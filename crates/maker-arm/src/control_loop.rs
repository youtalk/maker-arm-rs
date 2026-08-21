//! The fixed-rate control loop and its output stage — the ONLY place that
//! turns controller output into MIT frames, and therefore the single clamp
//! enforcement point (design §2). On any health fault or rejected command
//! the loop swaps to an internal hold at the last-good pose
//! (hold_on_fault, upstream semantics) or disables outright.

use crate::clamp::clamp_command;
use crate::controller::{Controller, HoldController};
use crate::health::FaultReason;
use crate::session::{Session, SessionError, SessionState};
use crate::state::ArmState;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct TickOutcome {
    pub fault: Option<FaultReason>,
    pub clamped: bool,
}

impl Session {
    pub fn fault(&self) -> Option<&FaultReason> {
        self.fault.as_ref()
    }

    fn enter_fault(&mut self, reason: FaultReason, at: &ArmState) -> Result<(), SessionError> {
        if self.config().hold_on_fault {
            let mut hold = HoldController::from_config(self.config());
            hold.retarget_to_state(at);
            self.fault_hold = Some(hold);
            self.fault = Some(reason);
            self.set_state(SessionState::Fault);
            Ok(())
        } else {
            self.fault = Some(reason);
            self.disable()
        }
    }

    /// One control cycle. Runs in `Enabled` (controller drives) and in
    /// `Fault` when holding (internal hold drives; the controller is not
    /// consulted).
    pub fn tick(&mut self, controller: &mut dyn Controller) -> Result<TickOutcome, SessionError> {
        if self.state() != SessionState::Enabled && self.state() != SessionState::Fault {
            return Err(SessionError::WrongState {
                expected: "Enabled",
            });
        }
        self.drain()?;
        // While not already faulted, solicit fresh feedback before deciding:
        // a motor's cached state only updates when a feedback frame is
        // dispatched, so without this a fault could go undetected for a
        // full tick (and the moving controller could be sent once more
        // before the loop ever notices).
        if self.fault.is_none() {
            self.refresh_feedback()?;
        }
        let snapshot = self.arm_state();
        // Owned copy: `self.health.check(&mut ...)` and `self.config()`
        // (&self method) cannot borrow `self` simultaneously.
        let cfg = self.config().clone();
        let dt = 1.0 / cfg.control_rate_hz;

        if self.fault.is_none() {
            if let Some(reason) = self.health.check(&snapshot, &cfg) {
                self.enter_fault(reason, &snapshot)?;
                if self.state() == SessionState::Connected {
                    // hold_on_fault = false path: motors already disabled.
                    return Ok(TickOutcome {
                        fault: self.fault.clone(),
                        clamped: false,
                    });
                }
            }
        }

        let raw_cmd = if self.state() == SessionState::Fault {
            self.fault_hold
                .as_mut()
                .expect("fault state implies fault_hold")
                .update(&snapshot, dt)
        } else {
            controller.update(&snapshot, dt)
        };

        let (cmd, clamped) = match clamp_command(&raw_cmd, &cfg) {
            Ok(x) => x,
            Err(e) => {
                // A bad command from a controller is a fault, not a crash:
                // hold the pose (or disable) and report.
                self.enter_fault(
                    FaultReason::BadCommand {
                        detail: e.to_string(),
                    },
                    &snapshot,
                )?;
                if self.state() == SessionState::Connected {
                    return Ok(TickOutcome {
                        fault: self.fault.clone(),
                        clamped: false,
                    });
                }
                let hold_cmd = self
                    .fault_hold
                    .as_mut()
                    .expect("holding")
                    .update(&snapshot, dt);
                clamp_command(&hold_cmd, &cfg)?
            }
        };

        self.send_mit_all(&cmd)?;
        // Drain this tick's own MIT replies now: a caller reading
        // `arm_state()` right after `tick()` returns must see the ACTUAL
        // (clamped) values just sent, not a stale cache.
        self.drain()?;
        self.tick += 1;
        Ok(TickOutcome {
            fault: self.fault.clone(),
            clamped,
        })
    }

    /// Actively solicits fresh feedback from every motor with a
    /// non-disruptive ENABLE re-send (a no-op on an already-enabled
    /// motor, mirroring how `probe()` uses DISABLE as a non-disruptive
    /// read while torque-free) and drains the replies. Never touches
    /// torque, position, or gains.
    fn refresh_feedback(&mut self) -> Result<(), SessionError> {
        let host = self.config().host_id;
        let motor_ids: Vec<u8> = self.config().joints.iter().map(|j| j.motor_id).collect();
        for m in motor_ids {
            let f = maker_arm_protocol::encode_enable(m, host);
            self.send_raw(f.id, &f.data)?;
        }
        self.drain()
    }

    fn send_mit_all(&mut self, cmd: &[crate::state::JointCommand]) -> Result<(), SessionError> {
        let spacing = self.config().inter_frame_us;
        for (i, c) in cmd.iter().enumerate() {
            let j = self.config().joints[i].clone();
            let motor_pos = j.to_motor(c.pos) - self.wrap_of(i);
            let f = maker_arm_protocol::encode_mit(
                j.motor_id,
                motor_pos,
                j.direction * c.vel,
                c.kp,
                c.kd,
                j.direction * c.tau,
                j.model.params(),
            );
            self.send_raw(f.id, &f.data)?;
            if spacing > 0 {
                let until = Instant::now() + Duration::from_micros(spacing);
                while Instant::now() < until {
                    std::hint::spin_loop();
                }
            }
        }
        Ok(())
    }

    /// Fixed-rate loop. Keeps running while `Fault`-holding; exits on
    /// `stop`, after `max_ticks`, or on a hard error. Does NOT disable on
    /// exit — the caller owns the release decision.
    pub fn run(
        &mut self,
        controller: &mut dyn Controller,
        stop: &AtomicBool,
        max_ticks: Option<u64>,
    ) -> Result<(), SessionError> {
        let period = Duration::from_secs_f64(1.0 / self.config().control_rate_hz);
        let mut next = Instant::now() + period;
        let mut n = 0u64;
        while !stop.load(Ordering::Relaxed) {
            self.tick(controller)?;
            if self.state() == SessionState::Connected {
                return Ok(()); // hold_on_fault=false path disabled the arm
            }
            n += 1;
            if let Some(max) = max_ticks {
                if n >= max {
                    return Ok(());
                }
            }
            let now = Instant::now();
            if next > now {
                std::thread::sleep(next - now);
            }
            next += period;
        }
        Ok(())
    }

    /// Moves the session into a background control thread.
    pub fn start(mut self, mut controller: Box<dyn Controller>) -> RunningArm {
        let shared = Arc::new(LoopShared {
            stop: AtomicBool::new(false),
            hold_now: AtomicBool::new(false),
            snapshot: Mutex::new(None),
        });
        let sh = Arc::clone(&shared);
        let handle = std::thread::spawn(move || {
            let period = Duration::from_secs_f64(1.0 / self.config().control_rate_hz);
            let mut next = Instant::now() + period;
            let mut result = Ok(());
            while !sh.stop.load(Ordering::Relaxed) {
                if sh.hold_now.swap(false, Ordering::Relaxed) {
                    let mut hold = HoldController::from_config(self.config());
                    hold.retarget_to_state(&self.arm_state());
                    controller = Box::new(hold);
                }
                match self.tick(controller.as_mut()) {
                    Ok(_) => {}
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                }
                *sh.snapshot.lock().unwrap() = Some(Snapshot {
                    state: self.state(),
                    fault: self.fault.clone(),
                    arm: self.arm_state(),
                });
                if self.state() == SessionState::Connected {
                    break; // disabled by hold_on_fault=false
                }
                let now = Instant::now();
                if next > now {
                    std::thread::sleep(next - now);
                }
                next += period;
            }
            if result.is_ok() && self.state() != SessionState::Connected {
                result = self.disable().and(result);
            }
            (self, result)
        });
        RunningArm { shared, handle }
    }
}

pub struct LoopShared {
    stop: AtomicBool,
    hold_now: AtomicBool,
    snapshot: Mutex<Option<Snapshot>>,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub state: SessionState,
    pub fault: Option<FaultReason>,
    pub arm: ArmState,
}

pub struct RunningArm {
    shared: Arc<LoopShared>,
    handle: std::thread::JoinHandle<(Session, Result<(), SessionError>)>,
}

impl RunningArm {
    pub fn snapshot(&self) -> Option<Snapshot> {
        self.shared.snapshot.lock().unwrap().clone()
    }

    /// Swap the active controller for a hold at the current pose. Torque
    /// stays ON — this is the Ctrl-C primitive, not a release.
    pub fn hold_now(&self) {
        self.shared.hold_now.store(true, Ordering::Relaxed);
    }

    /// Disable all motors and join the control thread.
    pub fn stop_and_disable(self) -> (Session, Result<(), SessionError>) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.handle.join().expect("control thread panicked")
    }
}
