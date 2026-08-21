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
            // `disable()` (`disable_all`) resets `fault`/`fault_hold`/
            // `health` as part of its general "back to a clean Connected
            // state" contract -- re-set `fault` right after so THIS tick's
            // caller still learns why the arm just went dark. An
            // externally-triggered disable/estop/clear_faults leaves it
            // cleared, as intended (see `disable_all`'s doc comment).
            let result = self.disable();
            self.fault = Some(reason);
            result
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
        // Health checks below run on whatever's cached: a motor's state
        // only updates when a feedback frame is dispatched, so a fault
        // that appears between ticks is visible starting from the tick
        // after its own MIT round trip lands (drained at the end of THIS
        // tick, below) -- one tick (one control period) of detection
        // latency, same as on real hardware.
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

        // Output-stage backstop (see `send_mit_all`): a command whose
        // motor-frame position would saturate the encoder range is a
        // fault, handled exactly like a clamp rejection -- hold the pose
        // (or disable) and report, rather than transmitting a truncated
        // position at full kp.
        if let Some(detail) = self.send_mit_all(&cmd)? {
            self.enter_fault(FaultReason::BadCommand { detail }, &snapshot)?;
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
            let (hold_cmd, _) = clamp_command(&hold_cmd, &cfg)?;
            // The hold target is the arm's own MEASURED pose, decoded
            // from the motor frame and then clamped into limits that
            // connect verified are mappable, so this second attempt
            // cannot be refused again. If some future change makes it
            // possible anyway, nothing goes out -- torque is held by the
            // motors' last command until their CAN_TIMEOUT watchdog
            // fires, which is the safe direction.
            let _refused_again = self.send_mit_all(&hold_cmd)?;
        }
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

    /// Encodes and sends one MIT frame per motor.
    ///
    /// Returns `Ok(Some(detail))` — with NOTHING sent — when a computed
    /// motor-frame position falls outside that motor's encoder window
    /// `[p_min, p_max]`. `encode_mit`'s `float_to_u16` would clamp such a
    /// value to the rail and report nothing, so a truncated position at
    /// full kp would go out as if it were the commanded pose. The
    /// connect-time `RangeNotMappable` check should make this unreachable;
    /// it is kept as a backstop because silently saturating a torque
    /// command is exactly the failure being eliminated, and two
    /// comparisons per joint are cheap insurance. The whole command is
    /// validated BEFORE the first frame goes out, so a refusal never
    /// leaves a partially-commanded arm.
    fn send_mit_all(
        &mut self,
        cmd: &[crate::state::JointCommand],
    ) -> Result<Option<String>, SessionError> {
        for (i, c) in cmd.iter().enumerate() {
            let j = &self.config().joints[i];
            let params = j.model.params();
            let motor_pos = j.to_motor(c.pos) - self.wrap_of(i);
            if !(motor_pos >= params.p_min && motor_pos <= params.p_max) {
                return Ok(Some(format!(
                    "motor {} position {motor_pos:.4} rad is outside the encoder range \
                     [{:.2}, {:.2}] (joint command {:.4} rad); refusing to send a command \
                     that would saturate",
                    j.motor_id, params.p_min, params.p_max, c.pos
                )));
            }
        }
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
        Ok(None)
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
            // Catch-up clamp: after a scheduling hiccup (a long GC-like
            // stall, a descheduled thread, a slow bus write) `next` can
            // already be in the past. Advancing it blindly makes the loop
            // sprint to "repay" the missed deadlines, bursting 7xN MIT
            // frames onto a real bus back to back. Drop the missed ticks
            // instead: a late control cycle is better than a flood.
            next = next.max(Instant::now());
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
                // Poison-tolerant like `RunningArm::snapshot`: a panicking
                // reader must not turn every later publish into a second
                // panic that kills the control thread outright.
                *sh.snapshot.lock().unwrap_or_else(|e| e.into_inner()) = Some(Snapshot {
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
                // Same catch-up clamp as `run()`: drop missed deadlines
                // rather than repaying them as a burst of MIT frames.
                next = next.max(Instant::now());
            }
            if result.is_ok() && self.state() != SessionState::Connected {
                result = self.disable().and(result);
            }
            (self, result)
        });
        RunningArm {
            shared,
            handle: Some(handle),
        }
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
    // `Option` so `stop_and_disable` (which consumes `self`) can `take()`
    // the handle and join it without leaving `Drop` to double-join.
    handle: Option<std::thread::JoinHandle<(Session, Result<(), SessionError>)>>,
}

impl RunningArm {
    /// The last published tick snapshot, or `None` before the first tick
    /// completes.
    ///
    /// Poison-tolerant on purpose: if the control thread panicked while
    /// holding this mutex, `lock().unwrap()` would panic here too, turning
    /// the operator's ONLY window into the arm's state into a second
    /// panic. A stale-but-readable snapshot is strictly more useful than
    /// that; pair it with [`RunningArm::loop_finished`] to tell a live
    /// loop from a dead one.
    pub fn snapshot(&self) -> Option<Snapshot> {
        self.shared
            .snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// True once the control thread has exited -- cleanly, by error, or by
    /// panic -- without joining it (`stop_and_disable` still owns that).
    ///
    /// The last snapshot keeps saying whatever the loop last published, so
    /// a caller that reports state must consult this too: a finished
    /// thread is no longer commanding the motors, and if it exited on an
    /// error it did NOT disable them (see `start`), so torque may still be
    /// on with only the motor-side CAN_TIMEOUT watchdog behind it.
    pub fn loop_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.is_finished())
    }

    /// Swap the active controller for a hold at the current pose. Torque
    /// stays ON — this is the Ctrl-C primitive, not a release.
    pub fn hold_now(&self) {
        self.shared.hold_now.store(true, Ordering::Relaxed);
    }

    /// Disable all motors and join the control thread.
    pub fn stop_and_disable(mut self) -> (Session, Result<(), SessionError>) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.handle
            .take()
            .expect("handle only ever taken here or in Drop, and self is consumed after")
            .join()
            .expect("control thread panicked")
    }
}

/// Clonable handle that can request a hold from signal context (e.g. a
/// Ctrl-C handler, which cannot borrow the `RunningArm` it interrupts).
#[derive(Clone)]
pub struct HoldHandle {
    shared: Arc<LoopShared>,
}

impl HoldHandle {
    /// Same effect as `RunningArm::hold_now`: torque stays ON.
    pub fn hold_now(&self) {
        self.shared.hold_now.store(true, Ordering::Relaxed);
    }
}

impl RunningArm {
    pub fn shared_hold_handle(&self) -> HoldHandle {
        HoldHandle {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Drop for RunningArm {
    /// Without this, dropping a `RunningArm` instead of calling
    /// `stop_and_disable()` leaves the background thread's own
    /// `Arc<LoopShared>` clone as the only thing keeping `stop` alive: it
    /// never gets set, and the detached thread keeps MIT-streaming with
    /// torque on forever (refreshing the motor-side CAN_TIMEOUT watchdog
    /// along the way, so even that safety net never fires). Signal `stop`
    /// and join so a dropped handle still leaves the arm disabled -- the
    /// spawned thread already disables on any non-error exit from its
    /// loop.
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
