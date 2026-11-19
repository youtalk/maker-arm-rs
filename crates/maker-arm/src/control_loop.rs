//! The fixed-rate control loop and its output stage — the ONLY place that
//! turns controller output into MIT frames, and therefore the single clamp
//! enforcement point (design §2). On any health fault or rejected command
//! the loop swaps to an internal hold at the last-good pose
//! (hold_on_fault, the vendor's semantics) or disables outright. A gripper
//! that faulted itself is released instead of held. A release ramps the
//! torque down inside the loop before the checked disable (ruling R36).

use crate::clamp::clamp_command;
use crate::controller::{Controller, HoldController};
use crate::health::FaultReason;
use crate::session::{Session, SessionError, SessionState};
use crate::state::{ArmCommand, ArmState, JointCommand, MotorState};
use maker_arm_protocol::mit;
use maker_arm_transport::TransportError;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct TickOutcome {
    pub fault: Option<FaultReason>,
    pub clamped: bool,
    /// What this tick read and sent; None when the tick sent nothing (the arm was disabled)
    /// or a transport error cut it short.
    pub record: Option<TickRecord>,
}

/// One tick for the telemetry ring: the state the controller read and the command that went
/// out after the clamp.
#[derive(Debug, Clone, PartialEq)]
pub struct TickRecord {
    /// `ArmState::t` of the state the controller read.
    pub t: f64,
    /// The session tick count after this tick, so the first tick is 1.
    pub tick: u64,
    pub motors: Vec<MotorState>,
    /// The command after `clamp_command`, as encoded for the motors. A
    /// released gripper gets a disable instead of its entry.
    pub sent: ArmCommand,
    pub clamped: bool,
}

/// Ticks the telemetry rings hold: 20 s at 200 Hz. Both rings (the loop's and the tracking
/// controller's) use this one value.
pub const RING_TICKS: usize = 4000;

/// A released gripper's disable is re-sent this often, as the vendor does:
/// one lost disable must not leave it energized.
const RELEASE_RESEND: Duration = Duration::from_millis(100);

/// `later_faults` keeps at most this many, as the vendor caps its list.
const LATER_FAULTS_MAX: usize = 100;

/// A release ramps `kp` and the feed-forward torque down to 0 over this long before the
/// checked disable, so a joint that rests above its stop does not fall onto it (ruling R36).
pub const RELEASE_RAMP_S: f64 = 2.0; // INITIAL: the owner checks the forearm's fold on the real arm

/// `LoopShared::ramp`. No release asked.
const RAMP_OFF: u8 = 0;
/// `RunningArm::ramp_down` asked: the ramp starts after the next tick.
const RAMP_ASKED: u8 = 1;
const RAMP_ON: u8 = 2;
/// A fault ended the ramp: the loop holds, torque on.
const RAMP_HELD: u8 = 3;
/// The last ramp tick went out and the loop ended. The disable is the caller's.
const RAMP_DONE: u8 = 4;

/// The release ramp (ruling R36): the last command sent, with `kp` and `tau` times `s`, which
/// falls linearly from 1 to 0 over the ramp's ticks. `kd` stays, so the fall is damped. The
/// position target stays, and the velocity target is 0.
#[derive(Debug)]
pub(crate) struct Ramp {
    from: ArmCommand,
    tick: u32,
    ticks: u32,
}

impl Ramp {
    /// The next tick's command. Not called once `done`.
    fn next(&mut self) -> ArmCommand {
        self.tick += 1;
        let s = 1.0 - f64::from(self.tick) / f64::from(self.ticks);
        self.from
            .iter()
            .map(|c| JointCommand {
                pos: c.pos,
                vel: 0.0,
                kp: c.kp * s,
                kd: c.kd,
                tau: c.tau * s,
            })
            .collect()
    }

    fn done(&self) -> bool {
        self.tick >= self.ticks
    }
}

/// A bounded ring of records and the number it dropped since the last `take`.
#[derive(Debug)]
pub struct TickRing<T> {
    buf: VecDeque<T>,
    dropped: u64,
}

impl<T> Default for TickRing<T> {
    fn default() -> Self {
        TickRing {
            buf: VecDeque::new(),
            dropped: 0,
        }
    }
}

impl<T> TickRing<T> {
    /// The records since the last call, oldest first, and the count dropped since then.
    pub fn take(&mut self) -> (Vec<T>, u64) {
        (
            self.buf.drain(..).collect(),
            std::mem::take(&mut self.dropped),
        )
    }
}

/// The loop side of a ring. Records wait in a local vector until `try_lock` gets the ring, so
/// a reader holding the lock delays records by a tick and never blocks the tick. A full ring
/// drops from the front, and every drop is counted for the next `take`.
#[derive(Debug)]
pub struct RingWriter<T> {
    pending: VecDeque<T>,
    dropped: u64,
}

impl<T> Default for RingWriter<T> {
    fn default() -> Self {
        RingWriter {
            pending: VecDeque::new(),
            dropped: 0,
        }
    }
}

impl<T> RingWriter<T> {
    fn flush(&mut self, ring: &mut TickRing<T>) {
        for item in self.pending.drain(..) {
            if ring.buf.len() >= RING_TICKS {
                ring.buf.pop_front();
                ring.dropped += 1;
            }
            ring.buf.push_back(item);
        }
        ring.dropped += std::mem::take(&mut self.dropped);
    }

    /// Queues `item` and moves everything queued into `ring` if the lock is free. A reader
    /// that never lets go cannot grow the queue past `RING_TICKS`: the oldest queued record is
    /// dropped and counted.
    pub fn record(&mut self, item: T, ring: &Mutex<TickRing<T>>) {
        if self.pending.len() >= RING_TICKS {
            self.pending.pop_front();
            self.dropped += 1;
        }
        self.pending.push_back(item);
        self.try_flush(ring);
    }

    /// Moves the queue into `ring` if the lock is free. Poison-tolerant.
    pub fn try_flush(&mut self, ring: &Mutex<TickRing<T>>) {
        match ring.try_lock() {
            Ok(mut r) => self.flush(&mut r),
            Err(TryLockError::Poisoned(e)) => self.flush(&mut e.into_inner()),
            Err(TryLockError::WouldBlock) => {}
        }
    }

    /// Forgets everything queued and counted, for a restart.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.dropped = 0;
    }

    /// Takes the queue out without a lock and returns how many records it lost: the queued
    /// ones plus the drops not yet reported. For a writer that is being dropped.
    pub fn abandon(&mut self) -> u64 {
        let n = self.pending.len() as u64 + std::mem::take(&mut self.dropped);
        self.pending.clear();
        n
    }

    /// Blocking flush for the end of a loop.
    pub fn flush_blocking(&mut self, ring: &Mutex<TickRing<T>>) {
        self.flush(&mut ring.lock().unwrap_or_else(|e| e.into_inner()));
    }
}

/// Real-time scheduling for the loop thread (tracking design section 10): SCHED_FIFO at
/// `priority`, pinned to `cpu`. Off unless the caller passes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopRt {
    pub priority: i32,
    pub cpu: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtError {
    /// Not Linux: there is no SCHED_FIFO or affinity call to make.
    Unsupported,
    /// `sched_setaffinity` refused the CPU.
    Cpu { cpu: usize, errno: i32 },
    /// The priority is outside SCHED_FIFO's range on Linux, 1..=99.
    PriorityRange { priority: i32 },
    /// `pthread_setschedparam` refused SCHED_FIFO at this priority, usually because the user
    /// has no rtprio limit that high.
    Priority { priority: i32, errno: i32 },
}

impl std::fmt::Display for RtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RtError::Unsupported => write!(f, "real-time scheduling is only available on Linux"),
            RtError::Cpu { cpu, errno } => {
                write!(f, "cannot pin the loop thread to CPU {cpu} (errno {errno})")
            }
            RtError::PriorityRange { priority } => write!(
                f,
                "SCHED_FIFO priority {priority} is outside the range 1..=99"
            ),
            RtError::Priority { priority, errno } => write!(
                f,
                "cannot run the loop thread at SCHED_FIFO priority {priority} (errno {errno}); \
                 the user needs an rtprio limit of at least {priority}"
            ),
        }
    }
}

impl std::error::Error for RtError {}

/// `Session::start` failed before the first tick. The loop thread has exited and the session
/// comes back unchanged, so the caller still owns it.
#[derive(Debug)]
pub struct StartError {
    pub session: Session,
    pub error: RtError,
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for StartError {}

/// Pins the calling thread to `rt.cpu`, then switches it to SCHED_FIFO at `rt.priority`.
/// The caller errors come first, before any syscall: a CPU past the set size and a priority
/// outside 1..=99 are wrong whatever the privileges are. Then the pin, then the priority.
#[cfg(target_os = "linux")]
fn apply_rt(rt: LoopRt) -> Result<(), RtError> {
    if rt.cpu >= libc::CPU_SETSIZE as usize {
        return Err(RtError::Cpu {
            cpu: rt.cpu,
            errno: libc::EINVAL,
        });
    }
    if !(1..=99).contains(&rt.priority) {
        return Err(RtError::PriorityRange {
            priority: rt.priority,
        });
    }
    // SAFETY: `set` and `param` are plain C structs, zeroed before use, and both calls act on
    // the calling thread only (pid 0 and `pthread_self`).
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(rt.cpu, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(RtError::Cpu {
                cpu: rt.cpu,
                errno: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            });
        }
        let mut param: libc::sched_param = std::mem::zeroed();
        param.sched_priority = rt.priority;
        let rc = libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &param);
        if rc != 0 {
            return Err(RtError::Priority {
                priority: rt.priority,
                errno: rc,
            });
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn apply_rt(_rt: LoopRt) -> Result<(), RtError> {
    Err(RtError::Unsupported)
}

impl Session {
    pub fn fault(&self) -> Option<&FaultReason> {
        self.fault.as_ref()
    }

    /// The faults found after the first one while the arm held, oldest
    /// first. `fault` stays the first.
    pub fn later_faults(&self) -> &[FaultReason] {
        &self.later_faults
    }

    /// Starts the release ramp from the last command sent. With none sent, this loop
    /// commanded no torque, and the ramp is done at once.
    fn start_ramp(&mut self) {
        let ticks = if self.sent.is_empty() {
            0
        } else {
            (RELEASE_RAMP_S * self.config().control_rate_hz)
                .round()
                .max(1.0) as u32
        };
        self.ramp = Some(Ramp {
            from: self.sent.clone(),
            tick: 0,
            ticks,
        });
    }

    fn enter_fault(&mut self, reason: FaultReason, at: &ArmState) -> Result<(), SessionError> {
        if self.fault.is_some() {
            // A later fault while the arm holds is kept, as the vendor's
            // `fault_reasons`. It does not retarget the hold, which would
            // walk a sagging arm downhill.
            if self.later_faults.len() < LATER_FAULTS_MAX {
                self.later_faults.push(reason);
            }
            return Ok(());
        }
        // The first fault ends a release ramp: the arm holds, or disables, as on any fault.
        self.ramp = None;
        if self.config().hold_on_fault {
            // Joints 1-6 hold the measured pose. The gripper keeps its last
            // sent target, within `1.0 Nm / kp` of the jaw under serve, so
            // a held object keeps its squeeze, as the vendor keeps its
            // gripper regulation in a fault hold. Every joint keeps the
            // gravity torque of the last command: without it, J3 sagged
            // 50 mrad on the real arm.
            let mut targets: Vec<f64> = at.motors.iter().map(|m| m.position).collect();
            if let Some((i, pos)) = self.grip_sent {
                targets[i] = pos;
            }
            let mut hold = HoldController::with_targets(self.config(), targets);
            hold.tau = self.gravity.clone();
            // The hold target is the pose the arm was measured at, but it
            // still goes through `clamp_command` like any other command,
            // and the clamp pulls it back inside `q_lo..q_hi`. A pose that
            // was legally outside the soft limits at fault time (enable
            // accepts up to `enable_limit_grace` = 0.35 rad beyond them)
            // therefore MOVES up to that much when the hold engages:
            // "hold" means "hold the nearest in-limit pose", not "freeze
            // exactly here". Deliberate -- the clamp is the single
            // enforcement point and must not be bypassed on the fault
            // path -- but it is operator-visible motion during a fault.
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
    /// consulted). During a release ramp the ramp drives, in either state.
    /// Every tick also sends at most one fault query
    /// (`Session::poll_fault`), after the last motor's reply.
    ///
    /// # Health checks keep running in `Fault`
    ///
    /// As the vendor's, health is checked every tick, in `Fault` too. A
    /// motor that faulted is skipped from then on, so a later fault on
    /// another motor still shows. Only the first fault enters `Fault`:
    /// it stays the session's reason, and a later one goes to
    /// `later_faults` and does not `retarget_to_state` the hold, which
    /// would walk a sagging arm downhill. A later fault on the gripper
    /// still releases it (`send_mit_all`).
    ///
    /// # A transport error holds
    ///
    /// As the vendor's loop, a failed send or receive does not end the
    /// loop: the arm holds (`FaultReason::Transport`, once per run of
    /// errors) and the next tick tries again, so one dropped frame never
    /// leaves an energized arm unsupervised. The error comes back, and
    /// ends `run` and the loop thread, when the interface is gone or down
    /// (`TransportError::Closed`) or the errors have lasted longer than
    /// `feedback_timeout`. Neither loop disables on its way out.
    pub fn tick(&mut self, controller: &mut dyn Controller) -> Result<TickOutcome, SessionError> {
        let e = match self.tick_once(controller) {
            Ok(out) => {
                self.failing_since = None;
                return Ok(out);
            }
            Err(e) => e,
        };
        let SessionError::Transport(TransportError::Io(detail)) = &e else {
            return Err(e);
        };
        let first = self.failing_since.is_none();
        let since = *self.failing_since.get_or_insert_with(Instant::now);
        if since.elapsed().as_secs_f64() > self.config().feedback_timeout {
            return Err(e);
        }
        if first {
            let at = self.arm_state();
            let detail = detail.clone();
            self.enter_fault(FaultReason::Transport { detail }, &at)?;
        }
        Ok(TickOutcome {
            fault: self.fault.clone(),
            clamped: false,
            record: None,
        })
    }

    fn tick_once(&mut self, controller: &mut dyn Controller) -> Result<TickOutcome, SessionError> {
        if self.state() != SessionState::Enabled && self.state() != SessionState::Fault {
            return Err(SessionError::WrongState {
                expected: "Enabled",
            });
        }
        self.drain()?;
        // Health checks below run on whatever's cached: a motor's state
        // only updates when a feedback frame is dispatched, so a fault
        // that appears between ticks is visible starting from the tick
        // after its own MIT round trip lands (dispatched while THIS tick
        // waits for it, below) -- one tick (one control period) of detection
        // latency, same as on real hardware.
        let snapshot = self.arm_state();
        // Owned copy: `self.health.check(&mut ...)` and `self.config()`
        // (&self method) cannot borrow `self` simultaneously.
        let cfg = self.config().clone();
        let dt = 1.0 / cfg.control_rate_hz;

        if let Some(reason) = self.health.check(&snapshot, &cfg) {
            self.enter_fault(reason, &snapshot)?;
            if self.state() == SessionState::Connected {
                // hold_on_fault = false path: motors already disabled.
                return Ok(TickOutcome {
                    fault: self.fault.clone(),
                    clamped: false,
                    record: None,
                });
            }
        }

        let raw_cmd = if let Some(ramp) = self.ramp.as_mut() {
            ramp.next()
        } else if self.state() == SessionState::Fault {
            self.fault_hold
                .as_mut()
                .expect("fault state implies fault_hold")
                .update(&snapshot, dt)
        } else {
            controller.update(&snapshot, dt)
        };

        let (mut cmd, clamped) = match clamp_command(&raw_cmd, &cfg) {
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
                        record: None,
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
                    record: None,
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
            // possible anyway, nothing goes out this tick and the motors
            // keep their last frame. Do not count on a motor watchdog:
            // on firmware without parameter access no motor has one.
            let _refused_again = self.send_mit_all(&hold_cmd)?;
            cmd = hold_cmd;
        } else if self.state() == SessionState::Enabled && self.ramp.is_none() {
            // Only a finite torque: a NaN in the hold would fail its clamp and end the loop.
            // A ramp tick keeps the controller's: a fault during the ramp holds at full kp,
            // so it needs the full gravity torque.
            self.gravity = controller
                .gravity()
                .into_iter()
                .map(|t| if t.is_finite() { t } else { 0.0 })
                .collect();
        }
        self.sent.clone_from(&cmd);
        self.poll_fault()?;
        // The waits already dispatched each motor's reply. Drain what came
        // late too: a caller reading `arm_state()` right after `tick()`
        // returns must see the ACTUAL (clamped) values just sent, not a
        // stale cache.
        self.drain()?;
        self.tick += 1;
        let record = TickRecord {
            t: snapshot.t,
            tick: self.tick,
            motors: snapshot.motors,
            sent: cmd,
            clamped,
        };
        Ok(TickOutcome {
            fault: self.fault.clone(),
            clamped,
            record: Some(record),
        })
    }

    /// Encodes and sends one MIT frame per motor. Each frame waits up to
    /// `reply_wait_us` for its motor's reply before the next goes out
    /// (`Session::send_to`); a silent motor costs that timeout and the
    /// others still get their frames. A gripper that faulted
    /// itself is released instead, as the vendor's `release_on_fault`: it
    /// gets a disable at once and then every `RELEASE_RESEND`, and no
    /// control frame. A fault on any other motor keeps the gripper held.
    ///
    /// Returns `Ok(Some(detail))` — with NOTHING sent — when a computed
    /// motor-frame position falls outside that motor's encoder window
    /// `[p_min, p_max]`, or when `mit::encode_control` refuses the command.
    /// `encode_control`'s `float_to_uint` would clamp such a
    /// value to the rail and report nothing, so a truncated position at
    /// full kp would go out as if it were the commanded pose. The
    /// `RangeNotMappable` check at connect and at enable should make this
    /// unreachable; it is kept as a backstop because silently saturating a
    /// torque command is exactly the failure being eliminated, and two
    /// comparisons per joint are cheap insurance. The whole command is
    /// validated BEFORE the first frame goes out, so a refusal never
    /// leaves a partially-commanded arm.
    fn send_mit_all(&mut self, cmd: &[JointCommand]) -> Result<Option<String>, SessionError> {
        let gripper = self.config().gripper.map(|g| g.motor_id);
        let mut released = false;
        let mut grip = None;
        let resend = self
            .release_sent
            .is_none_or(|t| t.elapsed() >= RELEASE_RESEND);
        let mut frames = Vec::with_capacity(cmd.len());
        for (i, c) in cmd.iter().enumerate() {
            let j = &self.config().joints[i];
            if gripper == Some(j.motor_id) && self.health.faulted(i) {
                released = true;
                if resend {
                    frames.push((i, mit::encode_disable(j.motor_id)));
                }
                continue;
            }
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
            let Some(f) = mit::encode_control(
                j.motor_id,
                motor_pos,
                j.direction * c.vel,
                c.kp,
                c.kd,
                j.direction * c.tau,
                params,
            ) else {
                return Ok(Some(format!(
                    "motor {} command refused by the MIT encoder",
                    j.motor_id
                )));
            };
            frames.push((i, f));
            if gripper == Some(j.motor_id) {
                grip = Some((i, c.pos));
            }
        }
        if released && resend {
            self.release_sent = Some(Instant::now());
        }
        let wait = Duration::from_micros(self.config().reply_wait_us);
        for (i, f) in frames {
            self.send_to(i, f, false, wait)?;
        }
        if grip.is_some() {
            self.grip_sent = grip;
        }
        Ok(None)
    }

    /// Fixed-rate loop. Keeps running while `Fault`-holding; exits on
    /// `stop`, after `max_ticks`, or on an error `tick` does not hold
    /// through. Does NOT disable on exit — the caller owns the release
    /// decision.
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
    ///
    /// With `rt`, the thread first pins itself and switches to SCHED_FIFO (tracking design
    /// section 10). If either call fails, the thread exits before its first tick, and the
    /// session comes back inside the error. There is no fallback to normal scheduling: a
    /// silent fallback would hide exactly the risk the option exists to remove.
    pub fn start(
        mut self,
        mut controller: Box<dyn Controller>,
        rt: Option<LoopRt>,
    ) -> Result<RunningArm, Box<StartError>> {
        let shared = Arc::new(LoopShared {
            stop: AtomicBool::new(false),
            hold_now: AtomicBool::new(false),
            ramp: AtomicU8::new(RAMP_OFF),
            snapshot: Mutex::new(None),
            ring: Mutex::new(TickRing::default()),
        });
        let sh = Arc::clone(&shared);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            if let Some(rt) = rt {
                if let Err(e) = apply_rt(rt) {
                    let _ = ready_tx.send(Err(e));
                    return (self, Ok(()));
                }
            }
            let _ = ready_tx.send(Ok(()));
            let period = Duration::from_secs_f64(1.0 / self.config().control_rate_hz);
            let mut next = Instant::now() + period;
            let mut result = Ok(());
            // A tick's snapshot that `try_lock` could not publish; the next tick replaces it,
            // and the blocking publish after the loop sends the last one.
            let mut unpublished = None;
            let mut records = RingWriter::default();
            while !sh.stop.load(Ordering::Relaxed) {
                if sh.hold_now.swap(false, Ordering::Relaxed) {
                    let mut hold = HoldController::from_config(self.config());
                    hold.retarget_to_state(&self.arm_state());
                    controller = Box::new(hold);
                }
                match self.tick(controller.as_mut()) {
                    Ok(outcome) => match outcome.record {
                        Some(r) => records.record(r, &sh.ring),
                        None => records.try_flush(&sh.ring),
                    },
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                }
                // `try_lock`, never `lock`: std mutexes have no priority inheritance, so
                // with `rt` the loop would wait behind a normal-priority reader cloning the
                // snapshot. Poison-tolerant like `RunningArm::snapshot`: a panicking
                // reader must not turn every later publish into a second panic that
                // kills the control thread outright.
                let snap = Snapshot {
                    state: self.state(),
                    fault: self.fault.clone(),
                    later_faults: self.later_faults.clone(),
                    arm: self.arm_state(),
                };
                unpublished = match sh.snapshot.try_lock() {
                    Ok(mut out) => {
                        *out = Some(snap);
                        None
                    }
                    Err(TryLockError::Poisoned(e)) => {
                        *e.into_inner() = Some(snap);
                        None
                    }
                    Err(TryLockError::WouldBlock) => Some(snap),
                };
                if self.state() == SessionState::Connected {
                    break; // disabled by hold_on_fault=false
                }
                // The ramp starts from the command this tick sent.
                if sh
                    .ramp
                    .compare_exchange(RAMP_ASKED, RAMP_ON, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    self.start_ramp();
                }
                match &self.ramp {
                    Some(r) if r.done() => {
                        sh.ramp.store(RAMP_DONE, Ordering::Relaxed);
                        break;
                    }
                    None if sh.ramp.load(Ordering::Relaxed) == RAMP_ON => {
                        sh.ramp.store(RAMP_HELD, Ordering::Relaxed);
                    }
                    _ => {}
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
            records.flush_blocking(&sh.ring);
            if let Some(snap) = unpublished {
                *sh.snapshot.lock().unwrap_or_else(|e| e.into_inner()) = Some(snap);
            }
            // No disable here: only the owner's release disables
            // (`stop_and_disable`, `release`, `Drop`), so a loop that
            // stopped on an error never drops the arm.
            (self, result)
        });
        match ready_rx
            .recv()
            .expect("the loop thread reports before its first tick")
        {
            Ok(()) => Ok(RunningArm {
                shared,
                handle: Some(handle),
            }),
            Err(error) => {
                let (session, _) = handle
                    .join()
                    .expect("the loop thread returns the session after an rt failure");
                Err(Box::new(StartError { session, error }))
            }
        }
    }

    /// The release after the loop thread is joined: the checked disable
    /// whenever the arm may hold torque, also after the loop had stopped
    /// on an error, because the bus may work again. Both errors come back.
    fn release(&mut self, looped: Result<(), SessionError>) -> Result<(), SessionError> {
        let released = match self.state() {
            SessionState::Connected => Ok(()),
            _ => self.disable(),
        };
        match looped {
            Ok(()) => released,
            Err(error) => Err(SessionError::LoopStopped {
                error: Box::new(error),
                release: released.err().map(Box::new),
            }),
        }
    }
}

pub struct LoopShared {
    stop: AtomicBool,
    hold_now: AtomicBool,
    /// The release ramp: `RAMP_OFF` and the other `RAMP_` values.
    ramp: AtomicU8,
    snapshot: Mutex<Option<Snapshot>>,
    ring: Mutex<TickRing<TickRecord>>,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub state: SessionState,
    pub fault: Option<FaultReason>,
    /// `Session::later_faults`.
    pub later_faults: Vec<FaultReason>,
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

    /// The records since the last drain, oldest first, and the number dropped since then
    /// because the ring (`RING_TICKS`) was full. A nonzero count marks a gap in the records.
    /// Poison-tolerant like `snapshot`.
    pub fn drain(&self) -> (Vec<TickRecord>, u64) {
        self.shared
            .ring
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    /// True once the control thread has exited -- cleanly, by error, or by
    /// panic -- without joining it (`stop_and_disable` still owns that).
    ///
    /// The last snapshot keeps saying whatever the loop last published, so
    /// a caller that reports state must consult this too: a finished
    /// thread is no longer commanding the motors, and if it exited on an
    /// error it did NOT disable them (see `start`), so torque may still be
    /// on. The motors keep their last frame: on firmware without parameter
    /// access no motor has a watchdog. `stop_and_disable` still tries the
    /// disable, and the emergency stop is the way out.
    pub fn loop_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.is_finished())
    }

    /// Swap the active controller for a hold at the current pose. Torque
    /// stays ON — this is the Ctrl-C primitive, not a release.
    pub fn hold_now(&self) {
        self.shared.hold_now.store(true, Ordering::Relaxed);
    }

    /// Asks for the release ramp (ruling R36) and returns at once. After its
    /// next tick, the loop sends the command of that tick for
    /// `RELEASE_RAMP_S` at the loop rate, with `kp` and the feed-forward
    /// torque falling linearly to 0, `kd` kept, the position target held
    /// and the velocity target 0. Then the loop ends (`ramp_done`), and
    /// `stop_and_disable` runs the checked disable. The health checks and
    /// the reply pacing keep running. A first fault during the ramp ends it,
    /// and the loop holds as on any fault (`ramp_held`). A ramp asked in
    /// `Fault` ramps the fault hold down, and a later fault does not end it.
    /// A gripper released on its own fault stays released.
    pub fn ramp_down(&self) {
        self.shared.ramp.store(RAMP_ASKED, Ordering::Relaxed);
    }

    /// True once a fault ended the ramp: the loop holds the arm, torque on.
    pub fn ramp_held(&self) -> bool {
        self.shared.ramp.load(Ordering::Relaxed) == RAMP_HELD
    }

    /// True once the last ramp tick went out. The loop then ends, and the
    /// motors wait for the disable at that tick's command: `kd` only.
    pub fn ramp_done(&self) -> bool {
        self.shared.ramp.load(Ordering::Relaxed) == RAMP_DONE
    }

    /// The release: `ramp_down`, then `stop_and_disable` once the loop has
    /// ended. A loop that had already stopped on an error is disabled at
    /// once. A fault during the ramp gives the arm back, holding, as the
    /// error: nothing was disabled.
    pub fn release(self) -> Result<(Session, Result<(), SessionError>), RunningArm> {
        self.ramp_down();
        while !self.loop_finished() {
            if self.ramp_held() {
                return Err(self);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(self.stop_and_disable())
    }

    /// Join the control thread, then disable all motors, also after the
    /// loop had stopped on an error. A loop error comes back as
    /// `SessionError::LoopStopped` with the release's own result. No ramp:
    /// the torque goes off at once. `release` ramps it down first.
    pub fn stop_and_disable(self) -> (Session, Result<(), SessionError>) {
        let (mut session, looped) = self.join().expect("control thread panicked");
        let result = session.release(looped);
        (session, result)
    }

    /// Stop and join the control thread, and send nothing: the session and
    /// the loop's own result come back, the motors keep their last frame.
    /// For a loop that stopped on its own, when the caller cannot know that
    /// a disable is safe (a supply drop or a loose cable look the same).
    /// `Err` when the thread panicked: then no session comes back.
    pub fn join(mut self) -> std::thread::Result<(Session, Result<(), SessionError>)> {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.handle
            .take()
            .expect("handle only ever taken here or in Drop, and self is consumed after")
            .join()
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
    /// torque on forever. Signal `stop`, join and release, as
    /// `stop_and_disable` does, so a dropped handle still leaves the arm
    /// disabled.
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(Ok((mut session, looped))) = self.handle.take().map(|h| h.join()) {
            let _ = session.release(looped);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArmConfig, SimArm};

    #[test]
    fn a_reader_holding_the_snapshot_lock_does_not_stall_the_loop() {
        let c = ArmConfig::maker_arm_v1();
        let mut s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).expect("connect");
        s.enable().expect("enable");
        let running = s
            .start(Box::new(HoldController::from_config(&c)), None)
            .expect("start");
        std::thread::sleep(Duration::from_millis(50));
        let before = running.snapshot().expect("ticked").arm.tick;
        {
            let _reader = running.shared.snapshot.lock().unwrap();
            std::thread::sleep(Duration::from_millis(200));
        }
        std::thread::sleep(Duration::from_millis(20));
        // About 44 ticks at 200 Hz; a loop blocked on the lock gets about 5.
        let ticks = running.snapshot().unwrap().arm.tick - before;
        assert!(ticks >= 20, "{ticks} ticks while a reader held the lock");
        let (_, res) = running.stop_and_disable();
        res.expect("clean stop");
    }

    fn sim_session() -> (ArmConfig, Session) {
        let c = ArmConfig::maker_arm_v1();
        let mut s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).expect("connect");
        s.enable().expect("enable");
        (c, s)
    }

    /// Hold that keeps the state and the raw command of its last update.
    struct Spy {
        hold: HoldController,
        seen: Option<(ArmState, ArmCommand)>,
    }

    impl Controller for Spy {
        fn update(&mut self, state: &ArmState, dt: f64) -> ArmCommand {
            let raw = self.hold.update(state, dt);
            self.seen = Some((state.clone(), raw.clone()));
            raw
        }
    }

    #[test]
    fn every_tick_leaves_one_record_with_the_clamped_command() {
        let (c, mut s) = sim_session();
        let mut spy = Spy {
            hold: HoldController::from_config(&c),
            seen: None,
        };
        for want in 1..=40u64 {
            let out = s.tick(&mut spy).expect("tick");
            let r = out.record.expect("a record per tick");
            let (state, raw) = spy.seen.take().expect("the controller ran");
            assert_eq!(r.tick, want);
            assert_eq!(r.t, state.t);
            assert_eq!(r.motors, state.motors);
            let (sent, clamped) = clamp_command(&raw, &c).expect("clamp");
            assert_eq!(r.sent, sent);
            assert_eq!(r.clamped, clamped);
        }
    }

    #[test]
    fn a_full_ring_drops_the_oldest_and_counts_them() {
        let ring = Mutex::new(TickRing::default());
        let mut w = RingWriter::default();
        for i in 0..RING_TICKS + 10 {
            w.record(i, &ring);
        }
        let (got, dropped) = ring.lock().unwrap().take();
        assert_eq!(got.len(), RING_TICKS);
        assert_eq!(dropped, 10);
        assert_eq!(got[0], 10);
        assert_eq!(*got.last().unwrap(), RING_TICKS + 9);
        // The count is per drain.
        assert_eq!(ring.lock().unwrap().take(), (vec![], 0));
    }

    #[test]
    fn a_reader_that_never_lets_go_is_counted_not_silent() {
        let ring = Mutex::new(TickRing::default());
        let mut w = RingWriter::default();
        let held = ring.lock().unwrap();
        for i in 0..RING_TICKS + 7 {
            w.record(i, &ring);
        }
        drop(held);
        w.try_flush(&ring);
        let (got, dropped) = ring.lock().unwrap().take();
        assert_eq!(got.len(), RING_TICKS);
        assert_eq!(dropped, 7);
        assert_eq!(got[0], 7);
    }

    #[test]
    fn a_reader_holding_the_ring_lock_does_not_stall_the_loop() {
        let (c, s) = sim_session();
        let running = s
            .start(Box::new(HoldController::from_config(&c)), None)
            .expect("start");
        std::thread::sleep(Duration::from_millis(50));
        let _ = running.drain();
        {
            let _reader = running.shared.ring.lock().unwrap();
            std::thread::sleep(Duration::from_millis(100));
        }
        std::thread::sleep(Duration::from_millis(20));
        let (records, dropped) = running.drain();
        // About 24 ticks at 200 Hz; a loop blocked on the lock gets about 5.
        assert!(records.len() >= 15, "{} records", records.len());
        assert_eq!(dropped, 0);
        assert!(records.windows(2).all(|w| w[1].tick == w[0].tick + 1));
        let (_, res) = running.stop_and_disable();
        res.expect("clean stop");
    }
}
