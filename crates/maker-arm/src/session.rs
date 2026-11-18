//! Arm session: state machine and bus I/O over the RobStride MIT protocol,
//! following the vendor SDK's arm.py (maker-arm-sdk 0245316, spec 6.2).
//! Connect sends each motor a non-clearing fault query and reads its state
//! only after a clean answer, infers parameter access from a RUN_MODE read,
//! and chooses the 2π wrap correction. Enable chooses the wrap again from
//! fresh positions, then refuses a reported fault, a pose outside the grace
//! window, a hot gripper and a joint at its health limit, arms the gripper
//! watchdog where parameters work, then preloads each motor with a
//! zero-gain frame before it enables it. Disable is checked per motor, then
//! the gripper watchdog is restored.
//!
//! One backend, one socket: commands and replies share a single
//! `CanBackend`, which also resolves the SocketCAN loopback hazard from the
//! MA0 report (a socket never receives its own sent frames; only OTHER
//! sockets on the interface do).
//!
//! One frame on the bus at a time: every motor replies on the same CAN id,
//! so two pending replies collide (P2 on the real arm: thousands of error
//! frames and fault-reply timeouts). Every frame goes through `exchange`,
//! which waits for the reply, or for a timeout, before the next frame. The
//! fast sweeps (the control loop, the fault poll, the release's first two
//! passes, the clearing frames) wait `reply_wait_us`; the checks that need
//! an answer wait `REPLY_TIMEOUT`. A parameter frame to legacy firmware gets
//! no reply: `read_param` still waits `PARAM_TIMEOUT` for one, because its
//! silence is how the session tells that firmware apart.

use crate::clamp::ClampError;
use crate::config::ArmConfig;
use crate::control_loop::Ramp;
use crate::controller::HoldController;
use crate::health::{FaultReason, HealthMonitor};
use crate::sim::SimArm;
use crate::state::{ArmCommand, ArmState, MotorState};
use maker_arm_protocol::{self as p, mit};
use maker_arm_transport::{CanBackend, TransportError};
use std::f64::consts::TAU;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Connected,
    Enabled,
    Fault,
}

#[derive(Debug)]
pub enum SessionError {
    Transport(TransportError),
    Clamp(ClampError),
    /// A motor did not reply in time: to the probe, the preload or the
    /// enable.
    Probe {
        motor_id: u8,
    },
    /// The motor reports a fault word. The fault is kept, never cleared.
    MotorFault {
        motor_id: u8,
        word: u32,
    },
    RunMode {
        motor_id: u8,
        value: u8,
    },
    /// The gripper is at or above its enable temperature, or a joint at or
    /// above `temp_hold_c`.
    TooHot {
        motor_id: u8,
        celsius: f64,
    },
    /// The enable reply shows a mode other than 2 (run).
    EnableMode {
        motor_id: u8,
        mode: u8,
    },
    EnableVerify {
        motor_id: u8,
        index: u16,
    },
    PositionOutOfRange {
        motor_id: u8,
        joint_pos: f64,
    },
    /// The 2π wrap chosen for the motor's position (at connect, or again
    /// before an enable) maps part of the joint's CONFIGURED range outside
    /// the encoder's `[p_min, p_max]` window. Distinct from
    /// [`SessionError::PositionOutOfRange`] on purpose: the arm's current
    /// pose is fine, but commands near this limit would silently saturate
    /// in `float_to_uint`, so the operator's action is different (re-zero
    /// the motor, or rotate it one full turn) rather than "move the joint
    /// back into range".
    RangeNotMappable {
        motor_id: u8,
        wrap: f64,
        joint_bound: f64,
        motor_pos: f64,
        encoder_limit: f64,
    },
    WrongState {
        expected: &'static str,
    },
    /// The checked disable could not confirm these motors off. They may
    /// still hold torque. `cause` is the first other error of the sweep.
    NotDisabled {
        motors: Vec<u8>,
        cause: Option<Box<SessionError>>,
    },
    /// The control loop had stopped on `error` before the release.
    /// `release` is the checked disable's error, `None` when it
    /// confirmed every motor off.
    LoopStopped {
        error: Box<SessionError>,
        release: Option<Box<SessionError>>,
    },
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Transport(e) => write!(f, "transport: {e}"),
            SessionError::Clamp(e) => write!(f, "clamp: {e}"),
            SessionError::Probe { motor_id } => {
                write!(f, "motor {motor_id} did not reply -- check bus/power")
            }
            SessionError::MotorFault { motor_id, word } => {
                // A word of 0 means only the feedback fault flag was seen.
                let text = match word {
                    0 => "fault flag set".to_string(),
                    _ => mit::fault_text(*word),
                };
                write!(
                    f,
                    "motor {motor_id} reports a fault: {text} (word {word:#010x}) -- fault kept. \
                     Fix the cause, then press and release the emergency stop, or run the \
                     vendor `maker-arm scan --clear-faults`"
                )
            }
            SessionError::RunMode { motor_id, value } => {
                write!(
                    f,
                    "motor {motor_id} RUN_MODE is {value}, not 0 (MIT control)"
                )
            }
            SessionError::TooHot { motor_id, celsius } => {
                write!(
                    f,
                    "motor {motor_id} is {celsius:.1} degC, too hot to enable"
                )
            }
            SessionError::EnableMode { motor_id, mode } => {
                write!(
                    f,
                    "motor {motor_id} enable rejected: mode {mode}, not 2 (run)"
                )
            }
            SessionError::EnableVerify { motor_id, index } => {
                write!(
                    f,
                    "motor {motor_id} failed param verify for index {index:#06x}"
                )
            }
            SessionError::PositionOutOfRange {
                motor_id,
                joint_pos,
            } => {
                write!(
                    f,
                    "motor {motor_id} at {joint_pos:.3} rad is outside its limit window"
                )
            }
            SessionError::RangeNotMappable {
                motor_id,
                wrap,
                joint_bound,
                motor_pos,
                encoder_limit,
            } => {
                write!(
                    f,
                    "motor {motor_id}: with the {wrap:+.3} rad wrap its position needs, its \
                     configured limit {joint_bound:.3} rad maps to {motor_pos:.3} rad in the \
                     motor frame, past the encoder limit {encoder_limit:.3} rad -- commands \
                     near that limit would saturate silently. Re-zero this motor, or rotate \
                     it one full turn, so its whole configured range fits inside the encoder \
                     range"
                )
            }
            SessionError::WrongState { expected } => {
                write!(f, "operation requires state {expected}")
            }
            SessionError::NotDisabled { motors, cause } => {
                let ids: Vec<String> = motors.iter().map(u8::to_string).collect();
                write!(f, "motors {} did not confirm the disable", ids.join(", "))?;
                if let Some(c) = cause {
                    write!(f, " ({c})")?;
                }
                write!(f, " -- they may still hold torque")
            }
            SessionError::LoopStopped { error, release } => {
                write!(f, "the control loop had stopped: {error}")?;
                match release {
                    Some(r) => write!(f, "; the release then failed: {r}"),
                    None => write!(f, "; the release confirmed every motor off"),
                }
            }
        }
    }
}

impl std::error::Error for SessionError {}

impl From<TransportError> for SessionError {
    fn from(e: TransportError) -> Self {
        SessionError::Transport(e)
    }
}

impl From<ClampError> for SessionError {
    fn from(e: ClampError) -> Self {
        SessionError::Clamp(e)
    }
}

struct MotorSlot {
    /// Joint-coordinate state; `feedback_age` is recomputed on read.
    state: MotorState,
    /// Raw motor position of the last feedback (before wrap/offset).
    raw_position: f64,
    /// 2π correction added to the raw motor position (0, +2π, or -2π), from
    /// `Session::choose_wraps`.
    wrap: f64,
    last_feedback: Option<Instant>,
    last_fault_reply: Option<Instant>,
    /// When `poll_fault` last sent this motor a fault query.
    last_fault_query: Option<Instant>,
    /// The motor did not reply to its last frame in time.
    missed: bool,
}

pub struct Session {
    backend: Box<dyn CanBackend>,
    config: ArmConfig,
    state: SessionState,
    slots: Vec<MotorSlot>,
    started: Instant,
    pub(crate) tick: u64,
    pub(crate) health: HealthMonitor,
    pub(crate) fault: Option<FaultReason>,
    /// Faults found after the first one while the arm holds, as the
    /// vendor's `fault_reasons`.
    pub(crate) later_faults: Vec<FaultReason>,
    pub(crate) fault_hold: Option<HoldController>,
    /// When the loop last sent the released gripper its disable.
    pub(crate) release_sent: Option<Instant>,
    /// When the current run of transport errors in `tick` began.
    pub(crate) failing_since: Option<Instant>,
    /// The gripper's slot and the last target sent to it (joint rad).
    pub(crate) grip_sent: Option<(usize, f64)>,
    /// The controller's gravity torque in the last command sent (N m), which the fault hold
    /// keeps sending. Empty before the first.
    pub(crate) gravity: Vec<f64>,
    /// The last command sent, after the clamp: the release ramp starts from it.
    pub(crate) sent: ArmCommand,
    /// The release ramp, from the tick it starts after to the end of the loop.
    pub(crate) ramp: Option<Ramp>,
    /// The gripper's CAN_TIMEOUT from before enable, kept for the restore.
    watchdog_restore: Option<u32>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("state", &self.state)
            .field("motors", &self.slots.len())
            .finish()
    }
}

// The vendor's timing (arm.py and motor.py).
const REPLY_TIMEOUT: Duration = Duration::from_millis(50);
const REPLY_TRIES: u32 = 3;
const PARAM_TIMEOUT: Duration = Duration::from_millis(120);
const PARAM_TRIES: u32 = 2;
/// Between a parameter write and its read-back.
const PARAM_SPACING: Duration = Duration::from_millis(5);
/// Between two motors' enables, and between two clearing frames.
const MOTOR_SPACING: Duration = Duration::from_millis(10);

impl Session {
    pub fn connect(
        backend: Box<dyn CanBackend>,
        config: ArmConfig,
    ) -> Result<Session, SessionError> {
        let slots = config
            .joints
            .iter()
            .map(|_| MotorSlot {
                state: MotorState::stale(),
                raw_position: 0.0,
                wrap: 0.0,
                last_feedback: None,
                last_fault_reply: None,
                last_fault_query: None,
                missed: false,
            })
            .collect();
        let health = HealthMonitor::new(&config);
        let mut s = Session {
            backend,
            config,
            state: SessionState::Connected,
            slots,
            started: Instant::now(),
            tick: 0,
            health,
            fault: None,
            later_faults: Vec::new(),
            fault_hold: None,
            release_sent: None,
            failing_since: None,
            grip_sent: None,
            gravity: Vec::new(),
            sent: Vec::new(),
            ramp: None,
            watchdog_restore: None,
        };
        let motor_ids: Vec<u8> = s.config.joints.iter().map(|j| j.motor_id).collect();
        for m in &motor_ids {
            s.probe(*m)?;
        }
        // Parameter access, inferred per motor from a RUN_MODE read.
        for (i, &m) in motor_ids.iter().enumerate() {
            let mut reply = None;
            for _ in 0..PARAM_TRIES {
                reply = s.read_param(m, p::param_index::RUN_MODE)?;
                if reply.is_some() {
                    break;
                }
            }
            if let Some(raw) = reply {
                s.slots[i].state.params = true;
                if raw[0] != 0 {
                    return Err(SessionError::RunMode {
                        motor_id: m,
                        value: raw[0],
                    });
                }
            }
        }
        s.choose_wraps()?;
        Ok(s)
    }

    /// The 2π wrap correction (upstream convention), from each motor's last
    /// raw position: the first of 0, +2π and -2π that puts the joint inside
    /// its limits widened by `enable_limit_grace`. Connect calls it, enable
    /// calls it again with fresh positions, and so does each released read:
    /// a motor that restarted, as after the emergency stop, can report its
    /// position one turn away. A motor that fits no wrap
    /// (`PositionOutOfRange`) or whose wrap cannot map its configured range
    /// (`RangeNotMappable`) keeps its old wrap. Every motor is tried, and
    /// the first error comes back.
    fn choose_wraps(&mut self) -> Result<(), SessionError> {
        let grace = self.config.enable_limit_grace;
        let mut first_err = None;
        for (j, slot) in self.config.joints.iter().zip(&mut self.slots) {
            let (lo, hi) = (j.q_lo - grace, j.q_hi + grace);
            let raw = slot.raw_position;
            let Some(wrap) = [0.0, TAU, -TAU].into_iter().find(|w| {
                let q = j.to_joint(raw + w);
                q.is_finite() && q >= lo && q <= hi
            }) else {
                first_err.get_or_insert(SessionError::PositionOutOfRange {
                    motor_id: j.motor_id,
                    joint_pos: j.to_joint(raw + slot.wrap),
                });
                continue;
            };
            // The wrap is chosen from where the arm HAPPENS to be parked,
            // but the control loop's output stage sends
            // `to_motor(q) - wrap` for every commanded q in
            // `q_lo..q_hi`. If either end of that configured range falls
            // outside the encoder's `[p_min, p_max]` window,
            // `float_to_uint` would clamp it to the rail with no error --
            // a silently truncated position command at full kp (the
            // reviewer measured 1.61 rad of truncation on J3 with
            // `clamped: false` reported). Refuse instead: the motor needs
            // re-zeroing or a full turn before it can be driven safely.
            // Checked against BOTH bounds, since a negative `direction`
            // swaps which joint bound maps highest.
            let params = j.model.params();
            let unmappable = [j.q_lo, j.q_hi]
                .into_iter()
                .map(|bound| (bound, j.to_motor(bound) - wrap))
                .find(|&(_, m)| !(m >= params.p_min && m <= params.p_max));
            if let Some((joint_bound, motor_pos)) = unmappable {
                first_err.get_or_insert(SessionError::RangeNotMappable {
                    motor_id: j.motor_id,
                    wrap,
                    joint_bound,
                    motor_pos,
                    encoder_limit: if motor_pos > params.p_max {
                        params.p_max
                    } else {
                        params.p_min
                    },
                });
                continue;
            }
            slot.wrap = wrap;
            slot.state.position = j.to_joint(raw + wrap);
        }
        first_err.map_or(Ok(()), Err)
    }

    /// The vendor's order: a non-clearing fault query first, then the
    /// clearing state read only after a clean answer, so a reported fault
    /// is never erased.
    fn probe(&mut self, motor_id: u8) -> Result<(), SessionError> {
        let idx = self.index_of(motor_id).expect("known motor");
        if !self.ask(idx, mit::encode_fault_query(motor_id), true, REPLY_TRIES)? {
            return Err(SessionError::Probe { motor_id });
        }
        let word = self.slots[idx].state.fault_word;
        if word != 0 {
            return Err(SessionError::MotorFault { motor_id, word });
        }
        if !self.ask(idx, mit::encode_clear_fault(motor_id), false, REPLY_TRIES)? {
            return Err(SessionError::Probe { motor_id });
        }
        let st = self.slots[idx].state;
        if st.fault_flag {
            return Err(SessionError::MotorFault {
                motor_id,
                word: st.fault_word,
            });
        }
        Ok(())
    }

    /// Drains the queue, then sends `f` to motor slot `idx` and waits
    /// `REPLY_TIMEOUT` for its reply, up to `tries` times. False when none
    /// came. The drain comes first, so a late reply to an earlier frame
    /// cannot pass for this one.
    fn ask(
        &mut self,
        idx: usize,
        f: p::Frame,
        fault: bool,
        tries: u32,
    ) -> Result<bool, SessionError> {
        for _ in 0..tries {
            self.drain()?;
            if self.send_to(idx, f, fault, REPLY_TIMEOUT)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Sends `f` to motor slot `idx`, then waits up to `timeout` for that
    /// motor's reply: a fault status when `fault`, else feedback. Every
    /// frame that arrives meanwhile is dispatched. False when no reply
    /// came, which is not a fault by itself.
    pub(crate) fn send_to(
        &mut self,
        idx: usize,
        f: p::Frame,
        fault: bool,
        timeout: Duration,
    ) -> Result<bool, SessionError> {
        let j = &self.config.joints[idx];
        let (m, params) = (j.motor_id, j.model.params());
        let got = self.exchange(f, timeout, &mut |id, data| {
            let parsed = mit::parse_frame(id, data, params);
            match parsed {
                Some(mit::Parsed::Feedback(fb)) => !fault && fb.motor_id == m,
                Some(mit::Parsed::Fault(s)) => fault && s.motor_id == m,
                _ => false,
            }
        })?;
        self.slots[idx].missed = !got;
        Ok(got)
    }

    /// `CanBackend::send_and_wait`, then every frame it saw is dispatched,
    /// also after a receive error.
    fn exchange(
        &mut self,
        f: p::Frame,
        timeout: Duration,
        reply: &mut dyn FnMut(u32, &[u8; 8]) -> bool,
    ) -> Result<bool, SessionError> {
        let mut seen = Vec::new();
        let got = self
            .backend
            .send_and_wait(f.id, &f.data, timeout, &mut |id, data| {
                seen.push((id, *data));
                reply(id, data)
            });
        for (id, data) in seen {
            self.dispatch(id, &data);
        }
        Ok(got?)
    }

    fn index_of(&self, motor_id: u8) -> Option<usize> {
        self.config
            .joints
            .iter()
            .position(|j| j.motor_id == motor_id)
    }

    /// Routes one raw frame into the right motor slot. Anything that is
    /// not an MIT reply is ignored, except a frame on a motor's own id in
    /// an enabled session: that is the motor's power-up announcement, so
    /// it restarted with torque off. Legacy firmware gives no other sign.
    pub(crate) fn dispatch(&mut self, id: u32, data: &[u8; 8]) {
        let Some(idx) = mit::reply_motor_id(id, data).and_then(|m| self.index_of(m)) else {
            let enabled = matches!(self.state, SessionState::Enabled | SessionState::Fault);
            if let Some(i) = u8::try_from(id).ok().and_then(|m| self.index_of(m)) {
                self.slots[i].state.restarted |= enabled;
            }
            return;
        };
        let j = &self.config.joints[idx];
        let slot = &mut self.slots[idx];
        match mit::parse_frame(id, data, j.model.params()) {
            Some(mit::Parsed::Feedback(fb)) => {
                slot.raw_position = fb.position;
                slot.state = MotorState {
                    position: j.to_joint(fb.position + slot.wrap),
                    velocity: j.direction * fb.velocity,
                    torque: j.direction * fb.torque,
                    temperature: fb.temperature,
                    mode: fb.mode,
                    fault_flag: fb.fault,
                    warning_flag: fb.warning,
                    feedback_age: 0.0,
                    ..slot.state
                };
                slot.last_feedback = Some(Instant::now());
            }
            Some(mit::Parsed::Fault(fs)) => {
                slot.state.fault_word = fs.word;
                slot.last_fault_reply = Some(Instant::now());
            }
            // Parameter replies are consumed synchronously by read_param.
            Some(mit::Parsed::Param(_)) | None => {}
        }
    }

    /// Non-blocking drain of everything queued on the bus.
    pub(crate) fn drain(&mut self) -> Result<(), SessionError> {
        while let Some((id, data)) = self.backend.recv(Duration::ZERO)? {
            self.dispatch(id, &data);
        }
        Ok(())
    }

    /// Sends a parameter read and waits for its reply; `None` when none
    /// comes within `PARAM_TIMEOUT`.
    fn read_param(&mut self, motor_id: u8, index: u16) -> Result<Option<[u8; 4]>, SessionError> {
        let f = mit::encode_read_param(motor_id, index);
        let mut value = None;
        self.exchange(f, PARAM_TIMEOUT, &mut |id, data| {
            match mit::parse_frame(id, data, &p::RS00) {
                Some(mit::Parsed::Param(r))
                    if !r.write && r.motor_id == motor_id && r.index == index =>
                {
                    value = Some(r.raw);
                }
                _ => {}
            }
            value.is_some()
        })?;
        Ok(value)
    }

    /// Writes a u32 parameter and reads it back `PARAM_SPACING` later.
    fn write_param(
        &mut self,
        motor_id: u8,
        index: u16,
        value: u32,
    ) -> Result<Option<u32>, SessionError> {
        let f = mit::encode_write_param(motor_id, index, p::ParamValue::U32(value));
        self.exchange(f, PARAM_SPACING, &mut |_, _| false)?;
        Ok(self.read_param(motor_id, index)?.map(u32::from_le_bytes))
    }

    pub fn enable(&mut self) -> Result<(), SessionError> {
        if self.state != SessionState::Connected {
            return Err(SessionError::WrongState {
                expected: "Connected",
            });
        }
        // Gates: fresh fault status and position from every motor (cached
        // values never authorize torque), then limits and temperatures: the
        // gripper below its enable limit, every joint below the health limit
        // that would fault the first tick (ruling R31), before any torque.
        let motor_ids: Vec<u8> = self.config.joints.iter().map(|j| j.motor_id).collect();
        for &m in &motor_ids {
            self.probe(m)?;
        }
        // The limit window: the wrap is chosen again from these fresh
        // positions, so a motor that restarted a turn away since connect
        // still enables, and one that fits no wrap is refused.
        self.choose_wraps()?;
        for (i, j) in self.config.joints.iter().enumerate() {
            let st = &self.slots[i].state;
            let limit = self
                .config
                .gripper
                .filter(|g| g.motor_id == j.motor_id)
                .map_or(self.config.temp_hold_c, |g| g.enable_temperature_c);
            if st.temperature >= limit {
                return Err(SessionError::TooHot {
                    motor_id: j.motor_id,
                    celsius: st.temperature,
                });
            }
        }
        if let Err(e) = self.arm_watchdog().and_then(|()| self.preload_and_enable()) {
            // A motor may already be torque-on: release every motor. A
            // release error wins, because it names a motor that may still
            // hold torque.
            return Err(self.disable_all().err().unwrap_or(e));
        }
        self.state = SessionState::Enabled;
        // A freshly (re-)enabled session starts clean: any fault info that
        // got us here no longer applies now that torque is back on, and the
        // health monitor must not carry forward stale per-motor counters.
        // Cleared on the success path only, so a failed enable keeps the
        // fault reason a caller most wants to read.
        self.fault = None;
        self.later_faults.clear();
        self.fault_hold = None;
        self.release_sent = None;
        self.failing_since = None;
        self.grip_sent = None;
        self.gravity.clear();
        self.sent.clear();
        self.ramp = None;
        self.health = HealthMonitor::new(&self.config);
        for slot in &mut self.slots {
            slot.state.restarted = false;
        }
        Ok(())
    }

    /// The vendor's motor watchdog: the gripper only, and only where the
    /// firmware has parameter access. The original value is kept for the
    /// restore after a checked disable.
    fn arm_watchdog(&mut self) -> Result<(), SessionError> {
        let Some(g) = self.config.gripper else {
            return Ok(());
        };
        let Some(idx) = self.index_of(g.motor_id) else {
            return Ok(());
        };
        if !self.slots[idx].state.params {
            return Ok(());
        }
        let index = p::param_index::CAN_TIMEOUT;
        let failed = SessionError::EnableVerify {
            motor_id: g.motor_id,
            index,
        };
        let Some(original) = self.read_param(g.motor_id, index)? else {
            return Err(failed);
        };
        self.watchdog_restore = Some(u32::from_le_bytes(original));
        let counts = g.can_timeout_ms * p::CAN_TIMEOUT_PER_MS;
        if self.write_param(g.motor_id, index, counts)? != Some(counts) {
            return Err(failed);
        }
        Ok(())
    }

    /// Per motor: a zero-gain frame at the measured pose, so the enable
    /// cannot resume a stale setpoint left by an earlier session, then the
    /// enable. Each needs a reply within `REPLY_TIMEOUT`, and the enable
    /// reply must show mode 2 where the firmware reports mode.
    fn preload_and_enable(&mut self) -> Result<(), SessionError> {
        for i in 0..self.config.joints.len() {
            let j = &self.config.joints[i];
            let m = j.motor_id;
            let f = mit::encode_control(
                m,
                self.slots[i].raw_position,
                0.0,
                0.0,
                0.0,
                0.0,
                j.model.params(),
            )
            .expect("a measured position always encodes");
            if !self.ask(i, f, false, 1)? || !self.ask(i, mit::encode_enable(m), false, 1)? {
                return Err(SessionError::Probe { motor_id: m });
            }
            let st = self.slots[i].state;
            if st.fault_flag {
                return Err(SessionError::MotorFault {
                    motor_id: m,
                    word: st.fault_word,
                });
            }
            if st.params && st.mode != 2 {
                return Err(SessionError::EnableMode {
                    motor_id: m,
                    mode: st.mode,
                });
            }
            std::thread::sleep(MOTOR_SPACING);
        }
        Ok(())
    }

    /// The vendor's checked disable, best effort across every motor: a
    /// zero-gain frame at the measured pose (so a later enable cannot
    /// resume the old setpoint), a disable to every motor (each of these
    /// waits up to `reply_wait_us` for its reply), then per motor a
    /// disable whose reply must come within `REPLY_TIMEOUT` and show mode 0
    /// where the firmware reports mode (3 tries). The gripper watchdog is
    /// restored only once the gripper is confirmed off. Every motor gets
    /// its frames whatever failed before. After the sweep, the motors that
    /// did not confirm come back as `NotDisabled`; with none, the first
    /// error (a send, a drain, the watchdog restore).
    ///
    /// Also resets `fault`/`fault_hold`/`health`: torque is off and the
    /// session is back at `Connected`, so the health monitor must start
    /// clean on the next `enable()` -- otherwise a stale `fault` would keep
    /// `tick()` from ever entering `Fault` again, and the motors that
    /// faulted would stay skipped by the health monitor. The gripper
    /// release (`release_sent`) resets with them. The hold_on_fault = false
    /// path in `enter_fault` calls `disable()` and then re-sets
    /// `self.fault` so the caller of that tick still sees why the arm just
    /// went dark.
    fn disable_all(&mut self) -> Result<(), SessionError> {
        let n = self.config.joints.len();
        let wait = Duration::from_micros(self.config.reply_wait_us);
        let mut first_err: Option<SessionError> = None;
        let mut note = |r: Result<bool, SessionError>| {
            if let Err(e) = r {
                first_err.get_or_insert(e);
            }
        };
        // No drain before these sends: a broken receive must not keep a
        // disable off the bus.
        for i in 0..n {
            let j = &self.config.joints[i];
            let f = mit::encode_control(
                j.motor_id,
                self.slots[i].raw_position,
                0.0,
                0.0,
                0.0,
                0.0,
                j.model.params(),
            )
            .expect("a measured position always encodes");
            note(self.send_to(i, f, false, wait));
        }
        for i in 0..n {
            let f = mit::encode_disable(self.config.joints[i].motor_id);
            note(self.send_to(i, f, false, wait));
        }
        let mut gripper_off = false;
        let mut unconfirmed = Vec::new();
        for i in 0..n {
            let m = self.config.joints[i].motor_id;
            let mut off = false;
            for _ in 0..REPLY_TRIES {
                match self.ask(i, mit::encode_disable(m), false, 1) {
                    Ok(true) => {
                        let st = self.slots[i].state;
                        if !st.params || st.mode == 0 {
                            off = true;
                            break;
                        }
                    }
                    Ok(false) => {}
                    Err(e) => note(Err(e)),
                }
            }
            if !off {
                unconfirmed.push(m);
            }
            if self.config.gripper.is_some_and(|g| g.motor_id == m) {
                gripper_off = off;
            }
        }
        if let (true, Some(g), Some(original)) = (
            gripper_off,
            self.config.gripper,
            self.watchdog_restore.take(),
        ) {
            let index = p::param_index::CAN_TIMEOUT;
            match self.write_param(g.motor_id, index, original) {
                Ok(Some(v)) if v == original => {}
                Ok(_) => note(Err(SessionError::EnableVerify {
                    motor_id: g.motor_id,
                    index,
                })),
                Err(e) => note(Err(e)),
            }
        }
        self.state = SessionState::Connected;
        self.fault = None;
        self.later_faults.clear();
        self.fault_hold = None;
        self.release_sent = None;
        self.health = HealthMonitor::new(&self.config);
        if !unconfirmed.is_empty() {
            // It names every motor that may still hold torque, so it wins.
            return Err(SessionError::NotDisabled {
                motors: unconfirmed,
                cause: first_err.map(Box::new),
            });
        }
        first_err.map_or(Ok(()), Err)
    }

    pub fn disable(&mut self) -> Result<(), SessionError> {
        self.disable_all()
    }

    /// Callable from any state; torque off immediately.
    pub fn estop(&mut self) -> Result<(), SessionError> {
        self.disable_all()
    }

    /// Operator action, as the vendor's: the checked disable, then the
    /// clearing frame to each motor, 10 ms apart after its reply.
    pub fn clear_faults(&mut self) -> Result<(), SessionError> {
        let mut result = self.disable_all();
        let wait = Duration::from_micros(self.config.reply_wait_us);
        for i in 0..self.config.joints.len() {
            let f = mit::encode_clear_fault(self.config.joints[i].motor_id);
            if let Err(e) = self.send_to(i, f, false, wait) {
                result = result.and(Err(e));
            }
            std::thread::sleep(MOTOR_SPACING);
        }
        let drained = self.drain();
        result.and(drained)
    }

    /// Sends one non-clearing fault query to the motor asked longest ago,
    /// once `fault_poll_interval` has passed since that query, as the
    /// vendor does. Paced by the query sent, not the reply: a motor that
    /// never answers is still asked only once per interval, so it cannot
    /// starve the others. The query waits up to `reply_wait_us` for its
    /// reply, except to a motor that missed its last reply: it is most
    /// likely silent, and a second timeout in one tick would only make the
    /// tick late.
    pub fn poll_fault(&mut self) -> Result<(), SessionError> {
        let Some((i, slot)) = self
            .slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.last_fault_query)
        else {
            return Ok(());
        };
        if slot
            .last_fault_query
            .is_some_and(|t| t.elapsed().as_secs_f64() < self.config.fault_poll_interval)
        {
            return Ok(());
        }
        self.slots[i].last_fault_query = Some(Instant::now());
        let f = mit::encode_fault_query(self.config.joints[i].motor_id);
        let wait = if self.slots[i].missed {
            Duration::ZERO
        } else {
            Duration::from_micros(self.config.reply_wait_us)
        };
        self.send_to(i, f, true, wait).map(drop)
    }

    /// Reads every motor with torque off, in `scan`'s order: a non-clearing fault query, then
    /// the clearing state read only after a clean answer, so a reported fault is never erased.
    /// Each frame waits up to `reply_wait_us` for its motor's reply. Never enables or
    /// commands a motor; only in `Connected`. Then the 2π wrap is chosen again
    /// (`choose_wraps`): a motor that fits no wrap keeps its old one, and the
    /// enable refuses it. The result is in `arm_state`.
    pub fn read_motors(&mut self) -> Result<(), SessionError> {
        if self.state != SessionState::Connected {
            return Err(SessionError::WrongState {
                expected: "Connected",
            });
        }
        let wait = Duration::from_micros(self.config.reply_wait_us);
        self.drain()?;
        for i in 0..self.config.joints.len() {
            let m = self.config.joints[i].motor_id;
            if self.send_to(i, mit::encode_fault_query(m), true, wait)?
                && self.slots[i].state.fault_word == 0
            {
                self.send_to(i, mit::encode_clear_fault(m), false, wait)?;
            }
        }
        // Not a read error: the enable reports it.
        let _ = self.choose_wraps();
        Ok(())
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn config(&self) -> &ArmConfig {
        &self.config
    }

    pub fn positions(&self) -> Vec<f64> {
        self.slots.iter().map(|s| s.state.position).collect()
    }

    pub fn arm_state(&self) -> ArmState {
        let now = Instant::now();
        let age = |t: Option<Instant>| t.map_or(f64::INFINITY, |t| (now - t).as_secs_f64());
        ArmState {
            motors: self
                .slots
                .iter()
                .map(|s| MotorState {
                    feedback_age: age(s.last_feedback),
                    fault_reply_age: age(s.last_fault_reply),
                    ..s.state
                })
                .collect(),
            tick: self.tick,
            t: (now - self.started).as_secs_f64(),
        }
    }

    /// Test support: reach the SimArm through the boxed backend.
    pub fn backend_as_sim(&mut self) -> Option<&mut SimArm> {
        self.backend
            .as_any_mut()
            .and_then(|a| a.downcast_mut::<SimArm>())
    }

    /// Supports the control loop's fault-entry transitions.
    pub(crate) fn set_state(&mut self, s: SessionState) {
        self.state = s;
    }

    /// The 2π wrap correction chosen for motor slot `idx` at the enable —
    /// the control loop's output stage subtracts it back out when
    /// converting a commanded joint position to the raw motor frame.
    pub(crate) fn wrap_of(&self, idx: usize) -> f64 {
        self.slots[idx].wrap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connected() -> Session {
        let c = ArmConfig::maker_arm_v1();
        Session::connect(Box::new(SimArm::new(&c)), c).expect("connect")
    }

    #[test]
    fn a_power_up_announcement_is_ignored_before_enable_and_holds_after() {
        let mut s = connected();
        let seen = |s: &Session| -> Vec<_> {
            s.slots
                .iter()
                .map(|m| (m.state, m.last_feedback, m.last_fault_reply))
                .collect()
        };
        // A synthetic MCU id, never a real motor's.
        let announce = |s: &mut Session| {
            s.backend_as_sim()
                .unwrap()
                .queue_raw(3, [0xA5, 0x5A, 0, 0, 0, 0, 0, 3])
        };
        // Before enable (connect, probe), it changes nothing.
        let before = seen(&s);
        announce(&mut s);
        s.drain().unwrap();
        assert_eq!(seen(&s), before);
        // In an enabled session, motor 3 restarted with torque off: hold.
        s.enable().expect("enable");
        let position = s.positions()[2];
        announce(&mut s);
        let mut hold = HoldController::from_config(&s.config().clone());
        s.tick(&mut hold).expect("tick");
        assert_eq!(s.state(), SessionState::Fault);
        assert_eq!(
            s.fault(),
            Some(&FaultReason::MotorRestarted { motor_id: 3 })
        );
        assert!((s.positions()[2] - position).abs() < 1e-3);
        // No enable went out.
        assert_eq!(s.backend_as_sim().unwrap().enables(3), 1);
    }

    #[test]
    fn a_fault_status_is_not_feedback() {
        let mut s = connected();
        std::thread::sleep(Duration::from_millis(20));
        let before = s.arm_state().motors[1];
        let last_feedback = s.slots[1].last_feedback;
        // Bytes 1 and 2 are 0: as feedback, this would read p_min.
        let f = mit::encode_fault_status(2, 0);
        s.dispatch(f.id, &f.data);
        let after = s.arm_state().motors[1];
        assert_eq!(after.position, before.position);
        assert_eq!(s.slots[1].last_feedback, last_feedback);
        assert!(before.fault_reply_age >= 0.02);
        assert!(after.fault_reply_age < 0.01);
    }
}
