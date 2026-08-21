//! Arm session: state machine and bus I/O, mirroring the official SDK's
//! arm.py semantics (verified upstream @ main, 2026-08-20): probe by
//! disable frame, 2π wrap correction at connect, RUN_MODE/CAN_TIMEOUT
//! write-and-verify at enable, positions checked against the grace window
//! before torque is ever enabled.
//!
//! One backend, one socket: commands and replies share a single
//! `CanBackend`, which also resolves the SocketCAN loopback hazard from the
//! MA0 report (a socket never receives its own sent frames; only OTHER
//! sockets on the interface do).

use crate::clamp::ClampError;
use crate::config::ArmConfig;
use crate::sim::SimArm;
use crate::state::{ArmState, MotorState};
use maker_arm_protocol as p;
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
    Probe { motor_id: u8 },
    EnableVerify { motor_id: u8, index: u16 },
    PositionOutOfRange { motor_id: u8, joint_pos: f64 },
    WrongState { expected: &'static str },
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Transport(e) => write!(f, "transport: {e}"),
            SessionError::Clamp(e) => write!(f, "clamp: {e}"),
            SessionError::Probe { motor_id } => {
                write!(
                    f,
                    "motor {motor_id} did not answer the probe -- check bus/power"
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
            SessionError::WrongState { expected } => {
                write!(f, "operation requires state {expected}")
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
    /// 2π correction added to the raw motor position (0, +2π, or -2π).
    wrap: f64,
    last_feedback: Option<Instant>,
}

pub struct Session {
    backend: Box<dyn CanBackend>,
    config: ArmConfig,
    state: SessionState,
    slots: Vec<MotorSlot>,
    started: Instant,
    pub(crate) tick: u64,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("state", &self.state)
            .field("motors", &self.slots.len())
            .finish()
    }
}

const PROBE_TIMEOUT: Duration = Duration::from_millis(50);
const PARAM_SPACING: Duration = Duration::from_millis(5);
const VERIFY_RETRY: Duration = Duration::from_millis(20);
const VERIFY_TRIES: u32 = 5;

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
            })
            .collect();
        let mut s = Session {
            backend,
            config,
            state: SessionState::Connected,
            slots,
            started: Instant::now(),
            tick: 0,
        };
        let motor_ids: Vec<u8> = s.config.joints.iter().map(|j| j.motor_id).collect();
        for m in &motor_ids {
            s.probe(*m)?;
        }
        // 2π wrap correction, once, at connect (upstream convention).
        for i in 0..s.config.joints.len() {
            let j = &s.config.joints[i];
            let (lo, hi) = (
                j.q_lo - s.config.enable_limit_grace,
                j.q_hi + s.config.enable_limit_grace,
            );
            let raw = s.slots[i].raw_position;
            let wrap = [0.0, TAU, -TAU]
                .into_iter()
                .find(|w| {
                    let q = j.to_joint(raw + w);
                    q.is_finite() && q >= lo && q <= hi
                })
                .ok_or(SessionError::PositionOutOfRange {
                    motor_id: j.motor_id,
                    joint_pos: j.to_joint(raw),
                })?;
            s.slots[i].wrap = wrap;
            s.slots[i].state.position = j.to_joint(raw + wrap);
        }
        Ok(s)
    }

    /// Read-only probe: a disable frame triggers one feedback frame (the
    /// RobStride probing convention); harmless on a torque-free motor.
    fn probe(&mut self, motor_id: u8) -> Result<(), SessionError> {
        let f = p::encode_disable(motor_id, false, self.config.host_id);
        self.backend.send(f.id, &f.data)?;
        let idx = self.index_of(motor_id).expect("known motor");
        let before = self.slots[idx].last_feedback;
        let deadline = Instant::now() + PROBE_TIMEOUT;
        loop {
            if let Some(frame) = self.backend.recv(Duration::from_millis(2))? {
                self.dispatch(frame.0, &frame.1);
            }
            if self.slots[idx].last_feedback != before && self.slots[idx].last_feedback.is_some() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(SessionError::Probe { motor_id });
            }
        }
    }

    fn index_of(&self, motor_id: u8) -> Option<usize> {
        self.config
            .joints
            .iter()
            .position(|j| j.motor_id == motor_id)
    }

    /// Routes one raw frame into the right motor slot.
    pub(crate) fn dispatch(&mut self, id: u32, data: &[u8; 8]) {
        let motor_id = ((id >> 8) & 0xFF) as u8;
        let Some(idx) = self.index_of(motor_id) else {
            return;
        };
        let j = &self.config.joints[idx];
        let Some(parsed) = p::parse_frame(id, data, j.model.params()) else {
            return;
        };
        if let p::ParsedFrame::Feedback(fb) = parsed {
            let slot = &mut self.slots[idx];
            slot.raw_position = fb.position;
            slot.state = MotorState {
                position: j.to_joint(fb.position + slot.wrap),
                velocity: j.direction * fb.velocity,
                torque: j.direction * fb.torque,
                temperature: fb.temperature,
                mode: fb.mode,
                fault_bits: fb.fault_bits,
                feedback_age: 0.0,
            };
            slot.last_feedback = Some(Instant::now());
        }
        // Param replies are consumed synchronously by read_param_raw.
    }

    /// Non-blocking drain of everything queued on the bus.
    pub(crate) fn drain(&mut self) -> Result<(), SessionError> {
        while let Some((id, data)) = self.backend.recv(Duration::ZERO)? {
            self.dispatch(id, &data);
        }
        Ok(())
    }

    /// Sends a param read and waits for the matching reply.
    fn read_param_raw(&mut self, motor_id: u8, index: u16) -> Option<[u8; 4]> {
        let f = p::encode_read_param(motor_id, index, self.config.host_id);
        self.backend.send(f.id, &f.data).ok()?;
        let deadline = Instant::now() + VERIFY_RETRY;
        loop {
            if let Some((id, data)) = self.backend.recv(Duration::from_millis(2)).ok()? {
                if let Some(p::ParsedFrame::ParamReply(r)) = p::parse_frame(id, &data, &p::RS00) {
                    if r.motor_id == motor_id && r.index == index {
                        return Some(r.raw);
                    }
                } else {
                    self.dispatch(id, &data);
                }
            }
            if Instant::now() >= deadline {
                return None;
            }
        }
    }

    pub fn enable(&mut self) -> Result<(), SessionError> {
        if self.state != SessionState::Connected {
            return Err(SessionError::WrongState {
                expected: "Connected",
            });
        }
        let host = self.config.host_id;
        let timeout_counts = self.config.motor_can_timeout_ms * p::CAN_TIMEOUT_PER_MS;
        let motor_ids: Vec<u8> = self.config.joints.iter().map(|j| j.motor_id).collect();
        // Write RUN_MODE=0 (MIT) and the motor-side watchdog, spaced 5 ms.
        for &m in &motor_ids {
            let f = p::encode_write_param(m, p::param_index::RUN_MODE, p::ParamValue::U8(0), host);
            self.backend.send(f.id, &f.data)?;
            std::thread::sleep(PARAM_SPACING);
            let f = p::encode_write_param(
                m,
                p::param_index::CAN_TIMEOUT,
                p::ParamValue::U32(timeout_counts),
                host,
            );
            self.backend.send(f.id, &f.data)?;
            std::thread::sleep(PARAM_SPACING);
        }
        // Read back and verify both params, up to 5 tries each.
        for &m in &motor_ids {
            for (index, want) in [
                (p::param_index::RUN_MODE, vec![0u8]),
                (
                    p::param_index::CAN_TIMEOUT,
                    timeout_counts.to_le_bytes().to_vec(),
                ),
            ] {
                let mut ok = false;
                for _ in 0..VERIFY_TRIES {
                    if let Some(raw) = self.read_param_raw(m, index) {
                        if raw[..want.len()] == want[..] {
                            ok = true;
                            break;
                        }
                    }
                    std::thread::sleep(VERIFY_RETRY);
                }
                if !ok {
                    return Err(SessionError::EnableVerify { motor_id: m, index });
                }
            }
        }
        // Fresh feedback, then the position sanity gate, then torque on.
        for &m in &motor_ids {
            self.probe(m)?;
        }
        for (i, j) in self.config.joints.iter().enumerate() {
            let q = self.slots[i].state.position;
            let (lo, hi) = (
                j.q_lo - self.config.enable_limit_grace,
                j.q_hi + self.config.enable_limit_grace,
            );
            if !q.is_finite() || q < lo || q > hi {
                return Err(SessionError::PositionOutOfRange {
                    motor_id: j.motor_id,
                    joint_pos: q,
                });
            }
        }
        for &m in &motor_ids {
            let f = p::encode_enable(m, host);
            self.backend.send(f.id, &f.data)?;
        }
        // Every enable frame has now gone out: the motors are physically
        // torque-on. Report that truthfully even if the trailing drain
        // below fails -- state must never lag reality in the unsafe
        // direction (Connected while motors are actually enabled).
        self.state = SessionState::Enabled;
        self.drain()?;
        Ok(())
    }

    /// Best-effort across every motor: an emergency stop must not give up
    /// on motors 3..7 just because motor 2's frame failed to send. Every
    /// motor gets an attempt regardless of earlier failures; the first
    /// error encountered (send or drain) is remembered and returned after
    /// the sweep, so the caller learns the bus is unreliable while every
    /// reachable motor has still been commanded off.
    fn disable_all(&mut self, clear_fault: bool) -> Result<(), SessionError> {
        let host = self.config.host_id;
        let motor_ids: Vec<u8> = self.config.joints.iter().map(|j| j.motor_id).collect();
        let mut first_err: Option<SessionError> = None;
        for &m in &motor_ids {
            let f = p::encode_disable(m, clear_fault, host);
            if let Err(e) = self.backend.send(f.id, &f.data) {
                if first_err.is_none() {
                    first_err = Some(e.into());
                }
            }
        }
        if let Err(e) = self.drain() {
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
        self.state = SessionState::Connected;
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub fn disable(&mut self) -> Result<(), SessionError> {
        self.disable_all(false)
    }

    /// Callable from any state; torque off immediately.
    pub fn estop(&mut self) -> Result<(), SessionError> {
        self.disable_all(false)
    }

    pub fn clear_faults(&mut self) -> Result<(), SessionError> {
        self.disable_all(true)
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
        ArmState {
            motors: self
                .slots
                .iter()
                .map(|s| MotorState {
                    feedback_age: s
                        .last_feedback
                        .map_or(f64::INFINITY, |t| (now - t).as_secs_f64()),
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
}
