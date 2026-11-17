//! Arm-level state and command types, in JOINT coordinates. Conversion to
//! motor coordinates (direction/offset/2π wrap) happens in the session.

/// Latest known state of one motor, joint coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorState {
    pub position: f64,    // rad
    pub velocity: f64,    // rad/s
    pub torque: f64,      // Nm (measured)
    pub temperature: f64, // °C
    pub mode: u8,         // 0=Reset 1=Cali 2=Motor
    /// Fault word from the last fault status reply.
    pub fault_word: u32,
    /// Fault and warning flags from the last feedback frame.
    pub fault_flag: bool,
    pub warning_flag: bool,
    /// Seconds since the last feedback frame; INFINITY before the first.
    pub feedback_age: f64,
    /// Seconds since the last fault status reply; INFINITY before the first.
    pub fault_reply_age: f64,
    /// The firmware answers parameter reads (found at connect).
    pub params: bool,
    /// A power-up announcement came while the session was enabled: the
    /// motor restarted with torque off.
    pub restarted: bool,
}

impl MotorState {
    pub fn stale() -> MotorState {
        MotorState {
            position: 0.0,
            velocity: 0.0,
            torque: 0.0,
            temperature: 0.0,
            mode: 0,
            fault_word: 0,
            fault_flag: false,
            warning_flag: false,
            feedback_age: f64::INFINITY,
            fault_reply_age: f64::INFINITY,
            params: false,
            restarted: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArmState {
    /// Same order as `ArmConfig::joints`.
    pub motors: Vec<MotorState>,
    pub tick: u64,
    /// Seconds since `Session::connect`.
    pub t: f64,
}

/// One joint's MIT command, joint coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointCommand {
    pub pos: f64,
    pub vel: f64,
    pub kp: f64,
    pub kd: f64,
    pub tau: f64,
}

/// Same order as `ArmConfig::joints`.
pub type ArmCommand = Vec<JointCommand>;
