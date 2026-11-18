//! RobStride MIT CAN protocol for the Maker Arm: the per-model mapping
//! ranges here, and the frame codec in [`mit`]. Pure functions, zero I/O.
//!
//! Ported (with modifications) from maker-arm-sdk maker_arm/protocol.py at
//! 0245316, Apache-2.0 - see the NOTICE file at the repository root.

pub const P_MIN: f64 = -12.57; // rad (encoding range, not joint limits)
pub const P_MAX: f64 = 12.57;
pub const V_MIN: f64 = -33.0; // rad/s
pub const V_MAX: f64 = 33.0;
pub const T_MIN: f64 = -14.0; // Nm
pub const T_MAX: f64 = 14.0;
pub const KP_MIN: f64 = 0.0;
pub const KP_MAX: f64 = 500.0;
pub const KD_MIN: f64 = 0.0;
pub const KD_MAX: f64 = 5.0; // protocol hard ceiling
/// The id the motors reply on.
pub const HOST_CAN_ID: u8 = 0xFD;

pub mod mit;

/// Per-model mapping ranges. The frame format is identical across models;
/// only the ranges differ. Picking the wrong table shows up as
/// torque/velocity scaled by the wrong factor.
///
/// **Invariant: every `*_min` must be strictly less than its `*_max`.**
/// The fields are public so per-joint tables can be written as consts, and
/// nothing enforces the invariant at construction: [`mit::float_to_uint`]
/// clamps with [`f64::clamp`], which **panics** on an inverted range
/// (`lo > hi`), and an equal pair (`lo == hi`) divides by zero. Tables are
/// expected to be compile-time constants that a test pins, not runtime
/// input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorParams {
    pub t_min: f64,
    pub t_max: f64,
    pub v_min: f64,
    pub v_max: f64,
    pub p_min: f64,
    pub p_max: f64,
}

pub const RS00: MotorParams = MotorParams {
    t_min: T_MIN,
    t_max: T_MAX,
    v_min: V_MIN,
    v_max: V_MAX,
    p_min: P_MIN,
    p_max: P_MAX,
};

/// RS02: ±17 Nm / ±44 rad/s (rated 6 Nm, 7.75:1 reduction).
/// Firmware 0.2.2.11 and older maps position to ±12.5, not ±12.57.
/// Parameter access implies 0.2.3.32 or later, where ±12.57 holds.
/// Without it, `maker-arm-cli doctor` warns.
pub const RS02: MotorParams = MotorParams {
    t_min: -17.0,
    t_max: 17.0,
    v_min: -44.0,
    v_max: 44.0,
    p_min: P_MIN,
    p_max: P_MAX,
};

pub fn motor_params(model: &str) -> Option<&'static MotorParams> {
    match model {
        "RS00" => Some(&RS00),
        "RS02" => Some(&RS02),
        _ => None,
    }
}

/// One standard CAN frame: an 11-bit id and an 8-byte payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    pub id: u32,
    pub data: [u8; 8],
}

/// The parameter indices this SDK reads and writes.
pub mod param_index {
    pub const RUN_MODE: u16 = 0x7005; // u8: 0 = control mode (MIT)
    pub const VBUS: u16 = 0x701C; // f32
    pub const CAN_TIMEOUT: u16 = 0x7028; // u32, 50 µs/count (20000 = 1 s)
}

/// canTimeout unit is 50 µs/count (protocol: 20000 = 1 s) — verified on
/// real hardware by the official SDK.
pub const CAN_TIMEOUT_PER_MS: u32 = 20;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamValue {
    F32(f32),
    U8(u8),
    U16(u16),
    U32(u32),
}
