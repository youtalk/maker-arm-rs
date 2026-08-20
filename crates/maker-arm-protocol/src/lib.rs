//! RobStride private CAN protocol: pure-function encode/decode, zero I/O.
//!
//! Byte order: control/feedback frames are big-endian; param read/write
//! frames (Type 17/18) are little-endian. τ_ff travels in bits 23..8 of the
//! 29-bit CAN ID, not in the 8-byte data field.
//!
//! Ported from maker-arm-sdk maker_arm/protocol.py (Apache-2.0).

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
pub const HOST_CAN_ID: u8 = 0xFD;

/// Per-model u16 mapping ranges. The frame format is identical across
/// models; only the ranges differ. Picking the wrong table shows up as
/// torque/velocity scaled by the wrong factor.
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
/// Firmware ≤ 0.2.2.11 maps position to ±12.5 (we assume the newer ±12.57)
/// — check the firmware version during bring-up.
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

/// Round-half-to-even matches Python's round(); MA1 golden-trace parity
/// depends on byte-exact agreement with the official SDK.
pub fn float_to_u16(x: f64, lo: f64, hi: f64) -> u16 {
    let x = x.clamp(lo, hi);
    ((x - lo) * 65535.0 / (hi - lo)).round_ties_even() as u16
}

pub fn u16_to_float(v: u16, lo: f64, hi: f64) -> f64 {
    v as f64 * (hi - lo) / 65535.0 + lo
}
