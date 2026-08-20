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

pub const COMM_GET_ID: u8 = 0;
pub const COMM_MIT: u8 = 1;
pub const COMM_FEEDBACK: u8 = 2;
pub const COMM_ENABLE: u8 = 3;
pub const COMM_DISABLE: u8 = 4;
pub const COMM_SET_ZERO: u8 = 6;
pub const COMM_SET_CAN_ID: u8 = 7;
pub const COMM_READ_PARAM: u8 = 17;
pub const COMM_WRITE_PARAM: u8 = 18;
pub const COMM_FAULT: u8 = 21;
pub const COMM_SAVE: u8 = 22;
pub const COMM_SET_PROTOCOL: u8 = 25;
pub const COMM_VERSION: u8 = 26;

/// One private-protocol frame: 29-bit extended CAN ID + fixed 8-byte payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    pub id: u32,
    pub data: [u8; 8],
}

pub fn make_can_id(comm: u8, data2: u16, target: u8) -> u32 {
    (((comm & 0x1F) as u32) << 24) | ((data2 as u32) << 8) | target as u32
}

pub fn encode_enable(motor_id: u8, host_id: u8) -> Frame {
    Frame {
        id: make_can_id(COMM_ENABLE, host_id as u16, motor_id),
        data: [0; 8],
    }
}

pub fn encode_disable(motor_id: u8, clear_fault: bool, host_id: u8) -> Frame {
    let mut data = [0u8; 8];
    if clear_fault {
        data[0] = 1;
    }
    Frame {
        id: make_can_id(COMM_DISABLE, host_id as u16, motor_id),
        data,
    }
}

pub fn encode_set_zero(motor_id: u8, host_id: u8) -> Frame {
    let mut data = [0u8; 8];
    data[0] = 1;
    Frame {
        id: make_can_id(COMM_SET_ZERO, host_id as u16, motor_id),
        data,
    }
}

pub fn encode_mit(
    motor_id: u8,
    pos: f64,
    vel: f64,
    kp: f64,
    kd: f64,
    tau: f64,
    params: &MotorParams,
) -> Frame {
    let mut data = [0u8; 8];
    data[0..2].copy_from_slice(&float_to_u16(pos, params.p_min, params.p_max).to_be_bytes());
    data[2..4].copy_from_slice(&float_to_u16(vel, params.v_min, params.v_max).to_be_bytes());
    data[4..6].copy_from_slice(&float_to_u16(kp, KP_MIN, KP_MAX).to_be_bytes());
    data[6..8].copy_from_slice(&float_to_u16(kd, KD_MIN, KD_MAX).to_be_bytes());
    let tau_u16 = float_to_u16(tau, params.t_min, params.t_max);
    Frame {
        id: make_can_id(COMM_MIT, tau_u16, motor_id),
        data,
    }
}
