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
///
/// **Invariant: every `*_min` must be strictly less than its `*_max`.**
/// The fields are public so per-joint tables can be written as consts, and
/// nothing enforces the invariant at construction: [`float_to_u16`] clamps
/// with [`f64::clamp`], which **panics** on an inverted range (`lo > hi`),
/// and an equal pair (`lo == hi`) divides by zero. Tables are expected to
/// be compile-time constants that a test pins, not runtime input.
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

/// Maps `x` into the `lo..=hi` range as a u16, clamping out-of-range input.
///
/// Rounding is round-half-to-even, matching Python's `round()`: MA1
/// golden-trace parity depends on byte-exact agreement with the official
/// SDK, and the two modes disagree on exact ties (see the
/// `round_half_to_even_is_load_bearing` vector).
///
/// # Non-finite input
///
/// This function never fails, so non-finite input is silently mapped to a
/// range extreme, and the mapping is *not* obviously safe:
///
/// * `NaN` → `0`, i.e. the **minimum** of the range (`P_MIN` is
///   −12.57 rad, `T_MIN` is −14 Nm) — `f64::clamp` propagates NaN and the
///   `as u16` cast then saturates it to 0.
/// * `+∞` → `65535` (range maximum), `-∞` → `0` (range minimum).
///
/// The Python oracle diverges here: `int(round(x))` raises `ValueError` on
/// NaN and `OverflowError` on ±∞ rather than encoding anything. Until the
/// session layer owns input validation, **callers must reject non-finite
/// values before calling** — a NaN escaping IK or a PD term would otherwise
/// be transmitted as a full-scale command, not as an error. Changing the
/// signature to report this is deliberately deferred; the behavior above is
/// pinned by tests so it cannot drift silently.
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

/// Well-known parameter indices (Type 17/18 frames).
pub mod param_index {
    pub const RUN_MODE: u16 = 0x7005; // u8: 0 = control mode (MIT)
    pub const LIMIT_TORQUE: u16 = 0x700B; // f32
    pub const LOC_REF: u16 = 0x7016; // f32
    pub const LIMIT_SPD: u16 = 0x7017; // f32
    pub const MECH_POS: u16 = 0x7019; // f32
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

pub fn encode_read_param(motor_id: u8, index: u16, host_id: u8) -> Frame {
    let mut data = [0u8; 8];
    data[0..2].copy_from_slice(&index.to_le_bytes());
    Frame {
        id: make_can_id(COMM_READ_PARAM, host_id as u16, motor_id),
        data,
    }
}

pub fn encode_write_param(motor_id: u8, index: u16, value: ParamValue, host_id: u8) -> Frame {
    let mut data = [0u8; 8];
    data[0..2].copy_from_slice(&index.to_le_bytes());
    match value {
        ParamValue::F32(v) => data[4..8].copy_from_slice(&v.to_le_bytes()),
        ParamValue::U8(v) => data[4] = v,
        ParamValue::U16(v) => data[4..6].copy_from_slice(&v.to_le_bytes()),
        ParamValue::U32(v) => data[4..8].copy_from_slice(&v.to_le_bytes()),
    }
    Frame {
        id: make_can_id(COMM_WRITE_PARAM, host_id as u16, motor_id),
        data,
    }
}

pub fn encode_save_params(motor_id: u8, host_id: u8) -> Frame {
    Frame {
        id: make_can_id(COMM_SAVE, host_id as u16, motor_id),
        data: [0; 8],
    }
}

/// Feedback in motor coordinates; direction/offset conversion is the arm
/// layer's job (MA1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorFeedback {
    pub motor_id: u8,
    pub position: f64,    // rad
    pub velocity: f64,    // rad/s
    pub torque: f64,      // Nm
    pub temperature: f64, // °C
    pub mode: u8,         // 0=Reset 1=Cali 2=Motor
    pub fault_bits: u8,   // 6-bit fault code, nonzero = faulted
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamReply {
    pub motor_id: u8,
    pub index: u16,
    pub raw: [u8; 4],
}

impl ParamReply {
    pub fn as_f32(&self) -> f32 {
        f32::from_le_bytes(self.raw)
    }
    pub fn as_u8(&self) -> u8 {
        self.raw[0]
    }
    pub fn as_u16(&self) -> u16 {
        u16::from_le_bytes([self.raw[0], self.raw[1]])
    }
    pub fn as_u32(&self) -> u32 {
        u32::from_le_bytes(self.raw)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaultReport {
    pub motor_id: u8,
    pub raw: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParsedFrame {
    Feedback(MotorFeedback),
    ParamReply(ParamReply),
    Fault(FaultReport),
}

pub fn parse_frame(id: u32, data: &[u8], params: &MotorParams) -> Option<ParsedFrame> {
    let comm = ((id >> 24) & 0x1F) as u8;
    if comm == COMM_FEEDBACK {
        if data.len() < 8 {
            return None;
        }
        let u = |i: usize| u16::from_be_bytes([data[i], data[i + 1]]);
        return Some(ParsedFrame::Feedback(MotorFeedback {
            motor_id: ((id >> 8) & 0xFF) as u8,
            position: u16_to_float(u(0), params.p_min, params.p_max),
            velocity: u16_to_float(u(2), params.v_min, params.v_max),
            torque: u16_to_float(u(4), params.t_min, params.t_max),
            // Unsigned on purpose: this mirrors the official SDK byte for
            // byte, which is the hard requirement. The cost is that a
            // sub-zero reading wraps instead of going negative (raw 0xFFF6,
            // two's-complement -1 °C, decodes as 6552.6 °C) — a nonsense
            // value that fails safe rather than a plausible one. Confirm the
            // sign convention against upstream during hardware bring-up.
            temperature: u(6) as f64 / 10.0,
            mode: ((id >> 22) & 0x03) as u8,
            fault_bits: ((id >> 16) & 0x3F) as u8,
        }));
    }
    if comm == COMM_READ_PARAM {
        if data.len() < 8 {
            return None;
        }
        return Some(ParsedFrame::ParamReply(ParamReply {
            motor_id: ((id >> 8) & 0xFF) as u8,
            index: u16::from_le_bytes([data[0], data[1]]),
            raw: [data[4], data[5], data[6], data[7]],
        }));
    }
    if comm == COMM_FAULT {
        let mut raw = [0u8; 8];
        raw[..data.len().min(8)].copy_from_slice(&data[..data.len().min(8)]);
        return Some(ParsedFrame::Fault(FaultReport {
            motor_id: ((id >> 8) & 0xFF) as u8,
            raw,
        }));
    }
    None
}

/// Private-protocol Type 25: switch the motor's communication protocol
/// (0 = private, 1 = CANopen, 2 = MIT). Persistent and mutually exclusive;
/// takes effect after a power cycle. Magic 01..06 at bytes 0..5, F_CMD at
/// byte 6 (anchored by real-hardware testing in the official SDK).
pub fn encode_set_protocol(motor_id: u8, f_cmd: u8, host_id: u8) -> Frame {
    Frame {
        id: make_can_id(COMM_SET_PROTOCOL, host_id as u16, motor_id),
        data: [1, 2, 3, 4, 5, 6, f_cmd, 0],
    }
}

/// MIT protocol command 8 (protocol switch) data field; the frame itself
/// uses an 11-bit standard ID equal to the motor id.
pub fn mit_switch_protocol_data(f_cmd: u8) -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, f_cmd, 0xFD]
}

/// MIT protocol command 5 (F_CMD=0, read fault status, no side effects) —
/// used as an MIT-mode probe ping.
pub fn mit_fault_query_data() -> [u8; 8] {
    [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFB]
}
