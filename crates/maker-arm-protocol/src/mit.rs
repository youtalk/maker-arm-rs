//! RobStride MIT wire protocol: standard 11-bit CAN ids, 1 Mbit/s, pure
//! functions, zero I/O.
//!
//! A command goes to id = motor id. Replies come on `HOST_CAN_ID` with the
//! motor id in byte 0. Typed parameter replies come on `0x300|id` (read) and
//! `0x400|id` (write).
//!
//! Ported (with modifications) from maker-arm-sdk maker_arm/protocol.py at
//! 0245316, Apache-2.0 - see the NOTICE file at the repository root.

use crate::{Frame, MotorParams, ParamValue, HOST_CAN_ID, KD_MAX, KD_MIN, KP_MAX, KP_MIN};

pub const PARAM_READ_ID: u32 = 0x300;
pub const PARAM_WRITE_ID: u32 = 0x400;

const OP_CLEAR_FAULT: u8 = 0xFB;
const OP_ENABLE: u8 = 0xFC;
const OP_DISABLE: u8 = 0xFD;
const OP_SET_ZERO: u8 = 0xFE;

/// Fault word bits, from the vendor's `FAULT_BITS`.
const FAULT_BITS: [(u32, &str); 11] = [
    (0, "motor overtemperature"),
    (1, "driver chip fault"),
    (2, "undervoltage"),
    (3, "overvoltage"),
    (4, "phase B overcurrent"),
    (5, "phase C overcurrent"),
    (7, "encoder uncalibrated"),
    (8, "hardware identification fault"),
    (9, "position initialization fault"),
    (14, "stall/overload"),
    (16, "phase A overcurrent"),
];

/// Clamp to `lo..hi`, then scale to `0..2^bits - 1`, halves to even.
/// A non-finite `x` has no meaning here: callers check it first.
pub fn float_to_uint(x: f64, lo: f64, hi: f64, bits: u32) -> u32 {
    ((x.clamp(lo, hi) - lo) * ((1u32 << bits) - 1) as f64 / (hi - lo)).round_ties_even() as u32
}

pub fn uint_to_float(v: u32, lo: f64, hi: f64, bits: u32) -> f64 {
    v as f64 * (hi - lo) / ((1u32 << bits) - 1) as f64 + lo
}

fn command(motor_id: u8, arg: u8, op: u8) -> Frame {
    Frame {
        id: motor_id as u32,
        data: [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, arg, op],
    }
}

pub fn encode_enable(motor_id: u8) -> Frame {
    command(motor_id, 0xFF, OP_ENABLE)
}

pub fn encode_disable(motor_id: u8) -> Frame {
    command(motor_id, 0xFF, OP_DISABLE)
}

pub fn encode_set_zero(motor_id: u8) -> Frame {
    command(motor_id, 0xFF, OP_SET_ZERO)
}

/// Clears faults and requests feedback.
pub fn encode_clear_fault(motor_id: u8) -> Frame {
    command(motor_id, 0xFF, OP_CLEAR_FAULT)
}

/// Reads the fault word without clearing it.
pub fn encode_fault_query(motor_id: u8) -> Frame {
    command(motor_id, 0x00, OP_CLEAR_FAULT)
}

/// MIT impedance command: `tau_out = kp*(pos - q) + kd*(vel - dq) + tau`.
/// `None` means refused: a non-finite input, or a payload that would look
/// like a command (bytes 0 to 5 all `FF`).
pub fn encode_control(
    motor_id: u8,
    pos: f64,
    vel: f64,
    kp: f64,
    kd: f64,
    tau: f64,
    params: &MotorParams,
) -> Option<Frame> {
    if ![pos, vel, kp, kd, tau].iter().all(|x| x.is_finite()) {
        return None;
    }
    let q = float_to_uint(pos, params.p_min, params.p_max, 16);
    let v = float_to_uint(vel, params.v_min, params.v_max, 12);
    let k = float_to_uint(kp, KP_MIN, KP_MAX, 12);
    let d = float_to_uint(kd, KD_MIN, KD_MAX, 12);
    let t = float_to_uint(tau, params.t_min, params.t_max, 12);
    let data = [
        (q >> 8) as u8,
        q as u8,
        (v >> 4) as u8,
        (((v & 0xF) << 4) | (k >> 8)) as u8,
        k as u8,
        (d >> 4) as u8,
        (((d & 0xF) << 4) | (t >> 8)) as u8,
        t as u8,
    ];
    if data[..6] == [0xFF; 6] {
        return None;
    }
    Some(Frame {
        id: motor_id as u32,
        data,
    })
}

/// Inverse of `encode_control`: (pos, vel, kp, kd, tau). `None` for a
/// command payload.
pub fn decode_control(data: &[u8; 8], params: &MotorParams) -> Option<(f64, f64, f64, f64, f64)> {
    if data[..6] == [0xFF; 6] {
        return None;
    }
    let w = |i: usize| data[i] as u32;
    let q = (w(0) << 8) | w(1);
    let v = (w(2) << 4) | (w(3) >> 4);
    let k = ((w(3) & 0xF) << 8) | w(4);
    let d = (w(5) << 4) | (w(6) >> 4);
    let t = ((w(6) & 0xF) << 8) | w(7);
    Some((
        uint_to_float(q, params.p_min, params.p_max, 16),
        uint_to_float(v, params.v_min, params.v_max, 12),
        uint_to_float(k, KP_MIN, KP_MAX, 12),
        uint_to_float(d, KD_MIN, KD_MAX, 12),
        uint_to_float(t, params.t_min, params.t_max, 12),
    ))
}

pub fn encode_read_param(motor_id: u8, index: u16) -> Frame {
    let mut data = [0u8; 8];
    data[..2].copy_from_slice(&index.to_le_bytes());
    Frame {
        id: PARAM_READ_ID | motor_id as u32,
        data,
    }
}

pub fn encode_write_param(motor_id: u8, index: u16, value: ParamValue) -> Frame {
    let mut data = [0u8; 8];
    data[..2].copy_from_slice(&index.to_le_bytes());
    match value {
        ParamValue::F32(v) => data[4..].copy_from_slice(&v.to_le_bytes()),
        ParamValue::U8(v) => data[4] = v,
        ParamValue::U16(v) => data[4..6].copy_from_slice(&v.to_le_bytes()),
        ParamValue::U32(v) => data[4..].copy_from_slice(&v.to_le_bytes()),
    }
    Frame {
        id: PARAM_WRITE_ID | motor_id as u32,
        data,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Feedback {
    pub motor_id: u8,
    pub position: f64,
    pub velocity: f64,
    pub torque: f64,
    /// Winding temperature, degrees C.
    pub temperature: f64,
    /// 0 reset, 1 calibration, 2 run.
    pub mode: u8,
    pub fault: bool,
    pub warning: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaultStatus {
    pub motor_id: u8,
    pub word: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamReply {
    pub motor_id: u8,
    pub index: u16,
    pub raw: [u8; 4],
    /// A write acknowledgement (`0x400|id`), never a read result.
    pub write: bool,
}

impl ParamReply {
    pub fn as_u8(&self) -> u8 {
        self.raw[0]
    }

    pub fn as_u32(&self) -> u32 {
        u32::from_le_bytes(self.raw)
    }

    pub fn as_f32(&self) -> f32 {
        f32::from_le_bytes(self.raw)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Parsed {
    Feedback(Feedback),
    Fault(FaultStatus),
    Param(ParamReply),
}

fn is_param_id(id: u32) -> bool {
    matches!(id, 0x300..=0x37F | 0x400..=0x47F)
}

/// Motor id of a reply, without decoding it. Parameter replies carry it in
/// the CAN id because byte 0 is the register index.
pub fn reply_motor_id(id: u32, data: &[u8; 8]) -> Option<u8> {
    if id == HOST_CAN_ID as u32 {
        Some(data[0])
    } else if is_param_id(id) {
        Some((id & 0x7F) as u8)
    } else {
        None
    }
}

/// Decode one received frame. Anything that is not an MIT reply is `None`.
///
/// A `HOST_CAN_ID` reply whose bytes 5 to 7 are zero is a fault status, never
/// a motion sample: a late fault reply must not become an encoder reading.
pub fn parse_frame(id: u32, data: &[u8; 8], params: &MotorParams) -> Option<Parsed> {
    if is_param_id(id) {
        return Some(Parsed::Param(ParamReply {
            motor_id: (id & 0x7F) as u8,
            index: u16::from_le_bytes([data[0], data[1]]),
            raw: [data[4], data[5], data[6], data[7]],
            write: id >= PARAM_WRITE_ID,
        }));
    }
    if id != HOST_CAN_ID as u32 {
        return None;
    }
    if data[5..] == [0; 3] {
        return Some(Parsed::Fault(FaultStatus {
            motor_id: data[0],
            word: u32::from_le_bytes([data[1], data[2], data[3], data[4]]),
        }));
    }
    let w = |i: usize| data[i] as u32;
    let q = (w(1) << 8) | w(2);
    let v = (w(3) << 4) | (w(4) >> 4);
    let t = ((w(4) & 0xF) << 8) | w(5);
    let status = data[6];
    Some(Parsed::Feedback(Feedback {
        motor_id: data[0],
        position: uint_to_float(q, params.p_min, params.p_max, 16),
        velocity: uint_to_float(v, params.v_min, params.v_max, 12),
        torque: uint_to_float(t, params.t_min, params.t_max, 12),
        temperature: ((((status & 0x0F) as u32) << 8) | w(7)) as f64 / 10.0,
        mode: status >> 6,
        fault: status & 0x20 != 0,
        warning: status & 0x10 != 0,
    }))
}

/// A motor's feedback reply, for tests and the simulator.
pub fn encode_feedback(fb: &Feedback, params: &MotorParams) -> Frame {
    let q = float_to_uint(fb.position, params.p_min, params.p_max, 16);
    let v = float_to_uint(fb.velocity, params.v_min, params.v_max, 12);
    let t = float_to_uint(fb.torque, params.t_min, params.t_max, 12);
    let temp = ((fb.temperature * 10.0).round_ties_even() as u32).min(0xFFF);
    let status = (fb.mode << 6) | ((fb.fault as u8) << 5) | ((fb.warning as u8) << 4);
    Frame {
        id: HOST_CAN_ID as u32,
        data: [
            fb.motor_id,
            (q >> 8) as u8,
            q as u8,
            (v >> 4) as u8,
            (((v & 0xF) << 4) | (t >> 8)) as u8,
            t as u8,
            status | (temp >> 8) as u8,
            temp as u8,
        ],
    }
}

/// A motor's fault status reply, for tests and the simulator.
pub fn encode_fault_status(motor_id: u8, word: u32) -> Frame {
    let mut data = [0u8; 8];
    data[0] = motor_id;
    data[1..5].copy_from_slice(&word.to_le_bytes());
    Frame {
        id: HOST_CAN_ID as u32,
        data,
    }
}

/// A motor's parameter reply, for tests and the simulator.
pub fn encode_param_reply(r: &ParamReply) -> Frame {
    let mut data = [0u8; 8];
    data[..2].copy_from_slice(&r.index.to_le_bytes());
    data[4..].copy_from_slice(&r.raw);
    let base = if r.write {
        PARAM_WRITE_ID
    } else {
        PARAM_READ_ID
    };
    Frame {
        id: base | r.motor_id as u32,
        data,
    }
}

pub fn fault_text(word: u32) -> String {
    if word == 0 {
        return "no fault".into();
    }
    (0..32)
        .filter(|b| word & (1 << b) != 0)
        .map(|b| match FAULT_BITS.iter().find(|(n, _)| *n == b) {
            Some((_, name)) => name.to_string(),
            None => format!("bit {b}"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}
