//! Golden vectors taken from maker-arm-sdk tests/unit/test_protocol.py
//! (main @ 2026-08-20), Apache-2.0 — see the NOTICE file at the repository
//! root. Do not edit values without re-checking upstream.
use maker_arm_protocol::*;

fn approx(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} !~ {b} (tol {tol})");
}

#[test]
fn constants() {
    assert_eq!(P_MAX, 12.57);
    assert_eq!(P_MIN, -12.57);
    assert_eq!(V_MAX, 33.0);
    assert_eq!(V_MIN, -33.0);
    assert_eq!(T_MAX, 14.0);
    assert_eq!(T_MIN, -14.0);
    assert_eq!(KP_MAX, 500.0);
    assert_eq!(KP_MIN, 0.0);
    assert_eq!(KD_MAX, 5.0);
    assert_eq!(KD_MIN, 0.0);
    assert_eq!(HOST_CAN_ID, 0xFD);
}

#[test]
fn u16_roundtrip_and_bounds() {
    assert_eq!(float_to_u16(P_MIN, P_MIN, P_MAX), 0);
    assert_eq!(float_to_u16(P_MAX, P_MIN, P_MAX), 65535);
    assert_eq!(float_to_u16(0.0, P_MIN, P_MAX), 32768);
    assert_eq!(float_to_u16(99.0, P_MIN, P_MAX), 65535); // clamps
    assert_eq!(float_to_u16(-99.0, P_MIN, P_MAX), 0);
    approx(u16_to_float(32768, P_MIN, P_MAX), 0.0, 1e-3);
    let x = 1.234;
    let back = u16_to_float(float_to_u16(x, P_MIN, P_MAX), P_MIN, P_MAX);
    approx(back, x, 25.14 / 65535.0);
}

#[test]
fn round_half_to_even_is_load_bearing() {
    // The ONLY discriminating case in the Rust suite. Every other vector
    // ties at 32767.5, where round-half-to-even and round-half-away-from-
    // zero both give 32768, so they cannot tell the two modes apart.
    //
    // 14.0 Nm on RS02's torque range is an exact tie:
    //   (14.0 - (-17.0)) * 65535 / 34 = 31 * 65535 / 34
    //                                 = 2031585 / 34 = 59752.5
    // exactly (every step is representable in f64, and 59752.5 * 34 is
    // exactly 2031585). Halfway between 59752 (even) and 59753 (odd):
    //   round_ties_even -> 59752   (matches Python's round(), our oracle)
    //   round           -> 59753   (half away from zero, WRONG here)
    //
    // Byte-exact agreement with the official SDK is the project's #1
    // requirement, so if this assertion fails the rounding mode has been
    // changed — do not "fix" the expected value.
    assert_eq!(float_to_u16(14.0, RS02.t_min, RS02.t_max), 59752);
    // Same tie seen through the encoder: tau rides in id bits 23..8.
    let f = encode_mit(1, 0.0, 0.0, 0.0, 0.0, 14.0, &RS02);
    assert_eq!(f.id, 0x01E96801); // 0xE968 == 59752

    // Ties that both modes agree on, kept as a contrast.
    assert_eq!(float_to_u16(0.0, P_MIN, P_MAX), 32768); // 32767.5 -> 32768
    assert_eq!(float_to_u16(2.5, KD_MIN, KD_MAX), 32768); // 32767.5 -> 32768
}

#[test]
fn non_finite_input_is_pinned_not_rejected() {
    // float_to_u16 cannot fail, so non-finite input silently lands on a
    // range extreme. This DIVERGES from the Python oracle, whose
    // int(round(x)) raises ValueError on NaN and OverflowError on +/-inf.
    // Pinned here so the hazard is documented and observable rather than
    // accidental; see the doc comment on float_to_u16. The fix (validating
    // at the session layer, or a fallible signature) is MA1 work.

    // NaN survives f64::clamp and the `as u16` cast saturates it to 0 --
    // which is the range MINIMUM, i.e. -12.57 rad of commanded position.
    assert_eq!(float_to_u16(f64::NAN, P_MIN, P_MAX), 0);
    assert_eq!(float_to_u16(f64::NAN, T_MIN, T_MAX), 0); // -14 Nm
    approx(
        u16_to_float(float_to_u16(f64::NAN, P_MIN, P_MAX), P_MIN, P_MAX),
        P_MIN,
        1e-12,
    );

    // Infinities clamp to the range ends before rounding.
    assert_eq!(float_to_u16(f64::INFINITY, P_MIN, P_MAX), 65535);
    assert_eq!(float_to_u16(f64::NEG_INFINITY, P_MIN, P_MAX), 0);
    assert_eq!(float_to_u16(f64::INFINITY, KP_MIN, KP_MAX), 65535);

    // Through encode_mit: a NaN position is transmitted, not rejected.
    let f = encode_mit(1, f64::NAN, 0.0, 0.0, 0.0, 0.0, &RS00);
    assert_eq!(&f.data[0..2], &[0x00, 0x00]);
}

#[test]
fn rs02_ranges_and_firmware_trap() {
    // RS02: ±17 Nm / ±44 rad/s. Position range ±12.57 assumes firmware
    // > 0.2.2.11 (older firmware maps ±12.5) — the bring-up checklist
    // MUST verify firmware before trusting position decode. Pinned here
    // so nobody "fixes" the constant without reading this.
    assert_eq!(RS02.t_min, -17.0);
    assert_eq!(RS02.t_max, 17.0);
    assert_eq!(RS02.v_min, -44.0);
    assert_eq!(RS02.v_max, 44.0);
    assert_eq!(RS02.p_max, 12.57);
    assert_eq!(motor_params("RS02").unwrap().t_max, 17.0);
    assert_eq!(motor_params("RS00").unwrap().t_max, 14.0);
    assert!(motor_params("RS99").is_none());
}

#[test]
fn can_id_and_simple_encoders() {
    // enable frame: comm=3, data2=host 0xFD, target=motor 1
    assert_eq!(make_can_id(COMM_ENABLE, HOST_CAN_ID as u16, 1), 0x0300FD01);
    let f = encode_enable(1, HOST_CAN_ID);
    assert_eq!(f.id, 0x0300FD01);
    assert_eq!(f.data, [0u8; 8]);
    let f = encode_disable(2, false, HOST_CAN_ID);
    assert_eq!(f.id, 0x0400FD02);
    assert_eq!(f.data, [0u8; 8]);
    let f = encode_disable(2, true, HOST_CAN_ID);
    assert_eq!(f.data[0], 1);
    let f = encode_set_zero(3, HOST_CAN_ID);
    assert_eq!(f.id, 0x0600FD03);
    assert_eq!(f.data[0], 1);
}

#[test]
fn encode_mit_golden() {
    // all-zero command + kp=10, kd=1.0 — hand-computed golden from the
    // official SDK: τ_ff=0 → 0x8000 in ID bits 23..8.
    let f = encode_mit(1, 0.0, 0.0, 10.0, 1.0, 0.0, &RS00);
    assert_eq!(f.id, 0x01800001);
    assert_eq!(f.data, [0x80, 0x00, 0x80, 0x00, 0x05, 0x1F, 0x33, 0x33]);
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn encode_params_little_endian() {
    let f = encode_read_param(1, param_index::VBUS, HOST_CAN_ID);
    assert_eq!(f.id, 0x1100FD01);
    assert_eq!(f.data.to_vec(), hex("1C70000000000000")); // index little-endian
    let f = encode_write_param(
        1,
        param_index::CAN_TIMEOUT,
        ParamValue::U32(200),
        HOST_CAN_ID,
    );
    assert_eq!(f.id, 0x1200FD01);
    assert_eq!(f.data.to_vec(), hex("28700000C8000000")); // value little-endian u32
    let f = encode_write_param(1, param_index::LIMIT_SPD, ParamValue::F32(2.0), HOST_CAN_ID);
    assert_eq!(&f.data[0..2], &hex("1770")[..]);
    // Independent literal, not 2.0f32.to_le_bytes(): IEEE-754 2.0f32 is
    // 0x40000000, so little-endian on the wire is 00 00 00 40.
    assert_eq!(&f.data[4..8], &hex("00000040")[..]);
    let f = encode_save_params(1, HOST_CAN_ID);
    assert_eq!(f.id, 0x1600FD01);
}

#[test]
fn encode_write_param_value_widths() {
    // The four ParamValue arms write to DIFFERENT byte ranges: U8 -> [4],
    // U16 -> [4..6], U32/F32 -> [4..8]. A wrong offset in the U8 arm is a
    // silent mode error on RUN_MODE, the first write MA1 makes to every
    // motor, so pin the exact payload of each.

    // U8: run_mode = 0 (MIT control mode). Index 0x7005 little-endian at
    // [0..2], value at [4] only, everything else zero.
    let f = encode_write_param(1, param_index::RUN_MODE, ParamValue::U8(0), HOST_CAN_ID);
    assert_eq!(f.id, 0x1200FD01);
    assert_eq!(f.data.to_vec(), hex("0570000000000000"));

    // U8 with a nonzero value: only byte 4 moves (0xAA is asymmetric, so a
    // byte swap or a wrong offset shows up immediately).
    let f = encode_write_param(2, param_index::RUN_MODE, ParamValue::U8(0xAA), HOST_CAN_ID);
    assert_eq!(f.id, 0x1200FD02);
    assert_eq!(f.data.to_vec(), hex("05700000AA000000"));

    // U16: two little-endian bytes at [4..6], bytes 6..8 untouched.
    // 0xBEEF little-endian is EF BE.
    let f = encode_write_param(
        3,
        param_index::RUN_MODE,
        ParamValue::U16(0xBEEF),
        HOST_CAN_ID,
    );
    assert_eq!(f.id, 0x1200FD03);
    assert_eq!(f.data.to_vec(), hex("05700000EFBE0000"));

    // U32 for contrast: four little-endian bytes fill [4..8].
    let f = encode_write_param(
        4,
        param_index::CAN_TIMEOUT,
        ParamValue::U32(0xDEADBEEF),
        HOST_CAN_ID,
    );
    assert_eq!(f.data.to_vec(), hex("28700000EFBEADDE"));
}

#[test]
fn parse_feedback_golden() {
    // comm=2, mode=2 (Motor), fault=0, motor=1, target=host
    let id = (2u32 << 24) | (2 << 22) | (1 << 8) | 0xFD;
    assert_eq!(id, 0x028001FD);
    let data = hex("8000800080000159"); // pos/vel/tau ≈ 0, temp = 34.5 °C
    let Some(ParsedFrame::Feedback(fb)) = parse_frame(id, &data, &RS00) else {
        panic!("expected feedback");
    };
    assert_eq!(fb.motor_id, 1);
    assert_eq!(fb.mode, 2);
    assert_eq!(fb.fault_bits, 0);
    approx(fb.position, 0.0, 1e-3);
    approx(fb.velocity, 0.0, 1e-3);
    approx(fb.torque, 0.0, 1e-3);
    approx(fb.temperature, 34.5, 1e-9);
}

#[test]
fn parse_feedback_fault_bits() {
    let id = (2u32 << 24) | (2 << 22) | (0x21 << 16) | (3 << 8) | 0xFD;
    let Some(ParsedFrame::Feedback(fb)) = parse_frame(id, &[0u8; 8], &RS00) else {
        panic!("expected feedback");
    };
    assert_eq!(fb.motor_id, 3);
    assert_eq!(fb.fault_bits, 0x21);
}

#[test]
fn parse_param_reply() {
    // motor 2 reads back VBUS = 24.5: comm=17, ID bits 15..8 = motor id
    let id = (17u32 << 24) | (2 << 8) | 0xFD;
    let mut data = vec![0x1C, 0x70, 0, 0];
    data.extend_from_slice(&24.5f32.to_le_bytes());
    let Some(ParsedFrame::ParamReply(r)) = parse_frame(id, &data, &RS00) else {
        panic!("expected param reply");
    };
    assert_eq!(r.motor_id, 2);
    assert_eq!(r.index, param_index::VBUS);
    assert_eq!(r.as_f32(), 24.5);
    let mut d2 = vec![0x28, 0x70, 0, 0];
    d2.extend_from_slice(&200u32.to_le_bytes());
    let Some(ParsedFrame::ParamReply(r2)) = parse_frame(id, &d2, &RS00) else {
        panic!("expected param reply");
    };
    assert_eq!(r2.as_u32(), 200);
}

#[test]
fn parse_unknown_returns_none() {
    assert!(parse_frame((22u32 << 24) | 0xFD, &[0u8; 8], &RS00).is_none());
}

#[test]
fn encode_set_protocol_golden() {
    // Anchored by the official SDK's real-hardware test (motor 7,
    // 2026-08-07): Type 25, magic 01..06 at bytes 0..5, F_CMD at byte 6.
    let f = encode_set_protocol(7, 2, HOST_CAN_ID);
    assert_eq!(f.id, 0x1900FD07);
    assert_eq!(f.data.to_vec(), hex("0102030405060200"));
}

#[test]
fn mit_interop_frames_golden() {
    // MIT protocol command 8 (switch), 11-bit standard frame at motor id.
    assert_eq!(
        mit_switch_protocol_data(0).to_vec(),
        hex("FFFFFFFFFFFF00FD")
    );
    assert_eq!(
        mit_switch_protocol_data(2).to_vec(),
        hex("FFFFFFFFFFFF02FD")
    );
    // MIT command 5 (F_CMD=0, read fault, side-effect-free probe ping).
    assert_eq!(mit_fault_query_data().to_vec(), hex("FFFFFFFFFFFF00FB"));
}

#[test]
fn parse_fault_frame() {
    // Case 1: Normal DLC-8 fault frame.
    // comm=21 (COMM_FAULT), motor=5, target=host (0xFD).
    // Motor ID 5 is distinctive to detect bit-shift errors.
    let id = (21u32 << 24) | (5 << 8) | 0xFD;
    assert_eq!(id, 0x150005FD);
    let data = hex("1122334455667788");
    let Some(ParsedFrame::Fault(fault)) = parse_frame(id, &data, &RS00) else {
        panic!("expected fault frame");
    };
    assert_eq!(fault.motor_id, 5);
    assert_eq!(fault.raw, [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);

    // Case 2: Short-payload fault frame — pins current, plan-mandated behavior.
    // Unlike COMM_FEEDBACK and COMM_READ_PARAM which reject data.len() < 8,
    // the COMM_FAULT arm zero-pads shorter payloads. This asymmetry is
    // unreachable in practice (all private-protocol frames are DLC 8), and
    // whether to reconcile it is deferred to the next milestone.
    let data_short = &[0xAA, 0xBB, 0xCC];
    let Some(ParsedFrame::Fault(fault)) = parse_frame(id, data_short, &RS00) else {
        panic!("expected fault frame with short payload");
    };
    assert_eq!(fault.motor_id, 5);
    assert_eq!(fault.raw, [0xAA, 0xBB, 0xCC, 0x00, 0x00, 0x00, 0x00, 0x00]);
}

#[test]
fn encode_feedback_inverts_parse() {
    let fb = MotorFeedback {
        motor_id: 3,
        position: 1.234,
        velocity: -2.5,
        torque: 0.75,
        temperature: 34.5,
        mode: 2,
        fault_bits: 0,
    };
    let f = encode_feedback(&fb, &RS00, HOST_CAN_ID);
    // 29-bit id: comm=2 | mode in bits 23..22 | fault in 21..16 | motor in 15..8 | host
    assert_eq!(f.id, (2u32 << 24) | (2 << 22) | (3 << 8) | 0xFD);
    let Some(ParsedFrame::Feedback(back)) = parse_frame(f.id, &f.data, &RS00) else {
        panic!("expected feedback");
    };
    assert_eq!(back.motor_id, 3);
    assert_eq!(back.mode, 2);
    assert_eq!(back.fault_bits, 0);
    approx(back.position, 1.234, 25.14 / 65535.0);
    approx(back.velocity, -2.5, 66.0 / 65535.0);
    approx(back.torque, 0.75, 28.0 / 65535.0);
    approx(back.temperature, 34.5, 1e-9);

    // fault bits and the golden zero-pose frame
    let fb2 = MotorFeedback {
        motor_id: 1,
        fault_bits: 0x21,
        ..fb
    };
    let f2 = encode_feedback(&fb2, &RS00, HOST_CAN_ID);
    assert_eq!((f2.id >> 16) & 0x3F, 0x21);
    let zero = MotorFeedback {
        motor_id: 1,
        position: 0.0,
        velocity: 0.0,
        torque: 0.0,
        temperature: 34.5,
        mode: 2,
        fault_bits: 0,
    };
    let f3 = encode_feedback(&zero, &RS00, HOST_CAN_ID);
    assert_eq!(f3.id, 0x028001FD);
    assert_eq!(f3.data.to_vec(), hex("8000800080000159"));
}
