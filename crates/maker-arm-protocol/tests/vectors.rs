//! Golden vectors ported from maker-arm-sdk tests/unit/test_protocol.py
//! (main @ 2026-08-20). Do not edit values without re-checking upstream.
use maker_arm_protocol::*;

fn approx(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} !~ {b} (tol {tol})");
}

#[test]
fn constants() {
    assert_eq!(P_MAX, 12.57);
    assert_eq!(P_MIN, -12.57);
    assert_eq!(V_MAX, 33.0);
    assert_eq!(T_MAX, 14.0);
    assert_eq!(KP_MAX, 500.0);
    assert_eq!(KD_MAX, 5.0);
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
    assert_eq!(&f.data[4..8], &2.0f32.to_le_bytes());
    let f = encode_save_params(1, HOST_CAN_ID);
    assert_eq!(f.id, 0x1600FD01);
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
