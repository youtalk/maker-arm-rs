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
