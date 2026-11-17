//! MIT codec vectors. Control and parameter frames come from the vendor
//! SDK's protocol.py at 0245316 (Apache-2.0, see NOTICE).
use maker_arm_protocol::mit::*;
use maker_arm_protocol::{Frame, ParamValue, RS00, RS02};

fn hex(s: &str) -> [u8; 8] {
    let mut d = [0u8; 8];
    for (i, b) in d.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    d
}

#[test]
fn commands_are_standard_frames_to_the_motor_id() {
    assert_eq!(
        encode_enable(7),
        Frame {
            id: 7,
            data: hex("FFFFFFFFFFFFFFFC")
        }
    );
    assert_eq!(encode_disable(7).data, hex("FFFFFFFFFFFFFFFD"));
    assert_eq!(encode_set_zero(7).data, hex("FFFFFFFFFFFFFFFE"));
    assert_eq!(encode_clear_fault(7).data, hex("FFFFFFFFFFFFFFFB"));
    assert_eq!(encode_fault_query(7).data, hex("FFFFFFFFFFFF00FB"));
}

#[test]
fn control_matches_the_vendor_oracle() {
    let cases = [
        (3, 0.0, 0.0, 500.0, 5.0, 0.0, &RS02, "8000800ffffff800"),
        (1, 0.0, 0.0, 0.0, 0.0, 0.0, &RS00, "8000800000000800"),
        (1, 1.0, 0.0, 60.0, 4.0, 0.5, &RS00, "8a2e8001ebccc849"),
        (2, -1.0, 0.0, 150.0, 4.5, 3.0, &RS02, "75d18004cce66969"),
        (4, 0.3, -1.2, 30.0, 2.0, -0.4, &RS00, "830e7b50f66667c5"),
        (1, 99.0, 0.0, 0.0, 0.0, 0.0, &RS00, "ffff800000000800"),
    ];
    for (id, p, v, kp, kd, t, params, want) in cases {
        let f = encode_control(id, p, v, kp, kd, t, params).unwrap();
        assert_eq!((f.id, f.data), (id as u32, hex(want)));
    }
}

#[test]
fn control_refuses_non_finite() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for i in 0..5 {
            let mut x = [0.0; 5];
            x[i] = bad;
            assert!(encode_control(1, x[0], x[1], x[2], x[3], x[4], &RS00).is_none());
        }
    }
}

#[test]
fn control_clamps_out_of_range() {
    assert_eq!(
        &encode_control(1, 99.0, 0.0, 0.0, 0.0, 0.0, &RS00)
            .unwrap()
            .data[..2],
        &[0xFF, 0xFF]
    );
}

#[test]
fn control_refuses_the_reserved_prefix() {
    // Every field at its top: bytes 0 to 5 are all FF.
    assert!(encode_control(1, 99.0, 99.0, 500.0, 5.0, 99.0, &RS00).is_none());
}

#[test]
fn decode_control_inverts_encode() {
    let f = encode_control(4, 0.3, -1.2, 30.0, 2.0, -0.4, &RS00).unwrap();
    let (p, v, kp, kd, t) = decode_control(&f.data, &RS00).unwrap();
    let near = |a: f64, b: f64, q: f64| assert!((a - b).abs() <= q, "{a} vs {b}");
    near(p, 0.3, 2.0 * 12.57 / 65535.0);
    near(v, -1.2, 2.0 * 33.0 / 4095.0);
    near(kp, 30.0, 500.0 / 4095.0);
    near(kd, 2.0, 5.0 / 4095.0);
    near(t, -0.4, 2.0 * 14.0 / 4095.0);
    assert_eq!(decode_control(&encode_enable(1).data, &RS00), None);
}

#[test]
fn p0_feedback_decodes() {
    // The real arm's motor 1 at rest, captured in P0 on 2026-11-16.
    let Some(Parsed::Feedback(f)) = parse_frame(0xFD, &hex("01BE907FD7FF00F0"), &RS00) else {
        panic!()
    };
    // Values from the vendor SDK's parse_frame at 0245316.
    assert_eq!(f.motor_id, 1);
    assert!((f.position - 6.144118562600138).abs() < 1e-12);
    assert!((f.velocity + 0.0402930402930437).abs() < 1e-12);
    assert!((f.torque + 0.0034188034188034067).abs() < 1e-12);
    assert_eq!(
        (f.temperature, f.mode, f.fault, f.warning),
        (24.0, 0, false, false)
    );
}

#[test]
fn p0_fault_status_decodes() {
    assert_eq!(
        parse_frame(0xFD, &hex("0100000000000000"), &RS00),
        Some(Parsed::Fault(FaultStatus {
            motor_id: 1,
            word: 0
        }))
    );
}

#[test]
fn status_bits_decode() {
    // byte 6 = 0xB1: mode 2, fault, warning, temperature high nibble 1. Byte 7 = 0x9D: 0x19D = 41.3 C.
    let Some(Parsed::Feedback(f)) = parse_frame(0xFD, &hex("058000800800B19D"), &RS00) else {
        panic!()
    };
    assert_eq!(
        (f.mode, f.fault, f.warning, f.temperature),
        (2, true, true, 41.3)
    );
}

#[test]
fn param_replies_are_attributed_by_id() {
    let d = hex("2870000010270000");
    assert_eq!(
        parse_frame(0x307, &d, &RS00),
        Some(Parsed::Param(ParamReply {
            motor_id: 7,
            index: 0x7028,
            raw: [0x10, 0x27, 0, 0],
            write: false
        }))
    );
    assert!(matches!(
        parse_frame(0x407, &d, &RS00),
        Some(Parsed::Param(ParamReply { write: true, .. }))
    ));
    let r = ParamReply {
        motor_id: 7,
        index: 0x7028,
        raw: [0x10, 0x27, 0, 0],
        write: false,
    };
    assert_eq!(r.as_u32(), 10000);
    assert_eq!(r.as_u8(), 0x10);
    assert_eq!(f32::from_le_bytes(r.raw), r.as_f32());
}

#[test]
fn param_frames_match_the_vendor_layout() {
    assert_eq!(
        encode_read_param(7, 0x7005),
        Frame {
            id: 0x307,
            data: hex("0570000000000000")
        }
    );
    assert_eq!(
        encode_write_param(7, 0x7028, ParamValue::U32(10000)),
        Frame {
            id: 0x407,
            data: hex("2870000010270000")
        }
    );
    assert_eq!(
        encode_write_param(7, 0x7005, ParamValue::U8(0)).data,
        hex("0570000000000000")
    );
}

#[test]
fn power_up_announcements_are_ignored() {
    let uid = hex("0102030405060708"); // synthetic: never the owner's MCU ids
    assert_eq!(parse_frame(3, &uid, &RS00), None);
    assert_eq!(reply_motor_id(3, &uid), None);
}

#[test]
fn reply_motor_id_reads_the_right_place() {
    let d = hex("0570000000000000");
    assert_eq!(reply_motor_id(0xFD, &d), Some(5));
    assert_eq!(reply_motor_id(0x307, &d), Some(7));
    assert_eq!(reply_motor_id(0x47F, &d), Some(0x7F));
}

#[test]
fn encoders_round_trip_through_parse() {
    let fb = Feedback {
        motor_id: 5,
        position: 1.0,
        velocity: -2.0,
        torque: 0.5,
        temperature: 41.3,
        mode: 2,
        fault: true,
        warning: false,
    };
    let Some(Parsed::Feedback(g)) = parse_frame(0xFD, &encode_feedback(&fb, &RS00).data, &RS00)
    else {
        panic!()
    };
    assert_eq!(
        (g.motor_id, g.mode, g.fault, g.warning),
        (5, 2, true, false)
    );
    assert_eq!(g.temperature, 41.3);
    assert!((g.position - 1.0).abs() < 1e-3);
    assert_eq!(
        parse_frame(0xFD, &encode_fault_status(3, 0x4004).data, &RS00),
        Some(Parsed::Fault(FaultStatus {
            motor_id: 3,
            word: 0x4004
        }))
    );
    let r = ParamReply {
        motor_id: 7,
        index: 0x7028,
        raw: [1, 2, 3, 4],
        write: true,
    };
    assert_eq!(
        parse_frame(0x407, &encode_param_reply(&r).data, &RS00),
        Some(Parsed::Param(r))
    );
}

#[test]
fn fault_text_names_the_vendor_bits() {
    assert_eq!(fault_text(0), "no fault");
    assert_eq!(
        fault_text((1 << 2) | (1 << 14)),
        "undervoltage, stall/overload"
    );
    assert_eq!(fault_text(1 << 6), "bit 6");
}
