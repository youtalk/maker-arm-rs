"""Golden parity with the protocol vectors — the same numbers as the Rust
tests and the official SDK, asserted through the Python surface."""

import maker_arm_rs as m


def test_encode_mit_golden():
    can_id, data = m.encode_mit(1, 0.0, 0.0, 10.0, 1.0, 0.0, "RS00")
    assert can_id == 0x01800001
    assert data == bytes.fromhex("80008000051F3333")


def test_encode_mit_rs02_scaling_differs():
    # tau = 14.0 Nm is RS00's exact max (t_min/t_max = -14/+14) but only
    # ~82% of RS02's range (t_min/t_max = -17/+17). encode_mit puts the
    # feed-forward torque as a u16 in bits 23..8 of the CAN id (via
    # make_can_id(COMM_MIT, tau_u16, motor_id)), computed by
    # float_to_u16(tau, t_min, t_max) with round-half-to-even.
    #
    # RS00: x = clamp(14.0, -14, 14) = 14.0
    #       tau_u16 = round_ties_even((14 - (-14)) * 65535 / 28)
    #               = round_ties_even(65535.0) = 65535 = 0xFFFF
    #       id = (COMM_MIT=1 << 24) | (0xFFFF << 8) | motor_id=1
    #          = 0x01000000 | 0x00FFFF00 | 0x00000001 = 0x01FFFF01
    #
    # RS02: x = clamp(14.0, -17, 17) = 14.0
    #       tau_u16 = round_ties_even((14 - (-17)) * 65535 / 34)
    #               = round_ties_even(31 * 65535 / 34)
    #               = round_ties_even(59752.5)
    #       59752.5 is exactly halfway between 59752 (even) and 59753 (odd),
    #       so round-half-to-even picks 59752 = 0xE968.
    #       id = 0x01000000 | (0xE968 << 8) | 0x00000001 = 0x01E96801
    rs00_id, rs00_data = m.encode_mit(1, 0.0, 0.0, 0.0, 0.0, 14.0, "RS00")
    rs02_id, rs02_data = m.encode_mit(1, 0.0, 0.0, 0.0, 0.0, 14.0, "RS02")

    assert rs00_id == 0x01FFFF01
    assert rs02_id == 0x01E96801
    assert rs00_id != rs02_id  # same torque, different mapping table

    # The payload only carries pos/vel/kp/kd (tau lives in the CAN id).
    # pos=vel=0.0 map to the midpoint of their respective ranges, and
    # since P_MIN/P_MAX (both models) and V_MIN/V_MAX (per model, but each
    # symmetric about 0) are all symmetric around zero, 0.0 always maps to
    # the same u16 (32768) regardless of the range's width. kp=kd=0.0 map
    # to 0 for both models too (KP/KD ranges don't vary by model). So the
    # two payloads are expected to be identical even though the torque
    # differs — the difference is confined entirely to the CAN id.
    assert rs00_data == rs02_data == bytes.fromhex("8000800000000000")


def test_parse_feedback_golden():
    fb = m.parse_frame(0x028001FD, bytes.fromhex("8000800080000159"), "RS00")
    assert fb["kind"] == "feedback"
    assert fb["motor_id"] == 1
    assert fb["mode"] == 2
    assert fb["fault_bits"] == 0
    assert abs(fb["temperature"] - 34.5) < 1e-9


def test_parse_unknown_returns_none():
    # Contrast with test_parse_frame_unknown_model_raises below: an
    # unrecognized *frame* (unknown comm type in the CAN id) is not an
    # error — parse_frame just has nothing to report and returns None.
    assert m.parse_frame((22 << 24) | 0xFD, bytes(8), "RS00") is None


def test_unknown_model_raises():
    import pytest

    with pytest.raises(ValueError):
        m.encode_mit(1, 0, 0, 0, 0, 0, "RS99")


def test_parse_frame_unknown_model_raises():
    # parse_frame resolves its `model` argument through the same params()
    # helper as encode_mit, so an unknown *model* raises ValueError — unlike
    # an unknown *frame* (test_parse_unknown_returns_none above), which
    # returns None. Use a well-known feedback frame (same id/data as
    # test_parse_feedback_golden) so the only variable is the model name.
    import pytest

    with pytest.raises(ValueError):
        m.parse_frame(0x028001FD, bytes.fromhex("8000800080000159"), "RS99")
