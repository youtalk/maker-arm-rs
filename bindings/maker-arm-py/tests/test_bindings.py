"""Golden parity with the protocol vectors — the same numbers as the Rust
tests and the official SDK, asserted through the Python surface."""

import threading
import time

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


def test_arm_sim_orchestration_lifecycle():
    import time

    arm = m.Arm.sim()
    assert arm.state() == "connected"
    arm.enable()
    assert arm.state() == "enabled"
    arm.start_hold()
    time.sleep(0.1)
    snap = arm.snapshot()
    assert snap is not None
    assert snap["state"] == "enabled"
    assert snap["fault"] is None
    assert len(snap["positions"]) == 7
    assert snap["tick"] > 0
    arm.hold_now()
    time.sleep(0.05)
    arm.stop()
    assert arm.state() == "connected"


def test_arm_stop_before_start_is_safe():
    arm = m.Arm.sim()
    arm.stop()  # no loop running: must not raise
    assert arm.state() == "connected"


def test_start_hold_requires_an_enabled_session():
    # A merely-connected session has no torque. start_hold() used to spawn
    # a loop anyway; it died on its first tick with a wrong-state error,
    # snapshot() then returned None forever, and state() reported
    # "enabled" for an un-energized arm -- an operator told the arm is
    # holding might stop supporting it.
    import pytest

    arm = m.Arm.sim()
    assert arm.state() == "connected"
    with pytest.raises(RuntimeError):
        arm.start_hold()
    # The refusal must not consume the handle: enable() then start_hold()
    # still works on the same object.
    assert arm.state() == "connected"
    arm.enable()
    arm.start_hold()
    arm.stop()
    assert arm.state() == "connected"


def test_state_and_snapshot_report_loop_liveness():
    # state() must never default to "enabled". While the loop is alive it
    # reports what the loop published; the dead-loop reading
    # ("loop_stopped", driven by JoinHandle::is_finished) is pinned in the
    # Rust suite instead -- no Python-reachable path kills a SimArm loop
    # now that start_hold() is gated, and adding a fault-injection hook to
    # the binding just to reach it would be new public API for a test.
    import time

    arm = m.Arm.sim()
    arm.enable()
    arm.start_hold()
    time.sleep(0.1)
    snap = arm.snapshot()
    assert snap is not None
    assert snap["loop_alive"] is True
    assert snap["state"] == "enabled"
    assert arm.state() == "enabled"
    arm.stop()
    # joined and back to an idle session: its own state, and no snapshot
    assert arm.state() == "connected"
    assert arm.snapshot() is None


import math


def test_profile_pins_every_upstream_joint():
    # Same table as crates/maker-arm/src/config.rs::v1_profile_matches_upstream_yaml,
    # asserted through the Python surface so the binding cannot drift from the crate.
    p = m.profile()
    expected = [
        (1, "j1", "RS00", -0.668, 4.818, 60.0, 4.0, 4.0),
        (2, "j2", "RS02", -2.024, 0.979, 150.0, 4.5, 12.0),
        (3, "j3", "RS02", 3.882, 7.955, 90.0, 3.0, 10.0),
        (4, "j4", "RS00", -0.832, 2.122, 30.0, 2.0, 4.0),
        (5, "j5", "RS00", 0.577, 3.641, 30.0, 2.0, 4.0),
        (6, "j6", "RS00", 0.966, 6.292, 30.0, 2.0, 4.0),
        (7, "gripper", "RS00", -2.092, -0.039, 20.0, 0.5, 2.0),
    ]
    assert len(p["joints"]) == 7
    for j, (mid, name, model, q_lo, q_hi, kp, kd, tau_max) in zip(p["joints"], expected):
        assert j["motor_id"] == mid
        assert j["name"] == name
        assert j["model"] == model
        assert j["q_lo"] == q_lo and j["q_hi"] == q_hi
        assert j["kp"] == kp and j["kd"] == kd and j["tau_max"] == tau_max
        assert j["direction"] == 1.0 and j["offset"] == 0.0
    assert p["control_rate_hz"] == 200.0
    assert p["max_velocity"] == 5.0
    assert p["feedback_timeout"] == 0.2
    assert p["limit_margin"] == 0.0
    assert p["kp_max"] == 200.0
    assert p["temp_hold_c"] == 70.0
    assert math.isfinite(p["kd_max"]) and p["kd_max"] > 0.0


def test_profile_returns_a_fresh_dict_each_call():
    a = m.profile()
    a["joints"][0]["q_lo"] = -99.0
    assert m.profile()["joints"][0]["q_lo"] == -0.668


import pytest


def _hold_cmds(p, pos=None):
    pos = pos if pos is not None else [0.0] * 7
    return [(pos[i], 0.0, j["kp"], j["kd"], 0.0) for i, j in enumerate(p["joints"])]


def test_clamp_command_passes_in_range_commands_unchanged():
    p = m.profile()
    mid = [(j["q_lo"] + j["q_hi"]) / 2.0 for j in p["joints"]]
    out, clamped = m.clamp_command(_hold_cmds(p, mid), p)
    assert clamped is False
    assert out == _hold_cmds(p, mid)


def test_clamp_command_clamps_position_to_profile_limits():
    p = m.profile()
    cmds = _hold_cmds(p)
    cmds[0] = (100.0, 0.0, cmds[0][2], cmds[0][3], 0.0)
    out, clamped = m.clamp_command(cmds, p)
    assert clamped is True
    assert out[0][0] == p["joints"][0]["q_hi"]


def test_clamp_command_honours_numeric_overrides():
    p = m.profile()
    p["joints"][2]["q_lo"], p["joints"][2]["q_hi"] = 0.0, 3.14  # URDF limits for j3
    p["limit_margin"] = 0.1
    cmds = _hold_cmds(p, [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    out, clamped = m.clamp_command(cmds, p)
    assert clamped is True
    assert out[2][0] == pytest.approx(3.04)


def test_clamp_command_caps_gains_velocity_and_torque():
    p = m.profile()
    cmds = _hold_cmds(p)
    cmds[1] = (0.0, 50.0, 1e6, 1e6, 1e6)
    out, clamped = m.clamp_command(cmds, p)
    assert clamped is True
    assert out[1][1] == p["max_velocity"]
    assert out[1][2] == p["kp_max"]
    assert out[1][3] == p["kd_max"]
    assert out[1][4] == p["joints"][1]["tau_max"]


def test_clamp_command_rejects_non_finite_and_wrong_length():
    p = m.profile()
    with pytest.raises(ValueError):
        m.clamp_command(_hold_cmds(p)[:6], p)
    bad = _hold_cmds(p)
    bad[3] = (float("nan"), 0.0, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        m.clamp_command(bad, p)


# --- Kinematics: the lab's global IK behind a thin surface --------------------------

import math

import pytest


def _planar_chain():
    """Two moving links (z then y), four inert joints at the same point, a fixed tool link."""
    inert = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), axis) for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0))]
    return m.Kinematics(
        mount_rpy=(0.0, 0.0, 0.0),
        links=[((0.0, 0.0, 0.1), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), ((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
        + inert
        + [((0.3, 0.0, 0.0), (0.0, 0.0, 0.0), None)],
        lower=[-3.0] * 6,
        upper=[3.0] * 6,
        tool_axis=(1.0, 0.0, 0.0),
        jaw_axis=(0.0, 1.0, 0.0),
    )


def test_kinematics_fk_walks_the_chain():
    pos, rot = _planar_chain().fk([math.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert pos == pytest.approx([0.0, 0.8, 0.1], abs=1e-15)
    assert rot[1][0] == pytest.approx(1.0) and rot[0][0] == pytest.approx(0.0, abs=1e-15)
    batch_pos, batch_rot = _planar_chain().fk_batch([[0.0] * 6, [math.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert batch_pos[1] == pytest.approx(pos) and batch_rot[1] == rot and len(batch_pos) == 2


def test_kinematics_pose_is_a_target_and_its_residual_vanishes():
    k = _planar_chain()
    q = [0.3, 0.6, 0.1, 0.2, 0.3, 0.4]
    pose = k.pose_of(q)
    position, tilt, jaw_heading, lean_azimuth = pose
    assert len(position) == 3 and 0.0 < tilt < math.pi
    assert max(abs(v) for v in k.residual(q, pose)) < 1e-12
    assert k.margins(q) == pytest.approx([2.7, 2.4, 2.9, 2.8, 2.7, 2.6])


def test_kinematics_jacobian_matches_a_coarser_central_difference():
    k = _planar_chain()
    q = [0.3, 0.6, 0.1, 0.2, 0.3, 0.4]
    target = k.pose_of(q)
    jac = k.residual_jacobian(q, target)
    assert len(jac) == 6 and all(len(row) == 6 for row in jac)
    eps = 1e-5
    for j in range(6):
        plus, minus = list(q), list(q)
        plus[j] += eps
        minus[j] -= eps
        rp, rm = k.residual(plus, target), k.residual(minus, target)
        for i in range(6):
            assert jac[i][j] == pytest.approx((rp[i] - rm[i]) / (2 * eps), abs=1e-6)


def _generic_chain():
    """Base yaw, three pitch joints, wrist yaw, wrist roll, tool: six constraints on six
    joints, so a pose pins its joint vector (the planar chain's wrist is redundant)."""
    links = [
        ((0.0, 0.0, 0.1), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.05), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.3, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.05, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.05, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.05, 0.0, 0.0), (0.0, 0.0, 0.0), None),
    ]
    return m.Kinematics(
        mount_rpy=(0.0, 0.0, 0.0), links=links, lower=[-3.0] * 6, upper=[3.0] * 6,
        tool_axis=(1.0, 0.0, 0.0), jaw_axis=(0.0, 1.0, 0.0),
    )


def test_kinematics_solve_recovers_a_pose_and_solve_path_walks():
    k = _generic_chain()
    q_true = [0.3, 0.3, 0.4, 0.3, 0.2, 0.1]
    target = k.pose_of(q_true)
    seed = [v + 0.1 for v in q_true]
    solution = k.solve(target, [seed])
    assert solution["position_error"] < 1e-5 and solution["q"] == pytest.approx(q_true, abs=1e-6)
    assert solution["margin"] == pytest.approx(2.6) and solution["step"] == 0.0
    assert k.solve(target, [seed], q_prev=seed, max_step=0.01) is None
    assert k.refine(seed, target) == pytest.approx(solution["q"])
    targets = [k.pose_of([a + t * (b - a) for a, b in zip(q_true, seed)]) for t in (0.5, 1.0)]
    path = k.solve_path(targets, q_true, [], 0.2)
    assert len(path) == 2 and path[-1]["q"] == pytest.approx(seed, abs=1e-6)
    far = ((10.0, 10.0, 10.0), 0.5, 0.0, 0.0)
    assert k.solve_path([targets[0], far], q_true, [], 0.2) is None


def test_start_hold_with_an_impossible_rt_fails_and_keeps_the_session():
    a = m.Arm.sim()
    a.enable()
    with pytest.raises(RuntimeError, match="CPU 1000"):
        a.start_hold(rt=(80, 1000))
    assert a.state() == "enabled"
    a.start_hold()  # the session is still usable
    a.stop()


def test_kinematics_rejects_a_chain_without_six_revolute_joints():
    with pytest.raises(ValueError, match="revolute"):
        m.Kinematics(mount_rpy=(0, 0, 0), links=[], lower=[0.0] * 6, upper=[1.0] * 6, tool_axis=(1, 0, 0), jaw_axis=(0, 1, 0))


# --- Dynamics: RNEA, mass matrix, gravity and friction behind a thin surface ---------

Z_AXIS = (0.0, 0.0, 1.0)
CHAIN = [((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), Z_AXIS)] * 6
BODY = (0.5, (0.05, 0.0, 0.0), ((1e-3, 0.0, 0.0), (0.0, 2e-3, 0.0), (0.0, 0.0, 2e-3)))


def _planar_dynamics(gravity=(0.0, -9.81, 0.0)):
    return m.Dynamics((0.0, 0.0, 0.0), CHAIN, [BODY] * 6, gravity=gravity)


def test_dynamics_gravity_of_a_planar_chain():
    # Joint i sits at x = 0.1 (i + 1), body k's center at 0.1 (k + 1) + 0.05; at q = 0 the
    # chain lies along +x, so G_i = m g sum_{k >= i} (0.1 (k - i) + 0.05).
    d = _planar_dynamics()
    g = d.gravity([0.0] * 6)
    for i in range(6):
        want = 0.5 * 9.81 * sum(0.1 * (k - i) + 0.05 for k in range(i, 6))
        assert abs(g[i] - want) < 1e-9
    assert all(abs(v) < 1e-12 for v in _planar_dynamics((0.0, 0.0, -9.81)).gravity([0.3] * 6))


def test_dynamics_mass_matrix_and_inverse_agree():
    d = _planar_dynamics()
    q = [0.1, -0.4, 0.7, 0.2, -0.3, 0.5]
    m_ = d.mass_matrix(q)
    g = d.gravity(q)
    for i in range(6):
        e = [0.0] * 6
        e[i] = 1.0
        col = d.inverse(q, [0.0] * 6, e)
        assert all(abs(col[k] - g[k] - m_[k][i]) < 1e-9 for k in range(6))
    assert d.friction([1.0] * 6) == [0.0] * 6


def test_dynamics_refuses_bad_models():
    bad = (-1.0, BODY[1], BODY[2])
    with pytest.raises(ValueError, match="mass"):
        m.Dynamics((0.0, 0.0, 0.0), CHAIN, [BODY] * 5 + [bad])
    with pytest.raises(ValueError, match="6 bodies"):
        m.Dynamics((0.0, 0.0, 0.0), CHAIN, [BODY] * 5)
    with pytest.raises(ValueError, match="friction"):
        m.Dynamics((0.0, 0.0, 0.0), CHAIN, [BODY] * 6, friction=[(0.1, 0.0, 0.0)] * 6)



def test_dynamics_armature_adds_to_the_diagonal_and_to_inverse():
    arm = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    plain = _planar_dynamics()
    heavy = m.Dynamics((0.0, 0.0, 0.0), CHAIN, [BODY] * 6, gravity=(0.0, -9.81, 0.0), armature=arm)
    q = [0.1, -0.4, 0.7, 0.2, -0.3, 0.5]
    ddq = [1.0, -2.0, 0.5, 3.0, -1.0, 0.25]
    a, b = plain.inverse(q, [0.0] * 6, ddq), heavy.inverse(q, [0.0] * 6, ddq)
    assert all(abs(b[j] - a[j] - arm[j] * ddq[j]) < 1e-12 for j in range(6))
    m0, m1 = plain.mass_matrix(q), heavy.mass_matrix(q)
    for i in range(6):
        for j in range(6):
            assert abs(m1[i][j] - m0[i][j] - (arm[i] if i == j else 0.0)) < 1e-12


def test_dynamics_friction_takes_three_or_five_entries():
    three = m.Dynamics((0, 0, 0), CHAIN, [BODY] * 6, friction=[(0.2, 0.05, 0.01)] * 6)
    five_off = m.Dynamics((0, 0, 0), CHAIN, [BODY] * 6, friction=[(0.2, 0.05, 0.01, 0.2, 1.0)] * 6)
    dq = [0.05] * 6
    assert three.friction(dq) == five_off.friction(dq)
    stribeck = m.Dynamics((0, 0, 0), CHAIN, [BODY] * 6, friction=[(0.2, 0.05, 0.01, 0.3, 0.1)] * 6)
    assert stribeck.friction(dq)[0] > three.friction(dq)[0] + 0.01
    mixed = [(0.2, 0.05, 0.01)] * 3 + [(0.2, 0.05, 0.01, 0.3, 0.1)] * 3
    out = m.Dynamics((0, 0, 0), CHAIN, [BODY] * 6, friction=mixed).friction(dq)
    assert out[0] == three.friction(dq)[0] and out[5] == stribeck.friction(dq)[5]


def test_dynamics_refuses_a_bad_stribeck_or_armature():
    kw = dict(mount_rpy=(0, 0, 0), links=CHAIN, bodies=[BODY] * 6)
    with pytest.raises(ValueError, match="static_"):
        m.Dynamics(**kw, friction=[(0.2, 0.0, 0.01, 0.1, 1.0)] * 6)
    with pytest.raises(ValueError, match="stribeck_v"):
        m.Dynamics(**kw, friction=[(0.2, 0.0, 0.01, 0.2, 0.0)] * 6)
    with pytest.raises(ValueError, match="friction"):
        m.Dynamics(**kw, friction=[(0.2, 0.0, 0.01, 0.2)] * 6)
    with pytest.raises(ValueError, match="armature"):
        m.Dynamics(**kw, armature=[0.0, 0.0, -0.1, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="armature"):
        m.Dynamics(**kw, armature=[float("nan")] + [0.0] * 5)
    with pytest.raises(ValueError, match="6 armature"):
        m.Dynamics(**kw, armature=[0.0] * 5)


# --- Tracking: the compliant tracking controller behind a thread-safe handle --------

def test_tracking_holds_the_start_pose_with_gravity_feed_forward():
    d = _planar_dynamics()
    tr = m.Tracking(d, params={"contact": False})
    q = [0.1, -0.2, 0.3, 0.0, 0.2, -0.1, -1.0]
    out = tr.update(0.0, q, [0.0] * 7, [0.0] * 7)
    assert len(out) == 7
    g = d.gravity(q[:6])
    for j in range(6):
        pos, vel, kp, kd, tau = out[j]
        assert pos == q[j] and vel == 0.0 and abs(tau - g[j]) < 1e-12
    assert out[6][0] == -1.0 and out[6][4] == 0.0
    tel = tr.telemetry()
    assert set(tel) == {
        "r", "gate", "offset", "q_r", "tau_ff", "tau_dob", "tau_res", "residual_skips",
        "late_ticks", "gaps", "max_interval", "tick_t",
    }
    assert tel["tau_res"] == [0.0] * 6 and tel["residual_skips"] == 0
    assert tel["tick_t"] == 0.0


def test_tracking_observer_compensation_cancels_a_held_push_below_the_deadband():
    d = _planar_dynamics()
    q = [0.1, -0.2, 0.3, 0.0, 0.2, -0.1, -1.0]
    tau = [0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]  # a static arm reads tau_ext = -tau_m
    g = d.gravity(q[:6])
    tau = [tau[j] + (g[j] if j < 6 else 0.0) for j in range(7)]
    off = m.Tracking(d, params={"tau_th": [0.5] * 6, "dob_f": 7.0})
    on = m.Tracking(d, params={"tau_th": [0.5] * 6, "dob": True, "dob_f": 7.0})
    for k in range(201):
        a = off.update(k * 0.005, q, [0.0] * 7, tau)
        b = on.update(k * 0.005, q, [0.0] * 7, tau)
    assert off.telemetry()["tau_dob"] == [0.0] * 6
    tel = on.telemetry()
    assert abs(tel["tau_dob"][1] + 0.3) < 1e-3
    assert abs(b[1][4] - (a[1][4] - 0.3)) < 1e-3
    assert on.drain()["tau_dob"][-1] == tel["tau_dob"]
    with pytest.raises(ValueError, match="dob"):
        m.Tracking(d, params={"dob": True})  # the default tau_th is infinite
    with pytest.raises(ValueError, match="dob_f"):
        m.Tracking(d, params={"tau_th": [0.5] * 6, "dob": True, "dob_f": 0.0})


def _residual(history=2, bias=(0.0,) * 6, out_max=1.0, seed=7):
    """The dict the lab's export writes: 12 * history + 6 -> 8 -> 6, torch's weight.tolist()."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = 12 * history + 6
    w0, b0 = rng.normal(0.0, 0.3, (8, n)), rng.normal(0.0, 0.1, 8)
    w1, b1 = rng.normal(0.0, 0.3, (6, 8)), np.asarray(bias, dtype=float)
    return {
        "history": history,
        "layers": [[w0.tolist(), b0.tolist()], [w1.tolist(), b1.tolist()]],
        "x_mean": rng.normal(0.0, 0.1, n).tolist(),
        "x_std": rng.uniform(0.5, 2.0, n).tolist(),
        "y_scale": [0.5] * 6,
        "out_max": [out_max] * 6,
    }


def test_mlp_matches_numpy_on_the_export_format():
    import numpy as np

    r = _residual()
    x = np.linspace(-1.0, 1.0, 30)
    (w0, b0), (w1, b1) = [(np.array(w), np.array(b)) for w, b in r["layers"]]
    want = w1 @ np.tanh(w0 @ x + b0) + b1
    got = m.Mlp(r["layers"]).forward(x.tolist())
    assert len(got) == 6
    assert np.max(np.abs(np.array(got) - want)) < 1e-12
    with pytest.raises(ValueError, match="layer 1"):
        m.Mlp([r["layers"][0], [[[0.0] * 7] * 6, [0.0] * 6]])
    with pytest.raises(ValueError, match="layer 0"):
        m.Mlp([[[[0.0] * 3, [0.0] * 2], [0.0, 0.0]]])  # ragged rows
    with pytest.raises(ValueError):
        m.Mlp(r["layers"]).forward([0.0] * 29)


def test_tracking_residual_enters_the_feed_forward_and_the_telemetry():
    import numpy as np

    d = _planar_dynamics()
    r = _residual(history=2)
    q = [0.1, -0.2, 0.3, 0.0, 0.2, -0.1, -1.0]
    temps = [31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 30.0]
    plain = m.Tracking(d, params={"contact": False})
    tr = m.Tracking(d, params={"contact": False}, residual=r)
    for k in range(3):
        a = plain.update(k * 0.005, q, [0.0] * 7, [0.0] * 7)
        b = tr.update(k * 0.005, q, [0.0] * 7, [0.0] * 7, temperature=temps)
    # The reference by hand: two ticks of [q, dq], oldest first, then the temperatures.
    x = np.array((q[:6] + [0.0] * 6) * 2 + temps[:6])
    x = (x - np.array(r["x_mean"])) / np.array(r["x_std"])
    (w0, b0), (w1, b1) = [(np.array(w), np.array(b)) for w, b in r["layers"]]
    want = np.clip((w1 @ np.tanh(w0 @ x + b0) + b1) * 0.5, -1.0, 1.0)
    tel = tr.telemetry()
    assert np.max(np.abs(np.array(tel["tau_res"]) - want)) < 1e-12
    assert tel["residual_skips"] == 1  # the first tick
    for j in range(6):
        assert abs(b[j][4] - (a[j][4] + want[j])) < 1e-12
    dr = tr.drain()
    assert dr["residual_skips"] == [1, 1, 1] and dr["tau_res"][0] == [0.0] * 6
    assert plain.telemetry()["tau_res"] == [0.0] * 6


def test_tracking_refuses_a_bad_residual():
    d = _planar_dynamics()
    r = _residual(history=2)
    with pytest.raises(ValueError, match="layer 0"):
        m.Tracking(d, residual=dict(r, history=3))
    with pytest.raises(ValueError, match="x_std"):
        m.Tracking(d, residual=dict(r, x_std=[0.0] * 30))
    with pytest.raises(ValueError, match="out_max"):
        m.Tracking(d, residual=dict(r, out_max=[-1.0] * 6))
    with pytest.raises(ValueError, match="unknown residual field"):
        m.Tracking(d, residual=dict(r, y_mean=[0.0] * 6))
    with pytest.raises(ValueError, match="missing"):
        m.Tracking(d, residual={k: v for k, v in r.items() if k != "x_mean"})


def test_tracking_rejects_bad_input():
    d = _planar_dynamics()
    with pytest.raises(ValueError, match="unknown tracking parameter"):
        m.Tracking(d, params={"f_R": 10.0})
    with pytest.raises(ValueError, match="f_r"):
        m.Tracking(d, params={"f_r": -1.0})
    tr = m.Tracking(d)
    with pytest.raises((ValueError, TypeError)):
        tr.update(0.0, [0.0] * 6, [0.0] * 7, [0.0] * 7)
    with pytest.raises(ValueError, match="push"):
        tr.push(0.0, 0.0, [[0.0] * 7])
    with pytest.raises(ValueError, match="push"):
        tr.push(0.0, 0.01, [])
    nan = float("nan")
    for bad in ([0.0] * 6 + [nan], [nan] + [0.0] * 6):
        with pytest.raises(ValueError, match="finite"):
            tr.update(0.0, bad, [0.0] * 7, [0.0] * 7)
        with pytest.raises(ValueError, match="finite"):
            tr.update(0.0, [0.0] * 7, bad, [0.0] * 7)
        with pytest.raises(ValueError, match="finite"):
            tr.update(0.0, [0.0] * 7, [0.0] * 7, bad)
    assert tr.telemetry() is None  # nothing reached the controller


def test_tracking_runs_in_the_loop_and_other_threads_can_reach_it():
    a = m.Arm.sim()
    a.enable()
    tr = m.Tracking(_planar_dynamics())
    a.start_tracking(tr)
    errors, clocks = [], []

    def worker():
        try:
            for _ in range(40):
                snap = a_positions[0]
                now = tr.now()
                if now is not None:
                    clocks.append(now)
                    tr.push(now + 0.02, 0.05, [snap, snap])
                tr.telemetry()
                time.sleep(0.005)
        except Exception as exc:  # noqa: BLE001 -- the assertion below reports it
            errors.append(exc)

    deadline = time.time() + 2.0
    while a.snapshot() is None and time.time() < deadline:
        time.sleep(0.01)
    a_positions = [list(a.snapshot()["positions"])]
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors == []
    assert clocks and clocks == sorted(clocks)
    assert tr.telemetry() is not None
    assert a.state() == "enabled"
    with pytest.raises(RuntimeError, match="loop"):
        tr.update(0.0, [0.0] * 7, [0.0] * 7, [0.0] * 7)
    b = m.Arm.sim()
    b.enable()
    with pytest.raises(RuntimeError, match="already runs"):
        b.start_tracking(tr)
    a.stop()
    b.stop()


def test_start_tracking_with_an_impossible_rt_keeps_the_tracking_usable():
    a = m.Arm.sim()
    a.enable()
    tr = m.Tracking(_planar_dynamics())
    with pytest.raises(RuntimeError, match="CPU 1000"):
        a.start_tracking(tr, rt=(80, 1000))
    assert a.state() == "enabled"
    a.start_tracking(tr)
    a.stop()


def test_start_tracking_requires_an_enabled_session():
    a = m.Arm.sim()
    with pytest.raises(RuntimeError, match="start_tracking requires an enabled session"):
        a.start_tracking(m.Tracking(_planar_dynamics()))


def test_a_stopped_loop_reads_as_not_running():
    # After stop() the loop no longer runs the controller: once the last tick is older than
    # the motors' CAN_TIMEOUT (0.2 s), now() is None and push raises.
    a = m.Arm.sim()
    a.enable()
    tr = m.Tracking(_planar_dynamics())
    a.start_tracking(tr)
    deadline = time.time() + 2.0
    while a.snapshot() is None and time.time() < deadline:
        time.sleep(0.01)
    tr.push(tr.now(), 0.01, [a.snapshot()["positions"]])  # accepted while the loop runs
    a.stop()
    tick_t = tr.telemetry()["tick_t"]
    time.sleep(0.3)
    assert tr.telemetry()["tick_t"] == tick_t  # the last tick's stamp shows the stop
    assert tr.now() is None
    with pytest.raises(RuntimeError, match="not running"):
        tr.push(0.0, 0.01, [[0.0] * 7])


def test_start_tracking_refuses_a_push_made_before_it():
    a = m.Arm.sim()
    a.enable()
    tr = m.Tracking(_planar_dynamics())
    tr.push(0.0, 0.01, [[0.0] * 7])
    with pytest.raises(RuntimeError, match="pending push"):
        a.start_tracking(tr)
    assert a.state() == "enabled" and a.snapshot() is None  # the arm stays idle
    tr.update(0.0, [0.0] * 7, [0.0] * 7, [0.0] * 7)  # and tr is not marked as in a loop
    a.stop()


def test_start_tracking_refuses_a_bench_tracking_with_a_plan():
    a = m.Arm.sim()
    a.enable()
    tr = m.Tracking(_planar_dynamics())
    tr.push(0.0, 0.01, [[0.0] * 7])
    tr.update(0.0, [0.0] * 7, [0.0] * 7, [0.0] * 7)
    with pytest.raises(RuntimeError, match="plan"):
        a.start_tracking(tr)
    assert a.state() == "enabled" and a.snapshot() is None
    a.stop()


def test_start_tracking_refuses_rung_f():
    a = m.Arm.sim()
    a.enable()
    tr = m.Tracking(_planar_dynamics(), params={"contact": False})
    with pytest.raises(RuntimeError, match="contact=False"):
        a.start_tracking(tr)
    assert a.state() == "enabled" and a.snapshot() is None
    assert len(tr.update(0.0, [0.0] * 7, [0.0] * 7, [0.0] * 7)) == 7  # still usable
    a.stop()


# --- The telemetry ring: every tick's state and sent command, drained from Python ------


def test_drain_returns_the_ticks_since_the_last_drain():
    arm = m.Arm.sim()
    arm.enable()
    arm.start_hold()
    time.sleep(0.25)
    d = arm.drain()
    assert 40 <= len(d["t"]) <= 60 and len(d["q"][0]) == 7 and d["dropped"] == 0
    for col in ("q", "dq", "tau", "temp", "fault_bits", "feedback_age"):
        assert len(d[col]) == len(d["t"]) and len(d[col][0]) == 7
    for col in ("cmd_pos", "cmd_vel", "cmd_kp", "cmd_kd", "cmd_tau"):
        assert len(d[col]) == len(d["t"]) and len(d[col][0]) == 7
    assert len(d["tick"]) == len(d["clamped"]) == len(d["t"])
    assert d["tick"] == list(range(d["tick"][0], d["tick"][0] + len(d["tick"])))
    time.sleep(0.05)
    assert arm.drain()["t"][0] > d["t"][-1]
    arm.stop()


def test_drain_is_none_without_a_loop():
    arm = m.Arm.sim()
    arm.enable()
    assert arm.drain() is None


def test_tracking_drain_gives_one_row_per_tick_and_counts_the_drops():
    a = m.Arm.sim()
    a.enable()
    tr = m.Tracking(_planar_dynamics())
    a.start_tracking(tr)
    time.sleep(0.25)
    loop = a.drain()
    d = tr.drain()
    a.stop()
    assert d["dropped"] == 0 and len(d["tick_t"]) >= len(loop["t"]) > 20
    assert d["tick_t"][: len(loop["t"])] == loop["t"]
    for col in ("r", "offset", "q_r", "tau_ff", "tau_dob", "tau_res"):
        assert len(d[col]) == len(d["tick_t"]) and len(d[col][0]) == 6
    assert len(d["gate"]) == len(d["tick_t"])
    assert tr.drain()["tick_t"] == []


# --- Reduced caps and joint calibration for an arm session ------------------------------


def _clamped_joint2_tau(profile):
    a = m.Arm.sim(profile=profile) if profile is not None else m.Arm.sim()
    a.enable()
    heavy = (100.0, BODY[1], BODY[2])  # gravity on joint 2 then asks for several N m
    a.start_tracking(m.Tracking(m.Dynamics((0.0, 0.0, 0.0), CHAIN, [heavy] * 6, gravity=(0.0, -9.81, 0.0))))
    time.sleep(0.3)
    d = a.drain()
    a.stop()
    return [row[1] for row in d["cmd_tau"]]


def test_a_reduced_cap_reaches_the_clamp():
    free = _clamped_joint2_tau(None)
    assert max(abs(t) for t in free) > 3.0  # the test bites: gravity asks for more than 3 N m
    p = m.profile()
    p["joints"][1]["tau_max"] = 3.0
    capped = _clamped_joint2_tau(p)
    assert capped and max(abs(t) for t in capped) <= 3.0 + 1e-9
    assert max(abs(t) for t in capped) > 2.9


def test_clamp_command_applies_a_reduced_tau_max_override():
    p = m.profile()
    p["joints"][1]["tau_max"] = 3.0
    out, clamped = m.clamp_command([(0.0, 0.0, 0.0, 0.0, 9.0)] * 7, p)
    assert clamped and out[1][4] == 3.0


@pytest.mark.parametrize(
    "edit, field",
    [
        (lambda p: p["joints"][1].update(tau_max=p["joints"][1]["tau_max"] * 1.5), "joints[1].tau_max"),
        (lambda p: p["joints"][2].update(kp=p["joints"][2]["kp"] * 2), "joints[2].kp"),
        (lambda p: p["joints"][3].update(kd=p["joints"][3]["kd"] + 1.0), "joints[3].kd"),
        (lambda p: p["joints"][0].update(q_lo=p["joints"][0]["q_lo"] - 0.1), "joints[0].q_lo"),
        (lambda p: p["joints"][4].update(q_hi=p["joints"][4]["q_hi"] + 0.1), "joints[4].q_hi"),
        (lambda p: p.update(max_velocity=p["max_velocity"] * 2), "max_velocity"),
        (lambda p: p["joints"][5].update(direction=0.5), "joints[5].direction"),
        (lambda p: p["joints"][5].update(direction=0.0), "joints[5].direction"),
        (lambda p: p["joints"][2].update(offset=3.2), "joints[2].offset"),
        (lambda p: p["joints"][2].update(offset=float("nan")), "joints[2].offset"),
        (lambda p: p["joints"][1].update(tau_max=float("inf")), "joints[1].tau_max"),
        (lambda p: p["joints"][1].update(kp=float("nan")), "joints[1].kp"),
        (lambda p: p.update(max_velocity=float("nan")), "max_velocity"),
        (lambda p: p.update(kp_max=-1.0), "kp_max"),
        (lambda p: p.update(kd_max=-1.0), "kd_max"),
        (lambda p: p.update(limit_margin=5.0), "limit_margin"),
        (lambda p: p["joints"][1].update(tau_mx=3.0), "tau_mx"),
        (lambda p: p.update(max_velcity=1.0), "max_velcity"),
        (lambda p: p.update(control_rate_hz=100.0), "control_rate_hz"),
        (lambda p: p.update(feedback_timeout=p["feedback_timeout"] * 2), "feedback_timeout"),
        (lambda p: p["joints"][0].update(motor_id=9), r"joints\[0\].motor_id"),
        (lambda p: p["joints"][0].update(model="RS02"), "model"),
    ],
)
def test_a_raised_or_non_finite_value_is_refused_before_connect(edit, field):
    p = m.profile()
    edit(p)
    with pytest.raises(ValueError, match=field if "\\" in field else field.replace("[", r"\[").replace("]", r"\]")):
        m.Arm.sim(profile=p)


def test_a_raised_cap_is_refused_before_connect():
    p = m.profile()
    p["joints"][1]["tau_max"] *= 1.5
    with pytest.raises(ValueError, match="tau_max"):
        m.Arm.sim(profile=p)


def test_socketcan_refuses_a_bad_profile_before_the_socket_opens():
    p = m.profile()
    p["joints"][1]["tau_max"] *= 1.5
    # An interface that does not exist: opening it would raise RuntimeError, so a ValueError
    # proves the profile was checked first.
    with pytest.raises(ValueError, match="tau_max"):
        m.Arm.socketcan("nocan99", profile=p)


def test_a_lowered_range_and_gains_are_accepted():
    p = m.profile()
    p["joints"][0]["q_lo"] += 0.1
    p["joints"][0]["q_hi"] -= 0.1
    p["joints"][0]["kp"] *= 0.5
    p["max_velocity"] *= 0.5
    a = m.Arm.sim(profile=p)
    a.enable()
    a.stop()


def test_a_flipped_direction_maps_motor_to_joint():
    v1 = m.profile()
    p = m.profile()
    for i, (d, o) in {0: (-1.0, 0.4), 2: (-1.0, -0.7), 4: (1.0, 0.3)}.items():
        p["joints"][i]["direction"] = d
        p["joints"][i]["offset"] = o
    a = m.Arm.sim(profile=p)
    a.enable()
    a.start_hold()
    time.sleep(0.2)
    snap = a.snapshot()
    d = a.drain()
    a.stop()
    # The sim parks every motor at the joint-range midpoint through the same map the session
    # reads back with, so joint coordinates come out as the midpoints for any direction/offset.
    for i, j in enumerate(v1["joints"][:6]):
        mid = (j["q_lo"] + j["q_hi"]) / 2.0
        assert abs(snap["positions"][i] - mid) < 1e-3, i
        assert abs(d["cmd_pos"][-1][i] - mid) < 1e-3, i


def test_clamp_command_reads_direction_and_offset_without_changing_clamping():
    p = m.profile()
    p["joints"][0]["direction"] = -1.0
    p["joints"][0]["offset"] = 0.4
    mids = [(j["q_lo"] + j["q_hi"]) / 2.0 for j in p["joints"]]
    out, clamped = m.clamp_command([(q, 0.0, 0.0, 0.0, 0.0) for q in mids], p)
    assert not clamped
