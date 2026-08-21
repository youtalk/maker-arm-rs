use maker_arm::{ArmConfig, Session, SessionError, SessionState, SimArm};
use maker_arm_protocol::param_index;
use std::f64::consts::TAU;

fn mid(config: &ArmConfig, i: usize) -> f64 {
    (config.joints[i].q_lo + config.joints[i].q_hi) / 2.0
}

#[test]
fn connect_finds_all_seven_motors_torque_free() {
    let c = ArmConfig::maker_arm_v1();
    let sim = SimArm::new(&c);
    let s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    assert_eq!(s.state(), SessionState::Connected);
    let q = s.positions();
    assert_eq!(q.len(), 7);
    #[allow(clippy::needless_range_loop)] // indexes both `q` and `mid(&c, i)` by `i`
    for i in 0..7 {
        assert!(
            (q[i] - mid(&c, i)).abs() < 1e-3,
            "joint {i}: {} vs {}",
            q[i],
            mid(&c, i)
        );
    }
    // probe-by-disable must not have enabled anything
    let st = s.arm_state();
    for m in &st.motors {
        assert_eq!(m.mode, 0);
        assert!(m.feedback_age.is_finite());
    }
}

#[test]
fn connect_fails_on_a_silent_motor() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_muted(4, true);
    match Session::connect(Box::new(sim), c) {
        Err(SessionError::Probe { motor_id: 4 }) => {}
        other => panic!("expected Probe(4), got {other:?}"),
    }
}

#[test]
fn connect_applies_two_pi_wrap_correction() {
    // J3 window is 3.882..7.955 (+0.35 grace); park the raw motor angle one
    // full turn low — connect must correct it instead of failing.
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    let true_j3 = mid(&c, 2);
    sim.set_position(3, true_j3 - TAU);
    let s = Session::connect(Box::new(sim), c).expect("connect with wrap");
    assert!((s.positions()[2] - true_j3).abs() < 1e-3);
}

#[test]
fn connect_rejects_unwrappable_position() {
    // J4 window is -0.832..2.122 ± 0.35; 3.5 rad is out, and so are
    // 3.5 ± 2π — no wrap can fix it.
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_position(4, 3.5);
    match Session::connect(Box::new(sim), c) {
        Err(SessionError::PositionOutOfRange { motor_id: 4, .. }) => {}
        other => panic!("expected PositionOutOfRange(4), got {other:?}"),
    }
}

#[test]
fn enable_writes_verifies_and_enables() {
    let c = ArmConfig::maker_arm_v1();
    let sim = SimArm::new(&c);
    let mut s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    s.enable().expect("enable");
    assert_eq!(s.state(), SessionState::Enabled);
    let sim = s.backend_as_sim().expect("sim backend");
    for j in &c.joints {
        assert!(sim.enabled(j.motor_id));
        assert_eq!(sim.param_u8(j.motor_id, param_index::RUN_MODE), Some(0));
        // 200 ms * 20 counts/ms = 4000 (CAN_TIMEOUT_PER_MS)
        assert_eq!(
            sim.param_u32(j.motor_id, param_index::CAN_TIMEOUT),
            Some(4000)
        );
    }
}

#[test]
fn enable_fails_when_verify_gets_no_reply() {
    let c = ArmConfig::maker_arm_v1();
    let sim = SimArm::new(&c);
    let mut s = Session::connect(Box::new(sim), c).expect("connect");
    s.backend_as_sim().unwrap().set_muted(2, true);
    match s.enable() {
        Err(SessionError::EnableVerify { motor_id: 2, .. }) => {}
        other => panic!("expected EnableVerify(2), got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Connected);
}

#[test]
fn disable_estop_and_clear_faults_transitions() {
    let c = ArmConfig::maker_arm_v1();
    let sim = SimArm::new(&c);
    let mut s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    s.enable().expect("enable");
    s.estop().expect("estop");
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    for j in &c.joints {
        assert!(!sim.enabled(j.motor_id));
    }
    // clear_faults sends disable(clear_fault=true)
    s.backend_as_sim().unwrap().inject_fault(5, 0x21);
    s.clear_faults().expect("clear");
    assert_eq!(s.state(), SessionState::Connected);
    let st = s.arm_state();
    assert_eq!(st.motors[4].fault_bits, 0);
}

#[test]
fn enable_requires_connected_state() {
    let c = ArmConfig::maker_arm_v1();
    let sim = SimArm::new(&c);
    let mut s = Session::connect(Box::new(sim), c).expect("connect");
    s.enable().expect("enable");
    match s.enable() {
        Err(SessionError::WrongState { .. }) => {}
        other => panic!("expected WrongState, got {other:?}"),
    }
}
