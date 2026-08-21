use maker_arm::{ArmConfig, Session, SessionError, SessionState, SimArm};
use maker_arm_protocol::param_index;
use maker_arm_transport::{CanBackend, TransportError};
use std::f64::consts::TAU;
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

/// Test-only backend that wraps a `SimArm` and can be told to fail `send`
/// or `recv` on demand -- something `SimArm` alone can never do, since it
/// always succeeds. Configuration is shared via `Arc<Mutex<_>>` so a test
/// can arm a failure *after* the backend is already boxed inside a
/// `Session`. Exists purely to exercise `Session`'s error-handling paths
/// (best-effort estop, enable-vs-drain-failure state ordering); not a
/// production type and not a new hook on `SimArm` itself.
#[derive(Default)]
struct FlakyConfig {
    /// Every `send` addressed to this motor id fails.
    fail_send_for: Option<u8>,
    /// Once true, every subsequent `recv` fails.
    fail_recv: bool,
    /// When set, `fail_recv` flips to true once this many `COMM_ENABLE`
    /// frames have gone out -- lets a test fail the *trailing drain* of
    /// `enable()` without touching the enable frames themselves.
    fail_recv_after_n_enables: Option<u8>,
    enables_seen: u8,
}

struct FlakyBackend {
    inner: SimArm,
    cfg: Arc<Mutex<FlakyConfig>>,
}

impl FlakyBackend {
    fn new(inner: SimArm) -> (Self, Arc<Mutex<FlakyConfig>>) {
        let cfg = Arc::new(Mutex::new(FlakyConfig::default()));
        (
            FlakyBackend {
                inner,
                cfg: cfg.clone(),
            },
            cfg,
        )
    }
}

impl CanBackend for FlakyBackend {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        let comm = ((id >> 24) & 0x1F) as u8;
        let target = (id & 0xFF) as u8;
        {
            let mut cfg = self.cfg.lock().unwrap();
            if cfg.fail_send_for == Some(target) {
                return Err(TransportError::Io(format!(
                    "injected send failure for motor {target}"
                )));
            }
            if let Some(n) = cfg.fail_recv_after_n_enables {
                if comm == maker_arm_protocol::COMM_ENABLE {
                    cfg.enables_seen += 1;
                    if cfg.enables_seen >= n {
                        cfg.fail_recv = true;
                    }
                }
            }
        }
        self.inner.send(id, data)
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        if self.cfg.lock().unwrap().fail_recv {
            return Err(TransportError::Io("injected recv failure".to_string()));
        }
        self.inner.recv(timeout)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut self.inner)
    }
}

#[test]
fn estop_send_failure_still_disables_every_reachable_motor() {
    // Critical fix 1: disable_all must be best-effort across every motor.
    // A bus hiccup on motor 2 must not stop estop from reaching 1 and 3..7.
    let c = ArmConfig::maker_arm_v1();
    let (backend, cfg) = FlakyBackend::new(SimArm::new(&c));
    let mut s = Session::connect(Box::new(backend), c.clone()).expect("connect");
    s.enable().expect("enable");
    cfg.lock().unwrap().fail_send_for = Some(2);
    match s.estop() {
        Err(_) => {}
        Ok(()) => panic!("expected estop to report the motor-2 send failure"),
    }
    // State still reflects "torque commanded off", since every reachable
    // motor was attempted -- the Err is what tells the caller the bus is
    // unreliable, not a state stuck at Enabled.
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    for j in &c.joints {
        if j.motor_id != 2 {
            assert!(
                !sim.enabled(j.motor_id),
                "motor {} should have received its disable frame",
                j.motor_id
            );
        }
    }
    assert!(
        sim.enabled(2),
        "motor 2 never got a disable frame -- expected, it was blocked"
    );
}

#[test]
fn enable_reports_enabled_even_if_the_trailing_drain_fails() {
    // Critical fix 2: state must reflect physical reality. All seven enable
    // frames go out successfully (motors are physically torque-on) and only
    // the trailing drain() fails -- state must be Enabled, not Connected.
    let c = ArmConfig::maker_arm_v1();
    let (backend, cfg) = FlakyBackend::new(SimArm::new(&c));
    let mut s = Session::connect(Box::new(backend), c.clone()).expect("connect");
    cfg.lock().unwrap().fail_recv_after_n_enables = Some(c.joints.len() as u8);
    match s.enable() {
        Err(SessionError::Transport(_)) => {}
        other => panic!("expected a transport error from the trailing drain, got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Enabled);
    let sim = s.backend_as_sim().unwrap();
    for j in &c.joints {
        assert!(sim.enabled(j.motor_id));
    }
}

#[test]
fn connect_refuses_a_wrap_that_cannot_map_the_configured_range() {
    // Critical: `connect` picks the 2π wrap from wherever the arm HAPPENS
    // to be parked, but the control loop then sends `to_motor(q) - wrap`
    // for every commanded q in `q_lo..q_hi`. J3's window is 3.882..7.955;
    // park its raw angle at 11.0 rad -- physically reachable -- and the
    // only wrap that lands the pose inside the grace window is -2π
    // (11.0 - 6.283 = 4.717). But then q_hi maps to 7.955 + 6.283 =
    // 14.238 rad, past the encoder's +12.57: `float_to_u16` would clamp
    // that to the rail with no error, truncating a position command by
    // 1.61 rad at kp 90 while reporting `clamped: false`. Refuse at
    // connect instead, with an error that says what to DO about it.
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_position(3, 11.0);
    match Session::connect(Box::new(sim), c) {
        Err(SessionError::RangeNotMappable {
            motor_id: 3,
            wrap,
            joint_bound,
            motor_pos,
            encoder_limit,
        }) => {
            assert!(
                (wrap - -TAU).abs() < 1e-9,
                "expected the -2π wrap, got {wrap}"
            );
            assert!(
                (joint_bound - 7.955).abs() < 1e-9,
                "expected J3's q_hi as the offending bound, got {joint_bound}"
            );
            assert!(
                motor_pos > 14.0,
                "expected the unmappable motor-frame position, got {motor_pos}"
            );
            assert!((encoder_limit - 12.57).abs() < 1e-9);
            // The message must tell the operator what to do, not just
            // that a number was out of range.
            let text = SessionError::RangeNotMappable {
                motor_id: 3,
                wrap,
                joint_bound,
                motor_pos,
                encoder_limit,
            }
            .to_string();
            assert!(text.contains("Re-zero"), "not actionable: {text}");
            assert!(text.contains("motor 3"), "no motor id: {text}");
        }
        other => panic!("expected RangeNotMappable(3), got {other:?}"),
    }
}

#[test]
fn connect_accepts_a_negative_wrap_whose_whole_range_still_maps() {
    // Guard against the fix above being over-broad: a -2π wrap is fine as
    // long as the joint's WHOLE configured range still fits in the
    // encoder window. J4's window is -0.832..2.122, so parking its raw
    // angle one full turn HIGH (mid + 2π = 6.928) forces wrap = -2π, and
    // both bounds still map comfortably (-0.832 + 6.283 = 5.451 and
    // 2.122 + 6.283 = 8.405, both inside ±12.57). Connect must succeed
    // and report the corrected joint position.
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    let true_j4 = mid(&c, 3);
    sim.set_position(4, true_j4 + TAU);
    let s = Session::connect(Box::new(sim), c).expect("connect with a mappable -2π wrap");
    assert!(
        (s.positions()[3] - true_j4).abs() < 1e-3,
        "expected {true_j4}, got {}",
        s.positions()[3]
    );
}
