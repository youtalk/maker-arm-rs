use maker_arm::{ArmConfig, HoldController, Session, SessionError, SessionState, SimArm};
use maker_arm_protocol::{mit, param_index, ParamValue};
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
    // the probe must not have enabled anything
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
fn connect_refuses_a_reported_fault() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.inject_fault(2, 1 << 2);
    let (backend, cfg) = FlakyBackend::new(sim);
    match Session::connect(Box::new(backend), c) {
        Err(
            e @ SessionError::MotorFault {
                motor_id: 2,
                word: 4,
            },
        ) => {
            assert!(e.to_string().contains("undervoltage"), "not decoded: {e}");
        }
        other => panic!("expected MotorFault(2, 4), got {other:?}"),
    }
    // The clearing state read never went to the faulted motor.
    let clear = mit::encode_clear_fault(2);
    assert!(!cfg.lock().unwrap().sent.contains(&(clear.id, clear.data)));
}

#[test]
fn connect_reads_run_mode_where_parameters_work() {
    let c = ArmConfig::maker_arm_v1();
    let s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).expect("connect");
    assert!(s.arm_state().motors.iter().all(|m| m.params));

    let mut sim = SimArm::new(&c);
    let w = mit::encode_write_param(4, param_index::RUN_MODE, ParamValue::U8(1));
    sim.send(w.id, &w.data).unwrap();
    match Session::connect(Box::new(sim), c) {
        Err(SessionError::RunMode {
            motor_id: 4,
            value: 1,
        }) => {}
        other => panic!("expected RunMode(4, 1), got {other:?}"),
    }
}

fn is_param_write(id: u32) -> bool {
    id & 0x780 == mit::PARAM_WRITE_ID
}

#[test]
fn legacy_firmware_connects_and_enables_without_parameters() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_params_supported(false);
    let (backend, cfg) = FlakyBackend::new(sim);
    let mut s = Session::connect(Box::new(backend), c.clone()).expect("connect");
    s.enable().expect("enable");
    assert_eq!(s.state(), SessionState::Enabled);
    assert!(s.arm_state().motors.iter().all(|m| !m.params));
    assert!(!cfg.lock().unwrap().sent.iter().any(|f| is_param_write(f.0)));
    let sim = s.backend_as_sim().unwrap();
    for j in &c.joints {
        assert!(sim.enabled(j.motor_id));
    }
}

#[test]
fn enable_never_writes_run_mode() {
    let c = ArmConfig::maker_arm_v1();
    let (backend, cfg) = FlakyBackend::new(SimArm::new(&c));
    let mut s = Session::connect(Box::new(backend), c).expect("connect");
    s.enable().expect("enable");
    s.disable().expect("disable");
    assert!(!cfg.lock().unwrap().sent.iter().any(|(id, d)| {
        is_param_write(*id) && u16::from_le_bytes([d[0], d[1]]) == param_index::RUN_MODE
    }));
}

#[test]
fn the_watchdog_is_armed_on_the_gripper_only_and_restored() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    // A non-zero original value, so the restore is visible.
    let w = mit::encode_write_param(7, param_index::CAN_TIMEOUT, ParamValue::U32(4000));
    sim.send(w.id, &w.data).unwrap();
    let mut s = Session::connect(Box::new(sim), c).expect("connect");
    s.enable().expect("enable");
    let sim = s.backend_as_sim().unwrap();
    // 500 ms * 20 counts/ms
    assert_eq!(sim.param_u32(7, param_index::CAN_TIMEOUT), Some(10000));
    for m in 1..=6 {
        assert_eq!(sim.param_u32(m, param_index::CAN_TIMEOUT), Some(0));
    }
    s.disable().expect("disable");
    let sim = s.backend_as_sim().unwrap();
    assert_eq!(sim.param_u32(7, param_index::CAN_TIMEOUT), Some(4000));
}

#[test]
fn enable_refuses_a_hot_gripper() {
    let c = ArmConfig::maker_arm_v1();
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c).expect("connect");
    s.backend_as_sim().unwrap().set_temperature(7, 45.0);
    match s.enable() {
        Err(SessionError::TooHot {
            motor_id: 7,
            celsius: 45.0,
        }) => {}
        other => panic!("expected TooHot(7, 45.0), got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| sim.enables(m) == 0));
    // Joints are not held to the gripper's limit, and 44.9 degrees C is fine.
    sim.set_temperature(1, 60.0);
    sim.set_temperature(7, 44.9);
    s.enable().expect("enable below 45 degrees C");
}

#[test]
fn enable_refuses_a_joint_at_its_health_limit() {
    // Ruling R31: before any torque, not after a first tick at kp.
    let c = ArmConfig::maker_arm_v1();
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).expect("connect");
    s.backend_as_sim()
        .unwrap()
        .set_temperature(3, c.temp_hold_c);
    let e = s.enable().unwrap_err();
    assert!(
        matches!(e, SessionError::TooHot { motor_id: 3, celsius } if celsius == c.temp_hold_c),
        "{e:?}"
    );
    assert_eq!(e.to_string(), "motor 3 is 70.0 degC, too hot to enable");
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| sim.enables(m) == 0));
    sim.set_temperature(3, c.temp_hold_c - 0.1);
    s.enable().expect("enable below the limit");
}

#[test]
fn enable_preloads_a_zero_gain_frame_at_the_measured_pose() {
    let c = ArmConfig::maker_arm_v1();
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).expect("connect");
    s.enable().expect("enable");
    let sim = s.backend_as_sim().unwrap();
    for j in &c.joints {
        let p = j.model.params();
        let (pos, vel, kp, kd, tau) = sim.preload(j.motor_id).expect("a preload");
        assert!((pos - sim.position(j.motor_id)).abs() <= (p.p_max - p.p_min) / 65535.0);
        assert!(vel.abs() <= (p.v_max - p.v_min) / 4095.0);
        assert_eq!((kp, kd), (0.0, 0.0));
        assert!(tau.abs() <= (p.t_max - p.t_min) / 4095.0);
    }
}

#[test]
fn a_failed_enable_disables_every_motor() {
    let c = ArmConfig::maker_arm_v1();
    let (backend, cfg) = FlakyBackend::new(SimArm::new(&c));
    let mut s = Session::connect(Box::new(backend), c).expect("connect");
    cfg.lock().unwrap().after_control_to = Some((5, |sim| sim.set_muted(5, true)));
    match s.enable() {
        Err(SessionError::NotDisabled { motors, .. }) if motors == [5] => {}
        other => panic!("expected motor 5 not disabled, got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    // Motors 1 to 5 had torque on before motor 5 went quiet.
    assert!((1..=5).all(|m| sim.enables(m) == 1));
    assert!((1..=7).all(|m| !sim.enabled(m)));
}

#[test]
fn enable_refuses_a_fault_flag_in_the_enable_reply() {
    // The fault comes up between motor 3's preload and its enable, so only
    // the enable reply carries it.
    let c = ArmConfig::maker_arm_v1();
    let (backend, cfg) = FlakyBackend::new(SimArm::new(&c));
    let mut s = Session::connect(Box::new(backend), c).expect("connect");
    cfg.lock().unwrap().after_control_to = Some((3, |sim| sim.inject_fault(3, 1 << 14)));
    match s.enable() {
        Err(SessionError::MotorFault { motor_id: 3, .. }) => {}
        other => panic!("expected MotorFault(3), got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=3).all(|m| sim.enables(m) == 1));
    assert!((1..=7).all(|m| !sim.enabled(m)));
}

#[test]
fn a_failed_release_after_a_failed_enable_names_the_motor() {
    // Motor 3's enable fails, and then motor 6 never shows mode 0 to the
    // checked disable. The caller must learn that motor 6 may still hold
    // torque, so the release error wins over the enable error.
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.force_mode(6, Some(2));
    let (backend, cfg) = FlakyBackend::new(sim);
    let mut s = Session::connect(Box::new(backend), c).expect("connect");
    cfg.lock().unwrap().after_control_to = Some((3, |sim| sim.inject_fault(3, 1 << 14)));
    match s.enable() {
        Err(SessionError::NotDisabled { motors, .. }) if motors == [6] => {}
        other => panic!("expected motor 6 not disabled, got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Connected);
}

#[test]
fn enable_refuses_a_silent_motor_before_any_enable() {
    let c = ArmConfig::maker_arm_v1();
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c).expect("connect");
    s.backend_as_sim().unwrap().set_muted(2, true);
    match s.enable() {
        Err(SessionError::Probe { motor_id: 2 }) => {}
        other => panic!("expected Probe(2), got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| sim.enables(m) == 0));
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
    // A reported fault blocks enable until clear_faults clears it.
    sim.inject_fault(5, 1 << 14);
    match s.enable() {
        Err(SessionError::MotorFault { motor_id: 5, .. }) => {}
        other => panic!("expected MotorFault(5), got {other:?}"),
    }
    s.clear_faults().expect("clear");
    assert_eq!(s.state(), SessionState::Connected);
    s.enable().expect("enable after clear_faults");
}

#[test]
fn query_faults_asks_every_motor_and_records_power_up_frames() {
    let c = ArmConfig::maker_arm_v1();
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c).expect("connect");
    s.enable().expect("enable");
    assert!(s.query_faults().unwrap());
    s.backend_as_sim().unwrap().set_muted(3, true);
    assert!(!s.query_faults().unwrap());
    let sim = s.backend_as_sim().unwrap();
    sim.set_muted(3, false);
    sim.power_cycle_motors(&[2, 3]);
    assert!(s.query_faults().unwrap());
    let restarted: Vec<bool> = s.arm_state().motors.iter().map(|m| m.restarted).collect();
    assert_eq!(restarted, [false, true, true, false, false, false, false]);
    let sim = s.backend_as_sim().unwrap();
    assert!(
        !sim.enabled(2) && sim.enabled(1),
        "a fault query enables or disables nothing"
    );
}

#[test]
fn power_ups_count_distinct_announcements_until_the_next_enable() {
    let c = ArmConfig::maker_arm_v1();
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c).expect("connect");
    s.enable().expect("enable");
    let sim = s.backend_as_sim().unwrap();
    sim.power_cycle_motors(&[2, 3]);
    // A second power-up of the same motor counts once.
    sim.power_cycle_motors(&[3]);
    assert!(s.query_faults().unwrap());
    assert_eq!(s.power_ups(), 2);
    // Kept in any state: after a disable, the session is Connected.
    s.disable().expect("disable");
    s.backend_as_sim().unwrap().power_cycle_motors(&[4]);
    s.query_faults().unwrap();
    assert_eq!(s.power_ups(), 3);
    s.enable().expect("enable");
    assert_eq!(s.power_ups(), 0);
}

#[test]
fn poll_fault_asks_each_motor_once_per_interval_by_send_time() {
    let c = ArmConfig::maker_arm_v1();
    let (backend, cfg) = FlakyBackend::new(SimArm::new(&c));
    let mut s = Session::connect(Box::new(backend), c.clone()).expect("connect");
    s.enable().expect("enable");
    let queries = |cfg: &Mutex<FlakyConfig>| -> Vec<u32> {
        let q: Vec<u32> = cfg
            .lock()
            .unwrap()
            .sent
            .iter()
            .filter(|(_, d)| d[6..] == [0x00, 0xFB])
            .map(|(id, _)| *id)
            .collect();
        q
    };
    // Paced by the last query sent, not the last reply: the motor asked
    // longest ago goes first, one query per call, each motor once per
    // interval. Enable's own queries do not count.
    for _ in 0..2 {
        cfg.lock().unwrap().sent.clear();
        for _ in 0..10 {
            s.poll_fault().unwrap();
        }
        assert_eq!(queries(&cfg), (1..=7).collect::<Vec<u32>>());
        std::thread::sleep(Duration::from_secs_f64(c.fault_poll_interval));
    }
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
type SimHook = fn(&mut SimArm);

#[derive(Default)]
struct FlakyConfig {
    /// Every `send` addressed to this motor id fails.
    fail_send_for: Option<u8>,
    /// Once true, every subsequent `recv` fails.
    fail_recv: bool,
    /// When set, `fail_recv` flips to true once this many enable frames
    /// have gone out.
    fail_recv_after_n_enables: Option<u8>,
    enables_seen: u8,
    /// Runs once on the sim, right after the first control frame to this
    /// motor reaches it.
    after_control_to: Option<(u8, SimHook)>,
    /// Every frame that reached the sim, in order.
    sent: Vec<(u32, [u8; 8])>,
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
        let target = (id & 0x7F) as u8;
        let mut cfg = self.cfg.lock().unwrap();
        if cfg.fail_send_for == Some(target) {
            return Err(TransportError::Io(format!(
                "injected send failure for motor {target}"
            )));
        }
        if let Some(n) = cfg.fail_recv_after_n_enables {
            if *data == mit::encode_enable(target).data {
                cfg.enables_seen += 1;
                if cfg.enables_seen >= n {
                    cfg.fail_recv = true;
                }
            }
        }
        cfg.sent.push((id, *data));
        self.inner.send(id, data)?;
        let command = data[..6] == [0xFF; 6];
        if id < 0x80 && !command {
            if let Some((_, hook)) = cfg.after_control_to.take_if(|(m, _)| *m == target) {
                hook(&mut self.inner);
            }
        }
        Ok(())
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
fn a_transport_error_during_enable_still_disables_every_motor() {
    // The bus fails while enable waits for the last motor's reply: every
    // enable frame went out, so every motor must get its disable frame,
    // and the state must say Connected.
    let c = ArmConfig::maker_arm_v1();
    let (backend, cfg) = FlakyBackend::new(SimArm::new(&c));
    let mut s = Session::connect(Box::new(backend), c.clone()).expect("connect");
    cfg.lock().unwrap().fail_recv_after_n_enables = Some(c.joints.len() as u8);
    // No reply confirms a disable, and the error keeps the cause.
    match s.enable() {
        Err(SessionError::NotDisabled {
            motors,
            cause: Some(c),
        }) if motors == [1, 2, 3, 4, 5, 6, 7] && matches!(*c, SessionError::Transport(_)) => {}
        other => panic!("expected a transport error, got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    for j in &c.joints {
        assert_eq!(sim.enables(j.motor_id), 1);
        assert!(!sim.enabled(j.motor_id));
    }
}

#[test]
fn connect_refuses_a_wrap_that_cannot_map_the_configured_range() {
    // Critical: `connect` picks the 2π wrap from wherever the arm HAPPENS
    // to be parked, but the control loop then sends `to_motor(q) - wrap`
    // for every commanded q in `q_lo..q_hi`. J3 gets a motor-frame offset
    // of 3.0 rad here, so its window is 3.049..7.121 rad in the motor
    // frame. Park the raw angle at 11.283 rad -- physically reachable --
    // and the only wrap that lands the pose inside the grace window is -2π
    // (the joint reads 2.0 rad). But then q_hi maps to 7.121 + 6.283 =
    // 13.404 rad, past the encoder's +12.57: `float_to_uint` would clamp
    // that to the rail with no error, truncating a position command
    // while reporting `clamped: false`. Refuse at connect instead, with
    // an error that says what to DO about it.
    let mut c = ArmConfig::maker_arm_v1();
    c.joints[2].offset = 3.0;
    let q_hi = c.joints[2].q_hi;
    let mut sim = SimArm::new(&c);
    sim.set_position(3, 3.0 + 2.0 + TAU);
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
                (joint_bound - q_hi).abs() < 1e-9,
                "expected J3's q_hi as the offending bound, got {joint_bound}"
            );
            assert!(
                motor_pos > 13.0,
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

#[test]
fn the_enable_chooses_the_wrap_again_after_a_motor_restart() {
    // The real arm, 2026-11-18: after the emergency stop, motor 1 reported
    // its position one turn lower. With the wrap chosen at connect it read
    // -6.293 rad, and the enable refused it.
    let c = ArmConfig::maker_arm_v1();
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).expect("connect");
    let true_j1 = s.positions()[0];
    let sim = s.backend_as_sim().unwrap();
    let raw = sim.position(1) - TAU;
    sim.set_position(1, raw);
    s.enable().expect("enable a turn away");
    assert!((s.positions()[0] - true_j1).abs() < 1e-3);
    // The output stage sends the hold with the new wrap: the motor stays.
    let mut hold = HoldController::from_config(&c);
    s.tick(&mut hold).expect("tick");
    assert!((s.backend_as_sim().unwrap().position(1) - raw).abs() < 1e-3);
}

#[test]
fn a_released_read_chooses_the_wrap_again_or_keeps_it() {
    let c = ArmConfig::maker_arm_v1();
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).expect("connect");
    let true_j1 = s.positions()[0];
    let sim = s.backend_as_sim().unwrap();
    sim.set_position(1, sim.position(1) - TAU);
    // J4's window is -1.488..2.165 with the grace: 3.5 rad fits no wrap.
    sim.set_position(4, 3.5);
    s.read_motors().expect("read");
    assert!((s.positions()[0] - true_j1).abs() < 1e-3);
    assert!((s.positions()[3] - 3.5).abs() < 1e-3, "the old wrap stays");
    match s.enable() {
        Err(SessionError::PositionOutOfRange { motor_id: 4, .. }) => {}
        other => panic!("expected PositionOutOfRange(4), got {other:?}"),
    }
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| sim.enables(m) == 0));
}

#[test]
fn the_enable_refuses_a_wrap_that_cannot_map_the_configured_range() {
    // As `connect_refuses_a_wrap_that_cannot_map_the_configured_range`, but
    // the motor comes to that position after the connect.
    let mut c = ArmConfig::maker_arm_v1();
    c.joints[2].offset = 3.0;
    let mut s = Session::connect(Box::new(SimArm::new(&c)), c).expect("connect");
    s.backend_as_sim().unwrap().set_position(3, 3.0 + 2.0 + TAU);
    match s.enable() {
        Err(SessionError::RangeNotMappable { motor_id: 3, .. }) => {}
        other => panic!("expected RangeNotMappable(3), got {other:?}"),
    }
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| sim.enables(m) == 0));
}

#[test]
fn p0_rest_pose_connects_in_urdf_frame() {
    // The vendor calibration: direction -1 on J5. Raw motor angles from the P0 rest pose.
    let mut c = ArmConfig::maker_arm_v1();
    c.joints[4].direction = -1.0;
    let raw = [352.0, 0.0, 7.9, 335.5, -2.6, 356.8, -0.9];
    let want = [-8.0, 0.0, 7.9, -24.5, 2.6, -3.2, -0.9];
    let mut sim = SimArm::new(&c);
    for (i, deg) in raw.iter().enumerate() {
        sim.set_position(i as u8 + 1, f64::to_radians(*deg));
    }
    let s = Session::connect(Box::new(sim), c).expect("connect");
    for (i, (q, deg)) in s.positions().iter().zip(want).enumerate() {
        // The wire carries 16-bit positions, so a reading is exact to half a step (1.92e-4 rad).
        assert!((q - f64::to_radians(deg)).abs() < 2e-4, "joint {i}: {q}");
    }
}
