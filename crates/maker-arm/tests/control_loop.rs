use maker_arm::{
    ArmCommand, ArmConfig, ArmState, Controller, FaultReason, HoldController, JointCommand,
    Session, SessionState, SimArm,
};
use std::f64::consts::TAU;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn fast(mut c: ArmConfig) -> ArmConfig {
    c.inter_frame_us = 0; // no wire pacing against an in-memory sim
    c
}

fn mid(c: &ArmConfig, i: usize) -> f64 {
    (c.joints[i].q_lo + c.joints[i].q_hi) / 2.0
}

fn enabled_session(c: &ArmConfig) -> Session {
    let sim = SimArm::new(c);
    let mut s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    s.enable().expect("enable");
    s
}

/// Test controller: constant joint targets with profile gains.
struct GoTo {
    targets: Vec<f64>,
    gains: Vec<(f64, f64)>,
    tau: f64,
}

impl GoTo {
    fn new(c: &ArmConfig, targets: Vec<f64>) -> GoTo {
        GoTo {
            targets,
            gains: c.joints.iter().map(|j| (j.kp, j.kd)).collect(),
            tau: 0.0,
        }
    }
}

impl Controller for GoTo {
    fn update(&mut self, _state: &ArmState, _dt: f64) -> ArmCommand {
        self.targets
            .iter()
            .zip(&self.gains)
            .map(|(&pos, &(kp, kd))| JointCommand {
                pos,
                vel: 0.0,
                kp,
                kd,
                tau: self.tau,
            })
            .collect()
    }
}

/// Test controller: emits NaN on every tick after the first.
struct GoesNan {
    inner: GoTo,
    ticks: u32,
}

impl Controller for GoesNan {
    fn update(&mut self, state: &ArmState, dt: f64) -> ArmCommand {
        self.ticks += 1;
        let mut cmd = self.inner.update(state, dt);
        if self.ticks > 1 {
            cmd[2].pos = f64::NAN;
        }
        cmd
    }
}

#[test]
fn hold_keeps_the_pose() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let before = s.positions();
    let mut hold = HoldController::from_config(&c);
    for _ in 0..10 {
        let out = s.tick(&mut hold).expect("tick");
        assert!(out.fault.is_none());
        assert!(!out.clamped);
    }
    let after = s.positions();
    for i in 0..7 {
        assert!((after[i] - before[i]).abs() < 1e-3);
    }
    assert_eq!(s.state(), SessionState::Enabled);
}

#[test]
fn commands_travel_through_wrap_and_conversion() {
    // Park motor 3 one turn low so connect assigns wrap = +2π; commanded
    // joint positions must come out on the RAW motor frame, minus the wrap.
    let c = fast(ArmConfig::maker_arm_v1());
    let mut sim = SimArm::new(&c);
    let true_j3 = mid(&c, 2);
    sim.set_position(3, true_j3 - TAU);
    let mut s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    s.enable().expect("enable");
    let mut targets = s.positions();
    targets[2] = true_j3 + 0.1;
    let mut go = GoTo::new(&c, targets);
    for _ in 0..3 {
        s.tick(&mut go).expect("tick");
    }
    // decoded joint position round-trips
    assert!((s.positions()[2] - (true_j3 + 0.1)).abs() < 1e-2);
    // and the raw motor angle stayed on the wrapped branch
    let raw = s.backend_as_sim().unwrap().position(3);
    assert!((raw - (true_j3 + 0.1 - TAU)).abs() < 1e-2);
}

#[test]
fn clamp_guards_the_output_stage() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let mut go = GoTo::new(&c, s.positions());
    go.tau = 99.0; // way over every tau_max
    let out = s.tick(&mut go).expect("tick");
    assert!(out.clamped);
    // the sim read back the CLAMPED torque, not 99
    let st = s.arm_state();
    assert!((st.motors[0].torque - 4.0).abs() < 1e-2); // J1 tau_max = 4.0
    assert!((st.motors[1].torque - 6.0).abs() < 1e-2); // J2 (RS02) tau_max = 6.0
}

#[test]
fn nan_from_a_controller_becomes_a_holding_fault() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let before = s.positions();
    let mut evil = GoesNan {
        inner: GoTo::new(&c, before.clone()),
        ticks: 0,
    };
    s.tick(&mut evil).expect("tick 1 fine");
    let out = s.tick(&mut evil).expect("tick 2 faults");
    assert!(matches!(out.fault, Some(FaultReason::BadCommand { .. })));
    assert_eq!(s.state(), SessionState::Fault);
    // still enabled, still holding the pre-fault pose, NaN never sent
    for _ in 0..5 {
        s.tick(&mut evil).expect("hold ticks");
    }
    let sim = s.backend_as_sim().unwrap();
    assert!(sim.enabled(3));
    let after = s.positions();
    for i in 0..7 {
        assert!((after[i] - before[i]).abs() < 1e-3);
    }
}

#[test]
fn motor_fault_bits_trigger_hold_and_ignore_the_controller() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let before = s.positions();
    s.backend_as_sim().unwrap().inject_fault(5, 0x21);
    let mut away = GoTo::new(&c, before.iter().map(|q| q + 0.2).collect());
    let out = s.tick(&mut away).expect("tick");
    assert_eq!(
        out.fault,
        Some(FaultReason::MotorFault {
            motor_id: 5,
            bits: 0x21
        })
    );
    assert_eq!(s.state(), SessionState::Fault);
    for _ in 0..5 {
        s.tick(&mut away).expect("hold");
    }
    // the moving controller was never listened to after the fault
    assert!((s.positions()[0] - before[0]).abs() < 1e-2);
}

#[test]
fn hold_on_fault_false_disables_outright() {
    let mut c = fast(ArmConfig::maker_arm_v1());
    c.hold_on_fault = false;
    let mut s = enabled_session(&c);
    s.backend_as_sim().unwrap().inject_fault(2, 0x01);
    let mut hold = HoldController::from_config(&c);
    let out = s.tick(&mut hold).expect("tick");
    assert!(out.fault.is_some());
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    for id in 1..=7u8 {
        assert!(!sim.enabled(id));
    }
}

#[test]
fn stale_feedback_faults() {
    let mut c = fast(ArmConfig::maker_arm_v1());
    c.feedback_timeout = 0.005;
    let mut s = enabled_session(&c);
    let mut hold = HoldController::from_config(&c);
    s.tick(&mut hold).expect("tick");
    s.backend_as_sim().unwrap().set_muted(4, true);
    std::thread::sleep(std::time::Duration::from_millis(15));
    let out = s.tick(&mut hold).expect("tick");
    assert!(matches!(
        out.fault,
        Some(FaultReason::FeedbackTimeout { motor_id: 4, .. })
    ));
}

#[test]
fn run_paces_ticks_and_running_arm_lifecycle_works() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let stop = AtomicBool::new(false);
    let mut hold = HoldController::from_config(&c);
    let t0 = Instant::now();
    s.run(&mut hold, &stop, Some(20)).expect("run");
    let dt = t0.elapsed().as_secs_f64();
    // 20 ticks at 200 Hz = 100 ms nominal; generous CI bounds
    assert!(dt > 0.08 && dt < 0.5, "elapsed {dt}");

    // threaded wrapper
    let running = s.start(Box::new(HoldController::from_config(&c)));
    std::thread::sleep(std::time::Duration::from_millis(50));
    let snap = running.snapshot().expect("snapshot");
    assert_eq!(snap.state, SessionState::Enabled);
    assert_eq!(snap.arm.motors.len(), 7);
    running.hold_now();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let (mut s, res) = running.stop_and_disable();
    res.expect("loop exit clean");
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    for id in 1..=7u8 {
        assert!(!sim.enabled(id));
    }
}
