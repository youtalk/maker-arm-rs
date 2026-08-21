use maker_arm::{
    ArmCommand, ArmConfig, ArmState, Controller, FaultReason, HoldController, JointCommand,
    Session, SessionState, SimArm,
};
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
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
    // `Session` only learns about a motor's fault bits when a feedback
    // frame carrying them is dispatched. `inject_fault` mutates the sim
    // directly with no frame, so detection lands one tick after it (the
    // reply to THIS tick's own MIT send, drained at the end of the tick,
    // per Task 5's/Task 7's convention) -- the same one-control-period
    // latency a real motor's feedback would have. That first tick DOES
    // still apply the moving controller's output (nothing could have
    // known better yet): on real hardware a motor ramps and moves only a
    // fraction of a 0.2 rad step in 5 ms, but `SimArm` is an ideal servo
    // that snaps instantly, so we only pin "never moved again" from the
    // tick the fault is actually reported onward, not from `before`.
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let before = s.positions();
    s.backend_as_sim().unwrap().inject_fault(5, 0x21);
    let mut away = GoTo::new(&c, before.iter().map(|q| q + 0.2).collect());
    let out1 = s.tick(&mut away).expect("tick 1: fault not visible yet");
    assert!(out1.fault.is_none());
    assert_eq!(s.state(), SessionState::Enabled);
    let out2 = s.tick(&mut away).expect("tick 2: fault now visible");
    assert_eq!(
        out2.fault,
        Some(FaultReason::MotorFault {
            motor_id: 5,
            bits: 0x21
        })
    );
    assert_eq!(s.state(), SessionState::Fault);
    let held = s.positions();
    for _ in 0..5 {
        s.tick(&mut away).expect("hold");
    }
    // the moving controller was never listened to once the fault was
    // actually detected and held
    let after = s.positions();
    for i in 0..7 {
        assert!((after[i] - held[i]).abs() < 1e-3);
    }
}

#[test]
fn hold_on_fault_false_disables_outright() {
    // Same one-tick detection latency as above; `HoldController` never
    // moves the arm, so there is no ideal-servo jump to account for here.
    let mut c = fast(ArmConfig::maker_arm_v1());
    c.hold_on_fault = false;
    let mut s = enabled_session(&c);
    s.backend_as_sim().unwrap().inject_fault(2, 0x01);
    let mut hold = HoldController::from_config(&c);
    let out1 = s.tick(&mut hold).expect("tick 1: fault not visible yet");
    assert!(out1.fault.is_none());
    let out2 = s.tick(&mut hold).expect("tick 2: fault now visible");
    assert_eq!(
        out2.fault,
        Some(FaultReason::MotorFault {
            motor_id: 2,
            bits: 0x01
        })
    );
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    for id in 1..=7u8 {
        assert!(!sim.enabled(id));
    }
}

#[test]
fn stale_feedback_faults() {
    // Every tick's own MIT reply refreshes ITS motor at the end of that
    // same tick, so a motor kept in the loop stays "fresh" every control
    // period; a muted motor's timestamp freezes at the moment it was
    // muted and its age grows without bound. Drive real-time-paced ticks
    // (via `run()`) long enough for a several-period gap to exceed a
    // feedback_timeout comfortably larger than one nominal tick period,
    // so ONLY the muted motor -- not all seven -- crosses it.
    let mut c = fast(ArmConfig::maker_arm_v1());
    c.feedback_timeout = 0.02; // 4x the nominal 5 ms tick period
    let mut s = enabled_session(&c);
    let mut hold = HoldController::from_config(&c);
    let stop = AtomicBool::new(false);
    // Warm up: every motor's own feedback is fresh after a few real ticks.
    s.run(&mut hold, &stop, Some(3)).expect("warm-up run");
    assert!(s.fault().is_none());
    s.backend_as_sim().unwrap().set_muted(4, true);
    // Drive enough further paced ticks for motor 4's un-refreshed age to
    // cross feedback_timeout while the other six, refreshed every tick by
    // their own MIT round trip, stay comfortably under it.
    s.run(&mut hold, &stop, Some(15)).expect("run to fault");
    assert!(matches!(
        s.fault(),
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

#[test]
fn clearing_a_fault_resets_the_health_monitor_and_reenables_detection() {
    // `Session::fault` is only ever set to `Some(..)`; unless something
    // clears it, `tick()`'s `if self.fault.is_none()` gates stay closed
    // forever, so the health monitor never runs again -- even across a
    // clean clear_faults() + enable() cycle. Pin the intended recovery
    // path: clear_faults() must reset `fault` (and the health monitor's
    // internal counters) so a freshly re-enabled arm has live detection.
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    s.backend_as_sim().unwrap().inject_fault(3, 0x01);
    let mut hold = HoldController::from_config(&c);
    s.tick(&mut hold).expect("tick 1: fault not visible yet");
    s.tick(&mut hold).expect("tick 2: fault now visible");
    assert!(s.fault().is_some());
    assert_eq!(s.state(), SessionState::Fault);

    s.clear_faults().expect("clear_faults");
    assert!(
        s.fault().is_none(),
        "clear_faults must reset the stale fault reason"
    );
    assert_eq!(s.state(), SessionState::Connected);
    s.enable().expect("re-enable");

    // Live again: a healthy tick after re-enable must not carry forward
    // the old fault, and the health check must actually be running.
    let mut hold2 = HoldController::from_config(&c);
    let out = s.tick(&mut hold2).expect("tick after re-enable");
    assert!(out.fault.is_none());
    assert!(s.fault().is_none());
    assert_eq!(s.state(), SessionState::Enabled);
}

#[test]
fn run_keeps_holding_through_a_fault_until_max_ticks() {
    // `run()`'s early-return condition (`state() == Connected`) is
    // deliberately narrow: it must NOT also fire for `Fault`, or a
    // hold-on-fault run would drop torque and let the arm fall instead of
    // holding. Pin it by counting ticks actually executed (not just the
    // final state, which would look the same either way): a `run()` that
    // wrongly exits on `Fault` executes far fewer than `max_ticks`.
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let before = s.positions();
    let mut evil = GoesNan {
        inner: GoTo::new(&c, before),
        ticks: 0,
    };
    let stop = AtomicBool::new(false);
    // Bounded: max_ticks caps the run even if something is badly wrong.
    s.run(&mut evil, &stop, Some(50))
        .expect("run through fault");
    assert_eq!(s.state(), SessionState::Fault);
    assert_eq!(
        s.arm_state().tick,
        50,
        "run() must keep ticking (holding) through a fault, not exit early"
    );
    // torque never dropped -- still holding, not disabled
    let sim = s.backend_as_sim().unwrap();
    for id in 1..=7u8 {
        assert!(sim.enabled(id));
    }
}

#[test]
fn command_side_respects_direction_and_offset() {
    // Every `maker_arm_v1` joint has direction = 1.0, offset = 0.0, so a
    // sign error in `j.direction * c.vel` / `j.direction * c.tau` inside
    // `send_mit_all`, or a mis-ordered offset term feeding `to_motor`,
    // would be invisible there -- the same class of latent, hardware-only
    // bug the wrap-sign test above guards against. Exercise a SYNTHETIC
    // joint with a nontrivial direction and offset instead of touching
    // the pinned profile. (`SimArm` never stores or replies with the
    // commanded velocity -- its MIT handler doesn't read it at all -- so
    // this only covers position and torque.)
    let mut c = fast(ArmConfig::maker_arm_v1());
    c.joints[0].direction = -1.0;
    c.joints[0].offset = 0.7;
    let mut s = enabled_session(&c);
    let target = mid(&c, 0);
    let mut targets = s.positions();
    targets[0] = target;
    let mut go = GoTo::new(&c, targets);
    go.tau = 1.0;
    s.tick(&mut go).expect("tick");

    // Position: read the RAW motor angle straight from the sim, bypassing
    // decode entirely -- `to_motor` must actually have been applied, not
    // skipped or applied with the operands swapped.
    let raw_pos = s.backend_as_sim().unwrap().position(1); // motor_id 1 == joint index 0
    let expected_raw_pos = c.joints[0].offset + c.joints[0].direction * target;
    assert!((raw_pos - expected_raw_pos).abs() < 1e-2);

    // Torque: decoded through Task 5's (unchanged) receive-side direction
    // multiply, so a missing or wrong direction factor in `send_mit_all`
    // flips its sign here (direction^2 == 1 only when BOTH sides apply
    // it consistently).
    let decoded_tau = s.arm_state().motors[0].torque;
    assert!((decoded_tau - 1.0).abs() < 1e-2);
}

#[test]
fn reenabling_without_clear_faults_still_resets_the_stale_fault() {
    // Critical 2's original fix lives in disable_all (reached by
    // disable/estop/clear_faults). But the hold_on_fault=false path
    // disables via enter_fault's own call to disable() -- so the same
    // fault-never-clears latch can re-form on a caller who calls
    // enable() again DIRECTLY, without an intervening clear_faults().
    // enable() must also clear a stale fault on its own success path.
    let mut c = fast(ArmConfig::maker_arm_v1());
    c.hold_on_fault = false;
    let mut s = enabled_session(&c);
    s.backend_as_sim().unwrap().inject_fault(6, 0x02);
    let mut hold = HoldController::from_config(&c);
    s.tick(&mut hold).expect("tick 1: fault not visible yet");
    s.tick(&mut hold).expect("tick 2: fault now visible");
    assert_eq!(
        s.fault().cloned(),
        Some(FaultReason::MotorFault {
            motor_id: 6,
            bits: 0x02
        })
    );
    assert_eq!(s.state(), SessionState::Connected);

    // Clear the sim-side fault bits so what's under test here is
    // enable()'s OWN reset, not a still-genuinely-faulted motor 6
    // re-tripping the health check immediately after re-enable.
    s.backend_as_sim().unwrap().inject_fault(6, 0x00);

    // Re-enable DIRECTLY: no clear_faults() call in between.
    s.enable().expect("re-enable without clear_faults");
    assert!(
        s.fault().is_none(),
        "enable() must clear a stale fault on its success path"
    );
    assert_eq!(s.state(), SessionState::Enabled);

    // Health check must be live again: inject a fresh fault on a
    // different motor and confirm it's still detected, not silently
    // swallowed by a permanently-closed `self.fault.is_none()` gate.
    let mut hold2 = HoldController::from_config(&c);
    s.backend_as_sim().unwrap().inject_fault(1, 0x04);
    let out1 = s.tick(&mut hold2).expect("tick 1 after re-enable");
    assert!(out1.fault.is_none());
    let out2 = s.tick(&mut hold2).expect("tick 2 after re-enable");
    assert_eq!(
        out2.fault,
        Some(FaultReason::MotorFault {
            motor_id: 1,
            bits: 0x04
        })
    );
}

#[test]
fn dropping_a_running_arm_stops_the_background_thread() {
    // Regression for RunningArm's Drop impl: dropping the handle instead
    // of calling stop_and_disable() used to leave the background
    // thread's OWN Arc<LoopShared> clone as the only thing keeping it
    // alive -- `stop` never got set, and the detached thread kept
    // ticking (and MIT-streaming with torque on) forever. Verify via a
    // counter that lives entirely in the test's own Arc -- no access to
    // the Session or backend is needed (and none is available: a bare
    // `drop()` never hands the Session back, unlike `stop_and_disable`).
    struct CountingController {
        ticks: Arc<AtomicU64>,
        inner: HoldController,
    }
    impl Controller for CountingController {
        fn update(&mut self, state: &ArmState, dt: f64) -> ArmCommand {
            self.ticks.fetch_add(1, Ordering::Relaxed);
            self.inner.update(state, dt)
        }
    }

    let c = fast(ArmConfig::maker_arm_v1());
    let s = enabled_session(&c);
    let ticks = Arc::new(AtomicU64::new(0));
    let controller = CountingController {
        ticks: ticks.clone(),
        inner: HoldController::from_config(&c),
    };
    let running = s.start(Box::new(controller));

    // Let it tick a handful of times before we drop instead of releasing
    // properly.
    std::thread::sleep(std::time::Duration::from_millis(50));
    drop(running); // NOT stop_and_disable() -- this is the regression path.

    // `Drop` joins the thread before returning, so by the time `drop()`
    // above returns, a CORRECT implementation has already stopped it.
    // Sample now, then again after a generous margin: a leaked thread
    // keeps incrementing at ~200 Hz and the counts would differ; a
    // stopped one leaves them equal. Generous sleep on purpose -- a
    // slow-but-correct result must never read as a false failure in CI.
    let after_drop = ticks.load(Ordering::Relaxed);
    assert!(after_drop > 0, "controller should have ticked before drop");
    std::thread::sleep(std::time::Duration::from_millis(300));
    let after_wait = ticks.load(Ordering::Relaxed);
    assert_eq!(
        after_drop, after_wait,
        "dropping RunningArm must stop the background thread, not leak it"
    );
}
