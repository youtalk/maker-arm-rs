use maker_arm::{
    ArmCommand, ArmConfig, ArmState, Controller, FaultReason, HoldController, JointCommand, LoopRt,
    RtError, RunningArm, Session, SessionError, SessionState, SimArm, RELEASE_RAMP_S,
};
use maker_arm_protocol::{mit, RS00};
use maker_arm_transport::{CanBackend, TransportError};
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn fast(mut c: ArmConfig) -> ArmConfig {
    // The sim replies at once: only a muted motor waits, so keep its wait short.
    c.reply_wait_us = 500;
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

/// Test controller: holds `inner`'s targets and sends a gravity torque that changes every tick.
struct Heavy {
    inner: GoTo,
    ticks: u32,
}

impl Controller for Heavy {
    fn update(&mut self, state: &ArmState, dt: f64) -> ArmCommand {
        self.ticks += 1;
        let g = self.gravity();
        let mut cmd = self.inner.update(state, dt);
        for (c, g) in cmd.iter_mut().zip(g) {
            c.tau = g;
        }
        cmd
    }

    fn gravity(&self) -> Vec<f64> {
        (0..6)
            .map(|j| 0.1 * f64::from(self.ticks) + 0.01 * j as f64)
            .collect()
    }
}

#[test]
fn a_fault_hold_keeps_the_gravity_of_the_tick_before() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let mut heavy = Heavy {
        inner: GoTo::new(&c, s.positions()),
        ticks: 0,
    };
    for _ in 0..3 {
        s.tick(&mut heavy).expect("tick");
    }
    s.backend_as_sim().unwrap().set_temperature(3, 80.0);
    // This tick's reply carries the heat; the next tick sees it.
    assert!(s.tick(&mut heavy).expect("tick").fault.is_none());
    let before = heavy.gravity();
    for _ in 0..3 {
        let out = s.tick(&mut heavy).expect("hold");
        assert!(matches!(
            out.fault,
            Some(FaultReason::OverTemperature { motor_id: 3, .. })
        ));
        let sent = out.record.expect("the hold sends").sent;
        for j in 0..6 {
            assert_eq!(sent[j].tau, before[j], "joint {j}");
        }
        assert_eq!(sent[6].tau, 0.0, "the gripper has no gravity");
    }
    assert_eq!(heavy.ticks, 4, "the hold never asks the controller");
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
    assert!((st.motors[1].torque - 15.0).abs() < 1e-2); // J2 (RS02) tau_max = 15.0
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
fn motor_fault_flag_triggers_hold_and_ignores_the_controller() {
    // `Session` only learns about a motor's fault flag when a feedback
    // frame carrying it is dispatched. `inject_fault` mutates the sim
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
    assert!(matches!(
        out2.fault,
        Some(FaultReason::MotorFault { motor_id: 5, .. })
    ));
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
    assert!(matches!(
        out2.fault,
        Some(FaultReason::MotorFault { motor_id: 2, .. })
    ));
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
    // Twice the ~70 ms that enable's 10 ms spacing puts between motor 1's
    // last reply and the first tick, and 30x the nominal 5 ms tick period.
    c.feedback_timeout = 0.15;
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
    s.run(&mut hold, &stop, Some(60)).expect("run to fault");
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
    let running = s
        .start(Box::new(HoldController::from_config(&c)), None)
        .expect("start");
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
    // clears it, `tick()`'s `if self.fault.is_none()` gate stays closed
    // forever, so no new fault ever enters `Fault` again -- even across a
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
    assert!(matches!(
        s.fault().cloned(),
        Some(FaultReason::MotorFault { motor_id: 6, .. })
    ));
    assert_eq!(s.state(), SessionState::Connected);

    // Clear the sim-side fault so what's under test here is
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
    assert!(matches!(
        out2.fault,
        Some(FaultReason::MotorFault { motor_id: 1, .. })
    ));
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
    let running = s.start(Box::new(controller), None).expect("start");

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

#[test]
fn commands_travel_through_a_negative_wrap_too() {
    // Mirror of `commands_travel_through_wrap_and_conversion` for the
    // OTHER sign: park motor 4 one turn HIGH so connect assigns
    // wrap = -2π (the branch that used to have zero coverage, and the one
    // that can push `to_motor(q) - wrap` past the encoder's +12.57).
    // J4's whole range still maps here (see the session-level test), so
    // commands must round-trip exactly as they do for +2π.
    let c = fast(ArmConfig::maker_arm_v1());
    let mut sim = SimArm::new(&c);
    let true_j4 = mid(&c, 3);
    sim.set_position(4, true_j4 + TAU);
    let mut s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    s.enable().expect("enable");
    let mut targets = s.positions();
    targets[3] = true_j4 + 0.1;
    let mut go = GoTo::new(&c, targets);
    for _ in 0..3 {
        s.tick(&mut go).expect("tick");
    }
    // decoded joint position round-trips
    assert!((s.positions()[3] - (true_j4 + 0.1)).abs() < 1e-2);
    // and the raw motor angle stayed on the wrapped (one-turn-high) branch
    let raw = s.backend_as_sim().unwrap().position(4);
    assert!(
        (raw - (true_j4 + 0.1 + TAU)).abs() < 1e-2,
        "raw motor angle {raw} left the -2π branch"
    );
}

#[test]
fn a_saturating_motor_position_faults_instead_of_being_truncated() {
    // Backstop for the same critical finding: if a computed motor-frame
    // position ever falls outside the encoder's ±12.57 window,
    // `mit::encode_control`'s `float_to_uint` would clamp it to the rail
    // and report nothing -- a silently truncated position command at full
    // kp. The output stage must fault instead.
    //
    // The connect-time check makes this unreachable through the normal
    // path (it verifies exactly the range the clamp permits), so the test
    // has to widen the clamp AFTER that check: `limit_margin` is
    // subtracted from q_hi and added to q_lo, so a NEGATIVE margin widens
    // the permitted command range beyond the bounds connect verified --
    // precisely the class of mistake this backstop exists to catch.
    // J4 parked one turn high gives wrap = -2π, so a commanded 6.3 rad
    // maps to 6.3 + 6.283 = 12.583 rad, just past +12.57.
    let mut c = fast(ArmConfig::maker_arm_v1());
    c.limit_margin = -4.6;
    let mut sim = SimArm::new(&c);
    let true_j4 = mid(&c, 3);
    sim.set_position(4, true_j4 + TAU);
    let mut s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    s.enable().expect("enable");
    let raw_before = s.backend_as_sim().unwrap().position(4);

    let mut targets = s.positions();
    targets[3] = 6.3; // inside the (widened) clamp, outside the encoder
    let mut go = GoTo::new(&c, targets);
    let out = s.tick(&mut go).expect("tick");
    match out.fault {
        Some(FaultReason::BadCommand { ref detail }) => {
            assert!(
                detail.contains("motor 4"),
                "detail names no motor: {detail}"
            );
            assert!(
                detail.contains("12.57"),
                "detail names no encoder range: {detail}"
            );
        }
        other => panic!("expected BadCommand from the output stage, got {other:?}"),
    }
    assert_eq!(s.state(), SessionState::Fault);

    // Nothing truncated ever went out: the motor is still where it was,
    // not slammed to the +12.57 rail (which would read back as 6.287 rad
    // in joint coordinates -- 1.7 rad away from the pose it was holding).
    for _ in 0..5 {
        s.tick(&mut go).expect("holding after the refusal");
    }
    let raw_after = s.backend_as_sim().unwrap().position(4);
    assert!(
        (raw_after - raw_before).abs() < 1e-2,
        "raw motor angle moved from {raw_before} to {raw_after}"
    );
    assert!(
        (s.positions()[3] - true_j4).abs() < 1e-2,
        "J4 left its pose: {}",
        s.positions()[3]
    );
}

#[test]
fn loop_finished_reports_a_dead_control_thread_honestly() {
    // `snapshot()` keeps returning whatever the loop last published, so a
    // caller that reports arm state from it alone (the Python surface
    // does) would keep claiming the loop is live long after the thread
    // exited. `loop_finished()` is that missing liveness signal, and it
    // must not consume the handle the way `stop_and_disable` does.
    //
    // With hold_on_fault = false a detected fault disables the arm and the
    // thread returns on its own while the `RunningArm` stays in the
    // caller's hands -- exactly the situation where an optimistic default
    // lies about a dead loop.
    let mut c = fast(ArmConfig::maker_arm_v1());
    c.hold_on_fault = false;
    let mut s = enabled_session(&c);
    s.backend_as_sim().unwrap().inject_fault(2, 0x01);
    let running = s
        .start(Box::new(HoldController::from_config(&c)), None)
        .expect("start");
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while !running.loop_finished() && Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        running.loop_finished(),
        "the control thread should have exited once the fault disabled the arm"
    );
    // The last snapshot is still readable -- that is the point: liveness
    // is reported separately instead of being guessed from the snapshot.
    let snap = running
        .snapshot()
        .expect("the exited loop still published its final snapshot");
    assert_eq!(snap.state, SessionState::Connected);
    assert!(
        snap.fault.is_some(),
        "the final snapshot must carry the fault"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn start_with_an_invalid_cpu_fails_and_returns_the_session() {
    let c = fast(ArmConfig::maker_arm_v1());
    let s = enabled_session(&c);
    let rt = Some(LoopRt {
        priority: 80,
        cpu: 1000,
    });
    let err = s
        .start(Box::new(HoldController::from_config(&c)), rt)
        .err()
        .expect("CPU 1000 does not exist on any test host");
    assert!(matches!(err.error, RtError::Cpu { cpu: 1000, .. }));
    // The loop thread is gone and the session came back as it was: it can start again.
    assert_eq!(err.session.state(), SessionState::Enabled);
    let running = err
        .session
        .start(Box::new(HoldController::from_config(&c)), None)
        .expect("a start without rt");
    let (_, res) = running.stop_and_disable();
    res.expect("clean stop");
}

#[cfg(target_os = "linux")]
#[test]
fn start_with_an_out_of_range_priority_fails_and_returns_the_session() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    for priority in [0, 100] {
        let rt = Some(LoopRt { priority, cpu: 0 });
        let err = s
            .start(Box::new(HoldController::from_config(&c)), rt)
            .err()
            .expect("SCHED_FIFO has no such priority");
        assert_eq!(err.error, RtError::PriorityRange { priority });
        assert!(err.error.to_string().contains("1..=99"), "{}", err.error);
        s = err.session;
    }
    assert_eq!(s.state(), SessionState::Enabled);
}

#[cfg(target_os = "linux")]
#[test]
fn start_without_the_rtprio_right_fails_and_returns_the_session() {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit writes one rlimit that we own; geteuid has no preconditions.
    let euid = unsafe {
        libc::getrlimit(libc::RLIMIT_RTPRIO, &mut lim);
        libc::geteuid()
    };
    if euid == 0 || lim.rlim_cur >= 80 {
        eprintln!("skipped: this host grants rtprio 80; T2 on the Thor covers the success path");
        return;
    }
    let c = fast(ArmConfig::maker_arm_v1());
    let s = enabled_session(&c);
    let rt = Some(LoopRt {
        priority: 80,
        cpu: 0,
    });
    let err = s
        .start(Box::new(HoldController::from_config(&c)), rt)
        .err()
        .expect("no rtprio right, so SCHED_FIFO 80 must be refused");
    assert!(matches!(err.error, RtError::Priority { priority: 80, .. }));
    assert_eq!(err.session.state(), SessionState::Enabled);
}

/// A `SimArm` that logs the motor id of every fault query it is sent and,
/// once the flag in `log` is set, drops motor 4's fault status replies.
struct Tap {
    sim: SimArm,
    /// (fault queries sent, motor 4 deaf)
    log: Arc<Mutex<(Vec<u8>, bool)>>,
}

impl CanBackend for Tap {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        if *data == mit::encode_fault_query(id as u8).data {
            self.log.lock().unwrap().0.push(id as u8);
        }
        self.sim.send(id, data)
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        loop {
            let r = self.sim.recv(timeout)?;
            let deaf = self.log.lock().unwrap().1;
            match r {
                Some((id, d))
                    if deaf
                        && matches!(mit::parse_frame(id, &d, &RS00),
                                    Some(mit::Parsed::Fault(f)) if f.motor_id == 4) => {}
                other => return Ok(other),
            }
        }
    }
}

#[test]
fn each_tick_sends_at_most_one_fault_query() {
    let c = fast(ArmConfig::maker_arm_v1());
    let log = Arc::new(Mutex::new((Vec::new(), false)));
    let tap = Tap {
        sim: SimArm::new(&c),
        log: log.clone(),
    };
    let mut s = Session::connect(Box::new(tap), c.clone()).expect("connect");
    s.enable().expect("enable");
    *log.lock().unwrap() = (Vec::new(), true);
    let mut hold = HoldController::from_config(&c);
    // (motor, before the tick, after the tick)
    let mut sent = Vec::new();
    for _ in 0..180 {
        let before = Instant::now();
        s.tick(&mut hold).expect("tick");
        let q = std::mem::take(&mut log.lock().unwrap().0);
        assert!(q.len() <= 1, "fault queries {q:?} in one tick");
        sent.extend(q.into_iter().map(|m| (m, before, Instant::now())));
        std::thread::sleep(Duration::from_millis(5));
    }
    let first: Vec<u8> = sent.iter().take(7).map(|q| q.0).collect();
    assert_eq!(first, (1..=7).collect::<Vec<u8>>());
    // Motor 4 never answers, yet it is asked once per interval like the
    // others, so it cannot starve them.
    let interval = Duration::from_secs_f64(c.fault_poll_interval);
    for m in 1..=7u8 {
        let mine: Vec<_> = sent.iter().filter(|q| q.0 == m).collect();
        assert!(mine.len() >= 3, "motor {m}: {} queries", mine.len());
        for w in mine.windows(2) {
            assert!(
                w[1].2 - w[0].1 >= interval,
                "motor {m} asked twice in an interval"
            );
        }
    }
    assert!(matches!(
        s.fault(),
        Some(FaultReason::FaultReplyTimeout { motor_id: 4, .. })
    ));
}

#[test]
fn a_gripper_fault_releases_the_gripper_and_holds_the_joints() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let before = s.positions();
    let sim = s.backend_as_sim().unwrap();
    // Connect and enable send disables too: count from here.
    let base = sim.disables(7);
    sim.inject_fault(7, 1 << 14);
    let mut hold = HoldController::from_config(&c);
    let stop = AtomicBool::new(false);
    s.run(&mut hold, &stop, Some(200)).expect("run 1 s");
    assert!(matches!(
        s.fault(),
        Some(FaultReason::MotorFault { motor_id: 7, .. })
    ));
    assert_eq!(s.state(), SessionState::Fault);
    let after = s.positions();
    for i in 0..6 {
        assert!((after[i] - before[i]).abs() < 1e-3, "joint {i} moved");
    }
    let sim = s.backend_as_sim().unwrap();
    assert!(!sim.enabled(7));
    assert!((1..=6).all(|m| sim.enabled(m)));
    let disables = sim.disables(7) - base;
    assert!(
        (9..=11).contains(&disables),
        "{disables} gripper disables in 1 s"
    );
}

#[test]
fn a_joint_fault_holds_every_motor_including_the_gripper() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let sim = s.backend_as_sim().unwrap();
    let base = sim.disables(7);
    sim.inject_fault(3, 1 << 14);
    let mut hold = HoldController::from_config(&c);
    let stop = AtomicBool::new(false);
    s.run(&mut hold, &stop, Some(200)).expect("run 1 s");
    assert!(matches!(
        s.fault(),
        Some(FaultReason::MotorFault { motor_id: 3, .. })
    ));
    assert_eq!(s.state(), SessionState::Fault);
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| sim.enabled(m)));
    assert_eq!(sim.disables(7), base);
}

#[test]
fn a_gripper_fault_during_a_hold_still_releases_it() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    s.backend_as_sim().unwrap().inject_fault(3, 1 << 14);
    let mut hold = HoldController::from_config(&c);
    let stop = AtomicBool::new(false);
    s.run(&mut hold, &stop, Some(5)).expect("run to the fault");
    assert_eq!(s.state(), SessionState::Fault);
    let sim = s.backend_as_sim().unwrap();
    let base = sim.disables(7);
    sim.set_temperature(7, 56.0);
    s.run(&mut hold, &stop, Some(200)).expect("run 1 s");
    // The first fault stays the session's reason, and the later one is kept.
    assert!(matches!(
        s.fault(),
        Some(FaultReason::MotorFault { motor_id: 3, .. })
    ));
    assert!(
        matches!(
            s.later_faults(),
            [FaultReason::OverTemperature { motor_id: 7, .. }]
        ),
        "{:?}",
        s.later_faults()
    );
    let sim = s.backend_as_sim().unwrap();
    assert!(!sim.enabled(7));
    assert!((1..=6).all(|m| sim.enabled(m)));
    let disables = sim.disables(7) - base;
    assert!(
        (9..=11).contains(&disables),
        "{disables} gripper disables in 1 s"
    );
}

#[test]
fn power_loss_then_return_never_re_enables() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let mut hold = HoldController::from_config(&c);
    let stop = AtomicBool::new(false);
    let sim = s.backend_as_sim().unwrap();
    for m in 1..=7 {
        sim.set_muted(m, true);
    }
    s.run(&mut hold, &stop, Some(100))
        .expect("run to the timeout");
    assert!(matches!(
        s.fault(),
        Some(FaultReason::FeedbackTimeout { .. })
    ));
    let sim = s.backend_as_sim().unwrap();
    sim.power_cycle();
    for m in 1..=7 {
        sim.set_muted(m, false);
    }
    s.run(&mut hold, &stop, Some(50))
        .expect("run after power returns");
    assert_eq!(s.state(), SessionState::Fault);
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| !sim.enabled(m)));
}

/// What the `Enobufs` bus does to the next sends.
#[derive(Default)]
struct Bus {
    /// This many sends fail.
    fail_next: u32,
    /// Every send to this motor fails.
    dead: Option<u8>,
    /// Disable frames tried, failed or not.
    disables_tried: u32,
}

/// A SimArm whose sends fail as a full CAN TX queue makes them fail on a
/// candleLight adapter: with ENOBUFS.
struct Enobufs {
    sim: SimArm,
    bus: Arc<Mutex<Bus>>,
}

impl Enobufs {
    fn session(c: &ArmConfig) -> (Session, Arc<Mutex<Bus>>) {
        let bus = Arc::new(Mutex::new(Bus::default()));
        let b = Enobufs {
            sim: SimArm::new(c),
            bus: bus.clone(),
        };
        let mut s = Session::connect(Box::new(b), c.clone()).expect("connect");
        s.enable().expect("enable");
        (s, bus)
    }
}

impl CanBackend for Enobufs {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        let mut bus = self.bus.lock().unwrap();
        if *data == mit::encode_disable(id as u8).data {
            bus.disables_tried += 1;
        }
        if bus.fail_next > 0 || bus.dead == Some(id as u8) {
            bus.fail_next = bus.fail_next.saturating_sub(1);
            return Err(TransportError::Io(
                "No buffer space available (os error 105)".into(),
            ));
        }
        self.sim.send(id, data)
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        self.sim.recv(timeout)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut self.sim)
    }
}

#[test]
fn one_failed_send_holds_and_the_loop_keeps_running() {
    let c = fast(ArmConfig::maker_arm_v1());
    let (mut s, bus) = Enobufs::session(&c);
    let before = s.positions();
    let mut hold = HoldController::from_config(&c);
    let stop = AtomicBool::new(false);
    s.run(&mut hold, &stop, Some(5)).expect("run");
    bus.lock().unwrap().fail_next = 1;
    s.run(&mut hold, &stop, Some(100))
        .expect("one failed send does not end the loop");
    assert_eq!(s.state(), SessionState::Fault);
    assert!(
        matches!(s.fault(), Some(FaultReason::Transport { .. })),
        "{:?}",
        s.fault()
    );
    let after = s.positions();
    for i in 0..7 {
        assert!((after[i] - before[i]).abs() < 1e-3, "joint {i} moved");
    }
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| sim.enabled(m)));
}

#[test]
fn a_release_after_the_loop_stopped_still_disables() {
    let c = fast(ArmConfig::maker_arm_v1());
    let (s, bus) = Enobufs::session(&c);
    let running = s
        .start(Box::new(HoldController::from_config(&c)), None)
        .expect("start");
    std::thread::sleep(Duration::from_millis(50));
    let tried = bus.lock().unwrap().disables_tried;
    bus.lock().unwrap().fail_next = u32::MAX;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !running.loop_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(running.loop_finished(), "errors that last end the loop");
    // The loop never disables on its own.
    assert_eq!(bus.lock().unwrap().disables_tried, tried);
    // The bus works again, except for motor 4.
    *bus.lock().unwrap() = Bus {
        dead: Some(4),
        ..Bus::default()
    };
    let (mut s, res) = running.stop_and_disable();
    match res {
        Err(SessionError::LoopStopped {
            error,
            release: Some(r),
        }) => {
            assert!(matches!(*error, SessionError::Transport(_)), "{error}");
            assert!(
                matches!(&*r, SessionError::NotDisabled { motors, .. } if motors == &[4]),
                "{r}"
            );
        }
        other => panic!("expected the loop error and motor 4, got {other:?}"),
    }
    let sim = s.backend_as_sim().unwrap();
    for m in 1..=7 {
        assert_eq!(sim.enabled(m), m == 4, "motor {m}");
    }
}

#[test]
fn join_gives_back_the_session_of_a_stopped_loop_and_sends_no_disable() {
    let c = fast(ArmConfig::maker_arm_v1());
    let (s, bus) = Enobufs::session(&c);
    let running = s
        .start(Box::new(HoldController::from_config(&c)), None)
        .expect("start");
    std::thread::sleep(Duration::from_millis(50));
    bus.lock().unwrap().fail_next = u32::MAX;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !running.loop_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(running.loop_finished(), "errors that last end the loop");
    // The bus works again: the join must still send nothing.
    *bus.lock().unwrap() = Bus::default();
    let (mut s, looped) = running.join().expect("no panic");
    assert!(
        matches!(looped, Err(SessionError::Transport(_))),
        "{looped:?}"
    );
    assert_eq!(s.state(), SessionState::Fault);
    assert_eq!(bus.lock().unwrap().disables_tried, 0);
    let sim = s.backend_as_sim().unwrap();
    assert!((1..=7).all(|m| sim.enabled(m)));
}

#[test]
fn a_motor_that_restarts_mid_session_holds_the_arm_and_stays_off() {
    // Legacy firmware leaves mode and the fault bits at 0, and a restarted
    // motor still answers every control frame. Only its power-up
    // announcement tells.
    let c = fast(ArmConfig::maker_arm_v1());
    let mut sim = SimArm::new(&c);
    sim.set_params_supported(false);
    let mut s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    s.enable().expect("enable");
    let mut hold = HoldController::from_config(&c);
    let stop = AtomicBool::new(false);
    s.run(&mut hold, &stop, Some(5)).expect("run");
    // Motor 2 loses power for a moment: torque off, then it announces
    // itself with a synthetic MCU id, never a real motor's.
    let sim = s.backend_as_sim().unwrap();
    let f = mit::encode_disable(2);
    sim.send(f.id, &f.data).unwrap();
    sim.queue_raw(2, [0xA5, 0x5A, 0, 0, 0, 0, 0, 2]);
    s.run(&mut hold, &stop, Some(200)).expect("run 1 s");
    assert_eq!(s.state(), SessionState::Fault);
    assert_eq!(
        s.fault(),
        Some(&FaultReason::MotorRestarted { motor_id: 2 })
    );
    let sim = s.backend_as_sim().unwrap();
    assert!(!sim.enabled(2));
    assert_eq!(sim.enables(2), 1);
    assert!([1, 3, 4, 5, 6, 7].iter().all(|&m| sim.enabled(m)));
}

/// A SimArm whose jaw is stopped by a held object: each gripper control
/// frame reaches the motor at the jaw's own position, and the commanded
/// position is logged.
struct HeldObject {
    sim: SimArm,
    sent: Arc<Mutex<Vec<f64>>>,
}

impl CanBackend for HeldObject {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        if id == 7 && data[..6] != [0xFF; 6] {
            let (pos, vel, kp, kd, tau) = mit::decode_control(data, &RS00).expect("control");
            self.sent.lock().unwrap().push(pos);
            let jaw = self.sim.position(7);
            let f = mit::encode_control(7, jaw, vel, kp, kd, tau, &RS00).unwrap();
            return self.sim.send(f.id, &f.data);
        }
        self.sim.send(id, data)
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        self.sim.recv(timeout)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut self.sim)
    }
}

#[test]
fn a_joint_fault_keeps_the_gripper_squeeze() {
    let c = fast(ArmConfig::maker_arm_v1());
    let sent = Arc::new(Mutex::new(Vec::new()));
    let b = HeldObject {
        sim: SimArm::new(&c),
        sent: sent.clone(),
    };
    let mut s = Session::connect(Box::new(b), c.clone()).expect("connect");
    s.enable().expect("enable");
    // The target is 0.02 rad past the jaw: a squeeze within 0.5 Nm / kp 20.
    let mut go = GoTo::new(&c, s.positions());
    let squeeze = s.positions()[6] - 0.02;
    go.targets[6] = squeeze;
    for _ in 0..5 {
        s.tick(&mut go).expect("tick");
    }
    s.backend_as_sim().unwrap().inject_fault(3, 1 << 14);
    for _ in 0..10 {
        s.tick(&mut go).expect("tick");
    }
    assert!(matches!(
        s.fault(),
        Some(FaultReason::MotorFault { motor_id: 3, .. })
    ));
    let last = *sent.lock().unwrap().last().unwrap();
    assert!(
        (last - squeeze).abs() < 1e-3,
        "the hold sent {last}, the squeeze was {squeeze}"
    );
}

/// A frame the host sent, and whether a reply was still on its way then.
type Sent = Arc<Mutex<Vec<(u32, [u8; 8], bool)>>>;

/// A `SimArm` whose replies reach the host `delay` after the frame that
/// caused them, as on a real bus. It logs every frame sent. A frame sent
/// while a reply is on its way is the collision P2 found: on the real bus
/// two motors then answer at once on one CAN id.
struct Delayed {
    sim: SimArm,
    delay: Duration,
    replies: std::collections::VecDeque<(Instant, u32, [u8; 8])>,
    sent: Sent,
}

impl Delayed {
    fn session(c: &ArmConfig, delay: Duration) -> (Session, Sent) {
        let sent = Sent::default();
        let b = Delayed {
            sim: SimArm::new(c),
            delay,
            replies: Default::default(),
            sent: sent.clone(),
        };
        let mut s = Session::connect(Box::new(b), c.clone()).expect("connect");
        s.enable().expect("enable");
        (s, sent)
    }
}

impl CanBackend for Delayed {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        let now = Instant::now();
        let busy = self.replies.iter().any(|r| r.0 > now);
        self.sent.lock().unwrap().push((id, *data, busy));
        self.sim.send(id, data)?;
        while let Some((id, data)) = self.sim.recv(Duration::ZERO)? {
            self.replies.push_back((now + self.delay, id, data));
        }
        Ok(())
    }

    fn recv(&mut self, _timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        match self.replies.front() {
            Some(r) if r.0 <= Instant::now() => Ok(self.replies.pop_front().map(|r| (r.1, r.2))),
            _ => Ok(None),
        }
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut self.sim)
    }
}

#[test]
fn no_frame_goes_out_while_a_reply_is_on_its_way() {
    let c = ArmConfig::maker_arm_v1();
    let (mut s, sent) = Delayed::session(&c, Duration::from_millis(1));
    let mut hold = HoldController::from_config(&c);
    // Long enough for every motor's first fault query.
    for _ in 0..20 {
        let out = s.tick(&mut hold).expect("tick");
        assert!(out.fault.is_none(), "{:?}", out.fault);
    }
    s.disable().expect("disable");
    let sent = sent.lock().unwrap();
    let early: Vec<_> = sent.iter().filter(|f| f.2).collect();
    assert!(early.is_empty(), "{} of {} frames", early.len(), sent.len());
}

#[test]
fn a_silent_motor_costs_one_reply_wait_per_tick() {
    let mut c = ArmConfig::maker_arm_v1();
    // Long against the in-memory replies, so scheduling noise cannot pass
    // for a second wait.
    c.reply_wait_us = 20_000;
    let wait = Duration::from_micros(c.reply_wait_us);
    let (mut s, sent) = Delayed::session(&c, Duration::ZERO);
    s.backend_as_sim().unwrap().set_muted(4, true);
    let mut hold = HoldController::from_config(&c);
    let start = Instant::now();
    // The fault query reaches motor 4 on the fourth tick, before the fault.
    while s.fault().is_none() {
        assert!(start.elapsed() < Duration::from_secs(2), "no fault");
        sent.lock().unwrap().clear();
        let t = Instant::now();
        s.tick(&mut hold).expect("tick");
        let took = t.elapsed();
        assert!(took >= wait && took < 2 * wait, "a tick took {took:?}");
        let control: Vec<u32> = sent
            .lock()
            .unwrap()
            .iter()
            .filter(|f| f.1[..6] != [0xFF; 6])
            .map(|f| f.0)
            .collect();
        assert_eq!(control, (1..=7).collect::<Vec<u32>>());
    }
    assert!(
        matches!(
            s.fault(),
            Some(FaultReason::FeedbackTimeout { motor_id: 4, .. })
        ),
        "{:?}",
        s.fault()
    );
}

/// A `SimArm` and every frame sent to it, in order.
struct Bench {
    sim: SimArm,
    sent: Vec<(u32, [u8; 8])>,
}

/// A `Bench` the test reaches while the loop thread owns the session.
#[derive(Clone)]
struct Shared(Arc<Mutex<Bench>>);

impl CanBackend for Shared {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        let mut g = self.0.lock().unwrap();
        g.sent.push((id, *data));
        g.sim.send(id, data)
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        self.0.lock().unwrap().sim.recv(timeout)
    }
}

/// A loop at the measured pose on a shared sim, with 0.3 N m of feed-forward on every joint.
fn shared_loop(c: &ArmConfig) -> (Shared, RunningArm) {
    let sim = Shared(Arc::new(Mutex::new(Bench {
        sim: SimArm::new(c),
        sent: Vec::new(),
    })));
    let mut s = Session::connect(Box::new(sim.clone()), c.clone()).expect("connect");
    s.enable().expect("enable");
    let mut ctrl = GoTo::new(c, s.positions());
    ctrl.tau = 0.3;
    let arm = s.start(Box::new(ctrl), None).expect("start");
    std::thread::sleep(Duration::from_millis(50));
    (sim, arm)
}

fn wait_for(what: &str, done: impl Fn() -> bool) {
    let end = Instant::now() + Duration::from_secs(5);
    while !done() {
        assert!(Instant::now() < end, "no {what} within 5 s");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn ramp_ticks(c: &ArmConfig) -> usize {
    (RELEASE_RAMP_S * c.control_rate_hz).round() as usize
}

#[test]
fn a_release_ramps_kp_and_the_feed_forward_to_zero_then_disables() {
    // Ruling R36.
    let c = fast(ArmConfig::maker_arm_v1());
    let (sim, running) = shared_loop(&c);
    let disable = |m: u8| mit::encode_disable(m).data;
    let start = sim.0.lock().unwrap().sent.len();
    let t = Instant::now();
    running.ramp_down();
    wait_for("end of the ramp", || running.loop_finished());
    assert!(t.elapsed() >= Duration::from_secs_f64(RELEASE_RAMP_S * 0.95));
    assert!(running.ramp_done() && !running.ramp_held());
    let (records, dropped) = running.drain();
    assert_eq!(dropped, 0);
    assert!(records.windows(2).all(|w| w[1].tick == w[0].tick + 1));
    let n = ramp_ticks(&c);
    let (before, ramp) = records.split_at(records.len() - n);
    let from = &before.last().expect("a tick before the ramp").sent;
    assert_eq!(from[2].kp, c.joints[2].kp);
    assert_eq!(from[2].tau, 0.3);
    for (k, r) in ramp.iter().enumerate() {
        let s = 1.0 - (k + 1) as f64 / n as f64;
        for (j, (got, was)) in r.sent.iter().zip(from).enumerate() {
            assert_eq!(
                (got.pos, got.vel, got.kd),
                (was.pos, 0.0, was.kd),
                "tick {k}"
            );
            assert!((got.kp - was.kp * s).abs() < 1e-12, "tick {k} joint {j}");
            assert!((got.tau - was.tau * s).abs() < 1e-12, "tick {k} joint {j}");
        }
    }
    let last = &ramp.last().unwrap().sent;
    assert!(last
        .iter()
        .all(|c| c.kp == 0.0 && c.tau == 0.0 && c.kd > 0.0));
    // The loop has ended after the last ramp tick, and no motor has had a disable yet.
    let end = {
        let g = sim.0.lock().unwrap();
        assert!((1..=7).all(|m| g.sim.enabled(m)));
        assert!(!g.sent[start..].iter().any(|f| f.1 == disable(f.0 as u8)));
        g.sent.len()
    };
    let (_, res) = running.stop_and_disable();
    res.expect("every motor confirmed off");
    let g = sim.0.lock().unwrap();
    assert!((1..=7).all(|m| !g.sim.enabled(m)));
    assert!((1..=7u8).all(|m| g.sent[end..].contains(&(m.into(), disable(m)))));
}

#[test]
fn a_fault_during_the_ramp_holds_and_disables_nothing() {
    let c = fast(ArmConfig::maker_arm_v1());
    let (sim, running) = shared_loop(&c);
    let disables = |sim: &Shared| -> Vec<u32> {
        let g = sim.0.lock().unwrap();
        (1..=7).map(|m| g.sim.disables(m)).collect()
    };
    // Motor `m` gets hot `after` from now, in the ramp.
    let heat = |m: u8, after: u64| {
        let sim = sim.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(after));
            sim.0.lock().unwrap().sim.set_temperature(m, 80.0);
        })
    };
    let before = disables(&sim);
    let hot = heat(3, 300);
    let Err(running) = running.release() else {
        panic!("released through a fault");
    };
    hot.join().unwrap();
    assert!(running.ramp_held());
    let snap = running.snapshot().unwrap();
    assert_eq!(snap.state, SessionState::Fault);
    assert!(matches!(
        snap.fault,
        Some(FaultReason::OverTemperature { motor_id: 3, .. })
    ));
    std::thread::sleep(Duration::from_millis(100));
    assert!(!running.loop_finished(), "the loop holds");
    assert!((1..=7).all(|m| sim.0.lock().unwrap().sim.enabled(m)));
    assert_eq!(disables(&sim), before);
    // The hold is back at full kp.
    let (records, _) = running.drain();
    assert_eq!(records.last().unwrap().sent[0].kp, c.joints[0].kp);
    // A release from the fault hold ramps the hold down, and a later fault does not end it.
    let hot = heat(4, 500);
    running.ramp_down();
    wait_for("later fault", || {
        running
            .snapshot()
            .unwrap()
            .later_faults
            .iter()
            .any(|f| matches!(f, FaultReason::OverTemperature { motor_id: 4, .. }))
    });
    hot.join().unwrap();
    assert!(!running.ramp_held() && !running.loop_finished());
    wait_for("end of the ramp", || running.loop_finished());
    assert!(running.ramp_done());
    let (s, res) = running.stop_and_disable();
    res.expect("every motor confirmed off");
    assert_eq!(s.state(), SessionState::Connected);
    assert!((1..=7).all(|m| !sim.0.lock().unwrap().sim.enabled(m)));
}

#[test]
fn a_gripper_released_on_its_own_fault_gets_no_ramp_frames() {
    let c = fast(ArmConfig::maker_arm_v1());
    let (sim, running) = shared_loop(&c);
    sim.0.lock().unwrap().sim.inject_fault(7, 1 << 14);
    wait_for("gripper fault", || {
        running
            .snapshot()
            .is_some_and(|s| matches!(s.fault, Some(FaultReason::MotorFault { motor_id: 7, .. })))
    });
    std::thread::sleep(Duration::from_millis(50));
    let start = sim.0.lock().unwrap().sent.len();
    running.ramp_down();
    wait_for("end of the ramp", || running.loop_finished());
    assert!(running.ramp_done());
    {
        let g = sim.0.lock().unwrap();
        let frames = &g.sent[start..];
        let control = |m: u32| {
            frames
                .iter()
                .filter(|f| f.0 == m && f.1[..6] != [0xFF; 6])
                .count()
        };
        assert_eq!(
            control(7),
            0,
            "a control frame reached the released gripper"
        );
        assert!((1..=6).all(|m| control(m) >= ramp_ticks(&c)));
        let resent = frames
            .iter()
            .filter(|f| **f == (7, mit::encode_disable(7).data))
            .count();
        assert!(resent >= 15, "{resent} gripper disables in the ramp");
        assert!(!g.sim.enabled(7) && (1..=6).all(|m| g.sim.enabled(m)));
    }
    let (_, res) = running.stop_and_disable();
    res.expect("every motor confirmed off");
}
