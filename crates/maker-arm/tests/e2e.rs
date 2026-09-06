//! The full session ladder over the in-memory simulator: connect → enable
//! → hold → fault → hold-on-fault → clear_faults → re-enable → release.
//! This is the end-to-end composition test the MA0 report called out as
//! missing (protocol encode → backend send → backend recv → protocol
//! parse, across all three crates).

use maker_arm::{ArmConfig, FaultReason, HoldController, Session, SessionState, SimArm};

#[test]
fn full_ladder_connect_fault_recover_release() {
    let mut c = ArmConfig::maker_arm_v1();
    c.inter_frame_us = 0;
    let sim = SimArm::new(&c);
    let mut s = Session::connect(Box::new(sim), c.clone()).expect("connect");
    let home = s.positions();

    // enable + hold
    s.enable().expect("enable");
    let mut hold = HoldController::from_config(&c);
    for _ in 0..20 {
        s.tick(&mut hold).expect("tick");
    }
    assert_eq!(s.state(), SessionState::Enabled);

    // fault → holding. `inject_fault` mutates the sim directly with no
    // frame, so detection lands one tick after it -- the reply to THIS
    // tick's own MIT send, drained at the end of the tick (Task 7's
    // convention; the same one-control-period latency a real motor's
    // feedback would have).
    s.backend_as_sim().unwrap().inject_fault(3, 0x08);
    let out1 = s.tick(&mut hold).expect("tick 1: fault not visible yet");
    assert!(out1.fault.is_none());
    let out2 = s.tick(&mut hold).expect("tick 2: fault now visible");
    assert_eq!(
        out2.fault,
        Some(FaultReason::MotorFault {
            motor_id: 3,
            bits: 0x08
        })
    );
    assert_eq!(s.state(), SessionState::Fault);
    for _ in 0..10 {
        s.tick(&mut hold).expect("holding");
    }
    for (i, q) in s.positions().iter().enumerate() {
        assert!((q - home[i]).abs() < 1e-2, "joint {i} drifted");
    }

    // recover: clear faults (motor-side clear included), re-enable, release
    s.clear_faults().expect("clear");
    assert_eq!(s.state(), SessionState::Connected);
    assert!(
        s.fault().is_none(),
        "clear_faults must clear the fault reason"
    );
    s.enable().expect("re-enable");
    let mut hold2 = HoldController::from_config(&c);
    for _ in 0..5 {
        let out = s.tick(&mut hold2).expect("tick");
        assert!(out.fault.is_none());
    }
    s.disable().expect("release");
    assert_eq!(s.state(), SessionState::Connected);
    let sim = s.backend_as_sim().unwrap();
    for id in 1..=7u8 {
        assert!(!sim.enabled(id));
    }
}
