use maker_arm::dynamics::{Body, Dynamics, Friction};
use maker_arm::kinematics::Link;
use maker_arm::tracking::{Push, TrackingController, TrackingParams, ROW};
use maker_arm::{ArmConfig, FaultReason, Session, SessionState, SimArm};

fn fast(mut c: ArmConfig) -> ArmConfig {
    c.inter_frame_us = 0;
    c
}

fn enabled_session(c: &ArmConfig) -> Session {
    let mut s = Session::connect(Box::new(SimArm::new(c)), c.clone()).expect("connect");
    s.enable().expect("enable");
    s
}

fn model() -> Dynamics {
    let z = Some([0.0, 0.0, 1.0]);
    let body = Body {
        mass: 0.5,
        com: [0.05, 0.0, 0.0],
        inertia: [[1e-3, 0.0, 0.0], [0.0, 2e-3, 0.0], [0.0, 0.0, 2e-3]],
    };
    let links: Vec<Link> = (0..6)
        .map(|_| Link::new([0.1, 0.0, 0.0], [0.0; 3], z))
        .collect();
    Dynamics::new(
        [0.0; 3],
        &links,
        [body; 6],
        [Friction::default(); 6],
        [0.0, -9.81, 0.0],
    )
    .unwrap()
}

#[test]
fn a_session_runs_1000_tracking_ticks_without_a_fault() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let mut ctrl = TrackingController::new(model(), TrackingParams::default(), &c).unwrap();
    let start = s.positions();
    let rows = (0..=100)
        .map(|k| {
            let mut r = [0.0; ROW];
            r[..ROW].copy_from_slice(&start[..ROW]);
            for v in r.iter_mut().take(ROW - 1) {
                *v += 0.001 * k as f64;
            }
            r
        })
        .collect();
    ctrl.shared().push(Push::new(0.0, 0.05, rows).unwrap());
    for _ in 0..1000 {
        let out = s.tick(&mut ctrl).expect("tick");
        assert!(out.fault.is_none(), "{:?}", out.fault);
    }
    assert_eq!(s.state(), SessionState::Enabled);
}

#[test]
fn a_non_finite_model_output_ends_in_the_fault_hold() {
    let c = fast(ArmConfig::maker_arm_v1());
    let mut s = enabled_session(&c);
    let mut ctrl = TrackingController::new(model(), TrackingParams::default(), &c).unwrap();
    s.tick(&mut ctrl).expect("first tick");
    let mut absurd = [1e300; ROW];
    absurd[ROW - 1] = s.positions()[ROW - 1];
    ctrl.shared()
        .push(Push::new(0.0, 0.01, vec![absurd]).unwrap());
    let mut faulted = false;
    for _ in 0..10 {
        let out = s.tick(&mut ctrl).expect("tick");
        if matches!(out.fault, Some(FaultReason::BadCommand { .. })) {
            faulted = true;
            break;
        }
    }
    assert!(
        faulted,
        "a 1e300 plan must overflow the model into a non-finite command"
    );
    assert_eq!(s.state(), SessionState::Fault);
}
