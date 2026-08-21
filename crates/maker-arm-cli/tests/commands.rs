use maker_arm::{ArmConfig, SimArm};
use maker_arm_cli::{doctor, scan};

#[test]
fn scan_reports_all_seven_and_flags_a_missing_motor() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_muted(6, true);
    let rows = scan(&mut sim, &c).expect("scan");
    assert_eq!(rows.len(), 7);
    for r in &rows {
        if r.motor_id == 6 {
            assert!(!r.present);
        } else {
            assert!(r.present, "motor {} should be present", r.motor_id);
            assert_eq!(r.mode, 0); // scan is read-only: nothing gets enabled
        }
    }
    assert_eq!(rows[1].model, "RS02");
    assert_eq!(rows[6].name, "gripper");
}

#[test]
fn doctor_reads_params_and_checks_limits() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_position(4, 3.5); // out of J4's window, unwrappable
    let report = doctor(&mut sim, &c).expect("doctor");
    assert_eq!(report.rows.len(), 7);
    let r1 = &report.rows[0];
    assert!(r1.present && r1.in_limits);
    assert_eq!(r1.run_mode, Some(0));
    assert_eq!(r1.vbus, Some(24.0));
    let r4 = &report.rows[3];
    assert!(r4.present && !r4.in_limits);
    // The RS02 firmware trap must be surfaced, not silently omitted:
    // the official SDK ships no READ_VERSION, so doctor cannot check it.
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("RS02") && w.contains("firmware")),
        "warnings: {:?}",
        report.warnings
    );
}
