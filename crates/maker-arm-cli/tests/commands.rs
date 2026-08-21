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

#[test]
fn doctor_reports_absent_motor_as_not_in_limits_with_no_params() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_muted(5, true);
    let report = doctor(&mut sim, &c).expect("doctor");
    let r = report
        .rows
        .iter()
        .find(|r| r.motor_id == 5)
        .expect("motor 5 row");
    assert!(!r.present);
    assert_eq!(r.run_mode, None);
    assert_eq!(r.can_timeout, None);
    assert_eq!(r.vbus, None);
    // An absent motor has no position reading at all -- it must never be
    // reported as confirmed in limits, since that would tell an operator
    // it is safe to energize when nothing was actually checked.
    assert!(!r.in_limits);
}

#[test]
fn scan_and_doctor_report_fault_bits_and_temperature_on_the_right_rows() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.inject_fault(2, 0x21);
    sim.set_temperature(5, 71.5);

    let rows = scan(&mut sim, &c).expect("scan");
    let faulted = rows.iter().find(|r| r.motor_id == 2).unwrap();
    assert_eq!(faulted.fault_bits, 0x21);
    assert!((faulted.temperature - 35.0).abs() < 1e-9); // unaffected motor stays at sim default
    let hot = rows.iter().find(|r| r.motor_id == 5).unwrap();
    assert!((hot.temperature - 71.5).abs() < 1e-9);
    assert_eq!(hot.fault_bits, 0); // unaffected motor stays clean

    let report = doctor(&mut sim, &c).expect("doctor");
    let faulted = report.rows.iter().find(|r| r.motor_id == 2).unwrap();
    assert_eq!(faulted.fault_bits, 0x21);
    let hot = report.rows.iter().find(|r| r.motor_id == 5).unwrap();
    assert!((hot.temperature - 71.5).abs() < 1e-9);
}
