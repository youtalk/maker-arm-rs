use maker_arm::{ArmConfig, SimArm};
use maker_arm_cli::{confirm_release, doctor, scan, zero_motor};
use std::io::{Cursor, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

#[test]
fn zero_motor_zeroes_and_reports() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_position(5, 2.0);
    let pos = zero_motor(&mut sim, &c, 5).expect("zero");
    assert!(pos.abs() < 1e-3);
    assert!(sim.position(5).abs() < 1e-3);
    // Note: J5's configured joint range is 0.577..3.641 -- it does not
    // include zero. zero_motor reports whatever the motor's raw position
    // maps to in joint coordinates; it does not validate that value
    // against the joint's soft limits (that gate belongs to `enable()`,
    // which runs later and would refuse to energize this joint at 0.0
    // rad). So this assertion is about what set-zero actually did to the
    // register, not a claim that 0.0 rad is a safe or reachable pose here.
}

#[test]
fn zero_motor_rejects_unknown_id() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    assert!(zero_motor(&mut sim, &c, 9).is_err());
}

#[test]
fn confirm_release_accepts_only_release() {
    // wrong input, then case-insensitive + whitespace-tolerant match
    let mut input = Cursor::new(b"nope\n  release \n".to_vec());
    let mut out = Vec::new();
    confirm_release(&mut input, &mut out).expect("returns after RELEASE");
    let text = String::from_utf8(out).unwrap();
    // the prompt and the mismatch warning both appeared
    assert!(text.contains("Type RELEASE"));
    assert!(text.contains("torque remains enabled"));
}

/// A `Write` sink that a background thread can fill while the test thread
/// reads it, used only by the EOF test below.
#[derive(Clone, Default)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn confirm_release_never_returns_on_eof_and_keeps_retrying() {
    // `confirm_release` sleeps 1s and retries forever on EOF (a closed
    // stdin must never be read as consent to release torque), so this test
    // cannot simply call it and wait for a result -- that would hang CI
    // forever. Instead it drives the call on a background thread and
    // bounds the wait with `recv_timeout`: a channel send only happens if
    // the function returns, so a timeout on the channel IS the assertion
    // that it did not return. The timeout (3.8s) comfortably observes
    // three 1s retry cycles with real margin left over -- a tighter bound
    // (this used to be 2.5s) leaves too little slack before the next
    // retry boundary on a loaded CI runner, which would turn a passing
    // test into a flaky failure rather than a real signal.
    let out = SharedBuf::default();
    let out_for_thread = out.clone();
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let mut input = Cursor::new(Vec::new()); // empty: read_line is EOF immediately
        let mut out = out_for_thread;
        let _ = confirm_release(&mut input, &mut out); // never expected to return
        let _ = tx.send(());
    });
    // Bounded wait: if this ever receives, confirm_release wrongly returned
    // on EOF -- fail loudly instead of hanging.
    assert!(
        rx.recv_timeout(Duration::from_millis(3800)).is_err(),
        "confirm_release returned on EOF -- it must never release torque on a dead stdin"
    );
    let text = String::from_utf8(out.0.lock().unwrap().clone()).unwrap();
    assert!(text.contains("Type RELEASE"));
    assert!(
        text.matches("input is unavailable").count() >= 2,
        "expected at least two EOF retries in 3.8s, got: {text}"
    );
    // The spawned thread loops forever (by design); it is intentionally
    // left detached and dies with the test process.
}

/// A mock `BufRead` whose `read_line` is overridden (a provided trait
/// method may always be overridden by a concrete impl) to hand back
/// specific `Err`s before eventually producing real data -- something a
/// `Cursor` cannot do, since it never fails.
struct FlakyThenRelease {
    calls: u32,
}

impl std::io::Read for FlakyThenRelease {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        unreachable!("confirm_release only calls read_line, which is overridden below")
    }
}

impl std::io::BufRead for FlakyThenRelease {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        unreachable!("confirm_release only calls read_line, which is overridden below")
    }

    fn consume(&mut self, _amt: usize) {}

    fn read_line(&mut self, buf: &mut String) -> std::io::Result<usize> {
        self.calls += 1;
        match self.calls {
            // First call: the exact error kind a retried-but-still-failing
            // EINTR would produce, if it ever reached this layer.
            1 => Err(std::io::Error::from(std::io::ErrorKind::Interrupted)),
            // Second call: a different error kind, to confirm the handling
            // is "any Err", not special-cased to Interrupted alone.
            2 => Err(std::io::Error::other("simulated stdin failure")),
            _ => {
                buf.push_str(
                    "RELEASE
",
                );
                Ok(buf.len())
            }
        }
    }
}

#[test]
fn confirm_release_treats_read_errors_like_eof() {
    // Regression test: a bare `?` on `read_line`'s result used to let any
    // non-EOF I/O error propagate out of `confirm_release`, and the
    // caller's `?` on THAT would drop the `RunningArm`, whose `Drop`
    // disables the motors -- releasing torque with no typed RELEASE at
    // all. Every read error must be treated exactly like EOF: keep torque
    // on, warn, sleep, and retry.
    let mut input = FlakyThenRelease { calls: 0 };
    let mut out = Vec::new();
    confirm_release(&mut input, &mut out)
        .expect("returns Ok after RELEASE despite two read errors first");
    assert_eq!(input.calls, 3);
    let text = String::from_utf8(out).unwrap();
    assert_eq!(
        text.matches("input is unavailable").count(),
        2,
        "expected one retry warning per read error, got: {text}"
    );
}

/// A mock `Write` whose `write`/`flush` fail on their very first call each,
/// then succeed and capture output afterwards -- used only by the write-
/// error regression test below.
struct FlakyWrite {
    write_calls: u32,
    flush_calls: u32,
    buf: Vec<u8>,
}

impl Write for FlakyWrite {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.write_calls += 1;
        if self.write_calls == 1 {
            return Err(std::io::Error::other("simulated broken stdout (write)"));
        }
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_calls += 1;
        if self.flush_calls == 1 {
            return Err(std::io::Error::other("simulated broken stdout (flush)"));
        }
        Ok(())
    }
}

#[test]
fn confirm_release_treats_write_errors_like_eof() {
    // Regression test: `writeln!`/`flush` inside `confirm_release` used to
    // propagate via a bare `?`. A broken stdout (or any other write
    // failure) would then make `confirm_release` return `Err`, and the
    // caller's `?` on that would drop the live `RunningArm` -- releasing
    // torque with no typed RELEASE at all, the mirror image of the read-
    // error bug fixed above, reached through the write side instead.
    let mut out = FlakyWrite {
        write_calls: 0,
        flush_calls: 0,
        buf: Vec::new(),
    };
    // "nope" forces one mismatch cycle before "RELEASE": the first
    // prompt's write AND flush both fail (see FlakyWrite above), so this
    // also exercises the write-failure pacing sleep before the second,
    // successful, iteration.
    let mut input = Cursor::new(b"nope\nRELEASE\n".to_vec());
    confirm_release(&mut input, &mut out)
        .expect("returns Ok after RELEASE despite the first write and flush failing");
    assert!(
        out.write_calls >= 2,
        "expected at least one write to succeed after the first failure"
    );
    assert!(
        out.flush_calls >= 2,
        "expected at least one flush to succeed after the first failure"
    );
    let text = String::from_utf8(out.buf).unwrap();
    // The first prompt's write failed and was never captured, but a later
    // (successful) prompt attempt must still have gotten through.
    assert!(
        text.contains("Type RELEASE"),
        "prompt never made it through: {text}"
    );
}
