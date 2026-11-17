use maker_arm::{ArmConfig, SimArm};
use maker_arm_cli::files::GripMap;
use maker_arm_cli::{confirm_release, doctor, scan, watch_sweep, zero_motor};
use maker_arm_protocol::mit;
use maker_arm_transport::{CanBackend, TransportError};
use std::io::{Cursor, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const BIN: &str = env!("CARGO_BIN_EXE_maker-arm-cli");

/// Runs `maker-arm-cli --sim <args>` with `stdin` typed in: its exit code and stdout.
fn sim_cli(args: &[&str], stdin: &str) -> (Option<i32>, String) {
    let mut child = Command::new(BIN)
        .arg("--sim")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

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
    assert!(r1.params);
    let r4 = &report.rows[3];
    assert!(r4.present && !r4.in_limits);
    assert!(!report.passed());
    // Parameter access on every motor: nothing to warn about.
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
}

#[test]
fn scan_shows_parameter_access_and_fault_text() {
    let c = ArmConfig::maker_arm_v1();
    for r in scan(&mut SimArm::new(&c), &c).expect("scan") {
        assert!(r.present && r.params, "motor {}", r.motor_id);
        assert_eq!(r.fault_word, 0);
    }
    let (code, out) = sim_cli(&["scan"], "");
    assert_eq!(code, Some(0), "{out}");
    let rows: Vec<&str> = out.lines().skip(1).collect();
    assert_eq!(rows.len(), 7, "{out}");
    for row in rows {
        let cols: Vec<&str> = row.split_whitespace().collect();
        // present and params, then the decoded fault word
        assert_eq!((cols[3], cols[7]), ("true", "true"), "{row}");
        assert!(row.ends_with("no fault"), "{row}");
    }
}

#[test]
fn doctor_warns_about_rs02_range_without_parameter_access() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_params_supported(false);
    let report = doctor(&mut sim, &c).expect("doctor");
    for r in &report.rows {
        assert!(r.present && !r.params, "motor {}", r.motor_id);
        assert_eq!((r.run_mode, r.can_timeout, r.vbus), (None, None, None));
    }
    let rs02 = report
        .warnings
        .iter()
        .find(|w| w.contains("RS02"))
        .expect("an RS02 warning");
    assert!(
        rs02.contains("motors 2, 3") && rs02.contains("no parameter access"),
        "{rs02}"
    );
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("motors 1, 2, 3, 4, 5, 6, 7") && w.contains("watchdog")),
        "{:?}",
        report.warnings
    );
    // Missing parameter access is a warning, not a failure.
    assert!(report.passed());
}

#[test]
fn doctor_fails_on_a_reported_fault() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    assert!(doctor(&mut sim, &c).expect("doctor").passed());
    sim.inject_fault(4, 1 << 9);
    let report = doctor(&mut sim, &c).expect("doctor");
    assert!(!report.passed());
    let r = &report.rows[3];
    assert!(r.present);
    assert_eq!(r.fault_word, 1 << 9);
    // No clearing frame went to the faulted motor: the fault is still there.
    assert_eq!(scan(&mut sim, &c).expect("scan")[3].fault_word, 1 << 9);
}

#[test]
fn zero_prompt_names_the_vendor_pose() {
    let (code, out) = sim_cli(&["zero", "--motor", "5"], "\nno\n");
    assert_eq!(code, Some(1), "{out}");
    assert!(
        out.contains("folded against the base, gripper fully closed"),
        "{out}"
    );
    assert!(out.contains("support the arm"), "{out}");
}

#[test]
fn zero_asks_before_it_disables_the_motors() {
    // Input ends at the first prompt: zero stops before the disables, so
    // the YES prompt, which comes after them, never shows.
    let (code, out) = sim_cli(&["zero", "--motor", "5"], "");
    assert_eq!(code, Some(1), "{out}");
    assert!(
        out.contains("every motor will be disabled: support the arm, then press ENTER"),
        "{out}"
    );
    assert!(!out.contains("type YES"), "{out}");
}

#[test]
fn zero_warns_only_outside_the_enable_window() {
    // The gripper's zero is just past its range, inside the grace that
    // enable accepts.
    let (code, out) = sim_cli(&["zero", "--motor", "7", "--yes"], "");
    assert_eq!(code, Some(0), "{out}");
    assert!(out.contains("motor 7 zeroed"), "{out}");
    assert!(!out.contains("WARNING"), "{out}");
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
fn scan_and_doctor_report_fault_word_and_temperature_on_the_right_rows() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.inject_fault(2, 0x21);
    sim.set_temperature(5, 71.5);

    let rows = scan(&mut sim, &c).expect("scan");
    let faulted = rows.iter().find(|r| r.motor_id == 2).unwrap();
    assert!(faulted.present);
    assert_eq!(faulted.fault_word, 0x21);
    // A faulted motor gets no state read, so it has no position or temperature.
    assert!(faulted.joint_pos.is_nan() && faulted.temperature.is_nan());
    let hot = rows.iter().find(|r| r.motor_id == 5).unwrap();
    assert!((hot.temperature - 71.5).abs() < 1e-9);
    assert_eq!(hot.fault_word, 0); // unaffected motor stays clean

    let report = doctor(&mut sim, &c).expect("doctor");
    let faulted = report.rows.iter().find(|r| r.motor_id == 2).unwrap();
    assert_eq!(faulted.fault_word, 0x21);
    let hot = report.rows.iter().find(|r| r.motor_id == 5).unwrap();
    assert!((hot.temperature - 71.5).abs() < 1e-9);
}

#[test]
fn zero_motor_zeroes_and_reports() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_position(5, 2.0);
    let pos = zero_motor(&mut sim, &c, 5, || true).expect("zero");
    assert!(pos.abs() < 1e-3);
    assert!(sim.position(5).abs() < 1e-3);
    // zero_motor reports the joint position that the new register value
    // maps to. It does not check it against the joint's limits: `enable()`
    // is that gate. So this assertion is about what set-zero did to the
    // register, not a claim that 0.0 rad is a safe or reachable pose.
}

#[test]
fn zero_motor_refuses_a_faulted_motor() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_position(5, 2.0);
    sim.inject_fault(5, 1 << 7);
    let e = zero_motor(&mut sim, &c, 5, || true).unwrap_err();
    assert!(e.contains("encoder uncalibrated"), "{e}");
    assert_eq!(sim.position(5), 2.0);
}

/// Every frame sent, in order: (id, data).
type SentLog = Arc<Mutex<Vec<(u32, [u8; 8])>>>;

/// A SimArm that records every frame sent to it.
struct Recorder {
    sim: SimArm,
    sent: SentLog,
}

impl CanBackend for Recorder {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        self.sent.lock().unwrap().push((id, *data));
        self.sim.send(id, data)
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        self.sim.recv(timeout)
    }
}

#[test]
fn zero_motor_disables_every_motor_before_the_prompt_and_set_zero() {
    let c = ArmConfig::maker_arm_v1();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let mut b = Recorder {
        sim: SimArm::new(&c),
        sent: sent.clone(),
    };
    // A dead daemon left every motor enabled on its last frame.
    for j in &c.joints {
        let f = mit::encode_enable(j.motor_id);
        b.sim.send(f.id, &f.data).unwrap();
        assert!(b.sim.enabled(j.motor_id));
    }
    let mut at_prompt = None;
    zero_motor(&mut b, &c, 5, || {
        at_prompt = Some(sent.lock().unwrap().len());
        true
    })
    .expect("zero");
    let at_prompt = at_prompt.expect("the prompt ran");
    let sent = sent.lock().unwrap();
    let at = |f: maker_arm_protocol::Frame| sent.iter().position(|s| *s == (f.id, f.data));
    let zero_at = at(mit::encode_set_zero(5)).expect("set-zero sent");
    assert!(at_prompt <= zero_at);
    for j in &c.joints {
        let off = at(mit::encode_disable(j.motor_id)).expect("disable sent");
        assert!(
            off < at_prompt,
            "motor {} disabled after the prompt",
            j.motor_id
        );
        assert!(!b.sim.enabled(j.motor_id), "motor {}", j.motor_id);
    }
}

#[test]
fn zero_motor_aborts_with_every_motor_disabled() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.set_position(5, 2.0);
    let f = mit::encode_enable(5);
    sim.send(f.id, &f.data).unwrap();
    assert_eq!(
        zero_motor(&mut sim, &c, 5, || false).unwrap_err(),
        "aborted"
    );
    assert!(!sim.enabled(5));
    assert_eq!(sim.position(5), 2.0);
}

#[test]
fn zero_motor_rejects_unknown_id() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    assert!(zero_motor(&mut sim, &c, 9, || true).is_err());
}

#[test]
fn confirm_release_accepts_only_release() {
    // wrong input, then case-insensitive + whitespace-tolerant match
    let mut input = Cursor::new(b"nope\n  release \n".to_vec());
    let mut out = Vec::new();
    confirm_release(&mut input, &mut out); // returns only on RELEASE
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
        confirm_release(&mut input, &mut out); // never expected to return
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
    // Returns (no `Result` to check: every failure mode inside is
    // warn-and-retry, so the signature has no `Err` variant at all).
    confirm_release(&mut input, &mut out);
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
    // Returns despite the first write AND flush failing.
    confirm_release(&mut input, &mut out);
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

/// The numbers of each `WATCH` line.
fn watch_rows(out: &str) -> Vec<Vec<f64>> {
    out.lines()
        .map(|l| {
            let mut w = l.split_whitespace();
            assert_eq!(w.next(), Some("WATCH"), "{l}");
            w.map(|x| x.parse().unwrap()).collect()
        })
        .collect()
}

#[test]
fn watch_prints_count_lines_of_seven_values() {
    let (code, out) = sim_cli(&["scan", "--watch", "--count", "2", "--hz", "50"], "");
    assert_eq!(code, Some(0));
    let rows = watch_rows(&out);
    assert_eq!(rows.len(), 2);
    for r in rows {
        assert_eq!(r.len(), 8); // t then seven values
        assert!(r.iter().all(|x| x.is_finite()), "{r:?}");
    }
}

#[test]
fn watch_applies_the_profile_direction() {
    let c = ArmConfig::maker_arm_v1();
    let mut flipped = c.clone();
    flipped.joints[4].direction = -1.0;
    // The same motors, read through the identity and through the flipped J5.
    let mut sim = SimArm::new(&c);
    let grip = GripMap::provisional();
    let (plain, _) = watch_sweep(&mut sim, &c, &grip, &mut None).unwrap();
    let (flip, _) = watch_sweep(&mut sim, &flipped, &grip, &mut None).unwrap();
    assert!((flip[4] + plain[4]).abs() < 1e-9, "{flip:?} {plain:?}");
    assert_eq!(flip[3], plain[3]);
}

#[test]
fn watch_never_enables() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    let mut wraps = None;
    for _ in 0..3 {
        let (q, _) = watch_sweep(&mut sim, &c, &GripMap::provisional(), &mut wraps).unwrap();
        assert!(q.iter().all(|x| x.is_finite()));
    }
    for j in &c.joints {
        assert_eq!(sim.enables(j.motor_id), 0);
    }
}

#[test]
fn watch_reports_the_fault_word_of_a_faulted_motor() {
    let c = ArmConfig::maker_arm_v1();
    let mut sim = SimArm::new(&c);
    sim.inject_fault(3, 1 << 14);
    let (q, faults) = watch_sweep(&mut sim, &c, &GripMap::provisional(), &mut None).unwrap();
    // A faulted motor gets no state read, so it reads nan like an absent one.
    assert!(q[2].is_nan());
    assert_eq!(faults, [0, 0, 1 << 14, 0, 0, 0, 0]);
}
