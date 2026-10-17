//! `maker-arm-cli serve` on the sim interface, driven over its socket as the ROS 2 plugin
//! drives it. No test opens a real interface.

mod common;

use common::{run_to_exit, scratch_dir, serve_command, Client, Daemon};
use maker_arm::ArmConfig;
use maker_arm_cli::files::GripMap;
use maker_arm_cli::wire::{Mode, Op, Role};
use std::time::Duration;

fn sleep_ms(ms: u64) {
    std::thread::sleep(Duration::from_millis(ms));
}

/// The 16 hex digits after `"digest": "` in a token line.
fn digest_of(line: &str) -> String {
    let at = line.find("\"digest\": \"").expect("a digest field") + 11;
    let hex = &line[at..at + 16];
    assert!(hex.chars().all(|c| c.is_ascii_hexdigit()), "{line}");
    hex.to_string()
}

/// A Command client, activated, and the measured pose it activated at.
fn activated(d: &Daemon) -> (Client, [f64; 7]) {
    let mut c = d.connect(Role::Command);
    assert_eq!(c.control(Op::Activate).unwrap().mode, Mode::Holding);
    let q = c.poll().q;
    (c, q)
}

#[test]
fn check_prints_the_digest_and_opens_nothing() {
    let dir = scratch_dir();
    let socket = dir.join("s.sock");
    let (status, out, _) = run_to_exit(serve_command(&socket, &["--check"]));
    assert_eq!(status.code(), Some(0));
    assert!(out.starts_with("SERVE_CHECK_OK {\"digest\": \""), "{out}");
    assert!(!socket.exists());
    let checked = digest_of(&out);
    let d = Daemon::start_in(dir, &[]);
    let mut m = d.connect(Role::Monitor);
    assert_eq!(format!("{:016x}", m.poll().digest), checked);
}

#[test]
fn usage_errors_exit_2() {
    let dir = scratch_dir();
    let mut c = serve_command(&dir.join("s.sock"), &[]);
    c.arg("--sim");
    let (status, _, _) = run_to_exit(c);
    assert_eq!(status.code(), Some(2));
    let mut c = std::process::Command::new(common::BIN);
    c.args([
        "serve",
        "--interface",
        "sim",
        "--dynamics",
        common::DYNAMICS,
    ])
    .env_remove("XDG_RUNTIME_DIR");
    let (status, _, err) = run_to_exit(c);
    assert_eq!(status.code(), Some(2));
    assert!(
        err.contains("XDG_RUNTIME_DIR") && err.contains("--socket"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_command_moves_the_target_and_the_state_follows() {
    let d = Daemon::start(&[]);
    let (mut c, mut q) = activated(&d);
    let mut last = c.poll();
    for _ in 0..100 {
        q[0] += 0.004;
        last = c.command(q);
        sleep_ms(10);
    }
    assert_eq!(last.mode, Mode::Following);
    assert_eq!(last.refused, 0);
    sleep_ms(300);
    let s = c.poll();
    assert!((s.q[0] - q[0]).abs() < 0.01, "{} vs {}", s.q[0], q[0]);
}

#[test]
fn a_jump_is_refused_and_held_until_reactivated() {
    let d = Daemon::start(&[]);
    let (mut c, q0) = activated(&d);
    let mut jump = q0;
    jump[0] += 0.2;
    let s = c.command(jump);
    assert_eq!((s.mode, s.refused), (Mode::Refused, 1));
    let mut small = q0;
    small[0] += 0.004;
    assert_eq!(c.command(small).mode, Mode::Refused);
    sleep_ms(100);
    let s = c.poll();
    assert!(
        (s.q[0] - q0[0]).abs() < 1e-3,
        "{} moved from {}",
        s.q[0],
        q0[0]
    );
    assert_eq!(s.refused, 1);
    // Activate alone does not clear a refusal: Deactivate first.
    assert_eq!(c.control(Op::Activate).unwrap().mode, Mode::Refused);
    assert_eq!(c.control(Op::Deactivate).unwrap().mode, Mode::Holding);
    assert_eq!(c.control(Op::Activate).unwrap().mode, Mode::Holding);
    assert_eq!(c.command(small).mode, Mode::Following);
}

#[test]
fn a_nan_command_is_refused() {
    let d = Daemon::start(&[]);
    let (mut c, mut q) = activated(&d);
    q[3] = f64::NAN;
    let s = c.command(q);
    assert_eq!((s.mode, s.refused), (Mode::Refused, 1));
}

#[test]
fn a_command_before_activate_is_ignored() {
    let d = Daemon::start(&[]);
    let mut c = d.connect(Role::Command);
    let mut q = c.poll().q;
    let q0 = q[0];
    q[0] += 0.04;
    assert_eq!(c.command(q).mode, Mode::Holding);
    sleep_ms(100);
    assert!((c.poll().q[0] - q0).abs() < 1e-3);
}

#[test]
fn the_stream_timeout_reports_holding() {
    let d = Daemon::start(&[]);
    let (mut c, mut q) = activated(&d);
    q[0] += 0.004;
    assert_eq!(c.command(q).mode, Mode::Following);
    sleep_ms(200);
    assert_eq!(c.poll().mode, Mode::Holding);
}

#[test]
fn a_second_command_client_is_closed() {
    let d = Daemon::start(&[]);
    let mut first = d.connect(Role::Command);
    first.poll();
    let mut second = d.connect(Role::Command);
    assert!(second.control(Op::Poll).is_err());
    first.poll();
}

#[test]
fn a_monitor_can_poll_while_a_client_commands() {
    let d = Daemon::start(&[]);
    let (mut c, mut q) = activated(&d);
    q[0] += 0.004;
    let s = c.command(q);
    let mut m = d.connect(Role::Monitor);
    let seen = m.poll();
    assert_eq!(seen.mode, Mode::Following);
    assert_eq!(seen.digest, s.digest);
    // A Monitor only polls: anything else closes it and leaves the client alone.
    assert!(m.control(Op::Activate).is_err());
    q[0] += 0.004;
    assert_eq!(c.command(q).mode, Mode::Following);
}

#[test]
fn the_gripper_target_stays_within_the_squeeze_window() {
    let grip = GripMap::provisional(&ArmConfig::maker_arm_v1());
    // GRIP_TAU_MAX_NM / kp of the gripper, in gripper_left_joint metres.
    let window = 0.5 / 20.0 * grip.joint_m_per_rad();
    let d = Daemon::start(&[]);
    let (mut c, mut q) = activated(&d);
    q[6] = 0.0;
    for _ in 0..3 {
        let before = c.poll().q[6];
        assert!(before > 10.0 * window, "the jaw starts open: {before}");
        c.command(q);
        sleep_ms(60);
        let moved = before - c.poll().q[6];
        assert!(
            (moved - window).abs() < 0.2 * window,
            "moved {moved} m, window {window} m"
        );
    }
}

#[test]
fn on_sim_the_daemon_exits_with_its_client_unless_persist() {
    let mut d = Daemon::start(&[]);
    let mut m = d.connect(Role::Monitor);
    m.poll();
    drop(m);
    sleep_ms(200);
    assert!(
        d.exit_within(Duration::ZERO).is_none(),
        "a Monitor ends nothing"
    );
    let mut c = d.connect(Role::Command);
    c.poll();
    drop(c);
    let status = d.exit_within(Duration::from_secs(2)).expect("exits");
    assert_eq!(status.code(), Some(0));
    assert!(d
        .line("SERVE_EXIT client gone", Duration::from_secs(1))
        .is_some());
    assert!(!d.socket.exists());

    let mut d = Daemon::start(&["--persist"]);
    let mut c = d.connect(Role::Command);
    c.poll();
    drop(c);
    sleep_ms(500);
    assert!(d.exit_within(Duration::ZERO).is_none());
    d.connect(Role::Command).poll();
}

#[test]
fn sigint_holds_and_keeps_a_persist_daemon_running() {
    let mut d = Daemon::start(&["--persist"]);
    let (mut c, q0) = activated(&d);
    d.signal("-INT");
    assert!(d.line("holding.", Duration::from_secs(2)).is_some());
    assert!(c.control(Op::Poll).is_err(), "the signal closes the client");
    let s = d.connect(Role::Monitor).poll();
    assert_eq!(s.mode, Mode::Holding);
    for (j, (a, b)) in s.q.iter().zip(&q0).enumerate() {
        assert!((a - b).abs() < 1e-3, "joint {j}");
    }
    d.signal("-TERM");
    sleep_ms(200);
    assert!(d.exit_within(Duration::ZERO).is_none());
}

#[test]
fn a_stale_socket_file_is_replaced() {
    let dir = scratch_dir();
    std::fs::write(dir.join("s.sock"), b"stale").unwrap();
    let d = Daemon::start_in(dir, &[]);
    d.connect(Role::Monitor).poll();
}

#[test]
fn a_second_daemon_on_the_same_socket_exits_1() {
    let d = Daemon::start(&[]);
    let (status, _, err) = run_to_exit(serve_command(&d.socket, &[]));
    assert_eq!(status.code(), Some(1));
    assert!(err.contains("a daemon already runs on"), "{err}");
    // The probe was no command client, so the first daemon still runs.
    d.connect(Role::Monitor).poll();
}

#[test]
fn a_second_daemon_on_a_held_bus_exits_1_before_it_opens_the_bus() {
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener};
    // No vcan device of this name exists: a daemon that opened the bus first would fail with
    // "open <iface>" instead of naming the lock.
    let iface = format!("vcan{}", std::process::id());
    let name = format!("maker-arm-bus-{iface}");
    let addr = SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
    let _held = UnixListener::bind_addr(&addr).unwrap();
    let dir = scratch_dir();
    let mut c = std::process::Command::new(common::BIN);
    c.args(["serve", "--interface", &iface, "--socket"])
        .arg(dir.join("s.sock"))
        .args(["--dynamics", common::DYNAMICS]);
    let (status, _, err) = run_to_exit(c);
    assert_eq!(status.code(), Some(1));
    assert!(err.contains(&name) && !err.contains("open "), "{err}");
    let hint = format!("maker-arm-cli status --interface {iface}");
    assert!(err.contains(&hint) && err.contains("ss -xlp"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn doctor_on_a_held_bus_exits_before_it_opens_the_bus() {
    // As above: no such device, so an opened bus would fail with "open <iface>". Not the name
    // above: both tests run in this process at once.
    let iface = format!("vcand{}", std::process::id());
    let _held = maker_arm_transport::bus_lock(&iface).unwrap();
    let mut c = std::process::Command::new(common::BIN);
    c.args(["--can", &iface, "doctor"]);
    let (status, out, err) = run_to_exit(c);
    assert_ne!(status.code(), Some(0));
    assert!(out.is_empty(), "{out}");
    assert!(
        err.contains(&format!("maker-arm-bus-{iface}")) && !err.contains("open "),
        "{err}"
    );
}

/// Runs `maker-arm-cli <sub>` against `socket` with `stdin` typed in.
fn client_cli(sub: &str, socket: &std::path::Path, stdin: &str) -> (Option<i32>, String, String) {
    use std::io::Write;
    let mut c = std::process::Command::new(common::BIN);
    c.args([sub, "--interface", "sim", "--socket"])
        .arg(socket)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = c.spawn().unwrap();
    let mut pipe = child.stdin.take().unwrap();
    pipe.write_all(stdin.as_bytes()).unwrap();
    // Keep the pipe open: EOF would make confirm_release retry, which is not under test.
    let out = child.wait_with_output().unwrap();
    drop(pipe);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn release_needs_the_typed_word_and_then_disables_and_exits() {
    let mut d = Daemon::start(&["--persist"]);
    let (mut c, _) = activated(&d);
    let (code, out, err) = client_cli("release", &d.socket, "no\nRELEASE\n");
    assert_eq!(code, Some(0), "{out}{err}");
    assert!(out.contains("torque remains enabled"), "{out}");
    assert!(out.contains("torque released."), "{out}");
    let status = d.exit_within(Duration::from_secs(2)).expect("exits");
    assert_eq!(status.code(), Some(0));
    assert!(d
        .line("SERVE_EXIT released", Duration::from_secs(1))
        .is_some());
    assert!(!d.socket.exists());
    assert!(c.control(Op::Poll).is_err(), "the Command client reads EOF");
}

#[test]
fn release_without_the_word_leaves_the_arm_holding() {
    // Without the typed word, nothing is sent: stdin ends after a wrong line, the process
    // is killed by the harness timeout, and the daemon still holds.
    let d = Daemon::start(&["--persist"]);
    let mut c = std::process::Command::new(common::BIN);
    c.args(["release", "--interface", "sim", "--socket"])
        .arg(&d.socket)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let mut child = c.spawn().unwrap();
    {
        use std::io::Write;
        let mut p = child.stdin.take().unwrap();
        p.write_all(b"no\n").unwrap();
        sleep_ms(300);
        child.kill().unwrap();
    }
    let _ = child.wait();
    assert_eq!(d.connect(Role::Monitor).poll().mode, Mode::Holding);
}

#[test]
fn release_without_a_daemon_names_the_socket() {
    let dir = scratch_dir();
    let socket = dir.join("none.sock");
    let (code, _, err) = client_cli("release", &socket, "RELEASE\n");
    assert_eq!(code, Some(1));
    assert!(err.contains(&socket.display().to_string()), "{err}");
    let (code, _, err) = client_cli("status", &socket, "");
    assert_eq!(code, Some(1));
    assert!(err.contains(&socket.display().to_string()), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn status_prints_one_status_line() {
    let d = Daemon::start(&[]);
    let (code, out, err) = client_cli("status", &d.socket, "");
    assert_eq!(code, Some(0), "{err}");
    let line = out.trim_end();
    assert_eq!(line.lines().count(), 1, "{out}");
    let json = line.strip_prefix("STATUS ").expect("STATUS prefix");
    let v: serde_json::Value = serde_json::from_str(json).unwrap();
    assert_eq!(v["mode"], "holding");
    assert_eq!(v["q"].as_array().unwrap().len(), 7);
    assert_eq!(
        (v["clamped"].as_u64(), v["refused"].as_u64()),
        (Some(0), Some(0))
    );
    assert_eq!(v["fault"].as_u64(), Some(0));
    assert!(v["t"].as_f64().is_some());
    assert_eq!(v["digest"].as_str().unwrap().len(), 16);
    // A status poll is a Monitor: the daemon still runs.
    d.connect(Role::Monitor).poll();
}
