//! `maker-arm-cli serve`: a resident daemon that owns the bus and the `Tracking` loop and
//! serves one command client over a Unix socket. When its client stops, on a signal and on a
//! fault, it holds the arm. On a real interface only an explicit release disables the
//! motors; on a virtual one the daemon ends with its client unless `--persist`.

use maker_arm::kinematics::ARM_DOF;
use maker_arm::tracking::{Push, Row, TrackingController, TrackingShared, ROW};
use maker_arm::{ArmConfig, FaultReason, LoopRt, RunningArm, Session, SimArm};
use maker_arm_cli::files::{self, GripMap, Loaded, ServeOptions};
use maker_arm_cli::wire::{self, ClientFrame, Mode, Op, Role, StateFrame, WireError};
use maker_arm_transport::CanBackend;
use std::io::{ErrorKind, Write};
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Each command is due this far ahead on the loop clock.
const LEAD_S: f64 = 0.02; // INITIAL: V2 checks the tracking lag
/// A Following client silent this long is reported as Holding.
const STREAM_TIMEOUT_S: f64 = 0.1; // INITIAL: spec section 6.1
/// An arm joint further than this from the last accepted target is refused.
const JUMP_MAX_RAD: f64 = 0.05; // INITIAL: 5 rad/s at 100 Hz, the clamp's max_velocity
/// The gripper target leads the measured angle by at most this spring torque.
const GRIP_TAU_MAX_NM: f64 = 0.5; // INITIAL: session 4 checks it with the gripper map
/// As the lab's rig runs the loop.
const RT: LoopRt = LoopRt {
    priority: 80,
    cpu: 5,
};
/// The accept loop's period: how late a signal, a disconnect or a fault is acted on.
const TICK: Duration = Duration::from_millis(10);
/// The first tick must publish a pose and start the tracking clock within this: 200 ticks.
const FIRST_TICK: Duration = Duration::from_secs(1);

/// Set by the SIGINT/SIGTERM handler, read by the accept loop.
static SIGNALED: AtomicBool = AtomicBool::new(false);

/// Best-effort output. `println!` panics on a closed stdout, and a panic in the accept loop
/// would unwind past the running arm.
fn say(line: &str) {
    let _ = writeln!(std::io::stdout(), "{line}");
}

fn log(line: &str) {
    let _ = writeln!(std::io::stderr(), "{line}");
}

struct State {
    arm: Option<RunningArm>,
    mode: Mode,
    /// Activate was sent and no Deactivate, disconnect or signal since.
    active: bool,
    /// The last accepted arm target (rad).
    target: [f64; ARM_DOF],
    last_command: Instant,
    clamped: u32,
    refused: u32,
    /// The command client's connection, kept to shut it down on a signal.
    command: Option<UnixStream>,
    /// Set when a virtual daemon without `--persist` must end.
    exit: Option<&'static str>,
}

impl State {
    fn faulted(&self) -> bool {
        self.arm
            .as_ref()
            .is_none_or(|a| a.loop_finished() || a.snapshot().is_some_and(|s| s.fault.is_some()))
    }

    /// Commands are ignored until the next Activate. The plan holds its last row.
    fn hold(&mut self) {
        self.active = false;
        if self.mode != Mode::Fault {
            self.mode = Mode::Holding;
        }
    }
}

struct Daemon {
    state: Mutex<State>,
    shared: Arc<TrackingShared>,
    grip: GripMap,
    /// `GRIP_TAU_MAX_NM / kp` of the gripper (rad).
    grip_window: f64,
    digest: u64,
    /// A virtual interface without `--persist`.
    exit_with_client: bool,
    socket: PathBuf,
}

/// Holds the command slot. Its drop, also on a panic, frees the slot and holds the arm.
struct CommandSlot<'a>(&'a Daemon);

impl Drop for CommandSlot<'_> {
    fn drop(&mut self) {
        let mut st = self.0.lock();
        st.command = None;
        st.hold();
        if self.0.exit_with_client {
            st.exit = Some("client gone");
        }
        log("the command client left; holding");
    }
}

fn fault_code(f: &FaultReason) -> u32 {
    1 + match f {
        FaultReason::FeedbackTimeout { .. } => 0,
        FaultReason::MotorFault { .. } => 1,
        FaultReason::ModeNotMotor { .. } => 2,
        FaultReason::OverTemperature { .. } => 3,
        FaultReason::BadCommand { .. } => 4,
    }
}

impl Daemon {
    /// Poison-tolerant: a panicked connection thread must not stop the others.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn take_slot(&self, stream: &UnixStream) -> Option<CommandSlot<'_>> {
        let mut st = self.lock();
        if st.command.is_some() {
            return None;
        }
        st.command = Some(stream.try_clone().ok()?);
        Some(CommandSlot(self))
    }

    /// Pushes `row` due `LEAD_S` ahead. False before the first tick.
    fn push(&self, row: Row) -> bool {
        let Some((now, _)) = self.shared.clock() else {
            return false;
        };
        match Push::new(now + LEAD_S, 1.0, vec![row]) {
            Ok(p) => {
                self.shared.push(p);
                true
            }
            Err(_) => false,
        }
    }

    fn activate(&self, st: &mut State) {
        if st.mode == Mode::Fault || st.mode == Mode::Refused {
            return;
        }
        let Some(snap) = st.arm.as_ref().and_then(RunningArm::snapshot) else {
            return;
        };
        let mut row = [0.0; ROW];
        for (r, m) in row.iter_mut().zip(&snap.arm.motors) {
            *r = m.position;
        }
        if self.push(row) {
            st.target.copy_from_slice(&row[..ARM_DOF]);
            st.active = true;
            st.mode = Mode::Holding;
        }
    }

    fn command(&self, st: &mut State, q: &[f64; ROW]) {
        if st.mode == Mode::Fault || st.mode == Mode::Refused || !st.active {
            return;
        }
        let jump = q[..ARM_DOF]
            .iter()
            .zip(&st.target)
            .any(|(a, b)| (a - b).abs() > JUMP_MAX_RAD);
        if jump || q.iter().any(|v| !v.is_finite()) {
            st.refused = st.refused.saturating_add(1);
            st.mode = Mode::Refused;
            log(&format!(
                "refused {q:?}: a non-finite value or a joint more than {JUMP_MAX_RAD} rad \
                 from the last target {:?}; holding until Deactivate, then Activate",
                st.target
            ));
            return;
        }
        let Some(snap) = st.arm.as_ref().and_then(RunningArm::snapshot) else {
            return;
        };
        let measured = snap.arm.motors[ARM_DOF].position;
        let mut row = [0.0; ROW];
        row[..ARM_DOF].copy_from_slice(&q[..ARM_DOF]);
        // max/min, not clamp: they cannot panic on a NaN.
        row[ARM_DOF] = self
            .grip
            .to_angle(q[ARM_DOF])
            .max(measured - self.grip_window)
            .min(measured + self.grip_window);
        if self.push(row) {
            st.target.copy_from_slice(&q[..ARM_DOF]);
            st.last_command = Instant::now();
            st.mode = Mode::Following;
        }
    }

    /// One State frame per client frame after the Hello. None closes the connection.
    fn handle(&self, role: Role, f: ClientFrame) -> Option<StateFrame> {
        let mut st = self.lock();
        if st.faulted() {
            st.mode = Mode::Fault;
        }
        let seq = match (role, f) {
            (_, ClientFrame::Control { seq, op: Op::Poll }) => seq,
            (
                Role::Command,
                ClientFrame::Control {
                    seq,
                    op: Op::Activate,
                },
            ) => {
                self.activate(&mut st);
                seq
            }
            (
                Role::Command,
                ClientFrame::Control {
                    seq,
                    op: Op::Deactivate,
                },
            ) => {
                st.hold();
                seq
            }
            (Role::Command, ClientFrame::Command { seq, q }) => {
                self.command(&mut st, &q);
                seq
            }
            _ => return None,
        };
        Some(self.state_frame(&st, seq))
    }

    fn state_frame(&self, st: &State, seq: u32) -> StateFrame {
        let stale = st.last_command.elapsed().as_secs_f64() > STREAM_TIMEOUT_S;
        let mut s = StateFrame {
            seq,
            mode: if st.mode == Mode::Following && stale {
                Mode::Holding
            } else {
                st.mode
            },
            digest: self.digest,
            t: self.shared.clock().map_or(0.0, |c| c.0),
            q: [0.0; ROW],
            dq: [0.0; ROW],
            effort: [0.0; ROW],
            temp: [0.0; ROW],
            fault: 0,
            clamped: st.clamped,
            refused: st.refused,
        };
        if let Some(snap) = st.arm.as_ref().and_then(RunningArm::snapshot) {
            for (j, m) in snap.arm.motors.iter().enumerate().take(ROW) {
                s.q[j] = m.position;
                s.dq[j] = m.velocity;
                s.effort[j] = m.torque;
                s.temp[j] = m.temperature;
            }
            s.q[ARM_DOF] = self.grip.to_joint_m(s.q[ARM_DOF]);
            s.dq[ARM_DOF] *= self.grip.joint_m_per_rad();
            s.fault = snap.fault.as_ref().map_or(0, fault_code);
        }
        s
    }

    /// The one way to drop torque on a real interface. Holds the lock to the exit, so no
    /// other client sees the arm gone. The reply goes out only after the motors are disabled.
    fn release(&self, stream: &mut UnixStream, seq: u32) -> ! {
        let mut st = self.lock();
        let mut frame = self.state_frame(&st, seq);
        frame.mode = Mode::Released;
        let failed = st.arm.take().and_then(|arm| arm.stop_and_disable().1.err());
        if failed.is_some() {
            // Never claim a release that did not happen.
            frame.mode = Mode::Fault;
        }
        let _ = stream.write_all(&wire::encode_state(&frame));
        let _ = std::fs::remove_file(&self.socket);
        if let Some(e) = failed {
            log(&format!(
                "release failed: disable: {e}. The motors may keep their last command until \
                 their CAN timeout. Support the arm."
            ));
            say("SERVE_EXIT release failed");
            std::process::exit(1);
        }
        say("SERVE_EXIT released");
        std::process::exit(0);
    }

    fn serve(&self, mut stream: UnixStream) {
        let role = match wire::read_client(&mut stream) {
            Ok(ClientFrame::Hello(role)) => role,
            Ok(f) => {
                log(&format!(
                    "closed a connection that began with {f:?}, not a Hello"
                ));
                return;
            }
            Err(e) => {
                log(&format!("closed a connection before its Hello: {e}"));
                return;
            }
        };
        let _slot = match role {
            Role::Command => match self.take_slot(&stream) {
                Some(slot) => Some(slot),
                None => {
                    log("another client is connected; closed the new command client");
                    return;
                }
            },
            Role::Monitor => None,
            Role::Release => None,
        };
        loop {
            let f = match wire::read_client(&mut stream) {
                Ok(f) => f,
                Err(WireError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return,
                Err(e) => {
                    log(&format!("closed a {role:?} client: {e}"));
                    return;
                }
            };
            if let (
                Role::Release,
                ClientFrame::Control {
                    seq,
                    op: Op::Release,
                },
            ) = (role, f)
            {
                self.release(&mut stream, seq);
            }
            let Some(state) = self.handle(role, f) else {
                log(&format!("closed a {role:?} client that sent {f:?}"));
                return;
            };
            if stream.write_all(&wire::encode_state(&state)).is_err() {
                return;
            }
        }
    }
}

fn open(interface: &str, config: &ArmConfig) -> Result<Box<dyn CanBackend>, String> {
    if interface == "sim" {
        return Ok(Box::new(SimArm::new(config)));
    }
    #[cfg(feature = "socketcan")]
    {
        let b = maker_arm_transport::SocketCanBackend::open(interface)
            .map_err(|e| format!("open {interface}: {e}"))?;
        Ok(Box::new(b))
    }
    #[cfg(not(feature = "socketcan"))]
    Err(format!("{interface}: built without the socketcan feature"))
}

/// `--socket`, else `$XDG_RUNTIME_DIR/maker-arm-<interface>.sock`. Shared by the daemon,
/// `release` and `status`.
pub fn socket_path(interface: &str, socket: Option<&Path>) -> PathBuf {
    if let Some(s) = socket {
        return s.to_path_buf();
    }
    match std::env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty()) {
        Some(d) => PathBuf::from(d).join(format!("maker-arm-{interface}.sock")),
        None => {
            log("serve: XDG_RUNTIME_DIR is not set; pass --socket PATH");
            std::process::exit(2);
        }
    }
}

/// Holds the abstract socket `maker-arm-bus-<interface>` while it lives, so a second daemon
/// never opens a bus that one drives: its `RunningArm` drop would disable the first one's
/// motors. The kernel frees the name when the process dies. `sim` needs no lock, because each
/// daemon has its own `SimArm`.
fn bus_lock(interface: &str) -> Result<Option<UnixListener>, String> {
    if interface == "sim" {
        return Ok(None);
    }
    let name = format!("maker-arm-bus-{interface}");
    SocketAddr::from_abstract_name(name.as_bytes())
        .and_then(|a| UnixListener::bind_addr(&a))
        .map(Some)
        .map_err(|e| match e.kind() {
            ErrorKind::AddrInUse => {
                format!("another daemon already drives {interface}: it holds the lock @{name}")
            }
            _ => format!("lock @{name} for {interface}: {e}"),
        })
}

/// Waits for the first tick's pose and tracking clock, which Activate needs. A signal, a
/// stopped loop, a fault or `timeout` ends the wait with an error. Only the controller starts
/// the clock, and a fault hold never calls it, so a first-tick fault would wait forever.
fn first_tick(
    arm: &RunningArm,
    shared: &TrackingShared,
    signaled: &AtomicBool,
    timeout: Duration,
) -> Result<(), String> {
    let end = Instant::now() + timeout;
    loop {
        if signaled.load(Ordering::Relaxed) {
            return Err("interrupted before the first tick".into());
        }
        if arm.loop_finished() {
            return Err("the control loop stopped before its first tick".into());
        }
        if let Some(snap) = arm.snapshot() {
            if let Some(f) = snap.fault {
                return Err(format!("fault at the first tick: {f}"));
            }
            if shared.clock().is_some() {
                return Ok(());
            }
        }
        if Instant::now() >= end {
            return Err(format!("no first tick within {timeout:?}"));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Ends a virtual daemon: stop the loop, disable, exit 0. Only `exit_with_client` reaches it.
fn finish(d: &Daemon, socket: &Path, why: &str) -> ! {
    let arm = d.lock().arm.take();
    let _ = std::fs::remove_file(socket);
    if let Some(arm) = arm {
        if let (_, Err(e)) = arm.stop_and_disable() {
            log(&format!("disable: {e}"));
        }
    }
    say(&format!("SERVE_EXIT {why}"));
    std::process::exit(0);
}

fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Holding => "holding",
        Mode::Following => "following",
        Mode::Refused => "refused",
        Mode::Fault => "fault",
        Mode::Released => "released",
    }
}

/// Connects as `role` and sends the Hello. Without a daemon the error names the socket.
fn connect(socket: &Path, role: Role) -> Result<UnixStream, String> {
    let mut s = UnixStream::connect(socket)
        .map_err(|e| format!("no daemon on {}: {e}", socket.display()))?;
    s.set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    s.write_all(&wire::encode_client(&ClientFrame::Hello(role)))
        .map_err(|e| e.to_string())?;
    Ok(s)
}

fn ask(s: &mut UnixStream, op: Op) -> Result<StateFrame, String> {
    let f = ClientFrame::Control { seq: 1, op };
    s.write_all(&wire::encode_client(&f))
        .map_err(|e| e.to_string())?;
    wire::read_state(s).map_err(|e| format!("the daemon's reply: {e}"))
}

/// `maker-arm-cli status`: one Monitor Poll as one token line.
pub fn status(interface: &str, socket: Option<&Path>) -> Result<(), String> {
    let mut s = connect(&socket_path(interface, socket), Role::Monitor)?;
    let f = ask(&mut s, Op::Poll)?;
    say(&format!(
        "STATUS {{\"mode\": \"{}\", \"t\": {}, \"q\": {}, \"clamped\": {}, \"refused\": {}, \
         \"fault\": {}, \"digest\": \"{:016x}\"}}",
        mode_name(f.mode),
        f.t,
        serde_json::json!(f.q),
        f.clamped,
        f.refused,
        f.fault,
        f.digest
    ));
    Ok(())
}

/// `maker-arm-cli release`: sends Op Release only after the typed word. Until then the daemon
/// holds, also if this process dies.
pub fn release(interface: &str, socket: Option<&Path>) -> Result<(), String> {
    let mut s = connect(&socket_path(interface, socket), Role::Release)?;
    let f = ask(&mut s, Op::Poll)?;
    say(&format!("daemon mode: {}", mode_name(f.mode)));
    maker_arm_cli::confirm_release(&mut std::io::stdin().lock(), &mut std::io::stdout());
    let f = ask(&mut s, Op::Release)?;
    released(f.mode)?;
    say("torque released.");
    Ok(())
}

/// Only `Released` is success: any other reply means the motors may still hold torque.
fn released(mode: Mode) -> Result<(), String> {
    if mode == Mode::Released {
        return Ok(());
    }
    Err(format!(
        "release failed: the daemon answered {}, not released. The motors may still hold \
         torque. Support the arm.",
        mode_name(mode)
    ))
}

pub fn run(o: &ServeOptions, check: bool) -> Result<(), String> {
    let Loaded {
        config,
        dynamics,
        params,
        residual,
        grip,
        digest,
    } = files::load(o)?;
    if check {
        say(&format!("SERVE_CHECK_OK {{\"digest\": \"{digest:016x}\"}}"));
        return Ok(());
    }
    let socket = socket_path(&o.interface, o.socket.as_deref());
    // Before the socket file and the bus: a refused daemon touches neither. Held to exit.
    let _bus = bus_lock(&o.interface)?;
    if UnixStream::connect(&socket).is_ok() {
        return Err(format!("a daemon already runs on {}", socket.display()));
    }
    match std::fs::remove_file(&socket) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(format!("remove the stale {}: {e}", socket.display()));
        }
        _ => {}
    }
    // Ctrl-C must hold, never release. Without the handler, a signal kills the process and
    // the motors' own CAN_TIMEOUT watchdog is the only backstop.
    if let Err(e) = ctrlc::set_handler(|| SIGNALED.store(true, Ordering::Relaxed)) {
        log(&format!(
            "WARNING: no SIGINT/SIGTERM handler ({e}); a signal kills the daemon and leaves \
             the motors to their CAN_TIMEOUT watchdog"
        ));
    }
    let grip_window = GRIP_TAU_MAX_NM / config.joints[ARM_DOF].kp;
    let mut ctrl =
        TrackingController::new(dynamics, params, &config, residual).map_err(|e| e.to_string())?;
    ctrl.restart_for_loop().map_err(|e| e.to_string())?;
    let shared = ctrl.shared();

    let mut session = Session::connect(open(&o.interface, &config)?, config.clone())
        .map_err(|e| format!("connect {}: {e}", o.interface))?;
    if SIGNALED.load(Ordering::Relaxed) {
        return Err("interrupted before the motors were enabled".into());
    }
    if let Err(e) = session.enable() {
        // Torque was off a moment ago; leave it off.
        let _ = session.disable();
        return Err(format!("enable {}: {e}", o.interface));
    }
    let arm = match session.start(Box::new(ctrl), o.rt.then_some(RT)) {
        Ok(arm) => arm,
        Err(e) => {
            let mut session = e.session;
            let _ = session.disable();
            return Err(format!("--rt: {}", e.error));
        }
    };
    // A failure from here to the bind comes before any client: dropping `arm` disables it,
    // as it was a moment before the start.
    first_tick(&arm, &shared, &SIGNALED, FIRST_TICK)?;
    let listener = UnixListener::bind(&socket)
        .and_then(|l| l.set_nonblocking(true).map(|()| l))
        .map_err(|e| format!("bind {}: {e}", socket.display()))?;

    let d = Arc::new(Daemon {
        state: Mutex::new(State {
            arm: Some(arm),
            mode: Mode::Holding,
            active: false,
            target: [0.0; ARM_DOF],
            last_command: Instant::now(),
            clamped: 0,
            refused: 0,
            command: None,
            exit: None,
        }),
        shared,
        grip,
        grip_window,
        digest,
        exit_with_client: files::is_virtual(&o.interface) && !o.persist,
        socket: socket.clone(),
    });
    say(&format!(
        "SERVE_READY {{\"socket\": {}, \"digest\": \"{digest:016x}\"}}",
        serde_json::json!(socket.display().to_string())
    ));
    let mut fault_logged = false;
    loop {
        if SIGNALED.swap(false, Ordering::Relaxed) {
            let mut st = d.lock();
            st.hold();
            if let Some(c) = &st.command {
                let _ = c.shutdown(std::net::Shutdown::Both);
            }
            drop(st);
            say("holding. Run `maker-arm-cli release` to release torque.");
            if d.exit_with_client {
                finish(&d, &socket, "signal");
            }
        }
        match listener.accept() {
            Ok((stream, _)) => {
                let d = Arc::clone(&d);
                let spawned = stream.set_nonblocking(false).and_then(|()| {
                    std::thread::Builder::new()
                        .name("serve-client".into())
                        .spawn(move || d.serve(stream))
                });
                if let Err(e) = spawned {
                    log(&format!("dropped a connection: {e}"));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(TICK),
            Err(e) => {
                log(&format!("accept: {e}"));
                std::thread::sleep(TICK);
            }
        }
        let mut st = d.lock();
        let clamped = st.arm.as_ref().map_or(0, |a| {
            let (records, _) = a.drain();
            records.iter().filter(|r| r.clamped).count()
        });
        st.clamped = st
            .clamped
            .saturating_add(u32::try_from(clamped).unwrap_or(u32::MAX));
        if st.faulted() {
            st.mode = Mode::Fault;
            if !fault_logged {
                fault_logged = true;
                let why = st.arm.as_ref().and_then(|a| a.snapshot()?.fault);
                log(&match why {
                    Some(f) => format!("FAULT: {f}; holding the measured pose"),
                    None => "FAULT: the control loop stopped; the motors' CAN_TIMEOUT \
                             watchdog takes over"
                        .into(),
                });
            }
        }
        if let Some(why) = st.exit {
            drop(st);
            finish(&d, &socket, why);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maker_arm::tracking::TrackingParams;

    /// A tracking loop on a sim arm, with `hook` applied to the sim before enable.
    fn started(hook: impl FnOnce(&mut SimArm)) -> (RunningArm, Arc<TrackingShared>) {
        let c = ArmConfig::maker_arm_v1();
        let mut s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).unwrap();
        hook(s.backend_as_sim().unwrap());
        s.enable().unwrap();
        let v = serde_json::from_str(include_str!("../tests/fixtures/dynamics_test.json"));
        let (model, _) = files::dynamics_from_json(&v.unwrap()).unwrap();
        let ctrl = TrackingController::new(model, TrackingParams::default(), &c, None).unwrap();
        let shared = ctrl.shared();
        (s.start(Box::new(ctrl), None).unwrap(), shared)
    }

    #[test]
    fn only_a_released_reply_counts_as_released() {
        assert!(released(Mode::Released).is_ok());
        for m in [Mode::Fault, Mode::Holding, Mode::Following, Mode::Refused] {
            let e = released(m).unwrap_err();
            assert!(
                e.contains("release failed") && e.contains("Support the arm"),
                "{e}"
            );
        }
    }

    #[test]
    fn the_first_tick_wait_returns_once_the_clock_runs() {
        let (arm, shared) = started(|_| {});
        let quiet = AtomicBool::new(false);
        first_tick(&arm, &shared, &quiet, Duration::from_secs(1)).unwrap();
        assert!(shared.clock().is_some());
    }

    #[test]
    fn a_fault_at_the_first_tick_ends_the_wait_with_its_reason() {
        // Enable checks no temperature. The first tick's health check does, and the fault hold
        // never calls the controller, so the tracking clock never starts.
        let (arm, shared) = started(|sim| sim.set_temperature(3, 80.0));
        let quiet = AtomicBool::new(false);
        let t = Instant::now();
        let e = first_tick(&arm, &shared, &quiet, Duration::from_secs(5)).unwrap_err();
        assert!(e.contains("over temperature"), "{e}");
        assert!(t.elapsed() < Duration::from_secs(1));
        assert!(shared.clock().is_none());
    }

    #[test]
    fn a_signal_ends_the_first_tick_wait() {
        let (arm, shared) = started(|sim| sim.set_temperature(3, 80.0));
        let e = first_tick(
            &arm,
            &shared,
            &AtomicBool::new(true),
            Duration::from_secs(5),
        );
        assert!(e.unwrap_err().contains("interrupted"));
    }

    #[test]
    fn the_bus_lock_is_exclusive_and_freed_on_drop() {
        let iface = format!("vcan-unit{}", std::process::id());
        let held = bus_lock(&iface).unwrap();
        assert!(held.is_some());
        let e = bus_lock(&iface).unwrap_err();
        assert!(e.contains(&iface), "{e}");
        drop(held);
        assert!(bus_lock(&iface).unwrap().is_some());
        assert!(bus_lock("sim").unwrap().is_none());
    }
}
