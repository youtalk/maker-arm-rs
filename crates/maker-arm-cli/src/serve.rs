//! `maker-arm-cli serve`: a resident daemon that owns the bus and the `Tracking` loop and
//! serves one command client over a Unix socket. When its client stops, on a signal and on a
//! fault, it holds the arm. Once it is ready, on a real interface only an explicit release
//! disables the motors, and the daemon stays up, released, until an explicit enable; on a
//! virtual one a release ends the daemon, and so does its client unless `--persist`.

use maker_arm::kinematics::ARM_DOF;
use maker_arm::tracking::{Push, Row, TrackingController, TrackingShared, ROW};
use maker_arm::{
    clamp_command, ArmConfig, ArmState, FaultReason, JointCommand, LoopRt, RunningArm, Session,
    SessionError, SimArm,
};
use maker_arm_cli::files::{self, GripMap, Loaded, ServeOptions};
use maker_arm_cli::wire::{self, ClientFrame, Mode, Op, Role, StateFrame, WireError};
use maker_arm_transport::CanBackend;
use std::io::Write;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

/// Each command is due this far ahead on the loop clock.
const LEAD_S: f64 = 0.02; // INITIAL: V2 checks the tracking lag
/// A Following client silent this long is reported as Holding.
const STREAM_TIMEOUT_S: f64 = 0.1; // INITIAL: spec section 6.1
/// An arm joint further than this from the last accepted target is refused.
const JUMP_MAX_RAD: f64 = 0.05; // INITIAL: 5 rad/s at 100 Hz, the clamp's max_velocity
/// The gripper target leads the measured angle by at most this spring torque, closing only.
/// The vendor SDK (`maker_arm/gripper.py`) caps the closing side at `max(1, 2 x hold)` Nm,
/// with a default hold of 0.5 Nm. Opening is not capped: a window there held the jaw shut.
const GRIP_TAU_MAX_NM: f64 = 1.0;
/// As the lab's rig runs the loop.
const RT: LoopRt = LoopRt {
    priority: 80,
    cpu: 5,
};
/// The accept loop's period: how late a signal, a disconnect or a fault is acted on.
const TICK: Duration = Duration::from_millis(10);
/// The first tick must publish a pose and start the tracking clock within this: 200 ticks.
const FIRST_TICK: Duration = Duration::from_secs(1);
/// Released, the motors are read this often: 5 Hz, as `scan --watch`.
const PARKED_READ: Duration = Duration::from_millis(200);

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
    /// The tracking half of the loop that runs: pushes in, the clock out.
    shared: Arc<TrackingShared>,
    /// Released: the last read of the motors, with torque off.
    parked: Option<ArmState>,
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
        self.mode != Mode::Released
            && self.arm.as_ref().is_none_or(|a| {
                a.loop_finished() || a.snapshot().is_some_and(|s| s.fault.is_some())
            })
    }

    /// Commands are ignored until the next Activate. The plan holds its last row.
    fn hold(&mut self) {
        self.active = false;
        if self.mode != Mode::Fault && self.mode != Mode::Released {
            self.mode = Mode::Holding;
        }
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
}

/// Builds the tracking controller for each enable.
type Tracking = Box<dyn Fn() -> Result<TrackingController, String> + Send + Sync>;

struct Daemon {
    /// The limits the loop clamps commands into; Activate clamps the measured pose the same way.
    config: ArmConfig,
    state: Mutex<State>,
    /// The session while released. Locked before `state`, and held across a release and an
    /// enable, so a release waits for an enable in flight.
    session: Mutex<Option<Session>>,
    tracking: Tracking,
    rt: Option<LoopRt>,
    grip: GripMap,
    /// `GRIP_TAU_MAX_NM / kp` of the gripper (rad), the closing lead limit.
    grip_window: f64,
    digest: u64,
    /// A virtual interface without `--persist`.
    exit_with_client: bool,
    /// A real interface: a release keeps the daemon up, released.
    real: bool,
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
        FaultReason::FaultReplyTimeout { .. } => 5,
        FaultReason::Warning { .. } => 6,
        FaultReason::MotorRestarted { .. } => 7,
        FaultReason::Transport { .. } => 8,
    }
}

impl Daemon {
    /// Poison-tolerant: a panicked connection thread must not stop the others.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Poison-tolerant like `lock`.
    fn lock_session(&self) -> MutexGuard<'_, Option<Session>> {
        self.session.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn take_slot(&self, stream: &UnixStream) -> Option<CommandSlot<'_>> {
        let mut st = self.lock();
        if st.command.is_some() {
            return None;
        }
        st.command = Some(stream.try_clone().ok()?);
        Some(CommandSlot(self))
    }

    fn activate(&self, st: &mut State) {
        if st.mode == Mode::Fault || st.mode == Mode::Refused {
            return;
        }
        let Some(snap) = st.arm.as_ref().and_then(RunningArm::snapshot) else {
            return;
        };
        // The loop pulls a joint that rests outside its limits to the clamped pose. Take that
        // pose as the jump reference, or the client's first command after the pull-in is refused.
        let pose: Vec<JointCommand> = snap
            .arm
            .motors
            .iter()
            .map(|m| JointCommand {
                pos: m.position,
                vel: 0.0,
                kp: 0.0,
                kd: 0.0,
                tau: 0.0,
            })
            .collect();
        let Ok((clamped, _)) = clamp_command(&pose, &self.config) else {
            return;
        };
        let mut row = [0.0; ROW];
        for (r, c) in row.iter_mut().zip(&clamped) {
            *r = c.pos;
        }
        if st.push(row) {
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
        row[ARM_DOF] = grip_target(self.grip.to_angle(q[ARM_DOF]), measured, self.grip_window);
        if st.push(row) {
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
            t: st.shared.clock().map_or(0.0, |c| c.0),
            q: [0.0; ROW],
            dq: [0.0; ROW],
            effort: [0.0; ROW],
            temp: [0.0; ROW],
            fault: 0,
            clamped: st.clamped,
            refused: st.refused,
        };
        let (arm, fault) = if let Some(snap) = st.arm.as_ref().and_then(RunningArm::snapshot) {
            (snap.arm, snap.fault.as_ref().map_or(0, fault_code))
        } else if let Some(parked) = &st.parked {
            // The same clock as the loop's: both count from the session's connect.
            s.t = parked.t;
            (parked.clone(), 0)
        } else {
            return s;
        };
        for (j, m) in arm.motors.iter().enumerate().take(ROW) {
            s.q[j] = m.position;
            s.dq[j] = m.velocity;
            s.effort[j] = m.torque;
            s.temp[j] = m.temperature;
        }
        s.q[ARM_DOF] = self.grip.to_joint_m(s.q[ARM_DOF]);
        s.dq[ARM_DOF] *= self.grip.joint_m_per_rad();
        s.fault = fault;
        s
    }

    /// The one way to drop torque on a real interface. The reply goes out only after the
    /// motors are disabled. On a real interface the daemon stays up, released, and returns the
    /// reply. Else, and after a failed disable, it exits holding the lock, so no other client
    /// sees the arm gone.
    fn release(&self, stream: &mut UnixStream, seq: u32) -> StateFrame {
        // An enable in flight finishes first.
        let mut parked = self.lock_session();
        let mut st = self.lock();
        let mut frame = self.state_frame(&st, seq);
        frame.mode = Mode::Released;
        // No arm: released already.
        let Some(arm) = st.arm.take() else {
            return frame;
        };
        let (session, result) = arm.stop_and_disable();
        let failed = match result {
            Ok(()) => None,
            // Every motor is confirmed off. Say why the loop had stopped.
            Err(SessionError::LoopStopped {
                error,
                release: None,
            }) => {
                log(&format!("the control loop had stopped: {error}"));
                None
            }
            Err(e) => Some(e),
        };
        if failed.is_none() && self.real {
            st.parked = Some(session.arm_state());
            *parked = Some(session);
            st.mode = Mode::Released;
            st.active = false;
            log("released: torque is off. `maker-arm-cli enable` enables the arm again.");
            return frame;
        }
        let Some(e) = failed else {
            self.exit(stream, &frame, 0, "released");
        };
        log(&format!(
            "release failed: disable: {e}. The motors may keep their last command. Support \
             the arm."
        ));
        // Never claim a release that did not happen.
        frame.mode = Mode::Fault;
        self.exit(stream, &frame, 1, "release failed");
    }

    /// Sends `frame`, removes the socket and exits with `code`. The caller holds the state
    /// lock, so no other client sees the arm gone.
    fn exit(&self, stream: &mut UnixStream, frame: &StateFrame, code: i32, why: &str) -> ! {
        let _ = stream.write_all(&wire::encode_state(frame));
        let _ = std::fs::remove_file(&self.socket);
        say(&format!("SERVE_EXIT {why}"));
        std::process::exit(code);
    }

    /// Enables a released arm at the measured pose behind every gate of the first enable, and
    /// starts a new loop: mode Holding, and the client must Activate again. In any other mode
    /// nothing is sent. A failed gate sends nothing more and stays released. A motor that got
    /// an enable frame and does not confirm its disable ends the daemon, as a failed release
    /// does.
    fn enable(&self, stream: &mut UnixStream, seq: u32) -> StateFrame {
        let mut parked = self.lock_session();
        let session = {
            let st = self.lock();
            if st.mode == Mode::Released {
                parked.take()
            } else {
                None
            }
        };
        // Off the state lock: the gates, the enable and the first tick take longer than a
        // client's poll may wait. Meanwhile the replies show the last read, released.
        let started = session.map(|session| match (self.tracking)() {
            Ok(ctrl) => start(session, ctrl, self.rt, false),
            Err(why) => Err(Box::new(NotStarted {
                session,
                why,
                torque: false,
            })),
        });
        let mut st = self.lock();
        match started {
            Some(Ok((arm, shared))) => {
                st.arm = Some(arm);
                st.shared = shared;
                st.parked = None;
                st.mode = Mode::Holding;
                log("enabled at the measured pose; holding until a client Activates.");
            }
            Some(Err(e)) => {
                let NotStarted {
                    session,
                    why,
                    torque,
                } = *e;
                if torque {
                    log(&format!(
                        "enable refused: {why}. The motors may keep their last command. \
                         Support the arm."
                    ));
                    let mut frame = self.state_frame(&st, seq);
                    frame.mode = Mode::Fault;
                    self.exit(stream, &frame, 1, "enable failed");
                }
                log(&format!("enable refused: {why}. Torque stays off."));
                st.parked = Some(session.arm_state());
                *parked = Some(session);
            }
            None => log(&format!(
                "enable refused: the daemon is {}, not released",
                mode_name(st.mode)
            )),
        }
        self.state_frame(&st, seq)
    }

    /// Released: one read of every motor, with torque off. Skipped while a release or an
    /// enable holds the session.
    fn read_parked(&self) -> Result<(), String> {
        let mut parked = match self.session.try_lock() {
            Ok(p) => p,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        let Some(session) = parked.as_mut() else {
            return Ok(());
        };
        let read = session.read_motors().map_err(|e| e.to_string());
        self.lock().parked = Some(session.arm_state());
        read
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
            let state = match (role, f) {
                (
                    Role::Release,
                    ClientFrame::Control {
                        seq,
                        op: Op::Release,
                    },
                ) => self.release(&mut stream, seq),
                (
                    Role::Release,
                    ClientFrame::Control {
                        seq,
                        op: Op::Enable,
                    },
                ) => self.enable(&mut stream, seq),
                _ => match self.handle(role, f) {
                    Some(state) => state,
                    None => {
                        log(&format!("closed a {role:?} client that sent {f:?}"));
                        return;
                    }
                },
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

/// Why a loop did not start, with the session back.
struct NotStarted {
    session: Session,
    why: String,
    /// A disable did not confirm every motor off: one may still hold torque.
    torque: bool,
}

/// A started loop and its tracking half.
type Started = Result<(RunningArm, Arc<TrackingShared>), Box<NotStarted>>;

/// Enables `session` behind every enable gate, starts the loop on `ctrl` and waits for its
/// first tick. The first enable and every later `Enable` take this one path. With `sweep`,
/// a failed enable also disables every motor: at the start, a dead daemon may have left one
/// on.
fn start(
    mut session: Session,
    ctrl: TrackingController,
    rt: Option<LoopRt>,
    sweep: bool,
) -> Started {
    let why = match session.enable() {
        // Released, every motor confirmed off at the release. After any frame that can enable
        // a motor, `Session::enable` disables every motor itself, and only its `NotDisabled`
        // names one that may hold torque. A failed gate sent no such frame, and a disable
        // here would only fail on a silent motor.
        Err(e) if !sweep => {
            let torque = matches!(e, SessionError::NotDisabled { .. });
            let why = format!("enable: {e}");
            return Err(Box::new(NotStarted {
                session,
                why,
                torque,
            }));
        }
        Err(e) => format!("enable: {e}"),
        Ok(()) => {
            let shared = ctrl.shared();
            match session.start(Box::new(ctrl), rt) {
                Ok(arm) => match first_tick(&arm, &shared, &SIGNALED, FIRST_TICK) {
                    Ok(()) => return Ok((arm, shared)),
                    Err(e) => {
                        session = arm.stop_and_disable().0;
                        e
                    }
                },
                Err(e) => {
                    session = e.session;
                    format!("--rt: {}", e.error)
                }
            }
        }
    };
    // Leave torque off on every motor, also on one that a dead daemon left on.
    let (why, torque) = match session.disable() {
        Ok(()) => (why, false),
        Err(d) => (format!("{why}; disable: {d}"), true),
    };
    Err(Box::new(NotStarted {
        session,
        why,
        torque,
    }))
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
    say(&status_line(&ask(&mut s, Op::Poll)?));
    Ok(())
}

/// The STATUS token line, with the hottest motor and its temperature.
fn status_line(f: &StateFrame) -> String {
    let hot = (0..ROW)
        .max_by(|&a, &b| f.temp[a].total_cmp(&f.temp[b]))
        .unwrap_or(0);
    format!(
        "STATUS {{\"mode\": \"{}\", \"t\": {}, \"q\": {}, \"clamped\": {}, \"refused\": {}, \
         \"fault\": {}, \"temp_max_c\": {}, \"temp_max_motor\": {}, \"digest\": \"{:016x}\"}}",
        mode_name(f.mode),
        f.t,
        serde_json::json!(f.q),
        f.clamped,
        f.refused,
        f.fault,
        serde_json::json!(f.temp[hot]),
        ArmConfig::maker_arm_v1().joints[hot].motor_id,
        f.digest
    )
}

/// `maker-arm-cli release`: sends Op Release only after the typed word, or at once with
/// `yes`. Until then the daemon holds, also if this process dies.
pub fn release(interface: &str, socket: Option<&Path>, yes: bool) -> Result<(), String> {
    let mut s = connect(&socket_path(interface, socket), Role::Release)?;
    let f = ask(&mut s, Op::Poll)?;
    say(&format!("daemon mode: {}", mode_name(f.mode)));
    if !yes {
        maker_arm_cli::confirm_release(&mut std::io::stdin().lock(), &mut std::io::stdout());
    }
    let f = ask(&mut s, Op::Release)?;
    released(f.mode)?;
    say("torque released.");
    Ok(())
}

/// `maker-arm-cli enable`: sends Op Enable to a released daemon only after a typed ENABLE, or
/// at once with `yes`. Any other line, an end of input or a read error aborts: torque stays
/// off.
pub fn enable(interface: &str, socket: Option<&Path>, yes: bool) -> Result<(), String> {
    let mut s = connect(&socket_path(interface, socket), Role::Release)?;
    let f = ask(&mut s, Op::Poll)?;
    if f.mode != Mode::Released {
        return Err(format!(
            "the daemon is {}, not released: nothing to enable",
            mode_name(f.mode)
        ));
    }
    if !yes {
        say(
            "the arm is released. Support the arm, then type ENABLE and press ENTER to enable \
             torque at the measured pose.",
        );
        if !typed(&mut std::io::stdin().lock(), "enable") {
            return Err("aborted: torque stays off".into());
        }
    }
    let f = ask(&mut s, Op::Enable)?;
    if f.mode != Mode::Holding {
        return Err(format!(
            "enable failed: the daemon answered {}, not holding. The daemon log says why.",
            mode_name(f.mode)
        ));
    }
    say("torque enabled; the daemon holds the measured pose.");
    Ok(())
}

/// True only for one line that is `word`, trimmed, in any case. An end of input or a read
/// error is false.
fn typed(input: &mut dyn std::io::BufRead, word: &str) -> bool {
    let mut line = String::new();
    matches!(input.read_line(&mut line), Ok(n) if n > 0) && line.trim().eq_ignore_ascii_case(word)
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

/// Caps only the closing side of the gripper target at `measured + window`; opening passes.
/// Closing is the positive motor-angle direction on this arm, so the bound is an upper one.
/// `min`, not `clamp`: it cannot panic on a NaN.
fn grip_target(angle: f64, measured: f64, window: f64) -> f64 {
    angle.min(measured + window)
}

#[cfg(test)]
mod grip_tests {
    use super::grip_target;

    #[test]
    fn opening_passes_and_closing_is_capped() {
        let (kp, measured) = (20.0, 0.3);
        let window = 1.0 / kp;
        assert_eq!(grip_target(-2.0, measured, window), -2.0);
        assert_eq!(grip_target(5.0, measured, window), measured + window);
        assert_eq!(
            grip_target(measured + 0.01, measured, window),
            measured + 0.01
        );
    }
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
    // Before --check and the bus: neither needs the arm, and a file the daemon refuses must
    // fail the check too.
    let grip_window = GRIP_TAU_MAX_NM / config.joints[ARM_DOF].kp;
    let loop_config = config.clone();
    let tracking = move || {
        let mut c =
            TrackingController::new(dynamics.clone(), params, &loop_config, residual.clone())
                .map_err(|e| e.to_string())?;
        c.restart_for_loop().map_err(|e| e.to_string())?;
        Ok(c)
    };
    let ctrl = tracking()?;
    if check {
        say(&format!("SERVE_CHECK_OK {{\"digest\": \"{digest:016x}\"}}"));
        return Ok(());
    }
    let socket = socket_path(&o.interface, o.socket.as_deref());
    // A missing directory fails only at the bind, after the motors were enabled.
    let dir = socket
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !dir.is_dir() {
        log(&format!(
            "serve: the socket directory {} does not exist or is not a directory",
            dir.display()
        ));
        std::process::exit(2);
    }
    // Before the socket file and the bus: a refused daemon touches neither. Held to exit.
    // `sim` needs no lock, because each daemon has its own `SimArm`.
    let _bus = (o.interface != "sim")
        .then(|| maker_arm_transport::bus_lock(&o.interface))
        .transpose()
        .map_err(|e| e.to_string())?;
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
    // the motors keep their last command.
    if let Err(e) = ctrlc::set_handler(|| SIGNALED.store(true, Ordering::Relaxed)) {
        log(&format!(
            "WARNING: no SIGINT/SIGTERM handler ({e}); a signal kills the daemon and leaves \
             the motors on their last command"
        ));
    }
    let session = Session::connect(open(&o.interface, &config)?, config.clone())
        .map_err(|e| format!("connect {}: {e}", o.interface))?;
    if SIGNALED.load(Ordering::Relaxed) {
        return Err("interrupted before the motors were enabled".into());
    }
    let rt = o.rt.then_some(RT);
    let (arm, shared) = start(session, ctrl, rt, true).map_err(|e| {
        let support = if e.torque {
            ". The motors may keep their last command. Support the arm."
        } else {
            ""
        };
        format!("{}: {}{support}", o.interface, e.why)
    })?;
    // A failure from here to the bind comes before any client: dropping `arm` disables it,
    // as it was a moment before the start.
    let listener = UnixListener::bind(&socket)
        .and_then(|l| l.set_nonblocking(true).map(|()| l))
        .map_err(|e| format!("bind {}: {e}", socket.display()))?;

    let d = Arc::new(Daemon {
        state: Mutex::new(State {
            arm: Some(arm),
            shared,
            parked: None,
            mode: Mode::Holding,
            active: false,
            target: [0.0; ARM_DOF],
            last_command: Instant::now(),
            clamped: 0,
            refused: 0,
            command: None,
            exit: None,
        }),
        session: Mutex::new(None),
        tracking: Box::new(tracking),
        rt,
        config: config.clone(),
        grip,
        grip_window,
        digest,
        exit_with_client: files::is_virtual(&o.interface) && !o.persist,
        real: !files::is_virtual(&o.interface),
        socket: socket.clone(),
    });
    say(&format!(
        "SERVE_READY {{\"socket\": {}, \"digest\": \"{digest:016x}\"}}",
        serde_json::json!(socket.display().to_string())
    ));
    let (mut fault_logged, mut later_logged, mut stop_logged) = (false, 0, false);
    let (mut read_at, mut read_failed) = (Instant::now(), false);
    loop {
        if SIGNALED.swap(false, Ordering::Relaxed) {
            let mut st = d.lock();
            st.hold();
            if let Some(c) = &st.command {
                let _ = c.shutdown(std::net::Shutdown::Both);
            }
            let released = st.mode == Mode::Released;
            drop(st);
            say(if released {
                "released. Run `maker-arm-cli enable` to enable torque."
            } else {
                "holding. Run `maker-arm-cli release` to release torque."
            });
            if d.exit_with_client {
                finish(&d, &socket, "signal");
            }
        }
        if read_at.elapsed() >= PARKED_READ {
            read_at = Instant::now();
            let read = d.read_parked();
            // Each run of read errors, once.
            if let (Err(e), false) = (&read, read_failed) {
                log(&format!("released: read the motors: {e}"));
            }
            read_failed = read.is_err();
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
            let (first, later) = st
                .arm
                .as_ref()
                .and_then(RunningArm::snapshot)
                .map_or((None, vec![]), |s| (s.fault, s.later_faults));
            if let Some(f) = first.filter(|_| !fault_logged) {
                fault_logged = true;
                log(&format!("FAULT: {f}; holding the measured pose"));
            }
            // Each later fault during the hold, once.
            for f in later.iter().skip(later_logged) {
                log(&format!("FAULT (later, still holding): {f}"));
            }
            later_logged = later_logged.max(later.len());
            // Also after a transport hold: errors that last end the loop.
            if !stop_logged && st.arm.as_ref().is_none_or(RunningArm::loop_finished) {
                stop_logged = true;
                log("FAULT: the control loop stopped; the motors keep their last command");
            }
        } else {
            // A release ends the loop, and an enable starts a new one with its own faults.
            (fault_logged, later_logged, stop_logged) = (false, 0, false);
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

    fn tracking() -> Result<TrackingController, String> {
        let c = ArmConfig::maker_arm_v1();
        let v = serde_json::from_str(include_str!("../tests/fixtures/dynamics_test.json"));
        let (model, _) = files::dynamics_from_json(&v.unwrap()).unwrap();
        TrackingController::new(model, TrackingParams::default(), &c, None)
            .map_err(|e| e.to_string())
    }

    /// A tracking loop on a sim arm, with `hook` applied to the sim after enable.
    fn started(hook: impl FnOnce(&mut SimArm)) -> (RunningArm, Arc<TrackingShared>) {
        let c = ArmConfig::maker_arm_v1();
        let mut s = Session::connect(Box::new(SimArm::new(&c)), c.clone()).unwrap();
        s.enable().unwrap();
        hook(s.backend_as_sim().unwrap());
        let ctrl = tracking().unwrap();
        let shared = ctrl.shared();
        (s.start(Box::new(ctrl), None).unwrap(), shared)
    }

    /// A daemon on a sim arm past its first tick; `real` as on a real interface. The loop runs
    /// with the stock limits and never moves the sim. The daemon gets a J2 upper limit 0.3 rad
    /// below the measured pose, so Activate must pull that pose in. Also returns that limit.
    fn daemon(real: bool) -> (Daemon, f64) {
        let (arm, shared) = started(|_| {});
        first_tick(
            &arm,
            &shared,
            &AtomicBool::new(false),
            Duration::from_secs(1),
        )
        .unwrap();
        let measured = arm.snapshot().unwrap().arm.motors[1].position;
        let mut c = ArmConfig::maker_arm_v1();
        c.joints[1].q_hi = measured - 0.3 + c.limit_margin;
        let hi = measured - 0.3;
        assert!(hi > c.joints[1].q_lo + c.limit_margin);
        let d = Daemon {
            config: c,
            state: Mutex::new(State {
                arm: Some(arm),
                shared,
                parked: None,
                mode: Mode::Holding,
                active: false,
                target: [0.0; ARM_DOF],
                last_command: Instant::now(),
                clamped: 0,
                refused: 0,
                command: None,
                exit: None,
            }),
            session: Mutex::new(None),
            tracking: Box::new(tracking),
            rt: None,
            grip: GripMap::new(1.0, 0.0).unwrap(),
            grip_window: 0.1,
            digest: 0,
            exit_with_client: false,
            real,
            socket: PathBuf::new(),
        };
        (d, hi)
    }

    /// R27: Activate takes the measured pose clamped into the limits as the jump reference.
    fn activate_clamps(d: &Daemon, hi: f64) {
        let mut st = d.lock();
        d.activate(&mut st);
        assert!(st.active);
        assert!((st.target[1] - hi).abs() < 1e-9, "{}", st.target[1]);
        let mut q = [0.0; ROW];
        q[..ARM_DOF].copy_from_slice(&st.target);
        d.command(&mut st, &q);
        assert_eq!(st.mode, Mode::Following);
        q[1] = hi - 0.06;
        d.command(&mut st, &q);
        assert_eq!(st.mode, Mode::Refused);
    }

    /// Runs `f` on the released session's sim.
    fn on_sim<T>(d: &Daemon, f: impl FnOnce(&mut SimArm) -> T) -> T {
        let mut s = d.lock_session();
        f(s.as_mut().expect("released").backend_as_sim().unwrap())
    }

    fn release_now(d: &Daemon) {
        let (mut a, _b) = UnixStream::pair().unwrap();
        assert_eq!(d.release(&mut a, 1).mode, Mode::Released);
        assert_eq!(d.lock().mode, Mode::Released);
    }

    #[test]
    fn activate_takes_the_jump_reference_from_inside_the_limits() {
        let (d, hi) = daemon(false);
        activate_clamps(&d, hi);
    }

    #[test]
    fn on_a_real_interface_a_release_reads_the_motors_and_an_enable_holds_again() {
        let (d, hi) = daemon(true);
        release_now(&d);
        assert!(d.lock().arm.is_none());
        let before = d.state_frame(&d.lock(), 2);
        assert_eq!(before.mode, Mode::Released);
        // Torque off on every motor; move J1 by hand.
        on_sim(&d, |sim| {
            assert!((1..=7).all(|m| !sim.enabled(m) && sim.enables(m) == 1));
            sim.set_position(1, sim.position(1) + 0.2);
        });
        d.read_parked().unwrap();
        let after = d.state_frame(&d.lock(), 3);
        assert!(
            ((after.q[0] - before.q[0]).abs() - 0.2).abs() < 1e-3,
            "{} then {}",
            before.q[0],
            after.q[0]
        );
        assert_eq!(after.mode, Mode::Released);
        on_sim(&d, |sim| {
            assert!((1..=7).all(|m| !sim.enabled(m) && sim.enables(m) == 1));
        });
        // A client stays released: Activate, a command and Deactivate change nothing.
        let ctl = |op| ClientFrame::Control { seq: 4, op };
        for f in [
            ctl(Op::Activate),
            ClientFrame::Command { seq: 5, q: after.q },
            ctl(Op::Deactivate),
        ] {
            assert_eq!(d.handle(Role::Command, f).unwrap().mode, Mode::Released);
        }
        // A second release answers released and sends nothing.
        release_now(&d);
        let (mut a, _b) = UnixStream::pair().unwrap();
        let f = d.enable(&mut a, 6);
        assert_eq!(f.mode, Mode::Holding);
        assert!(d.lock_session().is_none());
        assert!(!d.lock().active, "the client must Activate");
        activate_clamps(&d, hi);
    }

    #[test]
    fn enable_in_any_other_mode_is_refused_and_sends_nothing() {
        let (d, _) = daemon(true);
        for m in [Mode::Holding, Mode::Following, Mode::Refused, Mode::Fault] {
            d.lock().mode = m;
            let (mut a, _b) = UnixStream::pair().unwrap();
            d.enable(&mut a, 1);
            let st = d.lock();
            assert_eq!(st.mode, m);
            assert!(st.arm.is_some());
        }
        let arm = d.lock().arm.take().unwrap();
        let (mut s, _) = arm.stop_and_disable();
        let sim = s.backend_as_sim().unwrap();
        assert!((1..=7).all(|m| sim.enables(m) == 1));
    }

    #[test]
    fn a_failed_enable_gate_stays_released() {
        let (d, _) = daemon(true);
        release_now(&d);
        let (mut a, _b) = UnixStream::pair().unwrap();
        let c = ArmConfig::maker_arm_v1();
        let j2 = &c.joints[1];
        let at = on_sim(&d, |sim| {
            let at = sim.position(2);
            // Past the 0.35 rad enable window.
            sim.set_position(2, j2.to_motor(j2.q_hi + 0.5));
            at
        });
        assert_eq!(d.enable(&mut a, 2).mode, Mode::Released);
        on_sim(&d, |sim| {
            assert!((1..=7).all(|m| !sim.enabled(m) && sim.enables(m) == 1));
            sim.set_position(2, at);
            // Ruling R31: a joint at its 70 C health limit, refused before any torque.
            sim.set_temperature(3, 70.0);
        });
        assert_eq!(d.enable(&mut a, 3).mode, Mode::Released);
        on_sim(&d, |sim| {
            assert!((1..=7).all(|m| !sim.enabled(m) && sim.enables(m) == 1));
            sim.set_temperature(3, 35.0);
        });
        assert_eq!(d.enable(&mut a, 4).mode, Mode::Holding);
    }

    #[test]
    fn a_silent_motor_at_enable_stays_released_and_sends_no_disable() {
        // As with the emergency stop pressed while parked: the fault query gets no reply.
        let (d, _) = daemon(true);
        release_now(&d);
        let disables = |sim: &mut SimArm| (1..=7).map(|m| sim.disables(m)).collect::<Vec<_>>();
        let before = on_sim(&d, |sim| {
            sim.set_muted(3, true);
            disables(sim)
        });
        let (mut a, _b) = UnixStream::pair().unwrap();
        // An exit here would end the test binary.
        assert_eq!(d.enable(&mut a, 2).mode, Mode::Released);
        on_sim(&d, |sim| {
            assert!((1..=7).all(|m| !sim.enabled(m) && sim.enables(m) == 1));
            assert_eq!(disables(sim), before, "no disable after a failed gate");
            sim.set_muted(3, false);
        });
        assert_eq!(d.enable(&mut a, 3).mode, Mode::Holding);
    }

    #[test]
    fn only_a_disable_that_fails_claims_torque() {
        let muted = || {
            let c = ArmConfig::maker_arm_v1();
            let mut s = Session::connect(Box::new(SimArm::new(&c)), c).unwrap();
            s.backend_as_sim().unwrap().set_muted(3, true);
            s
        };
        // Released: the gate failed before any enable frame. The log line says torque off.
        let fail = |sweep| match start(muted(), tracking().unwrap(), None, sweep) {
            Ok(_) => panic!("a silent motor enabled"),
            Err(e) => e,
        };
        let e = fail(false);
        assert!(!e.torque);
        assert_eq!(e.why, "enable: motor 3 did not reply -- check bus/power");
        // At the start a dead daemon may have left motor 3 on, and it does not confirm off.
        let e = fail(true);
        assert!(e.torque && e.why.contains("; disable: "), "{}", e.why);
    }

    #[test]
    fn the_first_enable_refuses_a_joint_at_its_health_limit() {
        // Ruling R31, through the start path of `run`: no torque, and the reason names it.
        let c = ArmConfig::maker_arm_v1();
        let mut s = Session::connect(Box::new(SimArm::new(&c)), c).unwrap();
        s.backend_as_sim().unwrap().set_temperature(3, 70.0);
        let Err(e) = start(s, tracking().unwrap(), None, true) else {
            panic!("a hot joint enabled");
        };
        assert!(!e.torque);
        assert_eq!(e.why, "enable: motor 3 is 70.0 degC, too hot to enable");
        let mut s = e.session;
        let sim = s.backend_as_sim().unwrap();
        assert!((1..=7).all(|m| sim.enables(m) == 0));
    }

    #[test]
    fn only_the_typed_word_enables() {
        let t = |text: &str| typed(&mut text.as_bytes(), "enable");
        assert!(t("ENABLE\n") && t(" enable \n") && t("ENABLE"));
        for no in ["", "\n", "no\n", "ENABLED\n", "RELEASE\n", "\nENABLE\n"] {
            assert!(!t(no), "{no:?}");
        }
        struct Broken;
        impl std::io::Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::Other.into())
            }
        }
        let mut r = std::io::BufReader::new(Broken);
        assert!(!typed(&mut r, "enable"));
    }

    #[test]
    fn status_names_the_hottest_motor() {
        let mut f = StateFrame {
            seq: 0,
            mode: Mode::Released,
            digest: 0,
            t: 0.0,
            q: [0.0; ROW],
            dq: [0.0; ROW],
            effort: [0.0; ROW],
            temp: [35.0; ROW],
            fault: 0,
            clamped: 0,
            refused: 0,
        };
        f.temp[2] = 61.5;
        let line = status_line(&f);
        let v: serde_json::Value =
            serde_json::from_str(line.strip_prefix("STATUS ").unwrap()).unwrap();
        assert_eq!(v["mode"], "released");
        assert_eq!(v["temp_max_c"], 61.5);
        assert_eq!(v["temp_max_motor"], 3);
    }

    #[test]
    fn the_new_fault_reasons_have_their_own_codes() {
        let age = 1.0;
        assert_eq!(
            fault_code(&FaultReason::FaultReplyTimeout { motor_id: 7, age }),
            6
        );
        assert_eq!(fault_code(&FaultReason::Warning { motor_id: 7 }), 7);
        // The existing codes are unchanged.
        assert_eq!(
            fault_code(&FaultReason::FeedbackTimeout { motor_id: 1, age }),
            1
        );
        let detail = String::new();
        assert_eq!(fault_code(&FaultReason::BadCommand { detail }), 5);
        let detail = String::new();
        assert_eq!(fault_code(&FaultReason::Transport { detail }), 9);
        assert_eq!(fault_code(&FaultReason::MotorRestarted { motor_id: 2 }), 8);
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

    /// No feedback after enable for longer than the feedback timeout, so the first tick's
    /// health check faults.
    fn stale(_: &mut SimArm) {
        std::thread::sleep(Duration::from_millis(300));
    }

    #[test]
    fn a_fault_at_the_first_tick_ends_the_wait_with_its_reason() {
        // The fault hold never calls the controller, so the tracking clock never starts.
        let (arm, shared) = started(stale);
        let quiet = AtomicBool::new(false);
        let t = Instant::now();
        let e = first_tick(&arm, &shared, &quiet, Duration::from_secs(5)).unwrap_err();
        assert!(e.contains("feedback timeout"), "{e}");
        assert!(t.elapsed() < Duration::from_secs(1));
        assert!(shared.clock().is_none());
    }

    #[test]
    fn a_signal_ends_the_first_tick_wait() {
        let (arm, shared) = started(stale);
        let e = first_tick(
            &arm,
            &shared,
            &AtomicBool::new(true),
            Duration::from_secs(5),
        );
        assert!(e.unwrap_err().contains("interrupted"));
    }
}
