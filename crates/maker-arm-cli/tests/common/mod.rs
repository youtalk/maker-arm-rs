//! Starts `maker-arm-cli serve` on the sim interface and talks to it over its socket. Every
//! daemon is killed and its directory removed when its `Daemon` drops.

use maker_arm_cli::wire::{self, ClientFrame, Op, Role, StateFrame, WireError};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

pub const BIN: &str = env!("CARGO_BIN_EXE_maker-arm-cli");
pub const DYNAMICS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/dynamics_test.json"
);

static NEXT: AtomicUsize = AtomicUsize::new(0);

/// A fresh, short directory: a Unix socket path must stay under 108 bytes.
pub fn scratch_dir() -> PathBuf {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("mas-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The daemon's command line on `socket`, then `extra`.
pub fn serve_command(socket: &Path, extra: &[&str]) -> Command {
    let mut c = Command::new(BIN);
    c.args(["serve", "--interface", "sim", "--socket"])
        .arg(socket)
        .args(["--dynamics", DYNAMICS])
        .args(extra)
        .stdin(Stdio::null());
    c
}

/// Runs `c` to its exit (killed after 5 s) and returns its status, stdout and stderr.
pub fn run_to_exit(mut c: Command) -> (ExitStatus, String, String) {
    let mut child = c
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let end = Instant::now() + Duration::from_secs(5);
    while child.try_wait().unwrap().is_none() && Instant::now() < end {
        std::thread::sleep(Duration::from_millis(10));
    }
    let _ = child.kill();
    let status = child.wait().unwrap();
    let (mut out, mut err) = (String::new(), String::new());
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut err)
        .unwrap();
    (status, out, err)
}

pub struct Daemon {
    pub child: Child,
    pub socket: PathBuf,
    dir: PathBuf,
    lines: Receiver<String>,
}

impl Daemon {
    /// Starts a daemon in a fresh directory and waits for `SERVE_READY`.
    pub fn start(extra: &[&str]) -> Daemon {
        Daemon::start_in(scratch_dir(), extra)
    }

    pub fn start_in(dir: PathBuf, extra: &[&str]) -> Daemon {
        let socket = dir.join("s.sock");
        let mut child = serve_command(&socket, extra)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let out = child.stdout.take().unwrap();
        let (tx, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let _ = tx.send(line);
            }
        });
        let d = Daemon {
            child,
            socket,
            dir,
            lines,
        };
        d.line("SERVE_READY", Duration::from_secs(5))
            .expect("SERVE_READY within 5 s");
        d
    }

    /// The next stdout line that starts with `prefix`, or None after `timeout`.
    pub fn line(&self, prefix: &str, timeout: Duration) -> Option<String> {
        let end = Instant::now() + timeout;
        loop {
            let left = end.checked_duration_since(Instant::now())?;
            match self.lines.recv_timeout(left) {
                Ok(l) if l.starts_with(prefix) => return Some(l),
                Ok(_) => {}
                Err(_) => return None,
            }
        }
    }

    pub fn connect(&self, role: Role) -> Client {
        Client::connect(&self.socket, role)
    }

    /// The exit status within `timeout`, or None if it still runs.
    pub fn exit_within(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let end = Instant::now() + timeout;
        loop {
            if let Some(s) = self.child.try_wait().unwrap() {
                return Some(s);
            }
            if Instant::now() >= end {
                return None;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn signal(&self, sig: &str) {
        let pid = self.child.id().to_string();
        assert!(Command::new("kill")
            .args([sig, &pid])
            .status()
            .unwrap()
            .success());
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub struct Client {
    stream: UnixStream,
    seq: u32,
}

impl Client {
    pub fn connect(socket: &Path, role: Role) -> Client {
        let mut stream = UnixStream::connect(socket).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .write_all(&wire::encode_client(&ClientFrame::Hello(role)))
            .unwrap();
        Client { stream, seq: 0 }
    }

    fn send(&mut self, f: ClientFrame) -> Result<StateFrame, WireError> {
        self.stream.write_all(&wire::encode_client(&f))?;
        let s = wire::read_state(&mut self.stream)?;
        assert_eq!(s.seq, self.seq, "the reply answers the frame just sent");
        Ok(s)
    }

    pub fn control(&mut self, op: Op) -> Result<StateFrame, WireError> {
        self.seq += 1;
        let seq = self.seq;
        self.send(ClientFrame::Control { seq, op })
    }

    pub fn poll(&mut self) -> StateFrame {
        self.control(Op::Poll).unwrap()
    }

    pub fn command(&mut self, q: [f64; 7]) -> StateFrame {
        self.seq += 1;
        let seq = self.seq;
        self.send(ClientFrame::Command { seq, q }).unwrap()
    }
}
