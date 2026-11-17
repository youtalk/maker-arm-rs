//! Command cores for the maker-arm CLI, kept terminal-free so they run
//! against SimArm in tests. Rendering lives in main.rs.

pub mod export;
pub mod files;
pub mod wire;

use maker_arm::{ArmConfig, JointConfig, MotorModel};
use maker_arm_protocol::{self as p, mit};
use maker_arm_transport::{CanBackend, TransportError};
use std::f64::consts::TAU;
use std::time::{Duration, Instant};

// The session's timing, which is the vendor's: tries, and the wait for each.
const REPLY: (u32, Duration) = (3, Duration::from_millis(50));
const PARAM: (u32, Duration) = (2, Duration::from_millis(120));

#[derive(Debug, Clone)]
pub struct ScanRow {
    pub motor_id: u8,
    pub name: String,
    pub model: String,
    /// The motor answered the fault query.
    pub present: bool,
    /// NaN without a state read (an absent or faulted motor), as is `temperature`.
    pub joint_pos: f64,
    pub mode: u8,
    pub temperature: f64,
    pub fault_word: u32,
    /// The firmware answers parameter reads (a `RUN_MODE` read).
    pub params: bool,
}

/// Sends `f` up to `tries` times and waits `timeout` each time for the first reply, decoded
/// with `params`, that `pick` accepts. Frames already queued are dropped before each send.
fn ask<T>(
    backend: &mut dyn CanBackend,
    f: p::Frame,
    (tries, timeout): (u32, Duration),
    params: &p::MotorParams,
    pick: impl Fn(mit::Parsed) -> Option<T>,
) -> Result<Option<T>, TransportError> {
    for _ in 0..tries {
        while backend.recv(Duration::ZERO)?.is_some() {}
        backend.send(f.id, &f.data)?;
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some((id, data)) = backend.recv(Duration::from_millis(2))? {
                if let Some(x) = mit::parse_frame(id, &data, params).and_then(&pick) {
                    return Ok(Some(x));
                }
            }
        }
    }
    Ok(None)
}

/// What one motor answered.
struct Probe {
    fault_word: u32,
    /// The state read, sent only after a clean fault query.
    feedback: Option<mit::Feedback>,
}

/// The session's probe order: a non-clearing fault query, then the clearing state read only
/// after a clean answer, so a reported fault is never erased. Torque is never touched.
/// `None` when the motor does not answer the fault query.
fn probe_one(
    backend: &mut dyn CanBackend,
    j: &JointConfig,
) -> Result<Option<Probe>, TransportError> {
    let (m, params) = (j.motor_id, j.model.params());
    let word = ask(
        backend,
        mit::encode_fault_query(m),
        REPLY,
        params,
        |r| match r {
            mit::Parsed::Fault(s) if s.motor_id == m => Some(s.word),
            _ => None,
        },
    )?;
    let Some(fault_word) = word else {
        return Ok(None);
    };
    let feedback = if fault_word == 0 {
        ask(
            backend,
            mit::encode_clear_fault(m),
            REPLY,
            params,
            |r| match r {
                mit::Parsed::Feedback(fb) if fb.motor_id == m => Some(fb),
                _ => None,
            },
        )?
    } else {
        None
    };
    Ok(Some(Probe {
        fault_word,
        feedback,
    }))
}

/// The MIT parameter read. `None` when no reply comes: the firmware has no parameter access.
fn read_param(
    backend: &mut dyn CanBackend,
    motor_id: u8,
    index: u16,
) -> Result<Option<[u8; 4]>, TransportError> {
    let f = mit::encode_read_param(motor_id, index);
    // Parameter replies decode the same for every model.
    ask(backend, f, PARAM, &p::RS00, |r| match r {
        mit::Parsed::Param(r) if !r.write && r.motor_id == motor_id && r.index == index => {
            Some(r.raw)
        }
        _ => None,
    })
}

pub fn scan(
    backend: &mut dyn CanBackend,
    config: &ArmConfig,
) -> Result<Vec<ScanRow>, TransportError> {
    let mut rows = Vec::new();
    for j in &config.joints {
        let probe = probe_one(backend, j)?;
        let params =
            probe.is_some() && read_param(backend, j.motor_id, p::param_index::RUN_MODE)?.is_some();
        let fb = probe.as_ref().and_then(|pr| pr.feedback);
        rows.push(ScanRow {
            motor_id: j.motor_id,
            name: j.name.to_string(),
            model: j.model.name().to_string(),
            present: probe.is_some(),
            joint_pos: fb.map_or(f64::NAN, |fb| j.to_joint(fb.position)),
            mode: fb.map_or(0, |fb| fb.mode),
            temperature: fb.map_or(f64::NAN, |fb| fb.temperature),
            fault_word: probe.map_or(0, |pr| pr.fault_word),
            params,
        });
    }
    Ok(rows)
}

/// One `scan --watch` sweep: J1 to J6 in URDF-frame radians and the jaw in metres per jaw, NaN for
/// a motor that gives no state. It probes like `scan`, so it never enables or commands a motor
/// and never clears a reported fault. The first sweep that hears all six joints picks each
/// joint's 2-pi wrap like `Session::connect` does and `wraps` keeps it for the later sweeps.
pub fn watch_sweep(
    backend: &mut dyn CanBackend,
    config: &ArmConfig,
    grip: &files::GripMap,
    wraps: &mut Option<[f64; 6]>,
) -> Result<[f64; 7], TransportError> {
    let mut raw = [f64::NAN; 7];
    for (r, j) in raw.iter_mut().zip(&config.joints) {
        if let Some(fb) = probe_one(backend, j)?.and_then(|pr| pr.feedback) {
            *r = fb.position;
        }
    }
    let chosen = wraps.unwrap_or_else(|| {
        let mut w = [0.0; 6];
        for (w, (j, r)) in w.iter_mut().zip(config.joints.iter().zip(raw)) {
            let (lo, hi) = (
                j.q_lo - config.enable_limit_grace,
                j.q_hi + config.enable_limit_grace,
            );
            // No wrap inside the limits: show the plain reading.
            *w = [0.0, TAU, -TAU]
                .into_iter()
                .find(|w| (lo..=hi).contains(&j.to_joint(r + w)))
                .unwrap_or(0.0);
        }
        w
    });
    if raw[..6].iter().all(|r| r.is_finite()) {
        *wraps = Some(chosen);
    }
    let mut q = [f64::NAN; 7];
    for i in 0..6 {
        q[i] = config.joints[i].to_joint(raw[i] + chosen[i]);
    }
    q[6] = grip.to_joint_m(config.joints[6].to_joint(raw[6]));
    Ok(q)
}

#[derive(Debug, Clone)]
pub struct DoctorRow {
    pub motor_id: u8,
    pub present: bool,
    pub joint_pos: f64,
    pub in_limits: bool,
    pub run_mode: Option<u8>,
    pub can_timeout: Option<u32>,
    pub vbus: Option<f32>,
    pub temperature: f64,
    pub fault_word: u32,
    pub params: bool,
}

#[derive(Debug, Clone)]
pub struct DoctorReport {
    pub rows: Vec<DoctorRow>,
    pub warnings: Vec<String>,
}

impl DoctorReport {
    /// Every motor is present, in limits and fault-free.
    pub fn passed(&self) -> bool {
        self.rows
            .iter()
            .all(|r| r.present && r.in_limits && r.fault_word == 0)
    }
}

/// "motors 2, 3" for the rows of motors 2 and 3.
fn motor_list(rows: &[&DoctorRow]) -> String {
    let ids: Vec<String> = rows.iter().map(|r| r.motor_id.to_string()).collect();
    format!("motors {}", ids.join(", "))
}

pub fn doctor(
    backend: &mut dyn CanBackend,
    config: &ArmConfig,
) -> Result<DoctorReport, TransportError> {
    let scan_rows = scan(backend, config)?;
    let mut rows = Vec::new();
    for (row, j) in scan_rows.iter().zip(&config.joints) {
        let (run_mode, can_timeout, vbus) = if row.params {
            (
                read_param(backend, j.motor_id, p::param_index::RUN_MODE)?.map(|raw| raw[0]),
                read_param(backend, j.motor_id, p::param_index::CAN_TIMEOUT)?
                    .map(u32::from_le_bytes),
                read_param(backend, j.motor_id, p::param_index::VBUS)?.map(f32::from_le_bytes),
            )
        } else {
            (None, None, None)
        };
        let (lo, hi) = (
            j.q_lo - config.enable_limit_grace,
            j.q_hi + config.enable_limit_grace,
        );
        // Same wrap tolerance as connect: a position is "in limits" if some
        // 2π branch lands in the grace window.
        let in_limits = row.present
            && [0.0, std::f64::consts::TAU, -std::f64::consts::TAU]
                .iter()
                .any(|w| {
                    let q = row.joint_pos + j.direction * w;
                    q >= lo && q <= hi
                });
        rows.push(DoctorRow {
            motor_id: j.motor_id,
            present: row.present,
            joint_pos: row.joint_pos,
            in_limits,
            run_mode,
            can_timeout,
            vbus,
            temperature: row.temperature,
            fault_word: row.fault_word,
            params: row.params,
        });
    }
    let mut warnings = Vec::new();
    let no_params: Vec<&DoctorRow> = rows.iter().filter(|r| r.present && !r.params).collect();
    let rs02: Vec<&DoctorRow> = no_params
        .iter()
        .copied()
        .filter(|r| {
            config
                .joint_by_motor_id(r.motor_id)
                .is_some_and(|j| j.model == MotorModel::Rs02)
        })
        .collect();
    if !rs02.is_empty() {
        warnings.push(format!(
            "{} (RS02): no parameter access, so the firmware may be older than 0.2.3.32 and \
             the position range is unknown (+/-12.5 or +/-12.57 rad). Check the firmware with \
             the vendor tool before the first motion.",
            motor_list(&rs02)
        ));
    }
    if !no_params.is_empty() {
        warnings.push(format!(
            "{}: no parameter access, so no motor watchdog and no RUN_MODE or mode check.",
            motor_list(&no_params)
        ));
    }
    Ok(DoctorReport { rows, warnings })
}

/// Zeroes one motor as the vendor does. Every motor is disabled first, so torque stays off for
/// the whole procedure: a motor left enabled by a dead daemon must not drive toward its old
/// target once the zero moves. Then `confirm` runs (the operator's prompt), then set-zero and
/// a re-probe. A motor that does not answer or reports a fault is not zeroed, and nothing is
/// disabled. `confirm` returning false aborts with every motor disabled.
pub fn zero_motor(
    backend: &mut dyn CanBackend,
    config: &ArmConfig,
    motor_id: u8,
    confirm: impl FnOnce() -> bool,
) -> Result<f64, String> {
    let j = config
        .joint_by_motor_id(motor_id)
        .ok_or_else(|| format!("unknown motor id {motor_id}"))?;
    let probe = |backend: &mut dyn CanBackend| match probe_one(backend, j) {
        Err(e) => Err(e.to_string()),
        Ok(None) => Err(format!("motor {motor_id} did not answer")),
        Ok(Some(pr)) if pr.fault_word != 0 => Err(format!(
            "motor {motor_id} reports a fault: {}",
            mit::fault_text(pr.fault_word)
        )),
        Ok(Some(pr)) => pr
            .feedback
            .map(|fb| j.to_joint(fb.position))
            .ok_or_else(|| format!("motor {motor_id} did not answer the state read")),
    };
    probe(backend)?;
    let e = |e: TransportError| e.to_string();
    for j in &config.joints {
        let f = mit::encode_disable(j.motor_id);
        backend.send(f.id, &f.data).map_err(e)?;
    }
    if !confirm() {
        return Err("aborted".into());
    }
    let f = mit::encode_set_zero(motor_id);
    backend.send(f.id, &f.data).map_err(e)?;
    probe(backend)
}

/// The typed-RELEASE gate, mirroring the official SDK's cli/safety.py:
/// returns only once the operator typed RELEASE (case-insensitive,
/// trimmed). Anything else — including EOF, other read errors, and
/// prompt/warning write failures — keeps torque on and retries. The
/// caller disables the arm only after this returns.
///
/// Returns `()`, not `io::Result<()>`: every failure mode inside was made
/// non-releasing (read errors and write errors alike are warn-and-retry),
/// so there is no longer any `Err` to return. The plan pinned an
/// `io::Result` signature, but a `Result` that can never be `Err` is dead
/// failure surface on a safety gate — it invites a caller's `?`, and a
/// `?` here drops the live `RunningArm`, whose `Drop` disables the
/// motors: a release with no typed RELEASE at all. Deliberate,
/// approved deviation from the plan.
pub fn confirm_release(input: &mut dyn std::io::BufRead, output: &mut dyn std::io::Write) {
    loop {
        // Prompting is best-effort: writing/flushing here must NEVER
        // become a release path. If this function returned `Err` on a
        // write failure (e.g. a broken stdout), the caller's `?` would
        // drop the live `RunningArm`, and `RunningArm`'s `Drop` disables
        // the motors -- releasing torque with no typed RELEASE at all.
        // That is the exact same inversion closed for read errors below,
        // just reached through the write side instead of the read side.
        // The operator can't see a prompt that failed to print, but a
        // HOLDING arm is still the correct failure direction.
        let mut wrote_ok = writeln!(
            output,
            "arm is holding. Type RELEASE and press ENTER only when it is safe to release torque."
        )
        .is_ok();
        wrote_ok &= output.flush().is_ok();

        let mut line = String::new();
        // EOF (`Ok(0)`) and any read `Err` are treated identically: keep
        // torque on, warn, sleep, and retry -- NEVER propagate. If this
        // function returned `Err`, the caller's `?` would drop the
        // `RunningArm`, and `RunningArm`'s `Drop` disables the motors --
        // cutting torque with no typed RELEASE at all. Releasing torque
        // must be reachable only through that typed line.
        //
        // A literal Ctrl-C (SIGINT) cannot surface here as
        // `ErrorKind::Interrupted`: Rust's std retries EINTR at the raw
        // `read(2)` layer, below `BufRead`, so `read_line` blocks straight
        // through a signal and returns real data once it arrives. Do not
        // read that as license to drop this branch -- it still guards
        // every OTHER I/O error a real stdin can produce (and is exercised
        // directly by `confirm_release_treats_read_errors_like_eof` in
        // tests/commands.rs, which forces exactly that error kind).
        let is_data = matches!(input.read_line(&mut line), Ok(n) if n > 0);
        if !is_data {
            let _ = writeln!(
                output,
                "input is unavailable; torque remains enabled while this process is alive"
            );
            std::thread::sleep(std::time::Duration::from_secs(1));
            continue;
        }
        if line.trim().eq_ignore_ascii_case("release") {
            return;
        }
        wrote_ok &= writeln!(
            output,
            "torque remains enabled; type RELEASE only when the arm is supported"
        )
        .is_ok();
        // If every write this iteration failed, `read_line`'s normal
        // blocking can no longer be trusted to pace the loop on its own
        // (a pathological input could keep returning immediate mismatched
        // lines while output stays fully broken); sleep the same 1s used
        // for the dead-input case above so this can never spin at full
        // CPU speed.
        if !wrote_ok {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
}
