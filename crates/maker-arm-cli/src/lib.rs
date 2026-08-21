//! Command cores for the maker-arm CLI, kept terminal-free so they run
//! against SimArm in tests. Rendering lives in main.rs.

use maker_arm::{ArmConfig, JointConfig};
use maker_arm_protocol as p;
use maker_arm_transport::{CanBackend, TransportError};
use std::time::{Duration, Instant};

const PROBE_TIMEOUT: Duration = Duration::from_millis(50);

#[derive(Debug, Clone)]
pub struct ScanRow {
    pub motor_id: u8,
    pub name: String,
    pub model: String,
    pub present: bool,
    pub joint_pos: f64,
    pub mode: u8,
    pub temperature: f64,
    pub fault_bits: u8,
}

/// Read-only probe of one motor: disable frame → one feedback frame (the
/// RobStride probing convention; harmless while torque is off).
fn probe_one(
    backend: &mut dyn CanBackend,
    j: &JointConfig,
    host_id: u8,
) -> Result<Option<p::MotorFeedback>, TransportError> {
    let f = p::encode_disable(j.motor_id, false, host_id);
    backend.send(f.id, &f.data)?;
    let deadline = Instant::now() + PROBE_TIMEOUT;
    while Instant::now() < deadline {
        if let Some((id, data)) = backend.recv(Duration::from_millis(2))? {
            if ((id >> 8) & 0xFF) as u8 != j.motor_id {
                continue; // stale frame from an earlier probe
            }
            if let Some(p::ParsedFrame::Feedback(fb)) = p::parse_frame(id, &data, j.model.params())
            {
                return Ok(Some(fb));
            }
        }
    }
    Ok(None)
}

pub fn scan(
    backend: &mut dyn CanBackend,
    config: &ArmConfig,
) -> Result<Vec<ScanRow>, TransportError> {
    let mut rows = Vec::new();
    for j in &config.joints {
        let fb = probe_one(backend, j, config.host_id)?;
        rows.push(match fb {
            Some(fb) => ScanRow {
                motor_id: j.motor_id,
                name: j.name.to_string(),
                model: j.model.name().to_string(),
                present: true,
                joint_pos: j.to_joint(fb.position),
                mode: fb.mode,
                temperature: fb.temperature,
                fault_bits: fb.fault_bits,
            },
            None => ScanRow {
                motor_id: j.motor_id,
                name: j.name.to_string(),
                model: j.model.name().to_string(),
                present: false,
                joint_pos: f64::NAN,
                mode: 0,
                temperature: f64::NAN,
                fault_bits: 0,
            },
        });
    }
    Ok(rows)
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
    pub fault_bits: u8,
}

#[derive(Debug, Clone)]
pub struct DoctorReport {
    pub rows: Vec<DoctorRow>,
    pub warnings: Vec<String>,
}

fn read_param(
    backend: &mut dyn CanBackend,
    motor_id: u8,
    index: u16,
    host_id: u8,
) -> Result<Option<[u8; 4]>, TransportError> {
    let f = p::encode_read_param(motor_id, index, host_id);
    backend.send(f.id, &f.data)?;
    let deadline = Instant::now() + PROBE_TIMEOUT;
    while Instant::now() < deadline {
        if let Some((id, data)) = backend.recv(Duration::from_millis(2))? {
            // `parse_frame`'s ParamReply branch ignores `params` entirely
            // (only the Feedback branch is model-dependent), so `&p::RS00`
            // here is a deliberate placeholder, not a model mismatch bug.
            if let Some(p::ParsedFrame::ParamReply(r)) = p::parse_frame(id, &data, &p::RS00) {
                if r.motor_id == motor_id && r.index == index {
                    return Ok(Some(r.raw));
                }
            }
        }
    }
    Ok(None)
}

pub fn doctor(
    backend: &mut dyn CanBackend,
    config: &ArmConfig,
) -> Result<DoctorReport, TransportError> {
    let scan_rows = scan(backend, config)?;
    let mut rows = Vec::new();
    for (row, j) in scan_rows.iter().zip(&config.joints) {
        let (run_mode, can_timeout, vbus) = if row.present {
            (
                read_param(
                    backend,
                    j.motor_id,
                    p::param_index::RUN_MODE,
                    config.host_id,
                )?
                .map(|raw| raw[0]),
                read_param(
                    backend,
                    j.motor_id,
                    p::param_index::CAN_TIMEOUT,
                    config.host_id,
                )?
                .map(u32::from_le_bytes),
                read_param(backend, j.motor_id, p::param_index::VBUS, config.host_id)?
                    .map(f32::from_le_bytes),
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
            fault_bits: row.fault_bits,
        });
    }
    let warnings = vec![
        "RS02 firmware <= 0.2.2.11 maps position to +/-12.5 rad (this SDK assumes \
         +/-12.57). No READ_VERSION frame is pinnable yet (the official SDK ships \
         none) -- verify motors 2 and 3 with the vendor tool BEFORE first motion (MA1)."
            .to_string(),
    ];
    Ok(DoctorReport { rows, warnings })
}

/// Zeroes one motor (torque-free operation) and re-probes to confirm.
pub fn zero_motor(
    backend: &mut dyn CanBackend,
    config: &ArmConfig,
    motor_id: u8,
) -> Result<f64, String> {
    let j = config
        .joint_by_motor_id(motor_id)
        .ok_or_else(|| format!("unknown motor id {motor_id}"))?
        .clone();
    let f = p::encode_set_zero(motor_id, config.host_id);
    backend.send(f.id, &f.data).map_err(|e| e.to_string())?;
    // consume the set-zero feedback, then probe for a fresh reading
    let _ = probe_one(backend, &j, config.host_id).map_err(|e| e.to_string())?;
    match probe_one(backend, &j, config.host_id).map_err(|e| e.to_string())? {
        Some(fb) => Ok(j.to_joint(fb.position)),
        None => Err(format!("motor {motor_id} did not answer after set-zero")),
    }
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
