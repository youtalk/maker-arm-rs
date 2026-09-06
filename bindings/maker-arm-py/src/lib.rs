// `pyo3`'s `extension-module` feature (see Cargo.toml) links against no
// Python interpreter, which breaks `cargo test` at link time with
// undefined `Py*` symbols. It is therefore feature-gated (off by default)
// rather than always-on: `maturin` turns it on for wheel builds
// (`pyproject.toml`), and the workspace's `cargo test` step excludes this
// crate entirely (`--exclude maker-arm-py`) so `--all-features` can never
// reach it. Bindings tests live in the Python suite
// (`tests/test_bindings.py`), run against a `maturin develop` build.
use maker_arm::state::JointCommand;
use maker_arm::{ArmConfig, HoldController, RunningArm, Session, SessionState, SimArm};
use maker_arm_protocol as p;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn params(model: &str) -> PyResult<&'static p::MotorParams> {
    p::motor_params(model)
        .ok_or_else(|| PyValueError::new_err(format!("unknown motor model: {model}")))
}

// The `py` token pushes the arity to 8; the remaining 7 mirror
// `maker_arm_protocol::encode_mit` and the brief's documented Python
// signature `encode_mit(motor_id, pos, vel, kp, kd, tau, model)`, so
// grouping them into a struct would change the public binding API.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn encode_mit(
    py: Python<'_>,
    motor_id: u8,
    pos: f64,
    vel: f64,
    kp: f64,
    kd: f64,
    tau: f64,
    model: &str,
) -> PyResult<(u32, Py<PyBytes>)> {
    let f = p::encode_mit(motor_id, pos, vel, kp, kd, tau, params(model)?);
    Ok((f.id, PyBytes::new(py, &f.data).into()))
}

#[pyfunction]
fn parse_frame(
    py: Python<'_>,
    can_id: u32,
    data: &[u8],
    model: &str,
) -> PyResult<Option<Py<PyDict>>> {
    let parsed = match p::parse_frame(can_id, data, params(model)?) {
        Some(x) => x,
        None => return Ok(None),
    };
    let d = PyDict::new(py);
    match parsed {
        p::ParsedFrame::Feedback(fb) => {
            d.set_item("kind", "feedback")?;
            d.set_item("motor_id", fb.motor_id)?;
            d.set_item("position", fb.position)?;
            d.set_item("velocity", fb.velocity)?;
            d.set_item("torque", fb.torque)?;
            d.set_item("temperature", fb.temperature)?;
            d.set_item("mode", fb.mode)?;
            d.set_item("fault_bits", fb.fault_bits)?;
        }
        p::ParsedFrame::ParamReply(r) => {
            d.set_item("kind", "param_reply")?;
            d.set_item("motor_id", r.motor_id)?;
            d.set_item("index", r.index)?;
            d.set_item("raw", PyBytes::new(py, &r.raw))?;
        }
        p::ParsedFrame::Fault(fr) => {
            d.set_item("kind", "fault")?;
            d.set_item("motor_id", fr.motor_id)?;
            d.set_item("raw", PyBytes::new(py, &fr.raw))?;
        }
    }
    Ok(Some(d.into()))
}

/// The pinned `maker_arm_v1` profile as plain Python data. A fresh dict on
/// every call: callers may edit numeric fields and hand the result to
/// `clamp_command` (see that function for which fields it reads).
#[pyfunction]
fn profile(py: Python<'_>) -> PyResult<Py<PyDict>> {
    config_to_dict(py, &ArmConfig::maker_arm_v1())
}

fn config_to_dict(py: Python<'_>, c: &ArmConfig) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    let joints = pyo3::types::PyList::empty(py);
    for j in &c.joints {
        let jd = PyDict::new(py);
        jd.set_item("motor_id", j.motor_id)?;
        jd.set_item("name", j.name)?;
        jd.set_item("model", j.model.name())?;
        jd.set_item("kp", j.kp)?;
        jd.set_item("kd", j.kd)?;
        jd.set_item("tau_max", j.tau_max)?;
        jd.set_item("q_lo", j.q_lo)?;
        jd.set_item("q_hi", j.q_hi)?;
        jd.set_item("direction", j.direction)?;
        jd.set_item("offset", j.offset)?;
        joints.append(jd)?;
    }
    d.set_item("joints", joints)?;
    d.set_item("control_rate_hz", c.control_rate_hz)?;
    d.set_item("max_velocity", c.max_velocity)?;
    d.set_item("feedback_timeout", c.feedback_timeout)?;
    d.set_item("limit_margin", c.limit_margin)?;
    d.set_item("kp_max", c.kp_max)?;
    d.set_item("kd_max", c.kd_max)?;
    d.set_item("temp_hold_c", c.temp_hold_c)?;
    Ok(d.into())
}

fn override_f64(d: &Bound<'_, PyDict>, key: &str, slot: &mut f64) -> PyResult<()> {
    if let Some(v) = d.get_item(key)? {
        let x: f64 = v
            .extract()
            .map_err(|_| PyValueError::new_err(format!("profile field {key} must be a number")))?;
        *slot = x;
    }
    Ok(())
}

/// Build an ArmConfig from the v1 profile plus the numeric overrides in `profile`.
/// Names, models, and motor ids are never taken from Python.
fn config_from_dict(profile: &Bound<'_, PyDict>) -> PyResult<ArmConfig> {
    let mut c = ArmConfig::maker_arm_v1();
    if let Some(joints) = profile.get_item("joints")? {
        let joints = joints.downcast::<pyo3::types::PyList>()?;
        if joints.len() != c.joints.len() {
            return Err(PyValueError::new_err(format!(
                "profile has {} joints, arm has {}",
                joints.len(),
                c.joints.len()
            )));
        }
        for (j, item) in c.joints.iter_mut().zip(joints.iter()) {
            let d = item.downcast::<PyDict>()?;
            override_f64(d, "q_lo", &mut j.q_lo)?;
            override_f64(d, "q_hi", &mut j.q_hi)?;
            override_f64(d, "kp", &mut j.kp)?;
            override_f64(d, "kd", &mut j.kd)?;
            override_f64(d, "tau_max", &mut j.tau_max)?;
        }
    }
    override_f64(profile, "max_velocity", &mut c.max_velocity)?;
    override_f64(profile, "kp_max", &mut c.kp_max)?;
    override_f64(profile, "kd_max", &mut c.kd_max)?;
    override_f64(profile, "limit_margin", &mut c.limit_margin)?;
    Ok(c)
}

/// The single-point command clamp, as a pure function. `cmds` is a list of
/// 7 `(pos, vel, kp, kd, tau)` tuples in joint order; returns the clamped
/// list and whether anything changed. The control loop's own command path
/// always runs commands through this clamp before they reach a motor;
/// `encode_mit` is a separate low-level protocol encoder exposed for tests
/// and tooling and is not itself clamped.
// The 5-tuple mirrors JointCommand's fields one for one, and PyO3 maps it
// straight to/from a Python tuple; a named wrapper type would just move
// the same fields behind an extra layer with no gain in clarity.
#[allow(clippy::type_complexity)]
#[pyfunction]
fn clamp_command(
    cmds: Vec<(f64, f64, f64, f64, f64)>,
    profile: &Bound<'_, PyDict>,
) -> PyResult<(Vec<(f64, f64, f64, f64, f64)>, bool)> {
    let config = config_from_dict(profile)?;
    let input: Vec<JointCommand> = cmds
        .iter()
        .map(|&(pos, vel, kp, kd, tau)| JointCommand {
            pos,
            vel,
            kp,
            kd,
            tau,
        })
        .collect();
    let (out, clamped) = maker_arm::clamp_command(&input, &config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.iter()
            .map(|c| (c.pos, c.vel, c.kp, c.kd, c.tau))
            .collect(),
        clamped,
    ))
}

#[pymodule]
fn maker_arm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_mit, m)?)?;
    m.add_function(wrap_pyfunction!(parse_frame, m)?)?;
    m.add_function(wrap_pyfunction!(profile, m)?)?;
    m.add_function(wrap_pyfunction!(clamp_command, m)?)?;
    m.add_class::<Arm>()?;
    Ok(())
}

// `Session` (the `Idle` payload) is larger than `RunningArm`, but `Arm`
// holds exactly one `ArmHandle` per Python object (never an array of
// them), so the wasted stack space clippy is warning about does not
// apply here; boxing `Session` would only add an indirection with no
// benefit.
#[allow(clippy::large_enum_variant)]
enum ArmHandle {
    Idle(Session),
    Running(RunningArm),
    /// Transient state while moving between the two.
    Empty,
}

/// Orchestration-mode arm handle (design §2): Python selects and steers
/// Rust controllers; it never commands torque directly, so every command
/// path stays behind the Rust clamp.
#[pyclass(unsendable)]
struct Arm {
    handle: ArmHandle,
    config: ArmConfig,
}

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn state_str(s: SessionState) -> &'static str {
    match s {
        SessionState::Connected => "connected",
        SessionState::Enabled => "enabled",
        SessionState::Fault => "fault",
    }
}

/// Reported while the control thread is alive but has not published its
/// first tick snapshot yet. Distinct from `"enabled"`: torque is on, but
/// nothing has been observed coming back from the loop.
const STARTING: &str = "starting";

/// Reported once the control thread has exited without being joined. The
/// loop is NOT commanding the motors any more; if it exited on an error it
/// did not disable them either, so torque may still be on with only the
/// motor-side CAN_TIMEOUT watchdog behind it. `stop()` joins the thread
/// and surfaces the error.
const LOOP_STOPPED: &str = "loop_stopped";

#[pymethods]
impl Arm {
    /// Connect to the built-in 7-motor simulator.
    #[staticmethod]
    fn sim() -> PyResult<Arm> {
        let config = ArmConfig::maker_arm_v1();
        let session =
            Session::connect(Box::new(SimArm::new(&config)), config.clone()).map_err(err)?;
        Ok(Arm {
            handle: ArmHandle::Idle(session),
            config,
        })
    }

    /// Connect over SocketCAN, e.g. Arm.socketcan("can0").
    #[staticmethod]
    fn socketcan(interface: &str) -> PyResult<Arm> {
        let config = ArmConfig::maker_arm_v1();
        let backend = maker_arm_transport::SocketCanBackend::open(interface).map_err(err)?;
        let session = Session::connect(Box::new(backend), config.clone()).map_err(err)?;
        Ok(Arm {
            handle: ArmHandle::Idle(session),
            config,
        })
    }

    fn enable(&mut self) -> PyResult<()> {
        match &mut self.handle {
            ArmHandle::Idle(s) => s.enable().map_err(err),
            _ => Err(PyRuntimeError::new_err("loop already running")),
        }
    }

    /// Spawn the 200 Hz control loop holding the current pose.
    ///
    /// Requires an enabled session: raises RuntimeError otherwise. Without
    /// this gate the loop thread died on its first tick with a wrong-state
    /// error, `snapshot()` returned None forever, and `state()` reported
    /// "enabled" for an arm with no torque at all -- an operator told the
    /// arm is holding might stop supporting it.
    fn start_hold(&mut self) -> PyResult<()> {
        if let ArmHandle::Idle(s) = &self.handle {
            if s.state() != SessionState::Enabled {
                return Err(PyRuntimeError::new_err(format!(
                    "start_hold requires an enabled session, but this one is {}; \
                     call enable() first",
                    state_str(s.state())
                )));
            }
        }
        match std::mem::replace(&mut self.handle, ArmHandle::Empty) {
            ArmHandle::Idle(s) => {
                let hold = HoldController::from_config(&self.config);
                self.handle = ArmHandle::Running(s.start(Box::new(hold)));
                Ok(())
            }
            other => {
                self.handle = other;
                Err(PyRuntimeError::new_err("loop already running"))
            }
        }
    }

    /// Retarget the running loop to hold the pose it is at right now.
    fn hold_now(&self) -> PyResult<()> {
        match &self.handle {
            ArmHandle::Running(r) => {
                r.hold_now();
                Ok(())
            }
            _ => Err(PyRuntimeError::new_err("loop is not running")),
        }
    }

    /// The last published tick snapshot, or None if no loop is running (or
    /// none has ticked yet).
    ///
    /// `loop_alive` reports whether the control thread is still running.
    /// When it is False the snapshot is the LAST one the dead loop
    /// published, and `state` reads "loop_stopped" rather than repeating
    /// that stale session state as if it were current -- call stop() to
    /// join the thread and surface whatever error ended it.
    fn snapshot(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let (snap, alive) = match &self.handle {
            ArmHandle::Running(r) => (r.snapshot(), !r.loop_finished()),
            _ => (None, false),
        };
        let Some(snap) = snap else { return Ok(None) };
        let d = PyDict::new(py);
        d.set_item(
            "state",
            if alive {
                state_str(snap.state)
            } else {
                LOOP_STOPPED
            },
        )?;
        d.set_item("loop_alive", alive)?;
        d.set_item("fault", snap.fault.as_ref().map(|f| f.to_string()))?;
        d.set_item("tick", snap.arm.tick)?;
        d.set_item("t", snap.arm.t)?;
        d.set_item(
            "positions",
            snap.arm
                .motors
                .iter()
                .map(|m| m.position)
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "velocities",
            snap.arm
                .motors
                .iter()
                .map(|m| m.velocity)
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "torques",
            snap.arm.motors.iter().map(|m| m.torque).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "temperatures",
            snap.arm
                .motors
                .iter()
                .map(|m| m.temperature)
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "fault_bits",
            // as u16: a Vec<u8> would convert to Python bytes, not a list
            snap.arm
                .motors
                .iter()
                .map(|m| m.fault_bits as u16)
                .collect::<Vec<_>>(),
        )?;
        Ok(Some(d.into()))
    }

    /// Disable all motors; joins the loop if one is running.
    fn stop(&mut self) -> PyResult<()> {
        match std::mem::replace(&mut self.handle, ArmHandle::Empty) {
            ArmHandle::Running(r) => {
                let (session, res) = r.stop_and_disable();
                self.handle = ArmHandle::Idle(session);
                res.map_err(err)
            }
            ArmHandle::Idle(mut s) => {
                let r = s.estop().map_err(err);
                self.handle = ArmHandle::Idle(s);
                r
            }
            ArmHandle::Empty => Err(PyRuntimeError::new_err("arm handle poisoned")),
        }
    }

    /// Alias for stop(): immediate torque-off.
    fn estop(&mut self) -> PyResult<()> {
        self.stop()
    }

    /// One of "connected", "enabled", "fault", "starting", or
    /// "loop_stopped".
    ///
    /// With no loop running this is the session's own state. With a loop
    /// running it is the state the loop last published -- except that loop
    /// LIVENESS wins over a stale snapshot:
    ///
    /// * "starting" -- the thread is alive but has not completed its first
    ///   tick yet, so nothing has come back from it.
    /// * "loop_stopped" -- the thread has exited and has not been joined.
    ///   It is no longer commanding the motors, and if it exited on an
    ///   error it did not disable them either, so torque may still be on
    ///   with only the motor-side CAN_TIMEOUT watchdog behind it. Call
    ///   stop() to join it and surface the error.
    ///
    /// This used to default to "enabled" whenever no snapshot was
    /// available, which reported a dead loop -- or an un-energized arm --
    /// as if it were holding.
    fn state(&self) -> PyResult<&'static str> {
        match &self.handle {
            ArmHandle::Idle(s) => Ok(state_str(s.state())),
            ArmHandle::Running(r) => {
                if r.loop_finished() {
                    return Ok(LOOP_STOPPED);
                }
                Ok(r.snapshot().map_or(STARTING, |s| state_str(s.state)))
            }
            ArmHandle::Empty => Err(PyRuntimeError::new_err("arm handle poisoned")),
        }
    }
}
