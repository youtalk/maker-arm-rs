// `pyo3`'s `extension-module` feature (see Cargo.toml) links against no
// Python interpreter, which breaks `cargo test` at link time with
// undefined `Py*` symbols. It is therefore feature-gated (off by default)
// rather than always-on: `maturin` turns it on for wheel builds
// (`pyproject.toml`), and the workspace's `cargo test` step excludes this
// crate entirely (`--exclude maker-arm-py`) so `--all-features` can never
// reach it. Bindings tests live in the Python suite
// (`tests/test_bindings.py`), run against a `maturin develop` build.
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

#[pymodule]
fn maker_arm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_mit, m)?)?;
    m.add_function(wrap_pyfunction!(parse_frame, m)?)?;
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
    fn start_hold(&mut self) -> PyResult<()> {
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

    fn snapshot(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let snap = match &self.handle {
            ArmHandle::Running(r) => r.snapshot(),
            _ => None,
        };
        let Some(snap) = snap else { return Ok(None) };
        let d = PyDict::new(py);
        d.set_item("state", state_str(snap.state))?;
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

    fn state(&self) -> PyResult<&'static str> {
        match &self.handle {
            ArmHandle::Idle(s) => Ok(state_str(s.state())),
            ArmHandle::Running(r) => Ok(r
                .snapshot()
                .map(|s| state_str(s.state))
                .unwrap_or("enabled")),
            ArmHandle::Empty => Err(PyRuntimeError::new_err("arm handle poisoned")),
        }
    }
}
