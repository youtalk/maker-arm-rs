// `pyo3`'s `extension-module` feature (see Cargo.toml) is enabled
// unconditionally, which normally breaks `cargo test` at link time with
// undefined `Py*` symbols. It doesn't break here only because this crate
// has no `#[test]` items of its own: nothing in the test harness's
// synthetic `main()` reaches the PyO3-touching code below, so the
// linker's `--gc-sections` pass drops it before symbol resolution runs.
// Adding a `#[test]` here that calls into these bindings would bring that
// dead code back into the link and resurface the failure — put such tests
// in the Python suite (`tests/test_bindings.py`) instead.
use maker_arm_protocol as p;
use pyo3::exceptions::PyValueError;
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
    Ok(())
}
