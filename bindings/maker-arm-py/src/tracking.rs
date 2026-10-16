//! `Tracking`: one `TrackingController` behind a thread-safe Python handle (tracking design
//! sections 9 and 10). Python drives it with `update` (the bench), or `Arm.start_tracking`
//! moves it into the 200 Hz loop; either way `push`, `now` and `telemetry` stay reachable
//! from any Python thread, and the loop thread never takes the GIL.

use crate::dynamics::Dynamics;
use maker_arm::residual::{Layer, Mlp as RsMlp, ResidualModel};
use maker_arm::state::{ArmCommand, ArmState, MotorState};
use maker_arm::tracking::{Push, Row, TrackingController, TrackingParams, ROW};
use maker_arm::{ArmConfig, Controller};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

fn value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn params_from_dict(d: &Bound<'_, PyDict>) -> PyResult<TrackingParams> {
    let mut p = TrackingParams::default();
    for (key, value) in d.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "f_r" => p.f_r = value.extract()?,
            "k_o" => p.k_o = value.extract()?,
            "f_y" => p.f_y = value.extract()?,
            "ka_ratio" => p.ka_ratio = value.extract()?,
            "tau_th" => p.tau_th = value.extract()?,
            "e_max" => p.e_max = value.extract()?,
            "gate_fall" => p.gate_fall = value.extract()?,
            "gate_rise" => p.gate_rise = value.extract()?,
            "gate_quiet" => p.gate_quiet = value.extract()?,
            "contact" => p.contact = value.extract()?,
            "dob" => p.dob = value.extract()?,
            "dob_f" => p.dob_f = value.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown tracking parameter {other}"
                )))
            }
        }
    }
    Ok(p)
}

/// `layers` as the lab's export writes it: `[[w, b], ...]`, `w` torch's `weight.tolist()`
/// (out rows of `in` values), `b` a flat list.
fn mlp_from_layers(layers: &Bound<'_, PyAny>) -> PyResult<RsMlp> {
    let shape =
        |e: PyErr| PyValueError::new_err(format!("residual layers must be [[w, b], ...]: {e}"));
    let raw: Vec<Bound<'_, PyAny>> = layers.extract().map_err(shape)?;
    let mut out = Vec::with_capacity(raw.len());
    for (k, item) in raw.iter().enumerate() {
        let pair: Vec<Bound<'_, PyAny>> = item.extract().map_err(shape)?;
        let [w, b] = &pair[..] else {
            return Err(PyValueError::new_err(format!(
                "residual layer {k}: needs [w, b], has {} items",
                pair.len()
            )));
        };
        let at = |e: PyErr| PyValueError::new_err(format!("residual layer {k}: {e}"));
        let w: Vec<Vec<f64>> = w.extract().map_err(at)?;
        let b: Vec<f64> = b.extract().map_err(at)?;
        if w.len() != b.len() {
            return Err(PyValueError::new_err(format!(
                "residual layer {k}: w has {} rows, b has {} values",
                w.len(),
                b.len()
            )));
        }
        let n_in = w.first().map_or(0, Vec::len);
        if let Some(i) = w.iter().position(|row| row.len() != n_in) {
            return Err(PyValueError::new_err(format!(
                "residual layer {k}: row {i} of w has {} values, row 0 has {n_in}",
                w[i].len()
            )));
        }
        out.push(Layer {
            w: w.into_iter().flatten().collect(),
            b,
        });
    }
    RsMlp::new(out).map_err(value_err)
}

/// The residual dict the lab's export writes: `history`, `layers`, `x_mean`, `x_std`,
/// `y_scale` and `out_max`, every one required and no other.
fn residual_from_dict(d: &Bound<'_, PyDict>) -> PyResult<ResidualModel> {
    const FIELDS: [&str; 6] = ["history", "layers", "x_mean", "x_std", "y_scale", "out_max"];
    for key in d.keys() {
        let key: String = key.extract()?;
        if !FIELDS.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown residual field {key}"
            )));
        }
    }
    let get = |name: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("the residual is missing {name}")))
    };
    let field =
        |name: &'static str| move |e: PyErr| PyValueError::new_err(format!("residual {name}: {e}"));
    let mlp = mlp_from_layers(&get("layers")?)?;
    ResidualModel::new(
        mlp,
        get("history")?.extract().map_err(field("history"))?,
        get("x_mean")?.extract().map_err(field("x_mean"))?,
        get("x_std")?.extract().map_err(field("x_std"))?,
        get("y_scale")?.extract().map_err(field("y_scale"))?,
        get("out_max")?.extract().map_err(field("out_max"))?,
    )
    .map_err(value_err)
}

/// The residual's MLP alone, for the lab's export round-trip test: `Mlp(layers).forward(x)`
/// on an already normalized `x`.
#[pyclass]
pub struct Mlp {
    inner: RsMlp,
}

#[pymethods]
impl Mlp {
    #[new]
    fn new(layers: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Mlp {
            inner: mlp_from_layers(layers)?,
        })
    }

    fn forward(&mut self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        if x.len() != self.inner.inputs() {
            return Err(PyValueError::new_err(format!(
                "x has {} values, the MLP takes {}",
                x.len(),
                self.inner.inputs()
            )));
        }
        let mut out = vec![0.0; self.inner.outputs()];
        self.inner.forward(&x, &mut out);
        Ok(out)
    }
}

/// The loop's side of a `Tracking`: the controller behind the same mutex. The lock is
/// uncontended once the controller runs in a loop, because `update` refuses from then on.
pub(crate) struct LoopTracking(Arc<Mutex<TrackingController>>);

impl Controller for LoopTracking {
    fn update(&mut self, state: &ArmState, dt: f64) -> ArmCommand {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .update(state, dt)
    }
}

/// The compliant tracking controller (tracking design sections 6, 7, 9 and 10).
///
/// The default `tau_th` is infinite, which means no yield: until a calibrated deadband is
/// passed, only the error clamp `e_max` acts on the contact side.
#[pyclass(frozen)]
pub struct Tracking {
    pub(crate) ctrl: Arc<Mutex<TrackingController>>,
    shared: Arc<maker_arm::tracking::TrackingShared>,
    pub(crate) in_loop: AtomicBool,
    /// The motors' CAN_TIMEOUT (s): in a loop, a last tick older than this means the loop no
    /// longer runs this controller (stopped, fault hold or `hold_now`).
    stale_s: f64,
}

impl Tracking {
    /// The controller for `Arm.start_tracking`, restarted so its first loop tick begins from
    /// the measured pose, with no clock and fresh telemetry. Marks the handle as running in a
    /// loop. Refuses rung F and a handle with a plan or a pending push, and a refusal leaves
    /// the handle as it was.
    pub(crate) fn enter_loop(&self) -> PyResult<LoopTracking> {
        // The flag first: a handle already in a loop is refused without taking the lock its
        // loop thread ticks under.
        if self.in_loop.swap(true, Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "this Tracking already runs in a loop",
            ));
        }
        let ready = self
            .ctrl
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .restart_for_loop();
        if let Err(e) = ready {
            self.leave_loop();
            return Err(PyRuntimeError::new_err(e.to_string()));
        }
        Ok(LoopTracking(Arc::clone(&self.ctrl)))
    }

    /// The loop clock now, None before the first tick. In a loop whose last tick is older
    /// than `stale_s`, an error: the loop no longer runs this controller.
    fn live_now(&self) -> PyResult<Option<f64>> {
        match self.shared.clock() {
            Some((_, age)) if age > self.stale_s && self.in_loop.load(Ordering::SeqCst) => {
                Err(PyRuntimeError::new_err(format!(
                    "the loop is not running this Tracking: its last tick was {age:.3} s ago"
                )))
            }
            clock => Ok(clock.map(|(now, _)| now)),
        }
    }

    /// Undo `enter_loop` after a failed start.
    pub(crate) fn leave_loop(&self) {
        self.in_loop.store(false, Ordering::SeqCst);
    }
}

#[pymethods]
impl Tracking {
    #[new]
    #[pyo3(signature = (dynamics, params = None, profile = None, residual = None))]
    fn new(
        dynamics: &Dynamics,
        params: Option<&Bound<'_, PyDict>>,
        profile: Option<&Bound<'_, PyDict>>,
        residual: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let config = match profile {
            Some(p) => crate::config_from_dict(p)?,
            None => ArmConfig::maker_arm_v1(),
        };
        let params = match params {
            Some(d) => params_from_dict(d)?,
            None => TrackingParams::default(),
        };
        let residual = residual.map(residual_from_dict).transpose()?;
        let ctrl = TrackingController::new(dynamics.inner.clone(), params, &config, residual)
            .map_err(value_err)?;
        let shared = ctrl.shared();
        Ok(Tracking {
            ctrl: Arc::new(Mutex::new(ctrl)),
            shared,
            in_loop: AtomicBool::new(false),
            stale_s: f64::from(config.motor_can_timeout_ms) / 1000.0,
        })
    }

    /// Row k of `targets` (six joints and the gripper) is due at `t0 + k dt` on the loop
    /// clock. Replaces the plan from `t0` on. In a loop, raises RuntimeError once the loop no
    /// longer runs this controller (see `now`).
    fn push(&self, t0: f64, dt: f64, targets: Vec<Row>) -> PyResult<()> {
        let p = Push::new(t0, dt, targets).map_err(value_err)?;
        self.live_now()?;
        self.shared.push(p);
        Ok(())
    }

    /// One tick driven from Python (the bench). `q`, `dq` and `tau` hold all seven motors, and
    /// `temperature` (degC, all seven, default 0) is what a residual reads. Returns seven
    /// `(pos, vel, kp, kd, tau)` tuples before the clamp.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (t, q, dq, tau, temperature = None))]
    fn update(
        &self,
        t: f64,
        q: Row,
        dq: Row,
        tau: Row,
        temperature: Option<Row>,
    ) -> PyResult<Vec<(f64, f64, f64, f64, f64)>> {
        if self.in_loop.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "this Tracking runs in an Arm loop now; update is not available",
            ));
        }
        if !t.is_finite() {
            return Err(PyValueError::new_err("t must be finite"));
        }
        if q.iter().chain(&dq).chain(&tau).any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("q, dq and tau must be finite"));
        }
        let temperature = temperature.unwrap_or([0.0; ROW]);
        if temperature.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("temperature must be finite"));
        }
        let state = ArmState {
            motors: (0..ROW)
                .map(|j| MotorState {
                    position: q[j],
                    velocity: dq[j],
                    torque: tau[j],
                    temperature: temperature[j],
                    feedback_age: 0.0,
                    ..MotorState::stale()
                })
                .collect(),
            tick: 0,
            t,
        };
        let cmd = self
            .ctrl
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .update(&state, 0.0);
        Ok(cmd
            .iter()
            .map(|c| (c.pos, c.vel, c.kp, c.kd, c.tau))
            .collect())
    }

    /// The loop clock now: the last tick's `t` plus the wall time since that tick. In a loop
    /// that is seconds since connect. On the bench (`update` from Python) it is the caller's
    /// last `t` plus wall time, so it mixes the caller's sim time with wall time.
    ///
    /// None before the first tick, and None in a loop whose last tick is older than the
    /// motors' CAN_TIMEOUT (0.2 s): the loop is stopped, fault-holding or in `hold_now`, and
    /// no longer runs this controller.
    fn now(&self) -> Option<f64> {
        self.live_now().ok().flatten()
    }

    /// The telemetry of every tick since the last drain as columns, oldest first: `tick_t`,
    /// `gate`, the cumulative `residual_skips`, and `r`, `offset`, `q_r`, `tau_ff`, `tau_dob` and
    /// `tau_res` as one six-wide list per tick.
    /// `dropped` counts the ticks lost since the last drain because the ring held only 4000, or
    /// because a controller was dropped with rows still queued. Join with `Arm.drain` on
    /// `tick_t` and `t`, not on the index: a tick that faulted the arm has a row in one and not
    /// the other.
    fn drain(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let (rows, dropped) = self.shared.drain();
        let d = PyDict::new(py);
        d.set_item("tick_t", rows.iter().map(|t| t.tick_t).collect::<Vec<_>>())?;
        d.set_item("r", rows.iter().map(|t| t.r).collect::<Vec<_>>())?;
        d.set_item("gate", rows.iter().map(|t| t.gate).collect::<Vec<_>>())?;
        d.set_item("offset", rows.iter().map(|t| t.offset).collect::<Vec<_>>())?;
        d.set_item("q_r", rows.iter().map(|t| t.q_r).collect::<Vec<_>>())?;
        d.set_item("tau_ff", rows.iter().map(|t| t.tau_ff).collect::<Vec<_>>())?;
        d.set_item(
            "tau_dob",
            rows.iter().map(|t| t.tau_dob).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "tau_res",
            rows.iter().map(|t| t.tau_res).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "residual_skips",
            rows.iter().map(|t| t.residual_skips).collect::<Vec<_>>(),
        )?;
        d.set_item("dropped", dropped)?;
        Ok(d.into())
    }

    fn telemetry(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(t) = self.shared.telemetry() else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("r", t.r)?;
        d.set_item("gate", t.gate)?;
        d.set_item("offset", t.offset)?;
        d.set_item("q_r", t.q_r)?;
        d.set_item("tau_ff", t.tau_ff)?;
        d.set_item("tau_dob", t.tau_dob)?;
        d.set_item("tau_res", t.tau_res)?;
        d.set_item("residual_skips", t.residual_skips)?;
        d.set_item("late_ticks", t.late_ticks)?;
        d.set_item("gaps", t.gaps)?;
        d.set_item("max_interval", t.max_interval)?;
        d.set_item("tick_t", t.tick_t)?;
        Ok(Some(d.into()))
    }
}
