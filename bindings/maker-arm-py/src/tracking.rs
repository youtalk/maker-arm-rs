//! `Tracking`: one `TrackingController` behind a thread-safe Python handle (tracking design
//! sections 9 and 10). Python drives it with `update` (the bench), or `Arm.start_tracking`
//! moves it into the 200 Hz loop; either way `push`, `now` and `telemetry` stay reachable
//! from any Python thread, and the loop thread never takes the GIL.

use crate::dynamics::Dynamics;
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
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown tracking parameter {other}"
                )))
            }
        }
    }
    Ok(p)
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

#[pyclass(frozen)]
pub struct Tracking {
    pub(crate) ctrl: Arc<Mutex<TrackingController>>,
    shared: Arc<maker_arm::tracking::TrackingShared>,
    pub(crate) in_loop: AtomicBool,
}

impl Tracking {
    /// The controller for `Arm.start_tracking`, restarted so its first loop tick begins from
    /// the measured pose. Marks the handle as running in a loop.
    pub(crate) fn enter_loop(&self) -> PyResult<LoopTracking> {
        if self.in_loop.swap(true, Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "this Tracking already runs in a loop",
            ));
        }
        self.ctrl
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .restart();
        Ok(LoopTracking(Arc::clone(&self.ctrl)))
    }

    /// Undo `enter_loop` after a failed start.
    pub(crate) fn leave_loop(&self) {
        self.in_loop.store(false, Ordering::SeqCst);
    }
}

#[pymethods]
impl Tracking {
    #[new]
    #[pyo3(signature = (dynamics, params = None, profile = None))]
    fn new(
        dynamics: &Dynamics,
        params: Option<&Bound<'_, PyDict>>,
        profile: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let config = match profile {
            Some(p) => crate::config_from_dict(p)?,
            None => ArmConfig::maker_arm_v1(),
        };
        let params = match params {
            Some(d) => params_from_dict(d)?,
            None => TrackingParams::default(),
        };
        let ctrl =
            TrackingController::new(dynamics.inner.clone(), params, &config).map_err(value_err)?;
        let shared = ctrl.shared();
        Ok(Tracking {
            ctrl: Arc::new(Mutex::new(ctrl)),
            shared,
            in_loop: AtomicBool::new(false),
        })
    }

    /// Row k of `targets` (six joints and the gripper) is due at `t0 + k dt` on the loop
    /// clock. Replaces the plan from `t0` on.
    fn push(&self, t0: f64, dt: f64, targets: Vec<Row>) -> PyResult<()> {
        self.shared
            .push(Push::new(t0, dt, targets).map_err(value_err)?);
        Ok(())
    }

    /// One tick driven from Python (the bench). `q`, `dq` and `tau` hold all seven motors.
    /// Returns seven `(pos, vel, kp, kd, tau)` tuples before the clamp.
    #[allow(clippy::type_complexity)]
    fn update(
        &self,
        t: f64,
        q: Row,
        dq: Row,
        tau: Row,
    ) -> PyResult<Vec<(f64, f64, f64, f64, f64)>> {
        if self.in_loop.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "this Tracking runs in an Arm loop now; update is not available",
            ));
        }
        if !t.is_finite() {
            return Err(PyValueError::new_err("t must be finite"));
        }
        let state = ArmState {
            motors: (0..ROW)
                .map(|j| MotorState {
                    position: q[j],
                    velocity: dq[j],
                    torque: tau[j],
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

    /// The loop clock (seconds since connect) now, or None before the first tick.
    fn now(&self) -> Option<f64> {
        self.shared.now()
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
        d.set_item("late_ticks", t.late_ticks)?;
        d.set_item("gaps", t.gaps)?;
        d.set_item("max_interval", t.max_interval)?;
        Ok(Some(d.into()))
    }
}
