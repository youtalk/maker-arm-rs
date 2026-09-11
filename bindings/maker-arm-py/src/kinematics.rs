//! `Kinematics`: the lab's global grasp-pose solver (`maker_arm::kinematics`) behind a thin
//! Python surface. Plain Python data in and out -- tuples and lists for vectors, a dict per
//! solution -- so the lab keeps its dataclasses and torch tensors on its own side. A pose
//! read back with `pose_of` has the same shape as a target, so it can be solved for directly.

use maker_arm::kinematics::{
    Chain, GraspTarget, Joints, KinematicsError, Link, Mat3, Solution, Vec3, ANG_TOL, FD_EPS,
    MAX_ITER, POS_TOL,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `(position, tilt, jaw_heading, lean_azimuth)`, the shape `pose_of` returns.
type Target = (Vec3, f64, f64, f64);

fn target(t: Target) -> GraspTarget {
    GraspTarget {
        position: t.0,
        tilt: t.1,
        jaw_heading: t.2,
        lean_azimuth: t.3,
    }
}

fn solution_dict(py: Python<'_>, s: &Solution) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("q", s.q)?;
    d.set_item("margin", s.margin)?;
    d.set_item("position_error", s.position_error)?;
    d.set_item("tilt_error", s.tilt_error)?;
    d.set_item("heading_error", s.heading_error)?;
    d.set_item("azimuth_error", s.azimuth_error)?;
    d.set_item("step", s.step)?;
    Ok(d.into())
}

fn value_err(e: KinematicsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyclass(frozen)]
pub struct Kinematics {
    chain: Chain,
}

#[pymethods]
impl Kinematics {
    #[classattr]
    const POS_TOL: f64 = POS_TOL;
    #[classattr]
    const ANG_TOL: f64 = ANG_TOL;
    #[classattr]
    const FD_EPS: f64 = FD_EPS;
    #[classattr]
    const MAX_ITER: usize = MAX_ITER;

    /// `links` are `(xyz, rpy, axis)` per URDF joint from the mount to the tool, `axis`
    /// `None` for a fixed joint; exactly six revolute joints are required.
    #[new]
    #[pyo3(signature = (mount_rpy, links, lower, upper, tool_axis, jaw_axis))]
    fn new(
        mount_rpy: Vec3,
        links: Vec<(Vec3, Vec3, Option<Vec3>)>,
        lower: Joints,
        upper: Joints,
        tool_axis: Vec3,
        jaw_axis: Vec3,
    ) -> PyResult<Self> {
        let links = links
            .into_iter()
            .map(|(xyz, rpy, axis)| Link::new(xyz, rpy, axis))
            .collect();
        let chain =
            Chain::new(mount_rpy, links, lower, upper, tool_axis, jaw_axis).map_err(value_err)?;
        Ok(Kinematics { chain })
    }

    fn fk(&self, q: Joints) -> (Vec3, Mat3) {
        self.chain.fk(&q)
    }

    fn fk_batch(&self, q: Vec<Joints>) -> (Vec<Vec3>, Vec<Mat3>) {
        q.iter().map(|q| self.chain.fk(q)).unzip()
    }

    fn pose_of(&self, q: Joints) -> Target {
        let p = self.chain.pose_of(&q);
        (p.position, p.tilt, p.jaw_heading, p.lean_azimuth)
    }

    fn margins(&self, q: Joints) -> Joints {
        self.chain.margins(&q)
    }

    fn residual(&self, q: Joints, target: Target) -> Joints {
        self.chain.residual(&q, &self::target(target))
    }

    #[pyo3(signature = (q, target, eps = FD_EPS))]
    fn residual_jacobian(&self, q: Joints, target: Target, eps: f64) -> [Joints; 6] {
        self.chain.residual_jacobian(&q, &self::target(target), eps)
    }

    #[pyo3(signature = (seed, target, max_iter = MAX_ITER))]
    fn refine(&self, seed: Joints, target: Target, max_iter: usize) -> Joints {
        self.chain.refine(&seed, &self::target(target), max_iter)
    }

    #[pyo3(signature = (target, seeds, q_prev = None, max_step = None))]
    fn solve(
        &self,
        py: Python<'_>,
        target: Target,
        seeds: Vec<Joints>,
        q_prev: Option<Joints>,
        max_step: Option<f64>,
    ) -> PyResult<Option<Py<PyDict>>> {
        let t = self::target(target);
        let solution = py.allow_threads(|| self.chain.solve(&t, &seeds, q_prev.as_ref(), max_step));
        solution.map(|s| solution_dict(py, &s)).transpose()
    }

    fn solve_path(
        &self,
        py: Python<'_>,
        targets: Vec<Target>,
        seed: Joints,
        spread: Vec<Joints>,
        max_step: f64,
    ) -> PyResult<Option<Vec<Py<PyDict>>>> {
        let targets: Vec<GraspTarget> = targets.into_iter().map(self::target).collect();
        let path = py.allow_threads(|| self.chain.solve_path(&targets, &seed, &spread, max_step));
        path.map(|p| p.iter().map(|s| solution_dict(py, s)).collect())
            .transpose()
    }
}
