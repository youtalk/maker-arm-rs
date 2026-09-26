//! `Dynamics`: the arm's rigid-body model (`maker_arm::dynamics`) behind a thin Python
//! surface. Plain Python data in and out, like `Kinematics`: the chain in the same
//! `(xyz, rpy, axis)` form, one `(mass, com, inertia)` tuple per body.

use maker_arm::dynamics::{Body, Dynamics as Inner, Friction, JointMatrix};
use maker_arm::kinematics::{Joints, Link, Mat3, Vec3, ARM_DOF};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type BodyArg = (f64, Vec3, Mat3);
type FrictionArg = (f64, f64, f64);

fn six<T>(items: Vec<T>, what: &str) -> PyResult<[T; ARM_DOF]> {
    let n = items.len();
    items
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("expected {ARM_DOF} {what}, got {n}")))
}

#[pyclass(frozen)]
pub struct Dynamics {
    pub(crate) inner: Inner,
}

#[pymethods]
impl Dynamics {
    /// `links` as for `Kinematics`; `bodies` six `(mass, com, inertia)` tuples in the frame
    /// after each revolute joint; `friction` six `(coulomb, viscous, eps)` tuples or None.
    #[new]
    #[pyo3(signature = (mount_rpy, links, bodies, friction = None, gravity = [0.0, 0.0, -9.81]))]
    fn new(
        mount_rpy: Vec3,
        links: Vec<(Vec3, Vec3, Option<Vec3>)>,
        bodies: Vec<BodyArg>,
        friction: Option<Vec<FrictionArg>>,
        gravity: Vec3,
    ) -> PyResult<Self> {
        let links: Vec<Link> = links
            .into_iter()
            .map(|(xyz, rpy, axis)| Link::new(xyz, rpy, axis))
            .collect();
        let bodies = six(
            bodies
                .into_iter()
                .map(|(mass, com, inertia)| Body { mass, com, inertia })
                .collect(),
            "bodies",
        )?;
        let friction = match friction {
            None => [Friction::default(); ARM_DOF],
            Some(f) => six(
                f.into_iter()
                    .map(|(coulomb, viscous, eps)| Friction {
                        coulomb,
                        viscous,
                        eps,
                    })
                    .collect(),
                "friction tuples",
            )?,
        };
        Inner::new(mount_rpy, &links, bodies, friction, gravity)
            .map(|inner| Dynamics { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn inverse(&self, q: Joints, dq: Joints, ddq: Joints) -> Joints {
        self.inner.inverse(&q, &dq, &ddq)
    }

    fn gravity(&self, q: Joints) -> Joints {
        self.inner.gravity(&q)
    }

    fn coriolis(&self, q: Joints, dq: Joints) -> Joints {
        self.inner.coriolis(&q, &dq)
    }

    fn mass_matrix(&self, q: Joints) -> JointMatrix {
        self.inner.mass_matrix(&q)
    }

    fn friction(&self, dq: Joints) -> Joints {
        self.inner.friction(&dq)
    }
}
