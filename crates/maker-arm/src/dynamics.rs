//! Rigid-body dynamics of the six arm joints (tracking design, 2026-09-25, section 5):
//! inverse dynamics by recursive Newton-Euler, gravity, Coriolis and centrifugal torques,
//! the mass matrix and a joint friction model, over the same `Link` chain the kinematics
//! use. Pure `f64` and `std` with no allocation per call. The caller supplies every
//! inertial, so this module holds no robot data of its own.
//!
//! Body i lives in the frame after revolute joint i: its center of mass and its inertia
//! tensor (about the center of mass) are expressed there. Fixed joints before a revolute
//! joint fold into its origin; fixed joints after the last one carry no body and are
//! ignored.

use crate::kinematics::{
    axis_rotation, mat_mul, mat_vec, rpy_matrix, Joints, Link, Mat3, Vec3, ARM_DOF,
};
use std::fmt;

pub type JointMatrix = [[f64; ARM_DOF]; ARM_DOF];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub mass: f64,
    /// Center of mass (m), in the frame after the body's joint.
    pub com: Vec3,
    /// Inertia tensor about the center of mass (kg m^2), same frame.
    pub inertia: Mat3,
}

/// `F(dq) = coulomb * tanh(dq / eps) + viscous * dq`. All zero until MA2 fits them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Friction {
    pub coulomb: f64,
    pub viscous: f64,
    pub eps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynamicsError {
    WrongDof { got: usize },
    NonFinite { what: &'static str, index: usize },
    ZeroAxis { joint: usize },
    Mass { body: usize },
    Inertia { body: usize, reason: &'static str },
    Friction { joint: usize },
}

impl fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynamicsError::WrongDof { got } => {
                write!(f, "chain has {got} revolute joints, expected {ARM_DOF}")
            }
            DynamicsError::NonFinite { what, index } => {
                write!(f, "{what} {index} has a non-finite number")
            }
            DynamicsError::ZeroAxis { joint } => write!(f, "joint {joint} has a zero axis"),
            DynamicsError::Mass { body } => write!(f, "body {body} has a non-positive mass"),
            DynamicsError::Inertia { body, reason } => {
                write!(f, "body {body} inertia is {reason}")
            }
            DynamicsError::Friction { joint } => write!(
                f,
                "joint {joint} friction needs coulomb, viscous and eps >= 0, and eps > 0 \
                 when coulomb > 0"
            ),
        }
    }
}

impl std::error::Error for DynamicsError {}

const ZERO: Vec3 = [0.0; 3];
const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// Relative tolerance of the symmetry and triangle-inequality checks.
const INERTIA_TOL: f64 = 1e-9;

/// One revolute joint: the fixed rotation and origin from the previous joint's frame (or
/// the mount's parent), and the unit axis.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Joint {
    rot: Mat3,
    pos: Vec3,
    axis: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dynamics {
    joints: [Joint; ARM_DOF],
    bodies: [Body; ARM_DOF],
    friction: [Friction; ARM_DOF],
    gravity: Vec3,
}

fn add(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: &Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `m^T v`: takes a vector from the parent frame into the child frame.
fn mat_t_vec(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Sylvester's criterion on the leading minors of a symmetric 3x3 matrix.
fn positive_definite(m: &Mat3) -> bool {
    let d2 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let d3 = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    m[0][0] > 0.0 && d2 > 0.0 && d3 > 0.0
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn check_body(i: usize, b: &Body) -> Result<(), DynamicsError> {
    if !b.mass.is_finite() || !all_finite(&b.com) || !all_finite(b.inertia.as_flattened()) {
        return Err(DynamicsError::NonFinite {
            what: "body",
            index: i,
        });
    }
    if b.mass <= 0.0 {
        return Err(DynamicsError::Mass { body: i });
    }
    let m = &b.inertia;
    let size = m.as_flattened().iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    // Cross-indexed (m[r][c] vs m[c][r]): an iterator adapter would not read more clearly.
    #[allow(clippy::needless_range_loop)]
    for r in 0..3 {
        for c in 0..3 {
            if (m[r][c] - m[c][r]).abs() > INERTIA_TOL * size {
                return Err(DynamicsError::Inertia {
                    body: i,
                    reason: "not symmetric",
                });
            }
        }
    }
    if !positive_definite(m) {
        return Err(DynamicsError::Inertia {
            body: i,
            reason: "not positive definite",
        });
    }
    // The principal moments obey the triangle inequality exactly when tr(I)/2 - I is
    // positive semidefinite; the tolerance admits the equality (a lamina).
    let half = (m[0][0] + m[1][1] + m[2][2]) / 2.0;
    let mut s = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            s[r][c] = f64::from(u8::from(r == c)) * half * (1.0 + INERTIA_TOL) - m[r][c];
        }
    }
    if !positive_definite(&s) {
        return Err(DynamicsError::Inertia {
            body: i,
            reason: "breaking the triangle inequality",
        });
    }
    Ok(())
}

fn check_friction(j: usize, f: &Friction) -> Result<(), DynamicsError> {
    let ok = all_finite(&[f.coulomb, f.viscous, f.eps])
        && f.coulomb >= 0.0
        && f.viscous >= 0.0
        && f.eps >= 0.0
        && (f.coulomb == 0.0 || f.eps > 0.0);
    if ok {
        Ok(())
    } else {
        Err(DynamicsError::Friction { joint: j })
    }
}

impl Dynamics {
    pub fn new(
        mount_rpy: Vec3,
        links: &[Link],
        bodies: [Body; ARM_DOF],
        friction: [Friction; ARM_DOF],
        gravity: Vec3,
    ) -> Result<Dynamics, DynamicsError> {
        let got = links.iter().filter(|l| l.axis.is_some()).count();
        if got != ARM_DOF {
            return Err(DynamicsError::WrongDof { got });
        }
        if !all_finite(&mount_rpy) {
            return Err(DynamicsError::NonFinite {
                what: "mount",
                index: 0,
            });
        }
        if !all_finite(&gravity) {
            return Err(DynamicsError::NonFinite {
                what: "gravity",
                index: 0,
            });
        }
        for (i, l) in links.iter().enumerate() {
            let axis_ok = l.axis.is_none_or(|a| all_finite(&a));
            if !all_finite(&l.xyz) || !all_finite(l.rot.as_flattened()) || !axis_ok {
                return Err(DynamicsError::NonFinite {
                    what: "link",
                    index: i,
                });
            }
        }
        for (i, b) in bodies.iter().enumerate() {
            check_body(i, b)?;
        }
        for (j, f) in friction.iter().enumerate() {
            check_friction(j, f)?;
        }
        let mut joints = [Joint {
            rot: IDENTITY,
            pos: ZERO,
            axis: ZERO,
        }; ARM_DOF];
        let mut rot = rpy_matrix(mount_rpy[0], mount_rpy[1], mount_rpy[2]);
        let mut pos = ZERO;
        let mut index = 0;
        for link in links {
            if index == ARM_DOF {
                break;
            }
            pos = add(&pos, &mat_vec(&rot, &link.xyz));
            rot = mat_mul(&rot, &link.rot);
            if let Some(axis) = link.axis {
                let norm = dot(&axis, &axis).sqrt();
                if norm <= 0.0 {
                    return Err(DynamicsError::ZeroAxis { joint: index });
                }
                joints[index] = Joint {
                    rot,
                    pos,
                    axis: scale(&axis, 1.0 / norm),
                };
                rot = IDENTITY;
                pos = ZERO;
                index += 1;
            }
        }
        Ok(Dynamics {
            joints,
            bodies,
            friction,
            gravity,
        })
    }

    /// Recursive Newton-Euler, every quantity in its own body frame. The base accelerates
    /// against `gravity`, which puts every body's weight into the forward pass.
    fn rnea(&self, q: &Joints, dq: &Joints, ddq: &Joints, gravity: &Vec3) -> Joints {
        let mut rot = [IDENTITY; ARM_DOF];
        let mut force = [ZERO; ARM_DOF];
        let mut moment = [ZERO; ARM_DOF];
        let (mut w, mut wd, mut a) = (ZERO, ZERO, scale(gravity, -1.0));
        for i in 0..ARM_DOF {
            let j = &self.joints[i];
            let r = mat_mul(&j.rot, &axis_rotation(&j.axis, q[i]));
            let w_in = mat_t_vec(&r, &w);
            let spin = scale(&j.axis, dq[i]);
            let wi = add(&w_in, &spin);
            let wdi = add(
                &add(&mat_t_vec(&r, &wd), &scale(&j.axis, ddq[i])),
                &cross(&w_in, &spin),
            );
            let ai = mat_t_vec(
                &r,
                &add(
                    &add(&a, &cross(&wd, &j.pos)),
                    &cross(&w, &cross(&w, &j.pos)),
                ),
            );
            let b = &self.bodies[i];
            let ac = add(
                &add(&ai, &cross(&wdi, &b.com)),
                &cross(&wi, &cross(&wi, &b.com)),
            );
            force[i] = scale(&ac, b.mass);
            moment[i] = add(
                &mat_vec(&b.inertia, &wdi),
                &cross(&wi, &mat_vec(&b.inertia, &wi)),
            );
            rot[i] = r;
            w = wi;
            wd = wdi;
            a = ai;
        }
        let mut tau = [0.0; ARM_DOF];
        // The child's force and moment, already in this frame, and the child's origin here.
        let (mut f_c, mut n_c, mut p_c) = (ZERO, ZERO, ZERO);
        for i in (0..ARM_DOF).rev() {
            let f = add(&force[i], &f_c);
            let n = add(
                &add(&moment[i], &n_c),
                &add(&cross(&self.bodies[i].com, &force[i]), &cross(&p_c, &f_c)),
            );
            tau[i] = dot(&n, &self.joints[i].axis);
            f_c = mat_vec(&rot[i], &f);
            n_c = mat_vec(&rot[i], &n);
            p_c = self.joints[i].pos;
        }
        tau
    }

    /// Joint torques for the motion (q, dq, ddq), gravity included, friction not.
    pub fn inverse(&self, q: &Joints, dq: &Joints, ddq: &Joints) -> Joints {
        self.rnea(q, dq, ddq, &self.gravity)
    }

    /// G(q): the torque that holds the arm still against gravity.
    pub fn gravity(&self, q: &Joints) -> Joints {
        self.rnea(q, &[0.0; ARM_DOF], &[0.0; ARM_DOF], &self.gravity)
    }

    /// C(q, dq) dq: the Coriolis and centrifugal torques, without gravity.
    pub fn coriolis(&self, q: &Joints, dq: &Joints) -> Joints {
        self.rnea(q, dq, &[0.0; ARM_DOF], &ZERO)
    }

    /// M(q) by columns: column j is the torque a unit acceleration of joint j needs at rest
    /// without gravity. Six RNEA passes; the tracking loop budget allows it.
    pub fn mass_matrix(&self, q: &Joints) -> JointMatrix {
        let mut m = [[0.0; ARM_DOF]; ARM_DOF];
        for j in 0..ARM_DOF {
            let mut e = [0.0; ARM_DOF];
            e[j] = 1.0;
            let col = self.rnea(q, &[0.0; ARM_DOF], &e, &ZERO);
            for i in 0..ARM_DOF {
                m[i][j] = col[i];
            }
        }
        m
    }

    pub fn friction(&self, dq: &Joints) -> Joints {
        let mut out = [0.0; ARM_DOF];
        for j in 0..ARM_DOF {
            let f = &self.friction[j];
            let coulomb = if f.coulomb == 0.0 {
                0.0
            } else {
                f.coulomb * (dq[j] / f.eps).tanh()
            };
            out[j] = coulomb + f.viscous * dq[j];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::Link;
    use std::f64::consts::PI;

    const G: f64 = 9.81;
    const Z: Option<Vec3> = Some([0.0, 0.0, 1.0]);

    /// A body on the x axis whose inertia about z is `izz`; diag(izz/10, izz, izz) keeps
    /// the triangle inequality.
    fn rod(mass: f64, lc: f64, izz: f64) -> Body {
        Body {
            mass,
            com: [lc, 0.0, 0.0],
            inertia: [[izz / 10.0, 0.0, 0.0], [0.0, izz, 0.0], [0.0, 0.0, izz]],
        }
    }

    /// A near-massless body: fills the joints a planar test does not use.
    fn speck() -> Body {
        Body {
            mass: 1e-9,
            com: [0.0; 3],
            inertia: [[1e-12, 0.0, 0.0], [0.0, 1e-12, 0.0], [0.0, 0.0, 1e-12]],
        }
    }

    /// Two planar links about z, gravity along -y, then four specks.
    fn planar(b0: Body, b1: Body, l1: f64) -> Dynamics {
        let mut links = vec![
            Link::new([0.0; 3], [0.0; 3], Z),
            Link::new([l1, 0.0, 0.0], [0.0; 3], Z),
        ];
        links.extend((0..4).map(|_| Link::new([0.0; 3], [0.0; 3], Z)));
        Dynamics::new(
            [0.0; 3],
            &links,
            [b0, b1, speck(), speck(), speck(), speck()],
            [Friction::default(); ARM_DOF],
            [0.0, -G, 0.0],
        )
        .unwrap()
    }

    /// A Maker-Arm-like spatial chain with a fixed tool frame after the last joint.
    fn spatial() -> Dynamics {
        let links = [
            Link::new([0.0, 0.0, 0.03], [0.0; 3], Some([0.0, 0.0, 1.0])),
            Link::new(
                [0.0, 0.0, 0.06],
                [PI / 2.0, 0.0, 0.0],
                Some([0.0, 0.0, -1.0]),
            ),
            Link::new([0.19, 0.0, 0.0], [0.0; 3], Some([0.0, 0.0, -1.0])),
            Link::new([-0.17, 0.08, 0.0], [0.0; 3], Some([0.0, 0.0, -1.0])),
            Link::new(
                [-0.08, 0.03, 0.0],
                [PI / 2.0, 0.0, 0.17],
                Some([0.0, 0.0, 1.0]),
            ),
            Link::new(
                [0.0, 0.0, 0.05],
                [0.0, PI / 2.0, 0.0],
                Some([0.0, 0.0, 1.0]),
            ),
            Link::new([0.0, 0.0, -0.17], [0.0; 3], None),
        ];
        let body = |mass: f64, com: Vec3| Body {
            mass,
            com,
            inertia: [[2e-3, 1e-4, 0.0], [1e-4, 3e-3, 0.0], [0.0, 0.0, 2.5e-3]],
        };
        let bodies = [
            body(0.55, [0.0, 0.0, 0.04]),
            body(0.7, [0.13, 0.0, 0.0]),
            body(0.6, [-0.09, 0.06, 0.0]),
            body(0.4, [-0.06, 0.04, 0.0]),
            body(0.35, [0.0, 0.0, 0.04]),
            body(0.4, [0.0, 0.0, -0.08]),
        ];
        Dynamics::new(
            [PI / 2.0, 0.0, PI],
            &links,
            bodies,
            [Friction::default(); ARM_DOF],
            [0.0, 0.0, -G],
        )
        .unwrap()
    }

    fn pad(v: &[f64]) -> Joints {
        let mut out = [0.0; ARM_DOF];
        out[..v.len()].copy_from_slice(v);
        out
    }

    #[test]
    fn one_link_matches_its_closed_form() {
        let (m, lc, izz) = (2.0, 0.3, 0.05);
        let d = planar(rod(m, lc, izz), speck(), 0.5);
        let (q, dq, ddq) = (pad(&[0.7]), pad(&[1.3]), pad(&[-2.1]));
        let want = (izz + m * lc * lc) * ddq[0] + m * G * lc * q[0].cos();
        assert!((d.inverse(&q, &dq, &ddq)[0] - want).abs() < 1e-6);
        assert!((d.gravity(&q)[0] - m * G * lc * q[0].cos()).abs() < 1e-6);
    }

    #[test]
    fn two_links_match_their_closed_form() {
        let (m0, lc0, i0) = (1.5, 0.2, 0.03);
        let (m1, lc1, i1) = (1.0, 0.25, 0.02);
        let l1 = 0.4;
        let d = planar(rod(m0, lc0, i0), rod(m1, lc1, i1), l1);
        let (q, dq, ddq) = (pad(&[0.3, -0.8]), pad(&[1.1, -0.7]), pad(&[0.5, 2.0]));
        let (c1, s1) = (q[1].cos(), q[1].sin());
        let m00 = i0 + i1 + m0 * lc0 * lc0 + m1 * (l1 * l1 + lc1 * lc1 + 2.0 * l1 * lc1 * c1);
        let m01 = i1 + m1 * (lc1 * lc1 + l1 * lc1 * c1);
        let m11 = i1 + m1 * lc1 * lc1;
        let h = m1 * l1 * lc1 * s1;
        let g0 = (m0 * lc0 + m1 * l1) * G * q[0].cos() + m1 * lc1 * G * (q[0] + q[1]).cos();
        let g1 = m1 * lc1 * G * (q[0] + q[1]).cos();
        let t0 = m00 * ddq[0] + m01 * ddq[1] - h * (2.0 * dq[0] * dq[1] + dq[1] * dq[1]) + g0;
        let t1 = m01 * ddq[0] + m11 * ddq[1] + h * dq[0] * dq[0] + g1;
        let tau = d.inverse(&q, &dq, &ddq);
        assert!((tau[0] - t0).abs() < 1e-6, "{} vs {t0}", tau[0]);
        assert!((tau[1] - t1).abs() < 1e-6, "{} vs {t1}", tau[1]);
        let m = d.mass_matrix(&q);
        assert!((m[0][0] - m00).abs() < 1e-6 && (m[0][1] - m01).abs() < 1e-6);
        assert!((m[1][1] - m11).abs() < 1e-6);
    }

    /// Cholesky: true iff the symmetric matrix is positive definite.
    fn cholesky_ok(m: &JointMatrix) -> bool {
        let mut l = [[0.0; ARM_DOF]; ARM_DOF];
        for i in 0..ARM_DOF {
            for j in 0..=i {
                let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
                if i == j {
                    let v = m[i][i] - s;
                    if v <= 0.0 {
                        return false;
                    }
                    l[i][j] = v.sqrt();
                } else {
                    l[i][j] = (m[i][j] - s) / l[j][j];
                }
            }
        }
        true
    }

    #[test]
    fn mass_matrix_is_symmetric_positive_definite_and_matches_inverse_columns() {
        let d = spatial();
        let q = [0.3, -1.9, 1.6, -1.0, 0.4, -0.7];
        let m = d.mass_matrix(&q);
        // Cross-indexed (m[i][j] vs m[j][i]): an iterator adapter would not read more clearly.
        #[allow(clippy::needless_range_loop)]
        for i in 0..ARM_DOF {
            for j in 0..ARM_DOF {
                assert!((m[i][j] - m[j][i]).abs() < 1e-12);
            }
        }
        assert!(cholesky_ok(&m));
        let g = d.gravity(&q);
        for i in 0..ARM_DOF {
            let mut e = [0.0; ARM_DOF];
            e[i] = 1.0;
            let col = d.inverse(&q, &[0.0; ARM_DOF], &e);
            for k in 0..ARM_DOF {
                assert!((col[k] - g[k] - m[k][i]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn coriolis_obeys_the_skew_symmetry_of_mdot_minus_2c() {
        // dq . (Mdot dq) = 2 dq . (C dq), with Mdot along dq by central difference.
        let d = spatial();
        let q = [0.3, -1.9, 1.6, -1.0, 0.4, -0.7];
        let dq = [0.8, -1.2, 0.5, 1.5, -0.9, 2.0];
        let eps = 1e-6;
        let shift = |s: f64| {
            let mut out = q;
            for i in 0..ARM_DOF {
                out[i] += s * eps * dq[i];
            }
            out
        };
        let (mp, mm) = (d.mass_matrix(&shift(1.0)), d.mass_matrix(&shift(-1.0)));
        let c = d.coriolis(&q, &dq);
        let mut lhs = 0.0;
        for i in 0..ARM_DOF {
            for k in 0..ARM_DOF {
                lhs += dq[i] * (mp[i][k] - mm[i][k]) / (2.0 * eps) * dq[k];
            }
            lhs -= 2.0 * dq[i] * c[i];
        }
        assert!(lhs.abs() < 1e-6, "{lhs}");
    }

    #[test]
    fn friction_is_tanh_coulomb_plus_viscous() {
        let links: Vec<Link> = (0..6)
            .map(|_| Link::new([0.1, 0.0, 0.0], [0.0; 3], Z))
            .collect();
        let f = Friction {
            coulomb: 0.2,
            viscous: 0.05,
            eps: 0.01,
        };
        let d = Dynamics::new(
            [0.0; 3],
            &links,
            [rod(0.5, 0.05, 2e-3); 6],
            [f; 6],
            [0.0, 0.0, -G],
        )
        .unwrap();
        let out = d.friction(&[0.5, -0.5, 0.0, 0.001, 0.0, 0.0]);
        assert!((out[0] - (0.2 * 50.0f64.tanh() + 0.025)).abs() < 1e-12);
        assert!((out[1] + (0.2 * 50.0f64.tanh() + 0.025)).abs() < 1e-12);
        assert_eq!(out[2], 0.0);
        assert!((out[3] - (0.2 * 0.1f64.tanh() + 0.00005)).abs() < 1e-12);
    }

    #[test]
    fn the_constructor_refuses_bad_models() {
        let links: Vec<Link> = (0..6)
            .map(|_| Link::new([0.1, 0.0, 0.0], [0.0; 3], Z))
            .collect();
        let ok = rod(0.5, 0.05, 2e-3);
        let build = |bodies: [Body; 6], friction: [Friction; 6], links: &[Link]| {
            Dynamics::new([0.0; 3], links, bodies, friction, [0.0, 0.0, -G])
        };
        let nf = [Friction::default(); 6];
        assert!(build([ok; 6], nf, &links).is_ok());
        assert!(matches!(
            build([ok; 6], nf, &links[..5]),
            Err(DynamicsError::WrongDof { got: 5 })
        ));
        let mut b = [ok; 6];
        b[2].mass = -1.0;
        assert!(matches!(
            build(b, nf, &links),
            Err(DynamicsError::Mass { body: 2 })
        ));
        let mut b = [ok; 6];
        b[1].com[0] = f64::NAN;
        assert!(matches!(
            build(b, nf, &links),
            Err(DynamicsError::NonFinite { .. })
        ));
        let mut b = [ok; 6];
        b[3].inertia[0][1] = 1e-4; // not symmetric
        assert!(matches!(
            build(b, nf, &links),
            Err(DynamicsError::Inertia { body: 3, .. })
        ));
        let mut b = [ok; 6];
        b[4].inertia = [[1e-3, 0.0, 0.0], [0.0, -1e-3, 0.0], [0.0, 0.0, 1e-3]]; // not PD
        assert!(matches!(
            build(b, nf, &links),
            Err(DynamicsError::Inertia { body: 4, .. })
        ));
        let mut b = [ok; 6];
        b[5].inertia = [[1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0], [0.0, 0.0, 3e-3]]; // 1 + 1 < 3
        assert!(matches!(
            build(b, nf, &links),
            Err(DynamicsError::Inertia { body: 5, .. })
        ));
        let mut f = nf;
        f[0] = Friction {
            coulomb: 0.1,
            viscous: 0.0,
            eps: 0.0,
        };
        assert!(matches!(
            build([ok; 6], f, &links),
            Err(DynamicsError::Friction { joint: 0 })
        ));
    }
}
