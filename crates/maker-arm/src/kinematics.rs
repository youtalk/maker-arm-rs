//! Forward kinematics and the global grasp-pose solver for the six arm joints.
//!
//! A port of the lab's `core/global_ik.py`: Levenberg-Marquardt on a six-scalar grasp
//! residual with a central-difference Jacobian, multi-seed branch selection, and
//! continuity-constrained paths. It moved here because the planner spent about 80 % of its
//! time in torch dispatch overhead on 3x3 matrices. Pure `f64` and `std`; the caller supplies
//! the chain (URDF joint origins and axes), the joint box, and the body-frame tool and jaw
//! axes, so this module holds no robot geometry of its own.

use std::fmt;

pub const ARM_DOF: usize = 6;
pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];
pub type Joints = [f64; ARM_DOF];

/// One joint of the chain: the URDF `<origin>` (as a translation and a rotation matrix) and
/// the revolute axis in the joint frame, or `None` for a fixed joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub xyz: Vec3,
    pub rot: Mat3,
    pub axis: Option<Vec3>,
}

impl Link {
    pub fn new(xyz: Vec3, rpy: Vec3, axis: Option<Vec3>) -> Link {
        Link {
            xyz,
            rot: rpy_matrix(rpy[0], rpy[1], rpy[2]),
            axis,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    /// Rotation applied before the first link (the Z-up mount).
    pub mount: Mat3,
    pub links: Vec<Link>,
    pub lower: Joints,
    pub upper: Joints,
    /// Unit vectors in the tool frame: the axis the jaws close along is `jaw_axis`, the axis
    /// pointing out of the tool is `tool_axis`.
    pub tool_axis: Vec3,
    pub jaw_axis: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KinematicsError {
    WrongDof { got: usize },
    EmptyBox { joint: usize },
}

impl fmt::Display for KinematicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KinematicsError::WrongDof { got } => {
                write!(f, "chain has {got} revolute joints, expected {ARM_DOF}")
            }
            KinematicsError::EmptyBox { joint } => {
                write!(f, "joint {joint} has lower > upper (or a NaN bound)")
            }
        }
    }
}

impl std::error::Error for KinematicsError {}

impl Chain {
    pub fn new(
        mount_rpy: Vec3,
        links: Vec<Link>,
        lower: Joints,
        upper: Joints,
        tool_axis: Vec3,
        jaw_axis: Vec3,
    ) -> Result<Chain, KinematicsError> {
        let got = links.iter().filter(|l| l.axis.is_some()).count();
        if got != ARM_DOF {
            return Err(KinematicsError::WrongDof { got });
        }
        if let Some(joint) =
            (0..ARM_DOF).find(|&j| lower[j] > upper[j] || lower[j].is_nan() || upper[j].is_nan())
        {
            return Err(KinematicsError::EmptyBox { joint });
        }
        Ok(Chain {
            mount: rpy_matrix(mount_rpy[0], mount_rpy[1], mount_rpy[2]),
            links,
            lower,
            upper,
            tool_axis,
            jaw_axis,
        })
    }

    /// Position and rotation of the last link's frame in the mount's parent frame.
    pub fn fk(&self, q: &Joints) -> (Vec3, Mat3) {
        let mut rot = self.mount;
        let mut pos = [0.0; 3];
        let mut index = 0;
        for link in &self.links {
            let step = mat_vec(&rot, &link.xyz);
            for k in 0..3 {
                pos[k] += step[k];
            }
            rot = mat_mul(&rot, &link.rot);
            if let Some(axis) = link.axis {
                rot = mat_mul(&rot, &axis_rotation(&axis, q[index]));
                index += 1;
            }
        }
        (pos, rot)
    }

    /// Per-joint distance to the nearer stop (rad). Negative means past it.
    pub fn margins(&self, q: &Joints) -> Joints {
        let mut out = [0.0; ARM_DOF];
        for j in 0..ARM_DOF {
            out[j] = (q[j] - self.lower[j]).min(self.upper[j] - q[j]);
        }
        out
    }
}

pub fn rpy_matrix(roll: f64, pitch: f64, yaw: f64) -> Mat3 {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

/// Rodrigues' rotation about `axis` (normalised here) by `angle`.
pub fn axis_rotation(axis: &Vec3, angle: f64) -> Mat3 {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let u = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let skew = [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]];
    let skew2 = mat_mul(&skew, &skew);
    let (s, c) = angle.sin_cos();
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = f64::from(i == j) + s * skew[i][j] + (1.0 - c) * skew2[i][j];
        }
    }
    out
}

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

pub fn mat_vec(a: &Mat3, v: &Vec3) -> Vec3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

pub const POS_TOL: f64 = 1e-5;
pub const ANG_TOL: f64 = 0.05 * std::f64::consts::PI / 180.0;
/// Central-difference step of the residual Jacobian (rad).
pub const FD_EPS: f64 = 1e-7;
pub const MAX_ITER: usize = 120;

/// The six scalars the grasp task constrains, in the mount's parent frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraspTarget {
    pub position: Vec3,
    /// Angle of the tool axis off world -z (rad), magnitude only.
    pub tilt: f64,
    /// World heading of the jaw closing axis (rad), taken modulo pi.
    pub jaw_heading: f64,
    /// World azimuth the tool leans toward (rad).
    pub lean_azimuth: f64,
}

/// The same four quantities read back off a joint vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub tilt: f64,
    pub jaw_heading: f64,
    pub lean_azimuth: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solution {
    pub q: Joints,
    /// Distance from its nearer limit of the tightest joint (rad).
    pub margin: f64,
    pub position_error: f64,
    pub tilt_error: f64,
    pub heading_error: f64,
    pub azimuth_error: f64,
    /// Inf-norm joint distance from the reference pose continuity was asked against.
    pub step: f64,
}

/// Fold an angle into [-pi, pi); exactly +pi comes back as -pi.
pub fn wrap_to_pi(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

/// Fold an angular error modulo pi: a jaw is symmetric under a half turn.
pub fn fold_to_half_turn(angle: f64) -> f64 {
    wrap_to_pi(2.0 * angle) / 2.0
}

fn dot6(a: &Joints, b: &Joints) -> f64 {
    (0..ARM_DOF).map(|i| a[i] * b[i]).sum()
}

/// Gaussian elimination with partial pivoting; `None` on an exactly zero pivot, which is the
/// singular case LAPACK's `gesv` reports (torch raised on it, and the refine loop stopped).
pub fn solve6(a: &[[f64; ARM_DOF]; ARM_DOF], b: &Joints) -> Option<Joints> {
    let mut m = *a;
    let mut x = *b;
    for col in 0..ARM_DOF {
        let mut pivot = col;
        for row in col + 1..ARM_DOF {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        if m[pivot][col] == 0.0 {
            return None;
        }
        if pivot != col {
            m.swap(pivot, col);
            x.swap(pivot, col);
        }
        let pivot_row = m[col];
        for row in col + 1..ARM_DOF {
            let factor = m[row][col] / pivot_row[col];
            if factor != 0.0 {
                for (c, v) in m[row].iter_mut().enumerate().skip(col) {
                    *v -= factor * pivot_row[c];
                }
                x[row] -= factor * x[col];
            }
        }
    }
    let mut out = [0.0; ARM_DOF];
    for row in (0..ARM_DOF).rev() {
        let mut s = x[row];
        for c in row + 1..ARM_DOF {
            s -= m[row][c] * out[c];
        }
        out[row] = s / m[row][row];
    }
    Some(out)
}

impl Chain {
    pub fn pose_of(&self, q: &Joints) -> Pose {
        let (position, rot) = self.fk(q);
        let tool = mat_vec(&rot, &self.tool_axis);
        let jaw = mat_vec(&rot, &self.jaw_axis);
        Pose {
            position,
            tilt: (-tool[2]).clamp(-1.0, 1.0).acos(),
            jaw_heading: jaw[1].atan2(jaw[0]),
            lean_azimuth: tool[1].atan2(tool[0]),
        }
    }

    /// The six task-space errors of `q` against `target`, in the order position (3), tilt,
    /// folded jaw heading, wrapped lean azimuth.
    pub fn residual(&self, q: &Joints, target: &GraspTarget) -> Joints {
        let pose = self.pose_of(q);
        [
            pose.position[0] - target.position[0],
            pose.position[1] - target.position[1],
            pose.position[2] - target.position[2],
            pose.tilt - target.tilt,
            fold_to_half_turn(pose.jaw_heading - target.jaw_heading),
            wrap_to_pi(pose.lean_azimuth - target.lean_azimuth),
        ]
    }

    /// `d(residual_i) / d(q_j)` by central differences, `[i][j]`.
    pub fn residual_jacobian(
        &self,
        q: &Joints,
        target: &GraspTarget,
        eps: f64,
    ) -> [[f64; ARM_DOF]; ARM_DOF] {
        let mut jac = [[0.0; ARM_DOF]; ARM_DOF];
        for j in 0..ARM_DOF {
            let (mut plus, mut minus) = (*q, *q);
            plus[j] += eps;
            minus[j] -= eps;
            let (rp, rm) = (self.residual(&plus, target), self.residual(&minus, target));
            for i in 0..ARM_DOF {
                jac[i][j] = (rp[i] - rm[i]) / (2.0 * eps);
            }
        }
        jac
    }

    fn clamp_box(&self, q: &Joints) -> Joints {
        let mut out = *q;
        for (j, v) in out.iter_mut().enumerate() {
            *v = v.clamp(self.lower[j], self.upper[j]);
        }
        out
    }

    /// Levenberg-Marquardt onto the constraint manifold, clamped to the joint box. The
    /// damping schedule (1e-4 start, halve on success down to 1e-12, quadruple on failure up
    /// to 1e12) and the 1e-24 cost floor are the lab's.
    pub fn refine(&self, seed: &Joints, target: &GraspTarget, max_iter: usize) -> Joints {
        let mut q = self.clamp_box(seed);
        let mut residual = self.residual(&q, target);
        let mut cost = dot6(&residual, &residual);
        let mut damping = 1e-4;
        for _ in 0..max_iter {
            if cost < 1e-24 {
                break;
            }
            let jac = self.residual_jacobian(&q, target, FD_EPS);
            let mut normal = [[0.0; ARM_DOF]; ARM_DOF];
            let mut rhs = [0.0; ARM_DOF];
            for a in 0..ARM_DOF {
                for b in 0..ARM_DOF {
                    normal[a][b] = (0..ARM_DOF).map(|i| jac[i][a] * jac[i][b]).sum();
                }
                normal[a][a] += damping;
                rhs[a] = -(0..ARM_DOF).map(|i| jac[i][a] * residual[i]).sum::<f64>();
            }
            let Some(delta) = solve6(&normal, &rhs) else {
                break; // singular normal matrix at a kinematic singularity
            };
            let mut candidate = q;
            for j in 0..ARM_DOF {
                candidate[j] += delta[j];
            }
            let candidate = self.clamp_box(&candidate);
            let trial = self.residual(&candidate, target);
            let trial_cost = dot6(&trial, &trial);
            if trial_cost < cost {
                q = candidate;
                residual = trial;
                cost = trial_cost;
                damping = (damping * 0.5).max(1e-12);
            } else {
                damping *= 4.0;
                if damping > 1e12 {
                    break;
                }
            }
        }
        q
    }

    fn grade(&self, q: &Joints, target: &GraspTarget, reference: Option<&Joints>) -> Solution {
        let r = self.residual(q, target);
        Solution {
            q: *q,
            margin: self
                .margins(q)
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            position_error: (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt(),
            tilt_error: r[3].abs(),
            heading_error: r[4].abs(),
            azimuth_error: r[5].abs(),
            step: reference.map_or(0.0, |p| {
                (0..ARM_DOF)
                    .map(|j| (q[j] - p[j]).abs())
                    .fold(0.0, f64::max)
            }),
        }
    }

    fn feasible(s: &Solution) -> bool {
        s.position_error < POS_TOL
            && s.tilt_error < ANG_TOL
            && s.heading_error < ANG_TOL
            && s.azimuth_error < ANG_TOL
            && s.margin >= 0.0
    }

    /// Refine every seed and return the feasible branch furthest from the joint stops; with
    /// `q_prev` and `max_step`, branches further than `max_step` from `q_prev` on any joint
    /// are rejected outright. `None` when nothing feasible was found.
    pub fn solve(
        &self,
        target: &GraspTarget,
        seeds: &[Joints],
        q_prev: Option<&Joints>,
        max_step: Option<f64>,
    ) -> Option<Solution> {
        let mut best: Option<Solution> = None;
        for seed in seeds {
            let s = self.grade(&self.refine(seed, target, MAX_ITER), target, q_prev);
            if !Self::feasible(&s) {
                continue;
            }
            if let (Some(cap), Some(_)) = (max_step, q_prev) {
                if s.step > cap {
                    continue;
                }
            }
            if best.is_none_or(|b| s.margin > b.margin) {
                best = Some(s);
            }
        }
        best
    }

    /// Solve a waypoint list with continuity enforced between adjacent waypoints: each
    /// waypoint is seeded from the previous solution, and only when that fails is the
    /// `spread` of restart seeds tried. `None` at the first unwalkable waypoint.
    pub fn solve_path(
        &self,
        targets: &[GraspTarget],
        seed: &Joints,
        spread: &[Joints],
        max_step: f64,
    ) -> Option<Vec<Solution>> {
        let mut out = Vec::with_capacity(targets.len());
        let mut previous = *seed;
        for target in targets {
            let mut solution = self.solve(target, &[previous], Some(&previous), Some(max_step));
            if solution.is_none() && !spread.is_empty() {
                let mut seeds = Vec::with_capacity(spread.len() + 1);
                seeds.push(previous);
                seeds.extend_from_slice(spread);
                solution = self.solve(target, &seeds, Some(&previous), Some(max_step));
            }
            let s = solution?;
            previous = s.q;
            out.push(s);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn rpy_yaw_quarter_turn_maps_x_to_y() {
        let v = mat_vec(&rpy_matrix(0.0, 0.0, FRAC_PI_2), &[1.0, 0.0, 0.0]);
        assert!(close(v[0], 0.0, 1e-15) && close(v[1], 1.0, 1e-15) && close(v[2], 0.0, 1e-15));
    }

    #[test]
    fn axis_rotation_normalises_its_axis() {
        let v = mat_vec(
            &axis_rotation(&[0.0, 0.0, 2.0], FRAC_PI_2),
            &[1.0, 0.0, 0.0],
        );
        assert!(close(v[0], 0.0, 1e-15) && close(v[1], 1.0, 1e-15));
    }

    /// Two moving links (z then y), four inert joints at the same point, one fixed tool link.
    fn planar_chain() -> Chain {
        Chain::new(
            [0.0; 3],
            vec![
                Link::new([0.0, 0.0, 0.1], [0.0; 3], Some([0.0, 0.0, 1.0])),
                Link::new([0.5, 0.0, 0.0], [0.0; 3], Some([0.0, 1.0, 0.0])),
                Link::new([0.0; 3], [0.0; 3], Some([1.0, 0.0, 0.0])),
                Link::new([0.0; 3], [0.0; 3], Some([0.0, 1.0, 0.0])),
                Link::new([0.0; 3], [0.0; 3], Some([0.0, 0.0, 1.0])),
                Link::new([0.0; 3], [0.0; 3], Some([1.0, 0.0, 0.0])),
                Link::new([0.3, 0.0, 0.0], [0.0; 3], None),
            ],
            [-3.0; ARM_DOF],
            [3.0; ARM_DOF],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn fk_walks_the_chain() {
        let (pos, rot) = planar_chain().fk(&[FRAC_PI_2, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(
            close(pos[0], 0.0, 1e-15) && close(pos[1], 0.8, 1e-15) && close(pos[2], 0.1, 1e-15)
        );
        // The tool frame is the base frame yawed a quarter turn: its x axis points along +y.
        assert!(close(rot[1][0], 1.0, 1e-15) && close(rot[0][0], 0.0, 1e-15));
    }

    #[test]
    fn fk_applies_the_mount_first() {
        let mut chain = planar_chain();
        chain.mount = rpy_matrix(0.0, 0.0, FRAC_PI_2);
        let (pos, _) = chain.fk(&[0.0; ARM_DOF]);
        assert!(
            close(pos[0], 0.0, 1e-15) && close(pos[1], 0.8, 1e-15) && close(pos[2], 0.1, 1e-15)
        );
    }

    #[test]
    fn chain_needs_exactly_six_revolute_links() {
        let err = Chain::new(
            [0.0; 3],
            vec![],
            [0.0; ARM_DOF],
            [1.0; ARM_DOF],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        assert!(matches!(err, Err(KinematicsError::WrongDof { got: 0 })));
    }

    #[test]
    fn margins_measure_the_nearer_stop() {
        let m = planar_chain().margins(&[2.5, -2.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(close(m[0], 0.5, 1e-15) && close(m[1], 1.0, 1e-15) && close(m[2], 3.0, 1e-15));
    }

    /// A generic six-joint arm: base yaw, three pitch joints, wrist yaw, wrist roll, tool.
    fn generic_chain() -> Chain {
        Chain::new(
            [0.0; 3],
            vec![
                Link::new([0.0, 0.0, 0.1], [0.0; 3], Some([0.0, 0.0, 1.0])),
                Link::new([0.0, 0.0, 0.05], [0.0; 3], Some([0.0, 1.0, 0.0])),
                Link::new([0.3, 0.0, 0.0], [0.0; 3], Some([0.0, 1.0, 0.0])),
                Link::new([0.25, 0.0, 0.0], [0.0; 3], Some([0.0, 1.0, 0.0])),
                Link::new([0.05, 0.0, 0.0], [0.0; 3], Some([0.0, 0.0, 1.0])),
                Link::new([0.05, 0.0, 0.0], [0.0; 3], Some([1.0, 0.0, 0.0])),
                Link::new([0.05, 0.0, 0.0], [0.0; 3], None),
            ],
            [-3.0; ARM_DOF],
            [3.0; ARM_DOF],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
        .unwrap()
    }

    const Q_GENERIC: Joints = [0.3, 0.3, 0.4, 0.3, 0.2, 0.1];

    fn target_at(chain: &Chain, q: &Joints) -> GraspTarget {
        let pose = chain.pose_of(q);
        GraspTarget {
            position: pose.position,
            tilt: pose.tilt,
            jaw_heading: pose.jaw_heading,
            lean_azimuth: pose.lean_azimuth,
        }
    }

    #[test]
    fn wrap_to_pi_folds_exactly_pi_to_minus_pi() {
        assert!(close(wrap_to_pi(3.0 * PI), -PI, 1e-12));
        assert!(close(wrap_to_pi(-3.0 * PI), -PI, 1e-12));
        assert!(close(wrap_to_pi(0.4), 0.4, 1e-12));
        assert!(close(wrap_to_pi(2.0 * PI + 0.4), 0.4, 1e-12));
    }

    #[test]
    fn fold_to_half_turn_is_modulo_pi() {
        assert!(close(fold_to_half_turn(PI), 0.0, 1e-12));
        assert!(close(fold_to_half_turn(0.1), 0.1, 1e-12));
        assert!(close(fold_to_half_turn(PI - 0.1), -0.1, 1e-12));
    }

    #[test]
    fn residual_vanishes_at_the_pose_it_was_read_from() {
        let chain = generic_chain();
        let target = target_at(&chain, &Q_GENERIC);
        assert!(
            target.tilt > 0.2,
            "the test pose must sit away from the vertical singularity"
        );
        let r = chain.residual(&Q_GENERIC, &target);
        assert!(r.iter().all(|v| v.abs() < 1e-12), "{r:?}");
    }

    #[test]
    fn jacobian_position_rows_match_the_screw_of_the_base_joint() {
        // Planar chain at q = 0: the tool sits at (0.8, 0, 0.1) and joint 0 is the z axis
        // through the origin, so d(pos)/d(q0) = z x (0.8, 0, 0) = (0, 0.8, 0).
        let chain = planar_chain();
        let q = [0.0; ARM_DOF];
        let jac = chain.residual_jacobian(&q, &target_at(&chain, &q), FD_EPS);
        assert!(
            close(jac[0][0], 0.0, 1e-8)
                && close(jac[1][0], 0.8, 1e-8)
                && close(jac[2][0], 0.0, 1e-8)
        );
    }

    #[test]
    fn jacobian_agrees_with_a_coarser_central_difference() {
        let chain = generic_chain();
        let target = target_at(&chain, &Q_GENERIC);
        let jac = chain.residual_jacobian(&Q_GENERIC, &target, FD_EPS);
        let eps = 1e-5;
        for j in 0..ARM_DOF {
            let (mut plus, mut minus) = (Q_GENERIC, Q_GENERIC);
            plus[j] += eps;
            minus[j] -= eps;
            let (rp, rm) = (
                chain.residual(&plus, &target),
                chain.residual(&minus, &target),
            );
            for i in 0..ARM_DOF {
                assert!(
                    close(jac[i][j], (rp[i] - rm[i]) / (2.0 * eps), 1e-6),
                    "J[{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn solve6_inverts_a_well_posed_system_and_refuses_a_singular_one() {
        let a = [
            [4.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 5.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 6.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 7.0],
        ];
        let b = [1.0, -2.0, 3.0, 0.5, 0.0, 4.0];
        let x = solve6(&a, &b).unwrap();
        for i in 0..ARM_DOF {
            let ax: f64 = (0..ARM_DOF).map(|j| a[i][j] * x[j]).sum();
            assert!(close(ax, b[i], 1e-12));
        }
        let singular = [[1.0; ARM_DOF]; ARM_DOF];
        assert!(solve6(&singular, &b).is_none());
    }

    #[test]
    fn refine_converges_from_a_nearby_seed() {
        let chain = generic_chain();
        let target = target_at(&chain, &Q_GENERIC);
        let mut seed = Q_GENERIC;
        for (k, v) in seed.iter_mut().enumerate() {
            *v += 0.1 * if k % 2 == 0 { 1.0 } else { -1.0 };
        }
        let q = chain.refine(&seed, &target, MAX_ITER);
        let r = chain.residual(&q, &target);
        assert!(r.iter().all(|v| v.abs() < 1e-9), "{r:?}");
    }

    #[test]
    fn solve_grades_a_feasible_solution_and_enforces_continuity() {
        let chain = generic_chain();
        let target = target_at(&chain, &Q_GENERIC);
        let mut seed = Q_GENERIC;
        seed[1] += 0.1;
        let s = chain
            .solve(&target, &[seed], None, None)
            .expect("reachable");
        assert!(s.position_error < POS_TOL && s.tilt_error < ANG_TOL);
        assert!(s.heading_error < ANG_TOL && s.azimuth_error < ANG_TOL);
        assert!(close(
            s.margin,
            chain
                .margins(&s.q)
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
            0.0
        ));
        assert!(close(s.step, 0.0, 0.0));
        let stepped = chain
            .solve(&target, &[seed], Some(&seed), Some(1.0))
            .expect("within a radian");
        assert!(close(stepped.step, 0.1, 1e-6));
        assert!(chain
            .solve(&target, &[seed], Some(&seed), Some(0.01))
            .is_none());
    }

    #[test]
    fn solve_returns_none_for_an_unreachable_target() {
        let chain = generic_chain();
        let mut target = target_at(&chain, &Q_GENERIC);
        target.position = [10.0, 10.0, 10.0];
        assert!(chain.solve(&target, &[Q_GENERIC], None, None).is_none());
    }

    #[test]
    fn solve_path_walks_waypoints_within_the_step_cap() {
        // Waypoints are the poses of a joint-space line, so every one is reachable and the
        // continuous walk moves each joint 0.05 rad per waypoint, inside the 0.2 cap.
        let chain = generic_chain();
        let mut q_end = Q_GENERIC;
        q_end[0] += 0.5;
        q_end[2] -= 0.3;
        let n = 10;
        let targets: Vec<GraspTarget> = (1..=n)
            .map(|k| {
                let t = k as f64 / n as f64;
                let mut q = Q_GENERIC;
                for j in 0..ARM_DOF {
                    q[j] += t * (q_end[j] - Q_GENERIC[j]);
                }
                target_at(&chain, &q)
            })
            .collect();
        let path = chain
            .solve_path(&targets, &Q_GENERIC, &[], 0.2)
            .expect("walkable");
        assert_eq!(path.len(), n);
        let mut previous = Q_GENERIC;
        for s in &path {
            let step = (0..ARM_DOF)
                .map(|j| (s.q[j] - previous[j]).abs())
                .fold(0.0, f64::max);
            assert!(step <= 0.2 && close(step, s.step, 0.0));
            previous = s.q;
        }
        let last = path.last().unwrap();
        assert!(last.position_error < POS_TOL);
        assert!(
            (0..ARM_DOF).all(|j| close(last.q[j], q_end[j], 1e-6)),
            "{:?}",
            last.q
        );
        let mut broken = targets.clone();
        broken[5].position = [10.0, 10.0, 10.0];
        assert!(chain.solve_path(&broken, &Q_GENERIC, &[], 0.2).is_none());
    }
}
