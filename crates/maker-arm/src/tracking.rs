//! Compliant tracking controller (tracking design, 2026-09-25, sections 6, 7 and 10).
//!
//! Motion side: a time-stamped plan, interpolated linearly, through a critically damped
//! filter into the feed-forward `G(q_r) + g [RNEA(q_r, dq_r, ddq_d) - G(q_r) + F(dq_r)]`.
//! Contact side: a momentum observer estimates the external torque `r`; past a per-joint
//! deadband an admittance moves the reference along the push; a contact gate `g` removes
//! the inertial, Coriolis, friction and velocity feed-forward while the arm is blocked; an
//! error clamp caps the spring torque at `kp e_max`. Everything integrates over the MEASURED
//! tick interval, so a late tick on a non-RT kernel does not corrupt the state. The output
//! is a desire like any controller's: the loop's `clamp_command` decides what is sent.

use crate::config::ArmConfig;
use crate::controller::Controller;
use crate::dynamics::{Dynamics, JointMatrix};
use crate::kinematics::{Joints, ARM_DOF};
use crate::state::{ArmCommand, ArmState, JointCommand};
use std::f64::consts::PI;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// One plan row: six joint targets and the gripper.
pub const ROW: usize = ARM_DOF + 1;
pub type Row = [f64; ROW];

/// A tick interval over this is a gap: the observer restarts (design section 10).
pub const GAP_S: f64 = 0.020;
/// A tick interval over this counts as late in the telemetry (1.5 periods at 200 Hz).
pub const LATE_S: f64 = 0.0075;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingParams {
    /// Reference filter corner (Hz).
    pub f_r: f64,
    /// Observer gain (1/s): r follows tau_ext with a lag of 1/k_o.
    pub k_o: f64,
    /// Yield corner (Hz): D_a = kp / (2 pi f_y).
    pub f_y: f64,
    /// K_a = ka_ratio * kp.
    pub ka_ratio: f64,
    /// Deadband per joint (N m). Infinite means no yield.
    pub tau_th: Joints,
    /// Error clamp per joint (rad): |p - q| <= e_max.
    pub e_max: Joints,
    pub gate_fall: f64,
    pub gate_rise: f64,
    pub gate_quiet: f64,
    /// False is rung F: no observer, admittance, gate or error clamp.
    pub contact: bool,
}

impl Default for TrackingParams {
    fn default() -> Self {
        TrackingParams {
            f_r: 10.0,      // INITIAL
            k_o: 50.0,      // INITIAL
            f_y: 2.0,       // INITIAL
            ka_ratio: 0.25, // owner decision 2026-09-25: K_a = kp / 4
            // No yield until the calibration rule sets the deadband; the error clamp still
            // caps the spring torque.
            tau_th: [f64::INFINITY; ARM_DOF],
            e_max: [0.03; ARM_DOF], // INITIAL
            gate_fall: 0.020,       // INITIAL
            gate_rise: 0.100,       // INITIAL
            gate_quiet: 0.050,      // INITIAL
            contact: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackingError {
    Param { name: &'static str, detail: String },
    Profile { joints: usize },
    Push { detail: &'static str },
    Loop { detail: &'static str },
}

impl fmt::Display for TrackingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrackingError::Param { name, detail } => {
                write!(f, "tracking parameter {name}: {detail}")
            }
            TrackingError::Profile { joints } => {
                write!(f, "the profile has {joints} joints, tracking needs {ROW}")
            }
            TrackingError::Push { detail } => write!(f, "bad push: {detail}"),
            TrackingError::Loop { detail } => write!(f, "cannot run in a loop: {detail}"),
        }
    }
}

impl std::error::Error for TrackingError {}

fn positive(name: &'static str, v: f64) -> Result<(), TrackingError> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(TrackingError::Param {
            name,
            detail: format!("{v} is not a positive finite number"),
        })
    }
}

impl TrackingParams {
    pub fn validate(&self) -> Result<(), TrackingError> {
        positive("f_r", self.f_r)?;
        positive("k_o", self.k_o)?;
        positive("f_y", self.f_y)?;
        positive("ka_ratio", self.ka_ratio)?;
        positive("gate_fall", self.gate_fall)?;
        positive("gate_rise", self.gate_rise)?;
        positive("gate_quiet", self.gate_quiet)?;
        for &v in &self.e_max {
            positive("e_max", v)?;
        }
        for &v in &self.tau_th {
            // Infinity is allowed and means no yield.
            if v.is_nan() || v <= 0.0 {
                return Err(TrackingError::Param {
                    name: "tau_th",
                    detail: format!("{v} is not positive"),
                });
            }
        }
        // The yield loop is stable only well below the observer bandwidth.
        let bandwidth = self.k_o / (2.0 * PI);
        if self.f_y > bandwidth / 2.0 {
            return Err(TrackingError::Param {
                name: "f_y",
                detail: format!(
                    "{} Hz is above half the observer bandwidth ({bandwidth:.2} Hz)",
                    self.f_y
                ),
            });
        }
        Ok(())
    }
}

/// `rows[k]` is due at `t0 + k dt` on the loop clock (`ArmState::t`).
#[derive(Clone, Debug, PartialEq)]
pub struct Push {
    pub t0: f64,
    pub dt: f64,
    pub rows: Vec<Row>,
}

impl Push {
    pub fn new(t0: f64, dt: f64, rows: Vec<Row>) -> Result<Push, TrackingError> {
        let fail = |detail| Err(TrackingError::Push { detail });
        if !t0.is_finite() {
            return fail("t0 must be finite");
        }
        if !(dt.is_finite() && dt > 0.0) {
            return fail("dt must be a positive finite number");
        }
        if rows.is_empty() {
            return fail("a push needs at least one row");
        }
        if rows.iter().flatten().any(|v| !v.is_finite()) {
            return fail("every target must be finite");
        }
        Ok(Push { t0, dt, rows })
    }
}

/// The stored plan: time-stamped rows, strictly increasing in time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    samples: Vec<(f64, Row)>,
}

impl Plan {
    /// Replaces the plan from `p.t0` on; earlier samples stay, so the interpolation is
    /// continuous across the switch.
    pub fn push(&mut self, p: Push) {
        self.samples.retain(|s| s.0 < p.t0);
        self.samples.extend(
            p.rows
                .iter()
                .enumerate()
                .map(|(k, r)| (p.t0 + k as f64 * p.dt, *r)),
        );
    }

    /// Joints interpolated linearly at `t`, the gripper held from the last sample at or
    /// before `t`; before the first sample and after the last, the nearest sample.
    pub fn sample(&self, t: f64) -> Option<Row> {
        let s = &self.samples;
        let first = s.first()?;
        let k = s.partition_point(|x| x.0 <= t);
        if k == 0 {
            return Some(first.1);
        }
        if k == s.len() {
            return Some(s[k - 1].1);
        }
        let ((t0, a), (t1, b)) = (&s[k - 1], &s[k]);
        let f = (t - t0) / (t1 - t0);
        let mut out = *a;
        for j in 0..ARM_DOF {
            out[j] = a[j] + f * (b[j] - a[j]);
        }
        Some(out)
    }

    /// Adds `p` at `now`. A push due later than `now` onto a plan with no sample at or after
    /// `now` (empty or run out) first gets the sample `(now, current target)`, so the
    /// reference moves from where it is at `now` and reaches row 0 at `t0`, not at once.
    pub fn push_at(&mut self, p: Push, now: f64, hold: Row) {
        if p.t0 > now && self.samples.last().is_none_or(|s| s.0 < now) {
            let current = self.sample(now).unwrap_or(hold);
            self.samples.push((now, current));
        }
        self.push(p);
    }

    /// Drops the samples no interpolation at or after `t` can need.
    pub fn forget_before(&mut self, t: f64) {
        let k = self.samples.partition_point(|x| x.0 <= t);
        if k > 1 {
            self.samples.drain(..k - 1);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Telemetry {
    pub r: Joints,
    pub gate: f64,
    /// The admittance offset of the reference (rad), the design's dq_a.
    pub offset: Joints,
    pub q_r: Joints,
    pub tau_ff: Joints,
    pub late_ticks: u64,
    pub gaps: u64,
    pub max_interval: f64,
    /// The tick clock of the last tick (s, `ArmState::t`).
    pub tick_t: f64,
}

/// The half of the controller other threads reach: pushes in, telemetry and the clock out.
/// The loop side only `try_lock`s, so a Python caller holding a lock can delay a push or a
/// telemetry copy by a tick, never the tick itself.
#[derive(Debug, Default)]
pub struct TrackingShared {
    pending: Mutex<Vec<Push>>,
    telemetry: Mutex<Option<Telemetry>>,
    clock: Mutex<Option<(f64, Instant)>>,
}

impl TrackingShared {
    pub fn push(&self, p: Push) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(p);
    }

    /// The last tick's telemetry, or None before the first tick.
    pub fn telemetry(&self) -> Option<Telemetry> {
        *self.telemetry.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `(now, age)`: the loop clock now, which is the tick clock (the latest good
    /// `ArmState::t`, seconds since connect) plus the wall time since that tick, and that
    /// wall time (s). None before the first tick.
    pub fn clock(&self) -> Option<(f64, f64)> {
        self.clock
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map(|(t, at)| {
                let age = at.elapsed().as_secs_f64();
                (t + age, age)
            })
    }
}

fn mat6_vec(m: &JointMatrix, v: &Joints) -> Joints {
    let mut out = [0.0; ARM_DOF];
    for i in 0..ARM_DOF {
        out[i] = (0..ARM_DOF).map(|k| m[i][k] * v[k]).sum();
    }
    out
}

pub struct TrackingController {
    dynamics: Dynamics,
    params: TrackingParams,
    kp: [f64; ROW],
    kd: [f64; ROW],
    shared: Arc<TrackingShared>,
    plan: Plan,
    started: bool,
    t_prev: f64,
    /// The measured pose at start: the empty plan's target.
    hold: Row,
    u_prev: Joints,
    q_d: Joints,
    dq_d: Joints,
    ddq_d: Joints,
    offset: Joints,
    offset_rate: Joints,
    r: Joints,
    p_prev: Joints,
    m_prev: JointMatrix,
    gate: f64,
    quiet: f64,
    telemetry: Telemetry,
}

impl TrackingController {
    pub fn new(
        dynamics: Dynamics,
        params: TrackingParams,
        config: &ArmConfig,
    ) -> Result<TrackingController, TrackingError> {
        params.validate()?;
        if config.joints.len() != ROW {
            return Err(TrackingError::Profile {
                joints: config.joints.len(),
            });
        }
        let (mut kp, mut kd) = ([0.0; ROW], [0.0; ROW]);
        for (j, jc) in config.joints.iter().enumerate() {
            kp[j] = jc.kp;
            kd[j] = jc.kd;
        }
        for j in 0..ARM_DOF {
            positive("kp", kp[j])?;
            if !(kd[j].is_finite() && kd[j] >= 0.0) {
                return Err(TrackingError::Param {
                    name: "kd",
                    detail: format!("{} is not a finite non-negative number", kd[j]),
                });
            }
        }
        let zero = [0.0; ARM_DOF];
        Ok(TrackingController {
            dynamics,
            params,
            kp,
            kd,
            shared: Arc::new(TrackingShared::default()),
            plan: Plan::default(),
            started: false,
            t_prev: 0.0,
            hold: [0.0; ROW],
            u_prev: zero,
            q_d: zero,
            dq_d: zero,
            ddq_d: zero,
            offset: zero,
            offset_rate: zero,
            r: zero,
            p_prev: zero,
            m_prev: [[0.0; ARM_DOF]; ARM_DOF],
            gate: 1.0,
            quiet: 0.0,
            telemetry: Telemetry::default(),
        })
    }

    pub fn shared(&self) -> Arc<TrackingShared> {
        Arc::clone(&self.shared)
    }

    /// Readies the controller for a loop: the next update starts again from the measured pose
    /// at rest, with no tick clock and fresh telemetry. Refuses rung F, which has no error
    /// clamp, and a controller with a plan or a pending push: their stamps belong to another
    /// clock (the bench's, or a guess before the first tick), and dropping them silently
    /// would be worse.
    pub fn restart_for_loop(&mut self) -> Result<(), TrackingError> {
        let fail = |detail| Err(TrackingError::Loop { detail });
        if !self.params.contact {
            return fail("contact=False (rung F) has no error clamp and is for the bench only");
        }
        let pending = self
            .shared
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !self.plan.samples.is_empty() || !pending.is_empty() {
            return fail(
                "it already has a plan or a pending push, stamped on another clock; \
                 build a fresh Tracking and push after start_tracking",
            );
        }
        self.started = false;
        self.telemetry = Telemetry::default();
        *self
            .shared
            .telemetry
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *self.shared.clock.lock().unwrap_or_else(|e| e.into_inner()) = None;
        Ok(())
    }

    fn begin(&mut self, t: f64, q: &Joints, dq: &Joints, gripper: f64) {
        self.hold[..ARM_DOF].copy_from_slice(q);
        self.hold[ARM_DOF] = gripper;
        self.u_prev = *q;
        self.q_d = *q;
        self.dq_d = [0.0; ARM_DOF];
        self.ddq_d = [0.0; ARM_DOF];
        self.offset = [0.0; ARM_DOF];
        self.offset_rate = [0.0; ARM_DOF];
        self.r = [0.0; ARM_DOF];
        self.gate = 1.0;
        self.quiet = 0.0;
        self.m_prev = self.dynamics.mass_matrix(q);
        self.p_prev = mat6_vec(&self.m_prev, dq);
        self.t_prev = t;
        self.started = true;
    }

    /// One exact step of `ddq_d = w^2 (u - q_d) - 2 w dq_d` with the input moving linearly
    /// from `u_prev` to `u` over `h`: stable for any interval, and a steady ramp lags by
    /// exactly 2/w.
    fn filter(&mut self, u: &Joints, h: f64) {
        let w = 2.0 * PI * self.params.f_r;
        let e = (-w * h).exp();
        // Indexes u, u_prev, q_d and dq_d by the same joint: an iterator would not read better.
        #[allow(clippy::needless_range_loop)]
        for j in 0..ARM_DOF {
            let rho = (u[j] - self.u_prev[j]) / h;
            let y0 = self.q_d[j] - self.u_prev[j] + 2.0 * rho / w;
            let yd0 = self.dq_d[j] - rho;
            let c = yd0 + w * y0;
            self.q_d[j] = u[j] - 2.0 * rho / w + (y0 + c * h) * e;
            self.dq_d[j] = rho + (yd0 - w * c * h) * e;
        }
    }

    /// Momentum observer (De Luca 2006): r follows tau_ext with a lag of 1/k_o. The update
    /// is exact for a torque held over the tick, so one 10 ms tick and two 5 ms ticks agree.
    fn observe(&mut self, q: &Joints, dq: &Joints, tau_m: &Joints, h: f64, gap: bool) {
        let m = self.dynamics.mass_matrix(q);
        let p = mat6_vec(&m, dq);
        if gap {
            // Start again from the current momentum.
            self.r = [0.0; ARM_DOF];
        } else if h > 0.0 {
            let mut mdot_dq = [0.0; ARM_DOF];
            for i in 0..ARM_DOF {
                for k in 0..ARM_DOF {
                    mdot_dq[i] += (m[i][k] - self.m_prev[i][k]) / h * dq[k];
                }
            }
            let c_dq = self.dynamics.coriolis(q, dq);
            let g = self.dynamics.gravity(q);
            let f = self.dynamics.friction(dq);
            let decay = (-self.params.k_o * h).exp();
            for j in 0..ARM_DOF {
                // C^T dq = Mdot dq - C dq.
                let beta = mdot_dq[j] - c_dq[j] - g[j] - f[j];
                let ext = (p[j] - self.p_prev[j]) / h - tau_m[j] - beta;
                self.r[j] = decay * self.r[j] + (1.0 - decay) * ext;
            }
        }
        self.p_prev = p;
        self.m_prev = m;
    }

    /// Deadband admittance `D_a d(offset)/dt + K_a offset = dz(r)` by backward Euler (stable
    /// for any step), and the contact gate. On a gap tick the observer has just restarted and
    /// cannot see a push yet, so the admittance, the gate and its quiet timer keep their state
    /// for that tick (design section 10).
    fn yield_and_gate(&mut self, h: f64, gap: bool) {
        let mut over = false;
        for j in 0..ARM_DOF {
            let excess = self.r[j].abs() - self.params.tau_th[j];
            over |= excess > 0.0;
            if h > 0.0 && !gap {
                let dz = if excess > 0.0 {
                    excess.copysign(self.r[j])
                } else {
                    0.0
                };
                let ka = self.params.ka_ratio * self.kp[j];
                let da = self.kp[j] / (2.0 * PI * self.params.f_y);
                let x = (da * self.offset[j] + h * dz) / (da + ka * h);
                self.offset_rate[j] = (x - self.offset[j]) / h;
                self.offset[j] = x;
            } else {
                // A gap keeps the admittance where it is for that tick.
                self.offset_rate[j] = 0.0;
            }
        }
        if h > 0.0 && !gap {
            if over {
                self.quiet = 0.0;
                self.gate = (self.gate - h / self.params.gate_fall).max(0.0);
            } else {
                self.quiet += h;
                if self.quiet >= self.params.gate_quiet {
                    self.gate = (self.gate + h / self.params.gate_rise).min(1.0);
                }
            }
        }
    }
}

impl Controller for TrackingController {
    /// `_dt` is the nominal period; the controller integrates over the measured interval
    /// between `ArmState::t` stamps instead (design section 10).
    fn update(&mut self, state: &ArmState, _dt: f64) -> ArmCommand {
        if state.motors.len() != ROW {
            // Unreachable from the loop (the profile has ROW joints); a non-finite command
            // takes the fault-hold path instead of a panic.
            let nan = JointCommand {
                pos: f64::NAN,
                vel: 0.0,
                kp: 0.0,
                kd: 0.0,
                tau: 0.0,
            };
            return vec![nan; state.motors.len()];
        }
        let t = state.t;
        let (mut q, mut dq, mut tau_m) = ([0.0; ARM_DOF], [0.0; ARM_DOF], [0.0; ARM_DOF]);
        for j in 0..ARM_DOF {
            let m = &state.motors[j];
            q[j] = m.position;
            dq[j] = m.velocity;
            tau_m[j] = m.torque;
        }
        if !self.started {
            self.begin(t, &q, &dq, state.motors[ARM_DOF].position);
        }
        let raw = t - self.t_prev;
        // A repeated, backward or NaN stamp integrates nothing and leaves the tick clock at
        // the latest good stamp, so the clock never runs backward. A clock that is not finite
        // (a NaN first stamp) takes the next stamp as it is.
        let h = if raw.is_finite() && raw > 0.0 {
            raw
        } else {
            0.0
        };
        if h > 0.0 || !self.t_prev.is_finite() {
            self.t_prev = t;
        }
        let t = self.t_prev;
        if let Ok(mut clock) = self.shared.clock.try_lock() {
            *clock = Some((t, Instant::now()));
        }
        let gap = h > GAP_S;
        if h > LATE_S {
            self.telemetry.late_ticks += 1;
        }
        if gap {
            self.telemetry.gaps += 1;
        }
        self.telemetry.max_interval = self.telemetry.max_interval.max(h);

        // After `begin` and the clock update: a future push anchors at this tick's time and
        // the hold pose (design section 6: row k is due at t0 + k dt).
        if let Ok(mut pending) = self.shared.pending.try_lock() {
            for p in pending.drain(..) {
                self.plan.push_at(p, t, self.hold);
            }
        }
        let target = self.plan.sample(t).unwrap_or(self.hold);
        self.plan.forget_before(t);
        let mut u = [0.0; ARM_DOF];
        u.copy_from_slice(&target[..ARM_DOF]);
        if h > 0.0 {
            self.filter(&u, h);
        }
        self.u_prev = u;
        let w = 2.0 * PI * self.params.f_r;
        // Indexes ddq_d, u, q_d and dq_d by the same joint.
        #[allow(clippy::needless_range_loop)]
        for j in 0..ARM_DOF {
            self.ddq_d[j] = w * w * (u[j] - self.q_d[j]) - 2.0 * w * self.dq_d[j];
        }

        if self.params.contact {
            self.observe(&q, &dq, &tau_m, h, gap);
            self.yield_and_gate(h, gap);
        }

        let (mut q_r, mut dq_r) = ([0.0; ARM_DOF], [0.0; ARM_DOF]);
        for j in 0..ARM_DOF {
            q_r[j] = self.q_d[j] + self.offset[j];
            dq_r[j] = self.dq_d[j] + self.offset_rate[j];
        }
        let g_r = self.dynamics.gravity(&q_r);
        let full = self.dynamics.inverse(&q_r, &dq_r, &self.ddq_d);
        let fr = self.dynamics.friction(&dq_r);
        let mut tau_ff = [0.0; ARM_DOF];
        let mut cmd = Vec::with_capacity(ROW);
        for j in 0..ARM_DOF {
            tau_ff[j] = g_r[j] + self.gate * (full[j] - g_r[j] + fr[j]);
            let pos = if self.params.contact {
                let e = self.params.e_max[j];
                q[j] + (q_r[j] - q[j]).clamp(-e, e)
            } else {
                q_r[j]
            };
            cmd.push(JointCommand {
                pos,
                vel: self.gate * dq_r[j],
                kp: self.kp[j],
                kd: self.kd[j],
                tau: tau_ff[j],
            });
        }
        cmd.push(JointCommand {
            pos: target[ARM_DOF],
            vel: 0.0,
            kp: self.kp[ARM_DOF],
            kd: self.kd[ARM_DOF],
            tau: 0.0,
        });

        self.telemetry.r = self.r;
        self.telemetry.gate = self.gate;
        self.telemetry.offset = self.offset;
        self.telemetry.q_r = q_r;
        self.telemetry.tau_ff = tau_ff;
        self.telemetry.tick_t = t;
        if let Ok(mut out) = self.shared.telemetry.try_lock() {
            *out = Some(self.telemetry);
        }
        cmd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{Body, Friction};
    use crate::kinematics::{Link, Vec3};
    use crate::state::MotorState;

    const H: f64 = 0.005;

    /// Six vertical axes, gravity along them: G = 0 and, at rest, C dq = 0, so a static
    /// arm's observer sees tau_ext = -tau_m exactly.
    fn vertical(gravity: Vec3) -> Dynamics {
        let z = Some([0.0, 0.0, 1.0]);
        let body = Body {
            mass: 0.5,
            com: [0.05, 0.0, 0.0],
            inertia: [[1e-3, 0.0, 0.0], [0.0, 2e-3, 0.0], [0.0, 0.0, 2e-3]],
        };
        let links: Vec<Link> = (0..6)
            .map(|_| Link::new([0.1, 0.0, 0.0], [0.0; 3], z))
            .collect();
        Dynamics::new(
            [0.0; 3],
            &links,
            [body; 6],
            [Friction::default(); 6],
            gravity,
        )
        .unwrap()
    }

    fn ctrl(params: TrackingParams) -> TrackingController {
        TrackingController::new(
            vertical([0.0, 0.0, -9.81]),
            params,
            &ArmConfig::maker_arm_v1(),
        )
        .unwrap()
    }

    fn free() -> TrackingParams {
        TrackingParams {
            contact: false,
            ..TrackingParams::default()
        }
    }

    fn st(t: f64, q: Joints, dq: Joints, tau: Joints) -> ArmState {
        let mut motors: Vec<MotorState> = (0..ARM_DOF)
            .map(|j| MotorState {
                position: q[j],
                velocity: dq[j],
                torque: tau[j],
                feedback_age: 0.0,
                ..MotorState::stale()
            })
            .collect();
        motors.push(MotorState {
            position: -1.0,
            feedback_age: 0.0,
            ..MotorState::stale()
        });
        ArmState { motors, tick: 0, t }
    }

    fn at_rest(t: f64) -> ArmState {
        st(t, [0.0; 6], [0.0; 6], [0.0; 6])
    }

    fn row(j0: f64) -> Row {
        let mut r = [0.0; ROW];
        r[0] = j0;
        r[ARM_DOF] = -1.0;
        r
    }

    fn ramp(rate: f64, from: f64, dt: f64, n: usize) -> Vec<Row> {
        (0..=n).map(|k| row(from + rate * k as f64 * dt)).collect()
    }

    fn w() -> f64 {
        2.0 * PI * TrackingParams::default().f_r
    }

    #[test]
    fn plan_push_replaces_from_t0_and_keeps_earlier_samples() {
        let mut plan = Plan::default();
        plan.push(Push::new(0.0, 1.0, vec![row(0.0), row(1.0), row(2.0)]).unwrap());
        let mut late = row(10.0);
        late[ARM_DOF] = 5.0;
        plan.push(Push::new(1.5, 1.0, vec![late]).unwrap());
        let s = plan.sample(1.25).unwrap();
        assert!((s[0] - 5.5).abs() < 1e-12); // halfway from 1.0 (t=1) to 10.0 (t=1.5)
        assert_eq!(s[ARM_DOF], -1.0); // the gripper holds the t=1 sample
        assert_eq!(plan.sample(-3.0).unwrap()[0], 0.0); // before the first: the first
        assert_eq!(plan.sample(9.0).unwrap()[0], 10.0); // after the last: the last
        assert_eq!(plan.sample(9.0).unwrap()[ARM_DOF], 5.0);
    }

    #[test]
    fn filter_step_has_no_overshoot() {
        let mut c = ctrl(free());
        c.update(&at_rest(0.0), H);
        c.shared()
            .push(Push::new(0.0, 0.01, vec![row(0.1)]).unwrap());
        let mut prev = 0.0;
        for k in 1..=400 {
            c.update(&at_rest(k as f64 * H), H);
            assert!(
                c.q_d[0] <= 0.1 + 1e-12,
                "overshoot at tick {k}: {}",
                c.q_d[0]
            );
            assert!(c.q_d[0] >= prev - 1e-15);
            prev = c.q_d[0];
        }
        assert!((c.q_d[0] - 0.1).abs() < 1e-9);
    }

    #[test]
    fn filter_ramp_lag_is_two_over_w() {
        let mut c = ctrl(free());
        c.shared()
            .push(Push::new(0.0, 0.01, ramp(0.5, 0.0, 0.01, 300)).unwrap());
        for k in 0..=400 {
            c.update(&at_rest(k as f64 * H), H);
        }
        let lag = (0.5 * 2.0 - c.q_d[0]) / 0.5; // t = 2.0 s, u = 1.0
        assert!((lag - 2.0 / w()).abs() < 0.01 * 2.0 / w(), "lag {lag}");
        assert!((c.dq_d[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_future_plan_switch_keeps_the_reference_smooth() {
        let mut c = ctrl(free());
        c.shared()
            .push(Push::new(0.0, 0.01, ramp(0.5, 0.0, 0.01, 300)).unwrap());
        let (mut q_prev, mut v_prev) = (0.0, 0.0);
        for k in 0..=400 {
            let t = k as f64 * H;
            if k == 200 {
                // at t = 1.0, replace from 1.05 on with a ramp back down from the old value
                c.shared()
                    .push(Push::new(1.05, 0.01, ramp(-0.5, 0.525, 0.01, 100)).unwrap());
            }
            c.update(&at_rest(t), H);
            if k > 0 {
                assert!((c.q_d[0] - q_prev).abs() < 0.01, "q_d jump at {t}");
                assert!((c.dq_d[0] - v_prev).abs() < 0.2, "dq_d jump at {t}");
            }
            q_prev = c.q_d[0];
            v_prev = c.dq_d[0];
        }
    }

    #[test]
    fn a_push_into_the_past_is_absorbed_by_the_filter() {
        let mut c = ctrl(free());
        let mut q_prev = 0.0;
        for k in 0..=400 {
            let t = k as f64 * H;
            if k == 200 {
                c.shared()
                    .push(Push::new(0.5, 0.01, vec![row(0.3)]).unwrap());
            }
            let out = c.update(&at_rest(t), H);
            assert!(out.iter().all(|j| j.pos.is_finite() && j.tau.is_finite()));
            // The steepest step response is 0.3 w / e = 6.9 rad/s: 0.035 rad per tick.
            assert!((c.q_d[0] - q_prev).abs() < 0.05, "q_d jump at {t}");
            q_prev = c.q_d[0];
        }
        assert!((c.q_d[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn a_future_push_starts_from_the_current_target_now() {
        // First on the empty plan at start, then on a plan that has run out: a row due 1 s
        // ahead must not move the input or q_d on the next tick, and the input is halfway at
        // t0 - 0.5 s (design section 6: row k is due at t0 + k dt).
        let mut c = ctrl(free());
        c.update(&at_rest(0.0), H);
        let mut k = 0;
        for (from, to) in [(0.0, 0.1), (0.1, 0.2)] {
            let t0 = (k + 1) as f64 * H + 1.0;
            c.shared().push(Push::new(t0, 0.01, vec![row(to)]).unwrap());
            let q_d = c.q_d[0];
            for i in 1..=500 {
                k += 1;
                c.update(&at_rest(k as f64 * H), H);
                if i == 1 {
                    assert_eq!(c.u_prev[0], from, "the input moved on the push tick");
                    assert!((c.q_d[0] - q_d).abs() < 1e-12, "q_d moved on the push tick");
                }
                if i == 101 {
                    let half = (from + to) / 2.0;
                    assert!((c.u_prev[0] - half).abs() < 1e-12, "{}", c.u_prev[0]);
                }
            }
            // About 1.5 s past t0: the plan has run out and q_d has settled on `to`.
            assert!((c.q_d[0] - to).abs() < 1e-9, "{}", c.q_d[0]);
        }
    }

    #[test]
    fn feed_forward_is_gravity_at_rest_and_full_rnea_in_motion() {
        let d = vertical([0.0, -9.81, 0.0]);
        let mut c = TrackingController::new(d.clone(), free(), &ArmConfig::maker_arm_v1()).unwrap();
        let q0 = [0.2, -0.3, 0.4, 0.1, -0.2, 0.3];
        let out = c.update(&st(0.0, q0, [0.0; 6], [0.0; 6]), H);
        let g = d.gravity(&q0);
        for j in 0..ARM_DOF {
            assert!((out[j].tau - g[j]).abs() < 1e-12);
            assert_eq!(out[j].pos, q0[j]);
            assert_eq!(out[j].vel, 0.0);
        }
        c.shared()
            .push(Push::new(0.0, 0.01, ramp(0.5, 0.2, 0.01, 300)).unwrap());
        let mut out = Vec::new();
        for k in 1..=100 {
            out = c.update(&st(k as f64 * H, q0, [0.0; 6], [0.0; 6]), H);
        }
        let full = d.inverse(&c.q_d, &c.dq_d, &c.ddq_d);
        for j in 0..ARM_DOF {
            assert!((out[j].tau - full[j]).abs() < 1e-12);
            assert!((out[j].pos - c.q_d[j]).abs() < 1e-15); // F: no error clamp
            assert!((out[j].vel - c.dq_d[j]).abs() < 1e-15);
        }
        assert_eq!(out[ARM_DOF].pos, -1.0); // the gripper holds its start value
    }

    fn contact(tau_th: f64) -> TrackingParams {
        TrackingParams {
            tau_th: [tau_th; ARM_DOF],
            ..TrackingParams::default()
        }
    }

    #[test]
    fn observer_follows_a_step_with_time_constant_one_over_k_o() {
        let k_o = TrackingParams::default().k_o;
        let mut c = ctrl(contact(f64::INFINITY));
        for k in 0..=20 {
            c.update(&at_rest(k as f64 * H), H);
        }
        let mut tau = [0.0; 6];
        tau[2] = -2.0;
        for n in 1..=20 {
            c.update(&st((20 + n) as f64 * H, [0.0; 6], [0.0; 6], tau), H);
            let want = 2.0 * (1.0 - (-k_o * n as f64 * H).exp());
            assert!((c.r[2] - want).abs() < 1e-9, "n {n}: {} vs {want}", c.r[2]);
        }
    }

    #[test]
    fn admittance_settles_at_the_design_force_against_a_wall() {
        // Joint 0 (kp 60) is stuck at q = 0; the plan goes delta = 0.02 past it.
        let mut p = contact(f64::INFINITY);
        p.tau_th[0] = 0.5;
        let mut c = ctrl(p);
        let kp = 60.0;
        c.shared()
            .push(Push::new(0.0, 0.01, vec![row(0.02)]).unwrap());
        let mut tau = [0.0; 6];
        let mut spring = 0.0;
        for k in 0..=1200 {
            let out = c.update(&st(k as f64 * H, [0.0; 6], [0.0; 6], tau), H);
            // The motor's MIT torque at the wall is what it reports next tick.
            tau[0] = out[0].kp * (out[0].pos - 0.0) + out[0].kd * out[0].vel + out[0].tau;
            spring = kp * out[0].pos;
        }
        let want = (kp * 0.02 + 4.0 * 0.5) / 5.0;
        assert!((spring - want).abs() < 0.01, "spring {spring} vs {want}");
    }

    #[test]
    fn error_clamp_bounds_the_spring_for_any_reference() {
        let mut c = ctrl(contact(f64::INFINITY));
        let q = [0.3, -0.2, 0.1, 0.0, 0.4, -0.5];
        let mut seed: u64 = 12345;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for k in 0..400 {
            if k % 40 == 0 {
                let mut r = [0.0; ROW];
                for v in r.iter_mut().take(ARM_DOF) {
                    *v = next();
                }
                c.shared()
                    .push(Push::new(k as f64 * H, 0.01, vec![r]).unwrap());
            }
            let out = c.update(&st(k as f64 * H, q, [0.0; 6], [0.0; 6]), H);
            for j in 0..ARM_DOF {
                assert!((out[j].pos - q[j]).abs() <= 0.03 + 1e-12);
            }
        }
    }

    #[test]
    fn gate_falls_in_20_ms_and_rises_after_a_quiet_50_then_100_ms() {
        let mut c = ctrl(contact(1.0));
        let (mut t_over, mut t_zero, mut t_under, mut t_one) = (None, None, None, None);
        for k in 0..=200 {
            let t = k as f64 * H;
            let mut tau = [0.0; 6];
            if (0.1..0.3).contains(&t) {
                tau[1] = -3.0;
            }
            c.update(&st(t, [0.0; 6], [0.0; 6], tau), H);
            let over = c.r.iter().any(|r| r.abs() > 1.0);
            if over && t_over.is_none() {
                t_over = Some(t);
            }
            if c.gate == 0.0 && t_zero.is_none() {
                t_zero = Some(t);
            }
            if t_zero.is_some() && !over && t_under.is_none() && t > 0.3 {
                t_under = Some(t);
            }
            if t_under.is_some() && c.gate == 1.0 && t_one.is_none() {
                t_one = Some(t);
            }
        }
        let fall = t_zero.unwrap() - t_over.unwrap();
        assert!((0.010..=0.020 + 1e-9).contains(&fall), "fall {fall}");
        let rise = t_one.unwrap() - t_under.unwrap();
        assert!((0.130..=0.160 + 1e-9).contains(&rise), "rise {rise}");
    }

    #[test]
    fn a_30_ms_gap_restarts_the_observer_and_is_counted() {
        let k_o = TrackingParams::default().k_o;
        let mut c = ctrl(contact(f64::INFINITY));
        let mut tau = [0.0; 6];
        tau[0] = -2.0;
        let mut t = 0.0;
        for _ in 0..=100 {
            c.update(&st(t, [0.0; 6], [0.0; 6], tau), H);
            t += H;
        }
        assert!((c.r[0] - 2.0).abs() < 1e-3);
        t += 0.025; // this tick lands 30 ms after the last
        c.update(&st(t, [0.0; 6], [0.0; 6], tau), H);
        assert_eq!(c.r[0], 0.0);
        let tel = c.shared().telemetry().unwrap();
        assert_eq!(tel.gaps, 1);
        assert_eq!(tel.late_ticks, 1);
        assert!((tel.max_interval - 0.030).abs() < 1e-9);
        let ticks = (5.0 / k_o / H).round() as usize;
        for _ in 0..ticks {
            t += H;
            c.update(&st(t, [0.0; 6], [0.0; 6], tau), H);
        }
        assert!((c.r[0] - 2.0).abs() < 0.01 * 2.0, "r {}", c.r[0]);
    }

    #[test]
    fn restart_for_loop_refuses_rung_f_and_an_old_plan_and_resets_the_clock() {
        assert!(ctrl(free()).restart_for_loop().is_err(), "rung F");
        let mut c = ctrl(contact(f64::INFINITY));
        c.update(&at_rest(0.0), H);
        c.update(&at_rest(0.05), H); // a late tick and a gap on the bench clock
        c.restart_for_loop().unwrap();
        assert_eq!(c.shared().clock(), None);
        assert_eq!(c.shared().telemetry(), None);
        c.update(&at_rest(7.0), H);
        let tel = c.shared().telemetry().unwrap();
        assert_eq!((tel.late_ticks, tel.gaps, tel.tick_t), (0, 0, 7.0));
        assert_eq!(tel.max_interval, 0.0);
        c.shared()
            .push(Push::new(7.0, 0.01, vec![row(0.1)]).unwrap());
        assert!(c.restart_for_loop().is_err(), "a pending push");
        c.update(&at_rest(7.005), H);
        assert!(c.restart_for_loop().is_err(), "a plan");
    }

    #[test]
    fn one_10_ms_tick_equals_two_5_ms_ticks() {
        let mut a = ctrl(contact(f64::INFINITY));
        let mut b = ctrl(contact(f64::INFINITY));
        let rows = ramp(0.5, 0.0, 0.01, 300);
        a.shared().push(Push::new(0.0, 0.01, rows.clone()).unwrap());
        b.shared().push(Push::new(0.0, 0.01, rows).unwrap());
        let mut tau = [0.0; 6];
        tau[0] = -1.5;
        for k in 0..=200 {
            a.update(&st(k as f64 * H, [0.0; 6], [0.0; 6], tau), H);
            if k % 2 == 0 {
                b.update(&st(k as f64 * H, [0.0; 6], [0.0; 6], tau), 2.0 * H);
                assert!((a.q_d[0] - b.q_d[0]).abs() < 1e-9, "q_d at tick {k}");
                assert!((a.dq_d[0] - b.dq_d[0]).abs() < 1e-9, "dq_d at tick {k}");
                assert!((a.r[0] - b.r[0]).abs() < 1e-9, "r at tick {k}");
            }
        }
    }

    #[test]
    fn a_repeated_or_backward_stamp_integrates_nothing() {
        let mut c = ctrl(contact(f64::INFINITY));
        c.shared()
            .push(Push::new(0.0, 0.01, ramp(0.5, 0.0, 0.01, 300)).unwrap());
        for k in 0..=20 {
            c.update(&at_rest(k as f64 * H), H);
        }
        let (q_d, r) = (c.q_d, c.r);
        let mut tau = [0.0; 6];
        tau[0] = -5.0;
        let out = c.update(&st(20.0 * H, [0.0; 6], [0.0; 6], tau), H); // same t again
        assert_eq!(c.q_d, q_d);
        assert_eq!(c.r, r);
        assert!(out
            .iter()
            .all(|j| j.pos.is_finite() && j.vel.is_finite() && j.tau.is_finite()));
        let out = c.update(&st(19.0 * H, [0.0; 6], [0.0; 6], tau), H); // backward
        assert_eq!(c.q_d, q_d);
        assert!(out.iter().all(|j| j.tau.is_finite()));
    }

    #[test]
    fn a_backward_or_nan_stamp_keeps_the_tick_clock_at_the_later_time() {
        let rows = ramp(0.5, 0.0, 0.01, 300);
        let mut tau = [0.0; 6];
        tau[0] = -1.5;
        let tick = |c: &mut TrackingController, t: f64| {
            c.update(&st(t, [0.0; 6], [0.0; 6], tau), H);
        };
        let (mut a, mut b, mut n) = (
            ctrl(contact(f64::INFINITY)),
            ctrl(contact(f64::INFINITY)),
            ctrl(contact(f64::INFINITY)),
        );
        for c in [&mut a, &mut b, &mut n] {
            c.shared().push(Push::new(0.0, 0.01, rows.clone()).unwrap());
        }
        tick(&mut n, f64::NAN); // a NaN first stamp must not poison the clock
        for k in 0..=20 {
            for c in [&mut a, &mut b, &mut n] {
                tick(c, k as f64 * H);
            }
        }
        tick(&mut a, 19.0 * H); // backward
        tick(&mut a, f64::NAN);
        for k in 21..=22 {
            for c in [&mut a, &mut b, &mut n] {
                tick(c, k as f64 * H);
            }
            // a and n integrate exactly H, like b, which saw no bad stamp
            for c in [&a, &n] {
                assert_eq!(c.q_d, b.q_d, "q_d at tick {k}");
                assert_eq!(c.dq_d, b.dq_d, "dq_d at tick {k}");
                assert_eq!(c.r, b.r, "r at tick {k}");
            }
        }
    }

    #[test]
    fn a_gap_holds_the_gate_while_the_arm_is_blocked() {
        let mut c = ctrl(contact(1.0));
        let mut tau = [0.0; 6];
        tau[1] = -3.0;
        let mut t = 0.0;
        for _ in 0..=40 {
            c.update(&st(t, [0.0; 6], [0.0; 6], tau), H);
            t += H;
        }
        assert_eq!(c.gate, 0.0);
        t += 0.055; // this tick lands 60 ms after the last
        c.update(&st(t, [0.0; 6], [0.0; 6], tau), H);
        assert_eq!(c.r[1], 0.0); // the observer restarted
        assert_eq!(c.gate, 0.0, "gate on the gap tick");
        for _ in 0..40 {
            t += H;
            c.update(&st(t, [0.0; 6], [0.0; 6], tau), H);
            assert_eq!(c.gate, 0.0, "gate at {t}");
        }
    }

    #[test]
    fn params_and_pushes_are_validated() {
        let bad = |f: fn(&mut TrackingParams)| {
            let mut p = TrackingParams::default();
            f(&mut p);
            p.validate().is_err()
        };
        assert!(TrackingParams::default().validate().is_ok());
        assert!(bad(|p| p.f_r = -1.0));
        assert!(bad(|p| p.k_o = f64::NAN));
        assert!(bad(|p| p.e_max[3] = 0.0));
        assert!(bad(|p| p.tau_th[1] = 0.0));
        assert!(bad(|p| p.tau_th[1] = f64::NAN));
        assert!(bad(|p| p.f_y = 5.0)); // above half the 7.96 Hz observer bandwidth
        assert!(bad(|p| p.gate_rise = 0.0));
        assert!(Push::new(0.0, 0.0, vec![row(0.0)]).is_err());
        assert!(Push::new(f64::NAN, 0.01, vec![row(0.0)]).is_err());
        assert!(Push::new(0.0, 0.01, vec![]).is_err());
        assert!(Push::new(0.0, 0.01, vec![row(f64::INFINITY)]).is_err());
        let mut short = ArmConfig::maker_arm_v1();
        short.joints.pop();
        assert!(TrackingController::new(
            vertical([0.0, 0.0, -9.81]),
            TrackingParams::default(),
            &short
        )
        .is_err());
    }
}
