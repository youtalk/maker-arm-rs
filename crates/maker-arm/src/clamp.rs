//! THE single command-clamp enforcement point (design §2). Every command
//! that reaches a motor passes through [`clamp_command`] in the control
//! loop's output stage; nothing else in this workspace may feed controller
//! or user values into `encode_mit`.

use crate::config::ArmConfig;
use crate::state::{ArmCommand, JointCommand};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClampError {
    /// Non-finite values are rejected outright: `float_to_u16` would map
    /// NaN to the range minimum — a full-negative-rail command.
    NonFinite {
        joint: usize,
        field: &'static str,
    },
    WrongLength {
        got: usize,
        expected: usize,
    },
}

impl std::fmt::Display for ClampError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClampError::NonFinite { joint, field } => {
                write!(f, "non-finite {field} commanded for joint index {joint}")
            }
            ClampError::WrongLength { got, expected } => {
                write!(f, "command has {got} joints, arm has {expected}")
            }
        }
    }
}

impl std::error::Error for ClampError {}

pub fn clamp_command(
    cmd: &[JointCommand],
    config: &ArmConfig,
) -> Result<(ArmCommand, bool), ClampError> {
    if cmd.len() != config.joints.len() {
        return Err(ClampError::WrongLength {
            got: cmd.len(),
            expected: config.joints.len(),
        });
    }
    let mut out = Vec::with_capacity(cmd.len());
    let mut clamped = false;
    for (i, (c, j)) in cmd.iter().zip(&config.joints).enumerate() {
        for (field, v) in [
            ("pos", c.pos),
            ("vel", c.vel),
            ("kp", c.kp),
            ("kd", c.kd),
            ("tau", c.tau),
        ] {
            if !v.is_finite() {
                return Err(ClampError::NonFinite { joint: i, field });
            }
        }
        let lo = j.q_lo + config.limit_margin;
        let hi = j.q_hi - config.limit_margin;
        let g = JointCommand {
            pos: c.pos.clamp(lo, hi),
            vel: c.vel.clamp(-config.max_velocity, config.max_velocity),
            kp: c.kp.clamp(0.0, config.kp_max),
            kd: c.kd.clamp(0.0, config.kd_max),
            tau: c.tau.clamp(-j.tau_max, j.tau_max),
        };
        clamped |= g != *c;
        out.push(g);
    }
    Ok((out, clamped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ArmConfig;
    use crate::state::JointCommand;

    fn zero_cmd(n: usize) -> Vec<JointCommand> {
        vec![
            JointCommand {
                pos: 0.0,
                vel: 0.0,
                kp: 0.0,
                kd: 0.0,
                tau: 0.0
            };
            n
        ]
    }

    #[test]
    fn in_range_command_passes_unchanged() {
        let c = ArmConfig::maker_arm_v1();
        let mut cmd = zero_cmd(7);
        cmd[0] = JointCommand {
            pos: 1.0,
            vel: 0.5,
            kp: 60.0,
            kd: 4.0,
            tau: 1.0,
        };
        cmd[1].pos = 0.0; // J2: -2.024..0.979
        cmd[2].pos = 4.0; // J3: 3.882..7.955
        cmd[3].pos = 0.0; // J4: -0.832..2.122
        cmd[4].pos = 1.0; // J5: 0.577..3.641
        cmd[5].pos = 2.0; // J6: 0.966..6.292
        cmd[6].pos = -1.0; // gripper range is negative
        let (out, clamped) = clamp_command(&cmd, &c).unwrap();
        assert!(!clamped);
        assert_eq!(out[0].pos, 1.0);
        assert_eq!(out[0].tau, 1.0);
    }

    #[test]
    fn every_field_is_clamped() {
        let c = ArmConfig::maker_arm_v1();
        let mut cmd = zero_cmd(7);
        // J1 soft limits -0.668..4.818, tau_max 4.0
        cmd[0] = JointCommand {
            pos: 99.0,
            vel: -99.0,
            kp: 9999.0,
            kd: 99.0,
            tau: -99.0,
        };
        cmd[6].pos = 0.0; // above gripper q_hi = -0.039
        let (out, clamped) = clamp_command(&cmd, &c).unwrap();
        assert!(clamped);
        assert_eq!(out[0].pos, 4.818);
        assert_eq!(out[0].vel, -5.0);
        assert_eq!(out[0].kp, c.kp_max);
        assert_eq!(out[0].kd, c.kd_max);
        assert_eq!(out[0].tau, -4.0);
        assert_eq!(out[6].pos, -0.039);
    }

    #[test]
    fn limit_margin_shrinks_the_window() {
        let mut c = ArmConfig::maker_arm_v1();
        c.limit_margin = 0.1;
        let mut cmd = zero_cmd(7);
        cmd[0].pos = 4.818;
        cmd[6].pos = -1.0;
        let (out, clamped) = clamp_command(&cmd, &c).unwrap();
        assert!(clamped);
        assert!((out[0].pos - (4.818 - 0.1)).abs() < 1e-12);
    }

    #[test]
    fn non_finite_is_an_error_not_a_command() {
        // float_to_u16 maps NaN to the range MINIMUM (full negative rail) —
        // a NaN escaping a controller must become an error here, never a
        // motor command (MA0 carry-forward).
        let c = ArmConfig::maker_arm_v1();
        for (field, make) in [
            (
                "pos",
                &(|j: &mut JointCommand| j.pos = f64::NAN) as &dyn Fn(&mut JointCommand),
            ),
            ("vel", &|j: &mut JointCommand| j.vel = f64::INFINITY),
            ("kp", &|j: &mut JointCommand| j.kp = f64::NEG_INFINITY),
            ("kd", &|j: &mut JointCommand| j.kd = f64::NAN),
            ("tau", &|j: &mut JointCommand| j.tau = f64::NAN),
        ] {
            let mut cmd = zero_cmd(7);
            cmd[6].pos = -1.0;
            make(&mut cmd[3]);
            match clamp_command(&cmd, &c) {
                Err(ClampError::NonFinite { joint: 3, field: f }) => assert_eq!(f, field),
                other => panic!("{field}: expected NonFinite, got {other:?}"),
            }
        }
    }

    #[test]
    fn wrong_length_is_an_error() {
        let c = ArmConfig::maker_arm_v1();
        match clamp_command(&zero_cmd(6), &c) {
            Err(ClampError::WrongLength {
                got: 6,
                expected: 7,
            }) => {}
            other => panic!("expected WrongLength, got {other:?}"),
        }
    }

    #[test]
    fn negative_gains_clamp_to_zero() {
        let c = ArmConfig::maker_arm_v1();
        let mut cmd = zero_cmd(7);
        cmd[6].pos = -1.0;
        cmd[2] = JointCommand {
            pos: 4.0,
            vel: 0.0,
            kp: -5.0,
            kd: -1.0,
            tau: 0.0,
        };
        let (out, clamped) = clamp_command(&cmd, &c).unwrap();
        assert!(clamped);
        assert_eq!(out[2].kp, 0.0);
        assert_eq!(out[2].kd, 0.0);
    }
}
