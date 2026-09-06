//! The Controller trait (design §2): production controllers live in Rust
//! inside the control loop. Controllers OUTPUT desires; the loop's clamp
//! decides what actually reaches the motors — no controller may assume its
//! output is applied verbatim.

use crate::config::ArmConfig;
use crate::state::{ArmCommand, ArmState, JointCommand};

pub trait Controller: Send {
    /// Called once per tick with the latest state; returns the desired
    /// command in joint coordinates. `dt` is the nominal tick period (s).
    fn update(&mut self, state: &ArmState, dt: f64) -> ArmCommand;
}

/// Holds a pose with the profile's per-joint kp/kd. The first production
/// controller and the fault-hold primitive.
pub struct HoldController {
    gains: Vec<(f64, f64)>,
    targets: Option<Vec<f64>>,
}

impl HoldController {
    /// Targets are captured from the first `update` call's state.
    pub fn from_config(config: &ArmConfig) -> HoldController {
        HoldController {
            gains: config.joints.iter().map(|j| (j.kp, j.kd)).collect(),
            targets: None,
        }
    }

    pub fn with_targets(config: &ArmConfig, targets: Vec<f64>) -> HoldController {
        HoldController {
            gains: config.joints.iter().map(|j| (j.kp, j.kd)).collect(),
            targets: Some(targets),
        }
    }

    /// Re-captures targets from `state` — the loop calls this when entering
    /// hold-on-fault so the hold is at the last-good pose, not a stale one.
    pub fn retarget_to_state(&mut self, state: &ArmState) {
        self.targets = Some(state.motors.iter().map(|m| m.position).collect());
    }
}

impl Controller for HoldController {
    fn update(&mut self, state: &ArmState, _dt: f64) -> ArmCommand {
        let targets = self
            .targets
            .get_or_insert_with(|| state.motors.iter().map(|m| m.position).collect());
        targets
            .iter()
            .zip(&self.gains)
            .map(|(&pos, &(kp, kd))| JointCommand {
                pos,
                vel: 0.0,
                kp,
                kd,
                tau: 0.0,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ArmConfig;
    use crate::state::{ArmState, MotorState};

    fn state_at(positions: &[f64]) -> ArmState {
        ArmState {
            motors: positions
                .iter()
                .map(|&p| MotorState {
                    position: p,
                    feedback_age: 0.0,
                    ..MotorState::stale()
                })
                .collect(),
            tick: 0,
            t: 0.0,
        }
    }

    #[test]
    fn hold_captures_first_state_and_keeps_it() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HoldController::from_config(&c);
        let cmd1 = h.update(&state_at(&[0.1, -0.2, 4.0, 0.5, 1.0, 2.0, -1.0]), 0.005);
        assert_eq!(cmd1.len(), 7);
        assert_eq!(cmd1[0].pos, 0.1);
        assert_eq!(cmd1[0].kp, 60.0); // profile gains
        assert_eq!(cmd1[0].kd, 4.0);
        assert_eq!(cmd1[0].vel, 0.0);
        assert_eq!(cmd1[0].tau, 0.0);
        // second update with the arm moved: target must NOT follow
        let cmd2 = h.update(&state_at(&[0.9, -0.2, 4.0, 0.5, 1.0, 2.0, -1.0]), 0.005);
        assert_eq!(cmd2[0].pos, 0.1);
    }

    #[test]
    fn with_targets_and_retarget() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HoldController::with_targets(&c, vec![0.0, 0.0, 4.0, 0.0, 1.0, 2.0, -1.0]);
        let s = state_at(&[0.5, -0.5, 5.0, 0.5, 1.5, 2.5, -0.5]);
        assert_eq!(h.update(&s, 0.005)[2].pos, 4.0);
        h.retarget_to_state(&s);
        assert_eq!(h.update(&s, 0.005)[2].pos, 5.0);
    }
}
