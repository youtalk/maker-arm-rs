//! Per-tick health checks, following arm.py's _check_health (vendor SDK
//! 0245316, spec 6.2). Per motor, in this order: feedback age, fault word
//! or flag, fault reply age (three fault poll intervals), temperature,
//! the gripper's warning flag, and mode 2 for 5 ticks where the firmware
//! has parameter access. The gripper stops at the vendor's 55 degC; the
//! 70 degC limit on joints 1 to 6 is ours (design §4 item 3). Checks run
//! in motor order and the first failure wins. A motor that faulted is
//! skipped from then on, as the vendor does, so the checks keep running
//! while the arm holds and a later fault on another motor still shows.

use crate::config::{ArmConfig, JointConfig};
use crate::state::{ArmState, MotorState};

#[derive(Debug, Clone, PartialEq)]
pub enum FaultReason {
    FeedbackTimeout {
        motor_id: u8,
        age: f64,
    },
    MotorFault {
        motor_id: u8,
        word: u32,
    },
    /// No fault status reply for three fault poll intervals.
    FaultReplyTimeout {
        motor_id: u8,
        age: f64,
    },
    /// The gripper's warning flag, a fault as the vendor treats it.
    Warning {
        motor_id: u8,
    },
    ModeNotMotor {
        motor_id: u8,
        ticks: u32,
    },
    OverTemperature {
        motor_id: u8,
        celsius: f64,
    },
    /// A controller produced a command the clamp rejected (non-finite or
    /// wrong length). Raised by the control loop, not by `check`.
    BadCommand {
        detail: String,
    },
}

impl std::fmt::Display for FaultReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FaultReason::FeedbackTimeout { motor_id, age } => {
                write!(
                    f,
                    "motor {motor_id} feedback timeout {age:.3}s -- check bus/power"
                )
            }
            FaultReason::MotorFault { motor_id, word } => {
                write!(f, "motor {motor_id} fault word {word:#010x}")
            }
            FaultReason::FaultReplyTimeout { motor_id, age } => {
                write!(
                    f,
                    "motor {motor_id} sent no fault status for {age:.3}s -- check bus/power"
                )
            }
            FaultReason::Warning { motor_id } => write!(f, "motor {motor_id} warning flag set"),
            FaultReason::ModeNotMotor { motor_id, ticks } => {
                write!(f, "motor {motor_id} not in Motor mode for {ticks} ticks")
            }
            FaultReason::OverTemperature { motor_id, celsius } => {
                write!(f, "motor {motor_id} over temperature: {celsius:.1} degC")
            }
            FaultReason::BadCommand { detail } => write!(f, "bad command: {detail}"),
        }
    }
}

pub struct HealthMonitor {
    bad_mode_ticks: Vec<u32>,
    /// Motors that already faulted. They are skipped from then on, as the
    /// vendor does, so a later fault on another motor still shows.
    faulted: Vec<bool>,
}

impl HealthMonitor {
    pub fn new(config: &ArmConfig) -> HealthMonitor {
        HealthMonitor {
            bad_mode_ticks: vec![0; config.joints.len()],
            faulted: vec![false; config.joints.len()],
        }
    }

    /// The first failure among the motors that have not faulted yet. That
    /// motor is marked faulted.
    pub fn check(&mut self, state: &ArmState, config: &ArmConfig) -> Option<FaultReason> {
        for (i, (m, j)) in state.motors.iter().zip(&config.joints).enumerate() {
            if self.faulted[i] {
                continue;
            }
            if let Some(reason) = self.check_motor(i, m, j, config) {
                self.faulted[i] = true;
                return Some(reason);
            }
        }
        None
    }

    /// True once motor slot `i` has faulted in this session.
    pub(crate) fn faulted(&self, i: usize) -> bool {
        self.faulted[i]
    }

    fn check_motor(
        &mut self,
        i: usize,
        m: &MotorState,
        j: &JointConfig,
        config: &ArmConfig,
    ) -> Option<FaultReason> {
        let gripper = config.gripper.filter(|g| g.motor_id == j.motor_id);
        if m.feedback_age > config.feedback_timeout {
            return Some(FaultReason::FeedbackTimeout {
                motor_id: j.motor_id,
                age: m.feedback_age,
            });
        }
        if m.fault_flag || m.fault_word != 0 {
            return Some(FaultReason::MotorFault {
                motor_id: j.motor_id,
                word: m.fault_word,
            });
        }
        if m.fault_reply_age > 3.0 * config.fault_poll_interval {
            return Some(FaultReason::FaultReplyTimeout {
                motor_id: j.motor_id,
                age: m.fault_reply_age,
            });
        }
        if m.temperature >= gripper.map_or(config.temp_hold_c, |g| g.max_temperature_c) {
            return Some(FaultReason::OverTemperature {
                motor_id: j.motor_id,
                celsius: m.temperature,
            });
        }
        if gripper.is_some() && m.warning_flag {
            return Some(FaultReason::Warning {
                motor_id: j.motor_id,
            });
        }
        // Legacy firmware reports mode 0 even while enabled.
        if m.params && m.mode != 2 {
            self.bad_mode_ticks[i] += 1;
            if self.bad_mode_ticks[i] >= config.mode_fault_ticks {
                return Some(FaultReason::ModeNotMotor {
                    motor_id: j.motor_id,
                    ticks: self.bad_mode_ticks[i],
                });
            }
        } else {
            self.bad_mode_ticks[i] = 0;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ArmConfig;
    use crate::state::{ArmState, MotorState};

    fn healthy(c: &ArmConfig) -> ArmState {
        ArmState {
            motors: c
                .joints
                .iter()
                .map(|_| MotorState {
                    mode: 2,
                    temperature: 35.0,
                    feedback_age: 0.001,
                    fault_reply_age: 0.001,
                    params: true,
                    ..MotorState::stale()
                })
                .collect(),
            tick: 0,
            t: 0.0,
        }
    }

    #[test]
    fn healthy_arm_reports_nothing() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        for _ in 0..10 {
            assert_eq!(h.check(&healthy(&c), &c), None);
        }
    }

    #[test]
    fn stale_feedback_faults() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        let mut s = healthy(&c);
        s.motors[3].feedback_age = 0.26; // > 0.25 s timeout
        match h.check(&s, &c) {
            Some(FaultReason::FeedbackTimeout { motor_id: 4, .. }) => {}
            other => panic!("expected FeedbackTimeout(4), got {other:?}"),
        }
    }

    #[test]
    fn a_fault_word_faults_at_once() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        let mut s = healthy(&c);
        s.motors[4].fault_word = 1 << 14;
        assert_eq!(
            h.check(&s, &c),
            Some(FaultReason::MotorFault {
                motor_id: 5,
                word: 1 << 14
            })
        );
        let mut s = healthy(&c);
        s.motors[2].fault_flag = true;
        assert_eq!(
            h.check(&s, &c),
            Some(FaultReason::MotorFault {
                motor_id: 3,
                word: 0
            })
        );
    }

    #[test]
    fn a_missing_fault_reply_faults_after_three_intervals() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        let mut s = healthy(&c);
        s.motors[3].fault_reply_age = 0.74;
        assert_eq!(h.check(&s, &c), None);
        s.motors[3].fault_reply_age = 0.76;
        assert_eq!(
            h.check(&s, &c),
            Some(FaultReason::FaultReplyTimeout {
                motor_id: 4,
                age: 0.76
            })
        );
    }

    #[test]
    fn the_gripper_overheats_at_55_and_the_joints_at_70() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        let mut s = healthy(&c);
        s.motors[6].temperature = 54.9;
        s.motors[1].temperature = 69.9;
        assert_eq!(h.check(&s, &c), None);
        s.motors[6].temperature = 55.0;
        assert_eq!(
            h.check(&s, &c),
            Some(FaultReason::OverTemperature {
                motor_id: 7,
                celsius: 55.0
            })
        );
        let mut s = healthy(&c);
        s.motors[1].temperature = 70.0;
        assert_eq!(
            h.check(&s, &c),
            Some(FaultReason::OverTemperature {
                motor_id: 2,
                celsius: 70.0
            })
        );
    }

    #[test]
    fn a_gripper_warning_is_a_fault() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        // A joint's warning flag is not a fault.
        let mut s = healthy(&c);
        s.motors[2].warning_flag = true;
        assert_eq!(h.check(&s, &c), None);
        s.motors[6].warning_flag = true;
        assert_eq!(h.check(&s, &c), Some(FaultReason::Warning { motor_id: 7 }));
    }

    #[test]
    fn mode_is_checked_only_with_parameter_access() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        // Legacy firmware reports mode 0 even while enabled.
        let mut s = healthy(&c);
        for m in &mut s.motors {
            m.params = false;
            m.mode = 0;
        }
        for _ in 0..10 {
            assert_eq!(h.check(&s, &c), None);
        }
    }

    #[test]
    fn wrong_mode_needs_five_consecutive_ticks() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        let mut bad = healthy(&c);
        bad.motors[0].mode = 0;
        for _ in 0..4 {
            assert_eq!(h.check(&bad, &c), None);
        }
        // a healthy tick resets the counter
        assert_eq!(h.check(&healthy(&c), &c), None);
        for _ in 0..4 {
            assert_eq!(h.check(&bad, &c), None);
        }
        assert_eq!(
            h.check(&bad, &c),
            Some(FaultReason::ModeNotMotor {
                motor_id: 1,
                ticks: 5
            })
        );
    }

    #[test]
    fn never_heard_from_motor_faults() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        // ArmState with all motors at stale() — never received any feedback
        let s = ArmState {
            motors: vec![MotorState::stale(); c.joints.len()],
            tick: 0,
            t: 0.0,
        };
        // First check should find the stale feedback (age = INFINITY > timeout)
        match h.check(&s, &c) {
            Some(FaultReason::FeedbackTimeout { motor_id: 1, age }) => {
                assert!(age.is_infinite(), "expected infinite age, got {age}");
            }
            other => panic!("expected FeedbackTimeout(1) with infinite age, got {other:?}"),
        }
    }

    #[test]
    fn per_motor_mode_independence() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        // Two motors bad simultaneously on every tick: motors 1 and 2 (indices 0 and 1)
        // report mode != 2, while motors 3..7 (indices 2..6) are healthy.
        // Per-motor Vec: each motor gets 1 tick per call, motor 1 reaches 5 on tick 5,
        // faults with motor_id=1 and ticks=5 (it checks first).
        // Shared counter: increments by 2 per tick (both motors bad), crosses 5 on tick 3,
        // would fault early with wrong tick count.
        for tick in 1..=5 {
            let mut s = healthy(&c);
            s.motors[0].mode = 0; // Motor 1 bad
            s.motors[1].mode = 0; // Motor 2 bad
            if tick < 5 {
                assert_eq!(
                    h.check(&s, &c),
                    None,
                    "tick {}: expected no fault, counters per-motor still accumulating",
                    tick
                );
            } else {
                // Tick 5: motor 1's counter reaches 5 and faults
                assert_eq!(
                    h.check(&s, &c),
                    Some(FaultReason::ModeNotMotor {
                        motor_id: 1,
                        ticks: 5
                    }),
                    "tick 5: motor 1 should fault with exactly 5 ticks"
                );
            }
        }
    }
}
