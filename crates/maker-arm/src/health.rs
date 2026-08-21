//! Per-tick health checks, mirroring arm.py's _check_health (upstream @
//! 2026-08-20): feedback age, fault bits, and mode-not-Motor with a
//! 5-tick debounce. Over-temperature is OUR addition (design §4 item 3);
//! upstream does not check temperature. Checks run in motor order and the
//! first failure wins — one fault at a time is enough to stop on.

use crate::config::ArmConfig;
use crate::state::ArmState;

#[derive(Debug, Clone, PartialEq)]
pub enum FaultReason {
    FeedbackTimeout {
        motor_id: u8,
        age: f64,
    },
    MotorFault {
        motor_id: u8,
        bits: u8,
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
            FaultReason::MotorFault { motor_id, bits } => {
                write!(f, "motor {motor_id} fault bits {bits:#04x}")
            }
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
}

impl HealthMonitor {
    pub fn new(config: &ArmConfig) -> HealthMonitor {
        HealthMonitor {
            bad_mode_ticks: vec![0; config.joints.len()],
        }
    }

    pub fn check(&mut self, state: &ArmState, config: &ArmConfig) -> Option<FaultReason> {
        for (i, (m, j)) in state.motors.iter().zip(&config.joints).enumerate() {
            if m.feedback_age > config.feedback_timeout {
                return Some(FaultReason::FeedbackTimeout {
                    motor_id: j.motor_id,
                    age: m.feedback_age,
                });
            }
            if m.fault_bits != 0 {
                return Some(FaultReason::MotorFault {
                    motor_id: j.motor_id,
                    bits: m.fault_bits,
                });
            }
            if m.temperature >= config.temp_hold_c {
                return Some(FaultReason::OverTemperature {
                    motor_id: j.motor_id,
                    celsius: m.temperature,
                });
            }
            if m.mode != 2 {
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
        s.motors[3].feedback_age = 0.25; // > 0.2 s timeout
        match h.check(&s, &c) {
            Some(FaultReason::FeedbackTimeout { motor_id: 4, .. }) => {}
            other => panic!("expected FeedbackTimeout(4), got {other:?}"),
        }
    }

    #[test]
    fn fault_bits_fault_immediately() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        let mut s = healthy(&c);
        s.motors[4].fault_bits = 0x21;
        assert_eq!(
            h.check(&s, &c),
            Some(FaultReason::MotorFault {
                motor_id: 5,
                bits: 0x21
            })
        );
    }

    #[test]
    fn over_temperature_faults() {
        let c = ArmConfig::maker_arm_v1();
        let mut h = HealthMonitor::new(&c);
        let mut s = healthy(&c);
        s.motors[1].temperature = 70.0; // >= temp_hold_c
        assert_eq!(
            h.check(&s, &c),
            Some(FaultReason::OverTemperature {
                motor_id: 2,
                celsius: 70.0
            })
        );
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
}
