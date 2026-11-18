//! Arm configuration. The `maker_arm_v1` profile is the vendor's
//! `maker_arm_v1.yaml` (makermods-robotics/maker-arm-sdk @ 0245316): joint
//! limits, gains, rates and the gripper rules. The limits are in the vendor
//! zero: folded against the base, gripper fully closed. Torque caps and the
//! over-temperature hold threshold are OUR additions. The vendor loop sends
//! tau_ff = 0 and has no torque caps. `tau_max` covers the vendor gravity caps.
//!
//! Joint-coordinate contract: joint coordinates are the vendor URDF's
//! (`urdf/maker_arm/robot.urdf`), ordered `link_002_joint`..`link_007_joint`
//! = motor ids 1..6. `JointConfig::direction` / `offset` are the motor -> URDF
//! map. In this profile they are identity and `q_lo`/`q_hi` are motor-frame
//! values. The default calibration (direction -1 on J5, offset 0) turns them
//! into URDF coordinates, and `maker-arm-cli export --calibration` applies it.

use maker_arm_protocol::{MotorParams, RS00, RS02};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorModel {
    Rs00,
    Rs02,
}

impl MotorModel {
    pub fn params(&self) -> &'static MotorParams {
        match self {
            MotorModel::Rs00 => &RS00,
            MotorModel::Rs02 => &RS02,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            MotorModel::Rs00 => "RS00",
            MotorModel::Rs02 => "RS02",
        }
    }
}

#[derive(Debug, Clone)]
pub struct JointConfig {
    pub motor_id: u8,
    pub name: &'static str,
    pub model: MotorModel,
    /// +1.0 or -1.0; joint = direction * (motor - offset).
    pub direction: f64,
    /// Motor-frame position of joint zero, radians.
    pub offset: f64,
    /// Soft limits, radians. Motor-frame until MA1 calibration
    /// re-expresses them in URDF joint coordinates (see the module
    /// docstring's joint-coordinate contract).
    pub q_lo: f64,
    pub q_hi: f64,
    /// Default hold/impedance gains from the v1 profile.
    pub kp: f64,
    pub kd: f64,
    /// Experiment torque cap, Nm — below the model rating on purpose.
    pub tau_max: f64,
}

impl JointConfig {
    pub fn to_motor(&self, joint_pos: f64) -> f64 {
        self.offset + self.direction * joint_pos
    }

    pub fn to_joint(&self, motor_pos: f64) -> f64 {
        self.direction * (motor_pos - self.offset)
    }
}

/// The vendor's gripper rules. The gripper is also released on a fault,
/// and its warning flag counts as a fault.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GripperRules {
    pub motor_id: u8,
    /// Motor-side watchdog written to CAN_TIMEOUT at enable where the
    /// firmware has parameter access (ms).
    pub can_timeout_ms: u32,
    pub max_temperature_c: f64,
    /// Enable needs the gripper below this temperature.
    pub enable_temperature_c: f64,
}

#[derive(Debug, Clone)]
pub struct ArmConfig {
    /// Index 0..=6 = J1..J6, gripper. Motor ids 1..=7.
    pub joints: Vec<JointConfig>,
    pub control_rate_hz: f64,
    /// rad/s, applied to every joint by the clamp.
    pub max_velocity: f64,
    /// Seconds; feedback older than this is a health fault.
    pub feedback_timeout: f64,
    /// Seconds between non-clearing fault queries to one motor.
    pub fault_poll_interval: f64,
    /// Subtracted from q_hi / added to q_lo by the clamp (rad).
    pub limit_margin: f64,
    /// Connect/enable position tolerance beyond soft limits (rad).
    pub enable_limit_grace: f64,
    /// Consecutive ticks of mode != 2 before faulting.
    pub mode_fault_ticks: u32,
    /// Our addition: a joint at or above this temperature (°C) trips an
    /// automatic hold (design §4 item 3). The gripper uses its own
    /// `GripperRules::max_temperature_c`.
    pub temp_hold_c: f64,
    /// Our addition: experiment gain ceiling for kp (design §2) — well below
    /// the protocol's KP_MAX = 500.0, tuned by hardware sessions under
    /// pre-registered promotion criteria.
    pub kp_max: f64,
    /// Clamp ceiling for kd: tracks the protocol's KD_MAX constant.
    pub kd_max: f64,
    /// How long the session waits for one motor's reply before it sends the
    /// next frame (µs). Every motor replies on the same CAN id, so two
    /// replies on the bus at once collide. A missing reply is not a fault by
    /// itself: the health checks decide.
    pub reply_wait_us: u64,
    /// On health fault: freeze targets and keep holding (true, default)
    /// versus disable outright (false).
    pub hold_on_fault: bool,
    pub host_id: u8,
    pub gripper: Option<GripperRules>,
}

impl ArmConfig {
    pub fn maker_arm_v1() -> ArmConfig {
        let j = |motor_id: u8,
                 name: &'static str,
                 model: MotorModel,
                 q_lo: f64,
                 q_hi: f64,
                 kp: f64,
                 kd: f64,
                 tau_max: f64| JointConfig {
            motor_id,
            name,
            model,
            direction: 1.0,
            offset: 0.0,
            q_lo,
            q_hi,
            kp,
            kd,
            tau_max,
        };
        ArmConfig {
            joints: vec![
                j(
                    1,
                    "j1",
                    MotorModel::Rs00,
                    (-158.2f64).to_radians(),
                    156.1f64.to_radians(),
                    60.0,
                    4.0,
                    4.0,
                ),
                j(
                    2,
                    "j2",
                    MotorModel::Rs02,
                    (-175.3f64).to_radians(),
                    (-3.2f64).to_radians(),
                    150.0,
                    4.5,
                    15.0,
                ),
                j(
                    3,
                    "j3",
                    MotorModel::Rs02,
                    2.8f64.to_radians(),
                    236.1f64.to_radians(),
                    90.0,
                    3.0,
                    10.0,
                ),
                j(
                    4,
                    "j4",
                    MotorModel::Rs00,
                    (-65.2f64).to_radians(),
                    104.0f64.to_radians(),
                    30.0,
                    2.0,
                    4.0,
                ),
                j(
                    5,
                    "j5",
                    MotorModel::Rs00,
                    (-97.6f64).to_radians(),
                    78.0f64.to_radians(),
                    30.0,
                    2.0,
                    4.0,
                ),
                j(
                    6,
                    "j6",
                    MotorModel::Rs00,
                    (-153.2f64).to_radians(),
                    152.0f64.to_radians(),
                    30.0,
                    2.0,
                    4.0,
                ),
                j(
                    7,
                    "gripper",
                    MotorModel::Rs00,
                    (-96.0f64).to_radians(),
                    (-2.5f64).to_radians(),
                    20.0,
                    0.5,
                    2.0,
                ),
            ],
            control_rate_hz: 200.0,
            max_velocity: 5.0,
            feedback_timeout: 0.25,
            fault_poll_interval: 0.25,
            limit_margin: 0.0,
            enable_limit_grace: 0.35,
            mode_fault_ticks: 5,
            temp_hold_c: 70.0,
            kp_max: 200.0,
            kd_max: maker_arm_protocol::KD_MAX,
            reply_wait_us: 3000,
            hold_on_fault: true,
            host_id: maker_arm_protocol::HOST_CAN_ID,
            gripper: Some(GripperRules {
                motor_id: 7,
                can_timeout_ms: 500,
                max_temperature_c: 55.0,
                enable_temperature_c: 45.0,
            }),
        }
    }

    pub fn joint_by_motor_id(&self, motor_id: u8) -> Option<&JointConfig> {
        self.joints.iter().find(|j| j.motor_id == motor_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_profile_matches_the_vendor_yaml() {
        // maker_arm_v1.yaml at 0245316: (motor_id, model, lo_deg, hi_deg, kp, kd).
        // This test asserts every joint, so a local edit of any value fails it.
        let c = ArmConfig::maker_arm_v1();
        #[rustfmt::skip]
        let vendor: [(u8, MotorModel, f64, f64, f64, f64); 7] = [
            (1, MotorModel::Rs00, -158.2, 156.1,  60.0, 4.0),
            (2, MotorModel::Rs02, -175.3,  -3.2, 150.0, 4.5),
            (3, MotorModel::Rs02,    2.8, 236.1,  90.0, 3.0),
            (4, MotorModel::Rs00,  -65.2, 104.0,  30.0, 2.0),
            (5, MotorModel::Rs00,  -97.6,  78.0,  30.0, 2.0),
            (6, MotorModel::Rs00, -153.2, 152.0,  30.0, 2.0),
            (7, MotorModel::Rs00,  -96.0,  -2.5,  20.0, 0.5),
        ];
        assert_eq!(c.joints.len(), vendor.len());
        for (i, &(motor_id, model, lo, hi, kp, kd)) in vendor.iter().enumerate() {
            let j = &c.joints[i];
            assert_eq!(j.motor_id, motor_id, "joint index {i}: motor_id");
            assert_eq!(j.model, model, "joint index {i}: model");
            assert_eq!(j.q_lo, f64::to_radians(lo), "joint index {i}: q_lo");
            assert_eq!(j.q_hi, f64::to_radians(hi), "joint index {i}: q_hi");
            assert_eq!(j.kp, kp, "joint index {i}: kp");
            assert_eq!(j.kd, kd, "joint index {i}: kd");
            // The default calibration (direction -1 on J5) is applied by `export`.
            assert_eq!(j.direction, 1.0, "joint index {i}: direction");
            assert_eq!(j.offset, 0.0, "joint index {i}: offset");
        }
        assert_eq!(c.control_rate_hz, 200.0);
        assert_eq!(c.max_velocity, 5.0);
        assert_eq!(c.feedback_timeout, 0.25);
        assert_eq!(c.fault_poll_interval, 0.25);
        assert_eq!(
            c.gripper,
            Some(GripperRules {
                motor_id: 7,
                can_timeout_ms: 500,
                max_temperature_c: 55.0,
                enable_temperature_c: 45.0,
            })
        );
        assert_eq!(c.limit_margin, 0.0);
        assert_eq!(c.enable_limit_grace, 0.35);
        assert_eq!(c.mode_fault_ticks, 5);
        assert_eq!(c.host_id, 0xFD);
        assert_eq!(c.reply_wait_us, 3000);
        assert!(c.hold_on_fault);
    }

    #[test]
    fn torque_caps_cover_the_vendor_gravity_caps() {
        // The vendor's LIMIT_NM for J1..J6. Our caps cover them and stay below the rating.
        let c = ArmConfig::maker_arm_v1();
        let vendor = [0.5, 15.0, 7.5, 2.0, 0.5, 0.5];
        for (j, v) in c.joints.iter().zip(vendor) {
            assert!(j.tau_max >= v, "{}: {} < {v}", j.name, j.tau_max);
            assert!(j.tau_max <= j.model.params().t_max, "{}", j.name);
        }
        for j in &c.joints {
            assert!(j.tau_max > 0.0);
        }
        assert_eq!(c.joints[1].tau_max, 15.0);
        assert!(c.kp_max <= maker_arm_protocol::KP_MAX);
        assert!(c.kd_max <= maker_arm_protocol::KD_MAX);
        assert!(c.temp_hold_c > 0.0);
    }

    #[test]
    fn direction_offset_roundtrip() {
        let j = JointConfig {
            motor_id: 1,
            name: "test",
            model: MotorModel::Rs00,
            direction: -1.0,
            offset: 0.5,
            q_lo: -1.0,
            q_hi: 1.0,
            kp: 10.0,
            kd: 1.0,
            tau_max: 4.0,
        };
        let q = 0.75;
        assert!((j.to_joint(j.to_motor(q)) - q).abs() < 1e-12);
        // motor = offset + direction * joint
        assert!((j.to_motor(0.75) - (0.5 - 0.75)).abs() < 1e-12);
    }

    #[test]
    fn joint_lookup_by_motor_id() {
        let c = ArmConfig::maker_arm_v1();
        assert_eq!(c.joint_by_motor_id(3).unwrap().model, MotorModel::Rs02);
        assert!(c.joint_by_motor_id(8).is_none());
        assert!(c.joint_by_motor_id(0).is_none());
    }
}
