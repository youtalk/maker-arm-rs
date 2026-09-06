//! Arm configuration. The `maker_arm_v1` profile is pinned from the
//! official SDK's maker_arm/profiles/maker_arm_v1.yaml (main @ 2026-08-20):
//! joint limits, gains, directions, and rates are fixed model data there;
//! only motor zero position is user-calibrated. Torque caps and the
//! over-temperature hold threshold are OUR additions (design §2/§4) — the
//! official profile has no torque caps because its loop only ever sends
//! tau_ff = 0.
//!
//! Joint-coordinate contract (maker-arm-lab MS1, 2026-09-05): joint
//! coordinates are the vendor URDF's joint coordinates
//! (makermods-robotics/maker-arm-sdk `urdf/maker_arm/robot.urdf` @ b30d05a),
//! ordered `link_002_joint`..`link_007_joint` = motor ids 1..6.
//! `JointConfig::direction` / `offset` are the motor -> URDF map and are
//! filled by MA1 calibration; until then they are identity, and the
//! `q_lo`/`q_hi` values below are the upstream YAML's MOTOR-frame values,
//! to be re-expressed in URDF coordinates after calibration. Known
//! discrepancy to check on the arm: j3's range width is 4.07 rad here vs
//! 3.14 rad in the URDF.

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

#[derive(Debug, Clone)]
pub struct ArmConfig {
    /// Index 0..=6 = J1..J6, gripper. Motor ids 1..=7.
    pub joints: Vec<JointConfig>,
    pub control_rate_hz: f64,
    /// rad/s, applied to every joint by the clamp.
    pub max_velocity: f64,
    /// Seconds; feedback older than this is a health fault.
    pub feedback_timeout: f64,
    /// Motor-side watchdog written to CAN_TIMEOUT at enable (ms).
    pub motor_can_timeout_ms: u32,
    /// Subtracted from q_hi / added to q_lo by the clamp (rad).
    pub limit_margin: f64,
    /// Connect/enable position tolerance beyond soft limits (rad).
    pub enable_limit_grace: f64,
    /// Consecutive ticks of mode != 2 before faulting.
    pub mode_fault_ticks: u32,
    /// Our addition: any motor at or above this temperature (°C) trips an
    /// automatic hold (design §4 item 3). Not an upstream behavior.
    pub temp_hold_c: f64,
    /// Our addition: experiment gain ceiling for kp (design §2) — well below
    /// the protocol's KP_MAX = 500.0, tuned by hardware sessions under
    /// pre-registered promotion criteria.
    pub kp_max: f64,
    /// Clamp ceiling for kd: tracks the protocol's KD_MAX constant.
    pub kd_max: f64,
    /// Wire spacing between per-motor MIT frames within one tick (µs).
    pub inter_frame_us: u64,
    /// On health fault: freeze targets and keep holding (true, default)
    /// versus disable outright (false).
    pub hold_on_fault: bool,
    pub host_id: u8,
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
                j(1, "j1", MotorModel::Rs00, -0.668, 4.818, 60.0, 4.0, 4.0),
                j(2, "j2", MotorModel::Rs02, -2.024, 0.979, 150.0, 4.5, 6.0),
                j(3, "j3", MotorModel::Rs02, 3.882, 7.955, 90.0, 3.0, 6.0),
                j(4, "j4", MotorModel::Rs00, -0.832, 2.122, 30.0, 2.0, 4.0),
                j(5, "j5", MotorModel::Rs00, 0.577, 3.641, 30.0, 2.0, 4.0),
                j(6, "j6", MotorModel::Rs00, 0.966, 6.292, 30.0, 2.0, 4.0),
                j(
                    7,
                    "gripper",
                    MotorModel::Rs00,
                    -2.092,
                    -0.039,
                    20.0,
                    0.5,
                    2.0,
                ),
            ],
            control_rate_hz: 200.0,
            max_velocity: 5.0,
            feedback_timeout: 0.2,
            motor_can_timeout_ms: 200,
            limit_margin: 0.0,
            enable_limit_grace: 0.35,
            mode_fault_ticks: 5,
            temp_hold_c: 70.0,
            kp_max: 200.0,
            kd_max: maker_arm_protocol::KD_MAX,
            inter_frame_us: 150,
            hold_on_fault: true,
            host_id: maker_arm_protocol::HOST_CAN_ID,
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
    fn v1_profile_matches_upstream_yaml() {
        // Pinned from maker-arm-sdk maker_arm/profiles/maker_arm_v1.yaml
        // (main @ 2026-08-20). Do not edit without re-checking upstream.
        //
        // This test is the ONLY thing standing between the pinned upstream
        // data and a silent local edit, so it asserts EVERY joint --
        // limits, gains, direction, and offset -- not a sample of three.
        // (An earlier version covered J1, J2, and the gripper only, which
        // left a change to J3's q_hi passing CI unnoticed.)
        let c = ArmConfig::maker_arm_v1();
        #[rustfmt::skip]
        let upstream: [(u8, &str, MotorModel, f64, f64, f64, f64); 7] = [
            // motor_id, name, model,           q_lo,   q_hi,    kp,    kd
            (1, "j1",      MotorModel::Rs00, -0.668,  4.818,  60.0, 4.0),
            (2, "j2",      MotorModel::Rs02, -2.024,  0.979, 150.0, 4.5),
            (3, "j3",      MotorModel::Rs02,  3.882,  7.955,  90.0, 3.0),
            (4, "j4",      MotorModel::Rs00, -0.832,  2.122,  30.0, 2.0),
            (5, "j5",      MotorModel::Rs00,  0.577,  3.641,  30.0, 2.0),
            (6, "j6",      MotorModel::Rs00,  0.966,  6.292,  30.0, 2.0),
            (7, "gripper", MotorModel::Rs00, -2.092, -0.039,  20.0, 0.5),
        ];
        assert_eq!(c.joints.len(), upstream.len());
        for (i, &(motor_id, name, model, q_lo, q_hi, kp, kd)) in upstream.iter().enumerate() {
            let j = &c.joints[i];
            assert_eq!(j.motor_id, motor_id, "joint index {i}: motor_id");
            assert_eq!(j.name, name, "joint index {i}: name");
            assert_eq!(j.model, model, "joint index {i}: model");
            assert_eq!(j.q_lo, q_lo, "joint index {i}: q_lo");
            assert_eq!(j.q_hi, q_hi, "joint index {i}: q_hi");
            assert_eq!(j.kp, kp, "joint index {i}: kp");
            assert_eq!(j.kd, kd, "joint index {i}: kd");
            // Upstream ships no flipped axes and no offsets: only the
            // motor's own zero position is user-calibrated, on the motor.
            assert_eq!(j.direction, 1.0, "joint index {i}: direction");
            assert_eq!(j.offset, 0.0, "joint index {i}: offset");
        }
        assert_eq!(c.control_rate_hz, 200.0);
        assert_eq!(c.max_velocity, 5.0);
        assert_eq!(c.feedback_timeout, 0.2);
        assert_eq!(c.motor_can_timeout_ms, 200);
        assert_eq!(c.limit_margin, 0.0);
        assert_eq!(c.enable_limit_grace, 0.35);
        assert_eq!(c.mode_fault_ticks, 5);
        assert_eq!(c.host_id, 0xFD);
        assert_eq!(c.inter_frame_us, 150);
        assert!(c.hold_on_fault);
    }

    #[test]
    fn experiment_torque_caps_are_below_ratings() {
        // Design §2: experiment caps sit below the RS00 (±14) / RS02 (±17)
        // ratings. The clamp enforces these; MA1/MA2 sessions may tune them
        // per pre-registered promotion criteria, never silently.
        let c = ArmConfig::maker_arm_v1();
        for j in &c.joints {
            assert!(j.tau_max > 0.0);
            assert!(j.tau_max < j.model.params().t_max);
        }
        assert_eq!(c.joints[1].tau_max, 6.0); // RS02
        assert_eq!(c.joints[0].tau_max, 4.0); // RS00
        assert_eq!(c.joints[6].tau_max, 2.0); // gripper
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
