//! SimArm: an in-memory fake of the 7-motor Maker Arm bus, implementing
//! `CanBackend`. It emulates the RobStride conventions the session relies
//! on — disable as the read-only probe, feedback replies to
//! enable/disable/MIT/set-zero, param store reads — plus test hooks for
//! fault/temperature/mute/mode injection. Motion is an ideal servo:
//! an enabled motor snaps to the commanded position. Anything needing real
//! dynamics belongs on the real arm (MA1), not here.

use crate::config::ArmConfig;
use maker_arm_protocol as p;
use maker_arm_transport::{CanBackend, TransportError};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

struct SimMotor {
    model: p::MotorParams,
    position: f64,
    torque: f64,
    temperature: f64,
    enabled: bool,
    fault_bits: u8,
    muted: bool,
    forced_mode: Option<u8>,
    params: HashMap<u16, [u8; 4]>,
}

impl SimMotor {
    fn mode(&self) -> u8 {
        self.forced_mode.unwrap_or(if self.enabled { 2 } else { 0 })
    }

    fn feedback(&self, motor_id: u8) -> p::MotorFeedback {
        p::MotorFeedback {
            motor_id,
            position: self.position,
            velocity: 0.0,
            torque: self.torque,
            temperature: self.temperature,
            mode: self.mode(),
            fault_bits: self.fault_bits,
        }
    }
}

pub struct SimArm {
    motors: HashMap<u8, SimMotor>,
    host_id: u8,
    replies: VecDeque<(u32, [u8; 8])>,
}

impl SimArm {
    pub fn new(config: &ArmConfig) -> SimArm {
        let motors = config
            .joints
            .iter()
            .map(|j| {
                let params = *j.model.params();
                let mut store = HashMap::new();
                store.insert(p::param_index::RUN_MODE, [0u8, 0, 0, 0]);
                store.insert(p::param_index::CAN_TIMEOUT, 0u32.to_le_bytes());
                store.insert(p::param_index::VBUS, 24.0f32.to_le_bytes());
                (
                    j.motor_id,
                    SimMotor {
                        model: params,
                        position: j.to_motor((j.q_lo + j.q_hi) / 2.0),
                        torque: 0.0,
                        temperature: 35.0,
                        enabled: false,
                        fault_bits: 0,
                        muted: false,
                        forced_mode: None,
                        params: store,
                    },
                )
            })
            .collect();
        SimArm {
            motors,
            host_id: config.host_id,
            replies: VecDeque::new(),
        }
    }

    pub fn set_position(&mut self, motor_id: u8, motor_pos: f64) {
        self.motors.get_mut(&motor_id).unwrap().position = motor_pos;
    }

    pub fn position(&self, motor_id: u8) -> f64 {
        self.motors[&motor_id].position
    }

    pub fn enabled(&self, motor_id: u8) -> bool {
        self.motors[&motor_id].enabled
    }

    pub fn inject_fault(&mut self, motor_id: u8, bits: u8) {
        self.motors.get_mut(&motor_id).unwrap().fault_bits = bits & 0x3F;
    }

    pub fn set_temperature(&mut self, motor_id: u8, celsius: f64) {
        self.motors.get_mut(&motor_id).unwrap().temperature = celsius;
    }

    pub fn set_muted(&mut self, motor_id: u8, muted: bool) {
        self.motors.get_mut(&motor_id).unwrap().muted = muted;
    }

    pub fn force_mode(&mut self, motor_id: u8, mode: Option<u8>) {
        self.motors.get_mut(&motor_id).unwrap().forced_mode = mode;
    }

    pub fn param_u32(&self, motor_id: u8, index: u16) -> Option<u32> {
        self.motors[&motor_id]
            .params
            .get(&index)
            .map(|raw| u32::from_le_bytes(*raw))
    }

    pub fn param_u8(&self, motor_id: u8, index: u16) -> Option<u8> {
        self.motors[&motor_id].params.get(&index).map(|raw| raw[0])
    }

    fn queue_feedback(&mut self, motor_id: u8) {
        let m = &self.motors[&motor_id];
        if m.muted {
            return;
        }
        let f = p::encode_feedback(&m.feedback(motor_id), &m.model, self.host_id);
        self.replies.push_back((f.id, f.data));
    }
}

impl CanBackend for SimArm {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        let comm = ((id >> 24) & 0x1F) as u8;
        let target = (id & 0xFF) as u8;
        if !self.motors.contains_key(&target) {
            return Ok(());
        }
        match comm {
            p::COMM_ENABLE => {
                self.motors.get_mut(&target).unwrap().enabled = true;
                self.queue_feedback(target);
            }
            p::COMM_DISABLE => {
                let m = self.motors.get_mut(&target).unwrap();
                m.enabled = false;
                m.torque = 0.0;
                if data[0] == 1 {
                    m.fault_bits = 0;
                }
                self.queue_feedback(target);
            }
            p::COMM_MIT => {
                let m = self.motors.get_mut(&target).unwrap();
                if m.enabled {
                    let pos_u16 = u16::from_be_bytes([data[0], data[1]]);
                    m.position = p::u16_to_float(pos_u16, m.model.p_min, m.model.p_max);
                    let tau_u16 = ((id >> 8) & 0xFFFF) as u16;
                    m.torque = p::u16_to_float(tau_u16, m.model.t_min, m.model.t_max);
                }
                self.queue_feedback(target);
            }
            p::COMM_SET_ZERO => {
                self.motors.get_mut(&target).unwrap().position = 0.0;
                self.queue_feedback(target);
            }
            p::COMM_READ_PARAM => {
                let index = u16::from_le_bytes([data[0], data[1]]);
                let m = &self.motors[&target];
                if m.muted {
                    return Ok(());
                }
                if let Some(raw) = m.params.get(&index) {
                    let mut reply = [0u8; 8];
                    reply[0..2].copy_from_slice(&index.to_le_bytes());
                    reply[4..8].copy_from_slice(raw);
                    let rid = p::make_can_id(p::COMM_READ_PARAM, target as u16, self.host_id);
                    self.replies.push_back((rid, reply));
                }
            }
            p::COMM_WRITE_PARAM => {
                let index = u16::from_le_bytes([data[0], data[1]]);
                let raw = [data[4], data[5], data[6], data[7]];
                self.motors
                    .get_mut(&target)
                    .unwrap()
                    .params
                    .insert(index, raw);
            }
            _ => {}
        }
        Ok(())
    }

    fn recv(&mut self, _timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        Ok(self.replies.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ArmConfig;
    use maker_arm_protocol as p;
    use maker_arm_transport::CanBackend;
    use std::time::Duration;

    fn recv_parsed(sim: &mut SimArm, model: &p::MotorParams) -> p::ParsedFrame {
        let (id, data) = sim
            .recv(Duration::ZERO)
            .unwrap()
            .expect("sim should have replied");
        p::parse_frame(id, &data, model).expect("parseable reply")
    }

    #[test]
    fn disable_is_the_readonly_probe() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_position(3, 5.0);
        let f = p::encode_disable(3, false, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::Feedback(fb) = recv_parsed(&mut sim, &p::RS02) else {
            panic!("expected feedback");
        };
        assert_eq!(fb.motor_id, 3);
        assert_eq!(fb.mode, 0);
        assert!((fb.position - 5.0).abs() < 1e-3);
        assert!(!sim.enabled(3));
        // nothing else queued
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
    }

    #[test]
    fn enable_then_mit_moves_the_ideal_servo() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_position(1, 1.0);
        let f = p::encode_enable(1, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::Feedback(fb) = recv_parsed(&mut sim, &p::RS00) else {
            panic!("expected feedback");
        };
        assert_eq!(fb.mode, 2);
        assert!(sim.enabled(1));
        let f = p::encode_mit(1, 2.0, 0.0, 60.0, 4.0, 0.5, &p::RS00);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::Feedback(fb) = recv_parsed(&mut sim, &p::RS00) else {
            panic!("expected feedback");
        };
        assert!((fb.position - 2.0).abs() < 1e-3);
        assert!((fb.torque - 0.5).abs() < 1e-3);
        assert!((sim.position(1) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn mit_while_disabled_replies_but_does_not_move() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_position(1, 1.0);
        let f = p::encode_mit(1, 2.0, 0.0, 60.0, 4.0, 0.0, &p::RS00);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::Feedback(fb) = recv_parsed(&mut sim, &p::RS00) else {
            panic!("expected feedback");
        };
        assert!((fb.position - 1.0).abs() < 1e-3);
        assert!((sim.position(1) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn param_read_write_and_store() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        let f = p::encode_write_param(
            2,
            p::param_index::CAN_TIMEOUT,
            p::ParamValue::U32(4000),
            c.host_id,
        );
        sim.send(f.id, &f.data).unwrap();
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None); // write is fire-and-forget
        assert_eq!(sim.param_u32(2, p::param_index::CAN_TIMEOUT), Some(4000));
        let f = p::encode_read_param(2, p::param_index::CAN_TIMEOUT, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::ParamReply(r) = recv_parsed(&mut sim, &p::RS02) else {
            panic!("expected param reply");
        };
        assert_eq!(r.motor_id, 2);
        assert_eq!(r.index, p::param_index::CAN_TIMEOUT);
        assert_eq!(r.as_u32(), 4000);
        // defaults
        assert_eq!(sim.param_u8(2, p::param_index::RUN_MODE), Some(0));
    }

    #[test]
    fn seeded_param_defaults_are_pinned() {
        // CAN_TIMEOUT and VBUS are seeded at construction (sim.rs); a wrong
        // literal in either would pass silently today and only surface as a
        // confusing failure downstream (Task 5's enable sequence writes and
        // verifies CAN_TIMEOUT; Task 9's `doctor` command reads VBUS).
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        // CAN_TIMEOUT defaults to 0 before any write.
        assert_eq!(sim.param_u32(2, p::param_index::CAN_TIMEOUT), Some(0));
        // VBUS defaults to 24.0, read back over the wire (no param_f32 hook
        // exists, and the brief's hook list is fixed).
        let f = p::encode_read_param(2, p::param_index::VBUS, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::ParamReply(r) = recv_parsed(&mut sim, &p::RS02) else {
            panic!("expected param reply");
        };
        assert_eq!(r.motor_id, 2);
        assert_eq!(r.index, p::param_index::VBUS);
        assert_eq!(r.as_f32(), 24.0);
    }

    #[test]
    fn unknown_motor_and_muted_motor_do_not_reply() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        let f = p::encode_disable(9, false, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
        sim.set_muted(4, true);
        let f = p::encode_disable(4, false, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
    }

    #[test]
    fn unknown_comm_type_to_a_valid_motor_is_ignored() {
        // Distinct from unknown_motor_and_muted_motor_do_not_reply: the motor
        // id here (2) IS one of the arm's seven motors. COMM_SAVE is a real
        // RobStride comm type that SimArm does not implement, so it must
        // fall into send()'s wildcard arm: no reply, and no state mutation.
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.inject_fault(2, 0x05);
        let before_pos = sim.motors[&2].position;
        let before_mode = sim.motors[&2].mode();
        let before_fault = sim.motors[&2].fault_bits;
        let before_can_timeout = sim.param_u32(2, p::param_index::CAN_TIMEOUT);
        let f = p::encode_save_params(2, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
        assert_eq!(sim.motors[&2].position, before_pos);
        assert_eq!(sim.motors[&2].mode(), before_mode);
        assert_eq!(sim.motors[&2].fault_bits, before_fault);
        assert_eq!(
            sim.param_u32(2, p::param_index::CAN_TIMEOUT),
            before_can_timeout
        );
    }

    #[test]
    fn fault_injection_and_forced_mode_show_in_feedback() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.inject_fault(5, 0x21);
        sim.set_temperature(5, 71.0);
        sim.force_mode(5, Some(1));
        let f = p::encode_disable(5, false, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::Feedback(fb) = recv_parsed(&mut sim, &p::RS00) else {
            panic!("expected feedback");
        };
        assert_eq!(fb.fault_bits, 0x21);
        assert_eq!(fb.mode, 1);
        assert!((fb.temperature - 71.0).abs() < 1e-9);
        // disable with clear_fault byte clears injected faults
        sim.force_mode(5, None);
        let f = p::encode_disable(5, true, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::Feedback(fb) = recv_parsed(&mut sim, &p::RS00) else {
            panic!("expected feedback");
        };
        assert_eq!(fb.fault_bits, 0);
    }

    #[test]
    fn set_zero_zeroes_position() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_position(6, 3.0);
        let f = p::encode_set_zero(6, c.host_id);
        sim.send(f.id, &f.data).unwrap();
        let p::ParsedFrame::Feedback(fb) = recv_parsed(&mut sim, &p::RS00) else {
            panic!("expected feedback");
        };
        assert!(fb.position.abs() < 1e-3);
    }
}
