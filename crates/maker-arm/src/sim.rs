//! SimArm: an in-memory fake of the 7-motor Maker Arm bus over the MIT
//! protocol, implementing `CanBackend`. It answers the frames the session
//! sends: the non-clearing fault query with a fault status, every other
//! command and every control frame with feedback, and parameter frames
//! from a store (only where the firmware has parameter access). Test hooks
//! inject faults, temperature, silence, mode and legacy firmware. Motion is
//! an ideal servo: an enabled motor snaps to the commanded position.
//! Anything needing real dynamics belongs on the real arm (MA1), not here.

use crate::config::ArmConfig;
use maker_arm_protocol::{self as p, mit};
use maker_arm_transport::{CanBackend, TransportError};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

type Control = (f64, f64, f64, f64, f64);

struct SimMotor {
    model: p::MotorParams,
    position: f64,
    torque: f64,
    temperature: f64,
    enabled: bool,
    fault_word: u32,
    muted: bool,
    forced_mode: Option<u8>,
    params: HashMap<u16, [u8; 4]>,
    /// The last control frame received while disabled.
    pending: Option<Control>,
    /// `pending` as it was at the latest enable.
    preload: Option<Control>,
    enables: u32,
    disables: u32,
}

pub struct SimArm {
    motors: HashMap<u8, SimMotor>,
    /// Firmware with parameter access; legacy firmware also reports mode,
    /// fault flag and warning flag as 0.
    params_supported: bool,
    replies: VecDeque<(u32, [u8; 8])>,
}

impl SimArm {
    pub fn new(config: &ArmConfig) -> SimArm {
        let motors = config
            .joints
            .iter()
            .map(|j| {
                let mut store = HashMap::new();
                store.insert(p::param_index::RUN_MODE, [0u8, 0, 0, 0]);
                store.insert(p::param_index::CAN_TIMEOUT, 0u32.to_le_bytes());
                store.insert(p::param_index::VBUS, 24.0f32.to_le_bytes());
                (
                    j.motor_id,
                    SimMotor {
                        model: *j.model.params(),
                        position: j.to_motor((j.q_lo + j.q_hi) / 2.0),
                        torque: 0.0,
                        temperature: 35.0,
                        enabled: false,
                        fault_word: 0,
                        muted: false,
                        forced_mode: None,
                        params: store,
                        pending: None,
                        preload: None,
                        enables: 0,
                        disables: 0,
                    },
                )
            })
            .collect();
        SimArm {
            motors,
            params_supported: true,
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

    pub fn set_params_supported(&mut self, on: bool) {
        self.params_supported = on;
    }

    pub fn inject_fault(&mut self, motor_id: u8, word: u32) {
        self.motors.get_mut(&motor_id).unwrap().fault_word = word;
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

    /// Queues any frame as if a motor had sent it.
    pub fn queue_raw(&mut self, id: u32, data: [u8; 8]) {
        self.replies.push_back((id, data));
    }

    pub fn enables(&self, motor_id: u8) -> u32 {
        self.motors[&motor_id].enables
    }

    pub fn disables(&self, motor_id: u8) -> u32 {
        self.motors[&motor_id].disables
    }

    /// The last control frame (pos, vel, kp, kd, tau) the motor received
    /// while disabled, before its latest enable.
    pub fn preload(&self, motor_id: u8) -> Option<Control> {
        self.motors[&motor_id].preload
    }

    /// Power off and on: every motor is disabled, its fault and CAN_TIMEOUT
    /// reset, and it announces itself on its motor id.
    pub fn power_cycle(&mut self) {
        let mut ids: Vec<u8> = self.motors.keys().copied().collect();
        ids.sort();
        self.power_cycle_motors(&ids);
    }

    /// `power_cycle` for these motors only, in this order: as in the
    /// supply's restart loop, where only some motors live long enough to
    /// announce themselves.
    pub fn power_cycle_motors(&mut self, ids: &[u8]) {
        for &id in ids {
            let m = self.motors.get_mut(&id).unwrap();
            m.enabled = false;
            m.torque = 0.0;
            m.fault_word = 0;
            m.params
                .insert(p::param_index::CAN_TIMEOUT, 0u32.to_le_bytes());
            // A synthetic MCU id, never a real motor's.
            self.replies
                .push_back((id as u32, [0xA5, 0x5A, 0, 0, 0, 0, 0, id]));
        }
    }

    fn queue_feedback(&mut self, motor_id: u8) {
        let m = &self.motors[&motor_id];
        if m.muted {
            return;
        }
        let legacy = !self.params_supported;
        let fb = mit::Feedback {
            motor_id,
            position: m.position,
            velocity: 0.0,
            torque: m.torque,
            temperature: m.temperature,
            mode: match (legacy, m.forced_mode) {
                (true, _) => 0,
                (false, Some(mode)) => mode,
                (false, None) => 2 * m.enabled as u8,
            },
            fault: !legacy && m.fault_word != 0,
            warning: false,
        };
        let f = mit::encode_feedback(&fb, &m.model);
        self.replies.push_back((f.id, f.data));
    }

    fn command(&mut self, target: u8, arg: u8, op: u8) {
        let m = self.motors.get_mut(&target).unwrap();
        match op {
            0xFB if arg == 0x00 => {
                if !m.muted {
                    let f = mit::encode_fault_status(target, m.fault_word);
                    self.replies.push_back((f.id, f.data));
                }
                return;
            }
            0xFB => m.fault_word = 0,
            0xFC => {
                m.enabled = true;
                m.enables += 1;
                m.preload = m.pending.take();
            }
            0xFD => {
                m.enabled = false;
                m.torque = 0.0;
                m.disables += 1;
            }
            0xFE => m.position = 0.0,
            _ => return,
        }
        self.queue_feedback(target);
    }

    fn param(&mut self, id: u32, target: u8, data: &[u8; 8]) {
        let m = self.motors.get_mut(&target).unwrap();
        let index = u16::from_le_bytes([data[0], data[1]]);
        let write = id & 0x780 == mit::PARAM_WRITE_ID;
        if write {
            m.params.insert(index, [data[4], data[5], data[6], data[7]]);
        }
        let Some(&raw) = m.params.get(&index) else {
            return;
        };
        if !m.muted {
            let f = mit::encode_param_reply(&mit::ParamReply {
                motor_id: target,
                index,
                raw,
                write,
            });
            self.replies.push_back((f.id, f.data));
        }
    }
}

impl CanBackend for SimArm {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        let target = (id & 0x7F) as u8;
        if !self.motors.contains_key(&target) {
            return Ok(());
        }
        match id & 0x780 {
            0 if data[..6] == [0xFF; 6] => self.command(target, data[6], data[7]),
            0 => {
                let m = self.motors.get_mut(&target).unwrap();
                let c = mit::decode_control(data, &m.model).expect("not a command");
                if m.enabled {
                    m.position = c.0;
                    m.torque = c.4;
                } else {
                    m.pending = Some(c);
                }
                self.queue_feedback(target);
            }
            mit::PARAM_READ_ID | mit::PARAM_WRITE_ID if self.params_supported => {
                self.param(id, target, data)
            }
            _ => {}
        }
        Ok(())
    }

    fn recv(&mut self, _timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        Ok(self.replies.pop_front())
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ArmConfig;
    use maker_arm_protocol::mit::{self, Parsed};
    use maker_arm_protocol::{param_index, ParamValue, RS00, RS02};
    use maker_arm_transport::CanBackend;
    use std::time::Duration;

    fn recv_parsed(sim: &mut SimArm, model: &p::MotorParams) -> Parsed {
        let (id, data) = sim
            .recv(Duration::ZERO)
            .unwrap()
            .expect("sim should have replied");
        mit::parse_frame(id, &data, model).expect("parseable reply")
    }

    fn feedback(sim: &mut SimArm, model: &p::MotorParams) -> mit::Feedback {
        match recv_parsed(sim, model) {
            Parsed::Feedback(fb) => fb,
            other => panic!("expected feedback, got {other:?}"),
        }
    }

    fn send(sim: &mut SimArm, f: p::Frame) {
        sim.send(f.id, &f.data).unwrap();
    }

    #[test]
    fn disable_replies_with_feedback() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_position(3, 5.0);
        send(&mut sim, mit::encode_disable(3));
        let fb = feedback(&mut sim, &RS02);
        assert_eq!(fb.motor_id, 3);
        assert_eq!(fb.mode, 0);
        assert!((fb.position - 5.0).abs() < 1e-3);
        assert!(!sim.enabled(3));
        assert_eq!(sim.disables(3), 1);
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
    }

    #[test]
    fn enable_then_control_moves_the_ideal_servo() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_position(1, 1.0);
        send(&mut sim, mit::encode_enable(1));
        assert_eq!(feedback(&mut sim, &RS00).mode, 2);
        assert!(sim.enabled(1));
        assert_eq!(sim.enables(1), 1);
        send(
            &mut sim,
            mit::encode_control(1, 2.0, 0.0, 60.0, 4.0, 0.5, &RS00).unwrap(),
        );
        let fb = feedback(&mut sim, &RS00);
        assert!((fb.position - 2.0).abs() < 1e-3);
        assert!((fb.torque - 0.5).abs() < 1e-2);
        assert!((sim.position(1) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn control_while_disabled_replies_but_does_not_move() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_position(1, 1.0);
        send(
            &mut sim,
            mit::encode_control(1, 2.0, 0.0, 60.0, 4.0, 0.0, &RS00).unwrap(),
        );
        assert!((feedback(&mut sim, &RS00).position - 1.0).abs() < 1e-3);
        assert!((sim.position(1) - 1.0).abs() < 1e-3);
        // Not enabled yet, so nothing counts as a preload.
        assert_eq!(sim.preload(1), None);
        send(&mut sim, mit::encode_enable(1));
        let (pos, _, kp, _, _) = sim.preload(1).expect("the frame before the enable");
        assert!((pos - 2.0).abs() < 1e-3);
        assert!((kp - 60.0).abs() < 0.2);
    }

    #[test]
    fn fault_query_answers_without_clearing() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.inject_fault(3, 1 << 14);
        for _ in 0..2 {
            send(&mut sim, mit::encode_fault_query(3));
            assert_eq!(
                recv_parsed(&mut sim, &RS02),
                Parsed::Fault(mit::FaultStatus {
                    motor_id: 3,
                    word: 1 << 14
                })
            );
        }
        // The feedback flag shows the fault too.
        send(&mut sim, mit::encode_disable(3));
        assert!(feedback(&mut sim, &RS02).fault);
    }

    #[test]
    fn clear_fault_clears_and_answers_with_feedback() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.inject_fault(5, 1 << 2);
        send(&mut sim, mit::encode_clear_fault(5));
        let fb = feedback(&mut sim, &RS00);
        assert_eq!(fb.motor_id, 5);
        assert!(!fb.fault);
        send(&mut sim, mit::encode_fault_query(5));
        assert_eq!(
            recv_parsed(&mut sim, &RS00),
            Parsed::Fault(mit::FaultStatus {
                motor_id: 5,
                word: 0
            })
        );
    }

    #[test]
    fn legacy_motors_ignore_parameter_frames_and_leave_status_bits_zero() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_params_supported(false);
        send(&mut sim, mit::encode_read_param(7, param_index::RUN_MODE));
        send(
            &mut sim,
            mit::encode_write_param(7, param_index::CAN_TIMEOUT, ParamValue::U32(10000)),
        );
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
        assert_eq!(sim.param_u32(7, param_index::CAN_TIMEOUT), Some(0));
        // Enabled and faulted, yet mode, fault flag and warning flag read 0.
        sim.inject_fault(7, 1 << 14);
        send(&mut sim, mit::encode_enable(7));
        let (id, data) = sim.recv(Duration::ZERO).unwrap().expect("feedback");
        assert_eq!(id, p::HOST_CAN_ID as u32);
        assert_eq!(data[6] >> 4, 0);
        // The fault word still answers.
        send(&mut sim, mit::encode_fault_query(7));
        assert_eq!(
            recv_parsed(&mut sim, &RS00),
            Parsed::Fault(mit::FaultStatus {
                motor_id: 7,
                word: 1 << 14
            })
        );
    }

    #[test]
    fn param_read_write_and_store() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        send(
            &mut sim,
            mit::encode_write_param(2, param_index::CAN_TIMEOUT, ParamValue::U32(4000)),
        );
        let Parsed::Param(ack) = recv_parsed(&mut sim, &RS02) else {
            panic!("expected a write acknowledgement");
        };
        assert!(ack.write);
        assert_eq!(sim.param_u32(2, param_index::CAN_TIMEOUT), Some(4000));
        send(
            &mut sim,
            mit::encode_read_param(2, param_index::CAN_TIMEOUT),
        );
        let Parsed::Param(r) = recv_parsed(&mut sim, &RS02) else {
            panic!("expected param reply");
        };
        assert_eq!(
            (r.motor_id, r.index, r.write),
            (2, param_index::CAN_TIMEOUT, false)
        );
        assert_eq!(r.as_u32(), 4000);
        assert_eq!(sim.param_u8(2, param_index::RUN_MODE), Some(0));
    }

    #[test]
    fn seeded_param_defaults_are_pinned() {
        // The gripper watchdog restore and `doctor` read these seeds.
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        assert_eq!(sim.param_u32(2, param_index::CAN_TIMEOUT), Some(0));
        send(&mut sim, mit::encode_read_param(2, param_index::VBUS));
        let Parsed::Param(r) = recv_parsed(&mut sim, &RS02) else {
            panic!("expected param reply");
        };
        assert_eq!((r.motor_id, r.index), (2, param_index::VBUS));
        assert_eq!(r.as_f32(), 24.0);
    }

    #[test]
    fn unknown_motor_and_muted_motor_do_not_reply() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        send(&mut sim, mit::encode_disable(9));
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
        sim.set_muted(4, true);
        send(&mut sim, mit::encode_disable(4));
        send(&mut sim, mit::encode_fault_query(4));
        send(&mut sim, mit::encode_read_param(4, param_index::RUN_MODE));
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
    }

    #[test]
    fn unknown_command_to_a_valid_motor_is_ignored() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.inject_fault(2, 0x05);
        let before_pos = sim.position(2);
        sim.send(2, &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFA])
            .unwrap();
        assert_eq!(sim.recv(Duration::ZERO).unwrap(), None);
        assert_eq!(sim.position(2), before_pos);
        assert!(!sim.enabled(2));
        assert_eq!(sim.motors[&2].fault_word, 0x05);
    }

    #[test]
    fn fault_injection_and_forced_mode_show_in_feedback() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.inject_fault(5, 1 << 14);
        sim.set_temperature(5, 71.0);
        sim.force_mode(5, Some(1));
        send(&mut sim, mit::encode_disable(5));
        let fb = feedback(&mut sim, &RS00);
        assert!(fb.fault);
        assert_eq!(fb.mode, 1);
        assert!((fb.temperature - 71.0).abs() < 1e-9);
    }

    #[test]
    fn set_zero_zeroes_position() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        sim.set_position(6, 3.0);
        send(&mut sim, mit::encode_set_zero(6));
        assert!(feedback(&mut sim, &RS00).position.abs() < 1e-3);
    }

    #[test]
    fn power_cycle_disables_resets_and_announces() {
        let c = ArmConfig::maker_arm_v1();
        let mut sim = SimArm::new(&c);
        send(&mut sim, mit::encode_enable(7));
        send(
            &mut sim,
            mit::encode_write_param(7, param_index::CAN_TIMEOUT, ParamValue::U32(10000)),
        );
        sim.inject_fault(7, 1 << 14);
        while sim.recv(Duration::ZERO).unwrap().is_some() {}
        sim.power_cycle();
        assert!(!sim.enabled(7));
        assert_eq!(sim.param_u32(7, param_index::CAN_TIMEOUT), Some(0));
        assert_eq!(sim.motors[&7].fault_word, 0);
        let ids: Vec<u32> = std::iter::from_fn(|| sim.recv(Duration::ZERO).unwrap())
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, (1..=7).collect::<Vec<u32>>());
    }
}
