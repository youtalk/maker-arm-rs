//! The same stack over a real kernel CAN interface: the session talks
//! SocketCAN on vcan0; a pump thread bridges vcan0 to a SimArm playing the
//! 7 motors. Requires vcan0 (CI brings one up; locally:
//!   sudo modprobe vcan && sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
//! Run: cargo test -p maker-arm --features socketcan -- --ignored
#![cfg(feature = "socketcan")]

use maker_arm::{ArmConfig, HoldController, Session, SessionState, SimArm};
use maker_arm_transport::{CanBackend, SocketCanBackend};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

#[test]
#[ignore = "needs vcan0"]
fn session_ladder_over_vcan() {
    let c = ArmConfig::maker_arm_v1();
    let stop = Arc::new(AtomicBool::new(false));
    // Readiness handshake: CAN has no listen backlog, so a disable/probe
    // frame the host sends before the pump's socket is open is simply
    // lost (not queued for a late joiner), and the session's bounded
    // probe then times out. Block the host's `connect()` until the pump
    // has its socket open and is already polling.
    let (ready_tx, ready_rx) = mpsc::channel();

    // Motor-side pump: its own socket on vcan0, bridging to the simulator.
    let pump_stop = Arc::clone(&stop);
    let pump_config = c.clone();
    let pump = std::thread::spawn(move || {
        let mut sock = SocketCanBackend::open("vcan0").expect("pump socket");
        let mut sim = SimArm::new(&pump_config);
        let _ = ready_tx.send(());
        while !pump_stop.load(Ordering::Relaxed) {
            if let Ok(Some((id, data))) = sock.recv(Duration::from_millis(2)) {
                sim.send(id, &data).expect("sim send");
                while let Ok(Some((rid, rdata))) = sim.recv(Duration::ZERO) {
                    sock.send(rid, &rdata).expect("pump reply");
                }
            }
        }
    });
    ready_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("pump did not become ready");

    // Host side: a single shared socket (the loopback-safe arrangement).
    let backend = SocketCanBackend::open("vcan0").expect("host socket");
    let mut s = Session::connect(Box::new(backend), c.clone()).expect("connect over vcan");
    s.enable().expect("enable over vcan");
    let stop_loop = AtomicBool::new(false);
    let mut hold = HoldController::from_config(&c);
    s.run(&mut hold, &stop_loop, Some(40))
        .expect("40 held ticks at 200 Hz");
    assert_eq!(s.state(), SessionState::Enabled);
    assert!(s.fault().is_none());
    s.disable().expect("release");

    stop.store(true, Ordering::Relaxed);
    pump.join().expect("pump thread");
}
