//! Requires a vcan0 interface (CI brings one up; locally:
//!   sudo modprobe vcan && sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
//! Run with: cargo test -p maker-arm-transport --features socketcan -- --ignored
#![cfg(feature = "socketcan")]

use maker_arm_transport::{CanBackend, SocketCanBackend};
use std::time::Duration;

#[test]
#[ignore = "needs vcan0"]
fn roundtrip_extended_frame_over_vcan() {
    let mut tx = SocketCanBackend::open("vcan0").expect("open tx");
    let mut rx = SocketCanBackend::open("vcan0").expect("open rx");
    // golden MIT frame from the protocol vectors
    let id = 0x01800001;
    let data = [0x80, 0x00, 0x80, 0x00, 0x05, 0x1F, 0x33, 0x33];
    tx.send(id, &data).expect("send");
    let got = rx
        .recv(Duration::from_millis(500))
        .expect("recv")
        .expect("frame before timeout");
    assert_eq!(got, (id, data));
}
