//! Requires a vcan0 interface (CI brings one up; locally:
//!   sudo modprobe vcan && sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
//! Run with: cargo test -p maker-arm-transport --features socketcan -- --ignored
#![cfg(feature = "socketcan")]

use maker_arm_transport::{CanBackend, SocketCanBackend};
use socketcan::{CanFrame, CanSocket, EmbeddedFrame, ExtendedId, Socket, StandardId};
use std::sync::{mpsc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

/// vcan0 is a shared broadcast loopback: every socket bound to it sees every
/// frame. The rejection tests below assert that *nothing* arrives, so they
/// must not run while another test is transmitting. Serialize the whole
/// binary on one lock (poison is irrelevant — the guard protects a bus, not
/// data).
static BUS: Mutex<()> = Mutex::new(());

fn bus() -> MutexGuard<'static, ()> {
    BUS.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
#[ignore = "needs vcan0"]
fn roundtrip_extended_frame_over_vcan() {
    let _bus = bus();
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

/// `recv(Duration::ZERO)` must poll, not block forever. Driven from a worker
/// thread with a join deadline so a regression fails the test instead of
/// hanging the suite (`set_read_timeout(ZERO)` used to block indefinitely,
/// because Linux reads an all-zero `SO_RCVTIMEO` as "no timeout").
#[test]
#[ignore = "needs vcan0"]
fn zero_timeout_recv_returns_promptly_on_a_quiet_bus() {
    let _bus = bus();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut sock = SocketCanBackend::open("vcan0").expect("open");
        let started = Instant::now();
        let got = sock.recv(Duration::ZERO).expect("recv");
        let _ = tx.send((got, started.elapsed()));
    });
    let (got, elapsed) = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("recv(Duration::ZERO) did not return within 5 s");
    assert_eq!(got, None, "quiet bus must yield no frame");
    assert!(
        elapsed < Duration::from_millis(250),
        "recv(Duration::ZERO) took {elapsed:?}; it must not wait"
    );
}

/// Same poll, but with a frame already queued: a zero timeout still delivers.
#[test]
#[ignore = "needs vcan0"]
fn zero_timeout_recv_still_delivers_a_queued_frame() {
    let _bus = bus();
    let mut rx = SocketCanBackend::open("vcan0").expect("open rx");
    let mut tx = SocketCanBackend::open("vcan0").expect("open tx");
    let id = 0x028001FD;
    let data = [0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x01, 0x59];
    tx.send(id, &data).expect("send");
    // Give the loopback frame a moment to land in the rx socket's queue.
    let deadline = Instant::now() + Duration::from_secs(1);
    let got = loop {
        if let Some(f) = rx.recv(Duration::ZERO).expect("recv") {
            break f;
        }
        assert!(Instant::now() < deadline, "queued frame never arrived");
    };
    assert_eq!(got, (id, data));
}

/// Raw sender for the frames `SocketCanBackend::send` refuses to build.
fn raw_send(frame: CanFrame) {
    let sock = CanSocket::open("vcan0").expect("open raw tx");
    sock.write_frame(&frame).expect("raw write");
}

#[test]
#[ignore = "needs vcan0"]
fn remote_frames_are_not_delivered_as_motor_data() {
    let _bus = bus();
    let mut rx = SocketCanBackend::open("vcan0").expect("open rx");
    // Extended-id RTR frame at the golden feedback id: an empty payload that
    // would zero-pad into position -12.57 rad / velocity -33 rad/s /
    // torque -14 Nm if it were accepted.
    let id = ExtendedId::new(0x028001FD).unwrap();
    raw_send(CanFrame::new_remote(id, 8).expect("build rtr"));
    assert_eq!(rx.recv(Duration::from_millis(200)).expect("recv"), None);
}

#[test]
#[ignore = "needs vcan0"]
fn standard_id_frames_are_not_delivered_as_motor_data() {
    let _bus = bus();
    let mut rx = SocketCanBackend::open("vcan0").expect("open rx");
    // Standard id 0x001 must not surface as extended id 0x00000001.
    let sid = StandardId::new(0x001).unwrap();
    raw_send(CanFrame::new(sid, &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]).unwrap());
    assert_eq!(rx.recv(Duration::from_millis(200)).expect("recv"), None);
}

/// The rejections above must not be "nothing ever arrives": a good frame
/// sent right after a bad one still comes through.
#[test]
#[ignore = "needs vcan0"]
fn a_good_frame_after_a_rejected_one_still_arrives() {
    let _bus = bus();
    let mut rx = SocketCanBackend::open("vcan0").expect("open rx");
    raw_send(CanFrame::new_remote(ExtendedId::new(0x028001FD).unwrap(), 8).expect("build rtr"));
    raw_send(
        CanFrame::new(StandardId::new(0x001).unwrap(), &[0xFF; 8]).expect("build standard frame"),
    );
    let id = 0x01800001;
    let data = [0x80, 0x00, 0x80, 0x00, 0x05, 0x1F, 0x33, 0x33];
    raw_send(CanFrame::new(ExtendedId::new(id).unwrap(), &data).expect("build data frame"));
    // recv may report the rejected frames as Ok(None) before the good one,
    // so keep reading until the deadline.
    let deadline = Instant::now() + Duration::from_secs(1);
    let got = loop {
        if let Some(f) = rx.recv(Duration::from_millis(200)).expect("recv") {
            break f;
        }
        assert!(Instant::now() < deadline, "good frame never arrived");
    };
    assert_eq!(got, (id, data));
}
