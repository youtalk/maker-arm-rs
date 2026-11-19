use crate::{CanBackend, TransportError, POWER_UP_ID};
use socketcan::{CanFilter, CanFrame, CanSocket, EmbeddedFrame, Socket, SocketOptions, StandardId};
use std::time::Duration;

/// `can_id` bit the kernel sets on extended-id frames.
const CAN_EFF_FLAG: u32 = 0x8000_0000;
/// `can_id` bit the kernel sets on remote-transmission-request frames.
const CAN_RTR_FLAG: u32 = 0x4000_0000;
/// The 29 id bits of an extended `can_id`.
const CAN_EFF_MASK: u32 = 0x1FFF_FFFF;

/// Blocking SocketCAN backend. Standard (11-bit) ids only — the RobStride
/// MIT protocol never uses extended ids, and `recv` enforces that in
/// both directions (see [`accept_frame`]). The one extended frame it
/// delivers is the motors' power-up announcement (`POWER_UP_ID`).
pub struct SocketCanBackend {
    sock: CanSocket,
    /// Last value handed to `set_read_timeout`, so `recv` only pays for the
    /// `setsockopt` when the requested timeout actually changes: at ~1 kHz
    /// the timeout is the same value on every call.
    read_timeout: Option<Duration>,
}

/// Reduces one received frame to what the MIT protocol can carry, or
/// `None` if the frame must not be presented as motor data.
///
/// `CAN_RAW` sockets deliver remote (RTR) frames, extended-id frames
/// and error frames alongside the traffic we want:
///
/// * An RTR frame has an empty payload. Zero-padding it to `[0u8; 8]` would
///   yield a structurally valid feedback reading with every channel pinned
///   at its negative rail.
/// * A standard id `0x001` and an extended id `0x00000001` would both
///   surface as `(1, data)`, i.e. indistinguishable.
///
/// A data frame on extended id 0x7FFE, the power-up announcement, comes
/// back as `POWER_UP_ID`, which no standard id can equal.
fn accept_frame(frame: &CanFrame) -> Option<(u32, [u8; 8])> {
    let data_frame = match frame {
        CanFrame::Data(f) => f,
        CanFrame::Remote(_) | CanFrame::Error(_) => return None,
    };
    let id = match data_frame.id() {
        socketcan::Id::Standard(sid) => u32::from(sid.as_raw()),
        socketcan::Id::Extended(eid) if eid.as_raw() == POWER_UP_ID & CAN_EFF_MASK => POWER_UP_ID,
        socketcan::Id::Extended(_) => return None,
    };
    let mut data = [0u8; 8];
    let d = data_frame.data();
    data[..d.len().min(8)].copy_from_slice(&d[..d.len().min(8)]);
    Some((id, data))
}

/// `Closed` for an interface that is gone or down, `Io` for the rest. The
/// kernel's CAN_RAW answers ENODEV (19) once and then ENXIO (6) after the
/// adapter is unplugged, and ENETDOWN (100) while the interface is down.
fn io_error(e: std::io::Error) -> TransportError {
    match e.raw_os_error() {
        Some(6 | 19 | 100) => TransportError::Closed,
        _ => TransportError::Io(e.to_string()),
    }
}

/// The 11-bit id of a frame, or an error above `0x7FF`.
fn standard_id(id: u32) -> Result<StandardId, TransportError> {
    u16::try_from(id)
        .ok()
        .and_then(StandardId::new)
        .ok_or_else(|| TransportError::Io(format!("id {id:#x} exceeds 11 bits")))
}

impl SocketCanBackend {
    pub fn open(ifname: &str) -> Result<Self, TransportError> {
        let sock = CanSocket::open(ifname)
            .map_err(|e| TransportError::Io(format!("open CAN interface {ifname}: {e}")))?;
        // Kernel-side half of the frame filtering: match only frames whose
        // EFF bit and RTR bit are both clear, plus data frames on extended
        // id 0x7FFE (the power-up announcement), so other extended-id and
        // remote frames are dropped before they reach userspace. The
        // userspace check in `accept_frame` is the authoritative one — this
        // just keeps foreign traffic from eating the caller's timeout.
        sock.set_filters(&[
            CanFilter::new(0, CAN_EFF_FLAG | CAN_RTR_FLAG),
            CanFilter::new(POWER_UP_ID, CAN_EFF_FLAG | CAN_RTR_FLAG | CAN_EFF_MASK),
        ])
            .map_err(|e| TransportError::Io(format!("set CAN filter on {ifname}: {e}")))?;
        Ok(Self {
            sock,
            read_timeout: None,
        })
    }
}

impl CanBackend for SocketCanBackend {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        let frame = CanFrame::new(standard_id(id)?, data)
            .ok_or_else(|| TransportError::Io("frame construction failed".into()))?;
        self.sock.write_frame(&frame).map_err(io_error)
    }

    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        let read = if timeout.is_zero() {
            // `set_read_timeout(Duration::ZERO)` forwards to
            // `setsockopt(SO_RCVTIMEO, {0, 0})`, and Linux reads an all-zero
            // `timeval` as "no timeout at all" — the read then blocks
            // forever, which is the opposite of what a zero timeout means
            // for the other backends. Poll instead: `read_frame_timeout`
            // calls `poll(2)` with a zero timeout, which returns immediately
            // whether or not a frame is queued, and leaves `SO_RCVTIMEO`
            // (and therefore the cached value below) untouched.
            self.sock.read_frame_timeout(Duration::ZERO)
        } else {
            if self.read_timeout != Some(timeout) {
                self.sock.set_read_timeout(timeout).map_err(io_error)?;
                self.read_timeout = Some(timeout);
            }
            self.sock.read_frame()
        };
        match read {
            // A frame this transport does not deliver (remote, extended-id,
            // error) ends the call with `Ok(None)` instead of restarting the
            // read with the remaining budget: that keeps one `recv` bounded
            // by one blocking read and cannot spin on a busy foreign bus.
            // Per the `CanBackend::recv` contract `Ok(None)` never means
            // "the timeout elapsed", so an early return is legal, and with
            // the kernel filter installed by `open` this path is only
            // reachable for a socket that lost its filter.
            Ok(frame) => Ok(accept_frame(&frame)),
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                Ok(None)
            }
            Err(e) => Err(io_error(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socketcan::ExtendedId;

    #[test]
    fn accepts_standard_id_frames() {
        let id = StandardId::new(0x0FD).unwrap();
        let frame = CanFrame::new(id, &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(accept_frame(&frame), Some((0xFD, [1, 2, 3, 4, 5, 0, 0, 0])));
    }

    #[test]
    fn rejects_extended_id_frames_but_the_power_up_announcement() {
        for id in [0x001, 0x7FFD, 0x7FFF, 0x1000_7FFE] {
            let frame = CanFrame::new(ExtendedId::new(id).unwrap(), &[0x11; 8]).unwrap();
            assert_eq!(accept_frame(&frame), None, "{id:#x}");
        }
        let frame = CanFrame::new(ExtendedId::new(0x7FFE).unwrap(), &[0x11; 8]).unwrap();
        assert_eq!(accept_frame(&frame), Some((crate::POWER_UP_ID, [0x11; 8])));
        let rtr = CanFrame::new_remote(ExtendedId::new(0x7FFE).unwrap(), 8).unwrap();
        assert_eq!(accept_frame(&rtr), None);
    }

    #[test]
    fn rejects_remote_frames() {
        // An RTR frame carries no data; zero-padding it would decode as a
        // feedback frame with every channel at its negative rail.
        let frame = CanFrame::new_remote(StandardId::new(0x0FD).unwrap(), 8).unwrap();
        assert_eq!(accept_frame(&frame), None);
    }

    #[test]
    fn a_gone_or_down_interface_is_closed() {
        for errno in [6, 19, 100] {
            let e = io_error(std::io::Error::from_raw_os_error(errno));
            assert!(matches!(e, TransportError::Closed), "errno {errno}: {e}");
        }
        // A full TX queue (ENOBUFS) passes.
        let e = io_error(std::io::Error::from_raw_os_error(105));
        assert!(matches!(e, TransportError::Io(_)), "{e}");
    }

    #[test]
    fn send_refuses_ids_above_11_bits() {
        assert!(standard_id(0x7FF).is_ok());
        assert!(matches!(standard_id(0x800), Err(TransportError::Io(_))));
    }
}
