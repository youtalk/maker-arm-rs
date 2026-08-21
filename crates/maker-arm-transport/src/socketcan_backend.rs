use crate::{CanBackend, TransportError};
use socketcan::{CanFilter, CanFrame, CanSocket, EmbeddedFrame, ExtendedId, Socket, SocketOptions};
use std::time::Duration;

/// `can_id` bit the kernel sets on extended (29-bit) frames.
const CAN_EFF_FLAG: u32 = 0x8000_0000;
/// `can_id` bit the kernel sets on remote-transmission-request frames.
const CAN_RTR_FLAG: u32 = 0x4000_0000;

/// Blocking SocketCAN backend. Extended (29-bit) ids only — the RobStride
/// private protocol never uses standard ids, and `recv` enforces that in
/// both directions (see [`accept_frame`]).
pub struct SocketCanBackend {
    sock: CanSocket,
    /// Last value handed to `set_read_timeout`, so `recv` only pays for the
    /// `setsockopt` when the requested timeout actually changes: at ~1 kHz
    /// the timeout is the same value on every call.
    read_timeout: Option<Duration>,
}

/// Reduces one received frame to what the private protocol can carry, or
/// `None` if the frame must not be presented as motor data.
///
/// `CAN_RAW` sockets deliver remote (RTR) frames, 11-bit standard-id frames
/// and error frames alongside the traffic we want:
///
/// * An RTR frame has an empty payload. Zero-padding it to `[0u8; 8]` and
///   handing it to `parse_frame` yields a structurally valid feedback
///   reading with every channel pinned at its negative rail (position
///   −12.57 rad, velocity −33 rad/s, torque −14 Nm) — exactly the input a
///   PD loop turns into a large correction.
/// * A standard id `0x001` and an extended id `0x00000001` would both
///   surface as `(1, data)`, i.e. indistinguishable.
fn accept_frame(frame: &CanFrame) -> Option<(u32, [u8; 8])> {
    let data_frame = match frame {
        CanFrame::Data(f) => f,
        CanFrame::Remote(_) | CanFrame::Error(_) => return None,
    };
    let socketcan::Id::Extended(eid) = data_frame.id() else {
        return None;
    };
    let mut data = [0u8; 8];
    let d = data_frame.data();
    data[..d.len().min(8)].copy_from_slice(&d[..d.len().min(8)]);
    Some((eid.as_raw(), data))
}

impl SocketCanBackend {
    pub fn open(ifname: &str) -> Result<Self, TransportError> {
        let sock = CanSocket::open(ifname)
            .map_err(|e| TransportError::Io(format!("open CAN interface {ifname}: {e}")))?;
        // Kernel-side half of the frame filtering: match only frames whose
        // EFF bit is set and whose RTR bit is clear, so standard-id and
        // remote frames are dropped before they reach userspace. The
        // userspace check in `accept_frame` is the authoritative one — this
        // just keeps foreign traffic from eating the caller's timeout.
        sock.set_filters(&[CanFilter::new(CAN_EFF_FLAG, CAN_EFF_FLAG | CAN_RTR_FLAG)])
            .map_err(|e| TransportError::Io(format!("set CAN filter on {ifname}: {e}")))?;
        Ok(Self {
            sock,
            read_timeout: None,
        })
    }
}

impl CanBackend for SocketCanBackend {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        let eid = ExtendedId::new(id)
            .ok_or_else(|| TransportError::Io(format!("id {id:#x} exceeds 29 bits")))?;
        let frame = CanFrame::new(eid, data)
            .ok_or_else(|| TransportError::Io("frame construction failed".into()))?;
        self.sock
            .write_frame(&frame)
            .map_err(|e| TransportError::Io(e.to_string()))
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
                self.sock
                    .set_read_timeout(timeout)
                    .map_err(|e| TransportError::Io(e.to_string()))?;
                self.read_timeout = Some(timeout);
            }
            self.sock.read_frame()
        };
        match read {
            // A frame this transport does not deliver (remote, standard-id,
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
            Err(e) => Err(TransportError::Io(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socketcan::StandardId;

    #[test]
    fn accepts_extended_data_frames() {
        let id = ExtendedId::new(0x0280_01FD).unwrap();
        let data = [0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x01, 0x59];
        let frame = CanFrame::new(id, &data).unwrap();
        assert_eq!(accept_frame(&frame), Some((0x0280_01FD, data)));
    }

    #[test]
    fn zero_pads_short_extended_data_frames() {
        let id = ExtendedId::new(0x0400_FD02).unwrap();
        let frame = CanFrame::new(id, &[0x01]).unwrap();
        assert_eq!(
            accept_frame(&frame),
            Some((0x0400_FD02, [0x01, 0, 0, 0, 0, 0, 0, 0]))
        );
    }

    #[test]
    fn rejects_remote_frames() {
        // An RTR frame carries no data; zero-padding it would decode as a
        // feedback frame with every channel at its negative rail.
        let id = ExtendedId::new(0x0280_01FD).unwrap();
        let frame = CanFrame::new_remote(id, 8).unwrap();
        assert_eq!(accept_frame(&frame), None);
    }

    #[test]
    fn rejects_standard_id_frames() {
        // Standard id 0x001 must not be confused with extended id 0x00000001.
        let frame = CanFrame::new(StandardId::new(0x001).unwrap(), &[0x11; 8]).unwrap();
        assert_eq!(accept_frame(&frame), None);
    }

    #[test]
    fn rejects_standard_id_remote_frames() {
        let frame = CanFrame::new_remote(StandardId::new(0x001).unwrap(), 8).unwrap();
        assert_eq!(accept_frame(&frame), None);
    }
}
