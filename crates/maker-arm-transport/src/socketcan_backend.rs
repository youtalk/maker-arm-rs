use crate::{CanBackend, TransportError};
use socketcan::{CanFrame, CanSocket, EmbeddedFrame, ExtendedId, Socket};
use std::time::Duration;

/// Blocking SocketCAN backend. Extended (29-bit) ids only — the RobStride
/// private protocol never uses standard ids.
pub struct SocketCanBackend {
    sock: CanSocket,
}

impl SocketCanBackend {
    pub fn open(ifname: &str) -> Result<Self, TransportError> {
        let sock = CanSocket::open(ifname).map_err(|e| TransportError::Io(e.to_string()))?;
        Ok(Self { sock })
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
        self.sock
            .set_read_timeout(timeout)
            .map_err(|e| TransportError::Io(e.to_string()))?;
        match self.sock.read_frame() {
            Ok(frame) => {
                let id = match frame.id() {
                    socketcan::Id::Extended(e) => e.as_raw(),
                    socketcan::Id::Standard(s) => s.as_raw() as u32,
                };
                let mut data = [0u8; 8];
                let d = frame.data();
                data[..d.len().min(8)].copy_from_slice(&d[..d.len().min(8)]);
                Ok(Some((id, data)))
            }
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
