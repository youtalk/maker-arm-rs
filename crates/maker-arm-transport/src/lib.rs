//! CAN transports for the Maker Arm. The trait speaks raw
//! (extended-id, 8-byte payload) frames; encoding/decoding lives in
//! maker-arm-protocol.

mod mock;
pub mod replay;

pub use mock::MockBackend;

use std::time::Duration;

#[derive(Debug)]
pub enum TransportError {
    Io(String),
    Closed,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Io(m) => write!(f, "transport I/O error: {m}"),
            TransportError::Closed => write!(f, "transport closed"),
        }
    }
}

impl std::error::Error for TransportError {}

pub trait CanBackend: Send {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError>;
    /// Returns Ok(None) on timeout with no frame available.
    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError>;
}
