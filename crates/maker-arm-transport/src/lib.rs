//! CAN transports for the Maker Arm. The trait speaks raw
//! (extended-id, 8-byte payload) frames; encoding/decoding lives in
//! maker-arm-protocol.

mod mock;
pub mod replay;
#[cfg(feature = "socketcan")]
mod socketcan_backend;

pub use mock::MockBackend;
pub use replay::ReplayBackend;
#[cfg(feature = "socketcan")]
pub use socketcan_backend::SocketCanBackend;

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
    /// Sends one extended-id frame with an 8-byte payload.
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError>;

    /// Receives at most one frame, waiting up to `timeout`.
    ///
    /// Contract, identical for every backend:
    ///
    /// * `Ok(Some((id, data)))` — one extended-id (29-bit) **data** frame.
    ///   Payloads shorter than 8 bytes are zero-padded.
    /// * `Ok(None)` — nothing to report. This covers "the timeout expired",
    ///   "the source is drained", and "a frame arrived that this transport
    ///   does not deliver". It does **not** imply that `timeout` elapsed, so
    ///   callers must not use it to measure time; a caller that needs a
    ///   deadline has to keep its own clock.
    /// * `Err(_)` — the transport itself failed.
    ///
    /// Remote (RTR) frames, 11-bit standard-id frames and error frames are
    /// never returned as motor data: an RTR frame has an empty payload that
    /// would zero-pad into a full-scale-negative feedback reading, and a
    /// standard id is indistinguishable from the same numeric extended id.
    ///
    /// `Duration::ZERO` means "poll": return what is already buffered,
    /// otherwise `Ok(None)` — without blocking. (`MockBackend` and
    /// `ReplayBackend` pop from memory and never wait for any timeout;
    /// `SocketCanBackend` polls its socket with a zero timeout, because
    /// `SO_RCVTIMEO` reads an all-zero `timeval` as *no* timeout and would
    /// otherwise block forever.)
    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError>;
}
