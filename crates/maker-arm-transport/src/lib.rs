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

use std::io::ErrorKind;
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixListener};
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

    /// Test support: downcast access to the concrete backend.
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

/// Holds the abstract Unix socket `maker-arm-bus-<ifname>` while it lives. The kernel frees
/// the name when the process dies, so a crash leaves no stale lock.
pub struct BusLock {
    _socket: UnixListener,
}

/// Takes the bus lock of `ifname`. Every opener of a real bus takes it before its first
/// frame and holds it to the end: a probe or a connect sends disable frames, and a
/// `RunningArm` drop disables the motors, so a second opener would drop an arm that another
/// process holds. Kept apart from `SocketCanBackend::open`, because one process may open
/// several sockets on one interface (the vcan tests do).
pub fn bus_lock(ifname: &str) -> Result<BusLock, TransportError> {
    let name = format!("maker-arm-bus-{ifname}");
    SocketAddr::from_abstract_name(name.as_bytes())
        .and_then(|a| UnixListener::bind_addr(&a))
        .map(|s| BusLock { _socket: s })
        .map_err(|e| {
            TransportError::Io(match e.kind() {
                ErrorKind::AddrInUse => format!(
                    "{ifname} is in use: another process holds the bus lock @{name}. A daemon \
                     may hold the arm. Run `maker-arm-cli status --interface {ifname}`. Find \
                     the holder with `ss -xlp | grep {name}`"
                ),
                _ => format!("bus lock @{name} for {ifname}: {e}"),
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bus_lock_is_exclusive_and_freed_on_drop() {
        let iface = format!("vcan-unit{}", std::process::id());
        let held = bus_lock(&iface).unwrap();
        let e = bus_lock(&iface).err().unwrap().to_string();
        assert!(e.contains(&format!("maker-arm-bus-{iface}")), "{e}");
        assert!(e.contains("maker-arm-cli status --interface"), "{e}");
        assert!(e.contains("ss -xlp"), "{e}");
        drop(held);
        assert!(bus_lock(&iface).is_ok());
    }
}
