//! CAN transports for the Maker Arm. The trait speaks raw
//! (11-bit standard id, 8-byte payload) frames, plus the motors' power-up
//! announcement as `POWER_UP_ID`; encoding/decoding lives in
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
use std::time::{Duration, Instant};

/// The id under which every backend delivers a motor's power-up
/// announcement. A RobStride motor sends it on extended id 0x7FFE right
/// after power-up, with its MCU id as the payload. The extended-frame flag
/// keeps it apart from every 11-bit standard id.
pub const POWER_UP_ID: u32 = 0x8000_7FFE;

/// One read inside `send_and_wait`. A wait can end up to this much after
/// its timeout.
const RECV_SLICE: Duration = Duration::from_micros(500);

#[derive(Debug)]
pub enum TransportError {
    Io(String),
    /// The interface is gone (an unplugged adapter) or down. No frame can
    /// pass until someone acts.
    Closed,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Io(m) => write!(f, "transport I/O error: {m}"),
            TransportError::Closed => write!(f, "the CAN interface is gone or down"),
        }
    }
}

impl std::error::Error for TransportError {}

pub trait CanBackend: Send {
    /// Sends one standard-id data frame with an 8-byte payload. `id` must be
    /// at most `0x7FF`; `SocketCanBackend` refuses a larger one.
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError>;

    /// Receives at most one frame, waiting up to `timeout`.
    ///
    /// Contract, identical for every backend:
    ///
    /// * `Ok(Some((id, data)))` — one standard-id (11-bit) **data** frame.
    ///   Payloads shorter than 8 bytes are zero-padded.
    /// * `Ok(None)` — nothing to report. This covers "the timeout expired",
    ///   "the source is drained", and "a frame arrived that this transport
    ///   does not deliver". It does **not** imply that `timeout` elapsed, so
    ///   callers must not use it to measure time; a caller that needs a
    ///   deadline has to keep its own clock.
    /// * `Err(_)` — the transport itself failed.
    ///
    /// Remote (RTR) frames, extended-id frames and error frames are
    /// never returned as motor data: an RTR frame has an empty payload that
    /// would zero-pad into a full-scale-negative feedback reading, and an
    /// extended id is indistinguishable from the same numeric standard id.
    /// The one exception is a data frame on extended id 0x7FFE, the power-up
    /// announcement: it comes back as `(POWER_UP_ID, data)`.
    ///
    /// `Duration::ZERO` means "poll": return what is already buffered,
    /// otherwise `Ok(None)` — without blocking. (`MockBackend` and
    /// `ReplayBackend` pop from memory and never wait for any timeout;
    /// `SocketCanBackend` polls its socket with a zero timeout, because
    /// `SO_RCVTIMEO` reads an all-zero `timeval` as *no* timeout and would
    /// otherwise block forever.)
    fn recv(&mut self, timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError>;

    /// Sends one frame, then gives each frame that arrives to `reply` until
    /// `reply` returns true or `timeout` has passed. Returns true when
    /// `reply` did. A zero `timeout` sends and does not read.
    ///
    /// Every RobStride motor replies on the same CAN id, so two replies on
    /// the bus at once collide. Each sender that addresses motors one after
    /// another uses this, and sends the next frame only after the reply or
    /// the timeout.
    fn send_and_wait(
        &mut self,
        id: u32,
        data: &[u8; 8],
        timeout: Duration,
        reply: &mut dyn FnMut(u32, &[u8; 8]) -> bool,
    ) -> Result<bool, TransportError> {
        self.send(id, data)?;
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            // A fixed slice, never the time left: SocketCAN reads a timeout
            // under 1 us as no timeout at all.
            if let Some((id, data)) = self.recv(RECV_SLICE)? {
                if reply(id, &data) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

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
