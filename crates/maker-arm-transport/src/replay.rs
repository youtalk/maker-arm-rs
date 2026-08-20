//! Replay backend for candump -l logs: MA1 captures golden traces from the
//! official SDK's bring-up session; this backend feeds them back through
//! the CanBackend trait so parity tests run with zero hardware.

use crate::{CanBackend, TransportError};
use std::collections::VecDeque;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct RecordedFrame {
    pub timestamp: f64,
    pub interface: String,
    pub id: u32,
    pub data: [u8; 8],
    pub dlc: usize,
}

/// Parses one `candump -l` line: `(ts) iface ID#HEXDATA`. Returns None on
/// anything that isn't a well-formed frame line.
pub fn parse_candump_line(line: &str) -> Option<RecordedFrame> {
    let mut parts = line.split_whitespace();
    let ts = parts
        .next()?
        .strip_prefix('(')?
        .strip_suffix(')')?
        .parse::<f64>()
        .ok()?;
    let interface = parts.next()?.to_string();
    let frame = parts.next()?;
    let (id_hex, data_hex) = frame.split_once('#')?;
    let id = u32::from_str_radix(id_hex, 16).ok()?;
    if data_hex.len() % 2 != 0 || data_hex.len() > 16 {
        return None;
    }
    let mut data = [0u8; 8];
    let dlc = data_hex.len() / 2;
    for i in 0..dlc {
        data[i] = u8::from_str_radix(&data_hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(RecordedFrame {
        timestamp: ts,
        interface,
        id,
        data,
        dlc,
    })
}

/// Pops recorded frames in order; records everything sent.
pub struct ReplayBackend {
    frames: VecDeque<RecordedFrame>,
    pub sent: Vec<(u32, [u8; 8])>,
}

impl ReplayBackend {
    pub fn from_log(text: &str) -> Self {
        Self {
            frames: text.lines().filter_map(parse_candump_line).collect(),
            sent: Vec::new(),
        }
    }
}

impl CanBackend for ReplayBackend {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        self.sent.push((id, *data));
        Ok(())
    }

    fn recv(&mut self, _timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        Ok(self.frames.pop_front().map(|f| (f.id, f.data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CanBackend;
    use std::time::Duration;

    // candump -l format: "(ts) iface ID#HEXDATA" with 8-hex-digit extended ids.
    const LOG: &str = "\
(1755600000.000100) can0 0300FD01#0000000000000000
(1755600000.000350) can0 028001FD#8000800080000159
(1755600000.000500) can0 01800001#80008000051F3333
";

    #[test]
    #[allow(clippy::excessive_precision)]
    fn parses_candump_lines() {
        let f = parse_candump_line("(1755600000.000350) can0 028001FD#8000800080000159").unwrap();
        assert_eq!(f.timestamp, 1755600000.000350);
        assert_eq!(f.interface, "can0");
        assert_eq!(f.id, 0x028001FD);
        assert_eq!(f.data, [0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x01, 0x59]);
        assert_eq!(f.dlc, 8);
        // short payloads are zero-padded but keep their true dlc
        let s = parse_candump_line("(1.0) can0 0400FD02#01").unwrap();
        assert_eq!(s.dlc, 1);
        assert_eq!(s.data[0], 0x01);
        assert_eq!(s.data[1], 0x00);
        // garbage is None, not a panic
        assert!(parse_candump_line("not a frame").is_none());
        assert!(parse_candump_line("").is_none());
    }

    #[test]
    fn replay_backend_pops_frames_in_order() {
        let mut r = ReplayBackend::from_log(LOG);
        assert_eq!(
            r.recv(Duration::from_millis(1)).unwrap(),
            Some((0x0300FD01, [0u8; 8]))
        );
        let (id, _) = r.recv(Duration::from_millis(1)).unwrap().unwrap();
        assert_eq!(id, 0x028001FD);
        r.send(0x0600FD03, &[1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(r.sent.len(), 1);
        let (id, _) = r.recv(Duration::from_millis(1)).unwrap().unwrap();
        assert_eq!(id, 0x01800001);
        assert_eq!(r.recv(Duration::from_millis(1)).unwrap(), None);
    }
}
