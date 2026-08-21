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
///
/// Note the classic-CAN shape: a CAN FD capable adapter makes `candump -l`
/// emit `ID##flags+data` (a doubled `#`), which this parser rejects. Use
/// [`ReplayBackend::from_log_strict`] or [`ReplayBackend::skipped`] to find
/// out when that has happened instead of silently replaying nothing.
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
    // Byte length alone doesn't guarantee char boundaries: a multi-byte UTF-8
    // character can make `data_hex.len()` even and <= 16 while still landing
    // mid-character at a fixed 2-byte slice offset below, which panics. Reject
    // non-ASCII up front so every subsequent byte offset is a valid boundary.
    // An empty payload is valid—candump -l prints "ID#" for a legitimate
    // zero-length (DLC 0) CAN data frame; it parses to dlc: 0.
    if !data_hex.is_ascii() || data_hex.len() % 2 != 0 || data_hex.len() > 16 {
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

/// A log line [`ReplayBackend::from_log`] could not parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedLine {
    /// 1-based line number within the log text.
    pub number: usize,
    pub text: String,
}

/// Returned by [`ReplayBackend::from_log_strict`] when a log contains lines
/// that are not well-formed `candump -l` frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayLogError {
    /// Every line that was skipped, in order.
    pub skipped: Vec<SkippedLine>,
}

impl std::fmt::Display for ReplayLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "candump log: {} unparseable line(s)", self.skipped.len())?;
        if let Some(first) = self.skipped.first() {
            write!(f, "; first at line {}: {:?}", first.number, first.text)?;
        }
        Ok(())
    }
}

impl std::error::Error for ReplayLogError {}

/// Pops recorded frames in order; records everything sent.
#[derive(Debug, Default)]
pub struct ReplayBackend {
    frames: VecDeque<RecordedFrame>,
    skipped: Vec<SkippedLine>,
    pub sent: Vec<(u32, [u8; 8])>,
}

impl ReplayBackend {
    /// Forgiving constructor: unparseable lines are skipped, not rejected.
    ///
    /// Silence is the hazard here — a CAN FD capture (`ID##flags+data`)
    /// loses *every* frame, and the resulting empty backend's `recv` returns
    /// `Ok(None)` forever, which is indistinguishable from a trace that
    /// finished normally. Check [`skipped`](Self::skipped) afterwards, or
    /// use [`from_log_strict`](Self::from_log_strict), before trusting a
    /// replay for golden-trace fidelity. Blank and whitespace-only lines are
    /// not counted as skipped.
    pub fn from_log(text: &str) -> Self {
        let mut frames = VecDeque::new();
        let mut skipped = Vec::new();
        for (i, line) in text.lines().enumerate() {
            match parse_candump_line(line) {
                Some(frame) => frames.push_back(frame),
                None if line.trim().is_empty() => {}
                None => skipped.push(SkippedLine {
                    number: i + 1,
                    text: line.to_string(),
                }),
            }
        }
        Self {
            frames,
            skipped,
            sent: Vec::new(),
        }
    }

    /// Same as [`from_log`](Self::from_log), but a log with any unparseable
    /// line is an error — the right default when a malformed line most
    /// plausibly means a corrupt or wrong-format capture.
    pub fn from_log_strict(text: &str) -> Result<Self, ReplayLogError> {
        let backend = Self::from_log(text);
        if backend.skipped.is_empty() {
            Ok(backend)
        } else {
            Err(ReplayLogError {
                skipped: backend.skipped,
            })
        }
    }

    /// Lines [`from_log`](Self::from_log) discarded, in log order.
    pub fn skipped(&self) -> &[SkippedLine] {
        &self.skipped
    }

    /// Frames still waiting to be handed out by `recv`.
    pub fn remaining(&self) -> usize {
        self.frames.len()
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
        // empty payload is a legitimate zero-length CAN frame (DLC 0)
        let z = parse_candump_line("(1.0) can0 0300FD01#").unwrap();
        assert_eq!(z.dlc, 0);
        assert_eq!(z.data, [0u8; 8]);
        assert_eq!(z.id, 0x0300FD01);
        // garbage is None, not a panic
        assert!(parse_candump_line("not a frame").is_none());
        assert!(parse_candump_line("").is_none());
    }

    #[test]
    fn malformed_lines_return_none_never_panic() {
        // Payload guards: reject non-ASCII and non-even-length payloads.
        // Multi-byte UTF-8 character as the whole payload: byte length is 4
        // (even, <= 16) but no char boundary at offset 2 -- this panics
        // without the is_ascii() guard.
        assert!(parse_candump_line("(1.0) can0 0300FD01#\u{1f600}").is_none());
        // Multi-byte character positioned so total byte length is still even
        // and <= 16, but the fixed 2-byte slice offsets land mid-character.
        assert!(parse_candump_line("(1.0) can0 0300FD01#aa\u{1f600}").is_none());
        // Odd-length hex payload.
        assert!(parse_candump_line("(1.0) can0 0300FD01#010").is_none());
        // Hex payload longer than 16 characters (more than 8 bytes).
        assert!(parse_candump_line("(1.0) can0 0300FD01#0011223344556677889900").is_none());
        // Non-hex ASCII characters in the payload.
        assert!(parse_candump_line("(1.0) can0 0300FD01#ZZ").is_none());

        // Earlier parsing guards: reject malformed timestamps, field structure.
        // No '#' separator at all.
        assert!(parse_candump_line("(1.0) can0 0300FD01").is_none());
        // Timestamp missing its parentheses.
        assert!(parse_candump_line("1.0 can0 0300FD01#0102").is_none());
        // Too few whitespace-separated fields.
        assert!(parse_candump_line("(1.0) can0").is_none());
    }

    // What `candump -l` writes on a CAN FD capable adapter: `ID##flags+data`.
    // The doubled '#' leaves the payload starting with '#', so every line
    // fails to parse and the whole capture silently vanishes.
    const CAN_FD_LOG: &str = "\
(1755600000.000100) can0 0300FD01##10000000000000000
(1755600000.000350) can0 028001FD##18000800080000159
";

    #[test]
    fn from_log_reports_skipped_lines() {
        let text = "\
(1755600000.000100) can0 0300FD01#0000000000000000

(1755600000.000350) can0 028001FD##18000800080000159
this is not a frame line
";
        let r = ReplayBackend::from_log(text);
        // The one good line still replays: from_log stays forgiving.
        assert_eq!(r.remaining(), 1);
        // Line 2 is blank and does not count; lines 3 and 4 do, with their
        // 1-based numbers and original text.
        assert_eq!(
            r.skipped(),
            &[
                SkippedLine {
                    number: 3,
                    text: "(1755600000.000350) can0 028001FD##18000800080000159".to_string(),
                },
                SkippedLine {
                    number: 4,
                    text: "this is not a frame line".to_string(),
                },
            ]
        );
    }

    #[test]
    fn from_log_strict_rejects_a_can_fd_capture() {
        // Without a signal this returns an empty backend whose recv yields
        // Ok(None) forever -- indistinguishable from "trace finished".
        let forgiving = ReplayBackend::from_log(CAN_FD_LOG);
        assert_eq!(forgiving.remaining(), 0);
        assert_eq!(forgiving.skipped().len(), 2);

        let err = ReplayBackend::from_log_strict(CAN_FD_LOG).unwrap_err();
        assert_eq!(err.skipped.len(), 2);
        assert_eq!(err.skipped[0].number, 1);
        let msg = err.to_string();
        assert!(msg.contains("2 unparseable line(s)"), "{msg}");
        assert!(msg.contains("first at line 1"), "{msg}");
    }

    #[test]
    fn from_log_strict_accepts_a_clean_log() {
        let mut r = ReplayBackend::from_log_strict(LOG).expect("clean log");
        assert!(r.skipped().is_empty());
        assert_eq!(r.remaining(), 3);
        // Trailing and interior blank lines are not "dropped frames".
        let padded = format!("\n{LOG}\n\n");
        assert!(ReplayBackend::from_log_strict(&padded).is_ok());
        assert_eq!(
            r.recv(Duration::from_millis(1)).unwrap(),
            Some((0x0300FD01, [0u8; 8]))
        );
        assert_eq!(r.remaining(), 2);
    }

    #[test]
    fn zero_timeout_recv_polls_without_waiting() {
        // Same contract as SocketCanBackend::recv: Duration::ZERO is a poll.
        let mut r = ReplayBackend::from_log(LOG);
        assert_eq!(
            r.recv(Duration::ZERO).unwrap(),
            Some((0x0300FD01, [0u8; 8]))
        );
        let mut empty = ReplayBackend::from_log("");
        assert_eq!(empty.recv(Duration::ZERO).unwrap(), None);
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
