use crate::{CanBackend, TransportError};
use std::collections::VecDeque;
use std::time::Duration;

/// Records everything sent; hands back scripted incoming frames.
#[derive(Default)]
pub struct MockBackend {
    pub sent: Vec<(u32, [u8; 8])>,
    incoming: VecDeque<(u32, [u8; 8])>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_incoming(&mut self, id: u32, data: [u8; 8]) {
        self.incoming.push_back((id, data));
    }
}

impl CanBackend for MockBackend {
    fn send(&mut self, id: u32, data: &[u8; 8]) -> Result<(), TransportError> {
        self.sent.push((id, *data));
        Ok(())
    }

    fn recv(&mut self, _timeout: Duration) -> Result<Option<(u32, [u8; 8])>, TransportError> {
        Ok(self.incoming.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_records_sends_and_replays_incoming() {
        let mut m = MockBackend::new();
        m.send(0x0300FD01, &[0u8; 8]).unwrap();
        assert_eq!(m.sent, vec![(0x0300FD01, [0u8; 8])]);
        m.push_incoming(0x028001FD, [1u8; 8]);
        assert_eq!(
            m.recv(Duration::from_millis(1)).unwrap(),
            Some((0x028001FD, [1u8; 8]))
        );
        assert_eq!(m.recv(Duration::from_millis(1)).unwrap(), None);
    }

    #[test]
    fn mock_enforces_fifo_ordering_with_multiple_frames() {
        let mut m = MockBackend::new();

        // Push three distinct frames with recognizable ids and payloads
        m.push_incoming(0x0300FD01, [1u8, 2, 3, 4, 5, 6, 7, 8]);
        m.push_incoming(0x028001FD, [10u8, 20, 30, 40, 50, 60, 70, 80]);
        m.push_incoming(0x01800001, [100u8, 110, 120, 130, 140, 150, 160, 170]);

        // Verify they come back in FIFO order (not LIFO)
        assert_eq!(
            m.recv(Duration::from_millis(1)).unwrap(),
            Some((0x0300FD01, [1u8, 2, 3, 4, 5, 6, 7, 8]))
        );
        assert_eq!(
            m.recv(Duration::from_millis(1)).unwrap(),
            Some((0x028001FD, [10u8, 20, 30, 40, 50, 60, 70, 80]))
        );
        assert_eq!(
            m.recv(Duration::from_millis(1)).unwrap(),
            Some((0x01800001, [100u8, 110, 120, 130, 140, 150, 160, 170]))
        );

        // Verify the drained-queue contract
        assert_eq!(m.recv(Duration::from_millis(1)).unwrap(), None);
    }

    #[test]
    fn mock_copies_send_data_payload_faithfully() {
        let mut m = MockBackend::new();

        // Send frames with distinct, non-zero payloads to verify data is actually copied
        let payload1 = [11u8, 22, 33, 44, 55, 66, 77, 88];
        let payload2 = [99u8, 88, 77, 66, 55, 44, 33, 22];
        let id1 = 0x0300FD01u32;
        let id2 = 0x028001FDu32;

        m.send(id1, &payload1).unwrap();
        m.send(id2, &payload2).unwrap();

        // Verify both frames are recorded with their exact payloads
        assert_eq!(m.sent, vec![(id1, payload1), (id2, payload2)]);

        // Double-check that the payloads are distinct and non-zero
        assert_ne!(payload1, payload2);
        assert_ne!(payload1, [0u8; 8]);
        assert_ne!(payload2, [0u8; 8]);
    }
}
