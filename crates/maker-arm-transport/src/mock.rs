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
}
