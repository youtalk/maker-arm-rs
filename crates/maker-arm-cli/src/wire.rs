//! The daemon's Unix-socket wire. Little-endian, fixed-size frames, each
//! starting with an 8-byte header: magic u32, version u16, kind u16.

use std::io::Read;

const MAGIC: u32 = 0x4D41524D;
pub const VERSION: u16 = 1;
const HELLO: u16 = 1;
const CONTROL: u16 = 2;
const COMMAND: u16 = 3;
const STATE: u16 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Command = 1,
    Monitor = 2,
    Release = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Activate = 1,
    Deactivate = 2,
    Poll = 3,
    Release = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Holding = 0,
    Following = 1,
    Refused = 2,
    Fault = 3,
    Released = 4,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClientFrame {
    Hello(Role),
    Control {
        seq: u32,
        op: Op,
    },
    /// Six joints in rad, then `gripper_left_joint` in m.
    Command {
        seq: u32,
        q: [f64; 7],
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateFrame {
    /// The last client seq.
    pub seq: u32,
    pub mode: Mode,
    pub digest: u64,
    /// Loop clock.
    pub t: f64,
    pub q: [f64; 7],
    pub dq: [f64; 7],
    pub effort: [f64; 7],
    pub temp: [f64; 7],
    pub fault: u32,
    pub clamped: u32,
    pub refused: u32,
}

#[derive(Debug)]
pub enum WireError {
    BadMagic(u32),
    Version { got: u16, want: u16 },
    Kind(u16),
    Role(u8),
    Op(u8),
    Mode(u8),
    Io(std::io::Error),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic(m) => write!(f, "bad magic {m:#010x}"),
            Self::Version { got, want } => {
                write!(f, "wire version {got}, this build speaks {want}")
            }
            Self::Kind(k) => write!(f, "unknown frame kind {k}"),
            Self::Role(r) => write!(f, "unknown role {r}"),
            Self::Op(o) => write!(f, "unknown op {o}"),
            Self::Mode(m) => write!(f, "unknown mode {m}"),
            Self::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

impl std::error::Error for WireError {}

impl From<std::io::Error> for WireError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn header(kind: u16, len: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(len);
    b.extend_from_slice(&MAGIC.to_le_bytes());
    b.extend_from_slice(&VERSION.to_le_bytes());
    b.extend_from_slice(&kind.to_le_bytes());
    b
}

fn push_arr(b: &mut Vec<u8>, a: &[f64; 7]) {
    for v in a {
        b.extend_from_slice(&v.to_le_bytes());
    }
}

pub fn encode_client(f: &ClientFrame) -> Vec<u8> {
    match f {
        ClientFrame::Hello(role) => {
            let mut b = header(HELLO, 16);
            b.push(*role as u8);
            b.resize(16, 0);
            b
        }
        ClientFrame::Control { seq, op } => {
            let mut b = header(CONTROL, 16);
            b.extend_from_slice(&seq.to_le_bytes());
            b.push(*op as u8);
            b.resize(16, 0);
            b
        }
        ClientFrame::Command { seq, q } => {
            let mut b = header(COMMAND, 72);
            b.extend_from_slice(&seq.to_le_bytes());
            b.resize(16, 0);
            push_arr(&mut b, q);
            b
        }
    }
}

pub fn encode_state(s: &StateFrame) -> Vec<u8> {
    let mut b = header(STATE, 272);
    b.extend_from_slice(&s.seq.to_le_bytes());
    b.push(s.mode as u8);
    b.resize(16, 0);
    b.extend_from_slice(&s.digest.to_le_bytes());
    b.extend_from_slice(&s.t.to_le_bytes());
    for a in [&s.q, &s.dq, &s.effort, &s.temp] {
        push_arr(&mut b, a);
    }
    for v in [s.fault, s.clamped, s.refused] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.resize(272, 0);
    b
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

fn f64_at(b: &[u8], i: usize) -> f64 {
    f64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

fn arr_at(b: &[u8], i: usize) -> [f64; 7] {
    std::array::from_fn(|k| f64_at(b, i + 8 * k))
}

/// Read and check the header, return the kind.
fn read_header(r: &mut impl Read) -> Result<u16, WireError> {
    let mut h = [0u8; 8];
    r.read_exact(&mut h)?;
    let magic = u32_at(&h, 0);
    if magic != MAGIC {
        return Err(WireError::BadMagic(magic));
    }
    let got = u16::from_le_bytes([h[4], h[5]]);
    if got != VERSION {
        return Err(WireError::Version { got, want: VERSION });
    }
    Ok(u16::from_le_bytes([h[6], h[7]]))
}

/// Read the `len` bytes that follow the header.
fn read_rest(r: &mut impl Read, len: usize) -> Result<Vec<u8>, WireError> {
    let mut b = vec![0u8; len];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// Offsets below are relative to the end of the 8-byte header.
pub fn read_client(r: &mut impl Read) -> Result<ClientFrame, WireError> {
    match read_header(r)? {
        HELLO => {
            let b = read_rest(r, 8)?;
            let role = match b[0] {
                1 => Role::Command,
                2 => Role::Monitor,
                3 => Role::Release,
                x => return Err(WireError::Role(x)),
            };
            Ok(ClientFrame::Hello(role))
        }
        CONTROL => {
            let b = read_rest(r, 8)?;
            let op = match b[4] {
                1 => Op::Activate,
                2 => Op::Deactivate,
                3 => Op::Poll,
                4 => Op::Release,
                x => return Err(WireError::Op(x)),
            };
            Ok(ClientFrame::Control {
                seq: u32_at(&b, 0),
                op,
            })
        }
        COMMAND => {
            let b = read_rest(r, 64)?;
            Ok(ClientFrame::Command {
                seq: u32_at(&b, 0),
                q: arr_at(&b, 8),
            })
        }
        k => Err(WireError::Kind(k)),
    }
}

pub fn read_state(r: &mut impl Read) -> Result<StateFrame, WireError> {
    match read_header(r)? {
        STATE => {}
        k => return Err(WireError::Kind(k)),
    }
    let b = read_rest(r, 264)?;
    let mode = match b[4] {
        0 => Mode::Holding,
        1 => Mode::Following,
        2 => Mode::Refused,
        3 => Mode::Fault,
        4 => Mode::Released,
        x => return Err(WireError::Mode(x)),
    };
    Ok(StateFrame {
        seq: u32_at(&b, 0),
        mode,
        digest: u64::from_le_bytes(b[8..16].try_into().unwrap()),
        t: f64_at(&b, 16),
        q: arr_at(&b, 24),
        dq: arr_at(&b, 80),
        effort: arr_at(&b, 136),
        temp: arr_at(&b, 192),
        fault: u32_at(&b, 248),
        clamped: u32_at(&b, 252),
        refused: u32_at(&b, 256),
    })
}

/// FNV-1a 64 over each part as `name`, a 0 byte, the value, a 0 byte.
pub fn digest(parts: &[(&str, &[u8])]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (name, value) in parts {
        for &byte in name.as_bytes().iter().chain(&[0]).chain(*value).chain(&[0]) {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO_HEX: &str = "4d52414d010001000100000000000000";
    const CONTROL_HEX: &str = "4d52414d010002000700000001000000";
    const COMMAND_HEX: &str = "4d52414d010003002a000000000000009a9999999999b93f9a9999999999c9bf333333333333d33f9a9999999999d9bf000000000000e03f333333333333e3bf9a9999999999993f";
    const STATE_HEX: &str = "4d52414d010004002a00000001000000efcdab896745230100000000000029409a9999999999b93f9a9999999999c9bf333333333333d33f9a9999999999d9bf000000000000e03f333333333333e3bf9a9999999999993f0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007b14ae47e17a84bf000000000000f83f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d03f000000000080414000000000008041400000000000804140000000000080414000000000008041400000000000804140000000000080414000000000030000000100000000000000";
    const Q: [f64; 7] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.025];

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn client_frames_match_the_golden_vectors() {
        assert_eq!(
            hex(&encode_client(&ClientFrame::Hello(Role::Command))),
            HELLO_HEX
        );
        assert_eq!(
            hex(&encode_client(&ClientFrame::Control {
                seq: 7,
                op: Op::Activate
            })),
            CONTROL_HEX
        );
        assert_eq!(
            hex(&encode_client(&ClientFrame::Command { seq: 42, q: Q })),
            COMMAND_HEX
        );
        for f in [
            ClientFrame::Hello(Role::Release),
            ClientFrame::Control {
                seq: 9,
                op: Op::Release,
            },
            ClientFrame::Command { seq: 1, q: Q },
        ] {
            assert_eq!(read_client(&mut encode_client(&f).as_slice()).unwrap(), f);
        }
    }

    #[test]
    fn the_state_frame_matches_its_golden_vector_and_round_trips() {
        let s = StateFrame {
            seq: 42,
            mode: Mode::Following,
            digest: 0x0123456789ABCDEF,
            t: 12.5,
            q: Q,
            dq: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01],
            effort: [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            temp: [35.0; 7],
            fault: 0,
            clamped: 3,
            refused: 1,
        };
        let bytes = encode_state(&s);
        assert_eq!(bytes.len(), 272);
        assert_eq!(hex(&bytes), STATE_HEX);
        assert_eq!(read_state(&mut unhex(STATE_HEX).as_slice()).unwrap(), s);
    }

    #[test]
    fn the_digest_matches_its_golden_value() {
        let parts: [(&str, &[u8]); 6] = [
            ("interface", b"sim"),
            ("profile", br#"{"max_velocity": 2.5}"#),
            ("dynamics", b"{}"),
            ("params", b""),
            ("residual", b""),
            ("gripper_map", b""),
        ];
        assert_eq!(digest(&parts), 0xd89741bc0a7a1ac8);
    }

    #[test]
    fn another_version_is_an_error_that_names_both() {
        let mut b = unhex(HELLO_HEX);
        b[4] = 2;
        let e = read_client(&mut b.as_slice()).unwrap_err().to_string();
        assert!(e.contains('2') && e.contains('1'), "{e}");
    }

    #[test]
    fn a_short_read_is_an_io_error_not_a_panic() {
        let b = &unhex(COMMAND_HEX)[..10];
        assert!(matches!(read_client(&mut &b[..]), Err(WireError::Io(_))));
    }
}
