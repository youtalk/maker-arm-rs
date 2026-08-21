//! Arm session layer: the one place where protocol and transport compose.
//!
//! Safety invariant (design §2): every command that reaches a motor passes
//! through [`clamp::clamp_command`] in the control loop's output stage —
//! controllers, Rust or Python, never bypass it.

pub mod clamp;
pub mod config;
pub mod state;

pub use clamp::{clamp_command, ClampError};
pub use config::{ArmConfig, JointConfig, MotorModel};
pub use state::{ArmCommand, ArmState, JointCommand, MotorState};
