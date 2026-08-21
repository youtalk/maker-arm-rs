//! Arm session layer: the one place where protocol and transport compose.
//!
//! Safety invariant (design §2): every command that reaches a motor passes
//! through [`clamp::clamp_command`] in the control loop's output stage —
//! controllers, Rust or Python, never bypass it.

pub mod clamp;
pub mod config;
pub mod control_loop;
pub mod controller;
pub mod health;
pub mod session;
pub mod sim;
pub mod state;

pub use clamp::{clamp_command, ClampError};
pub use config::{ArmConfig, JointConfig, MotorModel};
pub use control_loop::{HoldHandle, RunningArm, Snapshot, TickOutcome};
pub use controller::{Controller, HoldController};
pub use health::{FaultReason, HealthMonitor};
pub use session::{Session, SessionError, SessionState};
pub use sim::SimArm;
pub use state::{ArmCommand, ArmState, JointCommand, MotorState};
