# maker-arm-rs

Rust SDK for the MakerMods Maker Arm (RobStride QDD actuators over CAN),
with Python bindings. A from-scratch implementation referencing the official
[maker-arm-sdk](https://github.com/makermods-robotics/maker-arm-sdk) (Apache-2.0)
as the protocol oracle.

- `crates/maker-arm-protocol` — pure encode/decode of the RobStride private
  CAN protocol; test vectors ported from the official SDK.
- `crates/maker-arm-transport` — `CanBackend` trait: mock, candump-log replay,
  and SocketCAN (feature `socketcan`).
- `bindings/maker-arm-py` — PyO3 bindings, Python module `maker_arm_rs`.

Status: MA0 (bench prep). The arm session layer, safety machinery, and CLI
land in MA1 — see the design doc in the maker-arm-lab project.
