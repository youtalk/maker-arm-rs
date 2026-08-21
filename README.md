# maker-arm-rs

Rust SDK for the MakerMods Maker Arm (RobStride QDD actuators over CAN),
with Python bindings. The protocol layer is a Rust port of the official
[maker-arm-sdk](https://github.com/makermods-robotics/maker-arm-sdk)
(Apache-2.0): it mirrors that project's frame layout, and its golden vectors
are values copied from upstream's test suite, so the two agree byte for byte.
See [NOTICE](NOTICE) for the attribution details.

- `crates/maker-arm-protocol` — pure encode/decode of the RobStride private
  CAN protocol; ported from `maker_arm/protocol.py`.
- `crates/maker-arm-transport` — `CanBackend` trait: mock, candump-log replay,
  and SocketCAN (feature `socketcan`).
- `bindings/maker-arm-py` — PyO3 bindings, Python module `maker_arm_rs`.

Status: MA0 (bench prep). The arm session layer, safety machinery, and CLI
land in MA1 — see the design doc in the maker-arm-lab project.

## Building

```sh
cargo build
cargo test --all-features
```

The SocketCAN tests are `#[ignore]`d because they need a virtual CAN
interface; bring one up first, then run them explicitly:

```sh
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
cargo test -p maker-arm-transport --features socketcan -- --ignored
```

Python bindings (module name: `maker_arm_rs`):

```sh
cd bindings/maker-arm-py
python3 -m venv .venv && . .venv/bin/activate && pip install maturin pytest
maturin develop && python -m pytest tests/ -v
```

## Quick start

```python
import maker_arm_rs as m

# MIT command: motor 1, pos/vel/tau 0, kp 10, kd 1.0 -> (0x1800001, b'\x80\x00\x80\x00\x05\x1f33')
can_id, data = m.encode_mit(1, 0.0, 0.0, 10.0, 1.0, 0.0, "RS00")
fb = m.parse_frame(0x028001FD, bytes.fromhex("8000800080000159"), "RS00")
print(fb["kind"], fb["motor_id"], fb["temperature"])  # feedback 1 34.5
```

The same functions in Rust are `maker_arm_protocol::{encode_mit, parse_frame}`;
put the frames on a bus with `maker_arm_transport::SocketCanBackend::open("can0")`
and its `CanBackend::{send, recv}`.
