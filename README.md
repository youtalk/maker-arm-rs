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
- `bindings/maker-arm-py` — PyO3 bindings, Python module `maker_arm_rs`:
  protocol-level `encode_mit`/`parse_frame`, plus the orchestration-mode
  `Arm` class (connect, enable, hold, snapshot, stop) described below.
- `crates/maker-arm` — arm session layer: the pinned `maker_arm_v1` profile,
  single-point command clamp, `Controller` trait, health monitoring with
  hold-on-fault, 200 Hz control loop, and `SimArm`, an in-memory 7-motor
  simulator for hardware-free development.
- `crates/maker-arm-cli` — `scan`, `doctor`, `zero`, and `hold` (typed-RELEASE
  safety gate), against `--can <iface>` or `--sim`.

Status: MA1 prep (hardware-independent session layer) complete. **No physical
arm has ever been driven by this code** — everything here is exercised against
`SimArm`, a virtual CAN interface, and golden vectors copied from upstream.
Golden-trace capture, live parity, the RS02 firmware check, and everything else
that needs the physical arm land in MA1 — see the design doc in the
maker-arm-lab project.

## Building

```sh
cargo build
cargo test --workspace --exclude maker-arm-py --all-features
```

The bindings crate (`maker-arm-py`) is excluded from that command because its
`extension-module` feature links against no Python interpreter, which breaks
`cargo test` at link time; it is built and tested separately, via `maturin`
(see below).

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

That raw pairing is for frame-level work and tests only — it bypasses the
single command clamp. The supported way to command a motor is through
`maker-arm`'s session and control loop (the `Arm` class below, or
`Session::start`), which passes every command through `clamp_command` before
it reaches a motor.

Orchestration mode: Python selects and steers controllers that run inside the
Rust control loop; it never commands torque itself, so every command still
passes through the Rust clamp.

```python
import time
import maker_arm_rs as m

arm = m.Arm.sim()  # or m.Arm.socketcan("can0") for a real bus
arm.enable()
arm.start_hold()  # spawns the 200 Hz loop, holding the current pose
time.sleep(0.1)
snap = arm.snapshot()
print(snap["state"], snap["positions"])
arm.stop()  # disable and join; also aliased as arm.estop()
```

## Profile and clamp from Python

`maker_arm_rs.profile()` returns the pinned `maker_arm_v1` profile as a dict
(7 joints: limits, gains, torque caps, direction/offset). `maker_arm_rs.clamp_command(cmds, profile)`
applies the crate's single-point clamp to a list of 7 `(pos, vel, kp, kd, tau)` tuples and
returns `(clamped_cmds, changed)`. The dict's numeric fields may be edited before the call
(a simulator passes URDF joint limits this way); names, models, and motor ids are never
read from Python, and `encode_mit` stays unreachable, so no Python value becomes a CAN
frame without passing the Rust clamp inside the control loop.

Joint-coordinate contract: joint coordinates are the vendor URDF's
(`maker-arm-sdk/urdf/maker_arm/robot.urdf` @ b30d05a), `link_002_joint`..`link_007_joint`
= motor ids 1..6. `direction`/`offset` map motor to URDF coordinates and are calibrated
on the arm (MA1); the profile's `q_lo`/`q_hi` are motor-frame until then.
