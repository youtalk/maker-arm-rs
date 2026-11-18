# maker-arm-rs

Rust SDK for the MakerMods Maker Arm (RobStride QDD actuators over CAN),
with Python bindings. The motors use the RobStride MIT protocol: standard
11-bit CAN ids at 1 Mbit/s. This code follows the vendor SDK
[maker-arm-sdk](https://github.com/makermods-robotics/maker-arm-sdk)
(Apache-2.0) at commit `0245316`. The protocol layer is a Rust port of its
`maker_arm/protocol.py`. Its golden vectors come from that commit, so the two
agree byte for byte. See [NOTICE](NOTICE) for the attribution details.

- `crates/maker-arm-protocol`: pure encode and decode of the RobStride MIT
  protocol, ported from `maker_arm/protocol.py`.
- `crates/maker-arm-transport`: the `CanBackend` trait, with a mock,
  candump-log replay and SocketCAN (feature `socketcan`).
- `bindings/maker-arm-py`: PyO3 bindings, Python module `maker_arm_rs`. It has
  the protocol-level `encode_control` and `parse_frame`, and the
  orchestration-mode `Arm` class (connect, enable, hold, snapshot, stop)
  described below.
- `crates/maker-arm`: the arm session layer, with the pinned `maker_arm_v1`
  profile, single-point command clamp, `Controller` trait, health monitoring with
  hold-on-fault, 200 Hz control loop, and `SimArm`, an in-memory 7-motor
  simulator for hardware-free development.
  Every motor replies on the same CAN id, so the session has one frame on the
  bus at a time: each frame waits for its motor's reply, or for
  `reply_wait_us` (3 ms), before the next frame goes out.
  Its `kinematics` module is the six-joint grasp-pose solver the lab's planner
  uses (forward kinematics over a caller-supplied URDF chain, central-difference
  Jacobian, Levenberg-Marquardt refine, multi-seed branch selection, and
  continuity-constrained paths), exposed to Python as `maker_arm_rs.Kinematics`.
- `crates/maker-arm-cli`: `scan`, `doctor`, `zero`, and `hold` (typed-RELEASE
  safety gate, then a 2 s torque ramp before the disable), against
  `--can <iface>` or `--sim`. No command clears a motor fault. To clear one,
  fix the cause, then press and release the emergency stop, or run the vendor
  `maker-arm scan --clear-faults`.

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

## ROS 2 daemon

`maker-arm-cli serve` is a resident daemon for ROS 2. It owns the CAN bus and the tracking loop. One command client sends targets over a Unix socket. The daemon holds the arm when that client stops.

Install it on Ubuntu 24.04. The apt Rust 1.85 is enough.

```sh
sudo apt install rustc-1.85 cargo-1.85 git
git clone https://github.com/youtalk/maker-arm-rs
PATH=/usr/lib/rust-1.85/bin:$PATH cargo install --locked --path maker-arm-rs/crates/maker-arm-cli
```

The apt cargo cannot install from a git URL, so clone the repository first.

The daemon sets the error clamp to `0.03 / kp_scale` rad per arm joint, where `kp_scale` is the profile's `kp` over the `maker_arm_v1` value (rule R33), and it refuses a params `e_max` above that cap at load time.

The daemon is not yet verified on a real arm. Support the arm before you start `serve` on a real interface.

Four commands. For `PROFILE` and `MODEL`, use the profile and model files that ship with your arm's ROS package. Do not write them by hand.

```sh
maker-arm-cli serve --interface can0 --profile PROFILE --dynamics MODEL   # start the daemon
maker-arm-cli status --interface can0                                    # print one STATUS line
maker-arm-cli release --interface can0                                   # ramp the torque down, then disable the motors
maker-arm-cli enable --interface can0                                    # enable them again
```

The socket is `--socket PATH`. If you omit it, the path is `$XDG_RUNTIME_DIR/maker-arm-<interface>.sock`. If that variable is not set, pass `--socket`. `status`, `release` and `enable` use the same rule. They never open the bus.

`status` prints `STATUS {"mode": ..., "t": ..., "q": [7 values], "clamped": ..., "refused": ..., "fault": ..., "temp_max_c": ..., "temp_max_motor": ..., "digest": ...}` and exits 0. `temp_max_c` is the temperature of the hottest motor, and `temp_max_motor` is its motor id. It exits 1 if no daemon runs on the socket.

### Export the daemon's files

`maker-arm-cli export` writes the profile and the dynamics model for your own arm. It takes the arm's URDF as an input. Every file passes the daemon's own loader before it is written. If a check fails, `export` prints one error line, exits non-zero and writes nothing. On success it prints each path it wrote.

```sh
# Simulation: a profile with the URDF's joint limits, and a model from the URDF's inertials
maker-arm-cli export --sim --urdf ARM.urdf --out_dir DIR
# Real arm: the v1 profile, gains and torque caps times F (0 < F <= 1), limits in your joint frame
maker-arm-cli export --calibration CAL.json --caps F --out PROFILE
# A model from an identification file (bodies, friction, armature) and the URDF's links
maker-arm-cli export --ident IDENT.json --urdf ARM.urdf --out MODEL
```

`CAL.json` holds exactly two keys: `direction` (7 values, each -1.0 or 1.0) and `offset` (7 values, each at most pi in absolute value, in radians). Any other key, or a missing key, is refused.

The procedure to measure `direction` and `offset` is not published yet. A guessed calibration can drive a joint into its stop. Do not enable a real arm on a profile whose calibration you did not measure. A profile with `--caps` below 1 cuts `kp`, so the daemon applies the R33 error clamp to it (`e_max = 0.03 / kp_scale`).

Hold rules:

- On a real interface, after `SERVE_READY`, the daemon does not disable the motors by itself. It holds the arm when the client leaves, on SIGINT and SIGTERM, and on a fault. A signal never releases the arm. The fault hold keeps the gravity torque of the last command, so the arm does not sag.
- Before `SERVE_READY`, a start-up failure can disable the motors that the daemon just enabled. The arm can then drop.
- A short CAN error does not stop the control loop. The arm holds as on a fault. The loop stops only when the errors last longer than 0.25 s or the CAN interface is gone. `release` then still tries to disable the motors.
- If the control loop stops, or the daemon process is killed, nothing commands the motors. Each motor keeps its last command. Only a gripper whose firmware has parameter access has a watchdog. To stop the motors, support the arm and press the emergency stop.
- After a daemon death, support the arm before you start `serve` or `hold` again. The start can release each joint for a moment.
- After `SERVE_READY`, only `release` disables the motors. It asks you to type `RELEASE`. Until you do, the arm holds. Release the torque only when the arm is supported. `release --yes` does not ask. It is for a script that knows the arm is supported, such as a park at `home`.
- A release ramps the torque down before it disables the motors (ruling R36). For 2 s, each motor keeps its last position target, its `kp` and its feed-forward torque fall linearly to 0, and its `kd` stays. Thus a joint that rests above its stop, such as J3 at `home`, sinks slowly onto the stop. The health checks keep running during the ramp. If a fault occurs during the ramp, the ramp stops, the arm holds as on any fault, and `release` reports that it did not release. Run `release` again: the fault hold then ramps down. A gripper that its own fault released gets no ramp frames. The reply to `release` comes after the disable, about 2 s later than before. During the ramp, the daemon answers the other clients with mode `holding` and ignores commands. `hold` ramps the same way after the typed `RELEASE`.
- On a real interface, a release does not end the daemon. The daemon goes to mode `released`. It keeps the bus and reads the motors 5 times a second with no torque, so `status` and the ROS 2 client still see the joints move.
- In mode `released`, SIGINT or SIGTERM ends the daemon. It prints `SERVE_EXIT signal while released`, removes the socket and frees the bus lock. To start a daemon with other files, run `release`, then stop the old daemon, for example with `pkill -TERM -f 'maker-arm-cli serve --interface can0'`, then start the new one.
- `enable` enables a released arm again at the measured pose. It asks you to type `ENABLE`; `enable --yes` does not ask. The daemon runs the same checks as at start: the motor fault query, the 0.35 rad limit window, the temperatures, and the first tick's health check. The gripper must be below 45 C, and every other joint below its 70 C health limit. A hotter joint gets no torque. If a check fails, the daemon stays `released` and its log says why. A check that fails before any enable frame sends nothing more. If a motor got an enable frame and then does not confirm its disable, the daemon ends, as after a failed release. After an enable the daemon holds the measured pose, and the client must send Activate again. Support the arm first, as before every enable.
- On `sim` or `vcan`, a release ends the daemon.
- While the daemon runs, `scan`, `doctor`, `zero`, `hold` and the Python binding refuse to open the same interface. Each one names the bus lock `maker-arm-bus-<interface>`.
- On `sim` or `vcan`, the daemon ends with its command client. Pass `--persist` to keep it running.
- A command more than 0.05 rad from the last target is refused. The daemon holds until the client sends Deactivate and then Activate.

## Quick start

```python
import maker_arm_rs as m

# MIT control frame: motor 1, pos 1.0, vel 0, kp 60, kd 4, tau 0.5
can_id, data = m.encode_control(1, 1.0, 0.0, 60.0, 4.0, 0.5, "RS00")
print(can_id, data.hex())  # 1 8a2e8001ebccc849
fb = m.parse_frame(0xFD, bytes.fromhex("01BE907FD7FF00F0"), "RS00")
print(fb["kind"], fb["motor_id"], fb["temperature"])  # feedback 1 24.0
```

The same functions in Rust are `maker_arm_protocol::mit::{encode_control, parse_frame}`.
Put the frames on a bus with `maker_arm_transport::SocketCanBackend::open("can0")`
and its `CanBackend::{send, recv}`.

That raw pairing is for frame-level work and tests only — it bypasses the
single command clamp. The supported way to command a motor is through
`maker-arm`'s session and control loop (the `Arm` class below, or
`Session::start`), which passes every command through `clamp_command` before
it reaches a motor.

Before you open a real bus in Rust, take `maker_arm_transport::bus_lock("can0")` and keep it while you use the bus. It fails if a `serve` daemon holds the arm.

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
arm.stop()  # disable at once, with no ramp, and join; also aliased as arm.estop()
```

## Profile and clamp from Python

`maker_arm_rs.profile()` returns the pinned `maker_arm_v1` profile as a dict (7 joints: limits, gains, torque caps, direction/offset). `maker_arm_rs.clamp_command(cmds, profile)` applies the crate's single-point clamp to a list of 7 `(pos, vel, kp, kd, tau)` tuples and returns `(clamped_cmds, changed)`. The dict's numeric fields may be edited before the call (a simulator passes URDF joint limits this way). Names, models, and motor ids are never read from Python. The control loop's own command path always runs commands through this clamp before they reach a motor. `encode_control` is a separate low-level protocol encoder for tests and tooling, and it is not clamped.

Joint-coordinate contract: joint coordinates are the vendor URDF's
(`maker-arm-sdk/urdf/maker_arm/robot.urdf` @ b30d05a), `link_002_joint`..`link_007_joint`
= motor ids 1..6. `direction`/`offset` map motor to URDF coordinates and are calibrated
on the arm (MA1); the profile's `q_lo`/`q_hi` are motor-frame until then.
