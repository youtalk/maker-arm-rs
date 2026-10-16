use std::io::Write;
use std::path::PathBuf;

use clap::{error::ErrorKind, CommandFactory, Parser, Subcommand};
use maker_arm::{ArmConfig, HoldController, Session, SimArm};
use maker_arm_cli::files::ServeOptions;
use maker_arm_transport::CanBackend;

mod serve;

#[derive(Parser)]
#[command(
    name = "maker-arm-cli",
    about = "MakerMods Maker Arm CLI (maker-arm-rs)"
)]
struct Cli {
    /// SocketCAN interface, e.g. can0
    #[arg(long, global = true, conflicts_with = "sim")]
    can: Option<String>,
    /// Use the built-in 7-motor simulator instead of hardware
    #[arg(long, global = true)]
    sim: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Probe motors 1..=7 read-only and print what answers
    Scan,
    /// Scan plus param reads, limit checks, and known-trap warnings
    Doctor,
    /// Zero one motor's position register (torque-free; confirm required)
    Zero {
        #[arg(long)]
        motor: u8,
        /// Skip the interactive confirmation
        #[arg(long)]
        yes: bool,
    },
    /// Connect, enable, and hold the current pose until a typed RELEASE
    Hold,
    /// Ask a running serve daemon to disable the motors, after a typed RELEASE
    Release {
        #[arg(long)]
        interface: String,
        /// Default: $XDG_RUNTIME_DIR/maker-arm-<interface>.sock
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Print one STATUS line from a running serve daemon
    Status {
        #[arg(long)]
        interface: String,
        /// Default: $XDG_RUNTIME_DIR/maker-arm-<interface>.sock
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Own the arm as a daemon for ROS 2: track a client's targets over a Unix socket, and
    /// hold when the client stops
    Serve {
        /// can0, vcan0 or sim
        #[arg(long)]
        interface: String,
        /// A profile JSON that only narrows maker_arm_v1
        #[arg(long)]
        profile: Option<PathBuf>,
        /// The dynamics model JSON
        #[arg(long)]
        dynamics: PathBuf,
        /// The tracking parameters JSON
        #[arg(long)]
        params: Option<PathBuf>,
        /// The learned residual JSON
        #[arg(long)]
        residual: Option<PathBuf>,
        /// The measured gripper map: jaw gap (m) = a * angle + b
        #[arg(long = "gripper_map", value_name = "A,B")]
        gripper_map: Option<String>,
        /// Run the loop at SCHED_FIFO priority 80 on CPU 5
        #[arg(long)]
        rt: bool,
        /// Default: $XDG_RUNTIME_DIR/maker-arm-<interface>.sock
        #[arg(long)]
        socket: Option<PathBuf>,
        /// On sim or vcan, keep running after the command client leaves
        #[arg(long)]
        persist: bool,
        /// Load and check the files, print their digest and exit; opens no bus
        #[arg(long)]
        check: bool,
    },
}

fn open_backend(cli: &Cli, config: &ArmConfig) -> Result<Box<dyn CanBackend>, String> {
    match (&cli.can, cli.sim) {
        (Some(iface), false) => {
            #[cfg(feature = "socketcan")]
            {
                let b = maker_arm_transport::SocketCanBackend::open(iface)
                    .map_err(|e| format!("open {iface}: {e}"))?;
                Ok(Box::new(b))
            }
            #[cfg(not(feature = "socketcan"))]
            {
                let _ = iface;
                Err("built without the socketcan feature".into())
            }
        }
        (None, true) => Ok(Box::new(SimArm::new(config))),
        _ => Err("pass exactly one of --can <iface> or --sim".into()),
    }
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    if let Command::Serve {
        interface,
        profile,
        dynamics,
        params,
        residual,
        gripper_map,
        rt,
        socket,
        persist,
        check,
    } = cli.command
    {
        if cli.can.is_some() || cli.sim {
            Cli::command()
                .error(
                    ErrorKind::ArgumentConflict,
                    "serve takes --interface, not --can or --sim",
                )
                .exit();
        }
        let o = ServeOptions {
            interface,
            profile,
            dynamics,
            params,
            residual,
            gripper_map,
            rt,
            socket,
            persist,
        };
        return serve::run(&o, check);
    }
    match &cli.command {
        Command::Release { interface, socket } => {
            return serve::release(interface, socket.as_deref());
        }
        Command::Status { interface, socket } => {
            return serve::status(interface, socket.as_deref());
        }
        _ => {}
    }
    let config = ArmConfig::maker_arm_v1();
    let mut backend = open_backend(&cli, &config)?;
    match cli.command {
        Command::Scan => {
            let rows = maker_arm_cli::scan(backend.as_mut(), &config).map_err(|e| e.to_string())?;
            println!(
                "{:>3} {:>8} {:>5} {:>7} {:>9} {:>4} {:>6} {:>5}",
                "id", "name", "model", "present", "pos(rad)", "mode", "temp", "fault"
            );
            for r in rows {
                // `{:>5:#04x}` is not valid Rust format syntax (a spec takes
                // at most one `:`-separated section); format the hex value
                // as a string first and align that, per the brief's escape
                // hatch.
                let fault = format!("{:#04x}", r.fault_bits);
                println!(
                    "{:>3} {:>8} {:>5} {:>7} {:>9.3} {:>4} {:>6.1} {:>5}",
                    r.motor_id,
                    r.name,
                    r.model,
                    r.present,
                    r.joint_pos,
                    r.mode,
                    r.temperature,
                    fault
                );
            }
        }
        Command::Doctor => {
            let report =
                maker_arm_cli::doctor(backend.as_mut(), &config).map_err(|e| e.to_string())?;
            for r in &report.rows {
                println!(
                    "motor {} present={} pos={:.3} in_limits={} run_mode={:?} \
                     can_timeout={:?} vbus={:?} temp={:.1} fault={:#04x}",
                    r.motor_id,
                    r.present,
                    r.joint_pos,
                    r.in_limits,
                    r.run_mode,
                    r.can_timeout,
                    r.vbus,
                    r.temperature,
                    r.fault_bits
                );
            }
            for w in &report.warnings {
                println!("WARNING: {w}");
            }
            if report.rows.iter().any(|r| !r.present || !r.in_limits) {
                return Err("doctor found problems (see rows above)".into());
            }
        }
        Command::Zero { motor, yes } => {
            if !yes {
                println!(
                    "zeroing motor {motor} overwrites its zero position. Type YES to continue."
                );
                let mut line = String::new();
                std::io::stdin()
                    .read_line(&mut line)
                    .map_err(|e| e.to_string())?;
                if line.trim() != "YES" {
                    return Err("aborted".into());
                }
            }
            let pos = maker_arm_cli::zero_motor(backend.as_mut(), &config, motor)?;
            println!("motor {motor} zeroed; joint position now {pos:.4} rad");
            // Presentation only: `zero_motor` reports a true register
            // value, not a claim that it is a safe pose. Some joints'
            // configured ranges do not include zero (e.g. J5, J6), so
            // warn here -- `Session::enable()` remains the real gate and
            // will refuse to energize a joint left outside its range.
            if let Some(j) = config.joint_by_motor_id(motor) {
                if pos < j.q_lo || pos > j.q_hi {
                    println!(
                        "WARNING: {pos:.4} rad is outside motor {motor}'s configured range \
                         [{:.3}, {:.3}]; enable will refuse this pose until it is moved back \
                         inside range",
                        j.q_lo, j.q_hi
                    );
                }
            }
        }
        Command::Hold => {
            let mut session =
                Session::connect(backend, config.clone()).map_err(|e| e.to_string())?;
            session.enable().map_err(|e| e.to_string())?;
            let running = session
                .start(Box::new(HoldController::from_config(&config)), None)
                .expect("a start without rt cannot fail");
            // EVERYTHING from here to `confirm_release` runs with a live,
            // torque-on `RunningArm` on the stack, and `RunningArm`'s
            // `Drop` disables the motors. So any panic or `?` in this span
            // releases the arm with no typed RELEASE at all -- the exact
            // inversion two earlier fix rounds closed INSIDE
            // `confirm_release`. Nothing here may unwind or return early:
            // prints are best-effort (`println!` panics on a broken
            // stdout, e.g. `... --sim hold | head`), and a handler that
            // cannot be installed is a warning, not an exit.
            let _ = writeln!(
                std::io::stdout(),
                "enabled; holding at the current pose (200 Hz)."
            );
            // Ctrl-C must HOLD, never release (design §4 / upstream safety.py).
            {
                let running_hold = running.shared_hold_handle();
                if let Err(e) = ctrlc::set_handler(move || {
                    running_hold.hold_now();
                    let _ = writeln!(
                        std::io::stderr(),
                        "\ntorque remains enabled; type RELEASE when safe"
                    );
                }) {
                    // Losing the Ctrl-C convenience is strictly better than
                    // dropping a live arm because a handler could not be
                    // installed. Without it, Ctrl-C terminates this process
                    // outright and the motors' own CAN_TIMEOUT watchdog is
                    // the only backstop -- so say so, and continue into the
                    // gate, which is still the only way to release cleanly.
                    let _ = writeln!(
                        std::io::stderr(),
                        "WARNING: could not install the Ctrl-C handler ({e}); Ctrl-C will \
                         kill this process instead of holding, leaving the motors to their \
                         own CAN_TIMEOUT watchdog. Type RELEASE to release cleanly."
                    );
                }
            }
            let stdin = std::io::stdin();
            let mut input = stdin.lock();
            let mut output = std::io::stdout();
            maker_arm_cli::confirm_release(&mut input, &mut output);
            // Past the gate: torque is commanded off before anything below
            // can fail, so `?` is safe again from here on.
            let (_session, res) = running.stop_and_disable();
            res.map_err(|e| e.to_string())?;
            let _ = writeln!(std::io::stdout(), "torque released.");
        }
        Command::Serve { .. } | Command::Release { .. } | Command::Status { .. } => {
            unreachable!("handled above")
        }
    }
    Ok(())
}
