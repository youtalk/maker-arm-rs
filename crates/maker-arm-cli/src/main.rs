use clap::{Parser, Subcommand};
use maker_arm::{ArmConfig, SimArm};
use maker_arm_transport::CanBackend;

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
    }
    Ok(())
}
