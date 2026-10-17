//! `maker-arm-cli export`: the sim files match the lab's reference files, every written file
//! loads through the daemon's loaders, and a refused input writes nothing.

use maker_arm::ArmConfig;
use maker_arm_cli::export::{self, ExportOptions};
use maker_arm_cli::files::{load, ServeOptions};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const BIN: &str = env!("CARGO_BIN_EXE_maker-arm-cli");
const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// A fresh directory under the system temp dir, for one test.
fn scratch_dir() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("mae-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rep<T: Into<Value>>(x: T, n: usize) -> Value {
    Value::Array(vec![x.into(); n])
}

fn fixture(name: &str) -> PathBuf {
    Path::new(FIX).join(name)
}

fn json_of(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn sim_options(dir: &Path) -> ExportOptions {
    ExportOptions {
        sim: true,
        urdf: Some(fixture("maker_arm_sdk.urdf")),
        out_dir: Some(dir.to_path_buf()),
        ..Default::default()
    }
}

fn rig_options(cal: &Path, caps: f64, out: &Path) -> ExportOptions {
    ExportOptions {
        calibration: Some(cal.to_path_buf()),
        caps: Some(caps),
        out: Some(out.to_path_buf()),
        ..Default::default()
    }
}

fn write_cal(dir: &Path, v: &Value) -> PathBuf {
    let p = dir.join("cal.json");
    std::fs::write(&p, v.to_string()).unwrap();
    p
}

/// Same shape and keys (except `provenance`), numbers within 1e-12; returns the largest gap.
fn compare(a: &Value, b: &Value, path: &str) -> f64 {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => (x.as_f64().unwrap() - y.as_f64().unwrap()).abs(),
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: length");
            x.iter()
                .zip(y)
                .enumerate()
                .map(|(i, (x, y))| compare(x, y, &format!("{path}[{i}]")))
                .fold(0.0, f64::max)
        }
        (Value::Object(x), Value::Object(y)) => {
            let keys = |m: &serde_json::Map<String, Value>| {
                m.keys()
                    .filter(|k| *k != "provenance")
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let (mut kx, mut ky) = (keys(x), keys(y));
            kx.sort();
            ky.sort();
            assert_eq!(kx, ky, "{path}: keys");
            kx.iter()
                .map(|k| compare(&x[k], &y[k], &format!("{path}.{k}")))
                .fold(0.0, f64::max)
        }
        _ => {
            assert_eq!(a, b, "{path}");
            0.0
        }
    }
}

fn serve_options(interface: &str, profile: &Path, dynamics: &Path) -> ServeOptions {
    ServeOptions {
        interface: interface.into(),
        profile: Some(profile.to_path_buf()),
        dynamics: dynamics.to_path_buf(),
        params: None,
        residual: None,
        gripper_map: None,
        rt: false,
        socket: None,
        persist: false,
    }
}

#[test]
fn sim_matches_the_reference_files() {
    let dir = scratch_dir();
    let written = export::run(&sim_options(&dir)).unwrap();
    assert_eq!(
        written,
        [dir.join("sim_profile.json"), dir.join("dynamics_urdf.json")]
    );
    let mut worst = 0.0_f64;
    for name in ["sim_profile.json", "dynamics_urdf.json"] {
        worst = worst.max(compare(
            &json_of(&dir.join(name)),
            &json_of(&fixture(name)),
            name,
        ));
    }
    eprintln!("largest parity difference: {worst:e}");
    assert!(worst <= 1e-12, "{worst:e}");
    let prov = &json_of(&dir.join("dynamics_urdf.json"))["provenance"];
    assert_eq!(prov["source"], "urdf");
    assert_eq!(prov["urdf_sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn sim_files_load() {
    let dir = scratch_dir();
    export::run(&sim_options(&dir)).unwrap();
    let o = serve_options(
        "sim",
        &dir.join("sim_profile.json"),
        &dir.join("dynamics_urdf.json"),
    );
    assert!(load(&o).is_ok());
}

#[test]
fn rig_profile_loads_for_can0_and_inverts_to_motor() {
    let dir = scratch_dir();
    let cal = write_cal(
        &dir,
        &json!({
            "direction": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            "offset": [0.1, -0.2, 0.3, 0.0, -0.4, 0.5, 0.05],
        }),
    );
    let out = dir.join("sub").join("rig.json");
    export::run(&rig_options(&cal, 0.5, &out)).unwrap();
    // The same code path as `serve --check --interface can0`.
    let l = load(&serve_options("can0", &out, &fixture("dynamics_urdf.json"))).unwrap();
    let v1 = ArmConfig::maker_arm_v1();
    for (j, b) in l.config.joints.iter().zip(&v1.joints) {
        let (m_lo, m_hi) = (j.to_motor(j.q_lo), j.to_motor(j.q_hi));
        let (lo, hi) = (m_lo.min(m_hi), m_lo.max(m_hi));
        assert!(
            (lo - b.q_lo).abs() < 1e-12 && (hi - b.q_hi).abs() < 1e-12,
            "{}",
            j.name
        );
        assert!(j.q_lo < j.q_hi);
        assert_eq!(j.kp, b.kp * 0.5);
        assert_eq!(j.kd, b.kd * 0.5);
        assert_eq!(j.tau_max, b.tau_max * 0.5);
    }
    // R33: a capped profile halves the error clamp's ceiling headroom (e_max / kp scale).
    assert!((l.params.e_max[0] - 0.06).abs() < 1e-12);
}

#[test]
fn rig_limits_survive_any_offset() {
    // direction * (v - offset) mapped back by to_motor can land one ulp outside the v1 range.
    let dir = scratch_dir();
    let out = dir.join("rig.json");
    for k in 0..200 {
        let o = -3.0 + 0.03 * f64::from(k);
        let d = if k % 2 == 0 { 1.0 } else { -1.0 };
        let cal = write_cal(&dir, &json!({"direction": rep(d, 7), "offset": rep(o, 7)}));
        match export::run(&rig_options(&cal, 1.0, &out)) {
            Ok(_) => {
                let l = load(&serve_options("can0", &out, &fixture("dynamics_urdf.json")));
                assert!(l.is_ok(), "offset {o}: {:?}", l.err());
            }
            // Only a joint range that cannot fit may refuse, never a rounding slip.
            Err(e) => assert!(
                !e.contains("outside the maker_arm_v1 range") || false,
                "{o}: {e}"
            ),
        }
    }
}

#[test]
fn bad_inputs_write_nothing() {
    let dir = scratch_dir();
    let ok = json!({"direction": rep(1.0, 7), "offset": rep(0.0, 7)});
    let cases: Vec<(Value, f64)> = vec![
        (
            json!({"direction": rep(1.0, 7), "offset": rep(0.0, 7), "extra": 1}),
            1.0,
        ),
        (json!({"direction": rep(1.0, 7)}), 1.0),
        (json!({"offset": rep(0.0, 7)}), 1.0),
        (
            json!({"direction": rep(1.0, 6), "offset": rep(0.0, 7)}),
            1.0,
        ),
        (
            json!({"direction": [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "offset": rep(0.0, 7)}),
            1.0,
        ),
        (
            json!({"direction": rep(1.0, 7), "offset": [3.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
            1.0,
        ),
        (ok.clone(), 0.0),
        (ok.clone(), 1.5),
        (ok.clone(), -0.5),
    ];
    for (i, (cal, caps)) in cases.iter().enumerate() {
        let out = dir.join(format!("out{i}.json"));
        let e = export::run(&rig_options(&write_cal(&dir, cal), *caps, &out)).unwrap_err();
        assert!(!e.is_empty() && !e.contains('\n'), "case {i}: {e}");
        assert!(!out.exists(), "case {i} wrote a file");
    }
    // A URDF that is not a URDF: nothing lands in the directory.
    let bad = dir.join("bad.urdf");
    std::fs::write(&bad, "<robot/>").unwrap();
    let outdir = dir.join("simout");
    let o = ExportOptions {
        urdf: Some(bad),
        ..sim_options(&outdir)
    };
    assert!(export::run(&o).is_err());
    assert!(!outdir.exists());
}

#[test]
fn ident_dynamics_carry_the_ident_bodies() {
    let dir = scratch_dir();
    let reference = json_of(&fixture("dynamics_urdf.json"));
    let ident = dir.join("ident.json");
    let friction = json!(rep(json!([0.1, 0.2, 0.01, 0.15, 0.05]), 6));
    let body = &reference["bodies"];
    std::fs::write(
        &ident,
        json!({"bodies": body, "friction": friction, "armature": rep(0.001, 6), "provenance": {"run": 7}})
            .to_string(),
    )
    .unwrap();
    let out = dir.join("dyn.json");
    let o = ExportOptions {
        ident: Some(ident),
        urdf: Some(fixture("maker_arm_sdk.urdf")),
        out: Some(out.clone()),
        ..Default::default()
    };
    export::run(&o).unwrap();
    let v = json_of(&out);
    assert_eq!(
        v["provenance"],
        json!({"source": "ident", "ident": {"run": 7}})
    );
    assert_eq!(v["friction"], friction);
    assert_eq!(v["armature"], json!(rep(0.001, 6)));
    assert!(compare(&v["links"], &reference["links"], "links") <= 1e-12);
    maker_arm_cli::files::dynamics_from_json(&v).unwrap();
    // A file without armature is refused.
    let short = dir.join("short.json");
    std::fs::write(
        &short,
        json!({"bodies": body, "friction": friction}).to_string(),
    )
    .unwrap();
    let o = ExportOptions {
        ident: Some(short),
        out: Some(dir.join("no.json")),
        ..o
    };
    assert!(export::run(&o).is_err());
    assert!(!dir.join("no.json").exists());
}

#[test]
fn the_command_prints_each_path_and_fails_with_one_line() {
    let dir = scratch_dir();
    let run = |args: &[&str]| Command::new(BIN).arg("export").args(args).output().unwrap();
    let urdf = fixture("maker_arm_sdk.urdf");
    let o = run(&[
        "--sim",
        "--urdf",
        urdf.to_str().unwrap(),
        "--out_dir",
        dir.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{o:?}");
    let out = String::from_utf8(o.stdout).unwrap();
    assert_eq!(out.lines().count(), 2, "{out}");
    assert!(out.lines().all(|l| Path::new(l).exists()), "{out}");
    let cal = write_cal(
        &dir,
        &json!({"direction": rep(0.5, 7), "offset": rep(0.0, 7)}),
    );
    let o = run(&[
        "--calibration",
        cal.to_str().unwrap(),
        "--caps",
        "1",
        "--out",
        "/nonexistent/x.json",
    ]);
    assert!(!o.status.success());
    let err = String::from_utf8(o.stderr).unwrap();
    assert_eq!(err.trim().lines().count(), 1, "{err}");
    assert!(err.contains("direction"), "{err}");
}
