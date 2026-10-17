//! `maker-arm-cli export`: writes the profile and dynamics files that `serve` loads, from a
//! URDF, an identification file or a joint calibration. Every file is built and checked with
//! the daemon's own loaders before the first byte is written, so a refusal writes nothing.
//! maker-arm-rs holds no robot data: the URDF is always an input.

use crate::files::{self, dynamics_from_json, profile_from_json};
use maker_arm::kinematics::ARM_DOF;
use maker_arm::ArmConfig;
use roxmltree::{Document, Node};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

// The constants below are ported from the private research repo that wrote the reference files
// (maker-arm-lab); each comment names its source.

/// `assets.urdf_prep.ARM_JOINT_NAMES`.
const ARM_JOINT_NAMES: [&str; ARM_DOF] = [
    "link_002_joint",
    "link_003_joint",
    "link_004_joint",
    "link_005_joint",
    "link_006_joint",
    "link_007_joint",
];
/// `assets.urdf_prep.GRIPPER_JOINT_NAMES`.
const GRIPPER_JOINT_NAMES: [&str; 2] = ["gripper_left_joint", "gripper_right_joint"];
/// `core.dynamics_model.ARM_LINKS`.
const ARM_LINKS: [&str; ARM_DOF] = [
    "link_002", "link_003", "link_004", "link_005", "link_006", "link_007",
];
/// `core.dynamics_model.JAW_LINKS`.
const JAW_LINKS: [&str; 2] = ["gripper_left", "gripper_right"];
/// `assets.urdf_prep.MOUNT_RPY`: the CAD's Y-up base to world Z-up, then a yaw of pi.
const MOUNT_RPY: [f64; 3] = [std::f64::consts::FRAC_PI_2, 0.0, std::f64::consts::PI];
/// `core.dynamics_model.GRAVITY`.
const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];
/// `core.dynamics_model.JAW_FRACTION`: the lumped jaws sit at half their travel.
const JAW_FRACTION: f64 = 0.5;

type Vec3 = [f64; 3];
type Mat3 = [Vec3; 3];
/// Mass, center of mass, inertia about the center of mass.
type Body = (f64, Vec3, Mat3);

#[derive(Debug, Default)]
pub struct ExportOptions {
    pub sim: bool,
    pub urdf: Option<PathBuf>,
    pub calibration: Option<PathBuf>,
    pub ident: Option<PathBuf>,
    pub caps: Option<f64>,
    pub out: Option<PathBuf>,
    pub out_dir: Option<PathBuf>,
}

/// Builds and checks the files, then writes them. Returns the paths written.
pub fn run(o: &ExportOptions) -> Result<Vec<PathBuf>, String> {
    let need = |p: &Option<PathBuf>, flag: &str, mode: &str| {
        p.clone().ok_or_else(|| format!("{mode} needs {flag}"))
    };
    // A flag that does not belong to the mode is an error, never silently ignored.
    let only = |mode: &str, stray: &[(&str, bool)]| match stray.iter().find(|s| s.1) {
        Some((flag, _)) => Err(format!("{mode} does not take {flag}")),
        None => Ok(()),
    };
    let files = match (o.sim, &o.calibration, &o.ident) {
        (true, None, None) => {
            only(
                "--sim",
                &[("--caps", o.caps.is_some()), ("--out", o.out.is_some())],
            )?;
            let dir = need(&o.out_dir, "--out_dir", "--sim")?;
            let urdf = need(&o.urdf, "--urdf", "--sim")?;
            sim_files(&urdf, &dir)?
        }
        (false, Some(cal), None) => {
            only(
                "--calibration",
                &[
                    ("--urdf", o.urdf.is_some()),
                    ("--out_dir", o.out_dir.is_some()),
                ],
            )?;
            let caps = o.caps.ok_or("--calibration needs --caps")?;
            let out = need(&o.out, "--out", "--calibration")?;
            vec![rig_profile(cal, caps, out)?]
        }
        (false, None, Some(ident)) => {
            only(
                "--ident",
                &[
                    ("--caps", o.caps.is_some()),
                    ("--out_dir", o.out_dir.is_some()),
                ],
            )?;
            let out = need(&o.out, "--out", "--ident")?;
            let urdf = need(&o.urdf, "--urdf", "--ident")?;
            vec![ident_dynamics(ident, &urdf, out)?]
        }
        _ => return Err("pass exactly one of --sim, --calibration or --ident".into()),
    };
    for (path, text) in &files {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(files.into_iter().map(|(p, _)| p).collect())
}

/// The exact text to write for `v`, after `check` accepted what a reader of that text gets.
/// Checking the reparsed text means the guarantee does not rest on how floats round-trip.
fn seal(
    path: PathBuf,
    v: &Value,
    check: impl Fn(&Value) -> Result<(), String>,
) -> Result<(PathBuf, String), String> {
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())? + "\n";
    let read: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    check(&read).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((path, text))
}

fn check_profile(v: &Value, interface: &str) -> Result<(), String> {
    files::check_tracking(&profile_from_json(v, interface)?)
}

fn check_dynamics(v: &Value) -> Result<(), String> {
    dynamics_from_json(v).map(|_| ())
}

fn read_text(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_str(&read_text(path)?).map_err(|e| format!("{}: {e}", path.display()))
}

fn sim_files(urdf: &Path, dir: &Path) -> Result<Vec<(PathBuf, String)>, String> {
    let text = read_text(urdf)?;
    let at = |e: String| format!("{}: {e}", urdf.display());
    let doc = Document::parse(&text).map_err(|e| at(e.to_string()))?;
    check_chain(&doc).map_err(at)?;
    let mut profile = profile_json(&ArmConfig::maker_arm_v1());
    let joints = profile["joints"].as_array_mut().expect("joints is a list");
    for (j, name) in joints.iter_mut().zip(ARM_JOINT_NAMES) {
        let limit = child(joint(&doc, name).map_err(at)?, "limit").map_err(at)?;
        j["q_lo"] = json!(attr_f64(limit, "lower").map_err(at)?);
        j["q_hi"] = json!(attr_f64(limit, "upper").map_err(at)?);
    }
    let provenance =
        json!({"source": "urdf", "urdf_sha256": hex(&Sha256::digest(text.as_bytes()))});
    let bodies = urdf_bodies(&doc).map_err(at)?;
    let bodies: Vec<Value> = bodies.iter().map(|b| json!([b.0, b.1, b.2])).collect();
    let dynamics =
        dynamics_spec(&doc, json!(bodies), Value::Null, Value::Null, provenance).map_err(at)?;
    Ok(vec![
        seal(dir.join("sim_profile.json"), &profile, |v| {
            check_profile(v, "sim")
        })?,
        seal(dir.join("dynamics_urdf.json"), &dynamics, check_dynamics)?,
    ])
}

fn ident_dynamics(ident: &Path, urdf: &Path, out: PathBuf) -> Result<(PathBuf, String), String> {
    let at = |e: String| format!("{}: {e}", ident.display());
    let data = read_json(ident)?;
    let data = data
        .as_object()
        .ok_or_else(|| at("must be an object".into()))?;
    let get = |k: &str| {
        data.get(k)
            .cloned()
            .ok_or_else(|| at(format!("{k}: missing")))
    };
    let provenance = json!({"source": "ident", "ident": data.get("provenance")});
    let text = read_text(urdf)?;
    let doc = Document::parse(&text).map_err(|e| format!("{}: {e}", urdf.display()))?;
    check_chain(&doc).map_err(|e| format!("{}: {e}", urdf.display()))?;
    let v = dynamics_spec(
        &doc,
        get("bodies")?,
        get("friction")?,
        get("armature")?,
        provenance,
    )
    .map_err(|e| format!("{}: {e}", urdf.display()))?;
    seal(out, &v, check_dynamics)
}

/// Port of the lab's `export_rig`. The v1 profile, with `kp`, `kd` and `tau_max` times `caps`
/// and each joint's limits in the joint frame: `sorted(direction * (v - offset))`, which
/// `JointConfig::to_motor` maps back to the v1 motor limits.
fn rig_profile(calibration: &Path, caps: f64, out: PathBuf) -> Result<(PathBuf, String), String> {
    if !(caps > 0.0 && caps <= 1.0) {
        return Err(format!("--caps must be in (0, 1], got {caps}"));
    }
    let at = |e: String| format!("{}: {e}", calibration.display());
    let cal = read_json(calibration)?;
    let cal = cal
        .as_object()
        .ok_or_else(|| at("must be an object".into()))?;
    if let Some(k) = cal
        .keys()
        .find(|k| !["direction", "offset"].contains(&k.as_str()))
    {
        return Err(at(format!("{k}: unknown field")));
    }
    let list = |k: &str| -> Result<Vec<f64>, String> {
        let a = cal.get(k).ok_or_else(|| at(format!("{k}: missing")))?;
        let a: Option<Vec<f64>> = a
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect());
        match (a, cal[k].as_array()) {
            (Some(x), Some(raw)) if x.len() == raw.len() && x.len() == 7 => Ok(x),
            _ => Err(at(format!("{k}: must be a list of 7 numbers"))),
        }
    };
    let (direction, offset) = (list("direction")?, list("offset")?);
    let v1 = ArmConfig::maker_arm_v1();
    let mut config = v1.clone();
    for ((j, b), (d, o)) in config
        .joints
        .iter_mut()
        .zip(&v1.joints)
        .zip(direction.iter().zip(&offset))
    {
        j.kp *= caps;
        j.kd *= caps;
        j.tau_max *= caps;
        (j.direction, j.offset) = (*d, *o);
        let (x, y) = (d * (b.q_lo - o), d * (b.q_hi - o));
        let (lo, hi) = (x.min(y), x.max(y));
        // The round trip can land an ulp outside the v1 range, which the loader refuses; a
        // step inward is the safe side.
        let inside = |q: f64| (b.q_lo..=b.q_hi).contains(&j.to_motor(q));
        let nudge = |mut q: f64, toward: f64| {
            for _ in 0..4 {
                if inside(q) {
                    break;
                }
                q += (toward - q).signum() * q.abs().max(f64::MIN_POSITIVE) * f64::EPSILON;
            }
            q
        };
        (j.q_lo, j.q_hi) = (nudge(lo, hi), nudge(hi, lo));
    }
    let v = profile_json(&config);
    seal(out, &v, |v| check_profile(v, "can0"))
}

/// The profile in the file format `profile_from_json` reads.
fn profile_json(c: &ArmConfig) -> Value {
    let joints: Vec<Value> = c
        .joints
        .iter()
        .map(|j| {
            json!({
                "motor_id": j.motor_id, "name": j.name, "model": j.model.name(),
                "kp": j.kp, "kd": j.kd, "tau_max": j.tau_max, "q_lo": j.q_lo, "q_hi": j.q_hi,
                "direction": j.direction, "offset": j.offset,
            })
        })
        .collect();
    json!({
        "control_rate_hz": c.control_rate_hz, "max_velocity": c.max_velocity,
        "feedback_timeout": c.feedback_timeout, "limit_margin": c.limit_margin,
        "kp_max": c.kp_max, "kd_max": c.kd_max, "temp_hold_c": c.temp_hold_c,
        "joints": joints,
    })
}

/// Port of the lab's `_dynamics_spec`; `links` is `urdf_prep.chain_links`.
fn dynamics_spec(
    doc: &Document,
    bodies: Value,
    friction: Value,
    armature: Value,
    provenance: Value,
) -> Result<Value, String> {
    let mut links = Vec::new();
    for name in ARM_JOINT_NAMES {
        let j = joint(doc, name)?;
        let axis = match j.children().find(|n| n.has_tag_name("axis")) {
            Some(a) => json!(floats(a.attribute("xyz"))?),
            None => Value::Null,
        };
        let o = child(j, "origin")?;
        links.push(json!([
            floats(o.attribute("xyz"))?,
            floats(o.attribute("rpy"))?,
            axis
        ]));
    }
    let mut m = Map::new();
    m.insert("mount_rpy".into(), json!(MOUNT_RPY));
    m.insert("links".into(), json!(links));
    m.insert("gravity".into(), json!(GRAVITY));
    m.insert("bodies".into(), bodies);
    m.insert("friction".into(), friction);
    m.insert("armature".into(), armature);
    m.insert("provenance".into(), provenance);
    Ok(Value::Object(m))
}

/// The arm joints must chain link_002 to link_007 and both jaws must hang on link_007, or the
/// bodies below would sit in the wrong frames.
fn check_chain(doc: &Document) -> Result<(), String> {
    let link_of = |name: &str, tag: &str| -> Result<String, String> {
        let l = child(joint(doc, name)?, tag)?;
        Ok(l.attribute("link").unwrap_or("").to_string())
    };
    let wrong = |name: &str, tag: &str, got: &str, want: &str| {
        Err(format!(
            "joint {name}: {tag} link is {got:?}, expected {want:?}"
        ))
    };
    for (name, link) in ARM_JOINT_NAMES.iter().zip(ARM_LINKS) {
        let got = link_of(name, "child")?;
        if got != link {
            return wrong(name, "child", &got, link);
        }
    }
    for (name, link) in GRIPPER_JOINT_NAMES.iter().zip(JAW_LINKS) {
        let (parent, kid) = (link_of(name, "parent")?, link_of(name, "child")?);
        if parent != ARM_LINKS[ARM_DOF - 1] {
            return wrong(name, "parent", &parent, ARM_LINKS[ARM_DOF - 1]);
        }
        if kid != link {
            return wrong(name, "child", &kid, link);
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn joint<'a, 'i>(doc: &'a Document<'i>, name: &str) -> Result<Node<'a, 'i>, String> {
    doc.descendants()
        .find(|n| n.has_tag_name("joint") && n.attribute("name") == Some(name))
        .ok_or_else(|| format!("joint {name}: missing in the URDF"))
}

fn child<'a, 'i>(n: Node<'a, 'i>, tag: &str) -> Result<Node<'a, 'i>, String> {
    n.children()
        .find(|c| c.has_tag_name(tag))
        .ok_or_else(|| format!("<{}> has no <{tag}>", n.tag_name().name()))
}

fn attr_f64(n: Node, attr: &str) -> Result<f64, String> {
    n.attribute(attr)
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(|| {
            format!(
                "<{}>: {attr} is missing or not a number",
                n.tag_name().name()
            )
        })
}

/// Three numbers from an attribute; a missing attribute reads as "0 0 0" (the lab's default).
fn floats(text: Option<&str>) -> Result<Vec3, String> {
    let v: Vec<f64> = text
        .unwrap_or("0 0 0")
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    <Vec3>::try_from(v).map_err(|_| format!("{text:?} is not three numbers"))
}

/// `core.dynamics_model._rpy`.
fn rpy(r: f64, p: f64, y: f64) -> Mat3 {
    let (cr, sr, cp, sp, cy, sy) = (r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin());
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn mv(m: &Mat3, v: &Vec3) -> Vec3 {
    std::array::from_fn(|r| (0..3).map(|k| m[r][k] * v[k]).sum())
}

/// `core.dynamics_model._rotate`: R I R^T.
fn rotate(r: &Mat3, i: &Mat3) -> Mat3 {
    let ri: Mat3 =
        std::array::from_fn(|a| std::array::from_fn(|b| (0..3).map(|k| r[a][k] * i[k][b]).sum()));
    std::array::from_fn(|a| std::array::from_fn(|b| (0..3).map(|k| ri[a][k] * r[b][k]).sum()))
}

/// `core.dynamics_model.lump`: total mass, joint center of mass, parallel-axis inertia.
fn lump(parts: &[Body]) -> Body {
    let mass: f64 = parts.iter().map(|p| p.0).sum();
    let com: Vec3 = std::array::from_fn(|k| parts.iter().map(|p| p.0 * p.1[k]).sum::<f64>() / mass);
    let mut inertia = [[0.0; 3]; 3];
    for (m, c, i) in parts {
        let d: Vec3 = std::array::from_fn(|k| c[k] - com[k]);
        let dd: f64 = d.iter().map(|v| v * v).sum();
        for a in 0..3 {
            for b in 0..3 {
                inertia[a][b] += i[a][b] + m * ((if a == b { dd } else { 0.0 }) - d[a] * d[b]);
            }
        }
    }
    (mass, com, inertia)
}

/// `core.dynamics_model.urdf_inertials` for one link; an error without an `<inertial>`.
fn inertial(doc: &Document, link: &str) -> Result<Body, String> {
    let l = doc
        .descendants()
        .find(|n| n.has_tag_name("link") && n.attribute("name") == Some(link))
        .ok_or_else(|| format!("link {link}: missing in the URDF"))?;
    let i = child(l, "inertial").map_err(|_| format!("link {link}: has no <inertial>"))?;
    let o = i.children().find(|n| n.has_tag_name("origin"));
    let xyz = floats(o.and_then(|o| o.attribute("xyz")))?;
    let r = floats(o.and_then(|o| o.attribute("rpy")))?;
    let a = child(i, "inertia")?;
    let g = |k: &str| attr_f64(a, k);
    let i0 = [
        [g("ixx")?, g("ixy")?, g("ixz")?],
        [g("ixy")?, g("iyy")?, g("iyz")?],
        [g("ixz")?, g("iyz")?, g("izz")?],
    ];
    let mass = attr_f64(child(i, "mass")?, "value")?;
    Ok((mass, xyz, rotate(&rpy(r[0], r[1], r[2]), &i0)))
}

/// `core.dynamics_model._jaw_in_link_007`: a jaw body moved into link_007's frame.
fn jaw_in_link_007(body: &Body, j: Node, opening: f64) -> Result<Body, String> {
    let o = child(j, "origin")?;
    let p = floats(o.attribute("xyz"))?;
    let r = floats(o.attribute("rpy"))?;
    let r = rpy(r[0], r[1], r[2]);
    let axis = floats(child(j, "axis")?.attribute("xyz"))?;
    let local: Vec3 = std::array::from_fn(|k| body.1[k] + opening * axis[k]);
    let com: Vec3 = std::array::from_fn(|k| mv(&r, &local)[k] + p[k]);
    Ok((body.0, com, rotate(&r, &body.2)))
}

/// `core.dynamics_model.urdf_bodies`: the six bodies in joint order, jaws lumped into link_007.
fn urdf_bodies(doc: &Document) -> Result<Vec<Body>, String> {
    let travel = attr_f64(
        child(joint(doc, GRIPPER_JOINT_NAMES[0])?, "limit")?,
        "upper",
    )?;
    let opening = JAW_FRACTION * travel;
    let mut bodies = Vec::new();
    for link in &ARM_LINKS[..ARM_DOF - 1] {
        bodies.push(inertial(doc, link)?);
    }
    let mut parts = vec![inertial(doc, ARM_LINKS[ARM_DOF - 1])?];
    for (name, link) in GRIPPER_JOINT_NAMES.iter().zip(JAW_LINKS) {
        parts.push(jaw_in_link_007(
            &inertial(doc, link)?,
            joint(doc, name)?,
            opening,
        )?);
    }
    bodies.push(lump(&parts));
    Ok(bodies)
}
