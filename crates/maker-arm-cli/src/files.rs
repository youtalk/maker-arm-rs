//! The daemon's files: the profile, the dynamics model, the tracking parameters and the
//! residual. Each loader checks its file and names the field that fails; `load` also
//! digests the raw bytes, so a peer that reads the same files gets the same digest.

use crate::wire;
use maker_arm::dynamics::{Body, Dynamics, Friction};
use maker_arm::kinematics::{Link, ARM_DOF};
use maker_arm::residual::{Layer, Mlp, ResidualModel};
use maker_arm::tracking::TrackingParams;
use maker_arm::ArmConfig;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// Half-stroke of the gripper's jaw, per finger (m).
const JAW_HALF_STROKE: f64 = 0.0524125;

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub interface: String,
    pub profile: Option<PathBuf>,
    pub dynamics: PathBuf,
    pub params: Option<PathBuf>,
    pub residual: Option<PathBuf>,
    /// The raw `--gripper_map` text, "a,b".
    pub gripper_map: Option<String>,
    pub rt: bool,
    pub socket: Option<PathBuf>,
    pub persist: bool,
}

/// `sim` and the `vcan` interfaces have no motors, so URDF-frame limits are fine there.
pub fn is_virtual(interface: &str) -> bool {
    interface == "sim" || interface.starts_with("vcan")
}

/// The jaw gap in m is `a * angle + b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GripMap {
    pub a: f64,
    pub b: f64,
}

impl GripMap {
    pub fn new(a: f64, b: f64) -> Result<GripMap, String> {
        if !a.is_finite() || !b.is_finite() || a == 0.0 {
            return Err(format!(
                "--gripper_map: a={a}, b={b} needs finite numbers and a != 0"
            ));
        }
        Ok(GripMap { a, b })
    }

    /// A gap of 0 at the gripper's `q_lo` and a full stroke at its `q_hi`.
    pub fn provisional(config: &ArmConfig) -> GripMap {
        let g = &config.joints[ARM_DOF];
        let a = 2.0 * JAW_HALF_STROKE / (g.q_hi - g.q_lo);
        GripMap { a, b: -a * g.q_lo }
    }

    fn parse(text: &str) -> Result<GripMap, String> {
        let nums: Vec<f64> = text
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        match nums[..] {
            [a, b] if text.split(',').count() == 2 => GripMap::new(a, b),
            _ => Err(format!(
                "--gripper_map: {text:?} is not two numbers \"a,b\""
            )),
        }
    }

    pub fn to_joint_m(&self, angle: f64) -> f64 {
        (self.a * angle + self.b) / 2.0
    }

    pub fn to_angle(&self, joint_m: f64) -> f64 {
        (2.0 * joint_m - self.b) / self.a
    }

    pub fn joint_m_per_rad(&self) -> f64 {
        self.a / 2.0
    }
}

pub struct Loaded {
    pub config: ArmConfig,
    pub dynamics: Dynamics,
    pub params: TrackingParams,
    pub residual: Option<ResidualModel>,
    pub grip: GripMap,
    pub digest: u64,
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn parse(path: &Path, bytes: &[u8]) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn load(o: &ServeOptions) -> Result<Loaded, String> {
    let at = |p: &Path, e: String| format!("{}: {e}", p.display());
    let profile = o.profile.as_deref().map(read).transpose()?;
    let dynamics = read(&o.dynamics)?;
    let params = o.params.as_deref().map(read).transpose()?;
    let residual = o.residual.as_deref().map(read).transpose()?;

    let config = match (&o.profile, &profile) {
        (Some(p), Some(b)) => {
            profile_from_json(&parse(p, b)?, &o.interface).map_err(|e| at(p, e))?
        }
        _ => ArmConfig::maker_arm_v1(),
    };
    let (model, source) =
        dynamics_from_json(&parse(&o.dynamics, &dynamics)?).map_err(|e| at(&o.dynamics, e))?;
    let (mut tracking, given) = match (&o.params, &params) {
        (Some(p), Some(b)) => {
            let v = parse(p, b)?;
            let given = v.get("e_max").is_some();
            (params_from_json(&v).map_err(|e| at(p, e))?, given)
        }
        _ => (TrackingParams::default(), false),
    };
    apply_e_max_cap(&mut tracking, given, &config).map_err(|e| match &o.params {
        Some(p) => at(p, e),
        None => e,
    })?;
    if tracking.tau_th.iter().any(|t| t.is_finite()) && source != "ident" {
        return Err(
            "a deadband ships only with the identified model: params set a finite tau_th \
                    but the dynamics file's provenance.source is not \"ident\""
                .into(),
        );
    }
    let residual_model = match (&o.residual, &residual) {
        (Some(p), Some(b)) => Some(residual_from_json(&parse(p, b)?).map_err(|e| at(p, e))?),
        _ => None,
    };
    let grip = match &o.gripper_map {
        Some(t) => GripMap::parse(t)?,
        None => GripMap::provisional(&config),
    };
    let empty: &[u8] = &[];
    let opt = |b: &Option<Vec<u8>>| b.as_deref().unwrap_or(empty).to_vec();
    let digest = wire::digest(&[
        ("interface", o.interface.as_bytes()),
        ("profile", &opt(&profile)),
        ("dynamics", &dynamics),
        ("params", &opt(&params)),
        ("residual", &opt(&residual)),
        (
            "gripper_map",
            o.gripper_map.as_deref().unwrap_or("").as_bytes(),
        ),
    ]);
    Ok(Loaded {
        config,
        dynamics: model,
        params: tracking,
        residual: residual_model,
        grip,
        digest,
    })
}

/// The part of the controller build that depends on the profile only: the default tracking
/// parameters under R33, then the checks `TrackingController::new` makes on them and on the
/// gains. `export` runs it so that it never writes a profile that `serve` refuses.
pub fn check_tracking(config: &ArmConfig) -> Result<(), String> {
    let mut p = TrackingParams::default();
    apply_e_max_cap(&mut p, false, config)?;
    p.validate().map_err(|e| e.to_string())?;
    for (i, j) in config.joints.iter().take(ARM_DOF).enumerate() {
        if j.kp <= 0.0 {
            return Err(format!("joints[{i}].kp: {} is not positive", j.kp));
        }
    }
    Ok(())
}

/// Ruling R33: `e_max_j = default / kp_scale_j`, so the spring ceiling `kp e_max` stays at the
/// v1 value when the profile cuts kp. A params `e_max` is kept only where it is within that cap.
fn apply_e_max_cap(p: &mut TrackingParams, given: bool, c: &ArmConfig) -> Result<(), String> {
    let v1 = ArmConfig::maker_arm_v1();
    for (j, base) in TrackingParams::default().e_max.into_iter().enumerate() {
        let cap = base / (c.joints[j].kp / v1.joints[j].kp);
        if !given {
            p.e_max[j] = cap;
        } else if p.e_max[j] > cap * (1.0 + 1e-12) {
            return Err(format!(
                "e_max[{j}]: {} exceeds the cap {cap} (R33: {base} / kp scale)",
                p.e_max[j]
            ));
        }
    }
    Ok(())
}

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>, String> {
    v.as_object()
        .ok_or_else(|| format!("{what}: must be an object"))
}

fn num(v: &Value, field: &str) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| format!("{field}: must be a number"))
}

fn nums(v: &Value, field: &str) -> Result<Vec<f64>, String> {
    let a = v
        .as_array()
        .ok_or_else(|| format!("{field}: must be a list of numbers"))?;
    a.iter()
        .enumerate()
        .map(|(i, x)| num(x, &format!("{field}[{i}]")))
        .collect()
}

fn fixed<const N: usize>(v: &Value, field: &str) -> Result<[f64; N], String> {
    let x = nums(v, field)?;
    let n = x.len();
    x.try_into()
        .map_err(|_| format!("{field}: expected {N} numbers, got {n}"))
}

/// A profile with the numeric overrides the binding's session path allows. The rules are
/// `reduced_config` in bindings/maker-arm-py: a profile may only be stricter than v1.
pub fn profile_from_json(v: &Value, interface: &str) -> Result<ArmConfig, String> {
    const OVERRIDE: [&str; 4] = ["max_velocity", "kp_max", "kd_max", "limit_margin"];
    const FIXED: [&str; 3] = ["control_rate_hz", "feedback_timeout", "temp_hold_c"];
    let v1 = ArmConfig::maker_arm_v1();
    let mut c = v1.clone();
    let top = object(v, "profile")?;
    let same = |field: String, got: &Value, want: Value| {
        if *got == want || (got.is_number() && got.as_f64() == want.as_f64()) {
            Ok(())
        } else {
            Err(format!(
                "{field}: cannot be changed from the maker_arm_v1 value"
            ))
        }
    };
    for (key, val) in top {
        if key == "joints" {
            let items = val.as_array().ok_or("joints: must be a list")?;
            if items.len() != c.joints.len() {
                return Err(format!(
                    "joints: {} joints, arm has {}",
                    items.len(),
                    c.joints.len()
                ));
            }
            for (i, (item, (j, b))) in items
                .iter()
                .zip(c.joints.iter_mut().zip(&v1.joints))
                .enumerate()
            {
                for (k, x) in object(item, &format!("joints[{i}]"))? {
                    let field = format!("joints[{i}].{k}");
                    let slot = match k.as_str() {
                        "kp" => &mut j.kp,
                        "kd" => &mut j.kd,
                        "tau_max" => &mut j.tau_max,
                        "q_lo" => &mut j.q_lo,
                        "q_hi" => &mut j.q_hi,
                        "direction" => &mut j.direction,
                        "offset" => &mut j.offset,
                        "motor_id" => {
                            same(field, x, b.motor_id.into())?;
                            continue;
                        }
                        "name" => {
                            same(field, x, b.name.into())?;
                            continue;
                        }
                        "model" => {
                            same(field, x, b.model.name().into())?;
                            continue;
                        }
                        _ => return Err(format!("{field}: unknown field")),
                    };
                    *slot = num(x, &field)?;
                }
            }
        } else if OVERRIDE.contains(&key.as_str()) {
            let slot = match key.as_str() {
                "max_velocity" => &mut c.max_velocity,
                "kp_max" => &mut c.kp_max,
                "kd_max" => &mut c.kd_max,
                _ => &mut c.limit_margin,
            };
            *slot = num(val, key)?;
        } else if FIXED.contains(&key.as_str()) {
            let want = match key.as_str() {
                "control_rate_hz" => v1.control_rate_hz,
                "feedback_timeout" => v1.feedback_timeout,
                _ => v1.temp_hold_c,
            };
            same(key.clone(), val, want.into())?;
        } else {
            return Err(format!("{key}: unknown field"));
        }
    }
    check_config(&c, &v1, is_virtual(interface))?;
    Ok(c)
}

fn check_config(c: &ArmConfig, v1: &ArmConfig, virt: bool) -> Result<(), String> {
    let bad = |field: String, why: String| Err(format!("{field}: {why}"));
    let finite = |field: String, x: f64| {
        if x.is_finite() {
            Ok(())
        } else {
            bad(field, format!("{x} is not finite"))
        }
    };
    let at_most = |field: String, x: f64, cap: f64| {
        if x <= cap {
            Ok(())
        } else {
            bad(field, format!("{x} exceeds the maker_arm_v1 value {cap}"))
        }
    };
    for (i, (j, b)) in c.joints.iter().zip(&v1.joints).enumerate() {
        for (name, x) in [
            ("kp", j.kp),
            ("kd", j.kd),
            ("tau_max", j.tau_max),
            ("q_lo", j.q_lo),
            ("q_hi", j.q_hi),
            ("direction", j.direction),
            ("offset", j.offset),
        ] {
            finite(format!("joints[{i}].{name}"), x)?;
        }
        at_most(format!("joints[{i}].kp"), j.kp, b.kp)?;
        at_most(format!("joints[{i}].kd"), j.kd, b.kd)?;
        at_most(format!("joints[{i}].tau_max"), j.tau_max, b.tau_max)?;
        for (name, x) in [("kp", j.kp), ("kd", j.kd), ("tau_max", j.tau_max)] {
            if x < 0.0 {
                return bad(format!("joints[{i}].{name}"), format!("{x} is negative"));
            }
        }
        if j.direction != 1.0 && j.direction != -1.0 {
            return bad(
                format!("joints[{i}].direction"),
                format!("{} is not -1.0 or 1.0", j.direction),
            );
        }
        if j.offset.abs() > std::f64::consts::PI {
            return bad(
                format!("joints[{i}].offset"),
                format!("|{}| exceeds pi", j.offset),
            );
        }
        if virt {
            if j.q_lo >= j.q_hi {
                return bad(
                    format!("joints[{i}].q_lo"),
                    format!("range [{}, {}] is empty", j.q_lo, j.q_hi),
                );
            }
        } else {
            // Both ends go to the motor frame; each must land inside the v1 range.
            for (name, x) in [("q_lo", j.q_lo), ("q_hi", j.q_hi)] {
                let m = j.to_motor(x);
                if m < b.q_lo || m > b.q_hi {
                    return bad(
                        format!("joints[{i}].{name}"),
                        format!("{x} is {m} in the motor frame, outside the maker_arm_v1 range [{}, {}]", b.q_lo, b.q_hi),
                    );
                }
            }
            if j.q_lo >= j.q_hi {
                return bad(
                    format!("joints[{i}].q_lo"),
                    format!("range [{}, {}] is empty", j.q_lo, j.q_hi),
                );
            }
        }
    }
    finite("max_velocity".into(), c.max_velocity)?;
    at_most("max_velocity".into(), c.max_velocity, v1.max_velocity)?;
    if c.max_velocity <= 0.0 {
        return bad("max_velocity".into(), "must be positive".into());
    }
    for (name, x, cap) in [
        ("kp_max", c.kp_max, v1.kp_max),
        ("kd_max", c.kd_max, v1.kd_max),
        ("limit_margin", c.limit_margin, v1.limit_margin),
    ] {
        finite(name.into(), x)?;
        if name == "limit_margin" {
            if x < cap {
                return bad(
                    name.into(),
                    format!("{x} is below the maker_arm_v1 value {cap}"),
                );
            }
        } else {
            if x < 0.0 {
                return bad(name.into(), format!("{x} is negative"));
            }
            at_most(name.into(), x, cap)?;
        }
    }
    // The clamp narrows each joint to [q_lo + m, q_hi - m]; an empty band would panic it.
    for (i, j) in c.joints.iter().enumerate() {
        if j.q_hi - j.q_lo <= 2.0 * c.limit_margin {
            return bad(
                "limit_margin".into(),
                format!(
                    "{} leaves no range on joints[{i}] ([{}, {}])",
                    c.limit_margin, j.q_lo, j.q_hi
                ),
            );
        }
    }
    Ok(())
}

/// The model and its `provenance.source`.
pub fn dynamics_from_json(v: &Value) -> Result<(Dynamics, String), String> {
    const KEYS: [&str; 7] = [
        "mount_rpy",
        "links",
        "gravity",
        "bodies",
        "friction",
        "armature",
        "provenance",
    ];
    let o = object(v, "dynamics")?;
    if let Some(k) = o.keys().find(|k| !KEYS.contains(&k.as_str())) {
        return Err(format!("{k}: unknown field"));
    }
    let get = |k: &str| o.get(k).ok_or_else(|| format!("{k}: missing"));
    let opt = |k: &str| o.get(k).filter(|x| !x.is_null());
    let source = get("provenance")?
        .get("source")
        .and_then(Value::as_str)
        .filter(|s| ["urdf", "ident"].contains(s))
        .ok_or("provenance.source: must be \"urdf\" or \"ident\"")?
        .to_string();
    let mount_rpy = fixed(get("mount_rpy")?, "mount_rpy")?;
    let gravity = fixed(get("gravity")?, "gravity")?;
    let links = get("links")?
        .as_array()
        .ok_or("links: must be a list")?
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let f = format!("links[{i}]");
            let [xyz, rpy, axis] = l.as_array().map(Vec::as_slice).unwrap_or(&[]) else {
                return Err(format!("{f}: needs [xyz, rpy, axis]"));
            };
            let axis = if axis.is_null() {
                None
            } else {
                Some(fixed(axis, &format!("{f}[2]"))?)
            };
            Ok(Link::new(
                fixed(xyz, &format!("{f}[0]"))?,
                fixed(rpy, &format!("{f}[1]"))?,
                axis,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let bodies = six(
        get("bodies")?
            .as_array()
            .ok_or("bodies: must be a list")?
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let f = format!("bodies[{i}]");
                let [mass, com, inertia] = b.as_array().map(Vec::as_slice).unwrap_or(&[]) else {
                    return Err(format!("{f}: needs [mass, com, inertia]"));
                };
                let rows = inertia.as_array().map(Vec::as_slice).unwrap_or(&[]);
                if rows.len() != 3 {
                    return Err(format!("{f}[2]: expected 3 rows, got {}", rows.len()));
                }
                let mut inertia = [[0.0; 3]; 3];
                for (r, row) in rows.iter().enumerate() {
                    inertia[r] = fixed(row, &format!("{f}[2][{r}]"))?;
                }
                Ok(Body {
                    mass: num(mass, &format!("{f}[0]"))?,
                    com: fixed(com, &format!("{f}[1]"))?,
                    inertia,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        "bodies",
    )?;
    let friction = match opt("friction") {
        None => [Friction::default(); ARM_DOF],
        Some(f) => six(
            f.as_array()
                .ok_or("friction: must be a list or null")?
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let f = format!("friction[{i}]");
                    match nums(t, &f)?[..] {
                        [c, v, e] => Ok(Friction::new(c, v, e)),
                        [coulomb, viscous, eps, static_, stribeck_v] => Ok(Friction {
                            coulomb,
                            viscous,
                            eps,
                            static_,
                            stribeck_v,
                        }),
                        _ => Err(format!("{f}: a friction tuple has 3 or 5 entries")),
                    }
                })
                .collect::<Result<Vec<_>, String>>()?,
            "friction",
        )?,
    };
    let armature = match opt("armature") {
        None => [0.0; ARM_DOF],
        Some(a) => fixed(a, "armature")?,
    };
    let d = Dynamics::new(mount_rpy, &links, bodies, friction, gravity, armature)
        .map_err(|e| e.to_string())?;
    Ok((d, source))
}

fn six<T>(items: Vec<T>, what: &str) -> Result<[T; ARM_DOF], String> {
    let n = items.len();
    items
        .try_into()
        .map_err(|_| format!("{what}: expected {ARM_DOF} entries, got {n}"))
}

/// The keys of the binding's `params_from_dict`. `contact: false` is rung F, which a loop
/// refuses.
pub fn params_from_json(v: &Value) -> Result<TrackingParams, String> {
    let mut p = TrackingParams::default();
    let flag = |x: &Value, k: &str| {
        x.as_bool()
            .ok_or_else(|| format!("{k}: must be true or false"))
    };
    for (k, x) in object(v, "params")? {
        match k.as_str() {
            "f_r" => p.f_r = num(x, k)?,
            "k_o" => p.k_o = num(x, k)?,
            "f_y" => p.f_y = num(x, k)?,
            "ka_ratio" => p.ka_ratio = num(x, k)?,
            "tau_th" => p.tau_th = fixed(x, k)?,
            "e_max" => p.e_max = fixed(x, k)?,
            "gate_fall" => p.gate_fall = num(x, k)?,
            "gate_rise" => p.gate_rise = num(x, k)?,
            "gate_quiet" => p.gate_quiet = num(x, k)?,
            "contact" => p.contact = flag(x, k)?,
            "dob" => p.dob = flag(x, k)?,
            "dob_f" => p.dob_f = num(x, k)?,
            other => return Err(format!("unknown tracking parameter {other}")),
        }
    }
    if !p.contact {
        return Err("contact: false (rung F) has no error clamp and is for the bench only".into());
    }
    Ok(p)
}

/// The six fields the lab's export writes, and no other.
pub fn residual_from_json(v: &Value) -> Result<ResidualModel, String> {
    const FIELDS: [&str; 6] = ["history", "layers", "x_mean", "x_std", "y_scale", "out_max"];
    let o = object(v, "residual")?;
    if let Some(k) = o.keys().find(|k| !FIELDS.contains(&k.as_str())) {
        return Err(format!("unknown residual field {k}"));
    }
    let get = |k: &str| {
        o.get(k)
            .ok_or_else(|| format!("the residual is missing {k}"))
    };
    let layers = get("layers")?
        .as_array()
        .ok_or("residual layers: must be [[w, b], ...]")?
        .iter()
        .enumerate()
        .map(|(k, item)| {
            let f = format!("residual layer {k}");
            let [w, b] = item.as_array().map(Vec::as_slice).unwrap_or(&[]) else {
                return Err(format!("{f}: needs [w, b]"));
            };
            let rows = w
                .as_array()
                .ok_or(format!("{f}: w must be a list of rows"))?;
            let w: Vec<Vec<f64>> = rows
                .iter()
                .enumerate()
                .map(|(i, r)| nums(r, &format!("{f}: row {i} of w")))
                .collect::<Result<_, _>>()?;
            let b = nums(b, &format!("{f}: b"))?;
            if w.len() != b.len() {
                return Err(format!(
                    "{f}: w has {} rows, b has {} values",
                    w.len(),
                    b.len()
                ));
            }
            let n_in = w.first().map_or(0, Vec::len);
            if let Some(i) = w.iter().position(|r| r.len() != n_in) {
                return Err(format!(
                    "{f}: row {i} of w has {} values, row 0 has {n_in}",
                    w[i].len()
                ));
            }
            Ok(Layer {
                w: w.into_iter().flatten().collect(),
                b,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let history = get("history")?
        .as_u64()
        .ok_or("residual history: must be a whole number")? as usize;
    ResidualModel::new(
        Mlp::new(layers).map_err(|e| e.to_string())?,
        history,
        nums(get("x_mean")?, "residual x_mean")?,
        nums(get("x_std")?, "residual x_std")?,
        fixed(get("y_scale")?, "residual y_scale")?,
        fixed(get("out_max")?, "residual out_max")?,
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn v1_json() -> Value {
        let c = ArmConfig::maker_arm_v1();
        let joints: Vec<Value> = c
            .joints
            .iter()
            .map(|j| {
                json!({"motor_id": j.motor_id, "name": j.name, "model": j.model.name(),
                       "kp": j.kp, "kd": j.kd, "tau_max": j.tau_max, "q_lo": j.q_lo,
                       "q_hi": j.q_hi, "direction": j.direction, "offset": j.offset})
            })
            .collect();
        json!({"joints": joints, "control_rate_hz": c.control_rate_hz,
               "max_velocity": c.max_velocity, "feedback_timeout": c.feedback_timeout,
               "limit_margin": c.limit_margin, "kp_max": c.kp_max, "kd_max": c.kd_max,
               "temp_hold_c": c.temp_hold_c})
    }

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dynamics_test.json")
    }

    fn scratch(name: &str, body: &str) -> PathBuf {
        // One directory per test thread: tests share file names and run in parallel.
        let t = std::thread::current();
        let dir = std::env::temp_dir().join(format!(
            "maker-arm-files-{}-{}",
            std::process::id(),
            t.name().unwrap_or("t").replace("::", "-")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn options() -> ServeOptions {
        ServeOptions {
            interface: "sim".into(),
            profile: None,
            dynamics: fixture(),
            params: None,
            residual: None,
            gripper_map: None,
            rt: false,
            socket: None,
            persist: false,
        }
    }

    fn half_kp_options(params: Option<&str>) -> ServeOptions {
        let mut v = v1_json();
        for j in v["joints"].as_array_mut().unwrap() {
            let kp = j["kp"].as_f64().unwrap();
            j["kp"] = json!(kp * 0.5);
        }
        let mut o = options();
        o.profile = Some(scratch("half_kp.json", &v.to_string()));
        o.params = params.map(|t| scratch("half_kp_params.json", t));
        o
    }

    #[test]
    fn a_half_kp_profile_doubles_the_default_e_max() {
        let l = load(&half_kp_options(None)).unwrap();
        for j in 0..ARM_DOF {
            assert!((l.params.e_max[j] - 0.06).abs() < 1e-12);
        }
    }

    #[test]
    fn a_params_e_max_within_the_cap_is_kept() {
        let l = load(&half_kp_options(Some(
            r#"{"e_max": [0.05,0.05,0.05,0.05,0.05,0.05]}"#,
        )))
        .unwrap();
        assert_eq!(l.params.e_max, [0.05; ARM_DOF]);
    }

    #[test]
    fn a_params_e_max_past_the_cap_is_refused() {
        let e = load(&half_kp_options(Some(
            r#"{"e_max": [0.05,0.05,0.07,0.05,0.05,0.05]}"#,
        )))
        .err()
        .unwrap();
        assert!(e.contains("e_max[2]") && e.contains("cap 0.06"), "{e}");
    }

    #[test]
    fn the_v1_kp_keeps_the_default_e_max() {
        assert_eq!(load(&options()).unwrap().params.e_max, [0.03; ARM_DOF]);
        let mut o = options();
        o.profile = Some(scratch("v1.json", &v1_json().to_string()));
        assert_eq!(load(&o).unwrap().params.e_max, [0.03; ARM_DOF]);
    }

    #[test]
    fn the_v1_profile_json_loads_as_v1() {
        let c = profile_from_json(&v1_json(), "can0").unwrap();
        let v1 = ArmConfig::maker_arm_v1();
        for (a, b) in c.joints.iter().zip(&v1.joints) {
            assert_eq!(
                (
                    a.kp,
                    a.kd,
                    a.tau_max,
                    a.q_lo,
                    a.q_hi,
                    a.direction,
                    a.offset,
                    a.motor_id,
                    a.name
                ),
                (
                    b.kp,
                    b.kd,
                    b.tau_max,
                    b.q_lo,
                    b.q_hi,
                    b.direction,
                    b.offset,
                    b.motor_id,
                    b.name
                )
            );
        }
        assert_eq!(
            (c.max_velocity, c.kp_max, c.kd_max, c.limit_margin),
            (v1.max_velocity, v1.kp_max, v1.kd_max, v1.limit_margin)
        );
    }

    #[test]
    fn a_raised_cap_is_refused_and_named() {
        let mut v = v1_json();
        v["joints"][1]["tau_max"] = json!(18.0);
        let e = profile_from_json(&v, "sim").unwrap_err();
        assert!(e.contains("joints[1].tau_max"), "{e}");
    }

    #[test]
    fn an_unknown_key_is_refused() {
        let mut v = v1_json();
        v["max_velocty"] = json!(1.0);
        assert!(profile_from_json(&v, "sim")
            .unwrap_err()
            .contains("max_velocty"));
    }

    #[test]
    fn urdf_limits_pass_on_sim_and_vcan0_but_not_on_can0() {
        let mut v = v1_json();
        v["joints"][2]["q_lo"] = json!(0.0);
        v["joints"][2]["q_hi"] = json!(3.1);
        assert!(profile_from_json(&v, "sim").is_ok());
        assert!(profile_from_json(&v, "vcan0").is_ok());
        let e = profile_from_json(&v, "can0").unwrap_err();
        assert!(e.contains("joints[2].q_lo"), "{e}");
    }

    #[test]
    fn a_flipped_joint_is_checked_in_the_motor_frame() {
        let mut v = v1_json();
        v["joints"][0]["direction"] = json!(-1.0);
        v["joints"][0]["offset"] = json!(0.5);
        v["joints"][0]["q_lo"] = json!(-4.318);
        v["joints"][0]["q_hi"] = json!(1.168);
        assert!(profile_from_json(&v, "can0").is_ok());
        v["joints"][0]["q_hi"] = json!(1.3);
        let e = profile_from_json(&v, "can0").unwrap_err();
        assert!(e.contains("joints[0].q_hi"), "{e}");
    }

    #[test]
    fn a_deadband_with_the_urdf_model_is_refused() {
        let mut o = options();
        o.params = Some(scratch(
            "deadband.json",
            r#"{"tau_th": [0.5,0.5,0.5,0.5,0.5,0.5]}"#,
        ));
        let e = load(&o).err().unwrap();
        assert!(
            e.contains("a deadband ships only with the identified model"),
            "{e}"
        );
        let ident = std::fs::read_to_string(fixture())
            .unwrap()
            .replace("\"urdf\"", "\"ident\"");
        o.dynamics = scratch("ident.json", &ident);
        assert!(load(&o).is_ok());
    }

    #[test]
    fn rung_f_params_are_refused() {
        let e = params_from_json(&json!({"contact": false})).unwrap_err();
        assert!(e.contains("contact"), "{e}");
    }

    #[test]
    fn the_provisional_grip_map_spans_the_gripper_range() {
        let g = GripMap::provisional(&ArmConfig::maker_arm_v1());
        assert!((g.to_joint_m(-2.092)).abs() < 1e-12);
        assert!((g.to_joint_m(-0.039) - 0.0524125).abs() < 1e-12);
        assert!((g.to_angle(g.to_joint_m(-1.0)) + 1.0).abs() < 1e-12);
        assert!(GripMap::new(0.0, 1.0).is_err());
    }

    #[test]
    fn the_gripper_map_text_is_parsed_and_checked() {
        let mut o = options();
        o.gripper_map = Some("0.05,-0.1".into());
        let g = load(&o).unwrap().grip;
        assert_eq!((g.a, g.b), (0.05, -0.1));
        for bad in ["0,1", "1", "1,2,3", "x,1"] {
            o.gripper_map = Some(bad.into());
            assert!(load(&o).err().unwrap().contains("--gripper_map"), "{bad}");
        }
    }

    #[test]
    fn the_digest_changes_with_any_file_byte_and_not_with_rt() {
        let mut o = options();
        let base = load(&o).unwrap().digest;
        o.rt = true;
        o.persist = true;
        assert_eq!(load(&o).unwrap().digest, base);
        o.interface = "vcan0".into();
        let vcan = load(&o).unwrap().digest;
        assert_ne!(vcan, base);
        o.gripper_map = Some("0.05,-0.1".into());
        assert_ne!(load(&o).unwrap().digest, vcan);
        o.gripper_map = None;
        o.params = Some(scratch("p1.json", r#"{"f_r": 10.0}"#));
        let p1 = load(&o).unwrap().digest;
        o.params = Some(scratch("p2.json", r#"{"f_r": 10.5}"#));
        assert_ne!(load(&o).unwrap().digest, p1);
        o.profile = Some(scratch("prof.json", &v1_json().to_string()));
        assert_ne!(load(&o).unwrap().digest, p1);
    }

    #[test]
    fn dynamics_rejects_unknown_keys_and_a_bad_source() {
        let text = std::fs::read_to_string(fixture()).unwrap();
        let mut v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(dynamics_from_json(&v).unwrap().1, "urdf");
        v["stiction"] = json!(1);
        assert!(dynamics_from_json(&v).err().unwrap().contains("stiction"));
        v.as_object_mut().unwrap().remove("stiction");
        v["provenance"]["source"] = json!("guess");
        assert!(dynamics_from_json(&v)
            .err()
            .unwrap()
            .contains("provenance.source"));
    }

    #[test]
    fn residual_refuses_an_unknown_field() {
        let e = residual_from_json(&json!({"layer": []})).err().unwrap();
        assert!(e.contains("layer"), "{e}");
    }
}
