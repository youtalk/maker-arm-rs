//! The learned residual torque (MS5 revision section 8.2): a small MLP, trained by the lab on
//! the arm's own logs, gives the per-joint torque the identified model misses. `Tracking`
//! adds it to the observer's model next to friction and to the feed-forward, gated like
//! friction.
//!
//! The format is the lab's export: torch's `weight.tolist()` per layer (out rows of `in`
//! values), tanh between layers and none after the last. The input is the last `history`
//! ticks of measured q (6) and dq (6), oldest first, then the six arm temperatures, so
//! `12 * history + 6` values, normalized as `(x - x_mean) / x_std` before the MLP. The output
//! times `y_scale`, clamped to `+-out_max`, is `tau_res`. There is no output mean: the lab
//! folds it into the last bias.
//!
//! It runs inside the 200 Hz loop, so a forward allocates nothing: every buffer is sized at
//! construction.

use crate::kinematics::{Joints, ARM_DOF};
use crate::tracking::TrackingError;

/// One dense layer, `y = w x + b`. `w` is row-major, `b.len()` rows of `inputs()` values.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub w: Vec<f64>,
    pub b: Vec<f64>,
}

impl Layer {
    pub fn inputs(&self) -> usize {
        self.w.len() / self.b.len().max(1)
    }

    pub fn outputs(&self) -> usize {
        self.b.len()
    }
}

fn bad(detail: String) -> TrackingError {
    TrackingError::Param {
        name: "residual",
        detail,
    }
}

/// A tanh MLP with no activation after the last layer.
#[derive(Clone, Debug)]
pub struct Mlp {
    layers: Vec<Layer>,
    /// Two ping-pong activations, each as wide as the widest layer.
    a: Vec<f64>,
    z: Vec<f64>,
}

impl Mlp {
    /// Refuses an empty net, a layer whose `w` is not `b.len()` whole rows, a layer whose
    /// input is not the previous layer's output, and a non-finite weight, naming the layer.
    pub fn new(layers: Vec<Layer>) -> Result<Mlp, TrackingError> {
        if layers.is_empty() {
            return Err(bad("the MLP has no layer".into()));
        }
        let mut width = 0;
        for (k, l) in layers.iter().enumerate() {
            if l.b.is_empty() || l.w.is_empty() || l.w.len() % l.b.len() != 0 {
                return Err(bad(format!(
                    "layer {k}: w has {} values, not a whole number of {} rows",
                    l.w.len(),
                    l.b.len()
                )));
            }
            if k > 0 && l.inputs() != layers[k - 1].outputs() {
                return Err(bad(format!(
                    "layer {k} takes {} inputs, layer {} gives {}",
                    l.inputs(),
                    k - 1,
                    layers[k - 1].outputs()
                )));
            }
            if l.w.iter().chain(&l.b).any(|v| !v.is_finite()) {
                return Err(bad(format!("layer {k} has a non-finite weight")));
            }
            width = width.max(l.inputs()).max(l.outputs());
        }
        Ok(Mlp {
            layers,
            a: vec![0.0; width],
            z: vec![0.0; width],
        })
    }

    pub fn inputs(&self) -> usize {
        self.layers[0].inputs()
    }

    pub fn outputs(&self) -> usize {
        self.layers[self.layers.len() - 1].outputs()
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// `out = MLP(x)` for an already normalized `x`. Allocates nothing.
    ///
    /// # Panics
    /// If `x` is not `inputs()` long or `out` not `outputs()` long.
    pub fn forward(&mut self, x: &[f64], out: &mut [f64]) {
        assert_eq!(x.len(), self.inputs(), "MLP input width");
        assert_eq!(out.len(), self.outputs(), "MLP output width");
        let last = self.layers.len() - 1;
        self.a[..x.len()].copy_from_slice(x);
        for (k, l) in self.layers.iter().enumerate() {
            let n_in = l.inputs();
            let input = &self.a[..n_in];
            for (i, (row, b)) in l.w.chunks_exact(n_in).zip(&l.b).enumerate() {
                let mut s = *b;
                for (w, v) in row.iter().zip(input) {
                    s += w * v;
                }
                self.z[i] = if k < last { s.tanh() } else { s };
            }
            std::mem::swap(&mut self.a, &mut self.z);
        }
        out.copy_from_slice(&self.a[..out.len()]);
    }
}

/// The trained residual and its runtime state: the q/dq history ring and the input buffer.
#[derive(Clone, Debug)]
pub struct ResidualModel {
    mlp: Mlp,
    history: usize,
    x_mean: Vec<f64>,
    x_std: Vec<f64>,
    y_scale: Joints,
    out_max: Joints,
    /// `history` blocks of [q(6), dq(6)]; `head` is the next block to write.
    ring: Vec<f64>,
    head: usize,
    filled: usize,
    x: Vec<f64>,
}

/// One tick's block in the history: q then dq.
const BLOCK: usize = 2 * ARM_DOF;

/// A normalized input past this many training standard deviations is a skip: a glitched
/// reading (a 6553.5 degC temperature, a q far outside the training range) must not drive the
/// net into a saturated output that `out_max` would then pass as torque.
pub const X_BOUND: f64 = 10.0; // INITIAL: session 8 checks it against the real data

impl ResidualModel {
    pub fn new(
        mlp: Mlp,
        history: usize,
        x_mean: Vec<f64>,
        x_std: Vec<f64>,
        y_scale: Joints,
        out_max: Joints,
    ) -> Result<ResidualModel, TrackingError> {
        if history == 0 {
            return Err(bad("history must be at least 1 tick".into()));
        }
        let n = BLOCK * history + ARM_DOF;
        if mlp.inputs() != n {
            return Err(bad(format!(
                "layer 0 takes {} inputs, a history of {history} needs 12 * {history} + 6 = {n}",
                mlp.inputs()
            )));
        }
        if mlp.outputs() != ARM_DOF {
            return Err(bad(format!(
                "layer {} gives {} outputs, the residual needs {ARM_DOF}",
                mlp.layers.len() - 1,
                mlp.outputs()
            )));
        }
        if x_mean.len() != n || x_std.len() != n {
            return Err(bad(format!(
                "x_mean and x_std need {n} values, they have {} and {}",
                x_mean.len(),
                x_std.len()
            )));
        }
        if let Some(v) = x_mean.iter().find(|v| !v.is_finite()) {
            return Err(bad(format!("x_mean has {v}, every entry must be finite")));
        }
        if let Some(v) = x_std.iter().find(|v| !(v.is_finite() && **v > 0.0)) {
            return Err(bad(format!(
                "x_std has {v}, every entry must be positive and finite"
            )));
        }
        if let Some(v) = y_scale.iter().find(|v| !v.is_finite()) {
            return Err(bad(format!("y_scale has {v}, every entry must be finite")));
        }
        if let Some(v) = out_max.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
            return Err(bad(format!(
                "out_max has {v}, every entry must be finite and non-negative"
            )));
        }
        Ok(ResidualModel {
            mlp,
            history,
            x_mean,
            x_std,
            y_scale,
            out_max,
            ring: vec![0.0; BLOCK * history],
            head: 0,
            filled: 0,
            x: vec![0.0; n],
        })
    }

    pub fn history(&self) -> usize {
        self.history
    }

    /// True once `history` ticks are recorded since the last reset.
    pub fn is_full(&self) -> bool {
        self.filled == self.history
    }

    /// Forgets the history: until `history` ticks are recorded again, `evaluate` gives None.
    pub fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
    }

    /// Adds one tick's measured q and dq. A non-finite reading restarts the history instead,
    /// so the ring never holds one.
    pub fn record(&mut self, q: &Joints, dq: &Joints) {
        if q.iter().chain(dq).any(|v| !v.is_finite()) {
            self.reset();
            return;
        }
        let block = &mut self.ring[self.head * BLOCK..(self.head + 1) * BLOCK];
        block[..ARM_DOF].copy_from_slice(q);
        block[ARM_DOF..].copy_from_slice(dq);
        self.head = (self.head + 1) % self.history;
        self.filled = (self.filled + 1).min(self.history);
    }

    /// True if this tick's reading is finite and, normalized as the newest tick, within
    /// `X_BOUND`: the reading `evaluate` would accept. For a hold, which evaluates nothing.
    pub fn reading_ok(&self, q: &Joints, dq: &Joints, temps: &Joints) -> bool {
        let base = (self.history - 1) * BLOCK;
        let t0 = self.history * BLOCK;
        let ok = |v: f64, i: usize| {
            let x = (v - self.x_mean[i]) / self.x_std[i];
            x.is_finite() && x.abs() <= X_BOUND
        };
        (0..ARM_DOF)
            .all(|j| ok(q[j], base + j) && ok(dq[j], base + ARM_DOF + j) && ok(temps[j], t0 + j))
    }

    /// `tau_res` from the recorded history and `temps`, or None (a skip) until `history` ticks
    /// exist, on a non-finite input or output, and on a normalized input past `X_BOUND`.
    /// Allocates nothing.
    pub fn evaluate(&mut self, temps: &Joints) -> Option<Joints> {
        if self.filled < self.history || temps.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Oldest first: the block at `head` is the oldest once the ring is full.
        let h = self.history;
        for k in 0..h {
            let src = ((self.head + k) % h) * BLOCK;
            self.x[k * BLOCK..(k + 1) * BLOCK].copy_from_slice(&self.ring[src..src + BLOCK]);
        }
        self.x[h * BLOCK..].copy_from_slice(temps);
        for ((x, m), s) in self.x.iter_mut().zip(&self.x_mean).zip(&self.x_std) {
            *x = (*x - m) / s;
            if x.is_nan() || x.abs() > X_BOUND {
                return None;
            }
        }
        let mut y = [0.0; ARM_DOF];
        self.mlp.forward(&self.x, &mut y);
        let mut out = [0.0; ARM_DOF];
        for j in 0..ARM_DOF {
            let v = y[j] * self.y_scale[j];
            if !v.is_finite() {
                return None;
            }
            out[j] = v.clamp(-self.out_max[j], self.out_max[j]);
        }
        Some(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn layer(w: &[&[f64]], b: &[f64]) -> Layer {
        Layer {
            w: w.iter().flat_map(|r| r.iter().copied()).collect(),
            b: b.to_vec(),
        }
    }

    #[test]
    fn a_two_by_two_mlp_matches_its_hand_computed_value() {
        let mut mlp = Mlp::new(vec![
            layer(&[&[0.5, -1.0], &[2.0, 0.25]], &[0.1, -0.2]),
            layer(&[&[1.0, -2.0], &[0.5, 3.0]], &[0.3, 0.0]),
        ])
        .unwrap();
        let x = [0.4, -0.6];
        let h0 = (0.1 + 0.5 * 0.4 + 1.0 * 0.6_f64).tanh(); // tanh(0.9)
        let h1 = (-0.2 + 2.0 * 0.4 - 0.25 * 0.6_f64).tanh(); // tanh(0.45)
        let want = [0.3 + h0 - 2.0 * h1, 0.5 * h0 + 3.0 * h1];
        let mut out = [0.0; 2];
        mlp.forward(&x, &mut out);
        for i in 0..2 {
            assert!((out[i] - want[i]).abs() < 1e-15, "{out:?} vs {want:?}");
        }
        // The buffers carry nothing from one call to the next.
        let mut again = [0.0; 2];
        mlp.forward(&x, &mut again);
        assert_eq!(again, out);
    }

    /// A residual with a history of `h` whose output is `bias` whatever the input.
    pub(crate) fn constant(h: usize, bias: Joints, out_max: f64) -> ResidualModel {
        let n = 12 * h + 6;
        let mlp = Mlp::new(vec![Layer {
            w: vec![0.0; 6 * n],
            b: bias.to_vec(),
        }])
        .unwrap();
        ResidualModel::new(mlp, h, vec![0.0; n], vec![1.0; n], [1.0; 6], [out_max; 6]).unwrap()
    }

    #[test]
    fn the_input_is_the_history_oldest_first_then_the_temperatures_normalized() {
        // One output reads input i through a one-hot weight; a history of 2.
        let n = 12 * 2 + 6;
        let probe = |i: usize| {
            let mut w = vec![0.0; 6 * n];
            w[i] = 1.0;
            let mlp = Mlp::new(vec![Layer { w, b: vec![0.0; 6] }]).unwrap();
            let mean: Vec<f64> = (0..n).map(|k| k as f64 * 0.01).collect();
            let std: Vec<f64> = (0..n).map(|k| 10.0 + k as f64 * 0.1).collect();
            ResidualModel::new(mlp, 2, mean, std, [2.0; 6], [100.0; 6]).unwrap()
        };
        let tick = |k: f64| -> (Joints, Joints) {
            (
                std::array::from_fn(|j| k + j as f64),
                std::array::from_fn(|j| 10.0 * k + j as f64),
            )
        };
        let temps: Joints = std::array::from_fn(|j| 30.0 + j as f64);
        for (i, raw) in [
            (0, 2.0),      // q[0] of the oldest kept tick (k = 2)
            (7, 21.0),     // dq[1] of the oldest
            (12 + 3, 6.0), // q[3] of the newest (k = 3)
            (24 + 5, 35.0),
        ] {
            let mut r = probe(i);
            for k in 1..=3 {
                let (q, dq) = tick(k as f64);
                r.record(&q, &dq);
            }
            let y = r.evaluate(&temps).unwrap();
            let want = 2.0 * (raw - i as f64 * 0.01) / (10.0 + i as f64 * 0.1);
            assert!((y[0] - want).abs() < 1e-12, "input {i}: {} vs {want}", y[0]);
        }
    }

    #[test]
    fn a_nan_temperature_gives_a_zero_residual_and_a_skip() {
        let mut r = constant(3, [0.1; 6], 1.0);
        let q = [0.0; 6];
        for _ in 0..2 {
            r.record(&q, &q);
            assert_eq!(r.evaluate(&[3.0; 6]), None, "the history is not full");
        }
        r.record(&q, &q);
        assert_eq!(r.evaluate(&[3.0; 6]), Some([0.1; 6]));
        let mut temps = [3.0; 6];
        temps[4] = f64::NAN;
        assert_eq!(r.evaluate(&temps), None);
        // q and dq were fine, so the history is: the next good temperature evaluates.
        r.record(&q, &q);
        assert_eq!(r.evaluate(&[3.0; 6]), Some([0.1; 6]));
        // A NaN reading restarts the history.
        let mut bad = q;
        bad[0] = f64::NAN;
        r.record(&bad, &q);
        r.record(&q, &q);
        assert_eq!(r.evaluate(&[3.0; 6]), None);
    }

    #[test]
    fn the_output_is_clamped_to_out_max() {
        let mut r = constant(1, [0.5, -0.5, 0.2, -0.2, 3.0, 0.0], 0.3);
        r.record(&[0.0; 6], &[0.0; 6]);
        assert_eq!(
            r.evaluate(&[0.0; 6]),
            Some([0.3, -0.3, 0.2, -0.2, 0.3, 0.0])
        );
        // An overflowing output is a skip, not a clamped infinity.
        let mut huge = constant(1, [1e300; 6], 0.3);
        huge.y_scale = [1e300; 6];
        huge.record(&[0.0; 6], &[0.0; 6]);
        assert_eq!(huge.evaluate(&[0.0; 6]), None);
    }

    #[test]
    fn a_normalized_input_past_x_bound_is_a_skip() {
        let mut r = constant(1, [0.1; 6], 1.0);
        r.record(&[0.0; 6], &[0.0; 6]);
        assert_eq!(
            r.evaluate(&[X_BOUND; 6]),
            Some([0.1; 6]),
            "the bound itself is in range"
        );
        assert_eq!(
            r.evaluate(&[0.0, 0.0, -X_BOUND - 1e-9, 0.0, 0.0, 0.0]),
            None
        );
        r.record(&[0.0, 11.0, 0.0, 0.0, 0.0, 0.0], &[0.0; 6]);
        assert_eq!(r.evaluate(&[0.0; 6]), None, "a q past the bound");
        assert!(!r.reading_ok(&[0.0; 6], &[0.0; 6], &[6553.5; 6]));
        assert!(r.reading_ok(&[0.0; 6], &[0.0; 6], &[1.0; 6]));
    }

    #[test]
    fn shapes_and_values_are_validated_naming_the_layer() {
        let msg = |e: TrackingError| e.to_string();
        let e = Mlp::new(vec![
            layer(&[&[1.0, 2.0, 3.0]], &[0.0]),
            layer(&[&[1.0, 2.0]], &[0.0]),
        ])
        .unwrap_err();
        assert!(msg(e).contains("layer 1"));
        let e = Mlp::new(vec![Layer {
            w: vec![1.0; 5],
            b: vec![0.0; 2],
        }])
        .unwrap_err();
        assert!(msg(e).contains("layer 0"));
        assert!(Mlp::new(vec![]).is_err());
        assert!(Mlp::new(vec![layer(&[&[f64::NAN]], &[0.0])]).is_err());
        let net = |n_in: usize, n_out: usize| {
            Mlp::new(vec![Layer {
                w: vec![0.0; n_in * n_out],
                b: vec![0.0; n_out],
            }])
            .unwrap()
        };
        let ok = |mean: Vec<f64>, std: Vec<f64>, y: f64, m: f64| {
            ResidualModel::new(net(18, 6), 1, mean, std, [y; 6], [m; 6])
        };
        assert!(ok(vec![0.0; 18], vec![1.0; 18], 1.0, 1.0).is_ok());
        assert!(ok(vec![0.0; 18], vec![1.0; 18], 1.0, 0.0).is_ok());
        assert!(ok(vec![0.0; 17], vec![1.0; 18], 1.0, 1.0).is_err());
        let mut std = vec![1.0; 18];
        std[3] = 0.0;
        assert!(ok(vec![0.0; 18], std.clone(), 1.0, 1.0).is_err());
        std[3] = f64::INFINITY;
        assert!(ok(vec![0.0; 18], std, 1.0, 1.0).is_err());
        assert!(ok(vec![f64::NAN; 18], vec![1.0; 18], 1.0, 1.0).is_err());
        assert!(ok(vec![0.0; 18], vec![1.0; 18], f64::NAN, 1.0).is_err());
        assert!(ok(vec![0.0; 18], vec![1.0; 18], 1.0, -0.1).is_err());
        assert!(ok(vec![0.0; 18], vec![1.0; 18], 1.0, f64::INFINITY).is_err());
        let e = ResidualModel::new(
            net(30, 6),
            1,
            vec![0.0; 30],
            vec![1.0; 30],
            [1.0; 6],
            [1.0; 6],
        )
        .unwrap_err();
        assert!(msg(e).contains("layer 0"));
        let e = ResidualModel::new(
            net(18, 5),
            1,
            vec![0.0; 18],
            vec![1.0; 18],
            [1.0; 6],
            [1.0; 6],
        )
        .unwrap_err();
        assert!(msg(e).contains("layer 0"));
        assert!(
            ResidualModel::new(net(6, 6), 0, vec![0.0; 6], vec![1.0; 6], [1.0; 6], [1.0; 6])
                .is_err()
        );
    }

    /// One forward of the spec's size (12 * 10 + 6 -> 64 -> 64 -> 6). Run with
    /// `cargo test --release -p maker-arm --lib one_forward -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn one_forward_of_the_spec_net_takes_microseconds() {
        let sizes = [126, 64, 64, 6];
        let layers = sizes
            .windows(2)
            .map(|s| Layer {
                w: (0..s[0] * s[1])
                    .map(|k| ((k % 7) as f64 - 3.0) * 0.01)
                    .collect(),
                b: vec![0.01; s[1]],
            })
            .collect();
        let mut mlp = Mlp::new(layers).unwrap();
        let x: Vec<f64> = (0..126).map(|k| (k as f64 * 0.1).sin()).collect();
        let mut out = [0.0; 6];
        let n = 100_000;
        let t = std::time::Instant::now();
        for _ in 0..n {
            mlp.forward(std::hint::black_box(&x), &mut out);
        }
        let us = t.elapsed().as_secs_f64() / n as f64 * 1e6;
        println!("one forward: {us:.2} us ({out:?})");
    }
}
