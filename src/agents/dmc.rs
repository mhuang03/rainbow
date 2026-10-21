//! Deep Monte Carlo agent (DouZero-style): a network scores each (observation, action) pair with
//! its expected return, and the agent plays the highest-scoring legal combo.
//!
//! Networks are trained in Python (`python/train_dmc.py`) and exported as a small binary MLP
//! file, loaded here without any ML dependencies. The input is the observation encoding
//! followed by the action features (see [`encoding`]).
//!
//! File format, little-endian: the magic bytes `RBMLP1\0\0`, a `u32` layer count, then for each
//! layer a `u32` input size, a `u32` output size, `out * in` weights (row-major, one row per
//! output) and `out` biases, all `f32`. Hidden layers use ReLU; the last layer is linear.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::agents::Agent;
use crate::game::encoding::{self, ACTION_FEATURES, OBS_SIZE};
use crate::game::{Combo, Observation};

const MAGIC: &[u8; 8] = b"RBMLP1\0\0";

#[derive(Debug, Clone)]
struct Layer {
    inputs: usize,
    outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

/// Dot product with eight independent partial sums (inlined so callers compiled with wider
/// SIMD features vectorise it with them). Float addition isn't associative, so a plain
/// `iter().sum()` must add one element at a time; separate lanes let the compiler use SIMD.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let split = a.len().min(b.len()) / LANES * LANES;
    let mut acc = [0.0f32; LANES];
    for (x, y) in a[..split]
        .chunks_exact(LANES)
        .zip(b[..split].chunks_exact(LANES))
    {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = a[split..].iter().zip(&b[split..]).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

impl Layer {
    /// [`Self::forward`] for `n` inputs stored row after row in `x`, writing `n` output rows.
    /// Uses AVX2 and FMA when the CPU has them.
    fn forward_batch(&self, x: &[f32], n: usize, out: &mut Vec<f32>, relu: bool) {
        out.clear();
        out.resize(n * self.outputs, 0.0);
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the CPU supports the features the function is compiled for.
            unsafe { self.forward_batch_avx2(x, out, relu) };
            return;
        }
        self.forward_batch_into(x, out, relu);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn forward_batch_avx2(&self, x: &[f32], out: &mut [f32], relu: bool) {
        self.forward_batch_into(x, out, relu);
    }

    #[inline(always)]
    fn forward_batch_into(&self, x: &[f32], out: &mut [f32], relu: bool) {
        for (o, (row, &b)) in self
            .weights
            .chunks_exact(self.inputs)
            .zip(&self.bias)
            .enumerate()
        {
            for (a, input) in x.chunks_exact(self.inputs).enumerate() {
                let y = dot(row, input) + b;
                out[a * self.outputs + o] = if relu { y.max(0.0) } else { y };
            }
        }
    }

    /// `out = W x + b`, applying ReLU if `relu`.
    fn forward(&self, x: &[f32], out: &mut Vec<f32>, relu: bool) {
        out.clear();
        out.extend(
            self.weights
                .chunks_exact(self.inputs)
                .zip(&self.bias)
                .map(|(row, b)| {
                    let y = dot(row, x) + b;
                    if relu { y.max(0.0) } else { y }
                }),
        );
    }
}

/// Little-endian cursor over a byte slice.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.0.len() < n {
            return Err("truncated MLP file".to_owned());
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    fn u32(&mut self) -> Result<usize, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")) as usize)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let bytes = self.take(n.checked_mul(4).ok_or("layer too large")?)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect())
    }
}

/// A multilayer perceptron with ReLU hidden layers.
#[derive(Debug, Clone)]
pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    /// Parses the binary format described in the [module docs](self).
    ///
    /// # Errors
    /// If the data is truncated, has the wrong magic bytes, or its layers don't chain.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader(
            bytes
                .strip_prefix(MAGIC)
                .ok_or("not a Rainbow MLP file (bad magic bytes)")?,
        );
        let count = r.u32()?;
        let mut layers = Vec::with_capacity(count);
        for _ in 0..count {
            let inputs = r.u32()?;
            let outputs = r.u32()?;
            let weights = r.f32s(inputs * outputs)?;
            let bias = r.f32s(outputs)?;
            if let Some(prev) = layers.last().map(|l: &Layer| l.outputs)
                && prev != inputs
            {
                return Err(format!("layer sizes don't chain: {prev} -> {inputs}"));
            }
            layers.push(Layer {
                inputs,
                outputs,
                weights,
                bias,
            });
        }
        if !r.0.is_empty() {
            return Err("trailing bytes after the last layer".to_owned());
        }
        if layers.is_empty() {
            return Err("MLP has no layers".to_owned());
        }
        Ok(Self { layers })
    }

    /// Loads a network file, sharing it between all callers that load the same path.
    ///
    /// # Errors
    /// If the file cannot be read or parsed.
    pub fn load_cached(path: &str) -> Result<Arc<Self>, String> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<Mlp>>>> = OnceLock::new();
        let mut cache = CACHE
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| "model cache poisoned")?;
        if let Some(mlp) = cache.get(path) {
            return Ok(Arc::clone(mlp));
        }
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let mlp = Arc::new(Self::from_bytes(&bytes)?);
        cache.insert(path.to_owned(), Arc::clone(&mlp));
        Ok(mlp)
    }

    pub fn inputs(&self) -> usize {
        self.layers[0].inputs
    }

    /// Runs the network on one input vector.
    ///
    /// # Panics
    /// If `x.len() != self.inputs()`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.inputs(), "wrong input size");
        let (mut h, mut next) = (x.to_vec(), Vec::new());
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(&h, &mut next, i < last);
            std::mem::swap(&mut h, &mut next);
        }
        h
    }

    pub fn outputs(&self) -> usize {
        self.layers.last().expect("at least one layer").outputs
    }
}

/// A network with the given layer sizes and deterministic pseudo-random weights, for tests.
#[cfg(test)]
pub(crate) fn test_mlp(layers: &[(usize, usize)], seed: u64) -> Mlp {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
    };
    Mlp {
        layers: layers
            .iter()
            .map(|&(inputs, outputs)| Layer {
                inputs,
                outputs,
                weights: (0..inputs * outputs).map(|_| next()).collect(),
                bias: (0..outputs).map(|_| next()).collect(),
            })
            .collect(),
    }
}

/// Plays the legal combo with the highest predicted return.
#[derive(Debug, Clone)]
pub struct DmcAgent {
    net: Arc<Mlp>,
}

impl DmcAgent {
    /// # Errors
    /// If the network doesn't take an observation plus action features and output one value.
    pub fn new(net: Arc<Mlp>) -> Result<Self, String> {
        if net.inputs() != OBS_SIZE + ACTION_FEATURES || net.outputs() != 1 {
            return Err(format!(
                "DMC network must map {} inputs to 1 output, got {} -> {}",
                OBS_SIZE + ACTION_FEATURES,
                net.inputs(),
                net.outputs()
            ));
        }
        Ok(Self { net })
    }

    /// Predicted return of each legal combo, in the order of `legal`.
    pub fn evaluate(&self, obs: &Observation, legal: &[Combo]) -> Vec<f32> {
        let first = &self.net.layers[0];
        let observation = obs.encoded();
        // The first layer splits into an observation part, shared by every action, and a small
        // action part.
        let shared: Vec<f32> = first
            .weights
            .chunks_exact(first.inputs)
            .zip(&first.bias)
            .map(|(row, b)| dot(&row[..OBS_SIZE], &observation) + b)
            .collect();

        // Evaluate every action together, layer by layer, so each weight row is loaded once
        // per evaluation rather than once per action.
        let n = legal.len();
        let single_layer = self.net.layers.len() == 1;
        let mut hidden = Vec::with_capacity(n * first.outputs);
        let mut action = [0.0; ACTION_FEATURES];
        for &combo in legal {
            encoding::encode_action(combo, &mut action);
            hidden.extend(
                first
                    .weights
                    .chunks_exact(first.inputs)
                    .zip(&shared)
                    .map(|(row, s)| {
                        let y = s + dot(&row[OBS_SIZE..], &action);
                        if single_layer { y } else { y.max(0.0) }
                    }),
            );
        }
        let last = self.net.layers.len() - 1;
        let mut next = Vec::new();
        for (i, layer) in self.net.layers.iter().enumerate().skip(1) {
            layer.forward_batch(&hidden, n, &mut next, i < last);
            std::mem::swap(&mut hidden, &mut next);
        }
        hidden
    }
}

impl Agent for DmcAgent {
    fn name(&self) -> &str {
        "dmc"
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        if legal.len() == 1 {
            return legal[0];
        }
        let values = self.evaluate(obs, legal);
        let best = (0..legal.len())
            .max_by(|&a, &b| values[a].total_cmp(&values[b]))
            .expect("legal moves are never empty");
        legal[best]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameState;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn write_mlp(layers: &[(usize, usize)], value: impl Fn(usize) -> f32) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend((layers.len() as u32).to_le_bytes());
        let mut i = 0;
        for &(inputs, outputs) in layers {
            bytes.extend((inputs as u32).to_le_bytes());
            bytes.extend((outputs as u32).to_le_bytes());
            for _ in 0..inputs * outputs + outputs {
                bytes.extend(value(i).to_le_bytes());
                i += 1;
            }
        }
        bytes
    }

    #[test]
    fn parses_and_validates_files() {
        let bytes = write_mlp(&[(3, 2), (2, 1)], |i| i as f32);
        let mlp = Mlp::from_bytes(&bytes).unwrap();
        assert_eq!((mlp.inputs(), mlp.outputs()), (3, 1));
        assert!(Mlp::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(Mlp::from_bytes(b"nonsense").is_err());
        assert!(Mlp::from_bytes(&write_mlp(&[(3, 2), (3, 1)], |_| 0.0)).is_err());
    }

    #[test]
    fn split_first_layer_matches_plain_forward() {
        let inputs = OBS_SIZE + ACTION_FEATURES;
        let bytes = write_mlp(&[(inputs, 8), (8, 4), (4, 1)], |i| {
            ((i * 7919) % 13) as f32 / 13.0 - 0.5
        });
        let net = Arc::new(Mlp::from_bytes(&bytes).unwrap());
        let agent = DmcAgent::new(Arc::clone(&net)).unwrap();

        let state = GameState::new(4, 0, &mut SmallRng::seed_from_u64(3));
        let obs = state.observe(0);
        let legal = state.legal_actions();
        let fast = agent.evaluate(&obs, &legal);
        for (&combo, &value) in legal.iter().zip(&fast) {
            let mut x = obs.encoded().to_vec();
            let mut a = [0.0; ACTION_FEATURES];
            encoding::encode_action(combo, &mut a);
            x.extend(a);
            let plain = net.forward(&x)[0];
            assert!((plain - value).abs() < 1e-4, "{plain} vs {value}");
        }
    }

    #[test]
    fn rejects_wrong_shapes() {
        let net = Arc::new(Mlp::from_bytes(&write_mlp(&[(5, 1)], |_| 0.0)).unwrap());
        assert!(DmcAgent::new(net).is_err());
    }
}
