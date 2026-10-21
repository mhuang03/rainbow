//! Agents whose network maps an observation to one output per action and play the legal action
//! with the highest output (or sample from their softmax): Q-values from `python/train_dqn.py`
//! (`dqn`), or policy logits from `python/train_ppo.py` (`ppo`) and `python/train_nfsp.py`
//! (`nfsp`). Networks use the same MLP format as
//! [`dmc`](crate::agents::dmc).

use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::agents::Agent;
use crate::agents::dmc::Mlp;
use crate::game::encoding::OBS_SIZE;
use crate::game::{Combo, NUM_ACTIONS, Observation};

#[derive(Debug, Clone)]
pub struct DqnAgent {
    net: Arc<Mlp>,
    name: &'static str,
    /// When set, sample from a softmax over the outputs (policy logits) instead of taking the
    /// highest.
    sampler: Option<SmallRng>,
}

impl DqnAgent {
    /// # Errors
    /// If the network doesn't map an observation to one value per action.
    pub fn new(net: Arc<Mlp>) -> Result<Self, String> {
        if net.inputs() != OBS_SIZE || net.outputs() != NUM_ACTIONS {
            return Err(format!(
                "DQN network must map {OBS_SIZE} inputs to {NUM_ACTIONS} outputs, got {} -> {}",
                net.inputs(),
                net.outputs()
            ));
        }
        Ok(Self {
            net,
            name: "dqn",
            sampler: None,
        })
    }

    /// The same agent reporting a different name, e.g. `"ppo"` for a policy network.
    #[must_use]
    pub fn named(self, name: &'static str) -> Self {
        Self { name, ..self }
    }

    /// Samples moves from the softmax of the outputs, treating them as policy logits (for PPO
    /// and NFSP average policies, whose mixing matters).
    #[must_use]
    pub fn sampling(self, seed: u64) -> Self {
        Self {
            sampler: Some(SmallRng::seed_from_u64(seed)),
            ..self
        }
    }

    /// Q-values of the legal combos, in the order of `legal`.
    pub fn evaluate(&self, obs: &Observation, legal: &[Combo]) -> Vec<f32> {
        let q = self.net.forward(&obs.encoded());
        legal.iter().map(|c| q[c.action_index()]).collect()
    }
}

impl Agent for DqnAgent {
    fn name(&self) -> &str {
        self.name
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        let q = self.evaluate(obs, legal);
        if let Some(rng) = &mut self.sampler {
            let max = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = q.iter().map(|&v| (v - max).exp()).collect();
            let mut x = rng.random::<f32>() * weights.iter().sum::<f32>();
            for (&combo, &w) in legal.iter().zip(&weights) {
                if x < w {
                    return combo;
                }
                x -= w;
            }
            return legal[legal.len() - 1];
        }
        let best = (0..legal.len())
            .max_by(|&a, &b| q[a].total_cmp(&q[b]))
            .expect("legal moves are never empty");
        legal[best]
    }
}
