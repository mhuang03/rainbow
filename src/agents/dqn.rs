//! Agents whose network maps an observation to one output per action and play the legal action
//! with the highest output: Q-values from `python/train_dqn.py` (`dqn`) or policy logits from
//! `python/train_ppo.py` (`ppo`). Networks use the same MLP format as
//! [`dmc`](crate::agents::dmc).

use std::sync::Arc;

use crate::agents::Agent;
use crate::agents::dmc::Mlp;
use crate::game::encoding::OBS_SIZE;
use crate::game::{Combo, NUM_ACTIONS, Observation};

#[derive(Debug, Clone)]
pub struct DqnAgent {
    net: Arc<Mlp>,
    name: &'static str,
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
        Ok(Self { net, name: "dqn" })
    }

    /// The same agent reporting a different name, e.g. `"ppo"` for a policy network.
    #[must_use]
    pub fn named(self, name: &'static str) -> Self {
        Self { name, ..self }
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
        let best = (0..legal.len())
            .max_by(|&a, &b| q[a].total_cmp(&q[b]))
            .expect("legal moves are never empty");
        legal[best]
    }
}
