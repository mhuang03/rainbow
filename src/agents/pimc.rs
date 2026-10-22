//! Perfect-information Monte Carlo (PIMC): sample full deals consistent with the observation,
//! try every legal move on each, value the result (by finishing the game with a fast rollout
//! policy, or with a DMC network), and play the move with the best mean value.
//!
//! Every move is evaluated on the same sampled deals with the same rollout seeds (common random
//! numbers), so differences between moves are not drowned out by the luck of the sample.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::agents::Agent;
use crate::agents::rollout::{Reward, Rollout};
use crate::agents::value::Evaluator;
use crate::game::{Combo, Observation};

#[derive(Debug, Clone)]
pub struct Params {
    /// Deals sampled per decision. Each legal move gets one evaluation per deal.
    pub determinizations: usize,
    /// How a deal is valued after each move: a rollout, or a DMC network.
    pub leaf: Evaluator,
    pub reward: Reward,
}

impl Default for Params {
    /// `counter` rollouts with 10% random moves: 1.80 against three Economists, where random
    /// rollouts score 0.94. (`cheapest` rollouts assume every opponent plays like `cheapest`,
    /// which wins against it but loses badly to stronger players; `counter` is close enough to
    /// good play that this bias stops hurting.)
    fn default() -> Self {
        Self {
            determinizations: 64,
            leaf: Evaluator::Rollout(Rollout::Counter { epsilon: 0.1 }),
            reward: Reward::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PimcAgent {
    params: Params,
    rng: SmallRng,
}

impl PimcAgent {
    pub fn new(seed: u64) -> Self {
        Self::with_params(Params::default(), seed)
    }

    pub fn with_params(params: Params, seed: u64) -> Self {
        Self {
            params,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Mean reward of each legal move, in the order of `legal`.
    pub fn evaluate(&mut self, obs: &Observation, legal: &[Combo]) -> Vec<f32> {
        let p = &self.params;
        let me = obs.observer();
        let mut totals = vec![0.0f32; legal.len()];
        for _ in 0..p.determinizations {
            let deal = obs.determinize(&mut self.rng);
            let rollout_seed: u64 = self.rng.random();
            for (total, &combo) in totals.iter_mut().zip(legal) {
                let mut state = deal.clone();
                state.apply(combo).expect("legal in every consistent deal");
                let mut rng = SmallRng::seed_from_u64(rollout_seed);
                *total += p.leaf.value(&state, me, p.reward, &mut rng);
            }
        }
        let k = p.determinizations.max(1) as f32;
        totals.iter().map(|t| t / k).collect()
    }
}

impl Agent for PimcAgent {
    fn name(&self) -> &str {
        "pimc"
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

    #[test]
    fn plays_legal_moves_for_whole_games() {
        for n in [2, 3, 6] {
            let mut rng = SmallRng::seed_from_u64(n as u64);
            let mut state = GameState::new(n, 0, &mut rng);
            let mut agent = PimcAgent::with_params(
                Params {
                    determinizations: 4,
                    ..Params::default()
                },
                1,
            );
            while !state.is_over() {
                let seat = state.current_player();
                let combo = agent.act(&state.observe(seat), &state.legal_actions());
                state.apply(combo).unwrap();
            }
        }
    }
}
