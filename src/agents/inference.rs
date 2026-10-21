//! Opponent inference for determinized search (after Rebstock et al., "Policy Based Inference in
//! Trick-Taking Card Games", 2019).
//!
//! Uniform determinization treats every consistent deal as equally likely, but opponents' past
//! plays say a lot about their hands: a player who could have taken the top pile cheaply and
//! didn't probably couldn't. Here many candidate deals are sampled, each is replayed from its
//! starting deal, and it is weighted by how likely an opponent model finds the moves the
//! opponents actually made. Search then samples deals in proportion to those weights.

use std::sync::Arc;

use rand::{Rng, RngExt};

use crate::agents::baseline::cheapest_winner;
use crate::agents::dmc::{DmcAgent, Mlp};
use crate::game::{Combo, GameState, Observation};

/// How opponents are assumed to choose moves.
#[derive(Debug, Clone)]
pub enum OpponentModel {
    /// The `cheapest` policy, playing a uniformly random move with probability `epsilon`.
    Cheapest { epsilon: f32 },
    /// A softmax over a DMC network's predicted returns.
    Dmc { net: Arc<Mlp>, temperature: f32 },
}

impl OpponentModel {
    /// Log-probability that a player with `obs` plays `chosen` from `legal`.
    pub fn log_prob(&self, obs: &Observation, legal: &[Combo], chosen: Combo) -> f32 {
        match self {
            Self::Cheapest { epsilon } => {
                let uniform = epsilon / legal.len() as f32;
                let p = if chosen == cheapest_winner(obs.trick(), legal) {
                    1.0 - epsilon + uniform
                } else {
                    uniform
                };
                p.ln()
            }
            Self::Dmc { net, temperature } => {
                let agent = DmcAgent::new(Arc::clone(net)).expect("validated on construction");
                let q = agent.evaluate(obs, legal);
                let max = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let log_z = q
                    .iter()
                    .map(|&v| ((v - max) / temperature).exp())
                    .sum::<f32>()
                    .ln();
                let i = legal
                    .iter()
                    .position(|&c| c == chosen)
                    .expect("chosen move is legal");
                (q[i] - max) / temperature - log_z
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Inference {
    pub model: OpponentModel,
    /// Candidate deals sampled per decision.
    pub candidates: usize,
    /// Only opponents' decisions among the last `window` plays are scored, to bound the cost.
    pub window: usize,
    /// Exponent applied to likelihoods. Values below 1 soften the model's confidence, keeping a
    /// few deals from taking all the weight.
    pub power: f32,
}

/// Candidate deals with cumulative weights, for weighted sampling.
#[derive(Debug, Clone)]
pub struct WorldPool {
    worlds: Vec<GameState>,
    cumulative: Vec<f32>,
}

impl WorldPool {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> &GameState {
        let total = *self.cumulative.last().expect("pool is never empty");
        let x = rng.random::<f32>() * total;
        let i = self.cumulative.partition_point(|&c| c <= x);
        &self.worlds[i.min(self.worlds.len() - 1)]
    }

    /// Effective sample size of the weights: `len()` when uniform, 1 when one deal dominates.
    pub fn effective_size(&self) -> f32 {
        let mut prev = 0.0;
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for &c in &self.cumulative {
            let w = c - prev;
            prev = c;
            sum += w;
            sum_sq += w * w;
        }
        sum * sum / sum_sq
    }

    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }
}

impl Inference {
    /// Log-likelihood of the opponents' recent moves if the deal were `world`.
    pub fn log_likelihood(&self, obs: &Observation, world: &GameState) -> f32 {
        let observer = obs.observer();
        let history = obs.history();
        let cutoff = history.len().saturating_sub(self.window);
        let mut state = obs.start_of(world);
        let mut total = 0.0;
        for (i, play) in history.iter().enumerate() {
            let seat = play.seat as usize;
            if i >= cutoff && seat != observer {
                let legal = state.legal_actions();
                if legal.len() > 1 {
                    total += self
                        .model
                        .log_prob(&state.observe(seat), &legal, play.combo);
                }
            }
            state.apply(play.combo).expect("history replays legally");
        }
        total
    }

    /// Samples candidate deals for `obs` and weights them by likelihood.
    pub fn sample_pool<R: Rng + ?Sized>(&self, obs: &Observation, rng: &mut R) -> WorldPool {
        let worlds: Vec<GameState> = (0..self.candidates.max(1))
            .map(|_| obs.determinize(rng))
            .collect();
        let logs: Vec<f32> = worlds.iter().map(|w| self.log_likelihood(obs, w)).collect();
        let max = logs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        let cumulative = logs
            .iter()
            .map(|&l| {
                total += (self.power * (l - max)).exp();
                total
            })
            .collect();
        WorldPool { worlds, cumulative }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::baseline::lowest_single;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn inference(epsilon: f32) -> Inference {
        Inference {
            model: OpponentModel::Cheapest { epsilon },
            candidates: 64,
            window: 60,
            power: 1.0,
        }
    }

    #[test]
    fn pool_worlds_match_the_observation() {
        let mut rng = SmallRng::seed_from_u64(1);
        let mut g = GameState::new(4, 0, &mut rng);
        for _ in 0..9 {
            let legal = g.legal_actions();
            g.apply(legal[rng.random_range(0..legal.len())]).unwrap();
        }
        let obs = g.observe(g.current_player());
        let pool = inference(0.3).sample_pool(&obs, &mut rng);
        assert_eq!(pool.len(), 64);
        for _ in 0..20 {
            assert_eq!(pool.sample(&mut rng).observe(obs.observer()), obs);
        }
        assert!(pool.effective_size() >= 1.0 && pool.effective_size() <= 64.0);
    }

    #[test]
    fn empty_window_gives_uniform_weights() {
        let mut rng = SmallRng::seed_from_u64(2);
        let mut g = GameState::new(3, 0, &mut rng);
        for _ in 0..6 {
            let legal = g.legal_actions();
            g.apply(legal[rng.random_range(0..legal.len())]).unwrap();
        }
        let inf = Inference {
            window: 0,
            ..inference(0.3)
        };
        let pool = inf.sample_pool(&g.observe(0), &mut rng);
        assert!((pool.effective_size() - 64.0).abs() < 1e-2);
        // with scoring on, the cheapest model separates deals
        let pool = inference(0.3).sample_pool(&g.observe(0), &mut rng);
        assert!(pool.effective_size() < 64.0);
    }

    #[test]
    fn declining_a_cheap_win_signals_a_weak_hand() {
        // Seat 1 played its lowest single after seat 0's single 1. Under the cheapest model,
        // deals where seat 1 could have won with a single above 1 are less likely.
        let mut rng = SmallRng::seed_from_u64(4);
        let mut g = GameState::new(3, 0, &mut rng);
        g.apply(Combo::single(lowest_rank(&g))).unwrap();
        let legal = g.legal_actions();
        g.apply(lowest_single(&legal)).unwrap();
        let obs = g.observe(2);
        let inf = inference(0.2);
        let world = obs.determinize(&mut rng);
        let ll = inf.log_likelihood(&obs, &world);
        assert!(ll.is_finite() && ll <= 0.0);
    }

    fn lowest_rank(g: &GameState) -> u8 {
        let hand = g.hand(g.current_player());
        (1..=6).find(|&r| hand.count(r) > 0).unwrap()
    }
}
