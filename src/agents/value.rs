//! Position evaluation shared by the search agents.

use std::sync::Arc;

use rand::Rng;

use crate::agents::dmc::{DmcAgent, Mlp};
use crate::agents::rollout::{Reward, Rollout};
use crate::constants::MAX_PLAYERS;
use crate::game::GameState;

/// How a search agent values a position where it stops searching.
#[derive(Debug, Clone)]
pub enum Evaluator {
    /// Play the game out with a rollout policy and score the result.
    Rollout(Rollout),
    /// For each seat, play `cheapest` moves until it is that seat's turn, then take the seat's
    /// best predicted return from a DMC network. Uses the true result if the game ends first.
    Dmc(Arc<Mlp>),
    /// A blend of the two, as AlphaGo mixed its value network with rollouts:
    /// `(1 - lambda) * network + lambda * rollout`.
    Mixed {
        net: Arc<Mlp>,
        rollout: Rollout,
        lambda: f32,
    },
}

/// Deterministic policy used to reach a seat's turn before asking the network.
const ADVANCE: Rollout = Rollout::Cheapest { epsilon: 0.0 };

impl Evaluator {
    /// Value of `state` for `seat`.
    pub fn value<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        seat: usize,
        reward: Reward,
        rng: &mut R,
    ) -> f32 {
        let mut s = state.clone();
        match self {
            Self::Rollout(rollout) => {
                rollout.play_out(&mut s, rng);
                reward.of(&s, seat)
            }
            Self::Mixed {
                net,
                rollout,
                lambda,
            } => {
                let network = Self::Dmc(Arc::clone(net)).value(state, seat, reward, rng);
                rollout.play_out(&mut s, rng);
                (1.0 - lambda) * network + lambda * reward.of(&s, seat)
            }
            Self::Dmc(net) => {
                while !s.is_over() && s.current_player() != seat {
                    let legal = s.legal_actions();
                    let combo = ADVANCE.choose(&s, &legal, rng);
                    s.apply(combo).expect("legal move");
                }
                if s.is_over() {
                    return reward.of(&s, seat);
                }
                DmcAgent::new(Arc::clone(net))
                    .expect("validated on construction")
                    .evaluate(&s.observe(seat), &s.legal_actions())
                    .into_iter()
                    .fold(f32::NEG_INFINITY, f32::max)
            }
        }
    }

    /// Values of `state` for every seat (unused seats are zero). A rollout evaluator scores all
    /// seats from one playout.
    pub fn values<R: Rng + ?Sized>(
        &self,
        state: &GameState,
        reward: Reward,
        rng: &mut R,
    ) -> [f32; MAX_PLAYERS] {
        match self {
            Self::Rollout(rollout) => {
                let mut s = state.clone();
                rollout.play_out(&mut s, rng);
                reward.all(&s)
            }
            Self::Dmc(_) => {
                let mut out = [0.0; MAX_PLAYERS];
                for (seat, v) in out.iter_mut().enumerate().take(state.num_players()) {
                    *v = self.value(state, seat, reward, rng);
                }
                out
            }
            Self::Mixed {
                net,
                rollout,
                lambda,
            } => {
                // one playout scores every seat; the network scores each seat separately
                let played = Self::Rollout(*rollout).values(state, reward, rng);
                let network = Self::Dmc(Arc::clone(net)).values(state, reward, rng);
                let mut out = [0.0; MAX_PLAYERS];
                for (o, (n, r)) in out.iter_mut().zip(network.iter().zip(played)) {
                    *o = (1.0 - lambda) * n + lambda * r;
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::encoding::{ACTION_FEATURES, OBS_SIZE};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn mixed_evaluator_interpolates_network_and_rollout() {
        let net = Arc::new(crate::agents::dmc::test_mlp(
            &[(OBS_SIZE + ACTION_FEATURES, 8), (8, 1)],
            2,
        ));
        let mut rng = SmallRng::seed_from_u64(3);
        let state = GameState::new(4, 0, &mut rng);
        let reward = Reward::default();
        let rollout = Rollout::Cheapest { epsilon: 0.0 };
        let mixed = |lambda| Evaluator::Mixed {
            net: Arc::clone(&net),
            rollout,
            lambda,
        };
        // deterministic policies make both parts reproducible
        let network = Evaluator::Dmc(Arc::clone(&net)).value(&state, 1, reward, &mut rng);
        let played = Evaluator::Rollout(rollout).value(&state, 1, reward, &mut rng);
        for lambda in [0.0, 0.3, 1.0] {
            let v = mixed(lambda).value(&state, 1, reward, &mut rng);
            assert!((v - ((1.0 - lambda) * network + lambda * played)).abs() < 1e-5);
            let all = mixed(lambda).values(&state, reward, &mut rng);
            assert!((all[1] - v).abs() < 1e-5);
        }
    }
}
