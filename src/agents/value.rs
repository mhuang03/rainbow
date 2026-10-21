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
        }
    }
}
