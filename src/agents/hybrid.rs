//! Plays a base agent (a network, or `counter`) until the endgame, then solves the rest of the
//! game exactly.
//!
//! Once few cards remain in all hands and reserves (a public count), every sampled deal is small
//! enough to search to the end with max-n, so the agent switches from the network to exact
//! endgame search over sampled deals (see [`minimax`](crate::agents::minimax)). Endgames are
//! where a fixed policy's small misjudgements, like ending the game a trick too early, cost
//! the most.

use crate::agents::Agent;
use crate::agents::minimax::{self, MinimaxAgent, Mode};
use crate::game::{Combo, Observation};

pub struct HybridAgent {
    base: Box<dyn Agent>,
    endgame: MinimaxAgent,
    /// Switch to exact search once at most this many cards remain in hands and reserves.
    threshold: usize,
}

impl HybridAgent {
    /// `deals` sampled deals are solved per endgame decision.
    pub fn new(base: Box<dyn Agent>, threshold: usize, deals: usize, seed: u64) -> Self {
        let params = minimax::Params {
            determinizations: deals,
            depth: 2,
            mode: Mode::MaxN,
            endgame_cards: threshold,
            ..minimax::Params::default()
        };
        Self {
            base,
            endgame: MinimaxAgent::with_params(params, seed),
            threshold,
        }
    }

    fn cards_left(obs: &Observation) -> usize {
        obs.players()
            .iter()
            .map(|p| usize::from(p.hand_size) + usize::from(p.reserve_size))
            .sum()
    }
}

impl Agent for HybridAgent {
    fn name(&self) -> &str {
        "hybrid"
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        if legal.len() == 1 {
            return legal[0];
        }
        if Self::cards_left(obs) <= self.threshold {
            self.endgame.act(obs, legal)
        } else {
            self.base.act(obs, legal)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameState;
    use crate::game::encoding::{ACTION_FEATURES, OBS_SIZE};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use std::sync::Arc;

    #[test]
    fn plays_legal_moves_for_whole_games() {
        let net = Arc::new(crate::agents::dmc::test_mlp(
            &[(OBS_SIZE + ACTION_FEATURES, 8), (8, 1)],
            2,
        ));
        let base = Box::new(crate::agents::DmcAgent::new(net).unwrap());
        let mut agent = HybridAgent::new(base, 6, 2, 0);
        for n in [2, 3, 5] {
            let mut state = GameState::new(n, 0, &mut SmallRng::seed_from_u64(n as u64));
            while !state.is_over() {
                let seat = state.current_player();
                let combo = agent.act(&state.observe(seat), &state.legal_actions());
                state.apply(combo).unwrap();
            }
        }
    }
}
