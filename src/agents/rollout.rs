//! Fast playout policies over full game states, and rewards for search agents.

use rand::{Rng, RngExt};

use crate::agents::Agent;
use crate::agents::baseline::cheapest_winner;
use crate::agents::counter::CounterAgent;
use crate::constants::MAX_PLAYERS;
use crate::game::{Combo, GameState};

/// How simulated players choose moves in playouts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rollout {
    /// Uniformly random legal moves.
    Random,
    /// The `cheapest` baseline, playing a random move with probability `epsilon` to keep
    /// playouts diverse.
    Cheapest { epsilon: f32 },
    /// The `counter` strategy, playing a random move with probability `epsilon`. Slower than
    /// `cheapest` (it observes the state every move) but much closer to good play.
    Counter { epsilon: f32 },
}

impl Default for Rollout {
    fn default() -> Self {
        Self::Cheapest { epsilon: 0.1 }
    }
}

impl Rollout {
    pub fn choose<R: Rng + ?Sized>(self, state: &GameState, legal: &[Combo], rng: &mut R) -> Combo {
        match self {
            Self::Cheapest { epsilon } if rng.random::<f32>() >= epsilon => {
                cheapest_winner(state.trick(), legal)
            }
            Self::Counter { epsilon } if rng.random::<f32>() >= epsilon => {
                CounterAgent::new().act(&state.observe(state.current_player()), legal)
            }
            _ => legal[rng.random_range(0..legal.len())],
        }
    }

    /// Plays `state` to the end.
    pub fn play_out<R: Rng + ?Sized>(self, state: &mut GameState, rng: &mut R) {
        while !state.is_over() {
            let legal = state.legal_actions();
            let combo = self.choose(state, &legal, rng);
            state
                .apply(combo)
                .expect("rollout policies only pick legal moves");
        }
    }
}

/// Converts a finished game into a per-seat reward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reward {
    /// Weight on the score margin over the best opponent (in units of 100 points), added to the
    /// win share. A little margin gives search a denser signal than wins alone.
    pub margin_weight: f32,
}

impl Default for Reward {
    fn default() -> Self {
        Self { margin_weight: 0.2 }
    }
}

impl Reward {
    pub fn of(self, state: &GameState, seat: usize) -> f32 {
        let returns = state.returns();
        if self.margin_weight == 0.0 {
            return returns[seat];
        }
        let scores = state.scores();
        let best_other = (0..scores.len())
            .filter(|&s| s != seat)
            .map(|s| scores[s])
            .max()
            .unwrap_or(0);
        let margin = (f32::from(scores[seat]) - f32::from(best_other)) / 100.0;
        returns[seat] + self.margin_weight * margin
    }

    /// Rewards for every seat (unused seats are zero).
    pub fn all(self, state: &GameState) -> [f32; MAX_PLAYERS] {
        let mut out = [0.0; MAX_PLAYERS];
        for (seat, r) in out.iter_mut().enumerate().take(state.num_players()) {
            *r = self.of(state, seat);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn playouts_finish() {
        let mut rng = SmallRng::seed_from_u64(0);
        for rollout in [Rollout::Random, Rollout::default()] {
            for n in 2..=6 {
                let mut state = GameState::new(n, 0, &mut rng);
                rollout.play_out(&mut state, &mut rng);
                assert!(state.is_over());
                let total: f32 = Reward { margin_weight: 0.0 }.all(&state).iter().sum();
                assert!((total - 1.0).abs() < 1e-6);
            }
        }
    }
}
