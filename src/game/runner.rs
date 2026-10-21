use std::fmt;

use crate::agents::Agent;
use crate::game::{ActionError, GameState};

/// An agent chose a combo that [`GameState::apply`] rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunError {
    pub seat: usize,
    pub error: ActionError,
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "agent in seat {} made an illegal move: {}",
            self.seat, self.error
        )
    }
}

impl std::error::Error for RunError {}

/// Plays `state` to the end, asking `agents[seat]` for each move.
///
/// # Panics
/// If there are fewer agents than players.
pub fn run_game(
    mut state: GameState,
    agents: &mut [&mut dyn Agent],
) -> Result<GameState, RunError> {
    assert!(
        agents.len() >= state.num_players(),
        "need one agent per seat"
    );
    while !state.is_over() {
        let seat = state.current_player();
        let legal = state.legal_actions();
        let combo = agents[seat].act(&state.observe(seat), &legal);
        state
            .apply(combo)
            .map_err(|error| RunError { seat, error })?;
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::RandomAgent;
    use crate::constants::{MAX_PLAYERS, MIN_PLAYERS};
    use crate::game::{Combo, Observation};
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    struct Cheater;
    impl Agent for Cheater {
        fn act(&mut self, _: &Observation, _: &[Combo]) -> Combo {
            Combo::set(6, 10)
        }
    }

    #[test]
    fn illegal_moves_are_reported() {
        let state = GameState::new(3, 1, &mut SmallRng::seed_from_u64(0));
        let (mut a, mut b, mut c) = (RandomAgent::new(0), Cheater, RandomAgent::new(1));
        let err = run_game(state, &mut [&mut a, &mut b, &mut c]).unwrap_err();
        assert_eq!(err.seat, 1);
        assert!(matches!(err.error, ActionError::CardsNotInHand { .. }));
    }

    proptest! {
        #[test]
        fn random_agents_finish(seed in any::<u64>(), n in MIN_PLAYERS..=MAX_PLAYERS) {
            let state = GameState::new(n, 0, &mut SmallRng::seed_from_u64(seed));
            let mut agents: Vec<RandomAgent> = (0..n as u64).map(|i| RandomAgent::new(seed ^ i)).collect();
            let mut refs: Vec<&mut dyn Agent> = agents.iter_mut().map(|a| a as &mut dyn Agent).collect();
            let end = run_game(state, &mut refs).unwrap();
            prop_assert!(end.is_over());
        }
    }
}
