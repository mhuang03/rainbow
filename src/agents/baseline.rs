//! Simple deterministic policies. The move rules are plain functions so search agents can reuse
//! them as fast rollout policies on full game states.

use crate::agents::Agent;
use crate::game::{Combo, ComboKind, Observation, Play};

/// The lowest single in `legal`. Every non-empty hand has one, and singles are always legal.
pub fn lowest_single(legal: &[Combo]) -> Combo {
    legal
        .iter()
        .copied()
        .filter(|c| matches!(c.kind(), ComboKind::Single { .. }))
        .min_by_key(|c| c.high_rank())
        .expect("a legal move list always contains a single")
}

/// The strongest combo in `legal`, preferring fewer cards among equals.
pub fn biggest(legal: &[Combo]) -> Combo {
    *legal
        .iter()
        .max_by_key(|c| c.strength())
        .expect("legal moves are never empty")
}

/// The cheapest combo that would currently win the trick (fewest cards, then lowest rank), or
/// the lowest single if nothing wins. Ties go to the earlier play, so a winner must be strictly
/// stronger than every play so far.
pub fn cheapest_winner(trick: &[Play], legal: &[Combo]) -> Combo {
    let best = trick.iter().map(|p| p.combo.strength()).max();
    legal
        .iter()
        .copied()
        .filter(|c| best.is_none_or(|b| c.strength() > b))
        .min_by_key(|c| c.strength())
        .unwrap_or_else(|| lowest_single(legal))
}

/// Always plays its lowest single, keeping cards to stay in the game as long as possible.
#[derive(Debug, Default, Clone, Copy)]
pub struct LowestAgent;

impl Agent for LowestAgent {
    fn name(&self) -> &str {
        "lowest"
    }

    fn act(&mut self, _: &Observation, legal: &[Combo]) -> Combo {
        lowest_single(legal)
    }
}

/// Always plays its strongest combo, spending cards freely to claim the top pile.
#[derive(Debug, Default, Clone, Copy)]
pub struct BiggestAgent;

impl Agent for BiggestAgent {
    fn name(&self) -> &str {
        "biggest"
    }

    fn act(&mut self, _: &Observation, legal: &[Combo]) -> Combo {
        biggest(legal)
    }
}

/// Plays the cheapest combo that currently wins the trick, otherwise its lowest single.
#[derive(Debug, Default, Clone, Copy)]
pub struct CheapestAgent;

impl Agent for CheapestAgent {
    fn name(&self) -> &str {
        "cheapest"
    }

    fn act(&mut self, observation: &Observation, legal: &[Combo]) -> Combo {
        cheapest_winner(observation.trick(), legal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_choices() {
        let legal = [
            Combo::single(2),
            Combo::single(5),
            Combo::set(2, 2),
            Combo::run(5, 6),
            Combo::single(6),
        ];
        assert_eq!(lowest_single(&legal), Combo::single(2));
        assert_eq!(biggest(&legal), Combo::run(5, 6));

        // leading: the weakest combo already "wins"
        assert_eq!(cheapest_winner(&[], &legal), Combo::single(2));
        // must strictly beat a single 5: a 6 is cheaper than any pair
        let trick = [Play {
            seat: 1,
            combo: Combo::single(5),
        }];
        assert_eq!(cheapest_winner(&trick, &legal), Combo::single(6));
        // beating a pair of 4s needs the 5-6 run; the pair of 2s is too weak
        let trick = [Play {
            seat: 1,
            combo: Combo::set(4, 2),
        }];
        assert_eq!(cheapest_winner(&trick, &legal), Combo::run(5, 6));
        // nothing beats three 3s: fall back to the lowest single
        let trick = [Play {
            seat: 1,
            combo: Combo::set(3, 3),
        }];
        assert_eq!(cheapest_winner(&trick, &legal), Combo::single(2));
    }
}
