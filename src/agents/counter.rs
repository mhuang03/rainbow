//! "Counter": a deterministic card-counting strategy.
//!
//! Every card that has been face up is public, so the unknown cards contain every opponent's
//! hand. That bounds the strongest combo each later opponent could possibly play: the biggest
//! set or the longest run they could hold, within their hand size and the trick type. A combo
//! beating the trick so far and every later opponent's bound is a sure winner.
//!
//! What a win is worth is the *difference* between the top pile and the pile the agent would get
//! anyway by dumping a single. Each combo beating the trick so far is credited with that gap,
//! times its chance of holding: 1 for every later opponent who provably can't beat it, and `q`
//! for every one who might (`q`). The agent plays the best such combo if its expected gap pays for its
//! extra cards; otherwise it dumps the single that hurts its hand's structure least.

use arrayvec::ArrayVec;

use crate::agents::Agent;
use crate::constants::{MAX_PLAYS_PER_TRICK, MAX_RANK, MIN_RANK};
use crate::game::{Combo, ComboKind, Hand, Observation, TrickType};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Points each card beyond the first must buy for a sure win to be worth it.
    pub card_value: f32,
    /// Penalty, in points, for breaking a pair or set or a run link when dumping.
    pub structure_value: f32,
    /// Assumed chance that a later opponent who could beat a combo doesn't.
    pub hold_chance: f32,
}

impl Default for Params {
    /// A quick sweep against `cheapest` opponents (q 0.3-0.9, card 1-2); not yet tuned.
    fn default() -> Self {
        Self {
            card_value: 1.0,
            structure_value: 1.0,
            hold_chance: 0.8,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CounterAgent {
    params: Params,
}

impl CounterAgent {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_params(params: Params) -> Self {
        Self { params }
    }
}

/// Upper bound on the strength of any combo a player holding `hand_size` of the `unknown`
/// cards could play in a trick of `trick_type`.
pub fn strongest_possible(unknown: &Hand, hand_size: usize, trick_type: TrickType) -> u8 {
    if hand_size == 0 {
        return 0;
    }
    let cap = hand_size.min(u8::MAX as usize) as u8;
    let mut best = 0;
    let mut run_start = None;
    for rank in MIN_RANK..=MAX_RANK {
        let count = unknown.count(rank).min(cap);
        if count == 0 {
            run_start = None;
            continue;
        }
        best = best.max(Combo::single(rank).strength());
        if trick_type != TrickType::Run && count >= 2 {
            best = best.max(Combo::set(rank, count).strength());
        }
        let start = *run_start.get_or_insert(rank);
        if trick_type != TrickType::Set && rank > start {
            // the longest run ending here that fits in the hand
            let start = start.max((rank + 1).saturating_sub(cap));
            if rank > start {
                best = best.max(Combo::run(start, rank).strength());
            }
        }
    }
    best
}

/// How much breaking the hand's structure costs when `combo` is removed: cards beyond the
/// first of a rank, and links between adjacent held ranks.
fn structure(hand: &Hand) -> u32 {
    let mut value = 0;
    for rank in MIN_RANK..=MAX_RANK {
        let count = u32::from(hand.count(rank));
        value += count.saturating_sub(1);
        if rank > MIN_RANK && count > 0 && hand.count(rank - 1) > 0 {
            value += 1;
        }
    }
    value
}

impl Agent for CounterAgent {
    fn name(&self) -> &str {
        "counter"
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        if legal.len() == 1 {
            return legal[0];
        }
        let p = self.params;
        let me = obs.observer();
        let n = obs.num_players();
        let players = obs.players();
        let trick = obs.trick();
        let piles = obs.piles();
        let unknown = obs.unknown();
        let hand = obs.hand();

        // Seats of the plays still to come after ours (ours again in 2-player tricks).
        let remaining = obs.trick_size().saturating_sub(trick.len() + 1);
        let mut later: ArrayVec<usize, MAX_PLAYS_PER_TRICK> = ArrayVec::new();
        let mut seat = me;
        for _ in 0..remaining {
            seat = (1..=n)
                .map(|i| (seat + i) % n)
                .find(|&s| s == me || players[s].hand_size > 0)
                .unwrap_or(me);
            later.push(seat);
        }

        let best_so_far = trick.iter().map(|t| t.combo.strength()).max().unwrap_or(0);
        let top = piles.first().map_or(0.0, |&v| f32::from(v));
        // Dumping a single lands us at or near the bottom; the bottom pile (or nothing) is the
        // fallback a sure win is measured against.
        let fallback = if piles.len() > trick.len() + later.len() {
            piles.last().map_or(0.0, |&v| f32::from(v))
        } else {
            0.0
        };

        // Chance a combo that beats the trick so far still wins it.
        let holds = |combo: Combo| {
            let strength = combo.strength();
            let trick_type = obs.trick_type().after(combo);
            later
                .iter()
                .filter(|&&s| s != me)
                .map(|&s| {
                    let bound =
                        strongest_possible(&unknown, players[s].hand_size.into(), trick_type);
                    if strength > bound { 1.0 } else { p.hold_chance }
                })
                .product::<f32>()
        };
        let best_win = legal
            .iter()
            .copied()
            .filter(|c| c.strength() > best_so_far)
            .map(|c| {
                let gain = holds(c) * (top - fallback);
                (c, gain - p.card_value * f32::from(c.num_cards() - 1))
            })
            .filter(|&(_, value)| value > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((combo, _)) = best_win {
            return combo;
        }

        // Dump: the single costing the least structure, lowest rank first.
        let base = structure(&hand);
        legal
            .iter()
            .copied()
            .filter(|c| matches!(c.kind(), ComboKind::Single { .. }))
            .min_by(|a, b| {
                let cost = |c: &Combo| {
                    let mut after = hand;
                    after.remove_combo(*c);
                    p.structure_value * (base - structure(&after)) as f32
                        + f32::from(c.high_rank()) * 0.1
                };
                cost(a).total_cmp(&cost(b))
            })
            .expect("a legal move list always contains a single")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameState;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn strongest_possible_bounds_sets_and_runs() {
        let unknown = Hand::from_counts([0, 3, 1, 1, 0, 2]);
        // sets: three 2s; runs: 2-3-4; open takes the bigger (both 3 cards, run ends higher)
        assert_eq!(
            strongest_possible(&unknown, 14, TrickType::Set),
            Combo::set(2, 3).strength()
        );
        assert_eq!(
            strongest_possible(&unknown, 14, TrickType::Run),
            Combo::run(2, 4).strength()
        );
        assert_eq!(
            strongest_possible(&unknown, 14, TrickType::Open),
            Combo::run(2, 4).strength()
        );
        // a two-card hand can't hold more than a pair or a two-card run
        assert_eq!(
            strongest_possible(&unknown, 2, TrickType::Run),
            Combo::run(3, 4).strength()
        );
        assert_eq!(
            strongest_possible(&unknown, 2, TrickType::Set),
            Combo::set(6, 2).strength()
        );
        assert_eq!(strongest_possible(&unknown, 0, TrickType::Open), 0);
    }

    #[test]
    fn bound_is_never_exceeded() {
        // No legal combo of any opponent can be stronger than the bound computed from the
        // observer's unknown cards.
        let mut rng = SmallRng::seed_from_u64(4);
        for n in 2..=6 {
            let mut state = GameState::new(n, 0, &mut rng);
            let mut agent = CounterAgent::new();
            while !state.is_over() {
                let seat = state.current_player();
                let observer = (seat + 1) % n;
                let unknown = state.observe(observer).unknown();
                if observer != seat {
                    let bound =
                        strongest_possible(&unknown, state.hand(seat).len(), state.trick_type());
                    for c in state.legal_actions() {
                        assert!(c.strength() <= bound, "{c} beats bound {bound}");
                    }
                }
                let combo = agent.act(&state.observe(seat), &state.legal_actions());
                state.apply(combo).unwrap();
            }
        }
    }
}
