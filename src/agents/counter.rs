//! "Counter": a deterministic card-counting strategy.
//!
//! Every card that has been face up is public, so the unknown cards contain every opponent's
//! hand. That bounds the strongest combo each later opponent could possibly play: the biggest
//! set or the longest run they could hold, within their hand size and the trick type. A combo
//! beating the trick so far and every later opponent's bound is a sure winner.
//!
//! Every play in a trick takes a pile by its finishing rank, so what matters is the *difference*
//! between piles, not their size. Each legal combo is scored by the pile it expects to finish
//! on: the plays already made that it doesn't beat finish ahead of it, and each later opponent
//! who could beat it (by the bound) is assumed to do so with probability `1 - q`. From that
//! expected pile it subtracts a cost per card beyond the first and the damage to the hand's
//! structure (pairs, sets and run links broken), and plays the best.

use arrayvec::ArrayVec;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::sync::Arc;

use crate::agents::Agent;
use crate::agents::belief::Belief;
use crate::constants::{MAX_PLAYERS, MAX_PLAYS_PER_TRICK, MAX_RANK, MIN_RANK};
use crate::game::{Combo, Hand, Observation, TrickType};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Points each card beyond the first must buy for a sure win to be worth it.
    pub card_value: f32,
    /// Penalty, in points, for each pair, set or run link the play breaks.
    pub structure_value: f32,
    /// Assumed chance that a later opponent who could beat a combo doesn't.
    pub hold_chance: f32,
    /// Learn each opponent's hold chance from the game so far: their rate of beating the trick
    /// when following, shrunk towards `1 - hold_chance` with this many pseudo-observations.
    pub adaptive: Option<f32>,
    /// Deals of the unseen cards sampled to estimate whether each later opponent can beat a
    /// combo. 0 uses the worst-case bound over all unseen cards instead.
    pub samples: usize,
}

impl Params {
    /// The `counter` agent's settings: 24 sampled deals instead of the bound, with the costs
    /// retuned for them (q 0, card 2, structure 0.5). Against three Economists: 1.20, where
    /// the bound version scores 1.12 (600 deals); head-to-head 1.12 to 0.88 (4p). Rollouts and
    /// search priors keep the faster bound version ([`Params::default`]).
    pub fn sampled() -> Self {
        Self {
            card_value: 2.0,
            structure_value: 0.5,
            hold_chance: 0.0,
            samples: 24,
            ..Self::default()
        }
    }
}

impl Default for Params {
    /// Grid search against three Economists (4p, 400 deals): q 0.2-0.9, card 0-2.5, structure
    /// 0-2. Results are flat around these values (1.09-1.11) and fall off with higher costs.
    fn default() -> Self {
        Self {
            card_value: 0.5,
            structure_value: 0.5,
            hold_chance: 0.5,
            adaptive: None,
            samples: 0,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CounterAgent {
    params: Params,
    /// With sampling: weights the sampled deals by a hand-prediction network.
    belief: Option<Arc<Belief>>,
}

impl CounterAgent {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_params(params: Params) -> Self {
        Self {
            params,
            belief: None,
        }
    }

    /// Samples deals weighted by `belief`'s predicted hands (when `samples` > 0).
    pub fn with_belief(mut self, belief: Option<Arc<Belief>>) -> Self {
        self.belief = belief;
        self
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

impl CounterAgent {
    /// The value of each legal combo, in the order of `legal`: the pile it expects to finish
    /// on, less the cost of the cards it spends and the hand structure it breaks.
    pub fn scores(&self, obs: &Observation, legal: &[Combo]) -> Vec<f32> {
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

        // Each later opponent's chance of not beating a combo they could beat.
        let mut hold = [p.hold_chance; MAX_PLAYERS];
        if let Some(weight) = p.adaptive {
            for (h, (follows, beats)) in hold.iter_mut().zip(obs.follow_record()) {
                let beat_rate =
                    (beats as f32 + weight * (1.0 - p.hold_chance)) / (follows as f32 + weight);
                *h = 1.0 - beat_rate;
            }
        }

        // With sampling, each later opponent's best possible play (per trick type) in each
        // sampled deal; the chance they can beat a combo is the fraction of deals where it's
        // stronger. Seeded from the position so scores stay a pure function of it.
        let mut sampled: Vec<[[u8; 3]; MAX_PLAYERS]> = Vec::with_capacity(p.samples);
        if p.samples > 0 {
            let seed = hand
                .counts()
                .iter()
                .fold(obs.history().len() as u64, |h, &c| {
                    h.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ u64::from(c)
                });
            let mut rng = SmallRng::seed_from_u64(seed);
            let weights = self.belief.as_ref().map(|b| b.weights(obs));
            for _ in 0..p.samples {
                let world = match &weights {
                    Some(w) => obs.determinize_weighted(w, &mut rng),
                    None => obs.determinize(&mut rng),
                };
                let mut best = [[0u8; 3]; MAX_PLAYERS];
                for &s in later.iter().filter(|&&s| s != me) {
                    let h = world.hand(s);
                    for (i, tt) in [TrickType::Open, TrickType::Set, TrickType::Run]
                        .into_iter()
                        .enumerate()
                    {
                        best[s][i] = strongest_possible(&h, h.len(), tt);
                    }
                }
                sampled.push(best);
            }
        }

        // Every play takes a pile by its finishing rank, so a combo is worth the pile it
        // expects to finish on: plays already made that it doesn't beat rank ahead of it (ties
        // go to the earlier play), and each later opponent who could beat it does so with
        // probability 1 - q.
        let base = structure(&hand);
        let score = |combo: Combo| {
            let strength = combo.strength();
            let ahead = trick
                .iter()
                .filter(|t| t.combo.strength() >= strength)
                .count();
            let trick_type = obs.trick_type().after(combo);
            // dist[k]: probability that k later opponents beat this combo
            let mut dist: ArrayVec<f32, { MAX_PLAYS_PER_TRICK + 1 }> = ArrayVec::new();
            dist.push(1.0);
            for &s in later.iter().filter(|&&s| s != me) {
                let can_beat = if sampled.is_empty() {
                    let bound =
                        strongest_possible(&unknown, players[s].hand_size.into(), trick_type);
                    f32::from(strength <= bound)
                } else {
                    let tt = trick_type as usize;
                    sampled.iter().filter(|b| b[s][tt] > strength).count() as f32
                        / sampled.len() as f32
                };
                let beat = can_beat * (1.0 - hold[s]);
                dist.push(0.0);
                for k in (0..dist.len()).rev() {
                    let stay = dist[k] * (1.0 - beat);
                    let moved = if k > 0 { dist[k - 1] * beat } else { 0.0 };
                    dist[k] = stay + moved;
                }
            }
            let pile: f32 = dist
                .iter()
                .enumerate()
                .map(|(k, &pr)| pr * piles.get(ahead + k).map_or(0.0, |&v| f32::from(v)))
                .sum();
            let mut after = hand;
            after.remove_combo(combo);
            pile - p.card_value * f32::from(combo.num_cards() - 1)
                - p.structure_value * base.saturating_sub(structure(&after)) as f32
                - f32::from(combo.high_rank()) * 0.01
        };
        legal.iter().map(|&c| score(c)).collect()
    }
}

impl Agent for CounterAgent {
    fn name(&self) -> &str {
        "counter"
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        if legal.len() == 1 {
            return legal[0];
        }
        let scores = self.scores(obs, legal);
        let best = (0..legal.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .expect("legal moves are never empty");
        legal[best]
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
