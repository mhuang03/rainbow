//! "Economist": a hand-written strategy that prices every legal combo.
//!
//! Each combo is scored as
//!
//! ```text
//! expected pile - card cost - structure damage + endgame
//! ```
//!
//! - **Expected pile.** The combo's rank among plays already made is exact. For each player still
//!   to play, the chance they beat it is estimated by sampling their hand from the unknown cards;
//!   together these give a distribution over the pile the combo claims.
//! - **Card cost.** Cards in hand are only worth something while they keep the player in the
//!   game. Cards beyond what is needed to last until the game is expected to end are nearly
//!   free; each card below that costs about one trick's income.
//! - **Structure damage.** Pairs, sets and runs left in hand are future winning combos, so
//!   spending or breaking them costs a little.
//! - **Endgame.** If playing out the hand would end the game, do it only when that wins.

use arrayvec::ArrayVec;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::agents::Agent;
use crate::constants::{DECK_SIZE, MAX_PLAYERS, MAX_PLAYS_PER_TRICK, MAX_RANK, MIN_RANK, Setup};
use crate::game::{Combo, Hand, Observation, TrickType};

/// Tunable weights for [`EconomistAgent`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Opponent hands sampled to estimate how likely each one can beat a combo.
    pub samples: usize,
    /// Probability an opponent who can beat a combo chooses to.
    pub beat_willingness: f32,
    /// Average cards an opponent spends per play, used to estimate when they run out.
    pub burn_rate: f32,
    /// Cost of each card spent beyond the first, as a fraction of a trick's income, when the
    /// card would otherwise have kept the player in the game.
    pub scarce_card_cost: f32,
    /// Cost of each extra card when the hand has cards to spare, in points.
    pub spare_card_cost: f32,
    /// Value of each card beyond the first of a rank (future sets), in points.
    pub set_potential: f32,
    /// Value of each adjacent pair of ranks held (future runs), in points.
    pub run_potential: f32,
    /// Bonus or penalty for ending the game while winning or losing, in points.
    pub endgame_weight: f32,
}

impl Default for Params {
    /// Tuned by `examples/tune_economist.rs` against a mixed field of opponents.
    fn default() -> Self {
        Self {
            samples: 48,
            beat_willingness: 0.68,
            burn_rate: 2.0,
            scarce_card_cost: 0.5,
            spare_card_cost: 1.0,
            set_potential: 0.8,
            run_potential: 0.6,
            endgame_weight: 50.0,
        }
    }
}

/// See the [module docs](self).
#[derive(Debug, Clone)]
pub struct EconomistAgent {
    params: Params,
    rng: SmallRng,
}

impl EconomistAgent {
    pub fn new(seed: u64) -> Self {
        Self::with_params(Params::default(), seed)
    }

    pub fn with_params(params: Params, seed: u64) -> Self {
        Self {
            params,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn params(&self) -> &Params {
        &self.params
    }
}

/// Strongest combo strength an opponent can play under each trick type, sorted ascending over
/// sampled hands so the fraction beating a strength is a binary search.
#[derive(Debug, Default)]
struct StrengthSamples {
    open: Vec<u8>,
    set: Vec<u8>,
    run: Vec<u8>,
}

impl StrengthSamples {
    fn prob_beats(&self, trick_type: TrickType, strength: u8) -> f32 {
        let samples = match trick_type {
            TrickType::Open => &self.open,
            TrickType::Set => &self.set,
            TrickType::Run => &self.run,
        };
        if samples.is_empty() {
            return 0.0;
        }
        let at_most = samples.partition_point(|&s| s <= strength);
        (samples.len() - at_most) as f32 / samples.len() as f32
    }
}

/// Strongest single-or-set and single-or-run in `hand`.
fn best_strengths(hand: &Hand) -> (u8, u8) {
    let mut best_set = 0;
    let mut best_run = 0;
    let mut run_len = 0u8;
    for rank in MIN_RANK..=MAX_RANK {
        let count = hand.count(rank);
        if count == 0 {
            run_len = 0;
            continue;
        }
        run_len += 1;
        let single = Combo::single(rank).strength();
        best_set = best_set.max(if count >= 2 {
            Combo::set(rank, count).strength()
        } else {
            single
        });
        best_run = best_run.max(if run_len >= 2 {
            Combo::run(rank + 1 - run_len, rank).strength()
        } else {
            single
        });
    }
    (best_set, best_run)
}

/// Future-combo potential of a hand: cards beyond the first of each rank, and adjacent held ranks.
fn structure(hand: &Hand, params: &Params) -> f32 {
    let mut value = 0.0;
    for rank in MIN_RANK..=MAX_RANK {
        let count = hand.count(rank);
        value += params.set_potential * f32::from(count.saturating_sub(1));
        if rank > MIN_RANK && count > 0 && hand.count(rank - 1) > 0 {
            value += params.run_potential;
        }
    }
    value
}

impl EconomistAgent {
    fn sample_opponent(&mut self, pool: &mut [u8], hand_size: usize) -> StrengthSamples {
        let mut out = StrengthSamples::default();
        let hand_size = hand_size.min(pool.len());
        for _ in 0..self.params.samples {
            // partial Fisher-Yates: the first `hand_size` cards form a uniform sample
            let mut hand = Hand::EMPTY;
            for i in 0..hand_size {
                let j = self.rng.random_range(i..pool.len());
                pool.swap(i, j);
                hand.add_card(pool[i]);
            }
            let (set, run) = best_strengths(&hand);
            out.open.push(set.max(run));
            out.set.push(set);
            out.run.push(run);
        }
        out.open.sort_unstable();
        out.set.sort_unstable();
        out.run.sort_unstable();
        out
    }
}

impl Agent for EconomistAgent {
    fn name(&self) -> &str {
        "economist"
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        if legal.len() == 1 {
            return legal[0];
        }
        let p = self.params;
        let me = obs.observer();
        let n = obs.num_players();
        let setup = Setup::for_players(n).expect("valid player count");
        let players = obs.players();
        let hand = obs.hand();
        let piles = obs.piles();
        let trick = obs.trick();

        // Seats of the plays still to come in this trick after ours.
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

        // Sampled strength profiles of each opponent still to play.
        let mut pool: ArrayVec<u8, DECK_SIZE> = obs.unknown().cards().collect();
        let mut profiles: [Option<StrengthSamples>; MAX_PLAYERS] = Default::default();
        for &s in &later {
            if s != me && profiles[s].is_none() {
                profiles[s] = Some(self.sample_opponent(&mut pool, players[s].hand_size.into()));
            }
        }

        // Tricks until the game is expected to end, from opponents' remaining cards.
        let lifetime = |s: usize| {
            let cards = f32::from(players[s].hand_size + players[s].reserve_size);
            cards / (p.burn_rate * f32::from(setup.combos_per_turn))
        };
        let mut opp_lives: ArrayVec<f32, MAX_PLAYERS> =
            (0..n).filter(|&s| s != me).map(lifetime).collect();
        opp_lives.sort_unstable_by(f32::total_cmp);
        let tricks_left = if n == 2 {
            opp_lives[0]
        } else {
            opp_lives.get(1).copied().unwrap_or(0.0)
        }
        .max(1.0);
        let my_cards = hand.len() + usize::from(players[me].reserve_size);
        let income = if piles.is_empty() {
            0.0
        } else {
            piles.iter().map(|&v| f32::from(v)).sum::<f32>() / piles.len() as f32
        };
        let max_opp_points = (0..n)
            .filter(|&s| s != me)
            .map(|s| f32::from(players[s].points))
            .fold(0.0, f32::max);
        let someone_out = (0..n).any(|s| s != me && players[s].hand_size == 0);
        let base_structure = structure(&hand, &p);

        let score = |combo: Combo| -> f32 {
            let strength = combo.strength();
            let trick_type = obs.trick_type().after(combo);

            // Distribution over our final rank in the trick.
            let ahead = trick
                .iter()
                .filter(|t| t.combo.strength() >= strength)
                .count();
            let mut dist = [0.0f32; MAX_PLAYS_PER_TRICK + 1];
            dist[0] = 1.0;
            for &s in &later {
                let Some(profile) = &profiles[s] else {
                    continue;
                };
                let beat = p.beat_willingness * profile.prob_beats(trick_type, strength);
                for j in (0..MAX_PLAYS_PER_TRICK).rev() {
                    dist[j + 1] += dist[j] * beat;
                    dist[j] *= 1.0 - beat;
                }
            }
            let expected_pile: f32 = dist
                .iter()
                .enumerate()
                .filter_map(|(j, &pr)| piles.get(ahead + j).map(|&v| pr * f32::from(v)))
                .sum();

            // Card cost: extra cards are cheap while we have more than we need.
            let extra = f32::from(combo.num_cards() - 1);
            let left = (my_cards - combo.num_cards() as usize) as f32;
            let needed = (tricks_left - 1.0) * f32::from(setup.combos_per_turn);
            let deficit = (needed - left).clamp(0.0, extra);
            let card_cost = deficit * p.scarce_card_cost * income + extra * p.spare_card_cost;

            let mut after = hand;
            after.remove_combo(combo);
            let damage = base_structure - structure(&after, &p);

            // Endgame: emptying our cards ends the game in 2-player games, or with 3+ players
            // when someone else is already out.
            let mut endgame = 0.0;
            if after.is_empty() && players[me].reserve_size == 0 && (n == 2 || someone_out) {
                let mine = f32::from(players[me].points) + expected_pile;
                let top_pile = piles.first().map_or(0.0, |&v| f32::from(v));
                let theirs = if n == 2 {
                    let other = 1 - me;
                    // the opponent may still claim this trick, and scores its leftover cards
                    f32::from(players[other].points)
                        + top_pile
                        + f32::from(players[other].hand_size)
                } else {
                    max_opp_points + top_pile
                };
                endgame = if mine > theirs {
                    p.endgame_weight
                } else {
                    -p.endgame_weight
                };
            }

            expected_pile - card_cost - damage + endgame
        };

        let mut best = legal[0];
        let mut best_score = f32::NEG_INFINITY;
        for &combo in legal {
            let s = score(combo);
            if s > best_score {
                best = combo;
                best_score = s;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameState;

    #[test]
    fn best_strengths_by_trick_type() {
        let hand = Hand::from_counts([1, 1, 1, 0, 3, 0]);
        let (set, run) = best_strengths(&hand);
        assert_eq!(set, Combo::set(5, 3).strength());
        assert_eq!(run, Combo::run(1, 3).strength());
    }

    #[test]
    fn plays_legal_moves_for_whole_games() {
        for n in 2..=6 {
            let mut rng = SmallRng::seed_from_u64(n as u64);
            let mut state = GameState::new(n, 0, &mut rng);
            let mut agent = EconomistAgent::new(1);
            while !state.is_over() {
                let seat = state.current_player();
                let combo = agent.act(&state.observe(seat), &state.legal_actions());
                state.apply(combo).unwrap();
            }
        }
    }

    #[test]
    fn takes_an_unbeatable_cheap_win() {
        // Last to play: a single 6 beats the trick's single 5 and takes the 12-point pile.
        let mut rng = SmallRng::seed_from_u64(3);
        let mut state = GameState::new(3, 0, &mut rng);
        while state.current_player() != 2 || state.trick().len() != 2 {
            let legal = state.legal_actions();
            state
                .apply(crate::agents::baseline::lowest_single(&legal))
                .unwrap();
        }
        let obs = state.observe(2);
        let legal = state.legal_actions();
        let choice = EconomistAgent::new(0).act(&obs, &legal);
        let best = obs
            .trick()
            .iter()
            .map(|t| t.combo.strength())
            .max()
            .unwrap();
        if legal.iter().any(|c| c.strength() > best) {
            assert!(choice.strength() > best, "{choice} should win the trick");
        }
    }
}
