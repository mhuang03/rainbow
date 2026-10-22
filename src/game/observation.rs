use arrayvec::ArrayVec;
use rand::seq::SliceRandom;
use rand::{Rng, RngExt, SeedableRng};

use crate::constants::{
    CARDS_PER_RANK, DECK_SIZE, MAX_GAME_PLAYS, MAX_HAND_SIZE, MAX_PILES, MAX_PLAYERS,
    MAX_PLAYS_PER_TRICK, MAX_RANK, MIN_PLAYERS, MIN_RANK, NUM_RANKS,
};
use crate::game::combo::{Combo, TrickType};
use crate::game::hand::Hand;
use crate::game::state::{GameState, Play, PlayerState};

/// Per-rank weights for [`Observation::determinize_weighted`], indexed by absolute seat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DealWeights {
    pub hands: [[f32; NUM_RANKS]; MAX_PLAYERS],
    pub reserves: [[f32; NUM_RANKS]; MAX_PLAYERS],
    /// For the cards set aside in 3-player games.
    pub aside: [f32; NUM_RANKS],
}

impl Default for DealWeights {
    /// Equal weights: the uniform deal.
    fn default() -> Self {
        Self {
            hands: [[1.0; NUM_RANKS]; MAX_PLAYERS],
            reserves: [[1.0; NUM_RANKS]; MAX_PLAYERS],
            aside: [1.0; NUM_RANKS],
        }
    }
}

/// What every player can see about a seat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PublicPlayer {
    pub hand_size: u8,
    /// Face-down reserve cards not yet picked up (2-player only).
    pub reserve_size: u8,
    pub points: u8,
}

/// One player's information set: their own hand plus all public information, assuming
/// perfect memory of every card that has been face up.
///
/// Fields are read-only so an observation always describes a reachable information set,
/// which [`Observation::determinize`] relies on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Observation {
    pub(crate) observer: u8,
    pub(crate) hand: Hand,
    pub(crate) players: ArrayVec<PublicPlayer, MAX_PLAYERS>,
    pub(crate) piles: ArrayVec<u8, MAX_PILES>,
    pub(crate) trick: ArrayVec<Play, MAX_PLAYS_PER_TRICK>,
    pub(crate) trick_type: TrickType,
    pub(crate) trick_size: u8,
    pub(crate) current: u8,
    pub(crate) tricks_played: u8,
    pub(crate) seen: Hand,
    pub(crate) is_over: bool,
    pub(crate) history: ArrayVec<Play, MAX_GAME_PLAYS>,
    pub(crate) initial_middle: Hand,
    pub(crate) first_leader: u8,
}

impl Observation {
    pub fn observer(&self) -> usize {
        self.observer as usize
    }

    pub fn hand(&self) -> Hand {
        self.hand
    }

    pub fn num_players(&self) -> usize {
        self.players.len()
    }

    /// Public information about each seat, indexed by seat.
    pub fn players(&self) -> &[PublicPlayer] {
        &self.players
    }

    pub fn piles(&self) -> &[u8] {
        &self.piles
    }

    pub fn trick(&self) -> &[Play] {
        &self.trick
    }

    pub fn trick_type(&self) -> TrickType {
        self.trick_type
    }

    pub fn current_player(&self) -> usize {
        self.current as usize
    }

    /// Number of plays the current trick will contain.
    pub fn trick_size(&self) -> usize {
        self.trick_size as usize
    }

    pub fn tricks_played(&self) -> usize {
        self.tricks_played as usize
    }

    pub fn is_over(&self) -> bool {
        self.is_over
    }

    /// Every card that has been face up: the initial point cards and every combo played.
    pub fn seen(&self) -> Hand {
        self.seen
    }

    /// Every play so far, in order.
    pub fn history(&self) -> &[Play] {
        &self.history
    }

    /// The face-up point cards dealt before the first trick.
    pub fn initial_middle(&self) -> Hand {
        self.initial_middle
    }

    /// The seat that led the first trick.
    pub fn first_leader(&self) -> usize {
        self.first_leader as usize
    }

    /// Cards `seat` has played so far.
    pub fn played_by(&self, seat: usize) -> Hand {
        let mut hand = Hand::EMPTY;
        for play in self.history.iter().filter(|p| p.seat as usize == seat) {
            hand.add_combo(play.combo);
        }
        hand
    }

    /// Cards the observer cannot locate: opponents' hands, all reserves and set-aside cards.
    pub fn unknown(&self) -> Hand {
        Hand::FULL_DECK
            .difference(&self.seen)
            .difference(&self.hand)
    }

    /// Rebuilds the starting deal of `world`, a determinization of this observation (or the
    /// true state), by returning every played card to the hand or reserve it came from.
    /// Replaying [`Self::history`] from the result reproduces `world`.
    ///
    /// # Panics
    /// If `world` is not consistent with this observation.
    pub fn start_of(&self, world: &GameState) -> GameState {
        let n = self.players.len();
        let setup = crate::constants::Setup::for_players(n).expect("valid player count");
        let mut hands: ArrayVec<Hand, MAX_PLAYERS> = ArrayVec::new();
        let mut reserves: ArrayVec<Hand, MAX_PLAYERS> = ArrayVec::new();
        for seat in 0..n {
            let mut hand = world.players[seat].hand;
            let mut reserve = world.players[seat].reserve;
            let plays = self.history.iter().filter(|p| p.seat as usize == seat);
            if setup.reserve_size > 0 && reserve.is_empty() {
                // The reserve was picked up when the first `hand_size` cards had been played:
                // those plays came from the starting hand, later ones from the reserve.
                let mut initial = Hand::EMPTY;
                for play in plays {
                    if initial.len() < setup.hand_size as usize {
                        initial.add_combo(play.combo);
                    } else {
                        hand.add_combo(play.combo);
                    }
                }
                reserve = hand;
                hand = initial;
            } else {
                for play in plays {
                    hand.add_combo(play.combo);
                }
            }
            hands.push(hand);
            reserves.push(reserve);
        }
        GameState::from_deal(
            &hands,
            &reserves,
            self.initial_middle,
            self.first_leader as usize,
        )
        .expect("world is consistent with the observation")
    }

    /// Samples a full game state consistent with this information set, by dealing the
    /// unknown cards uniformly at random into opponents' hands and all reserves.
    /// This is the determinization step of information-set MCTS.
    pub fn determinize<R: Rng + ?Sized>(&self, rng: &mut R) -> GameState {
        let mut pool: ArrayVec<u8, DECK_SIZE> = self.unknown().cards().collect();
        pool.shuffle(rng);
        let mut pool = pool.into_iter();
        let mut deal = |n: u8| {
            let mut hand = Hand::EMPTY;
            for rank in pool.by_ref().take(n as usize) {
                hand.add_card(rank);
            }
            assert_eq!(hand.len(), n as usize, "observation is inconsistent");
            hand
        };
        let mut hands = [Hand::EMPTY; MAX_PLAYERS];
        let mut reserves = [Hand::EMPTY; MAX_PLAYERS];
        for (seat, p) in self.players.iter().enumerate() {
            if seat != self.observer as usize {
                hands[seat] = deal(p.hand_size);
            }
            reserves[seat] = deal(p.reserve_size);
        }
        self.world(&hands, &reserves)
    }

    /// Like [`Self::determinize`], but each unknown card goes to a hidden hand, reserve or the
    /// set-aside cards with probability proportional to that container's weight for the card's
    /// rank (among containers with room left). Sizes always match the observation. Weights come
    /// from e.g. a hand-prediction network; equal weights give the uniform deal.
    pub fn determinize_weighted<R: Rng + ?Sized>(
        &self,
        weights: &DealWeights,
        rng: &mut R,
    ) -> GameState {
        self.determinize_weighted_with_ratio(weights, rng).0
    }

    /// [`Self::determinize_weighted`], also returning the log importance ratio of the uniform
    /// deal over the weighted one for the sampled cards.
    ///
    /// Shuffling the unknown cards and dealing them in order is the same as sending each card, in
    /// random order, to a container with probability proportional to its remaining room. The
    /// weighted dealer sends it with probability proportional to its weight instead, so the
    /// ratio is the product over cards of `(room / total room) / (weight / total weight)`. Weighting a
    /// belief-proposed deal by this ratio times a likelihood gives the correct posterior weight.
    pub fn determinize_weighted_with_ratio<R: Rng + ?Sized>(
        &self,
        weights: &DealWeights,
        rng: &mut R,
    ) -> (GameState, f32) {
        const FLOOR: f32 = 1e-3;
        let n = self.players.len();
        let observer = self.observer as usize;
        let mut pool: ArrayVec<u8, DECK_SIZE> = self.unknown().cards().collect();
        pool.shuffle(rng);

        // containers: hands of other seats, then every seat's reserve, then the set-aside cards
        let mut room: ArrayVec<usize, { 2 * MAX_PLAYERS + 1 }> = ArrayVec::new();
        let mut rank_weights: ArrayVec<&[f32; NUM_RANKS], { 2 * MAX_PLAYERS + 1 }> =
            ArrayVec::new();
        let mut owner: ArrayVec<(usize, bool), { 2 * MAX_PLAYERS + 1 }> = ArrayVec::new();
        for seat in (0..n).filter(|&s| s != observer) {
            room.push(self.players[seat].hand_size as usize);
            rank_weights.push(&weights.hands[seat]);
            owner.push((seat, false));
        }
        for seat in 0..n {
            room.push(self.players[seat].reserve_size as usize);
            rank_weights.push(&weights.reserves[seat]);
            owner.push((seat, true));
        }
        let hidden: usize = room.iter().sum();
        assert!(hidden <= pool.len(), "observation is inconsistent");
        room.push(pool.len() - hidden);
        rank_weights.push(&weights.aside);

        let mut hands = [Hand::EMPTY; MAX_PLAYERS];
        let mut reserves = [Hand::EMPTY; MAX_PLAYERS];
        let mut log_ratio = 0.0f64;
        let mut room_left = pool.len();
        for &rank in &pool {
            let r = (rank - MIN_RANK) as usize;
            let w = |c: usize| {
                if room[c] > 0 {
                    rank_weights[c][r].max(0.0) + FLOOR
                } else {
                    0.0
                }
            };
            let total: f32 = (0..room.len()).map(w).sum();
            let mut x = rng.random::<f32>() * total;
            let mut chosen = (0..room.len())
                .rev()
                .find(|&c| room[c] > 0)
                .expect("room for every card");
            for c in 0..room.len() {
                let wc = w(c);
                if x < wc {
                    chosen = c;
                    break;
                }
                x -= wc;
            }
            let uniform = room[chosen] as f64 / room_left as f64;
            let weighted = f64::from(w(chosen)) / f64::from(total);
            log_ratio += (uniform / weighted).ln();
            room[chosen] -= 1;
            room_left -= 1;
            if let Some(&(seat, reserve)) = owner.get(chosen) {
                if reserve {
                    reserves[seat].add_card(rank);
                } else {
                    hands[seat].add_card(rank);
                }
            }
        }
        (self.world(&hands, &reserves), log_ratio as f32)
    }

    /// The full state with the given hidden hands and reserves (the observer's hand is known).
    fn world(&self, hands: &[Hand; MAX_PLAYERS], reserves: &[Hand; MAX_PLAYERS]) -> GameState {
        let players = self
            .players
            .iter()
            .enumerate()
            .map(|(seat, p)| PlayerState {
                hand: if seat == self.observer as usize {
                    self.hand
                } else {
                    hands[seat]
                },
                reserve: reserves[seat],
                points: p.points,
            })
            .collect();

        GameState {
            players,
            piles: self.piles.clone(),
            trick: self.trick.clone(),
            trick_type: self.trick_type,
            trick_size: self.trick_size,
            current: self.current,
            tricks_played: self.tricks_played,
            seen: self.seen,
            is_over: self.is_over,
            history: self.history.clone(),
            initial_middle: self.initial_middle,
            first_leader: self.first_leader,
        }
    }

    /// Writes the fixed-size feature vector described in [`encoding`] into `out`.
    ///
    /// Seats are relative to the observer (slot 0 is the observer, slot 1 the next player
    /// clockwise, and so on), so one network can play every seat. All features lie in `[0, 1]`.
    ///
    /// # Panics
    /// If `out.len() != OBS_SIZE`.
    pub fn encode(&self, out: &mut [f32]) {
        use encoding::*;
        assert_eq!(out.len(), OBS_SIZE, "observation buffer has the wrong size");
        out.fill(0.0);
        let n = self.players.len();
        let rel = |seat: u8| (seat as usize + n - self.observer as usize) % n;
        let cards = |out: &mut [f32], hand: Hand| {
            for (o, c) in out.iter_mut().zip(hand.counts()) {
                *o = f32::from(c) / f32::from(CARDS_PER_RANK);
            }
        };

        cards(&mut out[HAND..HAND + NUM_RANKS], self.hand);
        cards(&mut out[UNKNOWN..UNKNOWN + NUM_RANKS], self.unknown());

        for (seat, p) in self.players.iter().enumerate() {
            let o = &mut out[SEATS + rel(seat as u8) * SEAT_FEATURES..][..SEAT_FEATURES];
            o[0] = 1.0;
            o[1] = f32::from(p.hand_size) / MAX_HAND_SIZE as f32;
            o[2] = f32::from(p.reserve_size) / MAX_HAND_SIZE as f32;
            o[3] = f32::from(p.points) / POINTS_SCALE;
            o[4] = f32::from(!self.is_over && seat == self.current as usize);
        }

        for (o, &v) in out[PILES..PILES + MAX_PILES].iter_mut().zip(&self.piles) {
            *o = f32::from(v) / f32::from(2 * MAX_RANK);
        }

        for (i, play) in self.trick.iter().enumerate() {
            let o = &mut out[TRICK + i * PLAY_FEATURES..][..PLAY_FEATURES];
            o[0] = 1.0;
            o[1 + rel(play.seat)] = 1.0;
            for (j, rank) in (MIN_RANK..=MAX_RANK).enumerate() {
                o[1 + MAX_PLAYERS + j] =
                    f32::from(play.combo.count_of(rank)) / f32::from(CARDS_PER_RANK);
            }
        }

        out[TRICK_TYPE + self.trick_type as usize] = 1.0;
        out[NUM_PLAYERS + n - MIN_PLAYERS] = 1.0;
        out[PROGRESS] = self.trick.len() as f32 / MAX_PLAYS_PER_TRICK as f32;
        out[PROGRESS + 1] = f32::from(self.tricks_played) / TRICKS_SCALE;

        let scale = 1.0 / f32::from(CARDS_PER_RANK);
        for play in &self.history {
            let o = &mut out[PLAYED + rel(play.seat) * NUM_RANKS..][..NUM_RANKS];
            for (x, rank) in o.iter_mut().zip(MIN_RANK..=MAX_RANK) {
                *x += f32::from(play.combo.count_of(rank)) * scale;
            }
        }
    }

    /// Writes the public-history features described in [`encoding::HISTORY`] into `out`: the
    /// most recent plays, and how often each seat beat the trick when following.
    ///
    /// # Panics
    /// If `out.len() != HISTORY_SIZE`.
    pub fn encode_history(&self, out: &mut [f32]) {
        use encoding::*;
        assert_eq!(out.len(), HISTORY_SIZE, "history buffer has the wrong size");
        out.fill(0.0);
        let n = self.players.len();
        let rel = |seat: u8| (seat as usize + n - self.observer as usize) % n;
        // Trick boundaries depend only on public hand sizes, so replaying the history from any
        // consistent deal finds them; a fixed seed keeps the encoding deterministic.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let mut state = self.start_of(&self.determinize(&mut rng));
        let mut follows = [0u32; MAX_PLAYERS];
        let mut beats = [0u32; MAX_PLAYERS];
        let recent = self.history.len().saturating_sub(HISTORY_PLAYS);
        for (i, play) in self.history.iter().enumerate() {
            let best = state.trick().iter().map(|t| t.combo.strength()).max();
            let lead = best.is_none();
            let beat = best.is_some_and(|b| play.combo.strength() > b);
            let seat = rel(play.seat);
            if !lead {
                follows[seat] += 1;
                beats[seat] += u32::from(beat);
            }
            if i >= recent {
                // newest play first
                let slot = self.history.len() - 1 - i;
                let o = &mut out[slot * HISTORY_FEATURES..][..HISTORY_FEATURES];
                o[0] = 1.0;
                o[1] = f32::from(lead);
                o[2] = f32::from(beat);
                o[3 + seat] = 1.0;
                for (x, rank) in o[3 + MAX_PLAYERS..].iter_mut().zip(MIN_RANK..=MAX_RANK) {
                    *x = f32::from(play.combo.count_of(rank)) / f32::from(CARDS_PER_RANK);
                }
            }
            state.apply(play.combo).expect("history replays legally");
        }
        let rates = &mut out[HISTORY_PLAYS * HISTORY_FEATURES..];
        for seat in 0..n {
            if follows[seat] > 0 {
                rates[2 * seat] = beats[seat] as f32 / follows[seat] as f32;
            }
            rates[2 * seat + 1] = (follows[seat] as f32 / TRICKS_SCALE).min(1.0);
        }
    }

    /// [`Self::encode`] followed by [`Self::encode_history`], for history-aware networks.
    ///
    /// # Panics
    /// If `out.len() != HIST_OBS_SIZE`.
    pub fn encode_with_history(&self, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            encoding::HIST_OBS_SIZE,
            "observation buffer has the wrong size"
        );
        let (obs, history) = out.split_at_mut(encoding::OBS_SIZE);
        self.encode(obs);
        self.encode_history(history);
    }

    /// Allocating convenience wrapper around [`Self::encode`].
    pub fn encoded(&self) -> [f32; encoding::OBS_SIZE] {
        let mut out = [0.0; encoding::OBS_SIZE];
        self.encode(&mut out);
        out
    }
}

/// Layout of the feature vector written by [`Observation::encode`]. Each constant is the
/// offset of a block.
pub mod encoding {
    use super::*;

    /// Own hand: count per rank / 10.
    pub const HAND: usize = 0;
    /// Unknown cards (opponents' hands, reserves, set-aside): count per rank / 10.
    pub const UNKNOWN: usize = HAND + NUM_RANKS;
    /// Per relative seat: present, hand size / 14, reserve size / 14, points / 224, is current.
    pub const SEATS: usize = UNKNOWN + NUM_RANKS;
    pub const SEAT_FEATURES: usize = 5;
    /// Point pile values / 12, highest first, zero-padded.
    pub const PILES: usize = SEATS + MAX_PLAYERS * SEAT_FEATURES;
    /// Per play in the current trick: present, relative seat one-hot, card count per rank / 10.
    pub const TRICK: usize = PILES + MAX_PILES;
    pub const PLAY_FEATURES: usize = 1 + MAX_PLAYERS + NUM_RANKS;
    /// One-hot: open, set, run.
    pub const TRICK_TYPE: usize = TRICK + MAX_PLAYS_PER_TRICK * PLAY_FEATURES;
    /// One-hot player count, 2 to 6.
    pub const NUM_PLAYERS: usize = TRICK_TYPE + 3;
    /// Plays so far in this trick / 6, tricks completed / 30.
    pub const PROGRESS: usize = NUM_PLAYERS + MAX_PLAYERS - MIN_PLAYERS + 1;
    /// Per relative seat: cards played so far this game, count per rank / 10.
    pub const PLAYED: usize = PROGRESS + 2;
    pub const OBS_SIZE: usize = PLAYED + MAX_PLAYERS * NUM_RANKS;

    /// Upper bound on a score: every card's value plus the 2-player end-of-game bonus.
    pub(super) const POINTS_SCALE: f32 = (CARDS_PER_RANK as usize
        * (MAX_RANK as usize * (MAX_RANK as usize + 1) / 2)
        + MAX_HAND_SIZE) as f32;
    /// Upper bound on tricks: every trick consumes at least two cards.
    pub(super) const TRICKS_SCALE: f32 = (DECK_SIZE / 2) as f32;

    /// Features per action written by [`encode_action`]: card count per rank / 10, one-hot
    /// single/set/run, card count / 10, strength / max strength.
    pub const ACTION_FEATURES: usize = NUM_RANKS + 3 + 2;

    /// Writes `combo`'s features, for networks that score (observation, action) pairs.
    ///
    /// # Panics
    /// If `out.len() != ACTION_FEATURES`.
    pub fn encode_action(combo: Combo, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            ACTION_FEATURES,
            "action buffer has the wrong size"
        );
        out.fill(0.0);
        for (o, rank) in out.iter_mut().zip(MIN_RANK..=MAX_RANK) {
            *o = f32::from(combo.count_of(rank)) / f32::from(CARDS_PER_RANK);
        }
        out[NUM_RANKS + combo.trick_type() as usize] = 1.0;
        out[NUM_RANKS + 3] = f32::from(combo.num_cards()) / f32::from(CARDS_PER_RANK);
        out[NUM_RANKS + 4] = f32::from(combo.strength()) / f32::from(MAX_STRENGTH);
    }

    /// Hidden information for full-information critics: per relative seat, hand then reserve,
    /// card count per rank / 10.
    pub const HIDDEN_SIZE: usize = MAX_PLAYERS * 2 * NUM_RANKS;
    /// [`OBS_SIZE`] observation features followed by [`HIDDEN_SIZE`] hidden features.
    pub const FULL_OBS_SIZE: usize = OBS_SIZE + HIDDEN_SIZE;

    /// Public history, written by [`Observation::encode_history`]: the last [`HISTORY_PLAYS`]
    /// plays, newest first, each as present, led the trick, beat the trick so far, relative seat
    /// one-hot and card count per rank / 10; then per relative seat, the fraction of its
    /// following plays that beat the trick, and its following plays / 30.
    pub const HISTORY: usize = 0;
    pub const HISTORY_PLAYS: usize = 12;
    pub const HISTORY_FEATURES: usize = 3 + MAX_PLAYERS + NUM_RANKS;
    pub const HISTORY_SIZE: usize = HISTORY_PLAYS * HISTORY_FEATURES + 2 * MAX_PLAYERS;
    /// [`OBS_SIZE`] observation features followed by [`HISTORY_SIZE`] history features.
    pub const HIST_OBS_SIZE: usize = OBS_SIZE + HISTORY_SIZE;

    /// Strength of the strongest combo, ten 6s.
    const MAX_STRENGTH: u8 = Combo::set(MAX_RANK, CARDS_PER_RANK).strength();
}

#[cfg(test)]
mod tests {
    use super::encoding::OBS_SIZE;
    use super::*;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    /// Plays `moves` random moves into a new game.
    fn midgame(seed: u64, num_players: usize, moves: usize) -> GameState {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut g = GameState::new(num_players, 0, &mut rng);
        for _ in 0..moves {
            if g.is_over() {
                break;
            }
            let actions = g.legal_actions();
            g.apply(actions[rng.random_range(0..actions.len())])
                .unwrap();
        }
        g
    }

    #[test]
    fn hides_opponent_hands() {
        let g = midgame(1, 4, 5);
        let obs = g.observe(2);
        assert_eq!(obs.hand(), g.hand(2));
        for (seat, p) in obs.players().iter().enumerate() {
            assert_eq!(p.hand_size as usize, g.hand(seat).len());
            assert_eq!(p.points, g.points(seat));
        }
    }

    #[test]
    fn importance_ratio_corrects_a_biased_deal() {
        let g = GameState::new(4, 0, &mut SmallRng::seed_from_u64(3));
        let obs = g.observe(0);
        let mut rng = SmallRng::seed_from_u64(5);
        // A biased dealer over-supplies 1s to seat 2; weighting each deal by its importance ratio
        // must recover the uniform deal's average.
        let mut weights = DealWeights::default();
        weights.hands[2] = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let (mut weighted_sum, mut weight_total, mut uniform_sum) = (0.0f64, 0.0f64, 0.0f64);
        let trials = 4000;
        for _ in 0..trials {
            let (d, log_ratio) = obs.determinize_weighted_with_ratio(&weights, &mut rng);
            let w = f64::from(log_ratio).exp();
            weighted_sum += w * f64::from(d.hand(2).count(1));
            weight_total += w;
            uniform_sum += f64::from(obs.determinize(&mut rng).hand(2).count(1));
        }
        let corrected = weighted_sum / weight_total;
        let uniform = uniform_sum / f64::from(trials);
        assert!(
            (corrected - uniform).abs() < 0.25,
            "{corrected} vs {uniform}"
        );
    }

    #[test]
    fn weights_bias_the_deal() {
        let g = GameState::new(4, 0, &mut SmallRng::seed_from_u64(3));
        let obs = g.observe(0);
        let mut weights = DealWeights::default();
        weights.hands[2] = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng = SmallRng::seed_from_u64(4);
        let (mut biased, mut uniform) = (0, 0);
        for _ in 0..200 {
            biased += obs
                .determinize_weighted(&weights, &mut rng)
                .hand(2)
                .count(1) as u32;
            uniform += obs.determinize(&mut rng).hand(2).count(1) as u32;
        }
        assert!(biased > uniform * 2, "{biased} vs {uniform}");
    }

    #[test]
    fn three_player_unknown_includes_set_aside() {
        let g = GameState::new(3, 0, &mut SmallRng::seed_from_u64(0));
        let obs = g.observe(0);
        assert_eq!(obs.unknown().len(), 2 * 14 + 15);
    }

    #[test]
    fn action_features_are_bounded_and_distinct() {
        use crate::game::ALL_COMBOS;
        use encoding::{ACTION_FEATURES, encode_action};
        let mut seen = std::collections::HashSet::new();
        for &combo in &ALL_COMBOS {
            let mut f = [0.0; ACTION_FEATURES];
            encode_action(combo, &mut f);
            assert!(f.iter().all(|x| (0.0..=1.0).contains(x)));
            assert!(seen.insert(f.map(f32::to_bits)));
        }
    }

    #[test]
    fn encoding_layout_is_contiguous() {
        assert_eq!(OBS_SIZE, 172);
    }

    proptest! {
        #[test]
        fn determinization_is_consistent(
            seed in any::<u64>(),
            num_players in MIN_PLAYERS..=MAX_PLAYERS,
            moves in 0..40usize,
        ) {
            let g = midgame(seed, num_players, moves);
            let mut rng = SmallRng::seed_from_u64(seed ^ 0xABCD);
            for seat in 0..num_players {
                let obs = g.observe(seat);
                let d = obs.determinize(&mut rng);
                // the sampled world looks identical from the observer's seat
                prop_assert_eq!(&d.observe(seat), &obs);
                prop_assert_eq!(!d.legal_actions().is_empty(), !g.is_over());
            }
        }

        #[test]
        fn weighted_determinization_is_consistent(
            seed in any::<u64>(),
            num_players in MIN_PLAYERS..=MAX_PLAYERS,
            moves in 0..40usize,
        ) {
            let g = midgame(seed, num_players, moves);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut weights = DealWeights::default();
            weights.hands[1] = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            for seat in 0..num_players {
                let obs = g.observe(seat);
                let d = obs.determinize_weighted(&weights, &mut rng);
                prop_assert_eq!(&d.observe(seat), &obs);
            }
        }

        #[test]
        fn history_encoding_is_public(
            seed in any::<u64>(),
            num_players in MIN_PLAYERS..=MAX_PLAYERS,
            moves in 0..70usize,
        ) {
            use super::encoding::HISTORY_SIZE;
            let g = midgame(seed, num_players, moves);
            let mut rng = SmallRng::seed_from_u64(seed);
            let seat = seed as usize % num_players;
            let mut a = [0.0; HISTORY_SIZE];
            g.observe(seat).encode_history(&mut a);
            prop_assert!(a.iter().all(|&x| (0.0..=1.0).contains(&x)));
            // the same for any deal the observer can't tell apart from this one
            let mut b = [0.0; HISTORY_SIZE];
            g.observe(seat).determinize(&mut rng).observe(seat).encode_history(&mut b);
            prop_assert_eq!(a, b);
            // one present slot per recent play
            let played = g.observe(seat).history().len().min(super::encoding::HISTORY_PLAYS);
            let present = a.chunks(super::encoding::HISTORY_FEATURES).take(super::encoding::HISTORY_PLAYS);
            prop_assert_eq!(present.filter(|o| o[0] == 1.0).count(), played);
        }

        #[test]
        fn start_of_replays_to_the_world(
            seed in any::<u64>(),
            num_players in MIN_PLAYERS..=MAX_PLAYERS,
            moves in 0..70usize,
        ) {
            let g = midgame(seed, num_players, moves);
            let mut rng = SmallRng::seed_from_u64(seed);
            let obs = g.observe(seed as usize % num_players);
            for world in [g.clone(), obs.determinize(&mut rng)] {
                let mut replay = obs.start_of(&world);
                for play in obs.history() {
                    replay.apply(play.combo).unwrap();
                }
                prop_assert_eq!(replay, world);
            }
        }

        #[test]
        fn encoding_is_bounded(
            seed in any::<u64>(),
            num_players in MIN_PLAYERS..=MAX_PLAYERS,
            moves in 0..40usize,
        ) {
            let g = midgame(seed, num_players, moves);
            for seat in 0..num_players {
                let v = g.observe(seat).encoded();
                prop_assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
            }
        }
    }
}
