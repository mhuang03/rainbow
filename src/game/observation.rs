use arrayvec::ArrayVec;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::constants::{
    CARDS_PER_RANK, DECK_SIZE, MAX_HAND_SIZE, MAX_PILES, MAX_PLAYERS, MAX_PLAYS_PER_TRICK,
    MAX_RANK, MIN_PLAYERS, MIN_RANK, NUM_RANKS,
};
use crate::game::combo::TrickType;
use crate::game::hand::Hand;
use crate::game::state::{GameState, Play, PlayerState};

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

    /// Cards the observer cannot locate: opponents' hands, all reserves and set-aside cards.
    pub fn unknown(&self) -> Hand {
        Hand::FULL_DECK
            .difference(&self.seen)
            .difference(&self.hand)
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

        let players = self
            .players
            .iter()
            .enumerate()
            .map(|(seat, p)| PlayerState {
                hand: if seat == self.observer as usize {
                    self.hand
                } else {
                    deal(p.hand_size)
                },
                reserve: deal(p.reserve_size),
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
    pub const OBS_SIZE: usize = PROGRESS + 2;

    /// Upper bound on a score: every card's value plus the 2-player end-of-game bonus.
    pub(super) const POINTS_SCALE: f32 = (CARDS_PER_RANK as usize
        * (MAX_RANK as usize * (MAX_RANK as usize + 1) / 2)
        + MAX_HAND_SIZE) as f32;
    /// Upper bound on tricks: every trick consumes at least two cards.
    pub(super) const TRICKS_SCALE: f32 = (DECK_SIZE / 2) as f32;
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
    fn three_player_unknown_includes_set_aside() {
        let g = GameState::new(3, 0, &mut SmallRng::seed_from_u64(0));
        let obs = g.observe(0);
        assert_eq!(obs.unknown().len(), 2 * 14 + 15);
    }

    #[test]
    fn encoding_layout_is_contiguous() {
        assert_eq!(OBS_SIZE, 136);
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
