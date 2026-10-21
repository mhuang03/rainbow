use std::fmt;

use arrayvec::ArrayVec;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::constants::{
    CARDS_PER_RANK, DECK_SIZE, EMPTY_HANDS_TO_END_GAME, MAX_GAME_PLAYS, MAX_LEGAL_ACTIONS,
    MAX_PILES, MAX_PLAYERS, MAX_PLAYS_PER_TRICK, MAX_RANK, MIN_PLAYERS, MIN_RANK, Setup,
};
use crate::game::combo::{Combo, TrickType};
use crate::game::hand::Hand;
use crate::game::observation::{Observation, PublicPlayer};

/// A combo played by the player in seat `seat`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Play {
    pub seat: u8,
    pub combo: Combo,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PlayerState {
    pub(crate) hand: Hand,
    /// Face-down cards picked up when the hand first empties. Always empty with 3+ players.
    pub(crate) reserve: Hand,
    /// At most 224 (see `encoding::POINTS_SCALE`), so it fits in a `u8`.
    pub(crate) points: u8,
}

/// Why an action was rejected by [`GameState::apply`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionError {
    GameOver,
    /// The current trick type forbids this combo (e.g. a run after a set).
    WrongTrickType {
        combo: Combo,
        trick_type: TrickType,
    },
    /// The current player does not hold the combo's cards.
    CardsNotInHand {
        combo: Combo,
    },
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GameOver => write!(f, "the game is over"),
            Self::WrongTrickType { combo, trick_type } => {
                write!(f, "{combo} cannot be played in a {trick_type:?} trick")
            }
            Self::CardsNotInHand { combo } => write!(f, "{combo} is not in the player's hand"),
        }
    }
}

impl std::error::Error for ActionError {}

/// The complete, perfect-information state of a game.
///
/// Players are identified by seat index `0..num_players`, in clockwise order. The state is
/// `Copy`-sized (no heap data), so cloning it for tree search is cheap.
///
/// Rules follow the official rulebook. Two decisions are made automatically because they
/// cannot change the outcome in the player's favour:
/// - claiming point piles: each player takes the most valuable pile left, since claimed
///   piles affect nothing but the claimant's score;
/// - 3-player leftover cards are set aside and never used.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GameState {
    pub(crate) players: ArrayVec<PlayerState, MAX_PLAYERS>,
    /// Point pile values in the middle, sorted in descending order.
    pub(crate) piles: ArrayVec<u8, MAX_PILES>,
    pub(crate) trick: ArrayVec<Play, MAX_PLAYS_PER_TRICK>,
    pub(crate) trick_type: TrickType,
    /// Number of plays the current trick will contain.
    pub(crate) trick_size: u8,
    pub(crate) current: u8,
    pub(crate) tricks_played: u8,
    /// Every card that has been face up: initial point cards and all played combos.
    pub(crate) seen: Hand,
    pub(crate) is_over: bool,
    /// Every play so far, in order.
    pub(crate) history: ArrayVec<Play, MAX_GAME_PLAYS>,
    /// The face-up point cards dealt before the first trick.
    pub(crate) initial_middle: Hand,
    /// The seat that led the first trick.
    pub(crate) first_leader: u8,
}

impl GameState {
    /// Deals a new game for `num_players` (2 to 6), with `leader` playing first.
    ///
    /// # Panics
    /// If `num_players` is out of range or `leader >= num_players`.
    pub fn new<R: Rng + ?Sized>(num_players: usize, leader: usize, rng: &mut R) -> Self {
        let setup = Setup::for_players(num_players).unwrap_or_else(|| {
            panic!("Rainbow supports {MIN_PLAYERS}-{MAX_PLAYERS} players, got {num_players}")
        });
        assert!(leader < num_players, "leader {leader} is not a seat");

        let mut deck: [u8; DECK_SIZE] =
            std::array::from_fn(|i| MIN_RANK + (i % MAX_RANK as usize) as u8);
        deck.shuffle(rng);
        let mut deck = deck.into_iter();
        let mut deal = |n: u8| {
            let mut hand = Hand::EMPTY;
            for rank in deck.by_ref().take(n as usize) {
                hand.add_card(rank);
            }
            hand
        };

        let middle = deal(setup.middle_cards);
        let hands: ArrayVec<Hand, MAX_PLAYERS> =
            (0..num_players).map(|_| deal(setup.hand_size)).collect();
        let reserves: ArrayVec<Hand, MAX_PLAYERS> =
            (0..num_players).map(|_| deal(setup.reserve_size)).collect();
        Self::start(&hands, &reserves, middle, leader)
    }

    /// Starts a game from an explicit deal: one hand and one reserve per seat (reserves are
    /// empty except in 2-player games), the face-up point cards, and the first leader.
    ///
    /// # Errors
    /// If the player count is unsupported, the sizes don't match the official setup, or the deal
    /// uses more copies of a rank than the deck has.
    pub fn from_deal(
        hands: &[Hand],
        reserves: &[Hand],
        middle: Hand,
        leader: usize,
    ) -> Result<Self, String> {
        let n = hands.len();
        let setup = Setup::for_players(n)
            .ok_or_else(|| format!("Rainbow supports {MIN_PLAYERS}-{MAX_PLAYERS} players"))?;
        if reserves.len() != n || leader >= n {
            return Err("need one reserve per hand and a valid leader".to_owned());
        }
        if middle.len() != setup.middle_cards as usize
            || hands.iter().any(|h| h.len() != setup.hand_size as usize)
            || reserves
                .iter()
                .any(|r| r.len() != setup.reserve_size as usize)
        {
            return Err("hand, reserve or middle sizes don't match the setup".to_owned());
        }
        let mut used = middle.counts();
        for h in hands.iter().chain(reserves) {
            for (u, c) in used.iter_mut().zip(h.counts()) {
                *u += c;
            }
        }
        if used.iter().any(|&u| u > CARDS_PER_RANK) {
            return Err("the deal uses a rank more than 10 times".to_owned());
        }
        Ok(Self::start(hands, reserves, middle, leader))
    }

    fn start(hands: &[Hand], reserves: &[Hand], middle: Hand, leader: usize) -> Self {
        let n = hands.len();
        let setup = Setup::for_players(n).expect("validated by callers");
        Self {
            players: hands
                .iter()
                .zip(reserves)
                .map(|(&hand, &reserve)| PlayerState {
                    hand,
                    reserve,
                    points: 0,
                })
                .collect(),
            piles: middle.cards().rev().collect(),
            trick: ArrayVec::new(),
            trick_type: TrickType::Open,
            trick_size: n as u8 * setup.combos_per_turn,
            current: leader as u8,
            tricks_played: 0,
            seen: middle,
            is_over: false,
            history: ArrayVec::new(),
            initial_middle: middle,
            first_leader: leader as u8,
        }
    }

    pub fn num_players(&self) -> usize {
        self.players.len()
    }

    fn setup(&self) -> Setup {
        Setup::for_players(self.num_players()).expect("player count validated on construction")
    }

    /// The seat whose turn it is. Meaningless once the game is over.
    pub fn current_player(&self) -> usize {
        self.current as usize
    }

    pub fn is_over(&self) -> bool {
        self.is_over
    }

    pub fn trick_type(&self) -> TrickType {
        self.trick_type
    }

    /// Plays made so far in the current trick, in order.
    pub fn trick(&self) -> &[Play] {
        &self.trick
    }

    /// Values of the point piles in the middle, highest first.
    pub fn piles(&self) -> &[u8] {
        &self.piles
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

    pub fn tricks_played(&self) -> usize {
        self.tricks_played as usize
    }

    /// The cards in `seat`'s hand. Hidden information: agents should use [`Self::observe`].
    pub fn hand(&self, seat: usize) -> Hand {
        self.players[seat].hand
    }

    /// The face-down reserve of `seat` (2-player games only). Hidden information.
    pub fn reserve(&self, seat: usize) -> Hand {
        self.players[seat].reserve
    }

    pub fn points(&self, seat: usize) -> u8 {
        self.players[seat].points
    }

    pub fn scores(&self) -> ArrayVec<u8, MAX_PLAYERS> {
        self.players.iter().map(|p| p.points).collect()
    }

    /// Seats with the highest score (ties share victory).
    pub fn winners(&self) -> ArrayVec<u8, MAX_PLAYERS> {
        let best = self.players.iter().map(|p| p.points).max().unwrap_or(0);
        (0..self.players.len() as u8)
            .filter(|&s| self.players[s as usize].points == best)
            .collect()
    }

    /// Terminal rewards: winners split 1.0 equally, everyone else gets 0.0.
    /// Returns zeros while the game is in progress.
    pub fn returns(&self) -> ArrayVec<f32, MAX_PLAYERS> {
        let mut returns: ArrayVec<f32, MAX_PLAYERS> = self.players.iter().map(|_| 0.0).collect();
        if self.is_over {
            let winners = self.winners();
            for &w in &winners {
                returns[w as usize] = 1.0 / winners.len() as f32;
            }
        }
        returns
    }

    /// Whether the current player may play `combo`.
    pub fn is_legal(&self, combo: Combo) -> bool {
        self.check(combo).is_ok()
    }

    fn check(&self, combo: Combo) -> Result<(), ActionError> {
        if self.is_over {
            Err(ActionError::GameOver)
        } else if !self.trick_type.allows(combo) {
            Err(ActionError::WrongTrickType {
                combo,
                trick_type: self.trick_type,
            })
        } else if !self.players[self.current as usize].hand.contains(combo) {
            Err(ActionError::CardsNotInHand { combo })
        } else {
            Ok(())
        }
    }

    /// Calls `f` with each legal combo for the current player, ordered by rank.
    fn for_each_legal(&self, mut f: impl FnMut(Combo)) {
        if self.is_over {
            return;
        }
        let hand = &self.players[self.current as usize].hand;
        let sets = self.trick_type != TrickType::Run;
        let runs = self.trick_type != TrickType::Set;
        for rank in MIN_RANK..=MAX_RANK {
            let count = hand.count(rank);
            if count == 0 {
                continue;
            }
            f(Combo::single(rank));
            if sets {
                (2..=count).for_each(|n| f(Combo::set(rank, n)));
            }
            if runs {
                (rank + 1..=MAX_RANK)
                    .take_while(|&end| hand.count(end) > 0)
                    .for_each(|end| f(Combo::run(rank, end)));
            }
        }
    }

    /// Legal combos for the current player. Empty once the game is over.
    pub fn legal_actions(&self) -> ArrayVec<Combo, MAX_LEGAL_ACTIONS> {
        let mut actions = ArrayVec::new();
        self.for_each_legal(|c| actions.push(c));
        actions
    }

    /// Legal actions as a bitmask over action indices (bit `i` set if
    /// `Combo::from_action_index(i)` is legal).
    pub fn legal_action_mask(&self) -> u128 {
        let mut mask = 0;
        self.for_each_legal(|c| mask |= 1 << c.action_index());
        mask
    }

    /// Plays `combo` for the current player and advances the game.
    pub fn apply(&mut self, combo: Combo) -> Result<(), ActionError> {
        self.check(combo)?;

        let seat = self.current;
        let player = &mut self.players[seat as usize];
        player.hand.remove_combo(combo);
        self.seen.add_combo(combo);
        self.trick_type = self.trick_type.after(combo);
        self.trick.push(Play { seat, combo });
        self.history.push(Play { seat, combo });

        if player.hand.is_empty() {
            if !player.reserve.is_empty() {
                player.hand = std::mem::take(&mut player.reserve);
            } else if self.setup().reserve_size > 0 {
                // 2-player: running out a second time ends the game at once. The plays made
                // so far still score, and the opponent gets 1 point per card left in hand.
                self.score_trick();
                let other = &mut self.players[1 - seat as usize];
                other.points += other.hand.len() as u8;
                self.is_over = true;
                return Ok(());
            }
        }

        if self.trick.len() == self.trick_size as usize {
            self.finish_trick();
        } else {
            self.current = self.next_seat_with_cards(seat + 1);
        }
        Ok(())
    }

    /// The first seat at or clockwise after `from` that holds cards.
    fn next_seat_with_cards(&self, from: u8) -> u8 {
        let n = self.players.len() as u8;
        (0..n)
            .map(|i| (from + i) % n)
            .find(|&s| !self.players[s as usize].hand.is_empty())
            .expect("a game in progress has a player with cards")
    }

    /// Hands out point piles for the current trick and returns the winning seat.
    fn score_trick(&mut self) -> u8 {
        // Stable insertion sort by descending strength: earlier plays win ties.
        let mut order: ArrayVec<Play, MAX_PLAYS_PER_TRICK> = self.trick.clone();
        for i in 1..order.len() {
            let mut j = i;
            while j > 0 && order[j - 1].combo.strength() < order[j].combo.strength() {
                order.swap(j - 1, j);
                j -= 1;
            }
        }
        for (play, &pile) in order.iter().zip(&self.piles) {
            self.players[play.seat as usize].points += pile;
        }
        self.tricks_played += 1;
        order[0].seat
    }

    fn finish_trick(&mut self) {
        let winner = self.score_trick();

        let empty = self.players.iter().filter(|p| p.hand.is_empty()).count();
        if empty >= EMPTY_HANDS_TO_END_GAME {
            self.is_over = true;
            return;
        }
        let active = (self.players.len() - empty) as u8;
        self.trick_size = active * self.setup().combos_per_turn;

        // The played cards become the next point piles: pairs of matching rank, then
        // leftover singles, keeping only the most valuable piles if there are too many.
        let mut played = Hand::EMPTY;
        for play in &self.trick {
            played.add_combo(play.combo);
        }
        self.piles.clear();
        let limit = self.trick_size as usize;
        for value in (1..=2 * MAX_RANK).rev() {
            let pairs = if value % 2 == 0 {
                played.count(value / 2) / 2
            } else {
                0
            };
            let single = if value <= MAX_RANK {
                played.count(value) % 2
            } else {
                0
            };
            for _ in 0..pairs + single {
                if self.piles.len() == limit {
                    break;
                }
                self.piles.push(value);
            }
        }

        self.trick.clear();
        self.trick_type = TrickType::Open;
        self.current = self.next_seat_with_cards(winner);
    }

    /// `seat`'s observation encoding followed by every hand and reserve, with seats relative to
    /// `seat`: features for a value network trained with perfect information (see
    /// [`encoding::FULL_OBS_SIZE`](crate::game::encoding::FULL_OBS_SIZE)). Agents must not act
    /// on this.
    ///
    /// # Panics
    /// If `out.len()` is not `FULL_OBS_SIZE`.
    pub fn encode_full(&self, seat: usize, out: &mut [f32]) {
        use crate::constants::NUM_RANKS;
        use crate::game::encoding::{FULL_OBS_SIZE, OBS_SIZE};
        assert_eq!(out.len(), FULL_OBS_SIZE, "buffer has the wrong size");
        let (obs, hidden) = out.split_at_mut(OBS_SIZE);
        self.observe(seat).encode(obs);
        hidden.fill(0.0);
        let n = self.players.len();
        for (s, p) in self.players.iter().enumerate() {
            let rel = (s + n - seat) % n;
            let block = &mut hidden[rel * 2 * NUM_RANKS..][..2 * NUM_RANKS];
            for (o, c) in block
                .iter_mut()
                .zip(p.hand.counts().into_iter().chain(p.reserve.counts()))
            {
                *o = f32::from(c) / f32::from(CARDS_PER_RANK);
            }
        }
    }

    /// Everything `seat` is allowed to know: their own hand plus public information.
    pub fn observe(&self, seat: usize) -> Observation {
        Observation {
            observer: seat as u8,
            hand: self.players[seat].hand,
            players: self
                .players
                .iter()
                .map(|p| PublicPlayer {
                    hand_size: p.hand.len() as u8,
                    reserve_size: p.reserve.len() as u8,
                    points: p.points,
                })
                .collect(),
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

    /// Cards not in any hand, reserve or the public record: the 3-player set-aside cards.
    #[cfg(test)]
    pub(crate) fn aside(&self) -> Hand {
        self.players
            .iter()
            .fold(Hand::FULL_DECK.difference(&self.seen), |acc, p| {
                acc.difference(&p.hand).difference(&p.reserve)
            })
    }

    #[cfg(test)]
    pub(crate) fn set_hand(&mut self, seat: usize, counts: [u8; crate::constants::NUM_RANKS]) {
        self.players[seat].hand = Hand::from_counts(counts);
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "trick {} ({:?}), piles {:?}",
            self.tricks_played + 1,
            self.trick_type,
            self.piles.as_slice()
        )?;
        for (seat, p) in self.players.iter().enumerate() {
            let marker = if !self.is_over && seat == self.current as usize {
                '>'
            } else {
                ' '
            };
            write!(f, "{marker} seat {seat}: {:>3} pts  {}", p.points, p.hand)?;
            if !p.reserve.is_empty() {
                write!(f, "  (+{} reserve)", p.reserve.len())?;
            }
            if let Some(play) = self.trick.iter().rev().find(|t| t.seat as usize == seat) {
                write!(f, "  played {}", play.combo)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::NUM_RANKS;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    fn game(num_players: usize) -> GameState {
        GameState::new(num_players, 0, &mut SmallRng::seed_from_u64(42))
    }

    /// A game with chosen hands and point piles.
    fn rigged(hands: &[[u8; NUM_RANKS]], piles: &[u8]) -> GameState {
        let mut g = game(hands.len());
        for (seat, &h) in hands.iter().enumerate() {
            g.set_hand(seat, h);
            g.players[seat].reserve = Hand::EMPTY;
        }
        g.piles = piles.iter().copied().collect();
        g
    }

    fn play(g: &mut GameState, combos: &[Combo]) {
        for &c in combos {
            g.apply(c).unwrap();
        }
    }

    #[test]
    fn deal_matches_setup() {
        for n in MIN_PLAYERS..=MAX_PLAYERS {
            let g = game(n);
            let setup = Setup::for_players(n).unwrap();
            assert_eq!(g.num_players(), n);
            assert_eq!(g.piles.len(), setup.middle_cards as usize);
            assert!(g.piles.is_sorted_by(|a, b| a >= b));
            for p in &g.players {
                assert_eq!(p.hand.len(), setup.hand_size as usize);
                assert_eq!(p.reserve.len(), setup.reserve_size as usize);
            }
            let aside = if n == 3 { 15 } else { 0 };
            assert_eq!(g.aside().len(), aside);
            assert_eq!(g.trick_size as usize, n * setup.combos_per_turn as usize);
            assert_eq!(g.current_player(), 0);
            assert!(!g.is_over());
        }
    }

    #[test]
    #[should_panic(expected = "supports")]
    fn rejects_one_player() {
        game(1);
    }

    #[test]
    fn legal_actions_respect_trick_type() {
        let mut g = game(3);
        g.set_hand(0, [2, 3, 1, 3, 0, 3]);
        assert_eq!(g.legal_actions().len(), 5 + 7 + 6);

        g.trick_type = TrickType::Run;
        let actions = g.legal_actions();
        assert_eq!(actions.len(), 11);
        assert!(actions.contains(&Combo::run(1, 4)));
        assert!(!actions.contains(&Combo::run(1, 5)));
        assert!(!actions.contains(&Combo::set(2, 2)));

        g.trick_type = TrickType::Set;
        let actions = g.legal_actions();
        assert_eq!(actions.len(), 12);
        assert!(actions.contains(&Combo::set(2, 3)));
        assert!(!actions.contains(&Combo::run(1, 2)));
    }

    #[test]
    fn legal_action_bound_is_tight() {
        let mut g = game(3);
        g.set_hand(0, [9, 1, 1, 1, 1, 1]);
        assert_eq!(g.legal_actions().len(), MAX_LEGAL_ACTIONS);
    }

    #[test]
    fn mask_matches_actions() {
        let mut g = game(4);
        g.set_hand(0, [2, 3, 1, 3, 0, 3]);
        let mask = g.legal_action_mask();
        assert_eq!(mask.count_ones() as usize, g.legal_actions().len());
        for c in g.legal_actions() {
            assert!(mask & (1 << c.action_index()) != 0);
        }
    }

    #[test]
    fn apply_rejects_illegal_actions() {
        let mut g = rigged(
            &[[2, 2, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0], [1; 6]],
            &[5, 3, 1],
        );
        assert_eq!(
            g.apply(Combo::single(6)),
            Err(ActionError::CardsNotInHand {
                combo: Combo::single(6)
            })
        );
        g.apply(Combo::set(1, 2)).unwrap();
        assert_eq!(
            g.apply(Combo::run(1, 2)),
            Err(ActionError::WrongTrickType {
                combo: Combo::run(1, 2),
                trick_type: TrickType::Set
            })
        );
        // a single is always allowed, and a lead single leaves the trick open
        g.apply(Combo::single(2)).unwrap();
        assert_eq!(g.trick_type(), TrickType::Set);
    }

    #[test]
    fn single_lead_leaves_trick_open() {
        let mut g = rigged(
            &[[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [1; 6]],
            &[5, 3, 1],
        );
        g.apply(Combo::single(1)).unwrap();
        assert_eq!(g.trick_type(), TrickType::Open);
        g.apply(Combo::run(2, 3)).unwrap();
        assert_eq!(g.trick_type(), TrickType::Run);
    }

    #[test]
    fn bigger_combo_wins_and_leads() {
        let mut g = rigged(
            &[[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0], [2, 0, 0, 1, 0, 0]],
            &[6, 4, 2],
        );
        play(
            &mut g,
            &[Combo::single(2), Combo::single(5), Combo::set(1, 2)],
        );
        // the pair of 1s is the biggest combo despite its low rank
        assert_eq!(g.scores().as_slice(), &[2, 4, 6]);
        assert_eq!(g.current_player(), 2);
        assert!(g.trick().is_empty());
        assert_eq!(g.trick_type(), TrickType::Open);
        assert_eq!(g.tricks_played(), 1);
    }

    #[test]
    fn ties_go_to_earlier_play() {
        let mut g = rigged(
            &[[1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1]],
            &[6, 4, 2],
        );
        play(
            &mut g,
            &[Combo::single(6), Combo::single(4), Combo::single(6)],
        );
        assert_eq!(g.scores().as_slice(), &[6, 2, 4]);
        assert_eq!(g.current_player(), 0);
    }

    #[test]
    fn next_piles_pair_matching_ranks() {
        let mut g = rigged(
            &[[1, 0, 3, 0, 0, 0], [1, 0, 0, 2, 0, 0], [1, 0, 0, 2, 0, 0]],
            &[6, 4, 2],
        );
        play(
            &mut g,
            &[Combo::set(3, 2), Combo::single(4), Combo::single(4)],
        );
        assert_eq!(g.piles(), &[8, 6]);
    }

    #[test]
    fn next_piles_discard_lowest() {
        let mut g = rigged(
            &[[0, 0, 3, 1, 1, 1], [0, 0, 2, 1, 1, 1], [1; 6]],
            &[6, 4, 2],
        );
        play(
            &mut g,
            &[Combo::run(3, 6), Combo::run(3, 6), Combo::run(1, 2)],
        );
        // played: 1, 2, 3x2, 4x2, 5x2, 6x2 -> piles 12, 10, 8, 6, 2, 1; keep 3
        assert_eq!(g.piles(), &[12, 10, 8]);
    }

    #[test]
    fn empty_hand_sits_out_and_lead_passes_clockwise() {
        let mut g = rigged(
            &[[0, 0, 0, 0, 0, 1], [3, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]],
            &[6, 4, 2],
        );
        play(
            &mut g,
            &[Combo::single(6), Combo::single(1), Combo::single(1)],
        );
        // seat 0 won but has no cards: seat 1 leads, and only 2 piles remain
        assert!(!g.is_over());
        assert_eq!(g.current_player(), 1);
        assert_eq!(g.trick_size, 2);
        // played 6, 1, 1 -> a single 6 and a pair of 1s
        assert_eq!(g.piles(), &[6, 2]);
        play(&mut g, &[Combo::single(1), Combo::single(1)]);
        assert_eq!(g.current_player(), 1);
    }

    #[test]
    fn two_empty_hands_end_the_game() {
        let mut g = rigged(
            &[
                [1, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0],
            ],
            &[4, 3, 2, 1],
        );
        play(&mut g, &[Combo::single(1); 4]);
        assert!(g.is_over());
        assert_eq!(g.scores().as_slice(), &[4, 3, 2, 1]);
        assert_eq!(g.winners().as_slice(), &[0]);
        assert_eq!(g.returns().as_slice(), &[1.0, 0.0, 0.0, 0.0]);
        assert!(g.legal_actions().is_empty());
        assert_eq!(g.apply(Combo::single(1)), Err(ActionError::GameOver));
    }

    #[test]
    fn tied_winners_share_returns() {
        let mut g = game(3);
        g.is_over = true;
        g.players[0].points = 10;
        g.players[2].points = 10;
        assert_eq!(g.winners().as_slice(), &[0, 2]);
        assert_eq!(g.returns().as_slice(), &[0.5, 0.0, 0.5]);
    }

    #[test]
    fn two_player_tricks_alternate_twice() {
        let mut g = rigged(&[[2, 2, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0]], &[6, 5, 4, 3]);
        assert_eq!(g.trick_size, 4);
        for (expected_seat, combo) in [(0, 1), (1, 4), (0, 2), (1, 5)] {
            assert_eq!(g.current_player(), expected_seat);
            g.apply(Combo::single(combo)).unwrap();
        }
        // singles 5 > 4 > 2 > 1 claim 6, 5, 4, 3
        assert_eq!(g.scores().as_slice(), &[3 + 4, 6 + 5]);
        assert_eq!(g.current_player(), 1);
        assert_eq!(g.piles(), &[5, 4, 2, 1]);
    }

    #[test]
    fn two_player_reserve_is_picked_up_immediately() {
        let mut g = rigged(&[[1, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0]], &[6, 5, 4, 3]);
        g.players[0].reserve = Hand::from_counts([0, 0, 0, 0, 0, 2]);
        play(&mut g, &[Combo::single(1), Combo::single(4)]);
        assert_eq!(g.hand(0).count(6), 2);
        assert!(g.players[0].reserve.is_empty());
        g.apply(Combo::single(6)).unwrap();
        assert!(!g.is_over());
    }

    #[test]
    fn two_player_second_run_out_ends_game() {
        let mut g = rigged(&[[1, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0]], &[6, 5, 4, 3]);
        play(&mut g, &[Combo::single(1)]);
        assert!(g.is_over());
        // seat 0's single claims the top pile; seat 1 scores 1 per card left
        assert_eq!(g.scores().as_slice(), &[6, 4]);
        assert_eq!(g.winners().as_slice(), &[0]);
    }

    #[test]
    fn full_encoding_appends_hidden_hands() {
        use crate::game::encoding::{FULL_OBS_SIZE, OBS_SIZE};
        let g = game(2);
        let mut full = [0.0; FULL_OBS_SIZE];
        g.encode_full(1, &mut full);
        assert_eq!(&full[..OBS_SIZE], &g.observe(1).encoded()[..]);
        // relative seat 1 from seat 1's view is seat 0: hand then reserve
        let h = &full[OBS_SIZE + 12..OBS_SIZE + 24];
        let expected: Vec<f32> = g.players[0]
            .hand
            .counts()
            .into_iter()
            .chain(g.players[0].reserve.counts())
            .map(|c| f32::from(c) / 10.0)
            .collect();
        assert_eq!(h, &expected[..]);
    }

    #[test]
    fn from_deal_validates() {
        let h = Hand::from_counts([3, 3, 2, 2, 2, 2]);
        let m = Hand::from_counts([1, 1, 1, 0, 0, 0]);
        assert!(GameState::from_deal(&[h; 3], &[Hand::EMPTY; 3], m, 0).is_ok());
        assert!(GameState::from_deal(&[h; 3], &[Hand::EMPTY; 3], m, 3).is_err());
        assert!(GameState::from_deal(&[h; 3], &[Hand::EMPTY; 2], m, 0).is_err());
        // four hands of three 1s each plus the middle's 1 would need 13 ones
        assert!(
            GameState::from_deal(
                &[h; 4],
                &[Hand::EMPTY; 4],
                Hand::from_counts([1, 1, 1, 1, 0, 0]),
                0
            )
            .is_err()
        );
        let small = Hand::from_counts([1, 0, 0, 0, 0, 0]);
        assert!(GameState::from_deal(&[small; 3], &[Hand::EMPTY; 3], m, 0).is_err());
    }

    proptest! {
        #[test]
        fn replaying_history_reproduces_the_game(
            seed in any::<u64>(),
            num_players in MIN_PLAYERS..=MAX_PLAYERS,
            moves in 0..80usize,
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let start = GameState::new(num_players, seed as usize % num_players, &mut rng);
            let hands: Vec<Hand> = start.players.iter().map(|p| p.hand).collect();
            let reserves: Vec<Hand> = start.players.iter().map(|p| p.reserve).collect();
            let mut g = start.clone();
            for _ in 0..moves {
                if g.is_over() {
                    break;
                }
                let actions = g.legal_actions();
                g.apply(actions[rng.random_range(0..actions.len())]).unwrap();
            }
            prop_assert_eq!(g.initial_middle(), start.seen);
            let mut replay = GameState::from_deal(&hands, &reserves, g.initial_middle(), g.first_leader()).unwrap();
            prop_assert_eq!(&replay, &start);
            for play in g.history() {
                prop_assert_eq!(replay.current_player(), play.seat as usize);
                replay.apply(play.combo).unwrap();
            }
            prop_assert_eq!(replay, g);
        }

        #[test]
        fn random_games_preserve_invariants(
            seed in any::<u64>(),
            num_players in MIN_PLAYERS..=MAX_PLAYERS,
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let leader = seed as usize % num_players;
            let mut g = GameState::new(num_players, leader, &mut rng);
            let aside = g.aside();
            let mut moves = 0;
            while !g.is_over() {
                let actions = g.legal_actions();
                prop_assert!(!actions.is_empty());
                prop_assert_eq!(g.legal_action_mask().count_ones() as usize, actions.len());
                let combo = actions[rng.random_range(0..actions.len())];
                g.apply(combo).unwrap();
                moves += 1;
                prop_assert!(moves <= crate::constants::DECK_SIZE);

                // card conservation
                let mut total = g.seen.len() + aside.len();
                for p in &g.players {
                    total += p.hand.len() + p.reserve.len();
                }
                prop_assert_eq!(total, DECK_SIZE);
                prop_assert!(g.piles.len() <= g.trick_size as usize);
                prop_assert!(g.piles.is_sorted_by(|a, b| a >= b));
            }
            let total: u32 = g.scores().iter().map(|&s| s as u32).sum();
            prop_assert!(total > 0);
            let returns: f32 = g.returns().iter().sum();
            prop_assert!((returns - 1.0).abs() < 1e-6);
        }
    }
}
