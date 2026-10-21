//! Rule constants for Rainbow, plus the per-player-count setup table.

/// Number of distinct card ranks (1 through 6).
pub const NUM_RANKS: usize = 6;
pub const MIN_RANK: u8 = 1;
pub const MAX_RANK: u8 = NUM_RANKS as u8;
pub const CARDS_PER_RANK: u8 = 10;
pub const DECK_SIZE: usize = NUM_RANKS * CARDS_PER_RANK as usize;

pub const MIN_PLAYERS: usize = 2;
pub const MAX_PLAYERS: usize = 6;

/// Largest hand a player can ever hold (the 3/4-player deal, and a 2-player hand or reserve).
pub const MAX_HAND_SIZE: usize = 14;

/// In games of 3+ players the game ends after a trick in which this many players have no cards.
pub const EMPTY_HANDS_TO_END_GAME: usize = 2;

/// Upper bound on plays in one trick: 6 players x 1 combo, or 2 players x 2 combos.
pub const MAX_PLAYS_PER_TRICK: usize = MAX_PLAYERS;

/// Upper bound on plays in a whole game: every play uses at least one card.
pub const MAX_GAME_PLAYS: usize = DECK_SIZE;

/// One point pile is available per play in a trick.
pub const MAX_PILES: usize = MAX_PLAYS_PER_TRICK;

/// Upper bound on legal combos for a single hand, reached by e.g. `[9, 1, 1, 1, 1, 1]`:
/// 6 singles + 8 sets + 15 runs. A run needs one card of each rank it spans, so a hand that
/// allows all 15 runs has at most `MAX_HAND_SIZE - 6 = 8` cards left over to form sets.
pub const MAX_LEGAL_ACTIONS: usize = 29;

/// How the deck is dealt for a given player count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Setup {
    /// Cards dealt to each player's hand.
    pub hand_size: u8,
    /// Face-down cards each player picks up when their hand first empties (2-player only).
    pub reserve_size: u8,
    /// Face-up point cards dealt to the middle before the first trick.
    pub middle_cards: u8,
    /// Combos each player plays per trick.
    pub combos_per_turn: u8,
}

impl Setup {
    /// The official setup for `num_players`, or `None` if the count is unsupported.
    ///
    /// Leftover cards (15 in a 3-player game) are set aside face down.
    pub const fn for_players(num_players: usize) -> Option<Self> {
        let (hand_size, reserve_size, middle_cards, combos_per_turn) = match num_players {
            2 => (14, 14, 4, 2),
            3 => (14, 0, 3, 1),
            4 => (14, 0, 4, 1),
            5 => (11, 0, 5, 1),
            6 => (9, 0, 6, 1),
            _ => return None,
        };
        Some(Self {
            hand_size,
            reserve_size,
            middle_cards,
            combos_per_turn,
        })
    }
}

const _: () = {
    let mut n = MIN_PLAYERS;
    while n <= MAX_PLAYERS {
        let Some(s) = Setup::for_players(n) else {
            panic!("missing setup")
        };
        let dealt = n * (s.hand_size + s.reserve_size) as usize + s.middle_cards as usize;
        assert!(dealt <= DECK_SIZE);
        assert!(s.hand_size as usize <= MAX_HAND_SIZE && s.reserve_size as usize <= MAX_HAND_SIZE);
        assert!(n * s.combos_per_turn as usize <= MAX_PLAYS_PER_TRICK);
        assert!(s.middle_cards as usize <= MAX_PILES);
        n += 1;
    }
};
