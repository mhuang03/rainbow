use std::fmt;

use crate::constants::{CARDS_PER_RANK, MAX_RANK, MIN_RANK, NUM_RANKS};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ComboKind {
    Single { rank: u8 },
    Set { rank: u8, count: u8 },
    Run { start: u8, end: u8 },
}

/// A validated combo: one card, two or more cards of one rank, or two or more consecutive ranks.
///
/// Every combo maps to a unique index in `0..NUM_ACTIONS`, giving learning agents a fixed
/// action space. Indices are laid out as 6 singles, then 54 sets (rank-major, count 2..=10),
/// then 15 runs (ordered by start, then end).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Combo(ComboKind);

const NUM_SET_COUNTS: u8 = CARDS_PER_RANK - 1;
const SET_BASE: u8 = NUM_RANKS as u8;
const RUN_BASE: u8 = SET_BASE + NUM_RANKS as u8 * NUM_SET_COUNTS;
const NUM_RUNS: u8 = (NUM_RANKS * (NUM_RANKS - 1) / 2) as u8;

/// Size of the action space: every distinct combo.
pub const NUM_ACTIONS: usize = (RUN_BASE + NUM_RUNS) as usize;

/// Index of the first run starting at `start`.
const fn run_offset(start: u8) -> u8 {
    let s = start - MIN_RANK;
    s * MAX_RANK - s * (s + 1) / 2
}

/// Every combo, indexed by its action index.
pub const ALL_COMBOS: [Combo; NUM_ACTIONS] = {
    let mut combos = [Combo(ComboKind::Single { rank: MIN_RANK }); NUM_ACTIONS];
    let mut i = 0;
    let mut rank = MIN_RANK;
    while rank <= MAX_RANK {
        combos[i] = Combo::single(rank);
        i += 1;
        rank += 1;
    }
    rank = MIN_RANK;
    while rank <= MAX_RANK {
        let mut count = 2;
        while count <= CARDS_PER_RANK {
            combos[i] = Combo::set(rank, count);
            i += 1;
            count += 1;
        }
        rank += 1;
    }
    let mut start = MIN_RANK;
    while start < MAX_RANK {
        let mut end = start + 1;
        while end <= MAX_RANK {
            combos[i] = Combo::run(start, end);
            i += 1;
            end += 1;
        }
        start += 1;
    }
    assert!(i == NUM_ACTIONS);
    combos
};

const fn valid_rank(rank: u8) -> bool {
    MIN_RANK <= rank && rank <= MAX_RANK
}

impl Combo {
    /// # Panics
    /// If `rank` is not in `1..=6`.
    pub const fn single(rank: u8) -> Self {
        assert!(valid_rank(rank), "rank must be in 1..=6");
        Self(ComboKind::Single { rank })
    }

    /// # Panics
    /// If `rank` is not in `1..=6` or `count` is not in `2..=10`.
    pub const fn set(rank: u8, count: u8) -> Self {
        assert!(valid_rank(rank), "rank must be in 1..=6");
        assert!(
            2 <= count && count <= CARDS_PER_RANK,
            "set count must be in 2..=10"
        );
        Self(ComboKind::Set { rank, count })
    }

    /// # Panics
    /// If either rank is not in `1..=6` or `start >= end`.
    pub const fn run(start: u8, end: u8) -> Self {
        assert!(
            valid_rank(start) && valid_rank(end),
            "rank must be in 1..=6"
        );
        assert!(start < end, "run start must be less than end");
        Self(ComboKind::Run { start, end })
    }

    pub const fn kind(self) -> ComboKind {
        self.0
    }

    /// Number of cards in the combo.
    pub const fn num_cards(self) -> u8 {
        match self.0 {
            ComboKind::Single { .. } => 1,
            ComboKind::Set { count, .. } => count,
            ComboKind::Run { start, end } => end - start + 1,
        }
    }

    /// Highest rank among the combo's cards.
    pub const fn high_rank(self) -> u8 {
        match self.0 {
            ComboKind::Single { rank } | ComboKind::Set { rank, .. } => rank,
            ComboKind::Run { end, .. } => end,
        }
    }

    /// Trick-resolution key: bigger combos (more cards) win, then the highest card.
    /// Remaining ties go to the combo played earlier, which callers handle with a stable sort.
    pub const fn strength(self) -> u8 {
        self.num_cards() * (MAX_RANK + 1) + self.high_rank()
    }

    /// Number of cards of `rank` in the combo.
    pub const fn count_of(self, rank: u8) -> u8 {
        match self.0 {
            ComboKind::Single { rank: r } => (r == rank) as u8,
            ComboKind::Set { rank: r, count } => {
                if r == rank {
                    count
                } else {
                    0
                }
            }
            ComboKind::Run { start, end } => (start <= rank && rank <= end) as u8,
        }
    }

    /// The trick type a combo imposes on the rest of the trick. Singles impose nothing.
    pub const fn trick_type(self) -> TrickType {
        match self.0 {
            ComboKind::Single { .. } => TrickType::Open,
            ComboKind::Set { .. } => TrickType::Set,
            ComboKind::Run { .. } => TrickType::Run,
        }
    }

    pub const fn action_index(self) -> usize {
        (match self.0 {
            ComboKind::Single { rank } => rank - MIN_RANK,
            ComboKind::Set { rank, count } => {
                SET_BASE + (rank - MIN_RANK) * NUM_SET_COUNTS + (count - 2)
            }
            ComboKind::Run { start, end } => RUN_BASE + run_offset(start) + (end - start - 1),
        }) as usize
    }

    pub const fn from_action_index(index: usize) -> Option<Self> {
        if index < NUM_ACTIONS {
            Some(ALL_COMBOS[index])
        } else {
            None
        }
    }
}

impl fmt::Display for Combo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            ComboKind::Single { rank } => write!(f, "{rank}"),
            ComboKind::Set { rank, count } => write!(f, "{count}x{rank}"),
            ComboKind::Run { start, end } => write!(f, "{start}-{end}"),
        }
    }
}

/// Which combos may be played in the current trick.
#[derive(Debug, Default, PartialEq, Eq, Hash, Copy, Clone)]
pub enum TrickType {
    /// Nothing but singles played so far: any combo is allowed.
    #[default]
    Open,
    /// A set has been played: only singles and sets are allowed.
    Set,
    /// A run has been played: only singles and runs are allowed.
    Run,
}

impl TrickType {
    pub const fn allows(self, combo: Combo) -> bool {
        !matches!(
            (self, combo.0),
            (TrickType::Set, ComboKind::Run { .. }) | (TrickType::Run, ComboKind::Set { .. })
        )
    }

    /// The trick type after `combo` is played.
    pub const fn after(self, combo: Combo) -> Self {
        match combo.trick_type() {
            TrickType::Open => self,
            t => t,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;

    impl Arbitrary for Combo {
        type Parameters = ();
        type Strategy = BoxedStrategy<Combo>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (0..NUM_ACTIONS).prop_map(|i| ALL_COMBOS[i]).boxed()
        }
    }

    #[test]
    fn action_index_round_trips() {
        assert_eq!(NUM_ACTIONS, 75);
        for (i, combo) in ALL_COMBOS.iter().enumerate() {
            assert_eq!(combo.action_index(), i);
            assert_eq!(Combo::from_action_index(i), Some(*combo));
        }
        assert_eq!(Combo::from_action_index(NUM_ACTIONS), None);
    }

    #[test]
    fn all_combos_are_distinct() {
        let set: std::collections::HashSet<_> = ALL_COMBOS.iter().collect();
        assert_eq!(set.len(), NUM_ACTIONS);
    }

    #[test]
    fn strength_follows_rulebook() {
        // more cards beat fewer, regardless of rank
        assert!(Combo::set(1, 2).strength() > Combo::single(6).strength());
        assert!(Combo::run(1, 3).strength() > Combo::set(6, 2).strength());
        assert!(Combo::set(1, 5).strength() > Combo::run(3, 6).strength());
        assert!(Combo::run(3, 6).strength() > Combo::set(5, 4).strength());
        // equal size: highest card wins
        assert!(Combo::set(5, 2).strength() > Combo::set(4, 2).strength());
        assert!(Combo::run(2, 4).strength() > Combo::run(1, 3).strength());
        assert!(Combo::run(4, 5).strength() < Combo::set(6, 2).strength());
        // equal size and high card: tied (earlier play breaks it)
        assert_eq!(Combo::single(3).strength(), Combo::single(3).strength());
    }

    #[test]
    fn trick_type_restrictions() {
        let (single, set, run) = (Combo::single(1), Combo::set(1, 2), Combo::run(1, 2));
        assert!(
            [single, set, run]
                .iter()
                .all(|&c| TrickType::Open.allows(c))
        );
        assert!(TrickType::Set.allows(single) && TrickType::Set.allows(set));
        assert!(!TrickType::Set.allows(run));
        assert!(TrickType::Run.allows(single) && TrickType::Run.allows(run));
        assert!(!TrickType::Run.allows(set));

        assert_eq!(TrickType::Open.after(single), TrickType::Open);
        assert_eq!(TrickType::Open.after(set), TrickType::Set);
        assert_eq!(TrickType::Run.after(single), TrickType::Run);
    }

    #[test]
    #[should_panic]
    fn set_of_one_is_rejected() {
        Combo::set(3, 1);
    }

    #[test]
    #[should_panic]
    fn reversed_run_is_rejected() {
        Combo::run(4, 2);
    }

    proptest! {
        #[test]
        fn count_of_sums_to_len(combo in any::<Combo>()) {
            let total: u8 = (MIN_RANK..=MAX_RANK).map(|r| combo.count_of(r)).sum();
            prop_assert_eq!(total, combo.num_cards());
        }

        #[test]
        fn strength_orders_by_len_then_high_rank(a in any::<Combo>(), b in any::<Combo>()) {
            let expected = (a.num_cards(), a.high_rank()).cmp(&(b.num_cards(), b.high_rank()));
            prop_assert_eq!(a.strength().cmp(&b.strength()), expected);
        }
    }
}
