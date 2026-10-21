use std::fmt;

use crate::constants::{CARDS_PER_RANK, MAX_RANK, MIN_RANK, NUM_RANKS};
use crate::game::combo::{Combo, ComboKind};

/// A multiset of cards stored as per-rank counts, with the total cached.
///
/// Used for player hands and reserves, and for tallies of publicly seen cards.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hand {
    counts: [u8; NUM_RANKS],
    len: u8,
}

const fn index(rank: u8) -> usize {
    (rank - MIN_RANK) as usize
}

impl Hand {
    pub const EMPTY: Self = Self {
        counts: [0; NUM_RANKS],
        len: 0,
    };

    /// The full 60-card deck.
    pub const FULL_DECK: Self = Self {
        counts: [CARDS_PER_RANK; NUM_RANKS],
        len: NUM_RANKS as u8 * CARDS_PER_RANK,
    };

    /// Builds a hand from per-rank counts, where `counts[0]` is the number of 1s.
    ///
    /// # Panics
    /// If any count exceeds the 10 copies of each rank in the deck.
    pub fn from_counts(counts: [u8; NUM_RANKS]) -> Self {
        assert!(
            counts.iter().all(|&c| c <= CARDS_PER_RANK),
            "a rank cannot appear more than {CARDS_PER_RANK} times"
        );
        Self {
            counts,
            len: counts.iter().sum(),
        }
    }

    /// Per-rank counts, where index 0 holds the number of 1s.
    pub const fn counts(&self) -> [u8; NUM_RANKS] {
        self.counts
    }

    /// Number of cards of `rank`.
    ///
    /// # Panics
    /// If `rank` is not in `1..=6`.
    pub const fn count(&self, rank: u8) -> u8 {
        self.counts[index(rank)]
    }

    pub const fn len(&self) -> usize {
        self.len as usize
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether every card of `combo` is in the hand.
    pub const fn contains(&self, combo: Combo) -> bool {
        match combo.kind() {
            ComboKind::Single { rank } => self.count(rank) >= 1,
            ComboKind::Set { rank, count } => self.count(rank) >= count,
            ComboKind::Run { start, end } => {
                let mut rank = start;
                while rank <= end {
                    if self.count(rank) == 0 {
                        return false;
                    }
                    rank += 1;
                }
                true
            }
        }
    }

    pub(crate) fn add_card(&mut self, rank: u8) {
        debug_assert!(self.count(rank) < CARDS_PER_RANK);
        self.counts[index(rank)] += 1;
        self.len += 1;
    }

    pub(crate) fn add_combo(&mut self, combo: Combo) {
        match combo.kind() {
            ComboKind::Single { rank } => self.counts[index(rank)] += 1,
            ComboKind::Set { rank, count } => self.counts[index(rank)] += count,
            ComboKind::Run { start, end } => {
                for c in &mut self.counts[index(start)..=index(end)] {
                    *c += 1;
                }
            }
        }
        self.len += combo.num_cards();
    }

    /// Removes `combo`'s cards. The caller must ensure `self.contains(combo)`.
    pub(crate) fn remove_combo(&mut self, combo: Combo) {
        debug_assert!(self.contains(combo));
        match combo.kind() {
            ComboKind::Single { rank } => self.counts[index(rank)] -= 1,
            ComboKind::Set { rank, count } => self.counts[index(rank)] -= count,
            ComboKind::Run { start, end } => {
                for c in &mut self.counts[index(start)..=index(end)] {
                    *c -= 1;
                }
            }
        }
        self.len -= combo.num_cards();
    }

    /// Cards in `self` but not in `other`, saturating at zero per rank.
    pub fn difference(&self, other: &Self) -> Self {
        let mut counts = self.counts;
        for (c, o) in counts.iter_mut().zip(other.counts) {
            *c = c.saturating_sub(o);
        }
        Self::from_counts(counts)
    }

    /// Iterates over every card's rank, in ascending order.
    pub fn cards(&self) -> impl DoubleEndedIterator<Item = u8> + '_ {
        (MIN_RANK..=MAX_RANK).flat_map(|r| std::iter::repeat_n(r, self.count(r) as usize))
    }
}

impl fmt::Display for Hand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, rank) in self.cards().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{rank}")?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::combo::ALL_COMBOS;
    use proptest::prelude::*;

    #[test]
    fn contains_checks_every_card() {
        let hand = Hand::from_counts([2, 3, 1, 3, 0, 3]);
        assert!(hand.contains(Combo::single(3)));
        assert!(!hand.contains(Combo::single(5)));
        assert!(hand.contains(Combo::set(2, 3)));
        assert!(!hand.contains(Combo::set(2, 4)));
        assert!(hand.contains(Combo::run(1, 4)));
        assert!(!hand.contains(Combo::run(3, 5)));
    }

    #[test]
    fn display_lists_cards() {
        assert_eq!(
            Hand::from_counts([2, 0, 1, 0, 0, 1]).to_string(),
            "[1 1 3 6]"
        );
    }

    #[test]
    #[should_panic]
    fn from_counts_rejects_impossible_hand() {
        Hand::from_counts([11, 0, 0, 0, 0, 0]);
    }

    proptest! {
        #[test]
        fn add_then_remove_is_identity(
            counts in proptest::array::uniform6(0..=4u8),
            index in 0..ALL_COMBOS.len(),
        ) {
            let hand = Hand::from_counts(counts);
            let combo = ALL_COMBOS[index];
            let mut h = hand;
            h.add_combo(combo);
            prop_assert!(h.contains(combo));
            prop_assert_eq!(h.len(), hand.len() + combo.num_cards() as usize);
            h.remove_combo(combo);
            prop_assert_eq!(h, hand);
        }
    }
}
