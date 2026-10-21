use crate::game::{Combo, Observation};

/// A player that picks a combo from the legal options given its information set.
///
/// Search agents can sample full states consistent with the observation through
/// [`Observation::determinize`].
pub trait Agent {
    /// Chooses one of `legal`, which is never empty.
    fn act(&mut self, observation: &Observation, legal: &[Combo]) -> Combo;
}

pub mod random;

pub use random::RandomAgent;
