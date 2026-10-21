//! Agents that play from a player's information set, and a registry for building them by name.

use crate::game::{Combo, Observation};

pub mod baseline;
pub mod random;

pub use baseline::{BiggestAgent, CheapestAgent, LowestAgent};
pub use random::RandomAgent;

/// A player that picks a combo from the legal options given its information set.
///
/// Search agents can sample full states consistent with the observation through
/// [`Observation::determinize`].
pub trait Agent {
    /// Short name used in reports.
    fn name(&self) -> &str;

    /// Chooses one of `legal`, which is never empty.
    fn act(&mut self, observation: &Observation, legal: &[Combo]) -> Combo;
}

/// Names accepted by [`create`].
pub const AGENT_NAMES: &[&str] = &["random", "lowest", "biggest", "cheapest"];

/// Builds the agent registered under `name`, seeding any randomness it uses with `seed`.
pub fn create(name: &str, seed: u64) -> Option<Box<dyn Agent>> {
    Some(match name {
        "random" => Box::new(RandomAgent::new(seed)),
        "lowest" => Box::new(LowestAgent),
        "biggest" => Box::new(BiggestAgent),
        "cheapest" => Box::new(CheapestAgent),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_name_builds() {
        for &name in AGENT_NAMES {
            let agent = create(name, 0).unwrap_or_else(|| panic!("{name} not registered"));
            assert_eq!(agent.name(), name);
        }
        assert!(create("nonsense", 0).is_none());
    }
}
