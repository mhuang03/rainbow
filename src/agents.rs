//! Agents that play from a player's information set, and a [`registry`] for building them
//! from specs like `ismcts:n=200`.

use crate::game::{Combo, Observation};

pub mod baseline;
pub mod belief;
pub mod counter;
pub mod dmc;
pub mod dqn;
pub mod economist;
pub mod inference;
pub mod ismcts;
pub mod minimax;
pub mod pimc;
pub mod random;
pub mod registry;
pub mod rollout;
pub mod value;

pub use baseline::{BiggestAgent, CheapestAgent, LowestAgent};
pub use counter::CounterAgent;
pub use dmc::DmcAgent;
pub use dqn::DqnAgent;
pub use economist::EconomistAgent;
pub use ismcts::IsmctsAgent;
pub use minimax::MinimaxAgent;
pub use pimc::PimcAgent;
pub use random::RandomAgent;
pub use registry::{AGENT_NAMES, create, ismcts_params};

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
