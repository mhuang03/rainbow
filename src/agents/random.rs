use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::agents::Agent;
use crate::game::{Combo, Observation};

/// Picks uniformly among legal combos.
#[derive(Debug, Clone)]
pub struct RandomAgent {
    rng: SmallRng,
}

impl RandomAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn from_entropy() -> Self {
        Self {
            rng: rand::make_rng(),
        }
    }
}

impl Agent for RandomAgent {
    fn name(&self) -> &str {
        "random"
    }

    fn act(&mut self, _observation: &Observation, legal: &[Combo]) -> Combo {
        legal[self.rng.random_range(0..legal.len())]
    }
}
