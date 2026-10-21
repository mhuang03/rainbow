//! Agents that play from a player's information set, and a registry for building them by name.

use crate::game::{Combo, Observation};

pub mod baseline;
pub mod economist;
pub mod pimc;
pub mod random;
pub mod rollout;

pub use baseline::{BiggestAgent, CheapestAgent, LowestAgent};
pub use economist::EconomistAgent;
pub use pimc::PimcAgent;
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
pub const AGENT_NAMES: &[&str] = &[
    "random",
    "lowest",
    "biggest",
    "cheapest",
    "economist",
    "pimc",
];

/// Options of a [`create`] spec, as `key=value` pairs.
struct Options<'a>(Vec<(&'a str, &'a str)>);

impl<'a> Options<'a> {
    fn parse(spec: &'a str) -> Result<Self, String> {
        spec.split(',')
            .filter(|kv| !kv.is_empty())
            .map(|kv| {
                kv.split_once('=')
                    .ok_or_else(|| format!("option {kv:?} is not key=value"))
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }

    /// Removes and parses option `key`, if present.
    fn take<T: std::str::FromStr>(&mut self, key: &str) -> Result<Option<T>, String> {
        let Some(i) = self.0.iter().position(|&(k, _)| k == key) else {
            return Ok(None);
        };
        let (_, value) = self.0.remove(i);
        value
            .parse()
            .map(Some)
            .map_err(|_| format!("invalid value {value:?} for option {key}"))
    }

    fn finish(self) -> Result<(), String> {
        match self.0.first() {
            Some((key, _)) => Err(format!("unknown option {key:?}")),
            None => Ok(()),
        }
    }
}

fn parse_rollout(
    opts: &mut Options,
    default: rollout::Rollout,
) -> Result<rollout::Rollout, String> {
    let epsilon = match default {
        rollout::Rollout::Cheapest { epsilon } => epsilon,
        rollout::Rollout::Random => 0.1,
    };
    let epsilon = opts.take("eps")?.unwrap_or(epsilon);
    Ok(match opts.take::<String>("rollout")?.as_deref() {
        None => match default {
            rollout::Rollout::Cheapest { .. } => rollout::Rollout::Cheapest { epsilon },
            other => other,
        },
        Some("random") => rollout::Rollout::Random,
        Some("cheapest") => rollout::Rollout::Cheapest { epsilon },
        Some(other) => return Err(format!("unknown rollout {other:?}")),
    })
}

/// Builds an agent from a spec `name[:key=value,...]`, seeding its randomness with `seed`.
///
/// Options: `pimc` takes `k` (deals per decision), `rollout` (`random` or `cheapest`), `eps`
/// (random-move rate of the cheapest rollout) and `margin` (score-margin reward weight).
///
/// # Errors
/// If the name or an option is unknown or invalid.
pub fn create(spec: &str, seed: u64) -> Result<Box<dyn Agent>, String> {
    let (name, options) = spec.split_once(':').unwrap_or((spec, ""));
    let mut opts = Options::parse(options)?;
    let agent: Box<dyn Agent> = match name {
        "random" => Box::new(RandomAgent::new(seed)),
        "lowest" => Box::new(LowestAgent),
        "biggest" => Box::new(BiggestAgent),
        "cheapest" => Box::new(CheapestAgent),
        "economist" => Box::new(EconomistAgent::new(seed)),
        "pimc" => {
            let d = pimc::Params::default();
            let params = pimc::Params {
                determinizations: opts.take("k")?.unwrap_or(d.determinizations),
                rollout: parse_rollout(&mut opts, d.rollout)?,
                reward: rollout::Reward {
                    margin_weight: opts.take("margin")?.unwrap_or(d.reward.margin_weight),
                },
            };
            Box::new(PimcAgent::with_params(params, seed))
        }
        _ => {
            return Err(format!(
                "unknown agent {name:?}; known agents: {}",
                AGENT_NAMES.join(", ")
            ));
        }
    };
    opts.finish()?;
    Ok(agent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_name_builds() {
        for &name in AGENT_NAMES {
            let agent = create(name, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(agent.name(), name);
        }
        assert!(create("nonsense", 0).is_err());
        assert!(create("pimc:k=8,rollout=random,margin=0", 0).is_ok());
        assert!(create("pimc:k=eight", 0).is_err());
        assert!(create("pimc:depth=3", 0).is_err());
        assert!(create("random:k=3", 0).is_err());
    }
}
