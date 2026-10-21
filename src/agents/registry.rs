//! Builds agents from spec strings such as `ismcts:n=200,prior=models/dmc_v2.bin`.

use crate::agents::{
    Agent, BiggestAgent, CheapestAgent, DmcAgent, DqnAgent, EconomistAgent, IsmctsAgent,
    LowestAgent, MinimaxAgent, PimcAgent, RandomAgent, belief, dmc, inference, ismcts, minimax,
    pimc, rollout, value,
};

/// Names accepted by [`create`].
pub const AGENT_NAMES: &[&str] = &[
    "random",
    "lowest",
    "biggest",
    "cheapest",
    "economist",
    "pimc",
    "ismcts",
    "minimax",
    "dmc",
    "dqn",
    "ppo",
    "nfsp",
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

/// Loads a DMC network and checks its shape.
fn load_dmc(path: &str) -> Result<std::sync::Arc<dmc::Mlp>, String> {
    let net = dmc::Mlp::load_cached(path)?;
    DmcAgent::new(std::sync::Arc::clone(&net))?;
    Ok(net)
}

/// Opponent inference options: `infer` (`cheapest` or a DMC network path), `icands` (candidate
/// deals), `iwindow` (recent plays scored), `ipower` (likelihood exponent), `ieps` (cheapest
/// model's random-move rate) and `itemp` (DMC softmax temperature).
fn parse_inference(opts: &mut Options) -> Result<Option<inference::Inference>, String> {
    let Some(model) = opts.take::<String>("infer")? else {
        return Ok(None);
    };
    let model = match model.as_str() {
        "cheapest" => inference::OpponentModel::Cheapest {
            epsilon: opts.take("ieps")?.unwrap_or(0.3),
        },
        path => {
            let net = load_dmc(path)?;
            inference::OpponentModel::Dmc {
                net,
                temperature: opts.take("itemp")?.unwrap_or(0.1),
            }
        }
    };
    Ok(Some(inference::Inference {
        model,
        candidates: opts.take("icands")?.unwrap_or(200),
        window: opts.take("iwindow")?.unwrap_or(12),
        power: opts.take("ipower")?.unwrap_or(1.0),
    }))
}

/// Parameters for an `ismcts` spec's options, e.g. `"n=200,prior=model.bin"`.
///
/// # Errors
/// If an option is unknown or invalid.
pub fn ismcts_params(options: &str) -> Result<ismcts::Params, String> {
    let mut opts = Options::parse(options)?;
    let params = parse_ismcts(&mut opts)?;
    opts.finish()?;
    Ok(params)
}

fn parse_ismcts(opts: &mut Options) -> Result<ismcts::Params, String> {
    let d = ismcts::Params::default();
    let rollout = match d.evaluator {
        value::Evaluator::Rollout(r) => r,
        value::Evaluator::Dmc(_) => unreachable!("default evaluator is a rollout"),
    };
    let rollout = parse_rollout(opts, rollout)?;
    let evaluator = match opts.take::<String>("leaf")? {
        None => value::Evaluator::Rollout(rollout),
        Some(path) => value::Evaluator::Dmc(load_dmc(&path)?),
    };
    let prior = match opts.take::<String>("prior")? {
        None => None,
        Some(path) => Some(ismcts::Prior {
            net: load_dmc(&path)?,
            temperature: opts.take("ptemp")?.unwrap_or(0.1),
            c_puct: opts.take("cpuct")?.unwrap_or(1.0),
        }),
    };
    let params = ismcts::Params {
        iterations: opts.take("n")?.unwrap_or(d.iterations),
        time_limit: opts
            .take::<f64>("ms")?
            .map(|ms| std::time::Duration::from_secs_f64(ms / 1000.0)),
        exploration: opts.take("c")?.unwrap_or(d.exploration),
        evaluator,
        reward: rollout::Reward {
            margin_weight: opts.take("margin")?.unwrap_or(d.reward.margin_weight),
        },
        prior,
        belief: match opts.take::<String>("belief")? {
            None => None,
            Some(path) => Some(std::sync::Arc::new(belief::Belief::new(
                dmc::Mlp::load_cached(&path)?,
            )?)),
        },
        inference: parse_inference(opts)?,
    };
    Ok(params)
}

/// Builds an agent from a spec `name[:key=value,...]`, seeding its randomness with `seed`.
///
/// Options:
/// - `pimc`: `k` (deals per decision), plus the rollout options.
/// - `ismcts`: `n` (iterations), `ms` (time budget per move; whichever limit comes first),
///   `c` (exploration), `leaf` (DMC network path for leaf values),
///   `prior` (DMC network path for PUCT priors) with `ptemp` (softmax temperature) and `cpuct`,
///   `belief` (hand-prediction network path for weighted deals), the rollout options, and the
///   inference options below.
/// - `minimax`: `k` (deals), `depth` (plies), `mode` (`paranoid` or `maxn`), `endgame` (cards
///   left at which to search to the end) and `leaf` (DMC network path for leaf values).
/// - `dmc`, `dqn`, `ppo`, `nfsp`: `path`, a trained network file. `dqn`, `ppo` and `nfsp` also
///   take `sample` (`true` to sample from the softmax of the outputs; the default for `nfsp`).
///
/// Rollout options are `rollout` (`random` or `cheapest`), `eps` (random-move rate of the
/// cheapest rollout) and `margin` (score-margin reward weight).
///
/// Inference options are `infer` (`cheapest` or a DMC network path, as the opponent model),
/// `icands` (candidate deals), `iwindow` (recent plays scored), `ipower` (likelihood exponent),
/// `ieps` (cheapest model's random-move rate) and `itemp` (DMC softmax temperature).
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
        "ismcts" => Box::new(IsmctsAgent::with_params(parse_ismcts(&mut opts)?, seed)),
        "minimax" => {
            let d = minimax::Params::default();
            let mode = match opts.take::<String>("mode")?.as_deref() {
                None => d.mode,
                Some("paranoid") => minimax::Mode::Paranoid,
                Some("maxn") => minimax::Mode::MaxN,
                Some(other) => return Err(format!("unknown minimax mode {other:?}")),
            };
            let params = minimax::Params {
                determinizations: opts.take("k")?.unwrap_or(d.determinizations),
                depth: opts.take("depth")?.unwrap_or(d.depth),
                mode,
                reward: rollout::Reward {
                    margin_weight: opts.take("margin")?.unwrap_or(d.reward.margin_weight),
                },
                endgame_cards: opts.take("endgame")?.unwrap_or(d.endgame_cards),
                leaf: match opts.take::<String>("leaf")? {
                    None => d.leaf,
                    Some(path) => value::Evaluator::Dmc(load_dmc(&path)?),
                },
            };
            Box::new(MinimaxAgent::with_params(params, seed))
        }
        "dqn" | "ppo" | "nfsp" => {
            let path: String = opts
                .take("path")?
                .ok_or_else(|| format!("{name} requires path=<network file>"))?;
            let mut agent = DqnAgent::new(dmc::Mlp::load_cached(&path)?)?;
            agent = match name {
                "ppo" => agent.named("ppo"),
                "nfsp" => agent.named("nfsp"),
                _ => agent,
            };
            // NFSP's average policy is a mixed strategy, so it samples by default.
            if opts.take("sample")?.unwrap_or(name == "nfsp") {
                agent = agent.sampling(seed);
            }
            Box::new(agent)
        }
        "dmc" => {
            let path: String = opts
                .take("path")?
                .ok_or("dmc requires path=<network file>")?;
            Box::new(DmcAgent::new(dmc::Mlp::load_cached(&path)?)?)
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
        for &name in AGENT_NAMES
            .iter()
            .filter(|&&n| !["dmc", "dqn", "ppo", "nfsp"].contains(&n))
        {
            let agent = create(name, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(agent.name(), name);
        }
        assert!(create("nonsense", 0).is_err());
        assert!(create("pimc:k=8,rollout=random,margin=0", 0).is_ok());
        assert!(create("pimc:k=eight", 0).is_err());
        assert!(create("pimc:depth=3", 0).is_err());
        assert!(create("random:k=3", 0).is_err());
        assert!(create("dmc", 0).is_err());
        assert!(create("ismcts:n=10,infer=cheapest,icands=20,iwindow=6", 0).is_ok());
        assert!(create("ismcts:infer=/nonexistent", 0).is_err());
        assert!(create("dmc:path=/nonexistent", 0).is_err());
    }
}
