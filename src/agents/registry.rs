//! Builds agents from spec strings such as `ismcts:n=200,prior=models/dmc_v2.bin`.

use crate::agents::{
    Agent, BiggestAgent, CheapestAgent, CounterAgent, DmcAgent, DqnAgent, EconomistAgent,
    HybridAgent, IsmctsAgent, LowestAgent, MinimaxAgent, PimcAgent, RandomAgent, belief, counter,
    dmc, economist, inference, ismcts, minimax, pimc, rollout, value,
};

/// Names accepted by [`create`].
pub const AGENT_NAMES: &[&str] = &[
    "random",
    "lowest",
    "biggest",
    "cheapest",
    "economist",
    "margin",
    "counter",
    "pimc",
    "ismcts",
    "minimax",
    "dmc",
    "dmc_hist",
    "hybrid",
    "dqn",
    "ppo",
    "nfsp",
    "distill",
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
        rollout::Rollout::Cheapest { epsilon } | rollout::Rollout::Counter { epsilon } => epsilon,
        rollout::Rollout::Random => 0.1,
    };
    let epsilon = opts.take("eps")?.unwrap_or(epsilon);
    Ok(match opts.take::<String>("rollout")?.as_deref() {
        None => match default {
            rollout::Rollout::Cheapest { .. } => rollout::Rollout::Cheapest { epsilon },
            rollout::Rollout::Counter { .. } => rollout::Rollout::Counter { epsilon },
            other => other,
        },
        Some("random") => rollout::Rollout::Random,
        Some("cheapest") => rollout::Rollout::Cheapest { epsilon },
        Some("counter") => rollout::Rollout::Counter { epsilon },
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
        Some(path) => {
            // a DMC network or a policy network (one logit per action)
            let net = dmc::Mlp::load_cached(&path)?;
            let policy = net.outputs() == crate::game::NUM_ACTIONS;
            if policy {
                DqnAgent::new(std::sync::Arc::clone(&net))?;
            } else {
                DmcAgent::new(std::sync::Arc::clone(&net))?;
            }
            Some(ismcts::Prior {
                net,
                temperature: opts
                    .take("ptemp")?
                    .unwrap_or(if policy { 1.0 } else { 0.1 }),
                c_puct: opts.take("cpuct")?.unwrap_or(1.0),
            })
        }
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
/// - `economist` and `margin`: `pressure` (`true` to make surplus cards free to spend),
///   `objective` (`points` or `margin`) and `adaptive` (prior weight for per-opponent
///   willingness estimated from the game so far). `margin` is the Economist with the margin
///   objective and its own tuned weights.
/// - `counter`: `card` (points each extra card must buy) and `structure` (penalty for breaking
///   pairs, sets and runs when dumping) and `q` (chance an opponent who could beat a combo
///   doesn't).
/// - `pimc`: `k` (deals per decision), plus the rollout options.
/// - `ismcts`: `n` (iterations), `ms` (time budget per move; whichever limit comes first),
///   `c` (exploration), `leaf` (DMC network path for leaf values),
///   `prior` (DMC or policy network path for PUCT priors) with `ptemp` (softmax temperature) and `cpuct`,
///   `belief` (hand-prediction network path for weighted deals), the rollout options, and the
///   inference options below.
/// - `minimax`: `k` (deals), `depth` (plies), `mode` (`paranoid`, `maxn` or `brs`), `endgame` (cards
///   left at which to search to the end) and `leaf` (DMC network path for leaf values).
/// - `dmc`, `dqn`, `ppo`, `nfsp`: `path`, a trained network file. `dmc` also takes several
///   `+`-separated paths and averages their predictions (an ensemble).
/// - `hybrid`: `path` (a DMC network played until the endgame), `cards` (cards left in all hands
///   and reserves at which to switch to exact search; default 8) and `k` (deals solved per
///   endgame move; default 8). `dqn`, `ppo` and `nfsp` also
///   take `sample` (`true` to sample from the softmax of the outputs; the default for `nfsp`).
///
/// Rollout options are `rollout` (`random`, `cheapest` or `counter`), `eps` (random-move rate of the
/// cheapest and counter rollouts) and `margin` (score-margin reward weight).
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
        "counter" => {
            let d = counter::Params::default();
            Box::new(CounterAgent::with_params(counter::Params {
                card_value: opts.take("card")?.unwrap_or(d.card_value),
                structure_value: opts.take("structure")?.unwrap_or(d.structure_value),
                hold_chance: opts.take("q")?.unwrap_or(d.hold_chance),
            }))
        }
        "economist" | "margin" => {
            let d = if name == "margin" {
                economist::Params::margin()
            } else {
                economist::Params::default()
            };
            let objective = match opts.take::<String>("objective")?.as_deref() {
                None => d.objective,
                Some("points") => economist::Objective::Points,
                Some("margin") => economist::Objective::Margin,
                Some(other) => return Err(format!("unknown objective {other:?}")),
            };
            let params = economist::Params {
                pressure_scaling: opts.take("pressure")?.unwrap_or(d.pressure_scaling),
                objective,
                adaptive_prior: opts.take("adaptive")?.or(d.adaptive_prior),
                ..d
            };
            let agent = EconomistAgent::with_params(params, seed);
            Box::new(if name == "margin" {
                agent.named("margin")
            } else {
                agent
            })
        }
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
                Some("brs") => minimax::Mode::BestReply,
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
        "dqn" | "ppo" | "nfsp" | "distill" => {
            let path: String = opts
                .take("path")?
                .ok_or_else(|| format!("{name} requires path=<network file>"))?;
            let mut agent = DqnAgent::new(dmc::Mlp::load_cached(&path)?)?;
            agent = match name {
                "ppo" => agent.named("ppo"),
                "nfsp" => agent.named("nfsp"),
                "distill" => agent.named("distill"),
                _ => agent,
            };
            // NFSP's average policy is a mixed strategy, so it samples by default.
            if opts.take("sample")?.unwrap_or(name == "nfsp") {
                agent = agent.sampling(seed);
            }
            Box::new(agent)
        }
        // `dmc_hist` is the same agent: a history network is told apart by its input width.
        "dmc" | "dmc_hist" => {
            let path: String = opts
                .take("path")?
                .ok_or_else(|| format!("{name} requires path=<network file>"))?;
            let mut members = path
                .split('+')
                .map(|p| DmcAgent::new(dmc::Mlp::load_cached(p)?))
                .collect::<Result<Vec<_>, String>>()?;
            if members.len() == 1 {
                Box::new(members.remove(0))
            } else {
                Box::new(dmc::EnsembleAgent::new(members))
            }
        }
        "hybrid" => {
            let path: String = opts
                .take("path")?
                .ok_or("hybrid requires path=<dmc network file>")?;
            Box::new(HybridAgent::new(
                DmcAgent::new(dmc::Mlp::load_cached(&path)?)?,
                opts.take("cards")?.unwrap_or(8),
                opts.take("k")?.unwrap_or(8),
                seed,
            ))
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
        for &name in AGENT_NAMES.iter().filter(|&&n| {
            !["dmc", "dmc_hist", "hybrid", "dqn", "ppo", "nfsp", "distill"].contains(&n)
        }) {
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
