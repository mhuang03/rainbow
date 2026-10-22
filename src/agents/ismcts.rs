//! Single-observer information-set MCTS (Cowling, Powley and Whitehouse, 2012) with max-n
//! backups.
//!
//! Each iteration samples a deal consistent with the observation and walks one shared tree whose
//! edges are public moves. Since every move in Rainbow is public, the tree's nodes are exactly
//! the observer's information sets. At each node the player to move picks by UCB on their own
//! reward (max-n), so opponents in the tree play to win rather than following a fixed policy.
//! Because a move is only legal in some sampled deals, UCB uses availability counts: how often a
//! child was selectable, rather than how often its parent was visited.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use std::sync::Arc;

use crate::agents::Agent;
use crate::agents::belief::Belief;
use crate::agents::counter::CounterAgent;
use crate::agents::dmc::{DmcAgent, Mlp};
use crate::agents::dqn::DqnAgent;
use crate::agents::inference::Inference;
use crate::agents::rollout::{Reward, Rollout};
use crate::agents::value::Evaluator;
use crate::game::{Combo, GameState, NUM_ACTIONS, Observation};

#[derive(Debug, Clone)]
pub struct Params {
    /// Iterations (sampled deals) per decision.
    pub iterations: usize,
    /// Optional time budget per decision; the search stops at whichever limit comes first.
    pub time_limit: Option<std::time::Duration>,
    /// UCB exploration constant.
    pub exploration: f32,
    /// Values new leaves: a rollout, or a DMC network.
    pub evaluator: Evaluator,
    pub reward: Reward,
    /// Network move priors. When set, selection uses PUCT (as in AlphaZero) instead of UCB.
    pub prior: Option<Prior>,
    /// Bias sampled deals towards a hand-prediction network's expectations.
    pub belief: Option<Arc<Belief>>,
    /// Weight sampled deals by how well they explain opponents' past moves, instead of
    /// sampling uniformly.
    ///
    /// Inference re-weights candidates by the likelihood of the history. With `belief` also set,
    /// candidates come from the belief network, which has already conditioned on that history;
    /// each is weighted by the importance ratio of the uniform deal over the belief proposal so
    /// the evidence isn't counted twice. In testing the combination still did worse than belief
    /// alone (the ratios vary widely, so a few deals take most of the weight); prefer one.
    pub inference: Option<Inference>,
    /// Independent searches run in parallel (root parallelisation), each with an equal share
    /// of the iterations; their root visit counts are summed.
    pub threads: usize,
    /// Searches up to this multiple of `iterations` while the decision is close: past the
    /// budget, the search continues until the most-visited root move has 1.5 times the
    /// visits of the runner-up. 1 turns it off.
    pub extend: f32,
}

impl Default for Params {
    /// Tuned against three Economists in 4-player games. `counter` rollouts with 20% random
    /// moves score 1.97 there, against 1.41 for mostly-`cheapest` rollouts (1.59 against 1.41
    /// at an equal 20 ms per move).
    /// The `counter` prior (temperature 1) raises that to 2.17, and wins head-to-head against
    /// the same search without a prior 1.25 to 0.75 (4p).
    fn default() -> Self {
        Self {
            iterations: 1000,
            time_limit: None,
            exploration: 0.7,
            evaluator: Evaluator::Rollout(Rollout::Counter { epsilon: 0.2 }),
            reward: Reward::default(),
            prior: Some(Prior {
                source: PriorSource::Counter,
                temperature: 1.0,
                counter_weight: 0.0,
                c_puct: 1.0,
            }),
            belief: None,
            inference: None,
            threads: 1,
            extend: 1.0,
        }
    }
}

/// Move priors from a network: a softmax over a DMC network's predicted returns, or over a
/// policy network's logits (one output per action, as trained by `train_distill.py`).
#[derive(Debug, Clone)]
pub struct Prior {
    pub source: PriorSource,
    pub temperature: f32,
    /// With a network source, the weight of the `counter` prior blended in (geometrically,
    /// in log space; the counter scores use temperature 1) in games of 3+ players. 0 uses the
    /// network alone.
    pub counter_weight: f32,
    /// PUCT exploration constant.
    pub c_puct: f32,
}

/// Where move priors come from.
#[derive(Debug, Clone)]
pub enum PriorSource {
    /// A DMC network (predicted returns) or a policy network (one logit per action), told
    /// apart by the output width.
    Network(Arc<Mlp>),
    /// The `counter` strategy's move scores, in points; no network needed.
    Counter,
}

impl Prior {
    fn probabilities(&self, state: &GameState, legal: &[Combo]) -> Vec<f32> {
        let obs = state.observe(state.current_player());
        let q = match &self.source {
            PriorSource::Network(net) if net.outputs() == NUM_ACTIONS => {
                DqnAgent::new(Arc::clone(net))
                    .expect("validated on construction")
                    .evaluate(&obs, legal)
            }
            PriorSource::Network(net) => DmcAgent::new(Arc::clone(net))
                .expect("validated on construction")
                .evaluate(&obs, legal),
            PriorSource::Counter => CounterAgent::new().scores(&obs, legal),
        };
        let max = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut logits: Vec<f32> = q.iter().map(|&v| (v - max) / self.temperature).collect();
        // The blend helps in multiplayer games and hurts heads-up (4p 1.05, 2p 0.93 at 0.5).
        if self.counter_weight > 0.0
            && state.num_players() >= 3
            && matches!(self.source, PriorSource::Network(_))
        {
            let c = CounterAgent::new().scores(&obs, legal);
            let cmax = c.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let w = self.counter_weight;
            for (l, &v) in logits.iter_mut().zip(&c) {
                *l = (1.0 - w) * *l + w * (v - cmax);
            }
        }
        let exp: Vec<f32> = logits.iter().map(|&l| l.exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.into_iter().map(|e| e / sum).collect()
    }
}

#[derive(Debug, Clone)]
struct Node {
    /// The move leading to this node, and the seat that made it.
    combo: Combo,
    mover: u8,
    children: Vec<u32>,
    visits: u32,
    available: u32,
    /// Sum of the mover's rewards over visits.
    total: f32,
    /// Prior probability of the move (PUCT only).
    prior: f32,
}

#[derive(Debug, Clone)]
pub struct IsmctsAgent {
    params: Params,
    rng: SmallRng,
    nodes: Vec<Node>,
}

impl IsmctsAgent {
    pub fn new(seed: u64) -> Self {
        Self::with_params(Params::default(), seed)
    }

    pub fn with_params(params: Params, seed: u64) -> Self {
        Self {
            params,
            rng: SmallRng::seed_from_u64(seed),
            nodes: Vec::new(),
        }
    }

    fn add_node(&mut self, parent: usize, combo: Combo, mover: u8) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            combo,
            mover,
            children: Vec::new(),
            visits: 0,
            available: 1,
            total: 0.0,
            prior: 1.0,
        });
        self.nodes[parent].children.push(id as u32);
        id
    }

    fn child_for(&self, node: usize, combo: Combo) -> Option<usize> {
        self.nodes[node]
            .children
            .iter()
            .map(|&ch| ch as usize)
            .find(|&ch| self.nodes[ch].combo == combo)
    }

    /// Counts a visit's availability for every legal child of `node`.
    fn mark_available(&mut self, node: usize, legal: &[Combo]) {
        for i in 0..self.nodes[node].children.len() {
            let ch = self.nodes[node].children[i] as usize;
            if legal.contains(&self.nodes[ch].combo) {
                self.nodes[ch].available += 1;
            }
        }
    }

    /// One UCB selection step from `node`, applying the chosen move to `state`. Expands a random
    /// untried move if there is one; returns whether it did.
    fn select_ucb(
        &mut self,
        node: &mut usize,
        state: &mut GameState,
        legal: &[Combo],
        c: f32,
    ) -> bool {
        let mover = state.current_player() as u8;
        let untried: Vec<Combo> = legal
            .iter()
            .copied()
            .filter(|&combo| self.child_for(*node, combo).is_none())
            .collect();
        if !untried.is_empty() {
            let combo = untried[self.rng.random_range(0..untried.len())];
            *node = self.add_node(*node, combo, mover);
            state.apply(combo).expect("legal move");
            return true;
        }
        let best = legal
            .iter()
            .map(|&combo| {
                self.child_for(*node, combo)
                    .expect("every legal move has a child")
            })
            .max_by(|&a, &b| {
                let ucb = |ch: usize| {
                    let n = &self.nodes[ch];
                    let visits = n.visits.max(1) as f32;
                    n.total / visits + c * ((n.available as f32).ln() / visits).sqrt()
                };
                ucb(a).total_cmp(&ucb(b))
            })
            .expect("legal moves are never empty");
        self.mark_available(*node, legal);
        *node = best;
        state
            .apply(self.nodes[best].combo)
            .expect("child moves are legal here");
        false
    }

    /// One PUCT selection step from `node`. Children for all legal moves are created with
    /// network priors on first sight; returns whether the chosen child was unvisited.
    fn select_puct(
        &mut self,
        node: &mut usize,
        state: &mut GameState,
        legal: &[Combo],
        prior: &Prior,
    ) -> bool {
        let mover = state.current_player() as u8;
        if legal.iter().any(|&c| self.child_for(*node, c).is_none()) {
            let probs = prior.probabilities(state, legal);
            for (&combo, &p) in legal.iter().zip(&probs) {
                if self.child_for(*node, combo).is_none() {
                    let id = self.add_node(*node, combo, mover);
                    self.nodes[id].prior = p;
                    self.nodes[id].available = 0;
                }
            }
        }
        let children: Vec<usize> = legal
            .iter()
            .map(|&c| self.child_for(*node, c).expect("created above"))
            .collect();
        let (visits, total) = children.iter().fold((0u32, 0.0f32), |(v, t), &ch| {
            (v + self.nodes[ch].visits, t + self.nodes[ch].total)
        });
        // Unvisited moves start at the mean value of their visited siblings.
        let fpu = if visits > 0 {
            total / visits as f32
        } else {
            0.0
        };
        let sqrt_n = (visits.max(1) as f32).sqrt();
        let best = *children
            .iter()
            .max_by(|&&a, &&b| {
                let score = |ch: usize| {
                    let n = &self.nodes[ch];
                    let q = if n.visits > 0 {
                        n.total / n.visits as f32
                    } else {
                        fpu
                    };
                    q + prior.c_puct * n.prior * sqrt_n / (1.0 + n.visits as f32)
                };
                score(a).total_cmp(&score(b))
            })
            .expect("legal moves are never empty");
        self.mark_available(*node, legal);
        *node = best;
        state.apply(self.nodes[best].combo).expect("legal move");
        self.nodes[best].visits == 0
    }

    /// Runs `threads` independent searches at once, each with its share of the iterations, and
    /// sums their root statistics.
    pub fn parallel_search(&mut self, obs: &Observation) -> Vec<RootMove> {
        let threads = self.params.threads.max(1);
        let mut params = self.params.clone();
        params.threads = 1;
        params.iterations = self.params.iterations.div_ceil(threads);
        let seeds: Vec<u64> = (0..threads).map(|_| self.rng.random()).collect();
        let results: Vec<Vec<RootMove>> = std::thread::scope(|scope| {
            let handles: Vec<_> = seeds
                .into_iter()
                .map(|seed| {
                    let params = params.clone();
                    scope.spawn(move || IsmctsAgent::with_params(params, seed).search(obs))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("search thread panicked"))
                .collect()
        });
        let mut merged: Vec<RootMove> = Vec::new();
        for moves in results {
            for m in moves {
                match merged.iter_mut().find(|x| x.combo == m.combo) {
                    Some(x) => {
                        let total = x.visits + m.visits;
                        if total > 0 {
                            let a = if x.mean.is_nan() {
                                0.0
                            } else {
                                x.mean * x.visits as f32
                            };
                            let b = if m.mean.is_nan() {
                                0.0
                            } else {
                                m.mean * m.visits as f32
                            };
                            x.mean = (a + b) / total as f32;
                        }
                        x.visits = total;
                    }
                    None => merged.push(m),
                }
            }
        }
        merged
    }

    /// Whether the most-visited root move leads the runner-up clearly (by half again).
    fn root_decided(&self) -> bool {
        let (mut first, mut second) = (0, 0);
        for &ch in &self.nodes[0].children {
            let v = self.nodes[ch as usize].visits;
            if v > first {
                (first, second) = (v, first);
            } else if v > second {
                second = v;
            }
        }
        first as f32 >= 1.5 * second as f32
    }

    /// Runs the search and returns statistics for each of the root's moves.
    pub fn search(&mut self, obs: &Observation) -> Vec<RootMove> {
        let p = self.params.clone();
        let weights = p.belief.as_ref().map(|b| b.weights(obs));
        let pool = p
            .inference
            .as_ref()
            .map(|inference| inference.sample_pool(obs, weights.as_ref(), &mut self.rng));
        self.nodes.clear();
        // The root's own fields are unused; only its children matter.
        self.nodes.push(Node {
            combo: Combo::single(1),
            mover: 0,
            children: Vec::new(),
            visits: 0,
            available: 0,
            total: 0.0,
            prior: 1.0,
        });
        let mut path: Vec<usize> = Vec::new();

        let start = std::time::Instant::now();
        let max_iterations = (p.iterations as f32 * p.extend.max(1.0)) as usize;
        for i in 0..max_iterations {
            if let Some(limit) = p.time_limit
                && i % 16 == 0
                && i > 0
                && start.elapsed() >= limit
            {
                break;
            }
            if i >= p.iterations && i % 16 == 0 && self.root_decided() {
                break;
            }
            let mut state = match &pool {
                Some(pool) => pool.sample(&mut self.rng).clone(),
                None => match &weights {
                    Some(w) => obs.determinize_weighted(w, &mut self.rng),
                    None => obs.determinize(&mut self.rng),
                },
            };
            let mut node = 0;
            path.clear();

            // Selection and expansion.
            while !state.is_over() {
                let legal = state.legal_actions();
                let expanded = match &p.prior {
                    None => self.select_ucb(&mut node, &mut state, &legal, p.exploration),
                    Some(prior) => self.select_puct(&mut node, &mut state, &legal, prior),
                };
                path.push(node);
                if expanded {
                    break;
                }
            }

            let rewards = p.evaluator.values(&state, p.reward, &mut self.rng);
            for &id in &path {
                let n = &mut self.nodes[id];
                n.visits += 1;
                n.total += rewards[n.mover as usize];
            }
        }

        self.nodes[0]
            .children
            .iter()
            .map(|&ch| {
                let n = &self.nodes[ch as usize];
                RootMove {
                    combo: n.combo,
                    visits: n.visits,
                    mean: if n.visits > 0 {
                        n.total / n.visits as f32
                    } else {
                        f32::NAN
                    },
                }
            })
            .collect()
    }
}

/// Search statistics for one of the root's moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootMove {
    pub combo: Combo,
    pub visits: u32,
    /// Mean reward for the root player over the move's visits (NaN if unvisited).
    pub mean: f32,
}

impl Agent for IsmctsAgent {
    fn name(&self) -> &str {
        "ismcts"
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        if legal.len() == 1 {
            return legal[0];
        }
        let moves = if self.params.threads > 1 {
            self.parallel_search(obs)
        } else {
            self.search(obs)
        };
        moves
            .into_iter()
            .filter(|m| legal.contains(&m.combo))
            .max_by_key(|m| m.visits)
            .map_or(legal[0], |m| m.combo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameState;

    #[test]
    fn plays_legal_moves_for_whole_games() {
        for n in [2, 3, 6] {
            let mut rng = SmallRng::seed_from_u64(n as u64);
            let mut state = GameState::new(n, 0, &mut rng);
            let mut agent = IsmctsAgent::with_params(
                Params {
                    iterations: 50,
                    ..Params::default()
                },
                1,
            );
            while !state.is_over() {
                let seat = state.current_player();
                let combo = agent.act(&state.observe(seat), &state.legal_actions());
                state.apply(combo).unwrap();
            }
        }
    }

    #[test]
    fn policy_network_prior_is_a_distribution_over_legal_moves() {
        use crate::game::encoding::OBS_SIZE;
        let prior = Prior {
            source: PriorSource::Network(Arc::new(crate::agents::dmc::test_mlp(
                &[(OBS_SIZE, 8), (8, NUM_ACTIONS)],
                5,
            ))),
            temperature: 1.0,
            counter_weight: 0.0,
            c_puct: 1.0,
        };
        let mut rng = SmallRng::seed_from_u64(1);
        let state = GameState::new(3, 0, &mut rng);
        let legal = state.legal_actions();
        let p = prior.probabilities(&state, &legal);
        assert_eq!(p.len(), legal.len());
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(p.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn parallel_search_shares_the_iterations() {
        let params = Params {
            iterations: 80,
            threads: 4,
            ..Params::default()
        };
        let mut rng = SmallRng::seed_from_u64(6);
        let state = GameState::new(4, 0, &mut rng);
        let mut agent = IsmctsAgent::with_params(params, 2);
        let moves = agent.parallel_search(&state.observe(0));
        assert_eq!(moves.iter().map(|m| m.visits).sum::<u32>(), 80);
        let legal = state.legal_actions();
        assert!(legal.contains(&agent.act(&state.observe(0), &legal)));
    }

    #[test]
    fn extended_search_stops_once_decided() {
        let mut rng = SmallRng::seed_from_u64(6);
        let state = GameState::new(4, 0, &mut rng);
        let params = Params {
            iterations: 64,
            extend: 4.0,
            ..Params::default()
        };
        let mut agent = IsmctsAgent::with_params(params, 3);
        let visits: u32 = agent.search(&state.observe(0)).iter().map(|m| m.visits).sum();
        assert!((64..=256).contains(&visits));
        // past the budget, the search only stops once the leader is clear (or at the cap)
        assert!(visits == 64 || visits == 256 || agent.root_decided());
    }

    #[test]
    fn counter_prior_guides_a_whole_game() {
        let params = Params {
            iterations: 40,
            prior: Some(Prior {
                source: PriorSource::Counter,
                temperature: 2.0,
                counter_weight: 0.0,
                c_puct: 1.0,
            }),
            ..Params::default()
        };
        let mut rng = SmallRng::seed_from_u64(8);
        let mut state = GameState::new(3, 0, &mut rng);
        let mut agent = IsmctsAgent::with_params(params, 1);
        while !state.is_over() {
            let seat = state.current_player();
            let combo = agent.act(&state.observe(seat), &state.legal_actions());
            state.apply(combo).unwrap();
        }
    }

    #[test]
    fn network_guided_search_plays_legal_moves() {
        use crate::game::encoding::{ACTION_FEATURES, OBS_SIZE};
        let net = Arc::new(crate::agents::dmc::test_mlp(
            &[(OBS_SIZE + ACTION_FEATURES, 16), (16, 1)],
            7,
        ));
        let params = Params {
            iterations: 60,
            evaluator: Evaluator::Dmc(Arc::clone(&net)),
            prior: Some(Prior {
                source: PriorSource::Network(net),
                temperature: 0.5,
                counter_weight: 0.0,
                c_puct: 1.0,
            }),
            ..Params::default()
        };
        for n in [2, 4] {
            let mut rng = SmallRng::seed_from_u64(n as u64);
            let mut state = GameState::new(n, 0, &mut rng);
            let mut agent = IsmctsAgent::with_params(params.clone(), 3);
            let visits = agent.search(&state.observe(0));
            assert_eq!(visits.iter().map(|m| m.visits).sum::<u32>(), 60);
            while !state.is_over() {
                let seat = state.current_player();
                let combo = agent.act(&state.observe(seat), &state.legal_actions());
                state.apply(combo).unwrap();
            }
        }
    }

    #[test]
    fn time_limit_stops_the_search_early() {
        let state = GameState::new(4, 0, &mut SmallRng::seed_from_u64(5));
        let mut agent = IsmctsAgent::with_params(
            Params {
                iterations: 10_000_000,
                time_limit: Some(std::time::Duration::from_millis(20)),
                ..Params::default()
            },
            2,
        );
        let start = std::time::Instant::now();
        let visits: u32 = agent
            .search(&state.observe(0))
            .iter()
            .map(|m| m.visits)
            .sum();
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
        assert!(visits > 0 && visits < 10_000_000);
    }

    #[test]
    fn root_visits_sum_to_iterations() {
        let state = GameState::new(4, 0, &mut SmallRng::seed_from_u64(5));
        let mut agent = IsmctsAgent::with_params(
            Params {
                iterations: 300,
                ..Params::default()
            },
            2,
        );
        let visits = agent.search(&state.observe(0));
        assert_eq!(visits.iter().map(|m| m.visits).sum::<u32>(), 300);
        assert!(
            visits
                .iter()
                .all(|m| m.visits == 0 || (-1.0..=2.0).contains(&m.mean))
        );
        // the observer's own hand fixes the root moves
        assert_eq!(visits.len(), state.legal_actions().len());
    }
}
