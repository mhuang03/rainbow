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

use crate::agents::Agent;
use crate::agents::rollout::{Reward, Rollout};
use crate::game::{Combo, Observation};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Iterations (sampled deals) per decision.
    pub iterations: usize,
    /// UCB exploration constant.
    pub exploration: f32,
    pub rollout: Rollout,
    pub reward: Reward,
}

impl Default for Params {
    /// Tuned against three Economists in 4-player games. Unlike PIMC, ISMCTS does best with
    /// mostly-`cheapest` rollouts: the tree models opponents' near-term choices itself.
    fn default() -> Self {
        Self {
            iterations: 1000,
            exploration: 0.7,
            rollout: Rollout::Cheapest { epsilon: 0.3 },
            reward: Reward::default(),
        }
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
        });
        self.nodes[parent].children.push(id as u32);
        id
    }

    /// Runs the search and returns visit counts of the root's moves.
    pub fn search(&mut self, obs: &Observation) -> Vec<(Combo, u32)> {
        let p = self.params;
        self.nodes.clear();
        // The root's own fields are unused; only its children matter.
        self.nodes.push(Node {
            combo: Combo::single(1),
            mover: 0,
            children: Vec::new(),
            visits: 0,
            available: 0,
            total: 0.0,
        });
        let mut path: Vec<usize> = Vec::new();

        for _ in 0..p.iterations {
            let mut state = obs.determinize(&mut self.rng);
            let mut node = 0;
            path.clear();

            // Selection and expansion.
            while !state.is_over() {
                let legal = state.legal_actions();
                let mover = state.current_player() as u8;
                let untried: Vec<Combo> = legal
                    .iter()
                    .copied()
                    .filter(|c| {
                        !self.nodes[node]
                            .children
                            .iter()
                            .any(|&ch| self.nodes[ch as usize].combo == *c)
                    })
                    .collect();
                if !untried.is_empty() {
                    let combo = untried[self.rng.random_range(0..untried.len())];
                    node = self.add_node(node, combo, mover);
                    state.apply(combo).expect("legal move");
                    path.push(node);
                    break;
                }

                let mut best = None;
                let mut best_ucb = f32::NEG_INFINITY;
                for &ch in &self.nodes[node].children {
                    let child = &self.nodes[ch as usize];
                    if !legal.contains(&child.combo) {
                        continue;
                    }
                    let mean = child.total / child.visits.max(1) as f32;
                    let explore = p.exploration
                        * ((child.available as f32).ln() / child.visits.max(1) as f32).sqrt();
                    if mean + explore > best_ucb {
                        best_ucb = mean + explore;
                        best = Some(ch as usize);
                    }
                }
                // Every legal move has a child here, so all of them were available.
                for i in 0..self.nodes[node].children.len() {
                    let ch = self.nodes[node].children[i] as usize;
                    if legal.contains(&self.nodes[ch].combo) {
                        self.nodes[ch].available += 1;
                    }
                }
                node = best.expect("legal moves are never empty");
                state
                    .apply(self.nodes[node].combo)
                    .expect("child moves are legal here");
                path.push(node);
            }

            p.rollout.play_out(&mut state, &mut self.rng);
            let rewards = p.reward.all(&state);
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
                (
                    self.nodes[ch as usize].combo,
                    self.nodes[ch as usize].visits,
                )
            })
            .collect()
    }
}

impl Agent for IsmctsAgent {
    fn name(&self) -> &str {
        "ismcts"
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        if legal.len() == 1 {
            return legal[0];
        }
        self.search(obs)
            .into_iter()
            .filter(|(c, _)| legal.contains(c))
            .max_by_key(|&(_, visits)| visits)
            .map_or(legal[0], |(c, _)| c)
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
        assert_eq!(visits.iter().map(|&(_, v)| v).sum::<u32>(), 300);
        // the observer's own hand fixes the root moves
        assert_eq!(visits.len(), state.legal_actions().len());
    }
}
