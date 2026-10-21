//! Depth-limited minimax over sampled deals.
//!
//! For each sampled deal the agent runs a perfect-information search to a fixed depth, scoring
//! leaves with a deterministic `cheapest` rollout or a DMC network, and averages each root move's value over the
//! deals (as in PIMC). When few cards remain the search runs to the end of the game, solving the
//! endgame exactly for that deal.
//!
//! Two multiplayer search rules are available:
//! - **Paranoid**: every opponent tries to minimise the root player's reward. This reduces the
//!   game to two players, so alpha-beta pruning applies.
//! - **Max-n**: every player maximises their own reward. More realistic, but only depth-limited
//!   with no pruning.

use arrayvec::ArrayVec;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::agents::Agent;
use crate::agents::baseline::cheapest_winner;
use crate::agents::rollout::{Reward, Rollout};
use crate::agents::value::Evaluator;
use crate::constants::{MAX_LEGAL_ACTIONS, MAX_PLAYERS};
use crate::game::{Combo, GameState, Observation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Paranoid,
    MaxN,
    /// Best-reply search (BRS+, Esser et al. 2013): between two of the root player's turns only
    /// one opponent may deviate to hurt the root; the others play the `cheapest` default move.
    /// Less pessimistic than paranoid search and with far less branching.
    BestReply,
}

#[derive(Debug, Clone)]
pub struct Params {
    pub determinizations: usize,
    /// Search depth in plies (single plays).
    pub depth: u32,
    pub mode: Mode,
    pub reward: Reward,
    /// Search to the end of the game once at most this many cards remain in hands and reserves.
    pub endgame_cards: usize,
    /// Values positions at the depth limit.
    pub leaf: Evaluator,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            determinizations: 16,
            depth: 3,
            mode: Mode::Paranoid,
            reward: Reward::default(),
            endgame_cards: 8,
            leaf: Evaluator::Rollout(Rollout::Cheapest { epsilon: 0.0 }),
        }
    }
}

fn cards_left(state: &GameState) -> usize {
    (0..state.num_players())
        .map(|s| state.hand(s).len() + state.reserve(s).len())
        .sum()
}

/// Legal moves, strongest first: good moves tend to come early, which helps pruning.
fn ordered_moves(state: &GameState) -> ArrayVec<Combo, MAX_LEGAL_ACTIONS> {
    let mut moves = state.legal_actions();
    moves.sort_unstable_by_key(|c| std::cmp::Reverse(c.strength()));
    moves
}

struct Search {
    params: Params,
    root: usize,
    rng: SmallRng,
    nodes: u64,
}

impl Search {
    /// Value of a depth-limited position for `seat`.
    fn leaf(&mut self, state: &GameState, seat: usize) -> f32 {
        self.params
            .leaf
            .value(state, seat, self.params.reward, &mut self.rng)
    }

    /// Paranoid alpha-beta. With `best_reply`, at most one opponent move between root turns
    /// may differ from the default policy (`deviated` records whether one already has).
    fn paranoid(
        &mut self,
        state: &GameState,
        depth: u32,
        mut alpha: f32,
        mut beta: f32,
        best_reply: bool,
        deviated: bool,
    ) -> f32 {
        self.nodes += 1;
        if state.is_over() {
            return self.params.reward.of(state, self.root);
        }
        if depth == 0 {
            return self.leaf(state, self.root);
        }
        let maximizing = state.current_player() == self.root;
        let mut best = if maximizing {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut moves = ordered_moves(state);
        let default = (best_reply && !maximizing).then(|| cheapest_winner(state.trick(), &moves));
        if deviated && let Some(d) = default {
            moves.clear();
            moves.push(d);
        }
        for combo in moves {
            let mut child = state.clone();
            child.apply(combo).expect("legal move");
            // the root's turn starts a new segment in which one opponent may deviate
            let child_deviated = if maximizing {
                false
            } else {
                deviated || default.is_some_and(|d| d != combo)
            };
            let value = self.paranoid(&child, depth - 1, alpha, beta, best_reply, child_deviated);
            if maximizing {
                best = best.max(value);
                alpha = alpha.max(value);
            } else {
                best = best.min(value);
                beta = beta.min(value);
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    fn max_n(&mut self, state: &GameState, depth: u32) -> [f32; MAX_PLAYERS] {
        self.nodes += 1;
        if state.is_over() {
            return self.params.reward.all(state);
        }
        if depth == 0 {
            return self
                .params
                .leaf
                .values(state, self.params.reward, &mut self.rng);
        }
        let mover = state.current_player();
        let mut best = [f32::NEG_INFINITY; MAX_PLAYERS];
        for combo in ordered_moves(state) {
            let mut child = state.clone();
            child.apply(combo).expect("legal move");
            let values = self.max_n(&child, depth - 1);
            if values[mover] > best[mover] {
                best = values;
            }
        }
        best
    }

    /// Value of playing `combo` in `state` for the root player.
    fn value(&mut self, state: &GameState, combo: Combo) -> f32 {
        let mut child = state.clone();
        child.apply(combo).expect("legal move");
        let depth = if cards_left(&child) <= self.params.endgame_cards {
            u32::MAX
        } else {
            self.params.depth.saturating_sub(1)
        };
        match self.params.mode {
            Mode::Paranoid => self.paranoid(
                &child,
                depth,
                f32::NEG_INFINITY,
                f32::INFINITY,
                false,
                false,
            ),
            Mode::BestReply => {
                self.paranoid(&child, depth, f32::NEG_INFINITY, f32::INFINITY, true, false)
            }
            Mode::MaxN => self.max_n(&child, depth)[self.root],
        }
    }
}

#[derive(Debug, Clone)]
pub struct MinimaxAgent {
    params: Params,
    rng: SmallRng,
}

impl MinimaxAgent {
    pub fn new(seed: u64) -> Self {
        Self::with_params(Params::default(), seed)
    }

    pub fn with_params(params: Params, seed: u64) -> Self {
        Self {
            params,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Mean value of each legal move over sampled deals, and the number of nodes searched.
    pub fn evaluate(&mut self, obs: &Observation, legal: &[Combo]) -> (Vec<f32>, u64) {
        let mut search = Search {
            params: self.params.clone(),
            root: obs.observer(),
            rng: SmallRng::seed_from_u64(0),
            nodes: 0,
        };
        let mut totals = vec![0.0; legal.len()];
        for _ in 0..self.params.determinizations.max(1) {
            let deal = obs.determinize(&mut self.rng);
            for (total, &combo) in totals.iter_mut().zip(legal) {
                *total += search.value(&deal, combo);
            }
        }
        let k = search.params.determinizations.max(1) as f32;
        (totals.iter().map(|t| t / k).collect(), search.nodes)
    }
}

impl Agent for MinimaxAgent {
    fn name(&self) -> &str {
        "minimax"
    }

    fn act(&mut self, obs: &Observation, legal: &[Combo]) -> Combo {
        if legal.len() == 1 {
            return legal[0];
        }
        let (values, _) = self.evaluate(obs, legal);
        let best = (0..legal.len())
            .max_by(|&a, &b| values[a].total_cmp(&values[b]))
            .expect("legal moves are never empty");
        legal[best]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plays_legal_moves_for_whole_games() {
        for mode in [Mode::Paranoid, Mode::MaxN, Mode::BestReply] {
            for n in [2, 3, 6] {
                let mut rng = SmallRng::seed_from_u64(n as u64);
                let mut state = GameState::new(n, 0, &mut rng);
                let mut agent = MinimaxAgent::with_params(
                    Params {
                        determinizations: 2,
                        depth: 2,
                        mode,
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
    }

    #[test]
    fn alpha_beta_matches_plain_paranoid_search() {
        // Without pruning (a full window at every node) paranoid search must give the same
        // values as with it.
        let mut rng = SmallRng::seed_from_u64(9);
        let state = GameState::new(3, 0, &mut rng);
        let params = Params {
            depth: 3,
            ..Params::default()
        };
        let mut s = Search {
            params: params.clone(),
            root: 0,
            rng: SmallRng::seed_from_u64(0),
            nodes: 0,
        };
        fn plain(s: &mut Search, state: &GameState, depth: u32) -> f32 {
            if state.is_over() || depth == 0 {
                return s.paranoid(state, 0, f32::NEG_INFINITY, f32::INFINITY, false, false);
            }
            let values = state.legal_actions().into_iter().map(|c| {
                let mut child = state.clone();
                child.apply(c).unwrap();
                plain(s, &child, depth - 1)
            });
            let values: Vec<f32> = values.collect();
            if state.current_player() == s.root {
                values.into_iter().fold(f32::NEG_INFINITY, f32::max)
            } else {
                values.into_iter().fold(f32::INFINITY, f32::min)
            }
        }
        let pruned = s.paranoid(&state, 3, f32::NEG_INFINITY, f32::INFINITY, false, false);
        let pruned_nodes = s.nodes;
        let full = plain(&mut s, &state, 3);
        assert_eq!(pruned, full);
        assert!(
            pruned_nodes < s.nodes - pruned_nodes,
            "pruning should save work"
        );
    }

    #[test]
    fn best_reply_searches_less_and_is_no_more_pessimistic() {
        let state = GameState::new(4, 0, &mut SmallRng::seed_from_u64(11));
        let search = |best_reply: bool| {
            let mut s = Search {
                params: Params::default(),
                root: 0,
                rng: SmallRng::seed_from_u64(0),
                nodes: 0,
            };
            let v = s.paranoid(
                &state,
                4,
                f32::NEG_INFINITY,
                f32::INFINITY,
                best_reply,
                false,
            );
            (v, s.nodes)
        };
        let (paranoid, paranoid_nodes) = search(false);
        let (brs, brs_nodes) = search(true);
        // opponents have fewer options under BRS, so the root can only do as well or better
        assert!(brs >= paranoid);
        assert!(brs_nodes < paranoid_nodes);
    }

    #[test]
    fn solves_a_tiny_endgame_exactly() {
        // Seat 0 holds a 1 and a 6; the others hold one card each, so the game ends after this
        // trick. Only leading the 6 can win the top pile, whatever the opponents hold.
        let mut state = GameState::new(3, 0, &mut SmallRng::seed_from_u64(1));
        state.set_hand(0, [1, 0, 0, 0, 0, 1]);
        state.set_hand(1, [0, 0, 0, 0, 1, 0]);
        state.set_hand(2, [0, 1, 0, 0, 0, 0]);
        let mut agent = MinimaxAgent::new(0);
        let combo = agent.act(&state.observe(0), &state.legal_actions());
        assert_eq!(combo, Combo::single(6));
    }
}
