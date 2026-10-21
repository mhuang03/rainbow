//! A fast, allocation-free engine for the card game Rainbow by Allplay, designed for
//! search (MCTS, minimax) and reinforcement-learning agents.
//!
//! - [`GameState`] holds the full game and is cheap to clone.
//! - Every [`Combo`] has a fixed action index in `0..NUM_ACTIONS`, with
//!   [`GameState::legal_action_mask`] for masking network outputs.
//! - [`GameState::observe`] returns a player's [`Observation`], which can be
//!   [encoded](Observation::encode) as a feature vector or
//!   [determinized](Observation::determinize) into a sampled full state.

pub mod agents;
pub mod arena;
pub mod constants;
pub mod game;

pub use game::{
    ActionError, Combo, ComboKind, GameState, Hand, NUM_ACTIONS, Observation, Play, TrickType,
};
