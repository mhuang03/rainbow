//! The agents people can play against, by friendly name, with their measured strength.

use std::path::Path;

use rainbow::agents::{self, Agent};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Preset {
    pub id: String,
    pub label: String,
    /// Difficulty group shown in menus: Beginner, Casual, Strong or Expert (from `rating`).
    pub level: String,
    pub description: String,
    /// Relative strength from the 4-player ladder in ROADMAP.md: the share of the win the agent
    /// takes against a mix of the other presets, over an even share (1 = even, 2 = double).
    pub rating: f32,
    /// An `agents::create` spec.
    #[serde(skip)]
    pub spec: String,
}

impl Preset {
    pub fn create(&self, seed: u64) -> Option<Box<dyn Agent>> {
        agents::create(&self.spec, seed).ok()
    }
}

/// The difficulty group for a rating.
fn level(rating: f32) -> &'static str {
    match rating {
        r if r < 0.6 => "Beginner",
        r if r < 1.3 => "Casual",
        r if r < 1.95 => "Strong",
        _ => "Expert",
    }
}

/// The presets whose agents can be built (network agents need their model files in `models`),
/// weakest first.
pub fn available(models: &Path) -> Vec<Preset> {
    let m = |file: &str| models.join(file).display().to_string();
    let league = m("dmc_league9.bin");
    // the league ensemble (rounds 8-10) distilled into one network; it plays like the ensemble
    let distilled = m("dmc_distilled1.bin");
    // played instead in 2-player games: trained on them alone, it beats the distilled one 1.05
    let two = m("dmc_spec2p_1.bin");
    let belief = m("belief_v3.bin");
    let ensemble = format!("{}+{}+{league}", m("dmc_league7.bin"), m("dmc_league8.bin"));
    let list = [
        (
            "random",
            "Random",
            "Plays a random legal move.",
            "random".to_string(),
            0.03,
        ),
        (
            "lowest",
            "Lowest",
            "Always plays its lowest single card.",
            "lowest".into(),
            0.1,
        ),
        (
            "biggest",
            "Biggest",
            "Always plays its biggest combination.",
            "biggest".into(),
            0.2,
        ),
        (
            "cheapest",
            "Cheapest",
            "Wins tricks as cheaply as it can, otherwise dumps its lowest card.",
            "cheapest".into(),
            0.31,
        ),
        (
            "dmc-playful",
            "DMC, playful",
            "The neural network with its moves loosened: often sensible, sometimes not.",
            format!("dmc:path={distilled},path2={two},temp=0.1"),
            0.45,
        ),
        (
            "economist",
            "Economist",
            "Hand-written strategy weighing piles against the cards they cost.",
            "economist".into(),
            0.78,
        ),
        (
            "margin",
            "Economist, margin",
            "The Economist aiming to beat the leader by the most, not just to score.",
            "margin".into(),
            0.8,
        ),
        (
            "counter",
            "Counter",
            "Counts cards to judge what opponents can still play.",
            "counter".into(),
            0.84,
        ),
        (
            "nfsp",
            "NFSP",
            "A network trained by neural fictitious self-play, mixing best replies with its average play.",
            format!("nfsp:path={}", m("nfsp_v1.bin")),
            0.7,
        ),
        (
            "dqn",
            "DQN",
            "A network trained by deep Q-learning, scoring every possible play.",
            format!("dqn:path={}", m("dqn_v1.bin")),
            0.9,
        ),
        (
            "ppo",
            "PPO",
            "A policy network trained by proximal policy optimisation.",
            format!("ppo:path={}", m("ppo_v2.bin")),
            0.9,
        ),
        (
            "dmc-relaxed",
            "DMC, relaxed",
            "The neural network playing a little loosely.",
            format!("dmc:path={distilled},path2={two},temp=0.04"),
            1.01,
        ),
        (
            "dmc-v2",
            "DMC, first version",
            "The network after self-play alone, before league training.",
            format!("dmc:path={}", m("dmc_v2.bin")),
            1.2,
        ),
        (
            "distill",
            "Distilled search",
            "A fast network trained to copy Guided ISMCTS's choices.",
            format!("distill:path={}", m("distill_v5.bin")),
            1.3,
        ),
        (
            "pimc",
            "PIMC",
            "Plays out many guesses at the hidden cards and picks the move that does best.",
            "pimc".into(),
            1.1,
        ),
        (
            "minimax",
            "Minimax",
            "Looks a few plays ahead over guessed deals, assuming everyone plays against it.",
            "minimax".into(),
            1.0,
        ),
        (
            "dmc",
            "DMC",
            "Neural network trained by self-play and league play, with card counting to break near-ties. Fast.",
            format!("dmc:path={distilled},path2={two},counter=0.04"),
            1.76,
        ),
        (
            "hybrid",
            "Hybrid",
            "The DMC network, solving the endgame exactly by search.",
            format!("hybrid:path={distilled}"),
            1.75,
        ),
        (
            "ensemble",
            "DMC ensemble",
            "Three league networks voting together.",
            format!("dmc:path={ensemble}"),
            1.75,
        ),
        (
            "ismcts",
            "ISMCTS",
            "Tree search over possible deals, guided by card counting.",
            "ismcts".into(),
            1.81,
        ),
        (
            "guided",
            "Guided ISMCTS",
            "Tree search guided by the DMC network. The strongest; thinks for a moment.",
            format!("ismcts:n=200,prior={league},leaf={league},belief={belief}"),
            2.08,
        ),
    ];
    let mut presets: Vec<Preset> = list
        .into_iter()
        .map(|(id, label, description, spec, rating)| Preset {
            id: id.into(),
            label: label.into(),
            level: level(rating).into(),
            description: description.into(),
            rating,
            spec,
        })
        .filter(|p| agents::create(&p.spec, 0).is_ok())
        .collect();
    // weakest first, so menus read up the difficulty groups and "the next stronger" is the next
    presets.sort_by(|a, b| a.rating.total_cmp(&b.rating));
    presets
}
