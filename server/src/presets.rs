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
    /// takes in random lineups of the presets, over an even share (1 = even, 2 = double).
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
    // likewise with 3 and 4 players: 1.63 to 1.50 and 1.55 to 1.43 in mixed fields
    let three = m("dmc_spec3p_1.bin");
    let four = m("dmc_spec4p_1.bin");
    let belief = m("belief_v3.bin");
    let ensemble = format!("{}+{}+{league}", m("dmc_league7.bin"), m("dmc_league8.bin"));
    let list = [
        (
            "random",
            "Random",
            "Plays a random legal move.",
            "random".to_string(),
            0.17,
        ),
        (
            "lowest",
            "Lowest",
            "Always plays its lowest single card.",
            "lowest".into(),
            0.0,
        ),
        (
            "biggest",
            "Biggest",
            "Always plays its biggest combination.",
            "biggest".into(),
            0.02,
        ),
        (
            "cheapest",
            "Cheapest",
            "Wins tricks as cheaply as it can, otherwise dumps its lowest card.",
            "cheapest".into(),
            0.48,
        ),
        (
            "dmc-playful",
            "DMC, playful",
            "The neural network with its moves loosened: often sensible, sometimes not.",
            format!("dmc:path={distilled},path2={two},path3={three},path4={four},temp=0.1"),
            0.79,
        ),
        (
            "economist",
            "Economist",
            "Hand-written strategy weighing piles against the cards they cost.",
            "economist".into(),
            0.85,
        ),
        (
            "margin",
            "Economist, margin",
            "The Economist aiming to beat the leader by the most, not just to score.",
            "margin".into(),
            1.09,
        ),
        (
            "counter",
            "Counter",
            "Counts cards to judge what opponents can still play.",
            "counter".into(),
            0.98,
        ),
        (
            "nfsp",
            "NFSP",
            "A network trained by neural fictitious self-play, mixing best replies with its average play.",
            format!("nfsp:path={}", m("nfsp_v1.bin")),
            0.73,
        ),
        (
            "dqn",
            "DQN",
            "A network trained by deep Q-learning, scoring every possible play.",
            format!("dqn:path={}", m("dqn_v1.bin")),
            0.8,
        ),
        (
            "ppo",
            "PPO",
            "A policy network trained by proximal policy optimisation.",
            format!("ppo:path={}", m("ppo_v2.bin")),
            0.52,
        ),
        (
            "dmc-relaxed",
            "DMC, relaxed",
            "The neural network playing a little loosely.",
            format!("dmc:path={distilled},path2={two},path3={three},path4={four},temp=0.04"),
            1.19,
        ),
        (
            "dmc-v2",
            "DMC, first version",
            "The network after self-play alone, before league training.",
            format!("dmc:path={}", m("dmc_v2.bin")),
            1.38,
        ),
        (
            "distill",
            "Distilled search",
            "A fast network trained to copy Guided ISMCTS's choices.",
            format!("distill:path={}", m("distill_v5.bin")),
            1.29,
        ),
        (
            "pimc",
            "PIMC",
            "Plays out many guesses at the hidden cards and picks the move that does best.",
            "pimc".into(),
            1.35,
        ),
        (
            "minimax",
            "Minimax",
            "Looks a few plays ahead over guessed deals, assuming everyone plays against it.",
            "minimax".into(),
            0.34,
        ),
        (
            "dmc",
            "DMC",
            "Neural network trained by self-play and league play, with card counting to break near-ties. Fast.",
            format!("dmc:path={distilled},path2={two},path3={three},path4={four},counter=0.04"),
            1.86,
        ),
        (
            "hybrid",
            "Hybrid",
            "The DMC network, solving the endgame exactly by search.",
            format!("hybrid:path={distilled}"),
            1.78,
        ),
        (
            "ensemble",
            "DMC ensemble",
            "Three league networks voting together.",
            format!("dmc:path={ensemble}"),
            1.83,
        ),
        (
            "ismcts",
            "ISMCTS",
            "Tree search over possible deals, guided by card counting.",
            "ismcts".into(),
            2.16,
        ),
        (
            "guided",
            "Guided ISMCTS",
            "Tree search guided by the DMC network. The strongest; thinks for a moment.",
            // steered by the player-count specialist where there is one tested for search (3 and
            // 4 players: 1.79 to 1.68 and 1.92 to 1.75 over the distilled network), else by the
            // distilled network (level with league 9 at 4 players, ahead at 6)
            {
                let guided =
                    |net: &str| format!("ismcts:n=200,prior={net},leaf={net},belief={belief}");
                format!(
                    "bycount:3={}|4={}|rest={}",
                    guided(&three),
                    guided(&four),
                    guided(&distilled)
                )
            },
            2.18,
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
