//! The agents people can play against, by friendly name.

use std::path::Path;

use rainbow::agents::{self, Agent};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Preset {
    pub id: String,
    pub label: String,
    pub description: String,
    /// An `agents::create` spec.
    #[serde(skip)]
    pub spec: String,
}

impl Preset {
    pub fn create(&self, seed: u64) -> Option<Box<dyn Agent>> {
        agents::create(&self.spec, seed).ok()
    }
}

/// The presets whose agents can be built: network agents need their model files in `models`.
pub fn available(models: &Path) -> Vec<Preset> {
    let m = |file: &str| models.join(file).display().to_string();
    let league = m("dmc_league3.bin");
    let ensemble = format!("{}+{}+{league}", m("dmc_league.bin"), m("dmc_league2.bin"));
    let belief = m("belief_v1.bin");
    let list = [
        (
            "random",
            "Random",
            "Plays a random legal move.",
            "random".to_string(),
        ),
        (
            "cheapest",
            "Cheapest",
            "Wins tricks as cheaply as it can, otherwise dumps its lowest card.",
            "cheapest".into(),
        ),
        (
            "economist",
            "Economist",
            "Hand-written strategy weighing piles against the cards they cost.",
            "economist".into(),
        ),
        (
            "counter",
            "Counter",
            "Counts cards to bound what opponents can still play.",
            "counter".into(),
        ),
        (
            "dmc",
            "DMC",
            "Neural network trained by self-play and league play. Fast and strong.",
            format!("dmc:path={league}"),
        ),
        (
            "ensemble",
            "DMC ensemble",
            "Three league networks voting together.",
            format!("dmc:path={ensemble}"),
        ),
        (
            "hybrid",
            "Hybrid",
            "DMC that solves the endgame exactly by search.",
            format!("hybrid:path={league}"),
        ),
        (
            "ismcts",
            "ISMCTS",
            "Monte Carlo tree search over possible deals.",
            "ismcts".into(),
        ),
        (
            "guided",
            "Guided ISMCTS",
            "Tree search guided by the DMC network. The strongest agent; thinks for a moment.",
            format!("ismcts:n=200,prior={league},leaf={league},belief={belief}"),
        ),
    ];
    list.into_iter()
        .map(|(id, label, description, spec)| Preset {
            id: id.into(),
            label: label.into(),
            description: description.into(),
            spec,
        })
        .filter(|p| agents::create(&p.spec, 0).is_ok())
        .collect()
}
