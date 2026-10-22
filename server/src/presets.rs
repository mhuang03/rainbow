//! The agents people can play against, by friendly name.

use std::path::Path;

use rainbow::agents::{self, Agent};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Preset {
    pub id: String,
    pub label: String,
    /// Difficulty group shown in menus: Beginner, Casual, Strong or Expert.
    pub level: String,
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
    let league = m("dmc_league9.bin");
    // the league ensemble (rounds 8-10) distilled into one network; it plays like the ensemble
    let distilled = m("dmc_distilled1.bin");
    let belief = m("belief_v3.bin");
    // Grouped by difficulty and ordered by strength (the 4-player ladder in ROADMAP.md).
    let list = [
        (
            "random",
            "Random",
            "Beginner",
            "Plays a random legal move.",
            "random".to_string(),
        ),
        (
            "cheapest",
            "Cheapest",
            "Beginner",
            "Wins tricks as cheaply as it can, otherwise dumps its lowest card.",
            "cheapest".into(),
        ),
        (
            "dmc-playful",
            "DMC, playful",
            "Beginner",
            "The neural network with its moves loosened: often sensible, sometimes not.",
            format!("dmc:path={distilled},temp=0.1"),
        ),
        (
            "economist",
            "Economist",
            "Casual",
            "Hand-written strategy weighing piles against the cards they cost.",
            "economist".into(),
        ),
        (
            "counter",
            "Counter",
            "Casual",
            "Counts cards to judge what opponents can still play.",
            "counter".into(),
        ),
        (
            "dmc-relaxed",
            "DMC, relaxed",
            "Casual",
            "The neural network playing a little loosely.",
            format!("dmc:path={distilled},temp=0.04"),
        ),
        (
            "dmc",
            "DMC",
            "Strong",
            "Neural network trained by self-play and league play, with card counting to break near-ties. Fast.",
            format!("dmc:path={distilled},counter=0.04"),
        ),
        (
            "ismcts",
            "ISMCTS",
            "Strong",
            "Tree search over possible deals, guided by card counting.",
            "ismcts".into(),
        ),
        (
            "guided",
            "Guided ISMCTS",
            "Expert",
            "Tree search guided by the DMC network. The strongest; thinks for a moment.",
            format!("ismcts:n=200,prior={league},leaf={league},belief={belief}"),
        ),
    ];
    list.into_iter()
        .map(|(id, label, level, description, spec)| Preset {
            id: id.into(),
            label: label.into(),
            level: level.into(),
            description: description.into(),
            spec,
        })
        .filter(|p| agents::create(&p.spec, 0).is_ok())
        .collect()
}
