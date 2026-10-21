//! Measures DMC network evaluation speed on observations from random games.
//!
//! Usage: `cargo run --release --example bench_dmc [model] [seconds]`

use std::time::{Duration, Instant};

use rainbow::GameState;
use rainbow::agents::DmcAgent;
use rainbow::agents::dmc::Mlp;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "models/dmc_v2.bin".to_owned());
    let seconds: f64 = args.next().map_or(3.0, |s| s.parse().unwrap());
    let agent = DmcAgent::new(Mlp::load_cached(&path).unwrap()).unwrap();

    // a pool of mid-game positions
    let mut rng = SmallRng::seed_from_u64(1);
    let mut positions = Vec::new();
    while positions.len() < 1000 {
        let mut state = GameState::new(rng.random_range(2..=6), 0, &mut rng);
        while !state.is_over() {
            positions.push((state.observe(state.current_player()), state.legal_actions()));
            let legal = state.legal_actions();
            state
                .apply(legal[rng.random_range(0..legal.len())])
                .unwrap();
        }
    }

    let start = Instant::now();
    let (mut calls, mut actions) = (0u64, 0u64);
    let mut checksum = 0.0f32;
    while start.elapsed() < Duration::from_secs_f64(seconds) {
        for (obs, legal) in &positions[..100] {
            checksum += agent.evaluate(obs, legal)[0];
            calls += 1;
            actions += legal.len() as u64;
        }
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{:.0} evaluations/s, {:.0} actions/s, {:.1} us/evaluation (checksum {checksum:.3})",
        calls as f64 / secs,
        actions as f64 / secs,
        secs * 1e6 / calls as f64
    );
}
