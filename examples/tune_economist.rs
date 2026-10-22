//! Tunes `EconomistAgent` weights by random search followed by hill climbing. Each candidate is
//! scored by its mean relative win share over 2-6 player games against a mixed field of
//! `cheapest`, `random` and default-weight economists, so it can't overfit one opponent.
//!
//! Usage: `cargo run --release --example tune_economist [iterations] [deals] [counts] [threads]`,
//! where `counts` is a comma-separated list of player counts (default `2,3,4,6`), followed by an
//! optional starting variant (`points` or `margin`).

use rainbow::agents::economist::{EconomistAgent, Params};
use rainbow::agents::{Agent, CheapestAgent, RandomAgent};
use rainbow::arena::{self, Entrant};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn opponent(kind: usize) -> Entrant {
    match kind % 3 {
        0 => Entrant::new("cheapest", |_| Box::new(CheapestAgent) as Box<dyn Agent>),
        1 => Entrant::new("random", |s| {
            Box::new(RandomAgent::new(s)) as Box<dyn Agent>
        }),
        _ => Entrant::new("default", |s| {
            Box::new(EconomistAgent::new(s)) as Box<dyn Agent>
        }),
    }
}

fn evaluate(params: Params, deals: u64, seed: u64, counts: &[usize], threads: usize) -> f64 {
    let mut total = 0.0;
    for (i, &n) in counts.iter().enumerate() {
        let mut lineup = vec![Entrant::new("candidate", move |s| {
            Box::new(EconomistAgent::with_params(params, s)) as Box<dyn Agent>
        })];
        lineup.extend((1..n).map(|seat| opponent(i + seat)));
        let config = arena::Config {
            deals,
            seed,
            threads,
        };
        let report = arena::run(&lineup, &config).expect("legal play");
        total += report
            .standings
            .iter()
            .find(|s| s.name == "candidate")
            .unwrap()
            .relative;
    }
    total / counts.len() as f64
}

fn perturb(p: Params, rng: &mut SmallRng, scale: f32) -> Params {
    let mut j = |x: f32, lo: f32, hi: f32| {
        (x * (1.0 + scale * rng.random_range(-1.0..1.0f32))).clamp(lo, hi)
    };
    Params {
        samples: p.samples,
        beat_willingness: j(p.beat_willingness, 0.05, 1.0),
        burn_rate: j(p.burn_rate, 1.0, 4.0),
        scarce_card_cost: j(p.scarce_card_cost, 0.0, 5.0),
        spare_card_cost: j(p.spare_card_cost, 0.0, 5.0),
        set_potential: j(p.set_potential, 0.0, 5.0),
        run_potential: j(p.run_potential, 0.0, 5.0),
        endgame_weight: p.endgame_weight,
        ..p
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let iterations: usize = args.next().map_or(60, |a| a.parse().unwrap());
    let deals: u64 = args.next().map_or(300, |a| a.parse().unwrap());
    let counts: Vec<usize> = args
        .next()
        .map_or("2,3,4,6".to_owned(), |a| a)
        .split(',')
        .map(|c| c.parse().unwrap())
        .collect();
    let threads: usize = args.next().map_or(0, |a| a.parse().unwrap());
    let start = match args.next().as_deref() {
        None | Some("points") => Params::default(),
        Some("margin") => Params::margin(),
        Some(other) => panic!("unknown variant {other:?}"),
    };
    let mut rng = SmallRng::seed_from_u64(7);

    let mut best = start;
    let mut best_score = evaluate(best, deals, 1, &counts, threads);
    println!("default: {best_score:.3}");
    for i in 0..iterations {
        // broad exploration first, then local refinement
        let scale = if i < iterations / 2 { 1.0 } else { 0.3 };
        let candidate = perturb(best, &mut rng, scale);
        let score = evaluate(candidate, deals, 1, &counts, threads);
        if score > best_score {
            best = candidate;
            best_score = score;
            println!("{i:>3}: {score:.3} {candidate:?}");
        }
    }
    // re-check the winner on fresh deals
    println!(
        "held-out: default {:.3}, tuned {:.3}",
        evaluate(start, deals * 4, 99, &counts, threads),
        evaluate(best, deals * 4, 99, &counts, threads)
    );
    println!("{best:#?}");
}
