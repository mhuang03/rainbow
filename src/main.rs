//! Plays random games and reports throughput and win rates by seat.
//!
//! Usage: `rainbow [num_players] [num_games] [seed]`

use std::time::Instant;

use rainbow::GameState;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut next = |default: u64| {
        args.next()
            .map_or(default, |a| a.parse().expect("expected a number"))
    };
    let num_players = next(4) as usize;
    let num_games = next(100_000);
    let seed = next(42);

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut wins = vec![0.0f64; num_players];
    let mut moves = 0u64;
    let start = Instant::now();
    for game in 0..num_games {
        let mut state = GameState::new(num_players, game as usize % num_players, &mut rng);
        while !state.is_over() {
            let actions = state.legal_actions();
            let combo = actions[rng.random_range(0..actions.len())];
            state.apply(combo).expect("legal action");
            moves += 1;
        }
        for (w, r) in wins.iter_mut().zip(state.returns()) {
            *w += f64::from(r);
        }
    }
    let secs = start.elapsed().as_secs_f64();

    println!(
        "{num_games} games, {num_players} players: {:.0} games/s, {:.1}M moves/s, {:.1} moves/game",
        num_games as f64 / secs,
        moves as f64 / secs / 1e6,
        moves as f64 / num_games as f64
    );
    for (seat, w) in wins.iter().enumerate() {
        println!("seat {seat}: win share {:.3}", w / num_games as f64);
    }
}
