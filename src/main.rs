//! Command-line tools for the Rainbow engine.
//!
//! ```text
//! rainbow bench [num_players] [num_games] [seed]
//! rainbow arena [-d deals] [-s seed] [-t threads] AGENT AGENT [AGENT...]
//! ```

use std::process::ExitCode;
use std::time::Instant;

use rainbow::GameState;
use rainbow::arena::{self, Entrant};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const USAGE: &str = "usage:
  rainbow bench [num_players] [num_games] [seed]
  rainbow arena [-d deals] [-s seed] [-t threads] AGENT AGENT [AGENT...]

AGENT is name[:key=value,...], e.g. pimc:k=64,rollout=random";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("bench") => bench(&args[1..]),
        Some("arena") => run_arena(&args[1..]),
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn parse<T: std::str::FromStr>(value: Option<&String>, what: &str) -> Result<T, String> {
    value
        .ok_or_else(|| format!("missing value for {what}"))?
        .parse()
        .map_err(|_| format!("invalid {what}"))
}

/// Plays random games and reports throughput.
fn bench(args: &[String]) -> Result<(), String> {
    let num_players: usize = args
        .first()
        .map_or(Ok(4), |a| parse(Some(a), "num_players"))?;
    let num_games: u64 = args
        .get(1)
        .map_or(Ok(100_000), |a| parse(Some(a), "num_games"))?;
    let seed: u64 = args.get(2).map_or(Ok(42), |a| parse(Some(a), "seed"))?;
    if rainbow::constants::Setup::for_players(num_players).is_none() {
        return Err("num_players must be 2 to 6".to_owned());
    }

    let mut rng = SmallRng::seed_from_u64(seed);
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
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{num_games} games, {num_players} players: {:.0} games/s, {:.1}M moves/s, {:.1} moves/game",
        num_games as f64 / secs,
        moves as f64 / secs / 1e6,
        moves as f64 / num_games as f64
    );
    Ok(())
}

/// Runs a duplicate tournament between registered agents.
fn run_arena(args: &[String]) -> Result<(), String> {
    let mut config = arena::Config::default();
    let mut lineup = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-d" => config.deals = parse(args.next(), "deals")?,
            "-s" => config.seed = parse(args.next(), "seed")?,
            "-t" => config.threads = parse(args.next(), "threads")?,
            spec => lineup.push(Entrant::registered(spec)?),
        }
    }
    if !(2..=6).contains(&lineup.len()) {
        return Err(format!("{USAGE}\n\nthe lineup needs 2 to 6 agents"));
    }

    let start = Instant::now();
    let report = arena::run(&lineup, &config).map_err(|e| e.to_string())?;
    print!("{report}");
    println!("({:.1}s)", start.elapsed().as_secs_f64());
    Ok(())
}
