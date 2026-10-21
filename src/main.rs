//! Command-line tools for the Rainbow engine.
//!
//! ```text
//! rainbow bench [num_players] [num_games] [seed]
//! rainbow arena [-d deals] [-s seed] [-t threads] AGENT AGENT [AGENT...]
//! rainbow ladder [-p players] [-l lineups] [-d deals] [-s seed] [-t threads] AGENT...
//! rainbow selfplay [-g games] [-s seed] [-t threads] [-p players] [-x explore] -o FILE [OPTIONS]
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

  rainbow ladder [-p players] [-l lineups] [-d deals] [-s seed] [-t threads] AGENT AGENT AGENT...
  rainbow selfplay [-g games] [-s seed] [-t threads] [-p players] [-x explore] -o FILE [OPTIONS]

AGENT is name[:key=value,...], e.g. pimc:k=64,rollout=random
selfplay OPTIONS are ismcts options, e.g. n=200,prior=models/dmc_v2.bin,leaf=models/dmc_v2.bin";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("bench") => bench(&args[1..]),
        Some("arena") => run_arena(&args[1..]),
        Some("selfplay") => selfplay(&args[1..]),
        Some("ladder") => run_ladder(&args[1..]),
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

/// Writes ISMCTS self-play records for expert iteration (see `rainbow::selfplay`).
fn selfplay(args: &[String]) -> Result<(), String> {
    let mut config = rainbow::selfplay::Config {
        games: 100,
        seed: 0,
        threads: 0,
        players: None,
        explore_moves: 3,
    };
    let mut out = None;
    let mut options = String::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-g" => config.games = parse(args.next(), "games")?,
            "-s" => config.seed = parse(args.next(), "seed")?,
            "-t" => config.threads = parse(args.next(), "threads")?,
            "-p" => config.players = Some(parse(args.next(), "players")?),
            "-x" => config.explore_moves = parse(args.next(), "explore")?,
            "-o" => out = Some(parse::<String>(args.next(), "output file")?),
            opts => options = opts.to_owned(),
        }
    }
    let out = out.ok_or_else(|| format!("{USAGE}\n\nselfplay needs -o FILE"))?;
    let params = rainbow::agents::ismcts_params(&options)?;
    let file = std::fs::File::create(&out).map_err(|e| format!("cannot create {out}: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);
    let start = Instant::now();
    let total = config.games;
    let report = move |games: u64, records: u64| {
        if games % 50 == 0 || games == total {
            eprintln!(
                "{games}/{total} games, {records} records, {:.0}s",
                start.elapsed().as_secs_f64()
            );
        }
    };
    let records = rainbow::selfplay::generate(&params, &config, &mut writer, &report)
        .map_err(|e| format!("writing {out}: {e}"))?;
    println!(
        "{records} records from {} games in {:.1}s -> {out}",
        config.games,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Ranks agents over random lineups drawn from a pool.
fn run_ladder(args: &[String]) -> Result<(), String> {
    let mut config = arena::LadderConfig {
        players: 4,
        lineups: 20,
        deals: 50,
        seed: 0,
        threads: 0,
    };
    let mut pool = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-p" => config.players = parse(args.next(), "players")?,
            "-l" => config.lineups = parse(args.next(), "lineups")?,
            "-d" => config.deals = parse(args.next(), "deals")?,
            "-s" => config.seed = parse(args.next(), "seed")?,
            "-t" => config.threads = parse(args.next(), "threads")?,
            spec => pool.push(Entrant::registered(spec)?),
        }
    }
    if rainbow::constants::Setup::for_players(config.players).is_none() {
        return Err("players must be 2 to 6".to_owned());
    }
    if pool.len() < config.players {
        return Err(format!(
            "{USAGE}\n\nthe pool needs at least {} agents",
            config.players
        ));
    }
    let start = Instant::now();
    let standings = arena::ladder(&pool, &config).map_err(|e| e.to_string())?;
    println!(
        "{} players, {} lineups x {} deals x {} rotations",
        config.players, config.lineups, config.deals, config.players
    );
    println!(
        "{:<44} {:>8} {:>9} {:>8}",
        "agent", "lineups", "relative", "+/-"
    );
    for s in standings {
        println!(
            "{:<44} {:>8} {:>9.3} {:>8.3}",
            s.name, s.lineups, s.relative, s.stderr
        );
    }
    println!("({:.1}s)", start.elapsed().as_secs_f64());
    Ok(())
}
