//! Duplicate tournaments: every deal is replayed once per seat rotation, so each agent in a
//! lineup plays every seat with the same cards. This removes most of the luck of the deal from
//! comparisons.

use std::collections::BTreeMap;
use std::fmt;
use std::thread;

use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::agents::Agent;
use crate::game::{GameState, RunError, run_game};

/// Builds a fresh agent for one game from a seed.
pub type Factory = Box<dyn Fn(u64) -> Box<dyn Agent> + Send + Sync>;

/// One seat in a lineup. Entrants with the same name are pooled in the report.
pub struct Entrant {
    pub name: String,
    pub make: Factory,
}

impl Entrant {
    pub fn new(
        name: impl Into<String>,
        make: impl Fn(u64) -> Box<dyn Agent> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            make: Box::new(make),
        }
    }

    /// An entrant for an agent from the [registry](crate::agents::create).
    pub fn registered(name: &str) -> Option<Self> {
        crate::agents::create(name, 0)?;
        let owned = name.to_owned();
        Some(Self::new(name, move |seed| {
            crate::agents::create(&owned, seed).expect("checked on construction")
        }))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Deals to play. Each is played once per rotation, so `deals * players` games in total.
    pub deals: u64,
    pub seed: u64,
    /// Worker threads; 0 means one per available core.
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            deals: 1000,
            seed: 0,
            threads: 0,
        }
    }
}

/// Returns are 0 or 1/k for k tied winners (k <= 6), so scaling by 60 makes every sum exact and
/// results independent of how games are split across threads.
const RETURN_SCALE: f64 = 60.0;

#[derive(Debug, Default, Clone, Copy)]
struct Tally {
    games: u64,
    returns: u64,
    returns_sq: u64,
    points: u64,
}

impl Tally {
    fn add(&mut self, ret: f32, points: u8) {
        let r = (f64::from(ret) * RETURN_SCALE).round() as u64;
        self.games += 1;
        self.returns += r;
        self.returns_sq += r * r;
        self.points += u64::from(points);
    }

    fn merge(&mut self, other: &Self) {
        self.games += other.games;
        self.returns += other.returns;
        self.returns_sq += other.returns_sq;
        self.points += other.points;
    }
}

/// Results for one agent name, pooled over every seat it occupied.
#[derive(Debug, Clone, PartialEq)]
pub struct Standing {
    pub name: String,
    /// Games played (one per seat occupied per game).
    pub games: u64,
    /// Mean return: the agent's share of wins, where tied winners split a win.
    pub win_share: f64,
    /// Standard error of `win_share`.
    pub stderr: f64,
    /// `win_share` times the player count: 1.0 is an average player.
    pub relative: f64,
    pub mean_points: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub num_players: usize,
    pub deals: u64,
    /// Sorted from strongest to weakest.
    pub standings: Vec<Standing>,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} players, {} deals x {} rotations",
            self.num_players, self.deals, self.num_players
        )?;
        writeln!(
            f,
            "{:<24} {:>8} {:>10} {:>8} {:>9} {:>8}",
            "agent", "games", "win share", "+/-", "relative", "points"
        )?;
        for s in &self.standings {
            writeln!(
                f,
                "{:<24} {:>8} {:>10.4} {:>8.4} {:>9.3} {:>8.2}",
                s.name, s.games, s.win_share, s.stderr, s.relative, s.mean_points
            )?;
        }
        Ok(())
    }
}

/// SplitMix64 finalizer, used to derive independent seeds.
const fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Plays one deal in every rotation, adding results to `tallies` (indexed by lineup position).
fn play_deal(
    lineup: &[Entrant],
    config: &Config,
    deal: u64,
    tallies: &mut [Tally],
) -> Result<(), RunError> {
    let n = lineup.len();
    let deal_seed = mix(config.seed ^ mix(deal));
    for rotation in 0..n {
        // seat s is played by lineup entry (s + rotation) % n
        let entry = |seat: usize| (seat + rotation) % n;
        let mut agents: Vec<Box<dyn Agent>> = (0..n)
            .map(|seat| {
                (lineup[entry(seat)].make)(mix(deal_seed ^ mix((rotation * n + seat) as u64 + 1)))
            })
            .collect();
        let mut refs: Vec<&mut dyn Agent> = agents
            .iter_mut()
            .map(|a| &mut **a as &mut dyn Agent)
            .collect();
        let state = GameState::new(n, 0, &mut SmallRng::seed_from_u64(deal_seed));
        let end = run_game(state, &mut refs)?;
        for (seat, (&ret, &points)) in end.returns().iter().zip(&end.scores()).enumerate() {
            tallies[entry(seat)].add(ret, points);
        }
    }
    Ok(())
}

/// Runs a duplicate tournament for `lineup`, which has one entrant per seat (2 to 6).
///
/// # Errors
/// If an agent makes an illegal move.
///
/// # Panics
/// If the lineup size is not a supported player count.
pub fn run(lineup: &[Entrant], config: &Config) -> Result<Report, RunError> {
    let n = lineup.len();
    assert!(
        crate::constants::Setup::for_players(n).is_some(),
        "lineup needs 2 to 6 entrants, got {n}"
    );
    let threads = match config.threads {
        0 => thread::available_parallelism().map_or(1, |p| p.get()),
        t => t,
    }
    .min(config.deals.max(1) as usize);

    let per_thread: Vec<Result<Vec<Tally>, RunError>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut tallies = vec![Tally::default(); n];
                    for deal in (t as u64..config.deals).step_by(threads) {
                        play_deal(lineup, config, deal, &mut tallies)?;
                    }
                    Ok(tallies)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("arena worker panicked"))
            .collect()
    });

    let mut by_name: BTreeMap<&str, Tally> = BTreeMap::new();
    for tallies in per_thread {
        for (entrant, tally) in lineup.iter().zip(tallies?) {
            by_name.entry(&entrant.name).or_default().merge(&tally);
        }
    }

    let mut standings: Vec<Standing> = by_name
        .into_iter()
        .map(|(name, t)| {
            let games = t.games.max(1) as f64;
            let mean = t.returns as f64 / RETURN_SCALE / games;
            let mean_sq = t.returns_sq as f64 / (RETURN_SCALE * RETURN_SCALE) / games;
            let variance = (mean_sq - mean * mean).max(0.0);
            Standing {
                name: name.to_owned(),
                games: t.games,
                win_share: mean,
                stderr: (variance / games).sqrt(),
                relative: mean * n as f64,
                mean_points: t.points as f64 / games,
            }
        })
        .collect();
    standings.sort_by(|a, b| b.win_share.total_cmp(&a.win_share));

    Ok(Report {
        num_players: n,
        deals: config.deals,
        standings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lineup(names: &[&str]) -> Vec<Entrant> {
        names
            .iter()
            .map(|n| Entrant::registered(n).unwrap())
            .collect()
    }

    #[test]
    fn identical_agents_split_evenly() {
        for n in 2..=6 {
            let names = vec!["lowest"; n];
            let config = Config {
                deals: 50,
                seed: 1,
                threads: 2,
            };
            let report = run(&lineup(&names), &config).unwrap();
            assert_eq!(report.standings.len(), 1);
            let s = &report.standings[0];
            assert_eq!(s.games, 50 * n as u64 * n as u64);
            // pooled returns of every seat always sum to one win per game
            assert!((s.relative - 1.0).abs() < 1e-9, "{report}");
        }
    }

    #[test]
    fn results_are_reproducible_and_thread_independent() {
        let names = ["cheapest", "random", "lowest", "biggest"];
        let a = run(
            &lineup(&names),
            &Config {
                deals: 40,
                seed: 9,
                threads: 1,
            },
        )
        .unwrap();
        let b = run(
            &lineup(&names),
            &Config {
                deals: 40,
                seed: 9,
                threads: 3,
            },
        )
        .unwrap();
        assert_eq!(a, b);
        let total: f64 = a.standings.iter().map(|s| s.win_share).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
}
