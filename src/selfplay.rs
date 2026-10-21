//! Self-play data generation for expert iteration: games are played by ISMCTS, and every
//! decision is recorded with the search's value estimate for each root move.
//!
//! Records are fixed-size runs of little-endian `f32`s, readable from Python with
//! `np.fromfile(path, "<f4").reshape(-1, RECORD_FLOATS)`:
//!
//! | Field | Floats |
//! |---|---|
//! | observation of the deciding player | `OBS_SIZE` |
//! | search mean value per action (NaN if illegal or unvisited) | `NUM_ACTIONS` |
//! | search visits per action | `NUM_ACTIONS` |
//! | the deciding player's final reward | 1 |

use std::io::{self, Write};
use std::sync::Mutex;
use std::thread;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::agents::ismcts::{IsmctsAgent, Params};
use crate::constants::{MAX_PLAYERS, MIN_PLAYERS};
use crate::game::encoding::OBS_SIZE;
use crate::game::{GameState, NUM_ACTIONS};

pub const RECORD_FLOATS: usize = OBS_SIZE + 2 * NUM_ACTIONS + 1;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub games: u64,
    pub seed: u64,
    /// Worker threads; 0 means one per available core.
    pub threads: usize,
    /// Player count for every game, or `None` for a uniformly random count per game.
    pub players: Option<usize>,
    /// For each player's first this-many decisions, sample the move in proportion to search
    /// visits instead of taking the most visited, so the data covers more positions.
    pub explore_moves: usize,
}

/// Plays one game, appending a record per decision to `out`.
fn play_game(params: &Params, config: &Config, game: u64, out: &mut Vec<f32>) {
    let mut rng = SmallRng::seed_from_u64(config.seed ^ game.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let n = config
        .players
        .unwrap_or_else(|| rng.random_range(MIN_PLAYERS..=MAX_PLAYERS));
    let mut state = GameState::new(n, rng.random_range(0..n), &mut rng);
    let mut agent = IsmctsAgent::with_params(params.clone(), rng.random());
    let mut decisions = [0usize; MAX_PLAYERS];
    let first = out.len();
    let mut movers = Vec::new();

    while !state.is_over() {
        let seat = state.current_player();
        let legal = state.legal_actions();
        let obs = state.observe(seat);
        let stats = agent.search(&obs);

        let mut record = vec![f32::NAN; RECORD_FLOATS];
        obs.encode(&mut record[..OBS_SIZE]);
        let (values, rest) = record[OBS_SIZE..].split_at_mut(NUM_ACTIONS);
        let visits = &mut rest[..NUM_ACTIONS];
        visits.fill(0.0);
        for m in &stats {
            values[m.combo.action_index()] = m.mean;
            visits[m.combo.action_index()] = m.visits as f32;
        }
        out.extend_from_slice(&record);
        movers.push(seat);

        let combo = if legal.len() == 1 {
            legal[0]
        } else if decisions[seat] < config.explore_moves {
            let total: u32 = stats.iter().map(|m| m.visits).sum();
            let mut x = rng.random_range(0..total.max(1));
            stats
                .iter()
                .find(|m| {
                    let hit = x < m.visits;
                    x = x.saturating_sub(m.visits);
                    hit
                })
                .map_or(legal[0], |m| m.combo)
        } else {
            stats
                .iter()
                .max_by_key(|m| m.visits)
                .map_or(legal[0], |m| m.combo)
        };
        decisions[seat] += 1;
        state.apply(combo).expect("search picks legal moves");
    }

    for (i, &seat) in movers.iter().enumerate() {
        out[first + (i + 1) * RECORD_FLOATS - 1] = params.reward.of(&state, seat);
    }
}

/// Plays `config.games` self-play games with ISMCTS, writing each game's records to `out` as
/// soon as it finishes. `progress` is called after every game with the games and records written
/// so far. Returns the number of records written.
///
/// # Errors
/// If writing fails; games already written stay in `out`.
pub fn generate<W: Write + Send>(
    params: &Params,
    config: &Config,
    out: &mut W,
    progress: &(dyn Fn(u64, u64) + Sync),
) -> io::Result<u64> {
    let threads = match config.threads {
        0 => thread::available_parallelism().map_or(1, |p| p.get()),
        t => t,
    }
    .min(config.games.max(1) as usize);
    // (writer, games written, records written)
    let shared = Mutex::new((out, 0u64, 0u64));
    thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let shared = &shared;
                scope.spawn(move || -> io::Result<()> {
                    let mut buf = Vec::new();
                    for game in (t as u64..config.games).step_by(threads) {
                        buf.clear();
                        play_game(params, config, game, &mut buf);
                        let bytes: Vec<u8> = buf.iter().flat_map(|x| x.to_le_bytes()).collect();
                        let mut guard = shared.lock().expect("writer lock poisoned");
                        let (writer, games, records) = &mut *guard;
                        writer.write_all(&bytes)?;
                        *games += 1;
                        *records += (buf.len() / RECORD_FLOATS) as u64;
                        progress(*games, *records);
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().expect("self-play worker panicked"))
    })?;
    let (writer, _, records) = shared.into_inner().expect("writer lock poisoned");
    writer.flush()?;
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_well_formed() {
        let params = Params {
            iterations: 30,
            ..Params::default()
        };
        let config = Config {
            games: 3,
            seed: 1,
            threads: 2,
            players: Some(3),
            explore_moves: 2,
        };
        let mut bytes = Vec::new();
        let records = generate(&params, &config, &mut bytes, &|_, _| {}).unwrap();
        assert!(records > 0);
        assert_eq!(bytes.len() as u64, records * RECORD_FLOATS as u64 * 4);
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for record in floats.chunks_exact(RECORD_FLOATS) {
            let values = &record[OBS_SIZE..OBS_SIZE + NUM_ACTIONS];
            let visits = &record[OBS_SIZE + NUM_ACTIONS..OBS_SIZE + 2 * NUM_ACTIONS];
            assert_eq!(visits.iter().sum::<f32>(), 30.0);
            for (v, n) in values.iter().zip(visits) {
                assert_eq!(v.is_nan(), *n == 0.0);
            }
            assert!(record[RECORD_FLOATS - 1].is_finite());
        }
    }
}
