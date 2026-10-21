//! Python bindings for the Rainbow engine.
//!
//! Actions are integers in `0..NUM_ACTIONS` (see `action_to_str`), and observations are
//! `float32` vectors of length `OBS_SIZE` with seats relative to the observer.
//!
//! Integer lists are returned as `u32` because PyO3 converts `Vec<u8>` to `bytes`.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rainbow::agents::{self, Agent};
use rainbow::constants::{MAX_PLAYERS, MIN_PLAYERS};
use rainbow::game::encoding::{self, ACTION_FEATURES, FULL_OBS_SIZE, HIDDEN_SIZE, OBS_SIZE};
use rainbow::{Combo, ComboKind, GameState, NUM_ACTIONS, TrickType};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn make_rng(seed: Option<u64>) -> SmallRng {
    seed.map_or_else(rand::make_rng, SmallRng::seed_from_u64)
}

fn combo(action: usize) -> PyResult<Combo> {
    Combo::from_action_index(action)
        .ok_or_else(|| PyValueError::new_err(format!("action must be in 0..{NUM_ACTIONS}")))
}

fn mask_bits(mask: u128) -> [bool; NUM_ACTIONS] {
    std::array::from_fn(|i| mask & (1 << i) != 0)
}

/// A game of Rainbow.
///
/// Game(num_players, seed=None, leader=0)
#[pyclass(module = "rainbow_engine", eq)]
#[derive(Clone, PartialEq)]
struct Game {
    state: GameState,
}

impl Game {
    fn seat(&self, player: usize) -> PyResult<usize> {
        if player < self.state.num_players() {
            Ok(player)
        } else {
            Err(PyIndexError::new_err(format!("no player {player}")))
        }
    }
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (num_players, seed=None, leader=0))]
    fn new(num_players: usize, seed: Option<u64>, leader: usize) -> PyResult<Self> {
        if !(MIN_PLAYERS..=MAX_PLAYERS).contains(&num_players) {
            return Err(PyValueError::new_err(format!(
                "num_players must be in {MIN_PLAYERS}..={MAX_PLAYERS}"
            )));
        }
        if leader >= num_players {
            return Err(PyValueError::new_err("leader must be a seat index"));
        }
        Ok(Self {
            state: GameState::new(num_players, leader, &mut make_rng(seed)),
        })
    }

    #[getter]
    fn num_players(&self) -> usize {
        self.state.num_players()
    }

    #[getter]
    fn current_player(&self) -> usize {
        self.state.current_player()
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.state.is_over()
    }

    #[getter]
    fn tricks_played(&self) -> usize {
        self.state.tricks_played()
    }

    /// "open", "set" or "run".
    #[getter]
    fn trick_type(&self) -> &'static str {
        match self.state.trick_type() {
            TrickType::Open => "open",
            TrickType::Set => "set",
            TrickType::Run => "run",
        }
    }

    /// Legal action indices for the current player.
    fn legal_actions(&self) -> Vec<usize> {
        self.state
            .legal_actions()
            .iter()
            .map(|c| c.action_index())
            .collect()
    }

    /// Boolean array of shape (NUM_ACTIONS,).
    fn legal_action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &mask_bits(self.state.legal_action_mask()))
    }

    fn is_legal(&self, action: usize) -> bool {
        Combo::from_action_index(action).is_some_and(|c| self.state.is_legal(c))
    }

    /// Plays `action` for the current player. Raises ValueError if it is illegal.
    fn step(&mut self, action: usize) -> PyResult<()> {
        self.state
            .apply(combo(action)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    fn __copy__(&self) -> Self {
        Clone::clone(self)
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        Clone::clone(self)
    }

    /// Compact encoding (starting deal and play history) for pickling and storage.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.state.to_bytes())
    }

    /// Rebuilds a game from `to_bytes`.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Ok(Self {
            state: GameState::from_bytes(data).map_err(PyValueError::new_err)?,
        })
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let py = slf.py();
        let from_bytes = slf.get_type().getattr("from_bytes")?;
        Ok((from_bytes, (slf.borrow().to_bytes(py),)))
    }

    fn scores(&self) -> Vec<u32> {
        self.state.scores().iter().map(|&s| s.into()).collect()
    }

    /// Winners split 1.0 equally; all zeros until the game is over.
    fn returns(&self) -> Vec<f32> {
        self.state.returns().to_vec()
    }

    fn winners(&self) -> Vec<u32> {
        self.state.winners().iter().map(|&s| s.into()).collect()
    }

    /// Per-rank card counts (index 0 = rank 1) of `player`'s hand. Hidden information.
    fn hand(&self, player: usize) -> PyResult<Vec<u32>> {
        let counts = self.state.hand(self.seat(player)?).counts();
        Ok(counts.iter().map(|&c| c.into()).collect())
    }

    /// Values of the point piles in the middle, highest first.
    fn piles(&self) -> Vec<u32> {
        self.state.piles().iter().map(|&v| v.into()).collect()
    }

    /// (seat, action) for each play in the current trick.
    fn trick(&self) -> Vec<(u8, usize)> {
        self.state
            .trick()
            .iter()
            .map(|p| (p.seat, p.combo.action_index()))
            .collect()
    }

    /// (seat, action) for every play so far this game, in order.
    fn history(&self) -> Vec<(u8, usize)> {
        self.state
            .history()
            .iter()
            .map(|p| (p.seat, p.combo.action_index()))
            .collect()
    }

    /// `player`'s information set as a float32 array of shape (OBS_SIZE,).
    fn observation<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let obs = self.state.observe(self.seat(player)?);
        Ok(PyArray1::from_slice(py, &obs.encoded()))
    }

    /// `player`'s observation followed by every hand and reserve (seats relative to `player`),
    /// as a float32 array of shape (FULL_OBS_SIZE,). For training value networks only.
    fn full_observation<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mut out = [0.0; FULL_OBS_SIZE];
        self.state.encode_full(self.seat(player)?, &mut out);
        Ok(PyArray1::from_slice(py, &out))
    }

    /// A full game sampled uniformly from the states `player` cannot tell apart from this one.
    #[pyo3(signature = (player, seed=None))]
    fn determinize(&self, player: usize, seed: Option<u64>) -> PyResult<Self> {
        let obs = self.state.observe(self.seat(player)?);
        Ok(Self {
            state: obs.determinize(&mut make_rng(seed)),
        })
    }

    /// Plays uniformly random moves from a copy of this game to the end and returns its
    /// `returns()`. Runs without holding the GIL.
    #[pyo3(signature = (seed=None))]
    fn rollout(&self, py: Python<'_>, seed: Option<u64>) -> Vec<f32> {
        let mut state = self.state.clone();
        py.detach(move || {
            let mut rng = make_rng(seed);
            while !state.is_over() {
                let actions = state.legal_actions();
                let c = actions[rng.random_range(0..actions.len())];
                state.apply(c).expect("legal action");
            }
            state.returns().to_vec()
        })
    }

    fn __str__(&self) -> String {
        self.state.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Game(num_players={}, current_player={}, tricks_played={}, is_over={})",
            self.state.num_players(),
            self.state.current_player(),
            self.state.tricks_played(),
            if self.state.is_over() {
                "True"
            } else {
                "False"
            },
        )
    }
}

/// Observations and legal-action masks for each game's current player, as arrays of shape
/// (len(games), OBS_SIZE) and (len(games), NUM_ACTIONS).
#[pyfunction]
fn batch_observations<'py>(
    py: Python<'py>,
    games: Vec<PyRef<'py, Game>>,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>) {
    let obs = PyArray2::<f32>::zeros(py, [games.len(), OBS_SIZE], false);
    let masks = PyArray2::<bool>::zeros(py, [games.len(), NUM_ACTIONS], false);
    {
        let mut obs_rw = obs.readwrite();
        let mut masks_rw = masks.readwrite();
        let obs_slice = obs_rw.as_slice_mut().expect("new arrays are contiguous");
        let mask_slice = masks_rw.as_slice_mut().expect("new arrays are contiguous");
        for ((game, o), m) in games
            .iter()
            .zip(obs_slice.chunks_exact_mut(OBS_SIZE))
            .zip(mask_slice.chunks_exact_mut(NUM_ACTIONS))
        {
            let state = &game.state;
            state.observe(state.current_player()).encode(o);
            m.copy_from_slice(&mask_bits(state.legal_action_mask()));
        }
    }
    (obs, masks)
}

/// Features of every action, as a float32 array of shape (NUM_ACTIONS, ACTION_FEATURES):
/// card count per rank / 10, one-hot single/set/run, card count / 10, strength / max strength.
#[pyfunction]
fn action_features<'py>(py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
    let table = PyArray2::<f32>::zeros(py, [NUM_ACTIONS, ACTION_FEATURES], false);
    {
        let mut rw = table.readwrite();
        let slice = rw.as_slice_mut().expect("new arrays are contiguous");
        for (row, combo) in slice
            .chunks_exact_mut(ACTION_FEATURES)
            .zip(rainbow::game::ALL_COMBOS)
        {
            encoding::encode_action(combo, row);
        }
    }
    table
}

/// A Rust agent from the registry, e.g. RustAgent("ismcts:n=2000", seed=1).
#[pyclass(module = "rainbow_engine", unsendable)]
struct RustAgent {
    agent: Box<dyn Agent>,
}

#[pymethods]
impl RustAgent {
    #[new]
    #[pyo3(signature = (spec, seed=0))]
    fn new(spec: &str, seed: u64) -> PyResult<Self> {
        Ok(Self {
            agent: agents::create(spec, seed).map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.agent.name().to_owned()
    }

    /// The action this agent chooses for `game`'s current player.
    fn act(&mut self, game: &Game) -> PyResult<usize> {
        let state = &game.state;
        if state.is_over() {
            return Err(PyValueError::new_err("the game is over"));
        }
        let obs = state.observe(state.current_player());
        Ok(self.agent.act(&obs, &state.legal_actions()).action_index())
    }
}

/// Full-information observations (see `Game.full_observation`) for each game's current player,
/// as an array of shape (len(games), FULL_OBS_SIZE).
#[pyfunction]
fn batch_full_observations<'py>(
    py: Python<'py>,
    games: Vec<PyRef<'py, Game>>,
) -> Bound<'py, PyArray2<f32>> {
    let out = PyArray2::<f32>::zeros(py, [games.len(), FULL_OBS_SIZE], false);
    {
        let mut rw = out.readwrite();
        let slice = rw.as_slice_mut().expect("new arrays are contiguous");
        for (game, row) in games.iter().zip(slice.chunks_exact_mut(FULL_OBS_SIZE)) {
            game.state.encode_full(game.state.current_player(), row);
        }
    }
    out
}

/// Human-readable form of an action: "4" (single), "3x4" (three 4s) or "2-5" (run).
#[pyfunction]
fn action_to_str(action: usize) -> PyResult<String> {
    Ok(combo(action)?.to_string())
}

/// (kind, a, b) for an action: ("single", rank, 1), ("set", rank, count) or ("run", start, end).
#[pyfunction]
fn action_info(action: usize) -> PyResult<(&'static str, u8, u8)> {
    Ok(match combo(action)?.kind() {
        ComboKind::Single { rank } => ("single", rank, 1),
        ComboKind::Set { rank, count } => ("set", rank, count),
        ComboKind::Run { start, end } => ("run", start, end),
    })
}

#[pymodule]
fn rainbow_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<RustAgent>()?;
    m.add_function(wrap_pyfunction!(action_features, m)?)?;
    m.add_function(wrap_pyfunction!(batch_observations, m)?)?;
    m.add_function(wrap_pyfunction!(batch_full_observations, m)?)?;
    m.add_function(wrap_pyfunction!(action_to_str, m)?)?;
    m.add_function(wrap_pyfunction!(action_info, m)?)?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("ACTION_FEATURES", ACTION_FEATURES)?;
    m.add("HIDDEN_SIZE", HIDDEN_SIZE)?;
    m.add("FULL_OBS_SIZE", FULL_OBS_SIZE)?;
    m.add("MIN_PLAYERS", MIN_PLAYERS)?;
    m.add("MAX_PLAYERS", MAX_PLAYERS)?;
    Ok(())
}
