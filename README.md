# Rainbow

This is an unofficial, open-source Rust engine for the card game [Rainbow](https://www.allplay.com/board-games/rainbow) by [Allplay](https://www.allplay.com/). Please support the original creators by purchasing the physical game from their website.

The goal of this project is to find optimal strategies and to see how well search algorithms (ISMCTS, minimax) and learned agents (e.g. deep Q-learning) play it.

## Rules coverage

All official rules for 2–6 players are implemented. They were checked against the [rules summary on Board Game Arena](https://en.doc.boardgamearena.com/Gamehelprainbowallplay):

- Deal: 14 cards each with 3–4 players (15 set aside with 3), 11 with 5, 9 with 6. One face-up point card per player goes in the middle.
- Combos: singles, sets (2+ of a rank) and runs (2+ consecutive ranks). After a set only singles and sets may follow; after a run only singles and runs.
- Trick resolution: bigger combos (more cards) claim point piles first. Ties go to the higher top card, then to the earlier play. The winner leads next, or the next player clockwise if the winner has no cards.
- Next point piles: the played cards are grouped into pairs and singles, and the lowest piles are discarded down to one per player.
- Players with no cards sit out. The game ends after a trick in which two or more players have run out.
- 2-player variant: 4 middle cards, a 14-card hand plus a 14-card face-down reserve each, and two combos per player per trick. A player whose hand empties picks up their reserve at once. Running out a second time ends the game, and the opponent gets 1 point per card left in hand.

Claiming point piles is resolved automatically: each player takes the most valuable remaining pile. Claimed piles only add to the claimant's score, so taking anything less can never help.

The official 2-player text could not be found online, so two details are interpretations: turns alternate within a trick (A, B, A, B), and when the game ends mid-trick, the plays already made still score.

## Engine

```rust
use rainbow::{GameState, Combo};
use rand::{SeedableRng, rngs::SmallRng};

let mut game = GameState::new(4, 0, &mut SmallRng::seed_from_u64(1));
while !game.is_over() {
    let combo: Combo = game.legal_actions()[0];
    game.apply(combo)?; // rejects illegal moves
}
println!("{:?} won with {:?}", game.winners(), game.scores());
```

- `GameState` is 400 bytes with no heap data (including the full play history), and is `Clone + Eq + Hash`, so it is cheap to copy for search and usable as a transposition-table key.
- There are 75 actions. `Combo::action_index` and `Combo::from_action_index` map combos to and from that fixed space, and `legal_action_mask()` returns a `u128` bitmask for masking network outputs.
- `observe(seat)` returns an `Observation`: that player's own hand plus all public information.
  - `Observation::encode` writes a 172-float feature vector, with seats relative to the observer. The layout is documented in `game::encoding`.
  - `Observation::determinize` samples a full `GameState` consistent with what the player knows (the determinization step of ISMCTS).
- `returns()`: winners split 1.0 equally. `scores()` gives raw points.
- `agents::Agent` and `game::run_game` play a game between agents.

`cargo run --release -- bench [num_players] [num_games] [seed]` benchmarks random playouts. It runs about 200k games/s (about 7M moves/s) on one core.

## Agents and the arena

| Agent | Description |
|---|---|
| `random` | Uniformly random legal move. |
| `lowest` | Always plays its lowest single. |
| `biggest` | Always plays its strongest combo. |
| `cheapest` | Plays the cheapest combo that currently wins the trick, otherwise its lowest single. |
| `economist` | Hand-written strategy: scores each combo by expected pile, card cost, hand-structure damage and endgame effects. |
| `pimc` | Samples deals consistent with what it knows and rolls out every move on each. |
| `ismcts` | Information-set MCTS with max-n backups. Options add network move priors (`prior=`), network leaf values (`leaf=`), belief-weighted deals (`belief=`) and opponent inference (`infer=`). With `prior` and `leaf` from `models/dmc_v2.bin` it is the strongest agent. |
| `minimax` | Paranoid alpha-beta or max-n search over sampled deals, with exact endgames. |
| `dmc` | A DouZero-style network trained by self-play (`dmc:path=models/dmc_v2.bin`). The strongest agent that doesn't search. |
| `dqn` | A masked Double DQN network. |
| `ppo` | A PPO policy trained with a full-information critic. |
| `nfsp` | An NFSP average policy (played by sampling). |

`rainbow arena` runs a duplicate tournament: every deal is replayed once per seat rotation, on all cores. Agents take options after a colon:

```sh
cargo run --release -- arena -d 1000 ismcts economist pimc:k=128,rollout=random cheapest
```

[ROADMAP.md](ROADMAP.md) tracks planned agents and records tournament results.

`rainbow selfplay` plays games with network-guided ISMCTS and records the search's value for every root move, as training data for expert iteration.

## Python bindings

The `python/` crate builds a `rainbow_engine` extension module with PyO3 and maturin:

```sh
cd python
pip install maturin
maturin develop --release
pytest tests
```

```python
import rainbow_engine as re

game = re.Game(4, seed=0)
while not game.is_over:
    obs = game.observation(game.current_player)   # float32[OBS_SIZE]
    mask = game.legal_action_mask()                # bool[NUM_ACTIONS]
    game.step(game.legal_actions()[0])
print(game.scores(), game.returns())
```

- `Game.clone()`, `determinize(player, seed)` and `rollout(seed)` support MCTS. `rollout` runs without holding the GIL.
- `VecEnv(num_envs, seed, players, opponents, learners)` steps many games per call and auto-resets finished ones (about 1M steps/s with random actions). With `opponents` (Rust agent specs), the environment plays the non-learner seats itself, for training against a league.
- `batch_observations(games)` fills `(B, OBS_SIZE)` and `(B, NUM_ACTIONS)` arrays for a list of `Game`s.
- `Game` pickles compactly (starting deal plus history), and `to_bytes` / `from_bytes` expose the same encoding.
- `full_observation(player)` and `batch_full_observations(games)` add every hidden hand, for training value networks.
- `history()` lists every play so far.
- `RustAgent(spec, seed)` plays any agent from the Rust registry, e.g. `RustAgent("ismcts").act(game)`.
- `action_features()` returns the per-action feature table used by the DMC network.
- Type hints are in `python/rainbow_engine.pyi`.

### Training a DMC agent

`python/train_dmc.py` trains a DouZero-style Q(observation, action) network by self-play on CPU with PyTorch, and exports it in a format the Rust `dmc` agent loads:

```sh
pip install torch numpy
cargo build --release
cd python && python train_dmc.py --minutes 30 --out models/dmc.bin
../target/release/rainbow arena dmc:path=python/models/dmc.bin economist economist economist
```

### Other trainers

All trainers play many games at once through `VecEnv`, check progress in the Rust arena, and export networks the Rust agents load. Pass `--threads` to share cores between runs; `train_dmc.py` and `train_ppo.py` also take `--opponents` to train against fixed Rust agents.

| Script | Trains |
|---|---|
| `train_dmc.py` | Deep Monte Carlo Q(observation, action) network (`dmc`). |
| `train_dqn.py` | Masked Double DQN over all 75 actions, with per-decision point rewards (`dqn`). |
| `train_ppo.py` | PPO policy with a value network that sees every hand; `--oracle-fraction` adds oracle guiding (`ppo`). |
| `train_nfsp.py` | NFSP best-response and average-policy networks (`nfsp`). |
| `train_belief.py` | Hand-prediction network for `ismcts:belief=` (supervised, from `Game.full_observation`). |
| `train_exit.py` | Expert iteration: fine-tunes a DMC network on `rainbow selfplay` search values. |
