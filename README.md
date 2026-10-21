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

- `GameState` is 148 bytes with no heap data, and is `Clone + Eq + Hash`, so it is cheap to copy for search and usable as a transposition-table key.
- There are 75 actions. `Combo::action_index` and `Combo::from_action_index` map combos to and from that fixed space, and `legal_action_mask()` returns a `u128` bitmask for masking network outputs.
- `observe(seat)` returns an `Observation`: that player's own hand plus all public information.
  - `Observation::encode` writes a 136-float feature vector, with seats relative to the observer. The layout is documented in `game::encoding`.
  - `Observation::determinize` samples a full `GameState` consistent with what the player knows (the determinization step of ISMCTS).
- `returns()`: winners split 1.0 equally. `scores()` gives raw points.
- `agents::Agent` and `game::run_game` play a game between agents. `RandomAgent` is included as a baseline.

`cargo run --release -- [num_players] [num_games] [seed]` benchmarks random playouts. It runs about 200k games/s (about 7M moves/s) on one core.

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
- `batch_observations(games)` fills `(B, OBS_SIZE)` and `(B, NUM_ACTIONS)` arrays for vectorised training.
- Type hints are in `python/rainbow_engine.pyi`.
