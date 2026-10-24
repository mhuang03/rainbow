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
| `margin` | The Economist scoring piles by their *difference* from what the other players take (`economist:objective=margin`). `economist:adaptive=N` also learns each opponent's willingness to contest tricks from the play history. |
| `counter` | Card counter: samples deals of the unseen cards to judge which later opponents can beat each combo, and scores each combo by the pile it expects to finish on. The strongest hand-written agent; its fast variant drives search rollouts and priors. |
| `pimc` | Samples deals consistent with what it knows (optionally weighted by a belief network) and finishes every move on each with `counter` rollouts, or values it with a network (`leaf=`). |
| `ismcts` | Information-set MCTS with max-n backups. Options add network move priors (`prior=`), network leaf values (`leaf=`), belief-weighted deals (`belief=`) and opponent inference (`infer=`). With `prior` and `leaf` from a DMC network and `belief=models/belief_v3.bin` it is the strongest agent ("Guided ISMCTS"); it plays best steered by the player-count specialist at 3, 4 and 5 players (`models/dmc_spec3p_1.bin` to `dmc_spec5p_1.bin`) and by the distilled network otherwise, and at 3 and 4 players with a `leaf=` averaging the specialist and the distilled network. Without a network it uses `counter` rollouts and a `counter` move prior, which bring it close to the best networks. |
| `minimax` | Paranoid alpha-beta or max-n search over sampled deals, with exact endgames. |
| `dmc` | A DouZero-style network trained by self-play (`models/dmc_v2.bin`) and fine-tuned in nine league rounds, widened to 512-512-256 in round 6 (`models/dmc_league9.bin`). Several `+`-joined paths make an ensemble, and `python/distill_ensemble.py` distils one into a single network (`models/dmc_distilled1.bin`, which plays like the ensemble of rounds 8-10 and beats round 9 1.13 at 4p). `path2=` to `path6=` play another network in games of that many players: specialists trained on one player count (`train_dmc.py --players`) beat the distilled network at 2, 3 and 4 players (`models/dmc_spec2p_1.bin`, `dmc_spec3p_1.bin`, `dmc_spec4p_1.bin`) but not at 5 or 6. `counter=0.04` breaks near-ties with the Counter agent's card counting (1.07 against the network alone, 3-5 players); `temp=` samples its moves for weaker, more human opponents. With the specialists and `counter=0.04` it is the strongest agent that doesn't search. |
| `dmc_hist` | A DMC network that also sees the recent plays and each seat's rate of beating the trick when following (`train_dmc.py --history`). |
| `dmc:path=a.bin+b.bin` | An ensemble averaging several DMC networks' values. |
| `hybrid` | A DMC network that hands over to exact max-n minimax once few cards are left (`cards=`, `k=` deals). The browser's Hybrid plays the player-count specialists through `bycount:`, like `dmc`. |
| `distill` | A policy network trained to imitate guided ISMCTS's visit counts (`train_distill.py`, `models/distill_v5.bin`). |
| `dqn` | A masked Double DQN network. |
| `ppo` | A PPO policy trained with a full-information critic. |
| `nfsp` | An NFSP average policy (played by sampling). |
| `bycount` | A different agent for each player count: `bycount:3=<spec>\|4=<spec>\|rest=<spec>` (any specs, searches included). The browser's Guided ISMCTS uses it to pick its networks. |

`rainbow arena` runs a duplicate tournament: every deal is replayed once per seat rotation, on all cores. Agents take options after a colon:

```sh
cargo run --release -- arena -d 1000 ismcts economist pimc:k=128,rollout=random cheapest
```

`rainbow ladder` rates a pool of agents over many random lineups drawn from it, so each is judged against a varied field:

```sh
cargo run --release -- ladder -p 4 -l 100 -d 30 ismcts counter economist pimc cheapest dmc:path=models/dmc_distilled1.bin
```

[ROADMAP.md](ROADMAP.md) tracks planned agents and records tournament results.

`scripts/league_round.sh` runs one round of DMC league training, then rates the result against its starting network in mixed 4- and 2-player ladders (`WIDEN=` grows the network first; `HEAD_TO_HEAD=1` adds the old head-to-head check).

`rainbow selfplay` plays games with network-guided ISMCTS and records the search's value for every root move, as training data for expert iteration.

### Results

Current 4-player ladder over the browser's opponents (160 random lineups of 4 from 21 agents, 30
duplicate deals each; 1.0 is an even share of the wins), top and selected others:

| Agent | Relative |
|---|---|
| Guided ISMCTS: `bycount:` routing `ismcts:n=200` with `prior`, `leaf` from the 3-, 4- and 5-player specialists (else `dmc_distilled1`), `belief=models/belief_v3.bin` | 2.19 |
| `ismcts` (no network: `counter` rollouts and prior) | 1.91 |
| `dmc:path=models/dmc_distilled1.bin,path2=…,path3=…,path4=…,counter=0.04` | 1.81 |
| `hybrid` on the distilled network (now on the specialists too, like `dmc`) | 1.78 |
| `dmc:path=models/dmc_league7.bin+models/dmc_league8.bin+models/dmc_league9.bin` | 1.70 |
| `pimc` | 1.28 |
| `distill:path=models/distill_v5.bin` | 1.19 |
| `counter` | 0.96 |
| `economist` | 0.88 |
| `ppo:path=models/ppo_v2.bin` | 0.51 |
| `cheapest` | 0.42 |

A field with many weak agents spreads the scores out; in stronger fields Economist rates about
0.4. Agents are judged by ladders over mixed fields like this rather than head-to-head, so a gain
has to hold against many kinds of opponent.

[ROADMAP.md](ROADMAP.md) has the full results and what each experiment taught.

## Play in the browser

`server/` is a web server for playing Rainbow yourself, alone against the agents or online with
other people:

```sh
cargo run --release -p rainbow-server -- --port 8080 --models models
```

Open http://localhost:8080 and enter your name.

- **Against the computer:** pick the number of players and an opponent from 21, grouped by their
  measured rating from Beginner (Random, Cheapest, PPO) through Casual (Economist, Counter, PIMC,
  a loosened DMC) and Strong (DMC, Hybrid, the DMC ensemble, ISMCTS) to Expert (Guided ISMCTS);
  the *?* beside the menu explains the ratings. Or choose each seat separately. *Suggest* asks
  the strongest available agent for a move.
- **Watch the computer:** *Watch computer players* sets up a table of 2-6 agents of your choice
  and shows every hand as they play; pause, step one move at a time, or change the speed.
- **With friends:** create a room and share its four-letter code or link; a link opens straight
  to a join screen. In the lobby the host can give open seats to the computer, and during a
  game can hand an absent player's seat to it. A player who reloads or loses their connection
  gets their seat back, and while they're away the computer plays their turns after 30 seconds;
  anyone joining a full room watches. While the host is away, the next player present acts as
  host. Rooms have a chat.

When a game ends, a score chart shows how the points built up; close it to look back over the
table and the full trick history, then play again or leave.

Play by tapping cards and pressing *Play* (or keys `1`-`6`, `Enter` and `Esc`). *How to play*
explains the rules and controls; *Settings* has sound, the computer's speed (slow, normal or fast, in solo and watched games),
reduced motion and a light or dark theme.

To play with people on other machines, the port must be reachable from them, for example on a
LAN, through port forwarding, or behind a reverse proxy that passes WebSockets through (the page
connects to `/ws`). Agents wait briefly before moving so people can follow the game. `--pace 0`
removes the wait, and `--pace 2` doubles it. Network agents are listed only when their model
files are found in `--models`.

The server runs each room on its own thread, which owns the game and its agents. Each browser
only receives its own hand and legal moves plus the public state, so nobody can see hidden
cards.

`server/e2e/run.sh` runs browser tests against a fast-paced server (needs Node with Playwright,
and optionally axe-core for the WCAG 2.2 AA checks): a solo game, an online room with a reload
mid-trick, keyboard-only play, a watched game paused and stepped, both colour themes and a
360-pixel-wide phone.

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
- `VecEnv(num_envs, seed, players, opponents, learners, pfsp)` steps many games per call and auto-resets finished ones (about 1M steps/s with random actions). With `opponents` (Rust agent specs), the environment plays the non-learner seats itself, for training against a league; `pfsp` weights the draw towards the opponents the learners beat least, and `opponent_win_rates()` reports those rates.
- `batch_observations(games)` fills `(B, OBS_SIZE)` and `(B, NUM_ACTIONS)` arrays for a list of `Game`s.
- `python/rainbow_gym.py` wraps a game as a Gymnasium environment (one learner seat against Rust agents, with `action_masks()` for MaskablePPO-style libraries).
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
| `train_distill.py` | Policy network imitating guided ISMCTS's visit counts from `rainbow selfplay` (`distill`). |
| `distill_ensemble.py` | Distils a DMC ensemble into one network, regressing every legal move's ensemble value in ensemble self-play. |

`train_dmc.py` also takes `--init` (continue a network), `--widen` (grow it first, function-preserving),
`--players N` (train on N-player games only, for the player-count specialists),
`--history` (public-history inputs) and, with `--opponents`, `--pfsp P` (draw opponents with weight
(1 - win rate)^P, favouring those the learner beats least). `scripts/league_round.sh` runs one league
round with these and a head-to-head check. `analyze_play.py` compares how agents play (leads,
trick wins, card economy, endgames). `train_belief.py --history` trains a belief network on
the public history as well.
