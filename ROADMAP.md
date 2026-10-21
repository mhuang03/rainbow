# Agent roadmap

The plan for building and evaluating Rainbow agents, in build order. Each milestone ends with the
agent in the tournament ladder and its results recorded under [Results](#results).

The design draws on research into climbing games (Dou Dizhu, GuanDan, Big 2) and trick-taking
games (Skat, Bridge); see [References](#references).

## Milestone 0: Evaluation infrastructure

- [x] **Duplicate tournament harness** (`arena`): plays each deal once per seat rotation so every
      lineup sees the same cards, runs on all cores, and reports mean return with standard error,
      win share and mean points per agent.
- [x] **Agent registry and CLI**: `rainbow arena -p 4 -d 1000 economist random random random`.
- [x] Agent `name()` for reports.

## Milestone 1: Baselines

- [x] `random`: uniformly random legal move (exists).
- [x] `lowest`: always plays its lowest single, conserving cards to stay in the game.
- [x] `biggest`: always plays its strongest combo, burning cards for the top pile.
- [x] `cheapest`: plays the cheapest combo that currently wins the trick, otherwise its lowest
      single.

## Milestone 2: Hand-written strategy ("Economist")

Scores every legal combo as expected pile value, minus card cost, minus hand-structure damage,
plus an endgame term. All weights live in a `Params` struct so they can be tuned in the arena.

- [x] **Expected pile**: exact rank against plays already made; the chance each later player
      beats the combo, estimated by sampling their hands from the unknown cards.
- [x] **Card cost**: cards beyond what's needed to last until the game is expected to end are
      nearly free; cards below that cost about one trick's pile each.
- [x] **Structure damage**: penalty for breaking pairs, sets and runs.
- [x] **Endgame**: when running out would end the game, play to end it only if winning.
- [x] Weights tuned by random search and hill climbing against a mixed field
      (`examples/tune_economist.rs`).
- [ ] Beat every baseline in 3-, 4- and 6-player games. Done for 2, 4 and 6 players; level with
      `cheapest` in 3-player games. Ideas: model which pile values the played cards will create
      for the next trick ("feeding"), and lead-type choice.

## Milestone 3: Sampled-deal Monte Carlo (PIMC)

- [x] For each legal move, play it out on K sampled deals (`Observation::determinize`) with a
      fast rollout policy; pick the move with the best mean result. The same deals are reused for
      every move to reduce variance.
- [x] Shared `Rollout` policies over the full state (random, cheapest with epsilon-random moves).
- [x] Objective option: win share plus a weighted score margin.
- [x] Registry specs with options, e.g. `pimc:k=128,rollout=cheapest,eps=0.3,margin=0`.

## Milestone 4: Information-set MCTS

- [x] Single-observer ISMCTS with max-n backups (each player maximises its own reward) and
      availability-count UCB.
- [x] Compare rollout policies and reward shapes; tune the exploration constant.

## Milestone 5: Deep Monte Carlo (DouZero-style)

- [x] Python training loop (PyTorch) with parallel self-play actors using the bindings.
- [x] Q(observation, action) network with action features (card counts per rank).
- [x] Rust or Python inference agent entered in the ladder.

## Milestone 6: Play history

- [x] `GameState` and `Observation` record every play, the opening point cards and the first
      leader; `GameState::from_deal` starts a game from an explicit deal.
- [x] Observation encoding adds each relative seat's played cards (172 features).
- [x] Retrain DMC with history features (v2, 60 minutes) and compare with v1.

## Milestone 7: Inference-weighted determinization

- [x] `Observation::start_of` rebuilds a sampled world's starting deal so its history can be
      replayed.
- [x] Candidate deals weighted by how likely an opponent model finds the opponents' recent plays
      (`cheapest`-with-noise or a DMC softmax), then sampled by weight in ISMCTS
      (`ismcts:infer=cheapest` or `infer=<model path>`).
- [x] Evaluate the DMC opponent model.
- [x] Hand-prediction network (`train_belief.py`, `ismcts:belief=`) and weighted determinization.
- [ ] Evaluate belief-weighted deals.

## Milestone 8: Depth-limited minimax over sampled deals

- [x] Paranoid alpha-beta and max-n search per sampled deal, with exact endgames
      (`minimax:mode=paranoid|maxn,depth=3,k=16`).
- [x] Pluggable leaf evaluation: deterministic `cheapest` playouts or a DMC network.
- [x] Best-reply search mode (`minimax:mode=brs`, BRS+).
- [ ] Evaluate DMC leaves and BRS.

## Milestone 9: Masked Double DQN

- [x] `python/train_dqn.py`: Q(observation) over all 75 actions with masking, Double DQN targets,
      Polyak target network, dense per-decision point rewards plus the final win share.
- [x] `dqn:path=...` Rust agent.
- [ ] Train and evaluate.

## Milestone 10: PPO with a perfect-information critic

- [x] `python/train_ppo.py`: masked policy over observations, value network over full information
      (`GameState::encode_full`), Monte Carlo advantages, clipped objective, optional oracle
      guiding (`--oracle-fraction`).
- [x] `ppo:path=...` Rust agent.
- [ ] Train and evaluate, with and without oracle guiding.

## Milestone 11: Network-guided ISMCTS

- [x] DMC leaf values (`ismcts:leaf=<model>`) and PUCT priors (`ismcts:prior=<model>`).
- [x] Evaluate.
- [x] Expert iteration tooling: `rainbow selfplay` records search values; `train_exit.py`
      fine-tunes the network on them.
- [ ] Run expert-iteration rounds and evaluate.

## Milestone 12: NFSP

- [x] `python/train_nfsp.py`: Double DQN best response plus a reservoir-trained average policy;
      `nfsp:path=...` samples from the average policy.
- [ ] Train and evaluate.

## Tooling added along the way

- [x] `VecEnv`: batched Python environment, 11x faster stepping than per-game calls; all
      trainers use it.
- [x] `rainbow ladder`: ranks a pool of agents over many random lineups.
- [x] `ismcts:ms=`: per-move time budget, for compute-matched comparisons.
- [x] `Game` pickling and `GameState::to_bytes` / `from_bytes`.
- [x] Auxiliary hidden-hand loss for DMC (`train_dmc.py --aux-weight`).
- [x] `scripts/evaluate.sh` (standard table) and `scripts/expert_iteration.sh` (automated rounds).

## Later

- **Larger DMC networks** and longer runs.
- **Hand-prediction as an auxiliary loss** for the DMC and PPO networks.

## Results

Ladder results are added here as agents land. "Relative" is win share times the player
count, so 1.0 is an average player. All runs are duplicate tournaments (`rainbow arena`, seed 0).

### Baselines (5000 deals)

| Lineup | Relative win share |
|---|---|
| 2p: cheapest, random | cheapest 1.47, random 0.53 |
| 3p: cheapest, lowest, biggest | cheapest 2.72, lowest 0.27, biggest 0.01 |
| 4p: cheapest, lowest, biggest, random | cheapest 2.43, random 1.43, biggest 0.09, lowest 0.05 |
| 6p: cheapest x2, lowest x2, biggest, random | cheapest 2.01, random 1.50, biggest 0.35, lowest 0.06 |

### Economist (3000 deals, tuned weights)

| Lineup | Economist relative win share |
|---|---|
| 2p vs cheapest | 1.45 |
| 3p vs cheapest x2 | 1.00 |
| 4p vs cheapest x3 | 1.34 |
| 6p vs cheapest x5 | 1.39 |
| 4p vs cheapest, random, lowest | 3.34 |

Untuned weights scored about 1.06 against `cheapest` on average. Tuning against `cheapest` alone
overfitted (it learned that opponents rarely contest), so the final weights were tuned against a
mixed field.

### PIMC (400 deals; k=64, random rollouts)

| Lineup | PIMC relative win share |
|---|---|
| 2p vs cheapest | 1.87 |
| 2p vs economist | 1.80 |
| 3p vs cheapest x2 | 1.47 |
| 3p vs economist x2 | 0.95 |
| 4p vs economist x3 | 0.98 |
| 6p vs economist x5 | 1.35 |

Rollout policy matters more than sample count. With `cheapest` rollouts, PIMC assumes every
opponent plays like `cheapest`: it scores 3.05 against three `cheapest` players but only 0.38
against two Economists, and k=128 is no better than k=32. Random rollouts are less biased, and
then more samples help (k=96 reaches parity with the Economist in 3- and 4-player games). Fixing
this properly means modelling opponents in the search itself (ISMCTS) or in the rollouts.

### ISMCTS (300 deals; 1000 iterations, c=0.7, cheapest rollouts with eps=0.3)

| Players | vs Economist | vs PIMC |
|---|---|---|
| 2 | 1.81 | 1.14 |
| 3 | 1.47 | 1.35 |
| 4 | 1.41 | 1.31 |
| 6 | 1.57 | - |

In a mixed 4-player table it scores ismcts 1.75, economist 0.98, pimc 0.93, cheapest 0.34.

Sweep against three Economists (4p, 400 deals, error about 0.05): random rollouts 1.22,
cheapest rollouts with eps 0.2 / 0.3 / 0.5 / 0.7 gave 1.37 / 1.45 / 1.29 / 1.32; 3000 iterations
1.31 (with random rollouts). Unlike PIMC, ISMCTS prefers the sharper `cheapest` rollouts, because
the tree itself models opponents' near-term choices and the rollout only covers the distant
future. Exploration between 0.5 and 1.0 and margin weights from 0 to 0.5 made little difference.

### DMC (`models/dmc_v1.bin`, 400 deals)

Trained for 40 minutes on 4 CPU cores with `python/train_dmc.py` defaults: about 2M self-play
games, MLP 256-256-128, epsilon 0.1 to 0.02, all player counts. Checks against three Economists
during training: 0.59 at 5 minutes, 1.17 at 10, 1.33 at 20, 1.34 at 40.

| Players | vs Economist | vs ISMCTS |
|---|---|---|
| 2 | 1.78 | 0.86 |
| 3 | 1.59 | 1.40 |
| 4 | 1.36 | 1.50 |
| 6 | 1.43 | 1.19 (150 deals) |

Mixed 4-player table: dmc 1.52, ismcts 1.22, economist 0.72, pimc 0.54.

The network picks moves with one forward pass per decision, yet beats 1000-iteration ISMCTS
in 3- to 6-player games. ISMCTS still wins 2-player games, where search can plan through the
whole game. The model is small and saw only 40 minutes of training, so longer runs, bigger
networks and play-history features are the obvious next steps.

### Opponent inference (ISMCTS, 4p, 200-300 deals)

| Lineup | Inference ISMCTS | Plain ISMCTS |
|---|---|---|
| with 2 Economists, `infer=cheapest` | 1.33 | 1.08 |
| with 2 Economists, `infer=cheapest,ipower=0.5` | 1.33 | 1.06 |
| with 2 Economists, `infer=cheapest,iwindow=30` | 1.39 | 1.11 |
| 3p vs 2 plain ISMCTS, `iwindow=30` | 0.99 | 1.01 |
| 2p vs plain ISMCTS, `iwindow=30` | 1.03 | 0.97 |

Inference helps when the opponent model fits the opponents: the `cheapest` model captures how
the Economist takes cheap wins, and gains about 0.25-0.3 (about three standard errors). Against
ISMCTS opponents it does nothing, because they don't play like `cheapest`.

### Minimax with `cheapest` leaves (4p vs 3 Economists, 40-50 deals)

| Variant | Relative |
|---|---|
| paranoid, depth 3, k=16 | 0.14 |
| paranoid, depth 2, k=8 | 0.17 |
| max-n, depth 2, k=8 | 0.30 |
| max-n, depth 1, k=8 | 0.20 |

Badly beaten. Even depth 1 (sampled deals plus a deterministic `cheapest` playout) loses, so the
leaf evaluation is the problem, not the search rule: it is the same biased opponent model that
sank PIMC with `cheapest` rollouts, compounded by determinized search assuming everyone sees all
cards. Paranoid is worse still because with three colluding opponents almost every move looks
lost.

### DMC v2 (`models/dmc_v2.bin`, 400 deals)

Same setup as v1 plus per-seat played-card features, trained for 60 minutes (2.4M games) while
other jobs shared the CPU. v1 can no longer be loaded (the observation size changed), so the
comparison is through common opponents:

| Players | v1 vs Economist | v2 vs Economist | v1 vs ISMCTS | v2 vs ISMCTS |
|---|---|---|---|---|
| 2 | 1.78 | 1.72 | 0.86 | 0.73 |
| 3 | 1.59 | 1.63 | 1.40 | 1.44 |
| 4 | 1.36 | 1.35 | 1.50 | 1.56 |
| 6 | 1.43 | 1.49 | 1.19 | - |

History features changed little at this budget; the unknown-card features already carry most of
what card counting gives.

### Network-guided ISMCTS (4p vs 3 Economists, 200 deals)

| Agent | Relative |
|---|---|
| ismcts (1000 iterations, rollouts) | 1.41 |
| dmc v2 alone | 1.35 |
| ismcts n=200, DMC priors (PUCT) | 1.39 |
| ismcts n=1000, DMC priors | 1.74 |
| ismcts n=200, DMC leaf values | 2.00 |
| ismcts n=200, DMC priors and leaf values | **2.20** |

ISMCTS with DMC priors (n=1000) also beats three copies of the DMC network itself: 1.16 against
0.95. Leaf values matter most: replacing rollouts with the network's estimate doubles the win
share at a fifth of the iterations. The combination is the strongest agent so far.

Opponent inference with the DMC network as the opponent model (4p, with plain ISMCTS and two
Economists): inference 1.26, plain 1.12. Smaller than the `cheapest` model's gain against
Economists, but positive.

### Learned agents: training curves (relative vs 3 Economists, 4p, 200-deal checks)

All runs used one CPU thread each while sharing 4 cores with several other jobs, so wall-clock
times overstate what the methods need.

| Minutes | DQN | PPO v1 (Monte Carlo advantages) | NFSP |
|---|---|---|---|
| 9 | 0.36 | 0.13 | 0.25 |
| 18 | 0.60 | 0.15 | 0.27 |
| 27 | 0.60 | 0.17 | 0.32 |
| 36 | 0.80 | 0.16 | 0.60 |
| 45 | 0.96 | 0.17 | - |

DMC reached 1.17 after 10 minutes on an idle machine, so DMC's action-feature Q-network learns
far faster than DQN's one-output-per-action head here, matching DouZero's finding. PPO v1 learns
(it beats random play easily) but only made 343 updates in 45 minutes and stays weak; PPO v2 adds
GAE and league opponents.

### Playing style (`python/analyze_play.py`, 4p, 300 games, one seat each)

| Agent | Win share | Cards/play | Points/card | Leads a set | Leads a run | Lead size | Wins when last | Cards left at end |
|---|---|---|---|---|---|---|---|---|
| dmc v2 | 0.611 | 1.64 | 3.51 | 0.19 | 0.20 | 1.53 | 0.42 | 0.2 |
| economist | 0.246 | 1.50 | 3.56 | 0.18 | 0.08 | 1.42 | 0.41 | 1.4 |
| cheapest | 0.136 | 1.52 | 3.28 | 0.00 | 0.00 | 1.00 | 0.69 | 1.1 |
| random | 0.007 | 1.73 | 2.56 | 0.30 | 0.26 | 1.82 | 0.36 | 0.2 |

The network trades cards for points no more efficiently than the Economist, but it leads runs
far more often and finishes with almost no cards left. Making surplus cards free in the
Economist (`economist:pressure=true`) made it worse (0.83-0.91 relative against default
Economists at 3-6 players), so the network's edge is in choosing which tricks to spend on, not in
spending more. Retuning the Economist for 3-player games alone gained only 0.05 (within noise).

Winning the trick cheaply matters most. Hoarding cards (`lowest`) and dumping them (`biggest`) are
both terrible, and are beaten even by random play.

## References

- DouZero (Zha et al., ICML 2021), DouZero+ (2022), PerfectDou (Yang et al., NeurIPS 2022)
- DanZero (Lu et al., 2022), DanZero+ (Zhao et al., 2023)
- Big 2 self-play PPO (Charlesworth, 2018)
- Suphx (Li et al., 2020)
- Information Set MCTS (Cowling, Powley and Whitehouse, 2012)
- Understanding PIMC (Long, Sturtevant, Buro and Furtak, AAAI 2010)
- Policy Based Inference in Trick-Taking Card Games (Rebstock et al., 2019)
- Best-Reply Search (Schadd and Winands, 2011)
- NFSP (Heinrich and Silver, 2016), ReBeL (Brown et al., 2020), Student of Games (Schmid et al., 2023)
