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
- [x] Evaluate belief-weighted deals.

## Milestone 8: Depth-limited minimax over sampled deals

- [x] Paranoid alpha-beta and max-n search per sampled deal, with exact endgames
      (`minimax:mode=paranoid|maxn,depth=3,k=16`).
- [x] Pluggable leaf evaluation: deterministic `cheapest` playouts or a DMC network.
- [x] Best-reply search mode (`minimax:mode=brs`, BRS+).
- [x] Evaluate DMC leaves and BRS.

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
- [x] Train and evaluate (v1 Monte Carlo advantages; v2 GAE and league opponents).
- [x] Oracle guiding run (v3: v2's setup plus `--oracle-fraction 0.5`): 0.43 at 10 minutes, 0.55
      at 20, 0.45 at 30 when the oracle inputs reached zero, then 0.76 at 40 against v2's 0.62.
      That run was cut short by a container restart. A full rerun reached 0.39 / 0.50 / 0.63 /
      0.62 / 0.74 / 0.87 at 10-60 minutes against v2's 0.51 / 0.56 / 0.58 / 0.62 / 0.65 / 0.89:
      no measurable benefit from oracle guiding at this budget.

## Milestone 11: Network-guided ISMCTS

- [x] DMC leaf values (`ismcts:leaf=<model>`) and PUCT priors (`ismcts:prior=<model>`).
- [x] Evaluate.
- [x] Expert iteration tooling: `rainbow selfplay` records search values; `train_exit.py`
      fine-tunes the network on them.
- [x] Round 1: 1500 guided self-play games (n=200, v2 prior and leaf; 46,557 decisions,
      436,630 state-move rows), 8 epochs of fine-tuning from v2. Validation error against search
      values fell from 0.0248 to 0.0117, but the network got worse: head-to-head against v2 it
      scores 0.99 (2p) and 0.83 (4p), though 1.37 against v2's 1.31 versus three Economists.
- [x] Round 1b on the same data: the most-visited move's target blends in the game result
      (`--return-weight 0.5`), 2 epochs. 1.54 against v2's 1.22 versus three Economists (100
      deals), but still worse head-to-head against v2: 0.94 (2p) and 0.92 (4p).
- [ ] At this budget expert iteration raises scores against the Economist while losing to its
      own predecessor; league training (+0.22 head-to-head) is the better use of CPU. Worth
      revisiting with stronger searches (more iterations) and much more data per round.

## Milestone 12: NFSP

- [x] `python/train_nfsp.py`: Double DQN best response plus a reservoir-trained average policy;
      `nfsp:path=...` samples from the average policy.
- [ ] Train and evaluate.

## Milestone 13: Second round of agents

Hand-written:
- [x] `margin` (`economist:objective=margin`): scores a claimed pile by how far it is above
      the average of what the other players in the trick take, not by its raw value. Default
      weights: 1.09 / 0.78 / 0.93 / 0.90 against Economists at 2 / 3 / 4 / 6 players, 1.59 / 1.05 /
      1.27 / 1.29 against `cheapest`. It helps head-to-head (2p), where a pile is zero-sum, and
      hurts with more players. Weights tuned for it (`tune_economist ... margin`, held-out 1.56 ->
      1.61 on the tuning field) did worse against Economists (2p 0.92), so they aren't used.
- [x] `economist:adaptive=N`: per-opponent willingness to contest tricks, estimated from the
      play history with a prior of N observations. Neutral: 0.99 / 0.95 / 1.00 / 0.93.
- [x] `counter`: bounds each later opponent's strongest possible combo from the unseen cards,
      then scores every legal combo by the pile it expects to finish on (a distribution over how
      many later opponents beat it) minus card and structure costs. The first version, which
      only compared the top pile with the bottom one, scored 0.46 against Economists; valuing the
      pile at each finishing rank raised it to 0.78, and a grid search to
      **1.14 / 1.07 / 1.10 / 0.96**, the strongest hand-written agent (0.31-0.49 against
      `dmc_league3`).

Meta-agents:
- [x] Ensembles (`dmc:path=a.bin+b.bin+...`): the three league rounds averaged score 1.05 (2p)
      and 1.02 (4p) head-to-head against round 3 alone (400 deals).
- [x] `hybrid`: DMC that switches to exact max-n minimax over k=8 sampled deals once at most 8
      cards are left in hands and reserves. 1.03 (2p) and 1.01 (4p) against `dmc_league3`;
      switching at 12 cards, 0.98. The network already plays endgames well.

Learned:
- [x] `distill` (`python/train_distill.py`): a policy network trained with cross-entropy on
      guided ISMCTS's root visit counts. On the 43k decisions of the first expert-iteration
      round it reaches only 36% top-move agreement with the search and 0.42 against three
      Economists; sharpening the targets (temperature 0.3 or 0.1) doesn't help (0.39, 0.40).
      It is data-limited: with search data from the round-3 network (n=200, league3 prior and
      leaf, belief), 91k decisions give 43% agreement and 0.88 against Economists (0.45 against
      three `dmc_league3`); all 137k decisions give 0.97 (0.52).
- [x] `dmc_hist`: DMC with public-history features (`Observation::encode_history`: the last 12
      plays with lead and beat flags, and each seat's rate of beating the trick when following;
      `train_dmc.py --history`). Existing networks are unaffected: `DmcAgent` tells a history
      network by its input width.
      60 minutes of self-play from scratch, as v2: 1.12M games (v2: 2.43M; rebuilding the history
      features each decision halves throughput), 0.95 against three Economists (v2 at a similar
      game count: 1.32). Head-to-head against v2: 0.80 (2p), 0.64 (4p); table against Economists
      1.56 / 1.04 / 0.95 / 0.89. Continued for 30 more minutes (about 2M games in all; a container
      restart ended the run at its last checkpoint): 1.12 against Economists, but still 0.90
      (2p) and 0.61 (4p) head-to-head against v2. **Negative result:** the history features
      don't help at this budget, and they halve self-play throughput.

### League round 4 (`models/dmc_league4.bin`)

Continued from round 3 for 60 minutes against economist, counter, dmc_league2 and dmc_league3
(1.97 against three Economists, 1.83 against three Counters). Against round 3: 1.07 (2p) and
1.06 (4p) head-to-head. The browser presets now use it.

### League round 5 (`models/dmc_league5.bin`)

Continued from round 4 for 60 minutes against economist, the sampled counter, dmc_league3 and
dmc_league4 (2.00 against three Economists, 1.69 against three sampled Counters). Against
round 4: 0.99 (2p) and 1.06 (4p). The gains per round are shrinking; the browser presets use
it.

### League round 6, widened (`models/dmc_league6.bin`)

Round 5 widened from 256-256-128 to 512-512-256 with Net2WiderNet (`train_dmc.py --widen`,
function-preserving, so training starts at round 5's strength), then 90 minutes of league
training against economist, counter, dmc_league4 and dmc_league5. Against round 5: **1.07
(2p)** and 1.02 (4p), where unwidened round 5 had stalled at 0.99 / 1.06 against round 4. The
extra capacity restarts progress; the browser presets use it. Under guided search (n=200, belief) round 6 against
round 5: level in 4p (1.00, 100 deals) and 1.08 in 2p (150 deals).

### League round 7 (`models/dmc_league7.bin`)

Continued from widened round 6 for 90 minutes against economist, counter, dmc_league5 and
dmc_league6 (1.97 against three Economists). Against round 6: 1.06 (2p) and **1.08 (4p)**, the
largest gain since round 2. The browser presets use it.

### League round 8 (`models/dmc_league8.bin`)

Continued from round 7 for 90 minutes against economist, counter, dmc_league6 and dmc_league7.
Against round 7: 1.06 (2p) and 1.08 (4p). Since widening, each round has gained about as
much as round 2 did. The browser presets use it.

## Milestone 14: Better playouts and search inputs

- [x] `rollout=counter`: playouts with the `counter` strategy (plus `eps` random moves).
      Against three Economists (4p, 200 deals):

      | Agent | Rollout | Relative |
      |---|---|---|
      | ISMCTS (1000 iterations) | cheapest, eps 0.3 (old default) | 1.41 |
      | | random | 1.20 |
      | | counter, eps 0.1 / 0.15 / **0.2** / 0.3 / 0.45 / 0.6 | 1.79 / 1.88 / **1.97** / 1.71 / 1.64 / 1.56 |
      | ISMCTS, 20 ms per move | cheapest / counter eps 0.2 | 1.41 / 1.59 |
      | PIMC (64 deals) | random (old default) | 0.94 |
      | | counter, eps 0 / **0.1** / 0.2 | 1.78 / **1.80** / 1.73 |
      | ISMCTS n=200 with the league3 prior | counter, eps 0.3 | 1.86 |

      Head-to-head (4p, two seats each), counter-rollout ISMCTS beats cheapest-rollout ISMCTS
      1.29 to 0.71. Counter rollouts are now the default for both ISMCTS (eps 0.2) and PIMC
      (eps 0.1). A much better playout policy matters more than the extra time it costs, as
      in Go before value networks.
- [x] `prior=` accepts a policy network (one logit per action, e.g. `models/distill_v5.bin`)
      as well as a DMC network; the kind is told by the output width. With `distill_v3` as the prior
      (DMC leaf, belief, n=200) the search loses to the same search with the DMC prior: 0.93 to
      1.07 (4p, two seats each) and 0.89 to 1.11 (2p). The distilled policy (48% top-move
      agreement) is still too weak to steer the search. With `distill_v5` (56% agreement, 0.92 against
      `dmc_league3`) the prior is level in 4p (1.05 and then 0.97 on fresh deals, 100 deals
      each) and still loses in 2p (0.93), so the DMC prior stays.
- [x] Mixed leaf evaluation (`leaf=...,mix=lambda`): `(1 - lambda) * DMC value + lambda *
      rollout`, as AlphaGo mixed its value network with rollouts. **Negative:** with lambda 0.25 and
      counter rollouts (eps 0.2), guided ISMCTS (league3 prior and leaf, belief, n=200) is
      level with the pure-network leaf in 4p (0.99 vs 1.01, two seats each) and loses in 2p
      (0.80 vs 1.20), at about twice the cost. The league network's values are already better
      than counter playouts.
- [x] Distillation keeps scaling with search data (round-3 guided search, n=200):

      | Decisions | Top-move agreement | vs 3 Economists | vs 3 `dmc_league3` |
      |---|---|---|---|
      | 43k | 36% | 0.42 | 0.22 |
      | 91k | 43% | 0.88 | 0.45 |
      | 137k | 46% | 0.97 | 0.52 |
      | 189k | 48% | 1.06 | 0.57 |
      | 287k | 51% | 1.21 | 0.68 |
      | 384k | 52% | 1.18 | 0.68 |
      | 384k, 512-512-256, 25 epochs (`models/distill_v5.bin`) | 56% | **1.45** | **0.92** |
      | 390k, 1024-1024-512, 25 epochs | 58.5% | 1.50 | 0.88 |

      The curve flattens at about 300k decisions for the 256-256-256 network; a bigger network
      breaks through, so capacity, not data, was the limit. Going larger again raises agreement
      but not strength, so `distill_v5` keeps the 512-512-256 size.

- [x] Blending the counter prior into the network prior (`pmix=w`, geometric, applied in games
      of 3+ players): w = 0.5 against no blend, guided ISMCTS with round 5: 1.05 (4p, 160
      deals), 0.95 (3p), 0.99 (6p), and 0.93 in 2p before it was limited to 3+ players; w =
      0.75: 0.94 (4p). Neutral overall, so it stays off by default.
- [x] Ensemble leaves (`leaf=a.bin+b.bin`, averaging the networks' values): guided ISMCTS
      with a round 3 + round 4 leaf is level with the round 4 leaf alone (1.00 vs 1.00, 4p, two
      seats each, 80 deals) at about twice the leaf cost. Not used by default.
- [x] `pimc:leaf=...`: PIMC valuing each move with a DMC network instead of a rollout. Weak
      (about 0.6 against three Economists, 10 deals) and slow: without a tree, opponents
      between the move and the network's turn are modelled by the deterministic `cheapest`
      advance. Using `counter` for that advance instead doesn't help guided ISMCTS either
      (1.35 vs 1.33 against three `dmc_league4`, 100 deals), so the advance stays `cheapest`.
- [x] Guided ISMCTS settings (league3 prior and leaf, belief, n=200), each variant
      head-to-head against cpuct 1 / ptemp 0.1 (4p, two seats each, 80 deals): cpuct 0.5:
      0.95, cpuct 2: 0.90, ptemp 0.05: 1.02, ptemp 0.2: 0.97, cpuct 2 + ptemp 0.2: 0.94. The
      current settings are at or near the best.
- [x] `prior=counter`: PUCT priors from a softmax over `counter`'s move scores (in points), no
      network needed. Plain ISMCTS (1000 iterations, counter rollouts) against three
      Economists: 1.97 without a prior; 2.17 / 2.08 / 2.04 with `ptemp` 1 / 2 / 4.
      Head-to-head (4p, two seats each, ptemp 2) it beats the same search without a prior 1.25
      to 0.75. Now the ISMCTS default when no network prior is given (`prior=none` turns it
      off).
- [x] `counter:samples=k`: estimate whether each later opponent can beat a combo from k sampled
      deals instead of the worst-case bound. Needs a higher card cost to compensate (q 0, card
      1.5: 1.10, level with the bound version). Tuned (q 0, card 2, structure 0.5, 24 deals):
      **1.20** against three Economists where the bound version scores 1.12 (600 deals), and
      1.12 to 0.88 head-to-head (4p). Now the `counter` agent's default; rollouts and priors
      keep the bound version for speed.
- [x] Network-free ISMCTS (counter prior and counter rollouts, 1000 iterations) against
      `dmc_league3`: 0.96 (4p, against three) and 0.95 (2p). Before the counter rollouts and
      prior, plain ISMCTS sat far below the league networks (1.06 against `dmc_league`'s 1.66
      on the earlier ladder).
- [x] `counter:adaptive=w`: each opponent's hold chance learned from their beat rate when
      following, shrunk to the prior with weight w. Neutral: against three Economists 1.10
      (fixed) vs 1.11-1.13 (w = 2-50, within noise), but worse against three `cheapest`
      (1.27 vs 1.05-1.21). Not the default.

- [x] `models/belief_v2.bin`: the hand-prediction network retrained on 12,000 games by today's
      agents (league rounds 5-6, counter, economist) with a bigger network (512-512). In guided
      ISMCTS (round 6 prior and leaf, n=200) it beats `belief_v1` 1.06 (4p, 100 deals) and 1.03
      (2p, 150 deals). The browser's Guided ISMCTS uses it.

- [x] `dmc:temp=T`: sample moves from a softmax over the network's predicted returns, for
      opponents of any strength. Round 8 against three Economists: greedy 1.92; T = 0.02 / 0.05 /
      0.1 / 0.2: 1.79 / 1.07 / 0.55 / 0.23. The browser offers T = 0.1 ("DMC, playful",
      Beginner) and T = 0.04 ("DMC, relaxed", Casual) to fill the gaps in its difficulty
      ladder.

## Tooling added along the way

- [x] `VecEnv`: batched Python environment, 11x faster stepping than per-game calls; all
      trainers use it.
- [x] `rainbow ladder`: ranks a pool of agents over many random lineups.
- [x] `ismcts:ms=`: per-move time budget, for compute-matched comparisons.
- [x] `Game` pickling and `GameState::to_bytes` / `from_bytes`.
- [x] Auxiliary hidden-hand loss for DMC (`train_dmc.py --aux-weight`).
- [x] `scripts/evaluate.sh` (standard table) and `scripts/expert_iteration.sh` (automated rounds).

## Later

- **Larger DMC networks** and longer runs. A first try (v3: 512-512-256 with the auxiliary
  hidden-hand loss, 90 minutes but only 546k games under contention) reached 0.94 against three
  Economists, below v2's 1.11 at a similar game count, at about four times the cost per game.
  Not worth it at this budget; it needs a run of millions of games to judge.
  The auxiliary loss alone (v2's size, `--aux-weight 1`) reached 1.05 at 552k games against v2's
  1.11 at 588k: no measurable benefit at this budget.
- **Hand-prediction as an auxiliary loss** for the DMC and PPO networks.

## Results

### Current ladder (4p, 40 random lineups of 4 from 10 agents, 40 duplicate deals each)

| Agent | Relative | +/- |
|---|---|---|
| ismcts:n=200,prior=dmc_league5.bin,leaf=dmc_league5.bin,belief=belief_v1.bin | **1.72** | 0.06 |
| dmc:path=dmc_league3.bin+dmc_league4.bin+dmc_league5.bin | 1.40 | 0.05 |
| hybrid:path=dmc_league5.bin | 1.36 | 0.08 |
| ismcts | 1.29 | 0.07 |
| dmc:path=dmc_league5.bin | 1.19 | 0.06 |
| pimc | 1.04 | 0.08 |
| distill v4 (since replaced by distill_v5) | 0.74 | 0.07 |
| counter | 0.64 | 0.04 |
| economist | 0.42 | 0.04 |
| cheapest | 0.26 | 0.02 |

A stronger field than the previous ladder, so scores are compressed. Network-free ISMCTS
(counter rollouts and prior) now ranks above the raw round-5 network, and counter clearly
above the Economist.

### Previous ladder (4p, 60 random lineups of 4 from 14 agents, 40 duplicate deals each)

| Agent | Relative | +/- |
|---|---|---|
| ismcts:n=200,prior=dmc_league3.bin,leaf=dmc_league3.bin,belief=belief_v1.bin | **2.12** | 0.12 |
| hybrid:path=dmc_league3.bin | 1.75 | 0.10 |
| dmc:path=dmc_league.bin+dmc_league2.bin+dmc_league3.bin | 1.66 | 0.10 |
| dmc:path=dmc_league3.bin | 1.62 | 0.10 |
| dmc:path=dmc_v2.bin | 1.28 | 0.11 |
| ismcts | 1.04 | 0.09 |
| dqn:path=dqn_v1.bin | 0.97 | 0.11 |
| economist | 0.73 | 0.07 |
| counter | 0.64 | 0.04 |
| pimc | 0.57 | 0.04 |
| margin | 0.54 | 0.03 |
| nfsp:path=nfsp_v1.bin | 0.51 | 0.05 |
| ppo:path=ppo_v2.bin | 0.47 | 0.06 |
| cheapest | 0.42 | 0.06 |

The hybrid (DMC plus endgame minimax) and the league ensemble edge out round 3 alone, matching
their small head-to-head gains. `counter` beats Economists head-to-head but is exploited more by
the learned agents that fill most lineups, and `margin` is weaker than the Economist at 4p.

### Ladder before the second round of agents (4p, 40 random lineups of 4 from 10 agents, 40 duplicate deals each)

| Agent | Relative | +/- |
|---|---|---|
| ismcts n=200, dmc_league prior and leaf, belief | **2.32** | 0.08 |
| dmc_league | 1.66 | 0.13 |
| dmc_v2 | 1.20 | 0.08 |
| ismcts (1000 iterations) | 1.06 | 0.09 |
| dqn_v1 | 0.85 | 0.08 |
| pimc | 0.68 | 0.08 |
| economist | 0.62 | 0.04 |
| ppo_v2 | 0.59 | 0.06 |
| nfsp_v1 | 0.57 | 0.04 |
| cheapest | 0.32 | 0.04 |

### League round 2 (`models/dmc_league2.bin`)

Continued from `dmc_league` against a pool of economist, cheapest, dmc_v2 and dmc_league (48 of
60 planned minutes; a container restart cut the run short, and its last checkpoint is shipped).
1.95 against three Economists. Against round 1: 0.98 / 1.14 / 1.14 / 0.98 at 2 / 3 / 4 / 6 players
(4p head-to-head 1.10); against 1000-iteration ISMCTS: 1.01 / 1.68 / 1.62 / 1.14. A second league
round gives a modest further gain at 3-4 players. Under search the gain disappears: guided ISMCTS (n=200, belief) with the
round-2 network against the same search with round 1 scores 1.74 to 1.71 (4p, with two
Economists, 120 deals) and 1.02 to 0.98 (2p head-to-head).

### League round 3 (`models/dmc_league3.bin`)

Continued from round 2 for 60 minutes against economist, cheapest, dmc_league and dmc_league2
(1.92 against three Economists). Against round 2: 0.99 / 1.09 / 1.11 / 1.09 at 2 / 3 / 4 / 6 players
(head-to-head 1.06 in 2p, 1.08 in 4p); against 1000-iteration ISMCTS: 1.03 / 1.51 / 1.55 / 1.22.
Each league round adds a small, consistent gain over the previous network.

### Earlier ladder

(4p, 40 random lineups of 4 from 9 agents, 40 duplicate deals each.)

| Agent | Relative | +/- |
|---|---|---|
| ismcts n=200, dmc_v2 prior and leaf | 2.30 | 0.07 |
| dmc_v2 | 1.49 | 0.09 |
| ismcts (1000 iterations) | 1.21 | 0.08 |
| dqn_v1 | 0.96 | 0.08 |
| pimc | 0.83 | 0.09 |
| economist | 0.81 | 0.08 |
| nfsp_v1 | 0.72 | 0.08 |
| ppo_v1 | 0.35 | 0.05 |
| cheapest | 0.32 | 0.03 |

Run before the league network, PPO v2 and belief sampling existed; their results are in the
sections below.

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

### Search guided by the league network (n=200, DMC prior and leaf, belief; 120 deals)

| Lineup | Guided search | Others |
|---|---|---|
| 4p: league-guided vs v2-guided + 2 Economists | 1.87 | v2-guided 1.54, Economists 0.30 |
| 4p: vs 3 raw `dmc_league` networks | 1.60 | 0.80 |
| 2p: vs raw `dmc_league` | 1.65 (82.5% of wins) | 0.35 |

A better network makes a better search, and 200 iterations of search still add about 0.6 on top
of the best network. This is the strongest agent:
`ismcts:n=200,prior=models/dmc_league.bin,leaf=models/dmc_league.bin,belief=models/belief_v1.bin`.

### Search budget (guided ismcts with dmc_league prior/leaf and belief, 4p vs 3 raw dmc_league)

| Iterations | Relative | Opponents | Time |
|---|---|---|---|
| 50 | 1.47 | 0.84 | 47 s for 100 deals |
| 200 | 1.60 | 0.80 | - |
| 800 | 1.60 | 0.80 | 6 min 41 s for 60 deals |

Most of search's gain comes in the first 50 iterations and it saturates by about 200; beyond that
the network and the deal sampling, not the search budget, limit strength.

Adding opponent inference (dmc_league as the opponent model, 100 candidate deals) to this agent
hurts: 1.18 against 1.50 without it at the same table (with two raw dmc_league networks, 80
deals). The search then draws all 200 iterations from only 100 candidates and the sharp softmax
(temperature 0.1) concentrates weight on a few of them; belief sampling already supplies most of
what inference adds. A softer setting (400 candidates, `ipower=0.3`) still lost (1.08 against
1.51). The cause is double counting: belief-proposed deals already condition on the history, and
inference then weights them by the same history's likelihood. Without belief, inference does help
guided search: 1.38 against 1.18 (with two raw dmc_league networks, 60 deals). Inference now applies
that correction (likelihood x the uniform deal's importance ratio over the belief proposal), but
the combination still loses: 1.10 against 1.36 for belief alone (60 deals). Belief proposals are
strongly shaped, so the importance ratios vary widely and a few deals take most of the weight,
narrowing what the search explores. Use belief or inference, not both; belief alone is best.

### Belief-weighted deals (`models/belief_v1.bin`, 4p unless noted, 200 deals)

| Lineup | With belief | Without |
|---|---|---|
| ismcts vs ismcts + 2 Economists | 1.33 | 1.08 |
| 3p: ismcts vs 2 plain ismcts | 1.04 | 0.98 |
| ismcts + `infer=cheapest` vs `infer=cheapest` + 2 Economists | 1.40 | 1.10 |
| guided ismcts (n=200, DMC prior and leaf) vs the same without belief + 2 Economists (150 deals) | 1.78 | 1.53 |

The hand-prediction network (trained on games by DMC, Economist, `cheapest` and random agents)
helps as much as the `cheapest` inference model against Economists, without needing a
hand-picked opponent model, and the two stack. It also adds 0.25 on top of network-guided
search, making `ismcts:n=200,prior=<dmc>,leaf=<dmc>,belief=<belief>` the strongest agent. Against ISMCTS opponents, whose play it never saw,
the gain is within noise.

### Minimax with `cheapest` leaves (4p vs 3 Economists, 40-50 deals)

| Variant | Relative |
|---|---|
| paranoid, depth 3, k=16 | 0.14 |
| paranoid, depth 2, k=8 | 0.17 |
| max-n, depth 2, k=8 | 0.30 |
| max-n, depth 1, k=8 | 0.20 |

| BRS, depth 3, k=8 (50 deals) | 0.10 |
| paranoid, depth 3, k=8 (same deals) | 0.13 |
| **BRS, depth 2, k=4, DMC v2 leaf values** (20 deals, about +/-0.2) | **1.25** |

Badly beaten. Even depth 1 (sampled deals plus a deterministic `cheapest` playout) loses, so the
leaf evaluation is the problem, not the search rule: it is the same biased opponent model that
sank PIMC with `cheapest` rollouts, compounded by determinized search assuming everyone sees all
cards. Paranoid is worse still because with three colluding opponents almost every move looks
lost.

With DMC leaf values instead of `cheapest` playouts, BRS jumps from 0.10 to 1.25 against
three Economists, confirming that leaf evaluation, not the search rule, was the problem. It is
slow, though (network values for every seat at every leaf), so guided ISMCTS remains the better
use of search time.

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
| 45 | 0.96 | 0.17 | 0.63 |

PPO v2 (GAE with lambda 0.95, two learner seats per game against economist and frozen dmc_v2)
reached 0.51 at 10 minutes, 0.62 at 40 and 0.89 at 60, against v1's 0.17.

DMC reached 1.17 after 10 minutes on an idle machine, so DMC's action-feature Q-network learns
far faster than DQN's one-output-per-action head here, matching DouZero's finding. PPO v1 learns
(it beats random play easily) but only made 343 updates in 45 minutes and stays weak; PPO v2 adds
GAE and league opponents.

### Learned agents: evaluation (`scripts/evaluate.sh`, 200 deals, agent in one seat)

| Agent | Opponent | 2p | 3p | 4p | 6p |
|---|---|---|---|---|---|
| dqn_v1 | economist | 1.79 | 1.17 | 0.96 | 1.04 |
| dqn_v1 | dmc_v2 | 0.86 | 0.57 | 0.51 | 0.76 |
| ppo_v1 | economist | 1.53 | 0.29 | 0.17 | 0.63 |
| ppo_v1 | dmc_v2 | 0.56 | 0.18 | 0.23 | 0.49 |
| ppo_v1 (sampling) | economist | 1.42 | 0.22 | 0.24 | 0.59 |
| nfsp_v1 (sampling) | economist | 1.16 | 0.79 | 0.63 | 0.73 |
| nfsp_v1 (sampling) | dmc_v2 | 0.50 | 0.33 | 0.37 | 0.57 |
| ppo_v2 (GAE, league) | economist | 1.68 | 0.81 | 0.89 | 0.97 |
| ppo_v2 (GAE, league) | dmc_v2 | 0.87 | 0.31 | 0.38 | 0.70 |
| dmc_league | economist | 1.81 | 2.09 | 1.74 | 1.78 |
| dmc_league | dmc_v2 | 1.23 | 1.14 | 1.22 | 0.92 |
| dmc_league | ismcts (1000 iterations) | 0.96 | 1.59 | 1.46 | 1.16 |
| dmc_v2 | ismcts (1000 iterations) | 0.68 | 1.41 | 1.54 | 1.05 |

DMC's (observation, action) scoring is the most effective learner per unit of compute; DQN beats
the Economist at 2-3 players but loses to DMC everywhere. Fine-tuning DMC v2 for 45 minutes
against a league (two learner seats, the rest drawn from economist, cheapest and frozen v2) beats
frozen v2 at 2-4 players, so the gain isn't only exploitation of the Economist it trained against.

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
