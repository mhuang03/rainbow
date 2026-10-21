"""Neural Fictitious Self-Play (Heinrich and Silver, 2016).

Each player mixes two networks. A best-response network is trained by Double DQN (shared with
`train_dqn.py`) against the current population. An average-policy network is trained by
supervised learning to imitate the best response's past actions, kept in a reservoir buffer so
it represents the average over training. Each seat plays the best response with probability eta
(per game) and the average policy otherwise. The average policy approximates an equilibrium
strategy, which makes it hard to exploit; it is the exported agent, played by sampling.

    python train_nfsp.py --minutes 30 --out models/nfsp.bin
    ../target/release/rainbow arena nfsp:path=python/models/nfsp.bin economist economist economist
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re
from rl_common import ARENA, arena_relative, export, mlp, new_game
from train_dqn import POINTS_SCALE, Replay, train_step


class Reservoir:
    """Uniform sample of every (obs, mask, action) ever added (reservoir sampling)."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.obs = np.zeros((capacity, re.OBS_SIZE), dtype=np.float32)
        self.mask = np.zeros((capacity, re.NUM_ACTIONS), dtype=bool)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.capacity = capacity
        self.size = 0
        self.seen = 0
        self.rng = rng

    def add(self, obs, mask, action) -> None:
        self.seen += 1
        if self.size < self.capacity:
            i = self.size
            self.size += 1
        else:
            i = int(self.rng.integers(self.seen))
            if i >= self.capacity:
                return
        self.obs[i], self.mask[i], self.action[i] = obs, mask, action

    def sample(self, n: int):
        idx = self.rng.integers(0, self.size, n)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.mask[idx]), torch.from_numpy(self.action[idx]))


class Actors:
    def __init__(self, count: int, rng: np.random.Generator, eta: float, points_weight: float):
        self.rng = rng
        self.eta = eta
        self.points_weight = points_weight
        self.games = [new_game(rng) for _ in range(count)]
        self.modes = [self.new_modes(g) for g in self.games]
        self.pending: list[dict[int, tuple]] = [{} for _ in range(count)]
        self.episodes = 0

    def new_modes(self, game: re.Game) -> list[bool]:
        """Per seat: True to play the best response this game."""
        return [bool(self.rng.random() < self.eta) for _ in range(game.num_players)]

    def step(self, best: nn.Module, average: nn.Module, epsilon: float, replay: Replay, reservoir: Reservoir) -> int:
        obs, masks = re.batch_observations(self.games)
        x = torch.from_numpy(obs)
        with torch.no_grad():
            q = best(x).numpy()
            logits = average(x).masked_fill(~torch.from_numpy(masks), -1e9)
            sampled = torch.distributions.Categorical(logits=logits).sample().numpy()
        q[~masks] = -np.inf

        for i, game in enumerate(self.games):
            seat = game.current_player
            scores = game.scores()
            if seat in self.pending[i]:
                p_obs, p_action, p_points = self.pending[i].pop(seat)
                reward = self.points_weight * (scores[seat] - p_points) / POINTS_SCALE
                replay.add(p_obs, p_action, reward, obs[i], masks[i], False)

            if self.modes[i][seat]:
                if self.rng.random() < epsilon:
                    action = int(self.rng.choice(np.flatnonzero(masks[i])))
                else:
                    action = int(np.argmax(q[i]))
                reservoir.add(obs[i], masks[i], action)
            else:
                action = int(sampled[i])
            self.pending[i][seat] = (obs[i], action, scores[seat])
            game.step(action)

            if game.is_over:
                final, returns = game.scores(), game.returns()
                for s, (p_obs, p_action, p_points) in self.pending[i].items():
                    reward = self.points_weight * (final[s] - p_points) / POINTS_SCALE + returns[s]
                    replay.add(p_obs, p_action, reward, None, None, True)
                self.pending[i] = {}
                self.games[i] = new_game(self.rng)
                self.modes[i] = self.new_modes(self.games[i])
                self.episodes += 1
        return len(self.games)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--out", type=Path, default=Path("models/nfsp.bin"))
    parser.add_argument("--actors", type=int, default=256)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--eta", type=float, default=0.1, help="probability a seat plays the best response")
    parser.add_argument("--rl-lr", type=float, default=1e-4)
    parser.add_argument("--sl-lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--replay", type=int, default=300_000)
    parser.add_argument("--reservoir", type=int, default=1_000_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--points-weight", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, nargs=2, default=[0.1, 0.0])
    parser.add_argument("--eval-every", type=float, default=5)
    parser.add_argument("--eval-deals", type=int, default=200)
    parser.add_argument("--arena", type=Path, default=ARENA)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    best = mlp(re.OBS_SIZE, args.hidden, re.NUM_ACTIONS)
    target = copy.deepcopy(best)
    average = mlp(re.OBS_SIZE, args.hidden, re.NUM_ACTIONS)
    rl_opt = torch.optim.Adam(best.parameters(), lr=args.rl_lr)
    sl_opt = torch.optim.Adam(average.parameters(), lr=args.sl_lr)
    replay = Replay(args.replay)
    reservoir = Reservoir(args.reservoir, rng)
    actors = Actors(args.actors, rng, args.eta, args.points_weight)

    start = time.monotonic()
    deadline = start + args.minutes * 60
    next_eval = start + args.eval_every * 60
    decisions = 0
    rl_losses: list[float] = []
    sl_losses: list[float] = []
    while time.monotonic() < deadline:
        progress = (time.monotonic() - start) / (deadline - start)
        epsilon = args.epsilon[0] + (args.epsilon[1] - args.epsilon[0]) * progress
        decisions += actors.step(best, average, epsilon, replay, reservoir)

        if replay.size >= args.batch * 8:
            rl_losses.append(train_step(best, target, rl_opt, replay.sample(args.batch, rng), 1.0))
            with torch.no_grad():
                for p, tp in zip(best.parameters(), target.parameters()):
                    tp.mul_(1 - args.tau).add_(p, alpha=args.tau)
        if reservoir.size >= args.batch * 8:
            obs, mask, action = reservoir.sample(args.batch)
            logits = average(obs).masked_fill(~mask, -1e9)
            loss = nn.functional.cross_entropy(logits, action)
            sl_opt.zero_grad()
            loss.backward()
            sl_opt.step()
            sl_losses.append(loss.item())

        if time.monotonic() >= next_eval:
            next_eval += args.eval_every * 60
            export(average, args.out)
            rel = arena_relative(f"nfsp:path={args.out}", ["economist"] * 3, args.eval_deals, args.arena)
            print(
                f"{(time.monotonic() - start) / 60:6.1f} min  episodes {actors.episodes:>8}  "
                f"decisions {decisions:>9}  rl {np.mean(rl_losses[-200:]) if rl_losses else float('nan'):.4f}  "
                f"sl {np.mean(sl_losses[-200:]) if sl_losses else float('nan'):.4f}  "
                f"reservoir {reservoir.size}  vs 3x economist {rel}",
                flush=True,
            )

    export(average, args.out)
    export(best, args.out.with_name(args.out.stem + "_best.bin"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
