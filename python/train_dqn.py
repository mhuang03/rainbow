"""Masked Double DQN self-play training for Rainbow.

The network maps an observation to one Q-value per action; illegal actions are masked out when
acting and when bootstrapping. A transition runs from one of a player's decisions to that
player's next decision, with everything the other players do in between treated as part of the
environment. Rewards are dense: the points a player gains between its decisions (scaled), plus
its win share when the game ends. Targets use Double DQN (the online network picks the next
action, a slowly updated target network values it) with a Huber loss.

    python train_dqn.py --minutes 30 --out models/dqn.bin
    ../target/release/rainbow arena dqn:path=python/models/dqn.bin economist economist economist
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

POINTS_SCALE = 100.0


class Replay:
    """Ring buffer of transitions (obs, action, reward, next_obs, next_mask, done)."""

    def __init__(self, capacity: int):
        self.obs = np.zeros((capacity, re.OBS_SIZE), dtype=np.float32)
        self.next_obs = np.zeros((capacity, re.OBS_SIZE), dtype=np.float32)
        self.next_mask = np.zeros((capacity, re.NUM_ACTIONS), dtype=bool)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.next = 0

    def add(self, obs, action, reward, next_obs, next_mask, done) -> None:
        i = self.next
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.done[i] = done
        if not done:
            self.next_obs[i] = next_obs
            self.next_mask[i] = next_mask
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, n)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.next_mask[idx]),
            torch.from_numpy(self.done[idx]),
        )


class Actors:
    """Concurrent self-play games, tracking each seat's unfinished transition."""

    def __init__(self, count: int, rng: np.random.Generator, win_weight: float, points_weight: float):
        self.rng = rng
        self.win_weight = win_weight
        self.points_weight = points_weight
        self.games = [new_game(rng) for _ in range(count)]
        # per game, per seat: (obs, action, points when acting) awaiting its next observation
        self.pending: list[dict[int, tuple[np.ndarray, int, int]]] = [{} for _ in range(count)]
        self.episodes = 0

    def step(self, net: nn.Module, epsilon: float, replay: Replay) -> int:
        obs, masks = re.batch_observations(self.games)
        with torch.no_grad():
            q = net(torch.from_numpy(obs)).numpy()
        q[~masks] = -np.inf

        for i, game in enumerate(self.games):
            seat = game.current_player
            scores = game.scores()
            # close this seat's previous transition now that it observes again
            if seat in self.pending[i]:
                p_obs, p_action, p_points = self.pending[i].pop(seat)
                reward = self.points_weight * (scores[seat] - p_points) / POINTS_SCALE
                replay.add(p_obs, p_action, reward, obs[i], masks[i], False)

            if self.rng.random() < epsilon:
                action = int(self.rng.choice(np.flatnonzero(masks[i])))
            else:
                action = int(np.argmax(q[i]))
            self.pending[i][seat] = (obs[i], action, scores[seat])
            game.step(action)

            if game.is_over:
                final = game.scores()
                returns = game.returns()
                for s, (p_obs, p_action, p_points) in self.pending[i].items():
                    reward = (
                        self.points_weight * (final[s] - p_points) / POINTS_SCALE
                        + self.win_weight * returns[s]
                    )
                    replay.add(p_obs, p_action, reward, None, None, True)
                self.pending[i] = {}
                self.games[i] = new_game(self.rng)
                self.episodes += 1
        return len(self.games)


def train_step(net, target, optimizer, batch, gamma: float) -> float:
    obs, action, reward, next_obs, next_mask, done = batch
    q = net(obs).gather(1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        online_next = net(next_obs).masked_fill(~next_mask, -1e9)
        best_next = online_next.argmax(1, keepdim=True)
        next_value = target(next_obs).gather(1, best_next).squeeze(1)
        y = reward + gamma * (1.0 - done) * next_value
    loss = nn.functional.smooth_l1_loss(q, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--out", type=Path, default=Path("models/dqn.bin"))
    parser.add_argument("--actors", type=int, default=256)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--replay", type=int, default=500_000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005, help="target network Polyak rate")
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--win-weight", type=float, default=1.0)
    parser.add_argument("--points-weight", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, nargs=2, default=[0.2, 0.02], help="start and end")
    parser.add_argument("--eval-every", type=float, default=5, help="minutes between arena checks")
    parser.add_argument("--eval-deals", type=int, default=200)
    parser.add_argument("--arena", type=Path, default=ARENA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0: torch default)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = mlp(re.OBS_SIZE, args.hidden, re.NUM_ACTIONS)
    target = copy.deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    actors = Actors(args.actors, rng, args.win_weight, args.points_weight)

    start = time.monotonic()
    deadline = start + args.minutes * 60
    next_eval = start + args.eval_every * 60
    decisions = updates = 0
    losses: list[float] = []
    while time.monotonic() < deadline:
        progress = (time.monotonic() - start) / (deadline - start)
        epsilon = args.epsilon[0] + (args.epsilon[1] - args.epsilon[0]) * min(1.0, 2 * progress)
        decisions += actors.step(net, epsilon, replay)

        if replay.size >= args.batch * 8:
            for _ in range(args.updates_per_step):
                losses.append(train_step(net, target, optimizer, replay.sample(args.batch, rng), args.gamma))
                updates += 1
                with torch.no_grad():
                    for p, tp in zip(net.parameters(), target.parameters()):
                        tp.mul_(1 - args.tau).add_(p, alpha=args.tau)

        if time.monotonic() >= next_eval:
            next_eval += args.eval_every * 60
            export(net, args.out)
            rel = arena_relative(f"dqn:path={args.out}", ["economist"] * 3, args.eval_deals, args.arena)
            loss = np.mean(losses[-200:]) if losses else float("nan")
            print(
                f"{(time.monotonic() - start) / 60:6.1f} min  episodes {actors.episodes:>8}  "
                f"decisions {decisions:>9}  updates {updates:>7}  loss {loss:.4f}  eps {epsilon:.3f}  "
                f"vs 3x economist {rel}",
                flush=True,
            )

    export(net, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
