"""Deep Monte Carlo (DouZero-style) self-play training for Rainbow.

A network Q(observation, action) is regressed onto the Monte Carlo return each player
eventually receives. Actors play many games at once, so each step scores every game's legal
moves in a single forward pass. All seats share the current network with epsilon-greedy
exploration, and player counts are sampled uniformly from 2 to 6.

Checkpoints are exported in the Rust MLP format, so they can enter the arena directly:

    python train_dmc.py --minutes 30 --out models/dmc.bin
    ../target/release/rainbow arena dmc:path=models/dmc.bin economist economist economist
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re
from rl_common import ARENA, arena_relative, export, final_returns, mlp, new_game

ACTIONS = re.action_features()  # (NUM_ACTIONS, ACTION_FEATURES)
INPUT_SIZE = re.OBS_SIZE + re.ACTION_FEATURES


class Buffer:
    """Ring buffer of (input row, target return) pairs."""

    def __init__(self, capacity: int):
        self.x = np.zeros((capacity, INPUT_SIZE), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.next = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        for i in range(len(x)):
            self.x[self.next] = x[i]
            self.y[self.next] = y[i]
            self.next = (self.next + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.size, n)
        return self.x[idx], self.y[idx]


class Actors:
    """A batch of concurrent self-play games."""

    def __init__(self, count: int, rng: np.random.Generator):
        self.rng = rng
        self.games = [new_game(rng) for _ in range(count)]
        # per game: list of (seat, input row) for every decision so far
        self.history: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(count)]
        self.episodes = 0

    def step(self, net: nn.Module, epsilon: float, buffer: Buffer) -> int:
        """Advances every game one move; returns the number of decisions made."""
        obs, masks = re.batch_observations(self.games)
        legal = [np.flatnonzero(m) for m in masks]
        counts = np.array([len(l) for l in legal])
        rows = np.concatenate(
            [np.repeat(obs, counts, axis=0), ACTIONS[np.concatenate(legal)]], axis=1
        )
        with torch.no_grad():
            q = net(torch.from_numpy(rows)).squeeze(1).numpy()

        start = 0
        for i, game in enumerate(self.games):
            n = counts[i]
            if self.rng.random() < epsilon:
                choice = int(self.rng.integers(n))
            else:
                choice = int(np.argmax(q[start : start + n]))
            self.history[i].append((game.current_player, rows[start + choice]))
            game.step(int(legal[i][choice]))
            start += n

            if game.is_over:
                returns = final_returns(game)
                hist = self.history[i]
                buffer.add(
                    np.stack([row for _, row in hist]),
                    np.array([returns[seat] for seat, _ in hist], dtype=np.float32),
                )
                self.episodes += 1
                self.games[i] = new_game(self.rng)
                self.history[i] = []
        return len(self.games)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--out", type=Path, default=Path("models/dmc.bin"))
    parser.add_argument("--actors", type=int, default=256, help="concurrent self-play games")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--buffer", type=int, default=200_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--epsilon", type=float, nargs=2, default=[0.1, 0.02], help="start and end")
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
    net = mlp(INPUT_SIZE, args.hidden, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = Buffer(args.buffer)
    actors = Actors(args.actors, rng)

    start = time.monotonic()
    deadline = start + args.minutes * 60
    next_eval = start + args.eval_every * 60
    decisions = updates = 0
    losses: list[float] = []
    while time.monotonic() < deadline:
        progress = (time.monotonic() - start) / (deadline - start)
        epsilon = args.epsilon[0] + (args.epsilon[1] - args.epsilon[0]) * progress
        decisions += actors.step(net, epsilon, buffer)

        if buffer.size >= args.batch * 4:
            for _ in range(args.updates_per_step):
                x, y = buffer.sample(args.batch, rng)
                pred = net(torch.from_numpy(x)).squeeze(1)
                loss = nn.functional.mse_loss(pred, torch.from_numpy(y))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                updates += 1

        if time.monotonic() >= next_eval:
            next_eval += args.eval_every * 60
            export(net, args.out)
            elapsed = (time.monotonic() - start) / 60
            rel = arena_relative(f"dmc:path={args.out}", ["economist"] * 3, args.eval_deals, args.arena)
            loss = np.mean(losses[-200:]) if losses else float("nan")
            print(
                f"{elapsed:6.1f} min  episodes {actors.episodes:>8}  decisions {decisions:>9}  "
                f"updates {updates:>7}  loss {loss:.4f}  eps {epsilon:.3f}  vs 3x economist {rel}",
                flush=True,
            )

    export(net, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
