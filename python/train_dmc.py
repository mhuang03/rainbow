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
from rl_common import ARENA, arena_relative, export, final_returns_row, mlp

ACTIONS = re.action_features()  # (NUM_ACTIONS, ACTION_FEATURES)
INPUT_SIZE = re.OBS_SIZE + re.ACTION_FEATURES


class DmcNet(nn.Module):
    """Q(observation, action) MLP whose first layer is split into an observation part and an
    action part, so an optional auxiliary head can predict hidden hands from the observation
    part alone. Exports as a plain MLP over the concatenated input."""

    def __init__(self, hidden: list[int]):
        super().__init__()
        self.obs_layer = nn.Linear(re.OBS_SIZE, hidden[0])
        self.act_layer = nn.Linear(re.ACTION_FEATURES, hidden[0], bias=False)
        self.rest = mlp(hidden[0], hidden[1:], 1)
        self.aux_head = nn.Linear(hidden[0], re.HIDDEN_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs, act = x[:, : re.OBS_SIZE], x[:, re.OBS_SIZE :]
        return self.rest(torch.relu(self.obs_layer(obs) + self.act_layer(act)))

    def aux(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted hidden cards (count / 10 per rank, relative seats) from the observation."""
        return self.aux_head(torch.relu(self.obs_layer(x[:, : re.OBS_SIZE])))

    def to_sequential(self) -> nn.Sequential:
        first = nn.Linear(INPUT_SIZE, self.obs_layer.out_features)
        with torch.no_grad():
            first.weight.copy_(torch.cat([self.obs_layer.weight, self.act_layer.weight], dim=1))
            first.bias.copy_(self.obs_layer.bias)
        return nn.Sequential(first, nn.ReLU(), *self.rest)


class Buffer:
    """Ring buffer of (input row, target return, hidden cards) triples."""

    def __init__(self, capacity: int):
        self.x = np.zeros((capacity, INPUT_SIZE), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.h = np.zeros((capacity, re.HIDDEN_SIZE), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.next = 0

    def add(self, x: np.ndarray, y: np.ndarray, h: np.ndarray) -> None:
        for i in range(len(x)):
            self.x[self.next] = x[i]
            self.y[self.next] = y[i]
            self.h[self.next] = h[i]
            self.next = (self.next + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.size, n)
        return self.x[idx], self.y[idx], self.h[idx]


class Actors:
    """A batch of concurrent self-play games."""

    def __init__(self, count: int, rng: np.random.Generator, opponents: list[str] | None = None, learners: int = 1):
        self.rng = rng
        self.env = re.VecEnv(count, seed=int(rng.integers(1 << 62)), opponents=opponents, learners=learners)
        # per game: (seat, input row, hidden cards) for every decision so far
        self.history: list[list[tuple[int, np.ndarray, np.ndarray]]] = [[] for _ in range(count)]
        self.episodes = 0

    def step(self, net: nn.Module, epsilon: float, buffer: Buffer) -> int:
        """Advances every game one move; returns the number of decisions made."""
        obs, masks, full = self.env.observe(full=True)
        hidden = full[:, re.OBS_SIZE :]
        seats = self.env.current_players()
        players = self.env.num_players()
        legal = [np.flatnonzero(m) for m in masks]
        counts = np.array([len(l) for l in legal])
        rows = np.concatenate([np.repeat(obs, counts, axis=0), ACTIONS[np.concatenate(legal)]], axis=1)
        with torch.no_grad():
            q = net(torch.from_numpy(rows)).squeeze(1).numpy()

        actions = np.empty(len(legal), dtype=np.int64)
        start = 0
        for i, n in enumerate(counts):
            if self.rng.random() < epsilon:
                choice = int(self.rng.integers(n))
            else:
                choice = int(np.argmax(q[start : start + n]))
            self.history[i].append((int(seats[i]), rows[start + choice], hidden[i]))
            actions[i] = legal[i][choice]
            start += n

        done, returns, scores = self.env.step(actions)
        for i in np.flatnonzero(done):
            final = final_returns_row(returns[i], scores[i], int(players[i]))
            hist = self.history[i]
            buffer.add(
                np.stack([row for _, row, _ in hist]),
                final[[seat for seat, _, _ in hist]],
                np.stack([h for _, _, h in hist]),
            )
            self.history[i] = []
            self.episodes += 1
        return len(legal)


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
    parser.add_argument("--aux-weight", type=float, default=0.0,
                        help="weight of the auxiliary hidden-hand prediction loss (0 disables)")
    parser.add_argument("--opponents", nargs="+", default=None,
                        help="Rust agent specs for non-learner seats (default: pure self-play)")
    parser.add_argument("--learners", type=int, default=1, help="learner seats per game with --opponents")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0: torch default)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = DmcNet(args.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = Buffer(args.buffer)
    actors = Actors(args.actors, rng, args.opponents, args.learners)

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
                x, y, h = (torch.from_numpy(a) for a in buffer.sample(args.batch, rng))
                loss = nn.functional.mse_loss(net(x).squeeze(1), y)
                if args.aux_weight > 0:
                    loss = loss + args.aux_weight * nn.functional.mse_loss(net.aux(x), h)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                updates += 1

        if time.monotonic() >= next_eval:
            next_eval += args.eval_every * 60
            export(net.to_sequential(), args.out)
            elapsed = (time.monotonic() - start) / 60
            rel = arena_relative(f"dmc:path={args.out}", ["economist"] * 3, args.eval_deals, args.arena)
            loss = np.mean(losses[-200:]) if losses else float("nan")
            print(
                f"{elapsed:6.1f} min  episodes {actors.episodes:>8}  decisions {decisions:>9}  "
                f"updates {updates:>7}  loss {loss:.4f}  eps {epsilon:.3f}  vs 3x economist {rel}",
                flush=True,
            )

    export(net.to_sequential(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
