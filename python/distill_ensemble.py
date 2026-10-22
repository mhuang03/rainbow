"""Distils an ensemble of DMC networks into one network of the same shape.

The ensemble (`dmc:path=a+b+c`) plays better than any of its members, at a member's cost
times the ensemble size. Here the ensemble plays self-play games (with a little exploration),
and every legal move's ensemble value becomes a regression target for the student, so the
student learns the ensemble's whole value function, not only its choices. The student starts
from one of the teachers.

    python distill_ensemble.py --teachers ../models/dmc_league7.bin ../models/dmc_league8.bin \\
        ../models/dmc_league9.bin --minutes 60 --out ../models/dmc_distilled.bin
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re
from rl_common import ARENA, arena_relative, export, load

ACTIONS = re.action_features()  # (NUM_ACTIONS, ACTION_FEATURES)


class Buffer:
    """Ring buffer of (input row, target value) pairs."""

    def __init__(self, capacity: int, width: int):
        self.x = np.zeros((capacity, width), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.next = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        idx = (self.next + np.arange(len(x))) % self.capacity
        self.x[idx] = x
        self.y[idx] = y
        self.next = int((self.next + len(x)) % self.capacity)
        self.size = min(self.size + len(x), self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.size, n)
        return self.x[idx], self.y[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teachers", type=Path, nargs="+", required=True)
    parser.add_argument("--init", type=Path, default=None, help="student's start (default: the last teacher)")
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--out", type=Path, default=Path("models/dmc_distilled.bin"))
    parser.add_argument("--actors", type=int, default=256, help="concurrent games")
    parser.add_argument("--epsilon", type=float, default=0.05, help="random-move rate while playing")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--buffer", type=int, default=1_000_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-every", type=float, default=15, help="minutes between arena checks")
    parser.add_argument("--eval-deals", type=int, default=200)
    parser.add_argument("--arena", type=Path, default=ARENA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0: torch default)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    teachers = [load(p).eval() for p in args.teachers]
    student = load(args.init or args.teachers[-1])
    width = student[0].in_features
    if any(t[0].in_features != width for t in teachers) or width != re.OBS_SIZE + re.ACTION_FEATURES:
        parser.error("teachers and student must all read observation + action features")
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    buffer = Buffer(args.buffer, width)
    env = re.VecEnv(args.actors, seed=int(rng.integers(1 << 62)))

    start = time.monotonic()
    deadline = start + args.minutes * 60
    next_eval = start + args.eval_every * 60
    decisions = updates = 0
    losses: list[float] = []
    while time.monotonic() < deadline:
        obs, masks = env.observe()
        legal = [np.flatnonzero(m) for m in masks]
        counts = np.array([len(l) for l in legal])
        rows = np.concatenate([np.repeat(obs, counts, axis=0), ACTIONS[np.concatenate(legal)]], axis=1)
        with torch.no_grad():
            x = torch.from_numpy(rows)
            q = torch.stack([t(x).squeeze(1) for t in teachers]).mean(0).numpy()
        buffer.add(rows, q)

        actions = np.empty(len(legal), dtype=np.int64)
        s = 0
        for i, n in enumerate(counts):
            choice = int(rng.integers(n)) if rng.random() < args.epsilon else int(np.argmax(q[s : s + n]))
            actions[i] = legal[i][choice]
            s += n
        env.step(actions)
        decisions += len(legal)

        for _ in range(args.updates_per_step):
            bx, by = buffer.sample(args.batch, rng)
            loss = nn.functional.mse_loss(student(torch.from_numpy(bx)).squeeze(1), torch.from_numpy(by))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            updates += 1

        if time.monotonic() >= next_eval:
            next_eval += args.eval_every * 60
            export(student, args.out)
            rel = arena_relative(f"dmc:path={args.out}", ["economist"] * 3, args.eval_deals, args.arena)
            print(
                f"{(time.monotonic() - start) / 60:6.1f} min  decisions {decisions:>9}  updates {updates:>7}  "
                f"loss {np.mean(losses[-200:]):.5f}  vs 3x economist {rel}",
                flush=True,
            )

    export(student, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
