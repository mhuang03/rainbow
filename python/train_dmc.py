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
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re

ACTIONS = re.action_features()  # (NUM_ACTIONS, ACTION_FEATURES)
INPUT_SIZE = re.OBS_SIZE + re.ACTION_FEATURES
MARGIN_WEIGHT = 0.2  # matches the Rust search agents' default reward


def make_net(hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    size = INPUT_SIZE
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def export(net: nn.Sequential, path: Path) -> None:
    """Writes the network in the format read by `rainbow::agents::dmc::Mlp`."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"RBMLP1\0\0")
        f.write(struct.pack("<I", len(linears)))
        for layer in linears:
            w = layer.weight.detach().cpu().numpy().astype("<f4")
            b = layer.bias.detach().cpu().numpy().astype("<f4")
            f.write(struct.pack("<II", w.shape[1], w.shape[0]))
            f.write(w.tobytes())
            f.write(b.tobytes())


def episode_returns(game: re.Game) -> list[float]:
    """Win share plus a small score-margin term for every seat."""
    scores = game.scores()
    returns = game.returns()
    out = []
    for seat, ret in enumerate(returns):
        best_other = max(s for i, s in enumerate(scores) if i != seat)
        out.append(ret + MARGIN_WEIGHT * (scores[seat] - best_other) / 100)
    return out


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
        self.games = [self.new_game() for _ in range(count)]
        # per game: list of (seat, input row) for every decision so far
        self.history: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(count)]
        self.episodes = 0

    def new_game(self) -> re.Game:
        n = int(self.rng.integers(re.MIN_PLAYERS, re.MAX_PLAYERS + 1))
        return re.Game(n, seed=int(self.rng.integers(1 << 62)), leader=int(self.rng.integers(n)))

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
                returns = episode_returns(game)
                hist = self.history[i]
                buffer.add(
                    np.stack([row for _, row in hist]),
                    np.array([returns[seat] for seat, _ in hist], dtype=np.float32),
                )
                self.episodes += 1
                self.games[i] = self.new_game()
                self.history[i] = []
        return len(self.games)


def evaluate(model: Path, arena: Path, deals: int) -> str:
    """Runs the Rust arena: the model against three Economists in 4-player games."""
    result = subprocess.run(
        [str(arena), "arena", "-d", str(deals), f"dmc:path={model}", "economist", "economist", "economist"],
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(l for l in result.stdout.splitlines() if l.startswith("dmc"))
    return line.split()[4]  # relative win share


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
    parser.add_argument("--arena", type=Path, default=Path(__file__).parent.parent / "target/release/rainbow")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = make_net(args.hidden)
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
            rel = evaluate(args.out, args.arena, args.eval_deals) if args.arena.exists() else "n/a"
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
