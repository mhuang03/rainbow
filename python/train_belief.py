"""Hand-prediction (belief) network: predicts every hidden hand and reserve from an observation.

Training data comes for free: games are played by a mix of Rust agents, and at every step each
seat's observation is paired with the true hidden cards from `Game.full_observation`. The network
outputs card count / 10 per rank for each relative seat's hand then reserve, and is used by
`ismcts:belief=<path>` to bias sampled deals.

    python train_belief.py --games 4000 --out models/belief.bin
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re
from rl_common import export, mlp, new_game

DEFAULT_AGENTS = ["dmc:path=../models/dmc_v2.bin"] * 3 + ["economist", "cheapest", "random"]


def generate(games: int, specs: list[str], rng: np.random.Generator,
             history: bool = False, keep: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Observation and hidden-card rows from `games` games. `keep` < 1 keeps that random
    fraction of the (position, seat) pairs: neighbouring positions are nearly alike, and all of
    them from tens of thousands of games don't fit in memory."""
    agents = [re.RustAgent(s, seed=int(rng.integers(1 << 62))) for s in specs]
    xs, ys = [], []
    start = time.monotonic()
    for g in range(games):
        game = new_game(rng)
        seats = [agents[int(rng.integers(len(agents)))] for _ in range(game.num_players)]
        while not game.is_over:
            for seat in range(game.num_players):
                if keep < 1.0 and rng.random() >= keep:
                    continue
                full = game.full_observation(seat)
                xs.append(game.history_observation(seat) if history else full[: re.OBS_SIZE])
                ys.append(full[re.OBS_SIZE :])
            game.step(seats[game.current_player].act(game))
        if (g + 1) % 500 == 0:
            print(f"{g + 1} games, {len(xs)} samples, {time.monotonic() - start:.0f}s", flush=True)
    return np.stack(xs), np.stack(ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    parser.add_argument("--out", type=Path, default=Path("models/belief.bin"))
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=float, default=1.0,
                        help="fraction of (position, seat) samples kept, to bound memory")
    parser.add_argument("--history", action="store_true",
                        help="also read the public-history features (who played what, in order)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x, y = generate(args.games, args.agents, rng, args.history, args.keep)
    perm = rng.permutation(len(x))
    split = len(x) // 10
    val, train = perm[:split], perm[split:]
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)

    net = mlp(re.HIST_OBS_SIZE if args.history else re.OBS_SIZE, args.hidden, re.HIDDEN_SIZE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    def val_loss() -> float:
        with torch.no_grad():
            i = torch.from_numpy(val)
            return nn.functional.mse_loss(net(xt[i]), yt[i]).item()

    print(f"{len(x)} samples; predict-zero val loss {float((y[val] ** 2).mean()):.5f}")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        for s in range(0, len(train), args.batch):
            i = torch.from_numpy(train[s : s + args.batch])
            loss = nn.functional.mse_loss(net(xt[i]), yt[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}  val {val_loss():.5f}", flush=True)
    export(net, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
