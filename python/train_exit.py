"""Expert iteration for the DMC network: fit it to ISMCTS's move values.

`rainbow selfplay` plays games with network-guided ISMCTS and records, for every decision, the
search's mean value for each root move. Search with the network is stronger than the network
alone, so regressing the network onto those values improves it, and the improved network makes
the next round of search stronger.

    ../target/release/rainbow selfplay -g 1000 -o data.bin n=200,prior=models/dmc_v2.bin,leaf=models/dmc_v2.bin
    python train_exit.py --init ../models/dmc_v2.bin --data data.bin --out models/dmc_exit1.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re
from rl_common import ARENA, arena_relative, export, load

ACTIONS = re.action_features()
RECORD = re.OBS_SIZE + 2 * re.NUM_ACTIONS + 1


def training_rows(paths: list[Path], return_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inputs, targets, weights): one row per visited root move, weighted by visit share.

    With `return_weight` w, the most-visited move's target (usually the move played) blends in the
    game's actual result: (1 - w) * search value + w * final return."""
    records = np.concatenate([np.fromfile(p, "<f4").reshape(-1, RECORD) for p in paths])
    obs = records[:, : re.OBS_SIZE]
    values = records[:, re.OBS_SIZE : re.OBS_SIZE + re.NUM_ACTIONS]
    visits = records[:, re.OBS_SIZE + re.NUM_ACTIONS : RECORD - 1]
    rec, act = np.nonzero(visits > 0)
    x = np.concatenate([obs[rec], ACTIONS[act]], axis=1)
    y = values[rec, act].copy()
    if return_weight > 0:
        chosen = visits[rec, act] == visits[rec].max(axis=1)
        final = records[rec, RECORD - 1]
        y[chosen] = (1 - return_weight) * y[chosen] + return_weight * final[chosen]
    w = visits[rec, act] / visits[rec].sum(axis=1)
    print(f"{len(records)} decisions, {len(x)} (state, move) rows")
    return x.astype(np.float32), y.astype(np.float32), w.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", type=Path, required=True, help="network to start from")
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="selfplay record files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2, help="validation error usually stops improving after 1-2")
    parser.add_argument("--return-weight", type=float, default=0.0,
                        help="blend the game result into the most-visited move's target")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--eval-deals", type=int, default=200)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    net = load(args.init)
    x, y, w = training_rows(args.data, args.return_weight)
    # hold out 10% of rows to watch for overfitting
    perm = rng.permutation(len(x))
    split = len(x) // 10
    val, train = perm[:split], perm[split:]
    xt, yt, wt = (torch.from_numpy(a) for a in (x, y, w))

    def loss_on(idx: np.ndarray) -> float:
        with torch.no_grad():
            i = torch.from_numpy(idx)
            err = net(xt[i]).squeeze(1) - yt[i]
            return float((wt[i] * err**2).sum() / wt[i].sum())

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"epoch 0  train {loss_on(train):.5f}  val {loss_on(val):.5f}")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        for start in range(0, len(train), args.batch):
            i = torch.from_numpy(train[start : start + args.batch])
            err = net(xt[i]).squeeze(1) - yt[i]
            loss = (wt[i] * err**2).sum() / wt[i].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}  train {loss_on(train):.5f}  val {loss_on(val):.5f}", flush=True)

    export(net, args.out)
    print(f"saved {args.out}")
    for spec in (f"dmc:path={args.out}", f"dmc:path={args.init}"):
        print(spec, "vs 3x economist:", arena_relative(spec, ["economist"] * 3, args.eval_deals))


if __name__ == "__main__":
    main()
