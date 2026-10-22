"""Policy distillation: train a fast policy network to imitate guided ISMCTS.

`rainbow selfplay` records, for every decision, how often the search visited each root move.
Those visit counts are a policy target (as in AlphaZero): a network mapping the observation to
one logit per action is trained with cross-entropy against the visit distribution. The result
plays in microseconds, like DMC, but learns from the search's choices rather than from noisy
game returns.

    ../target/release/rainbow selfplay -g 2000 -o data.bin n=200,prior=models/dmc_league.bin,leaf=models/dmc_league.bin
    python train_distill.py --data data.bin --out models/distill.bin
    ../target/release/rainbow arena distill:path=python/models/distill.bin economist economist economist
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re
from rl_common import arena_relative, export, mlp

RECORD = re.OBS_SIZE + 2 * re.NUM_ACTIONS + 1


def training_rows(paths: list[Path], temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """(observations, target policies). Visit counts are raised to 1 / temperature and normalized,
    so temperatures below 1 sharpen the target towards the most-visited move."""
    records = np.concatenate([np.fromfile(p, "<f4").reshape(-1, RECORD) for p in paths])
    obs = records[:, : re.OBS_SIZE]
    visits = records[:, re.OBS_SIZE + re.NUM_ACTIONS : RECORD - 1].astype(np.float64)
    # forced moves teach nothing
    keep = (visits > 0).sum(axis=1) > 1
    obs, visits = obs[keep], visits[keep]
    target = visits ** (1.0 / temperature)
    target /= target.sum(axis=1, keepdims=True)
    print(f"{len(records)} decisions, {len(obs)} with a choice")
    return obs.astype(np.float32), target.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="selfplay record files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0, help="visit-count temperature")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval-deals", type=int, default=200)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x, p = training_rows(args.data, args.temperature)
    perm = rng.permutation(len(x))
    split = max(len(x) // 10, 1)
    val, train = perm[:split], perm[split:]
    xt, pt = torch.from_numpy(x), torch.from_numpy(p)
    net = mlp(re.OBS_SIZE, args.hidden, re.NUM_ACTIONS)
    # illegal and unvisited moves get target 0, so the full softmax learns to push them down
    lsm = nn.LogSoftmax(dim=1)

    def loss_on(idx: np.ndarray) -> tuple[float, float]:
        """(cross-entropy, how often the network's top move is the search's most-visited)."""
        with torch.no_grad():
            i = torch.from_numpy(idx)
            logits = net(xt[i])
            ce = float(-(pt[i] * lsm(logits)).sum(1).mean())
            agree = float((logits.argmax(1) == pt[i].argmax(1)).float().mean())
            return ce, agree

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(args.epochs, 1))
    best, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        net.train()
        for start in range(0, len(train), args.batch):
            i = torch.from_numpy(train[start : start + args.batch])
            loss = -(pt[i] * lsm(net(xt[i]))).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        ce, agree = loss_on(val)
        print(f"epoch {epoch}  val ce {ce:.4f}  top-move agreement {agree:.3f}", flush=True)
        if ce < best:
            best, best_state = ce, {k: v.clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    export(net, args.out)
    print(f"saved {args.out}")
    print("vs 3x economist:", arena_relative(f"distill:path={args.out}", ["economist"] * 3, args.eval_deals))


if __name__ == "__main__":
    main()
