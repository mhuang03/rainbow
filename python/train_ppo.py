"""PPO self-play training with a perfect-information critic (after PerfectDou, NeurIPS 2022).

The policy network sees only the acting player's observation and outputs one logit per action,
with illegal actions masked. The value network also sees every hand and reserve: it is only used
to compute advantages during training, so it may use information players don't have, which
makes its estimates far less noisy. Returns are Monte Carlo (GAE with lambda = 1): each
decision's return is the deciding player's final win share plus a small score-margin term.

With --oracle-fraction F, the policy also gets the hidden features for the first part of
training, scaled by a factor that decays from 1 to 0 over the first F of the run, so the policy
learns with perfect information and then weans itself off it (Suphx's "oracle guiding"). The
exported network keeps only the observation inputs.

    python train_ppo.py --minutes 30 --out models/ppo.bin
    ../target/release/rainbow arena ppo:path=python/models/ppo.bin economist economist economist
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


class Rollouts:
    """Decisions from finished episodes, waiting for a policy update."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.obs: list[np.ndarray] = []
        self.full: list[np.ndarray] = []
        self.mask: list[np.ndarray] = []
        self.action: list[int] = []
        self.logp: list[float] = []
        self.ret: list[float] = []

    def __len__(self) -> int:
        return len(self.action)

    def tensors(self):
        return (
            torch.from_numpy(np.stack(self.obs)),
            torch.from_numpy(np.stack(self.full)),
            torch.from_numpy(np.stack(self.mask)),
            torch.tensor(self.action),
            torch.tensor(self.logp),
            torch.tensor(self.ret, dtype=torch.float32),
        )


def policy_input(obs: np.ndarray, full: np.ndarray, oracle: float, use_oracle: bool) -> np.ndarray:
    if not use_oracle:
        return obs
    return np.concatenate([obs, full[:, re.OBS_SIZE :] * oracle], axis=1)


def masked_logits(policy: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return policy(x).masked_fill(~mask, -1e9)


class Actors:
    def __init__(self, count: int, rng: np.random.Generator):
        self.rng = rng
        self.games = [new_game(rng) for _ in range(count)]
        # per game: (seat, obs, full, mask, action, logp) for every decision so far
        self.history: list[list[tuple]] = [[] for _ in range(count)]
        self.episodes = 0

    def step(self, policy: nn.Module, oracle: float, use_oracle: bool, out: Rollouts) -> int:
        obs, masks = re.batch_observations(self.games)
        full = re.batch_full_observations(self.games)
        x = policy_input(obs, full, oracle, use_oracle)
        with torch.no_grad():
            logits = masked_logits(policy, torch.from_numpy(x), torch.from_numpy(masks))
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logps = dist.log_prob(actions).numpy()
        actions = actions.numpy()

        for i, game in enumerate(self.games):
            a = int(actions[i])
            self.history[i].append((game.current_player, obs[i], full[i], masks[i], a, float(logps[i])))
            game.step(a)
            if game.is_over:
                returns = final_returns(game)
                for seat, o, f, m, act, lp in self.history[i]:
                    out.obs.append(o)
                    out.full.append(f)
                    out.mask.append(m)
                    out.action.append(act)
                    out.logp.append(lp)
                    out.ret.append(returns[seat])
                self.history[i] = []
                self.games[i] = new_game(self.rng)
                self.episodes += 1
        return len(self.games)


def export_policy(policy: nn.Sequential, use_oracle: bool, path: Path) -> None:
    """Exports the policy with observation inputs only (dropping oracle columns)."""
    if use_oracle:
        first = policy[0]
        trimmed = nn.Linear(re.OBS_SIZE, first.out_features)
        with torch.no_grad():
            trimmed.weight.copy_(first.weight[:, : re.OBS_SIZE])
            trimmed.bias.copy_(first.bias)
        policy = nn.Sequential(trimmed, *list(policy)[1:])
    export(policy, path)


def update(policy, value, optimizer, data, oracle, use_oracle, args) -> tuple[float, float, float]:
    obs, full, mask, action, old_logp, ret = data
    if use_oracle:
        x_all = torch.cat([obs, full[:, re.OBS_SIZE :] * oracle], dim=1)
    else:
        x_all = obs
    with torch.no_grad():
        adv = ret - value(full).squeeze(1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(action)
    stats = []
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, args.minibatch):
            idx = perm[start : start + args.minibatch]
            logits = masked_logits(policy, x_all[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(action[idx])
            ratio = (logp - old_logp[idx]).exp()
            clipped = ratio.clamp(1 - args.clip, 1 + args.clip)
            policy_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
            entropy = dist.entropy().mean()
            value_loss = nn.functional.mse_loss(value(full[idx]).squeeze(1), ret[idx])
            loss = policy_loss - args.entropy * entropy + args.value_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), 1.0)
            optimizer.step()
            stats.append((policy_loss.item(), value_loss.item(), entropy.item()))
    return tuple(np.mean(stats, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--out", type=Path, default=Path("models/ppo.bin"))
    parser.add_argument("--actors", type=int, default=256)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout", type=int, default=32_768, help="decisions per update")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--oracle-fraction", type=float, default=0.0,
                        help="fraction of the run over which oracle inputs fade out (0 disables)")
    parser.add_argument("--eval-every", type=float, default=5, help="minutes between arena checks")
    parser.add_argument("--eval-deals", type=int, default=200)
    parser.add_argument("--arena", type=Path, default=ARENA)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    use_oracle = args.oracle_fraction > 0
    policy = mlp(re.OBS_SIZE + (re.HIDDEN_SIZE if use_oracle else 0), args.hidden, re.NUM_ACTIONS)
    value = mlp(re.FULL_OBS_SIZE, args.hidden, 1)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=args.lr)
    actors = Actors(args.actors, rng)
    rollouts = Rollouts()

    start = time.monotonic()
    deadline = start + args.minutes * 60
    next_eval = start + args.eval_every * 60
    decisions = updates = 0
    stats = (float("nan"),) * 3
    while time.monotonic() < deadline:
        progress = (time.monotonic() - start) / (deadline - start)
        oracle = max(0.0, 1.0 - progress / args.oracle_fraction) if use_oracle else 0.0
        decisions += actors.step(policy, oracle, use_oracle, rollouts)
        if len(rollouts) >= args.rollout:
            stats = update(policy, value, optimizer, rollouts.tensors(), oracle, use_oracle, args)
            rollouts.clear()
            updates += 1

        if time.monotonic() >= next_eval:
            next_eval += args.eval_every * 60
            export_policy(policy, use_oracle, args.out)
            # with oracle inputs still active the exported policy is not yet meaningful
            rel = arena_relative(f"ppo:path={args.out}", ["economist"] * 3, args.eval_deals, args.arena)
            print(
                f"{(time.monotonic() - start) / 60:6.1f} min  episodes {actors.episodes:>8}  "
                f"decisions {decisions:>9}  updates {updates:>5}  policy {stats[0]:+.4f}  "
                f"value {stats[1]:.4f}  entropy {stats[2]:.3f}  oracle {oracle:.2f}  vs 3x economist {rel}",
                flush=True,
            )

    export_policy(policy, use_oracle, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
