"""Rainbow Max: the strongest DMC network this project can train, on one machine with a GPU.

Everything the experiments in ROADMAP.md found to work, at a much larger scale:

- Starts from the best network so far (the distilled league ensemble), grown function-preservingly
  (Net2Wider + Net2Deeper) to a bigger network, so it never starts worse than that.
- Learns Deep Monte Carlo values (every legal move's final return) from games of 2 to 6 players.
- League play: learner seats face a pool of the strongest Rust agents plus snapshots of itself,
  drawn by prioritized fictitious self-play (the opponents it beats least come up most). New
  snapshots join every `--snapshot-every` minutes.
- Many games at once: one Rust game batch per CPU core, stepped in parallel threads (the Rust
  side releases the GIL), with every learner move scored in a single GPU batch.
- GPU learner: a large replay buffer kept on the GPU, bf16 autocast, gradient clipping.
- Resource limits: `--cores` physical cores (affinity pinned; children inherit it) and
  `--gpu-share` of the GPU's time and memory.
- Checkpoints: Ctrl-C (or the time limit) saves everything; `--resume` carries on.

    python train_max.py --cores 4 --gpu-share 0.5 --out models/max.bin

The exported network plays as `dmc:path=models/max.bin` (add `counter=0.04` for card-counting
tie breaks) and steers the search as `ismcts:n=200,prior=models/max.bin,leaf=models/max.bin`.
`eval_max.py` rates it in random mixed fields.
"""

from __future__ import annotations

import argparse
import copy
import math
import re as regex
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# resource limits before anything starts thread pools
from resources import GpuShare, limit_cpus

import torch
from torch import nn

import rainbow_engine as re
from rl_common import ARENA, arena_relative, deepen, export, final_returns_row, load, mlp, widen

ACTIONS = re.action_features()  # (NUM_ACTIONS, ACTION_FEATURES)
MODELS = Path(__file__).parent.parent / "models"


def league(models: Path) -> list[str]:
    """The fixed part of the opponent pool: the strongest fast agents, of every kind."""
    m = lambda f: str(models / f)  # noqa: E731
    d = m("dmc_distilled1.bin")
    pool = [
        "economist",
        "counter",
        f"dmc:path={d},path2={m('dmc_spec2p_1.bin')},path3={m('dmc_spec3p_1.bin')}+{d},"
        f"path4={m('dmc_spec4p_1.bin')},counter=0.04",
        f"dmc:path={m('dmc_league7.bin')}+{m('dmc_league8.bin')}+{m('dmc_league9.bin')}",
        f"hybrid:path={d}",
        f"dmc:path={m('dmc_v2.bin')}",
    ]
    return pool


def build_network(args: argparse.Namespace) -> nn.Sequential:
    """`--hidden` from scratch, or `--init` grown to `--hidden`: widened to the target widths,
    then deepened with identity layers. Both steps keep the function, so play starts as strong
    as the starting network."""
    if args.init is None:
        return mlp(re.OBS_SIZE + re.ACTION_FEATURES, args.hidden, 1)
    net = load(args.init)
    depth = sum(isinstance(m, nn.Linear) for m in net) - 1
    target = args.hidden
    if len(target) < depth:
        raise SystemExit(f"--hidden needs at least {depth} layers to grow {args.init}")
    # widen the starting layers to the target's first depth-1 widths and its last width, then
    # insert the remaining layers (as wide as the one before them) just before the last
    widths = target[: depth - 1] + [target[-1]]
    extra = len(target) - depth
    if extra and any(w != target[depth - 2] for w in target[depth - 2 : depth - 1 + extra]):
        raise SystemExit("inserted layers copy the width before them: e.g. --hidden 1024 1024 1024 512")
    # a little noise lets copied units diverge; 1e-4 keeps the outputs within about 1%
    net = widen(net, widths, noise=1e-4, seed=args.seed)
    if extra:
        net = deepen(net, after=depth - 2, count=extra)
    return net


class Replay:
    """Ring buffer of (input row, target return) on the training device."""

    def __init__(self, capacity: int, width: int, device: torch.device):
        self.x = torch.zeros((capacity, width), dtype=torch.float32, device=device)
        self.y = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.capacity, self.size, self.next = capacity, 0, 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        n = len(x)
        if n > self.capacity:
            x, y, n = x[-self.capacity :], y[-self.capacity :], self.capacity
        xt = torch.from_numpy(x).to(self.x.device, non_blocking=True)
        yt = torch.from_numpy(y).to(self.y.device, non_blocking=True)
        first = min(n, self.capacity - self.next)
        self.x[self.next : self.next + first] = xt[:first]
        self.y[self.next : self.next + first] = yt[:first]
        if first < n:
            self.x[: n - first] = xt[first:]
            self.y[: n - first] = yt[first:]
        self.next = (self.next + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (n,), device=self.x.device)
        return self.x[idx], self.y[idx]


class Worker:
    """One batch of games, stepped on its own thread, and each game's learner decisions so far."""

    def __init__(self, envs: int, opponents: list[str], args: argparse.Namespace, seed: int):
        self.env = re.VecEnv(envs, seed=seed, players=args.players, opponents=opponents,
                             learners=args.learners, pfsp=args.pfsp)
        self.history: list[list[tuple[int, np.ndarray, int]]] = [[] for _ in range(envs)]
        self.opponents = opponents


def choose(q: np.ndarray, game_of_row: np.ndarray, starts: np.ndarray, counts: np.ndarray,
           epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Per game, the row index of the best-valued legal move, or a random one with probability
    epsilon (vectorized over games)."""
    order = np.lexsort((-q, game_of_row))  # within each game, best first
    best = order[starts]
    explore = rng.random(len(counts)) < epsilon
    random_pick = starts + (rng.random(len(counts)) * counts).astype(np.int64)
    return np.where(explore, random_pick, best)


def label(spec: str) -> str:
    """A short name for an agent spec: its kind and first network file."""
    nets = regex.findall(r"([\w.-]+)\.bin", spec)
    return spec.split(":", 1)[0] + (f" {nets[0]}" if nets else "")


def save_checkpoint(path: Path, net: nn.Module, opt: torch.optim.Optimizer, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "state": state}, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=MODELS / "max.bin", help="exported network")
    parser.add_argument("--minutes", type=float, default=0, help="stop after this long (0: until Ctrl-C)")
    parser.add_argument("--resume", action="store_true", help="carry on from the checkpoint next to --out")
    parser.add_argument("--cores", type=int, default=4, help="physical CPU cores to use")
    parser.add_argument("--gpu-share", type=float, default=0.5, help="share of GPU time and memory to use")
    parser.add_argument("--init", type=Path, default=MODELS / "dmc_distilled1.bin",
                        help="network to grow from ('none' to start from scratch)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 1024, 512])
    parser.add_argument("--envs", type=int, default=256, help="games per core")
    parser.add_argument("--players", type=int, default=None, help="one player count only (default 2 to 6)")
    parser.add_argument("--learners", type=int, default=2, help="learner seats per game")
    parser.add_argument("--opponents", nargs="*", default=None, help="Rust agent specs (default: the league)")
    parser.add_argument("--pfsp", type=float, default=1.0, help="opponent weight (1 - win rate)^P")
    parser.add_argument("--snapshots", type=int, default=4, help="own snapshots kept in the pool")
    parser.add_argument("--snapshot-every", type=float, default=30, help="minutes between snapshots")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2000,
                        help="updates over which the learning rate ramps up from zero, so the grown "
                             "network and fresh optimizer don't jolt it off its start")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--replay-ratio", type=float, default=4,
                        help="times each decision is trained on, on average")
    parser.add_argument("--buffer", type=int, default=2_000_000)
    parser.add_argument("--epsilon", type=float, nargs=2, default=[0.05, 0.01],
                        help="exploration at the start and after --epsilon-hours")
    parser.add_argument("--epsilon-hours", type=float, default=12)
    parser.add_argument("--eval-every", type=float, default=30, help="minutes between progress checks")
    parser.add_argument("--eval-deals", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if str(args.init).lower() == "none":
        args.init = None

    cpus = limit_cpus(args.cores)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = GpuShare(args.gpu_share, device)
    print(f"device {device}{' ' + torch.cuda.get_device_name(device) if device.type == 'cuda' else ''}, "
          f"cpus {cpus or 'unpinned'}, gpu share {args.gpu_share if gpu.enabled else 'n/a'}", flush=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    checkpoint = args.out.with_suffix(".pt")
    snap_dir = args.out.parent / f"{args.out.stem}_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    net = build_network(args).to(device)
    width = net[0].in_features
    if width != re.OBS_SIZE + re.ACTION_FEATURES:
        raise SystemExit(f"network takes {width} inputs; DMC networks take {re.OBS_SIZE + re.ACTION_FEATURES}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    state = {"seconds": 0.0, "decisions": 0, "updates": 0, "games": 0, "snapshots": []}
    if args.resume:
        ck = torch.load(checkpoint, map_location=device)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        print(f"resumed after {state['seconds'] / 3600:.1f} h, {state['decisions']:,} decisions", flush=True)
    sizes = [m.out_features for m in net if isinstance(m, nn.Linear)][:-1]
    params = sum(p.numel() for p in net.parameters())
    print(f"network {width}-{'-'.join(map(str, sizes))}-1, {params / 1e6:.1f}M parameters", flush=True)

    base_pool = args.opponents if args.opponents is not None else league(MODELS)
    snapshot_specs = lambda: [f"dmc:path={p}" for p in state["snapshots"][-args.snapshots:]]  # noqa: E731

    def make_workers() -> list[Worker]:
        pool = base_pool + snapshot_specs()
        return [Worker(args.envs, pool, args, seed=int(rng.integers(1 << 62))) for _ in range(args.cores)]

    workers = make_workers()
    replay = Replay(args.buffer, width, device)
    threads = ThreadPoolExecutor(max_workers=args.cores)

    def amp():
        """bf16 autocast on the GPU; full precision on the CPU."""
        if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    stop = False

    def request_stop(*_):
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True
        print("stopping after this step (Ctrl-C again to quit at once)", flush=True)

    signal.signal(signal.SIGINT, request_stop)

    session_start = time.monotonic()
    base_seconds = state["seconds"]
    next_eval = session_start + args.eval_every * 60
    next_snapshot = session_start + args.snapshot_every * 60
    losses: list[float] = []
    rate_mark = (session_start, state["decisions"])
    owed_updates = 0.0

    def elapsed() -> float:
        return base_seconds + time.monotonic() - session_start

    def export_to(path: Path) -> None:
        export(copy.deepcopy(net).cpu(), path)

    while not stop and (args.minutes <= 0 or time.monotonic() - session_start < args.minutes * 60):
        progress = min(1.0, elapsed() / 3600 / max(args.epsilon_hours, 1e-9))
        epsilon = args.epsilon[0] + (args.epsilon[1] - args.epsilon[0]) * progress

        # observe every game and score every legal move of every learner in one batch
        views = [w.env.observe() for w in workers]
        obs = np.concatenate([v[0] for v in views])
        masks = np.concatenate([v[1] for v in views])
        game_of_row, actions = np.nonzero(masks)
        counts = masks.sum(axis=1)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        rows = np.concatenate([obs[game_of_row], ACTIONS[actions]], axis=1)
        with gpu.section(), torch.no_grad(), amp():
            q = net(torch.from_numpy(rows).to(device, non_blocking=True)).float().squeeze(1).cpu().numpy()
        picks = choose(q, game_of_row, starts, counts, epsilon, rng)
        chosen = actions[picks]

        # remember each decision, then step every batch in parallel while the learner trains
        offset = 0
        steps = []
        players = []  # per batch, each game's player count (a finished game is replaced in step)
        for w, (o, _) in zip(workers, views):
            n = len(o)
            seats = w.env.current_players()
            players.append(w.env.num_players())
            for i in range(n):
                w.history[i].append((int(seats[i]), o[i].copy(), int(chosen[offset + i])))
            steps.append(threads.submit(w.env.step, chosen[offset : offset + n].astype(np.int64)))
            offset += n
        state["decisions"] += len(obs)

        owed_updates += args.replay_ratio * len(obs) / args.batch
        if replay.size >= args.batch * 4:
            while owed_updates >= 1:
                owed_updates -= 1
                x, y = replay.sample(args.batch)
                for group in opt.param_groups:
                    group["lr"] = args.lr * min(1.0, (state["updates"] + 1) / max(args.warmup, 1))
                with gpu.section():
                    with amp():
                        loss = nn.functional.mse_loss(net(x).float().squeeze(1), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                losses.append(loss.item())
                state["updates"] += 1

        # finished games become training rows: every learner decision, valued by its final return
        for w, step, counts_now in zip(workers, steps, players):
            done, returns, scores = step.result()
            new_x, new_y = [], []
            for i in np.flatnonzero(done):
                hist = w.history[i]
                if hist:
                    final = final_returns_row(returns[i], scores[i], int(counts_now[i]))
                    o = np.stack([h[1] for h in hist])
                    a = np.array([h[2] for h in hist])
                    new_x.append(np.concatenate([o, ACTIONS[a]], axis=1))
                    new_y.append(final[[h[0] for h in hist]])
                w.history[i] = []
                state["games"] += 1
            if new_x:
                replay.add(np.concatenate(new_x).astype(np.float32), np.concatenate(new_y).astype(np.float32))

        now = time.monotonic()
        if now >= next_snapshot:
            next_snapshot = now + args.snapshot_every * 60
            path = snap_dir / f"snap{len(state['snapshots']):04d}.bin"
            export_to(path)
            state["snapshots"].append(str(path))
            workers = make_workers()  # partial games are dropped; their decisions are few
            print(f"snapshot {path.name} joins the league", flush=True)

        if now >= next_eval:
            next_eval = now + args.eval_every * 60
            state["seconds"] = elapsed()
            export_to(args.out)
            save_checkpoint(checkpoint, net, opt, state)
            rate = (state["decisions"] - rate_mark[1]) / max(now - rate_mark[0], 1e-9)
            rate_mark = (now, state["decisions"])
            vs = arena_relative(f"dmc:path={args.out}", ["economist"] * 3,
                                args.eval_deals, ARENA, threads=args.cores)
            loss = np.mean(losses[-500:]) if losses else math.nan
            print(f"{state['seconds'] / 3600:6.2f} h  games {state['games']:>10,}  decisions "
                  f"{state['decisions']:>12,} ({rate:,.0f}/s)  updates {state['updates']:>9,}  "
                  f"loss {loss:.4f}  eps {epsilon:.3f}  vs 3x economist {vs}", flush=True)
            names = [label(s) for s in base_pool + snapshot_specs()]
            rates = workers[0].env.opponent_win_rates()
            print("           learner ahead of: " + "  ".join(f"{n} {r:.2f}" for n, r in zip(names, rates)),
                  flush=True)

    state["seconds"] = elapsed()
    export_to(args.out)
    save_checkpoint(checkpoint, net, opt, state)
    print(f"saved {args.out} and {checkpoint} after {state['seconds'] / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
