"""Shared helpers for the training scripts: model export, game sampling and arena evaluation."""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re

ARENA = Path(__file__).parent.parent / "target/release" / ("rainbow.exe" if os.name == "nt" else "rainbow")
MARGIN_WEIGHT = 0.2  # matches the Rust search agents' default reward


def export(net: nn.Sequential, path: Path) -> None:
    """Writes a ReLU MLP in the format read by `rainbow::agents::dmc::Mlp`."""
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


def load(path: Path) -> nn.Sequential:
    """Reads a network written by `export` (or any `RBMLP1` file)."""
    data = Path(path).read_bytes()
    if data[:8] != b"RBMLP1\0\0":
        raise ValueError(f"{path} is not a Rainbow MLP file")
    (count,) = struct.unpack_from("<I", data, 8)
    offset = 12
    layers: list[nn.Module] = []
    for i in range(count):
        inputs, outputs = struct.unpack_from("<II", data, offset)
        offset += 8
        w = np.frombuffer(data, "<f4", inputs * outputs, offset).reshape(outputs, inputs)
        offset += 4 * inputs * outputs
        b = np.frombuffer(data, "<f4", outputs, offset)
        offset += 4 * outputs
        layer = nn.Linear(inputs, outputs)
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(w.copy()))
            layer.bias.copy_(torch.from_numpy(b.copy()))
        layers.append(layer)
        if i < count - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def widen(net: nn.Sequential, hidden: list[int], noise: float = 1e-3, seed: int = 0) -> nn.Sequential:
    """Net2WiderNet (Chen et al., 2016): a wider ReLU MLP computing the same function.

    Each hidden layer grows to the given width by copying randomly chosen units; the next layer's
    weights from a unit and its copies are divided among them, so outputs are unchanged. A little
    noise on the copies' incoming weights lets them diverge during training."""
    rng = np.random.default_rng(seed)
    linears = [m for m in net if isinstance(m, nn.Linear)]
    if len(hidden) != len(linears) - 1:
        raise ValueError(f"need {len(linears) - 1} hidden widths, got {len(hidden)}")
    weights = [l.weight.detach().clone() for l in linears]
    biases = [l.bias.detach().clone() for l in linears]
    for i, width in enumerate(hidden):
        old = weights[i].shape[0]
        if width < old:
            raise ValueError(f"layer {i}: can't narrow {old} units to {width}")
        g = np.concatenate([np.arange(old), rng.integers(0, old, width - old)])
        counts = torch.from_numpy(np.bincount(g, minlength=old)[g]).float()
        idx = torch.from_numpy(g)
        weights[i] = weights[i][idx].clone()
        biases[i] = biases[i][idx].clone()
        weights[i][old:] += noise * torch.randn_like(weights[i][old:])
        weights[i + 1] = weights[i + 1][:, idx] / counts
    layers: list[nn.Module] = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        layer = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            layer.weight.copy_(w)
            layer.bias.copy_(b)
        layers.append(layer)
        if i < len(weights) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def deepen(net: nn.Sequential, after: int, count: int = 1) -> nn.Sequential:
    """Net2DeeperNet (Chen et al., 2016): inserts `count` identity layers after hidden layer
    `after` (0-based). Its inputs have passed a ReLU, so they are non-negative and ReLU(I x) = x:
    the network computes the same function, and training then puts the new layers to use."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    if not 0 <= after < len(linears) - 1:
        raise ValueError(f"hidden layer {after} doesn't exist")
    width = linears[after].out_features
    layers: list[nn.Module] = []
    for i, linear in enumerate(linears):
        layers.append(linear)
        if i < len(linears) - 1:
            layers.append(nn.ReLU())
        if i == after:
            for _ in range(count):
                identity = nn.Linear(width, width)
                with torch.no_grad():
                    identity.weight.copy_(torch.eye(width))
                    identity.bias.zero_()
                layers += [identity, nn.ReLU()]
    return nn.Sequential(*layers)


def mlp(inputs: int, hidden: list[int], outputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    size = inputs
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, outputs))
    return nn.Sequential(*layers)


def new_game(rng: np.random.Generator) -> re.Game:
    """A game with 2 to 6 players, a random deal and a random first leader."""
    n = int(rng.integers(re.MIN_PLAYERS, re.MAX_PLAYERS + 1))
    return re.Game(n, seed=int(rng.integers(1 << 62)), leader=int(rng.integers(n)))


def final_returns(game: re.Game) -> list[float]:
    """Win share plus a small score-margin term for every seat."""
    scores = game.scores()
    out = []
    for seat, ret in enumerate(game.returns()):
        best_other = max(s for i, s in enumerate(scores) if i != seat)
        out.append(ret + MARGIN_WEIGHT * (scores[seat] - best_other) / 100)
    return out


def final_returns_row(returns: np.ndarray, scores: np.ndarray, n: int) -> np.ndarray:
    """`final_returns` from one row of `VecEnv.step` results, for an `n`-player game."""
    r, sc = returns[:n], scores[:n].astype(np.float64)
    best_other = np.array([np.max(np.delete(sc, s)) for s in range(n)])
    return (r + MARGIN_WEIGHT * (sc - best_other) / 100).astype(np.float32)


def arena_relative(spec: str, opponents: list[str], deals: int, arena: Path = ARENA, threads: int = 0) -> str:
    """Relative win share of `spec` in a Rust arena tournament, or "n/a" without the binary.
    `threads` caps the arena's worker threads (0: one per core)."""
    if not arena.exists():
        return "n/a"
    result = subprocess.run(
        [str(arena), "arena", "-d", str(deals), *(["-t", str(threads)] if threads else []), spec, *opponents],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith(spec):
            return line[len(spec) :].split()[3]
    raise RuntimeError(f"no result for {spec} in:\n{result.stdout}")
