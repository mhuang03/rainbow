"""Shared helpers for the training scripts: model export, game sampling and arena evaluation."""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rainbow_engine as re

ARENA = Path(__file__).parent.parent / "target/release/rainbow"
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


def arena_relative(spec: str, opponents: list[str], deals: int, arena: Path = ARENA) -> str:
    """Relative win share of `spec` in a Rust arena tournament, or "n/a" without the binary."""
    if not arena.exists():
        return "n/a"
    result = subprocess.run(
        [str(arena), "arena", "-d", str(deals), spec, *opponents],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith(spec):
            return line[len(spec) :].split()[3]
    raise RuntimeError(f"no result for {spec} in:\n{result.stdout}")
