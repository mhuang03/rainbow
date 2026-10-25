"""Rates a network from train_max.py against random lineups of every browser opponent.

For each player count, `rainbow ladder` deals many random lineups from a pool of the trained
network and all of the browser's opponents (as `rainbow-server --list-presets` reports them, so
the field matches the one behind the menu's ratings) and reports each agent's share of the wins
over an even share. The 4-player rating is written next to the network (`max.rating`), and the
browser then offers it as the opponent "Max".

    python eval_max.py --model models/max.bin --cores 4
    python eval_max.py --model models/max.bin --players 4 --lineups 200 --search
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from resources import limit_cpus

ROOT = Path(__file__).parent.parent
EXE = ".exe" if os.name == "nt" else ""
RAINBOW = ROOT / "target" / "release" / f"rainbow{EXE}"
SERVER = ROOT / "target" / "release" / f"rainbow-server{EXE}"


def presets() -> list[tuple[str, float, str]]:
    """The browser's opponents: (id, rating, spec)."""
    if not SERVER.exists():
        raise SystemExit(f"{SERVER} not found: run `cargo build --release -p rainbow-server` first")
    out = subprocess.run([str(SERVER), "--models", "models", "--list-presets"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    rows = [line.split("\t") for line in out.splitlines() if line.strip()]
    return [(i, float(r), s) for i, r, s in rows if i != "max"]


def ladder(players: int, lineups: int, deals: int, cores: int, specs: list[str]) -> dict[str, tuple[float, float]]:
    """Relative win share and its standard error for every spec in one ladder."""
    cmd = [str(RAINBOW), "ladder", "-p", str(players), "-l", str(lineups), "-d", str(deals),
           "-t", str(cores), *specs]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    result = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in specs and parts[1].isdigit():
            result[parts[0]] = (float(parts[2]), float(parts[3]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/max.bin"),
                        help="network to rate (relative to the repository root, or absolute)")
    parser.add_argument("--players", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--lineups", type=int, default=80, help="random lineups per player count")
    parser.add_argument("--deals", type=int, default=20, help="duplicate deals per lineup")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--search", action="store_true",
                        help="also rate Guided ISMCTS steered by the network (slow)")
    parser.add_argument("--no-rating", action="store_true", help="don't write max.rating")
    args = parser.parse_args()
    limit_cpus(args.cores)
    if not RAINBOW.exists():
        raise SystemExit(f"{RAINBOW} not found: run `cargo build --release` first")
    model = args.model if args.model.is_absolute() else (ROOT / args.model)
    if not model.exists():
        raise SystemExit(f"{model} not found")
    rel = os.path.relpath(model, ROOT).replace(os.sep, "/")

    field = presets()
    names = {spec: pid for pid, _, spec in field}
    candidates = {f"dmc:path={rel},counter=0.04": "max (trained)"}
    if args.search:
        candidates[f"ismcts:n=200,prior={rel},leaf={rel}+models/dmc_distilled1.bin,"
                   f"belief=models/belief_v3.bin"] = "max-guided search (trained)"
    names.update(candidates)
    specs = list(candidates) + [spec for _, _, spec in field]
    print(f"rating {rel} against {len(field)} browser opponents, {args.lineups} random lineups "
          f"x {args.deals} deals per player count, {args.cores} cores", flush=True)

    four = None
    for players in args.players:
        start = time.monotonic()
        result = ladder(players, args.lineups, args.deals, args.cores, specs)
        ranked = sorted(result.items(), key=lambda kv: -kv[1][0])
        print(f"\n{players} players ({(time.monotonic() - start) / 60:.0f} min):")
        for rank, (spec, (rating, err)) in enumerate(ranked, 1):
            mark = "  <--" if spec in candidates else ""
            print(f"  {rank:>2}. {names[spec]:<28} {rating + 0.0:5.2f} +/- {abs(err):.2f}{mark}")
        if players == 4:
            four = result.get(next(iter(candidates)))
    if four and not args.no_rating:
        path = model.with_suffix(".rating")
        path.write_text(f"{four[0]:.2f}\n")
        print(f"\nwrote {path}: the browser offers this network as \"Max\" at {four[0]:.2f} "
              f"(restart the server to pick it up)")


if __name__ == "__main__":
    main()
