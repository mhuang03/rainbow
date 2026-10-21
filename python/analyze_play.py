"""Describes how agents play: lead choices, trick-winning, card economy and endgame behaviour.

Plays duplicate-style games where every seat is one of the given agents (rotating seats), then
prints per-agent statistics, so learned agents' habits can be compared with hand-written ones.

    python analyze_play.py --players 4 --games 200 economist dmc:path=../models/dmc_v2.bin
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

import rainbow_engine as re


def combo_kind(action: int) -> tuple[str, int]:
    kind, a, b = re.action_info(action)
    size = 1 if kind == "single" else (b if kind == "set" else b - a + 1)
    return kind, size


def strength(action: int) -> tuple[int, int]:
    kind, a, b = re.action_info(action)
    if kind == "single":
        return 1, a
    if kind == "set":
        return b, a
    return b - a + 1, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agents", nargs="+")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    agents = {spec: re.RustAgent(spec, seed=int(rng.integers(1 << 62))) for spec in args.agents}

    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for g in range(args.games):
        seed = int(rng.integers(1 << 62))
        lineup = [args.agents[(g + s) % len(args.agents)] for s in range(args.players)]
        game = re.Game(args.players, seed=seed, leader=g % args.players)
        while not game.is_over:
            seat = game.current_player
            spec = lineup[seat]
            trick = game.trick()
            st = stats[spec]
            action = agents[spec].act(game)
            kind, size = combo_kind(action)
            st["plays"] += 1
            st["cards"] += size
            if not trick:
                st["leads"] += 1
                st[f"lead_{kind}"] += 1
                st["lead_cards"] += size
            best = max((strength(a) for _, a in trick), default=None)
            beats = best is None or strength(action) > best
            if len(trick) == len(game.piles()) - 1 or (len(game.piles()) <= len(trick) + 1):
                st["last"] += 1
                st["last_wins"] += beats
            game.step(action)
        scores = game.scores()
        returns = game.returns()
        for seat, spec in enumerate(lineup):
            st = stats[spec]
            st["games"] += 1
            st["points"] += scores[seat]
            st["wins"] += returns[seat]
            st["cards_left"] += sum(game.hand(seat))

    header = f"{'agent':<34}{'win share':>10}{'pts':>7}{'cards/play':>11}{'pts/card':>9}{'lead set':>9}{'lead run':>9}{'lead size':>10}{'last wins':>10}{'left':>6}"
    print(f"{args.players} players, {args.games} games")
    print(header)
    for spec, st in sorted(stats.items(), key=lambda kv: -kv[1]["wins"] / kv[1]["games"]):
        leads = max(st["leads"], 1)
        print(
            f"{spec[:33]:<34}{st['wins'] / st['games']:>10.3f}{st['points'] / st['games']:>7.1f}"
            f"{st['cards'] / st['plays']:>11.2f}{st['points'] / max(st['cards'], 1):>9.2f}"
            f"{st['lead_set'] / leads:>9.2f}{st['lead_run'] / leads:>9.2f}{st['lead_cards'] / leads:>10.2f}"
            f"{st['last_wins'] / max(st['last'], 1):>10.2f}{st['cards_left'] / st['games']:>6.1f}"
        )


if __name__ == "__main__":
    main()
