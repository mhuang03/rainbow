#!/usr/bin/env bash
# Expert iteration: alternate network-guided ISMCTS self-play and fine-tuning the DMC network on
# the search's move values. Each round's network guides the next round's search.
#
# Usage: scripts/expert_iteration.sh INIT_MODEL OUT_DIR [ROUNDS] [GAMES] [SEARCH_OPTIONS]
#   e.g. scripts/expert_iteration.sh models/dmc_v2.bin runs/exit 3 1500 n=200
# Needs the Python bindings and PyTorch (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
init=${1:?usage: scripts/expert_iteration.sh INIT_MODEL OUT_DIR [ROUNDS] [GAMES] [SEARCH_OPTIONS]}
out=${2:?usage: scripts/expert_iteration.sh INIT_MODEL OUT_DIR [ROUNDS] [GAMES] [SEARCH_OPTIONS]}
rounds=${3:-3}
games=${4:-1500}
search=${5:-n=200}
threads=${THREADS:-0}
mkdir -p "$out"
cargo build --release -q
model=$init
for round in $(seq 1 "$rounds"); do
    data="$out/selfplay_$round.bin"
    next="$out/dmc_exit_$round.bin"
    echo "== round $round: self-play with $model"
    ./target/release/rainbow selfplay -g "$games" -s "$round" -t "$threads" -o "$data" \
        "$search,prior=$model,leaf=$model"
    echo "== round $round: fine-tune"
    (cd python && python train_exit.py --init "../$model" --data "../$data" --out "../$next")
    model=$next
done
echo "final model: $model"
