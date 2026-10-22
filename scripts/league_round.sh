#!/usr/bin/env bash
# One round of DMC league training: continue a network against a pool of opponents, then
# compare it head-to-head with the network it started from (2 and 4 players).
#
# Usage: scripts/league_round.sh INIT OUT [MINUTES] [OPPONENT...]
#   e.g. scripts/league_round.sh models/dmc_league8.bin models/dmc_league9.bin 90 \
#            economist counter dmc:path=models/dmc_league7.bin dmc:path=models/dmc_league8.bin
#
# With no opponents given, the pool is economist, counter and the starting network itself.
# Set WIDEN="1024 1024 512" to grow the network first (Net2Net, function-preserving).
# Needs the Python environment with torch and the rainbow_engine module (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
init=${1:?usage: scripts/league_round.sh INIT OUT [MINUTES] [OPPONENT...]}
out=${2:?usage: scripts/league_round.sh INIT OUT [MINUTES] [OPPONENT...]}
minutes=${3:-90}
shift $(( $# >= 3 ? 3 : 2 ))
opponents=("$@")
if [ ${#opponents[@]} -eq 0 ]; then
    opponents=(economist counter "dmc:path=$init")
fi
# relative paths are from the repository root; the trainer runs in python/
python_opponents=()
for o in "${opponents[@]}"; do
    case $o in *path=/*) python_opponents+=("$o") ;; *) python_opponents+=("${o//path=/path=../}") ;; esac
done
from_python() { case $1 in /*) echo "$1" ;; *) echo "../$1" ;; esac; }
widen=()
if [ -n "${WIDEN:-}" ]; then read -ra sizes <<< "$WIDEN"; widen=(--widen "${sizes[@]}"); fi

cargo build --release -q
(cd python && python train_dmc.py --init "$(from_python "$init")" "${widen[@]}" --minutes "$minutes" \
    --lr 1e-4 --epsilon 0.05 0.02 --opponents "${python_opponents[@]}" --learners 2 \
    --eval-every 15 --out "$(from_python "$out")")

arena=./target/release/rainbow
new="dmc:path=$out"
old="dmc:path=$init"
echo "== $out vs $init, 2 players"
"$arena" arena -d 400 "$new" "$old" | sed -n 3,4p
echo "== $out vs $init, 4 players (one seat against three)"
"$arena" arena -d 400 "$new" "$old" "$old" "$old" | sed -n 3,4p
