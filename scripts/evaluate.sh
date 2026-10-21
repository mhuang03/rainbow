#!/usr/bin/env bash
# Standard evaluation table for one agent spec: relative win share against each reference
# opponent at 2, 3, 4 and 6 players (duplicate tournaments, the agent in one seat).
#
# Usage: scripts/evaluate.sh SPEC [DEALS] [OPPONENT...]
#   e.g. scripts/evaluate.sh dqn:path=python/models/dqn.bin 300 economist dmc:path=models/dmc_v2.bin
set -euo pipefail
cd "$(dirname "$0")/.."
spec=${1:?usage: scripts/evaluate.sh SPEC [DEALS] [OPPONENT...]}
deals=${2:-300}
shift $(( $# >= 2 ? 2 : 1 ))
opponents=("$@")
if [ ${#opponents[@]} -eq 0 ]; then
    opponents=(economist dmc:path=models/dmc_v2.bin)
fi
arena=./target/release/rainbow
[ -x "$arena" ] || cargo build --release -q

printf '%-34s' "opponent"
for n in 2 3 4 6; do printf '%8s' "${n}p"; done
echo
for opp in "${opponents[@]}"; do
    printf '%-34s' "$opp"
    for n in 2 3 4 6; do
        lineup=("$spec")
        for _ in $(seq 2 "$n"); do lineup+=("$opp"); done
        rel=$("$arena" arena -d "$deals" "${lineup[@]}" | awk -v s="$spec" 'index($0, s) == 1 { print $(NF-1) }')
        printf '%8s' "$rel"
    done
    echo
done
