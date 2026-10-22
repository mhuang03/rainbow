#!/usr/bin/env bash
# Builds the server, starts it with agents moving instantly, and runs the browser tests.
# Needs Node with Playwright (and optionally axe-core): `npm install playwright axe-core`.
set -euo pipefail
cd "$(dirname "$0")/../.."
port=${PORT:-8099}
cargo build --release -q -p rainbow-server
./target/release/rainbow-server --port "$port" --pace 0.05 > /dev/null 2>&1 &
server=$!
trap 'kill $server' EXIT
for _ in $(seq 50); do curl -s "localhost:$port/api/agents" > /dev/null && break; sleep 0.1; done
BASE_URL="http://localhost:$port" node server/e2e/e2e.js
