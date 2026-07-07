#!/usr/bin/env bash
# Idempotently (re)launch the main sweep and the chained experiment queue
# (anomaly -> ablation -> quant check) as detached daemons. Safe to run any
# number of times: liveness is checked via pidfiles before starting anything.
# Prereq: scripts/setup_env.sh <scratch-dir> has completed.
set -uo pipefail

SCRATCH="${1:?usage: launch_runs.sh <scratch-dir>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
GGUF="$SCRATCH/models/qwen2-0.5b-q8_0.gguf"

for f in "$PY" "$SCRATCH/ppl-dump" "$GGUF" "$SCRATCH/corpora/manifest.tsv"; do
  [ -e "$f" ] || { echo "missing $f — run scripts/setup_env.sh first"; exit 1; }
done

alive() { # alive <pidfile> <cmdline-substring>
  local pid
  pid=$(cat "$1" 2>/dev/null) || return 1
  [ -n "$pid" ] && [ -d "/proc/$pid" ] && grep -qa "$2" "/proc/$pid/cmdline" 2>/dev/null
}

cd "$REPO"
mkdir -p "$SCRATCH/work"

if alive "$SCRATCH/work/run_sweep.pid" run_sweep; then
  echo "[launch] sweep already running"
else
  setsid nohup "$PY" scripts/run_sweep.py \
    --corpora "$SCRATCH/corpora" --results results/main --scratch "$SCRATCH/work" \
    --binary "$SCRATCH/ppl-dump" --model "$GGUF" \
    --languages lean,python,haskell,rust,c,javascript \
    --sorted-windows 8 --shuffled-windows 4 --ctx 16384 --budget 90000 --threads 4 \
    >> "$SCRATCH/sweep.log" 2>&1 &
  echo "[launch] sweep started"
fi

if alive "$SCRATCH/chain.pid" chain_after_sweep; then
  echo "[launch] chain already running"
else
  setsid nohup bash scripts/chain_after_sweep.sh "$SCRATCH" "$REPO" \
    >> "$SCRATCH/chain.log" 2>&1 &
  echo "[launch] chain started"
fi
