#!/usr/bin/env bash
# Waits for the main sweep to finish, then runs the anomaly sweep, the
# type-signature ablation, and the quantization sanity check, sequentially
# (they share the 4 CPU cores). Everything is resumable.
set -uo pipefail

SCRATCH="${1:?usage: chain_after_sweep.sh <scratch-dir> <repo-dir>}"
REPO="${2:?}"
PY="$REPO/.venv/bin/python"
BIN="$SCRATCH/ppl-dump"
MODEL="$SCRATCH/models/qwen2-0.5b-q8_0.gguf"

echo $$ > "$SCRATCH/chain.pid"
cd "$REPO"

# Liveness via the pidfile run_sweep.py writes at startup. (pgrep -f is
# unreliable: transient monitoring shells embed "run_sweep.py" in their
# command lines and match.)
sweep_alive() {
  local pf="$SCRATCH/work/run_sweep.pid" pid
  [ -f "$pf" ] || return 1
  pid=$(cat "$pf" 2>/dev/null) || return 1
  [ -n "$pid" ] && [ -d "/proc/$pid" ] && \
    grep -qa "run_sweep" "/proc/$pid/cmdline" 2>/dev/null
}

echo "[chain] waiting for main sweep to complete..."
restarts=0
while ! grep -q "SWEEP_COMPLETE" "$SCRATCH/sweep.log" 2>/dev/null; do
  # Sleep first: a just-launched sweep gets a full minute to write its
  # pidfile before the first liveness check.
  sleep 60
  grep -q "SWEEP_COMPLETE" "$SCRATCH/sweep.log" 2>/dev/null && break
  if ! sweep_alive; then
    if [ "$restarts" -ge 5 ]; then
      echo "[chain] sweep died $restarts times; giving up"
      echo "SWEEP_FAILED"
      exit 1
    fi
    restarts=$((restarts + 1))
    echo "[chain] sweep process died without completing; restarting it ($restarts/5)"
    setsid nohup "$PY" scripts/run_sweep.py \
      --corpora "$SCRATCH/corpora" --results results/main --scratch "$SCRATCH/work" \
      --binary "$BIN" --model "$MODEL" \
      --languages lean,python,haskell,rust,c,javascript \
      --sorted-windows 8 --shuffled-windows 4 --ctx 16384 --budget 90000 --threads 4 \
      >> "$SCRATCH/sweep.log" 2>&1
  fi
done
echo "[chain] sweep complete; starting anomaly sweep"

"$PY" scripts/run_anomaly.py \
  --corpora "$SCRATCH/corpora" --results results/main --scratch "$SCRATCH/work" \
  --binary "$BIN" --model "$MODEL" \
  --languages lean,python,haskell,rust,c,javascript \
  --windows 6 --depths 2000,8000,24000,45000 --ctx 16384 --budget 90000 --threads 4 \
  > "$SCRATCH/anomaly.log" 2>&1
echo "[chain] anomaly done (rc=$?)"

"$PY" scripts/run_ablation.py \
  --corpora "$SCRATCH/corpora" --results results/main --scratch "$SCRATCH/work" \
  --binary "$BIN" --model "$MODEL" \
  --languages python,haskell --windows 8 --ctx 16384 --budget 90000 --threads 4 \
  > "$SCRATCH/ablation.log" 2>&1
echo "[chain] ablation done (rc=$?)"

# Quantization sanity check on one python + one lean window (first 2048 tokens).
mkdir -p results/analysis
for lang in python lean; do
  wid="${lang}-sorted-w0"
  txt="$SCRATCH/work/quantcheck_${wid}.txt"
  "$PY" - "$SCRATCH/corpora" "$lang" "$txt" <<'EOF'
import sys
sys.path.insert(0, ".")
from leanscale.corpus import scan_language
from leanscale.windows import build_windows
corpora, lang, out = sys.argv[1], sys.argv[2], sys.argv[3]
recs = scan_language(corpora, lang)
w = build_windows(recs, lang, "sorted", 1, 90000, seed=0)[0]
open(out, "w", encoding="utf-8").write(w.text)
EOF
  "$PY" scripts/run_quant_check.py \
    --model-dir "$SCRATCH/models/qwen2-0.5b" \
    --window-text "$txt" \
    --q8-dump "results/main/dumps/${wid}.tsv.gz" \
    --n-tokens 2048 \
    --out "results/analysis/quant_check_${lang}.json" \
    > "$SCRATCH/quantcheck_${lang}.log" 2>&1
  echo "[chain] quant check $lang done (rc=$?)"
done

echo "CHAIN_COMPLETE"
