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

cd "$REPO"

echo "[chain] waiting for main sweep to complete..."
while ! grep -q "SWEEP_COMPLETE" "$SCRATCH/sweep.log" 2>/dev/null; do
  if ! pgrep -f "run_sweep.py" > /dev/null 2>&1; then
    echo "[chain] sweep process died without completing; restarting it"
    setsid nohup "$PY" scripts/run_sweep.py \
      --corpora "$SCRATCH/corpora" --results results/main --scratch "$SCRATCH/work" \
      --binary "$BIN" --model "$MODEL" \
      --languages lean,python,haskell,rust,c,javascript \
      --sorted-windows 8 --shuffled-windows 4 --ctx 16384 --budget 90000 --threads 4 \
      >> "$SCRATCH/sweep.log" 2>&1
    sleep 60
  fi
  sleep 60
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
