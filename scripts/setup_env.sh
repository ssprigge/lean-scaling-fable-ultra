#!/usr/bin/env bash
# One-command environment rebuild for the leanscale measurement pipeline.
# Everything outside the git repo (venv, corpora, llama.cpp, model weights)
# is reconstructed here, pinned, so an ephemeral-container restart costs one
# command:   scripts/setup_env.sh /home/user/scratch
#
# Idempotent: every step is skipped if its artifact already exists.
set -euo pipefail

SCRATCH="${1:?usage: setup_env.sh <scratch-dir>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_TAG=b6100   # llama.cpp pin (commit 65c797c4fad4d9966695ac4b4a1560be44109267)
MODEL_S3="https://s3.amazonaws.com/jumpstart-cache-prod-us-east-1/huggingface-llm/huggingface-llm-qwen2-0-5b/artifacts/inference-prepack/v1.0.0"

mkdir -p "$SCRATCH"

# ---- venv ------------------------------------------------------------------
if [ ! -x "$REPO/.venv/bin/python" ]; then
  python3.11 -m venv "$REPO/.venv"
  "$REPO/.venv/bin/pip" install --quiet --upgrade pip
fi
"$REPO/.venv/bin/pip" install --quiet -e "$REPO" pytest
# torch/transformers are only needed for the GGUF conversion and the fp32
# quantization cross-check. (download.pytorch.org is blocked in this
# environment; the default PyPI index works.)
"$REPO/.venv/bin/python" -c "import torch, transformers, gguf" 2>/dev/null || \
  "$REPO/.venv/bin/pip" install --quiet torch transformers safetensors gguf sentencepiece

# ---- corpora (pinned public clones; SHAs recorded in manifest) --------------
if [ ! -f "$SCRATCH/corpora/manifest.tsv" ] || \
   [ "$(grep -c 'ok\|cached' "$SCRATCH/corpora/manifest.tsv")" -lt 26 ]; then
  bash "$REPO/scripts/fetch_corpora.sh" "$SCRATCH/corpora"
fi

# ---- llama.cpp @ pinned tag --------------------------------------------------
if [ ! -f "$SCRATCH/llama.cpp/build/bin/libllama.so" ]; then
  if [ ! -d "$SCRATCH/llama.cpp" ]; then
    git clone --quiet --depth 1 --branch "$LLAMA_TAG" \
      https://github.com/ggml-org/llama.cpp.git "$SCRATCH/llama.cpp"
  fi
  # Portable x86-64-v3 baseline, NOT -march=native: this environment's
  # containers can migrate between hosts with different ISA extensions
  # mid-session (observed: a GGML_NATIVE build SIGILLed after a migration).
  cmake -S "$SCRATCH/llama.cpp" -B "$SCRATCH/llama.cpp/build" \
        -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SCRATCH/llama.cpp/build" --config Release -j 3 -t llama
fi

# ---- model weights (Qwen2-0.5B base; SageMaker JumpStart public S3 mirror,
# ---- since huggingface.co is unreachable from this environment) -------------
MD="$SCRATCH/models/qwen2-0.5b"
if [ ! -f "$MD/model.safetensors" ]; then
  mkdir -p "$MD"
  for f in config.json generation_config.json merges.txt tokenizer.json \
           tokenizer_config.json vocab.json model.safetensors; do
    curl -sS --retry 3 -o "$MD/$f" "$MODEL_S3/$f"
  done
fi

# ---- GGUF Q8_0 conversion ----------------------------------------------------
GGUF="$SCRATCH/models/qwen2-0.5b-q8_0.gguf"
if [ ! -f "$GGUF" ]; then
  "$REPO/.venv/bin/python" "$SCRATCH/llama.cpp/convert_hf_to_gguf.py" \
    "$MD" --outfile "$GGUF" --outtype q8_0
fi

# ---- ppl-dump ----------------------------------------------------------------
L="$SCRATCH/llama.cpp"
g++ -O2 -std=c++17 "$REPO/tools/ppl_dump.cpp" \
    -I"$L/include" -I"$L/ggml/include" -L"$L/build/bin" \
    -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin" \
    -o "$SCRATCH/ppl-dump"

"$REPO/.venv/bin/python" -m pytest "$REPO/tests" -q
echo "SETUP_COMPLETE scratch=$SCRATCH model=$GGUF"
