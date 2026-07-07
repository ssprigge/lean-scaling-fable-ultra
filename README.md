# leanscale — Lean Software Scaling Laws, measured

An implementation of the measurement design in Gwern's
[*Lean Software Scaling Laws*](https://gwern.net/lean-scaling)
(`lean-scaling-essay.md` in this repo): estimate how the *predictability* of
source code — a frozen coding LLM's loss, in bits per byte — scales with the
amount of codebase context already read, per programming language; fit
power-law scaling curves; and extrapolate to find *crossovers* (at what
codebase size would a formally-disciplined language like Lean become more
absolutely predictable than a permissive one like Python?).

## Method

1. **Corpus construction** — pinned shallow clones of well-known open-source
   codebases per language (Lean 4: mathlib4, batteries, lean4 core; Python:
   sympy, django, sqlalchemy, flask; Haskell: Agda, pandoc, lens, aeson;
   Rust: nalgebra, tokio, serde, clap; C: zlib, redis, curl, sqlite;
   JavaScript: eslint, webpack, jquery, moment, express, axios, fastify).
   Files are filtered (size bounds, UTF-8, no vendored/generated/minified
   code, no test suites — uniformly across languages), content-deduplicated,
   and concatenated into ~16k-token *windows* with native-comment file
   headers, in two orders: **sorted** (path order ≈ module locality) and
   **shuffled** (locality destroyed).
2. **Loss measurement** — a frozen base LLM (Qwen2-0.5B, Q8_0, via a custom
   `llama.cpp`-based dumper, `tools/ppl_dump.cpp`) scores every token of every
   window; losses are converted to **bits and attributed to the byte offset of
   context already read**, normalizing away tokenizer verbosity differences
   between languages on both axes.
3. **Position averaging** — token losses are pooled into log-spaced
   context-depth bins across windows: bits-per-byte vs. bytes-of-context
   curves per (language, ordering).
4. **Curve fitting** — weighted least squares of
   `L(c) = L_inf + A·c^(−α)` (and the pure power law) per language, with
   window-bootstrap confidence intervals. `α` is the *predictability scaling
   exponent*; `L_inf` the extrapolated floor.
5. **Extrapolation** — solve `L_i(c) = L_j(c)` for language pairs to locate
   crossovers, with bootstrap uncertainty.

Cross-checks implemented (essay §Measurement Design):

* **Anomaly/bug detection** — inject stylistically plausible bugs
  (comparison swaps, sign flips, off-by-one constants, identifier swaps) at
  several context depths; measure extra bits to encode the buggy line vs. the
  original line under the identical prefix (KV-cache prefix reuse). A model
  that understands the codebase should grow *more* surprised at bugs with
  more context.
* **Type-signature ablation** — strip Python annotations (AST-guided surgery)
  and Haskell top-level `::` signatures from the same windows; compare curves.
* **Quantization sanity** — fp32 HF-transformers logprobs vs. the Q8_0
  llama.cpp path on shared windows.

## Layout

```
leanscale/          measurement + analysis library (corpus, windows,
                    aggregation, fitting, crossover, anomaly, ablation, plots)
tools/ppl_dump.cpp  per-token logprob dumper (llama.cpp); modes: full, variants
scripts/            fetch_corpora.sh, run_sweep.py, run_anomaly.py,
                    run_ablation.py, run_quant_check.py, fit_and_report.py
tests/              unit tests (pytest)
results/            committed measurement outputs: manifests, gzipped
                    per-token dumps, analysis CSVs, figures
REPORT.md           results write-up
```

## Reproducing

```bash
# one-command environment rebuild: venv, corpora (pinned SHAs -> results/
# corpora_manifest.tsv), llama.cpp @ b6100, Qwen2-0.5B base weights (public
# SageMaker JumpStart S3 mirror; huggingface.co is unreachable in the build
# environment), GGUF Q8_0 conversion, ppl-dump build, unit tests:
scripts/setup_env.sh /home/user/scratch

.venv/bin/python scripts/run_sweep.py    --corpora ... --results results/main ...
.venv/bin/python scripts/run_anomaly.py  --corpora ... --results results/main ...
.venv/bin/python scripts/run_ablation.py --corpora ... --results results/main ...
.venv/bin/python scripts/fit_and_report.py
```

All runs are resumable (completed windows are skipped via manifests).

## Caveats

This is the essay's *cheap* arm (frozen model, in-context scaling), run at
small scale: a 0.5B-parameter model, 16k-token windows, on CPU. Absolute
numbers will differ for frontier models; the pipeline is model-agnostic —
point `--model` at any GGUF. See REPORT.md §Limitations for the full list
(domain confounds, ecosystem-maturity effects, single model, etc.).
