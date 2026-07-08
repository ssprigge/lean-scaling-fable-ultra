# HANDOFF — state as of 2026-07-08 ~00:45 UTC

Implementation of the measurement design in `lean-scaling-essay.md` (Gwern's
*Lean Software Scaling Laws*). Pipeline code is complete and running; review
findings are triaged but mostly **not yet fixed**; the measurement sweep is
mid-flight.

## What is running right now (detached daemons, survive session turns)

- **Main sweep** — `scripts/run_sweep.py`: 70 windows (6 languages ×
  {8 sorted + 4 shuffled}, JS capped at 6 sorted by corpus size), 16384-token
  context, Qwen2-0.5B base Q8_0 via `/home/user/scratch/ppl-dump`.
  ~19/70 done, ~260 s/window ⇒ finishes ~4 h from timestamp above.
  Log: `/home/user/scratch/sweep.log`; pidfile `/home/user/scratch/work/run_sweep.pid`.
- **Chain babysitter** — `scripts/chain_after_sweep.sh` (pidfile
  `/home/user/scratch/chain.pid`): restarts a dead sweep (cap 5), then runs
  `run_anomaly.py` → `run_ablation.py` → quant-check, then prints
  CHAIN_COMPLETE. Logs: `/home/user/scratch/{chain,anomaly,ablation}.log`.
- Results checkpoint continuously into `results/main/` (gzipped per-token
  dumps + `manifest.jsonl`); every window is resumable/skippable.

**After everything completes**: `scripts/fit_and_report.py` then
`scripts/make_report.py` produce `results/analysis/` + REPORT.md. Neither has
been run on real data yet — and both have known findings to fix first (below).

## IMPORTANT — fix before the chain reaches anomaly/ablation (~4 h)

`results/review/findings.json` holds 18 confirmed findings (adversarially
verified, majority-vote) + 4 rejected, from a 71-agent review. Full verdict
text includes suggested fixes. Priority order:

1. **anomaly.py:78** [major] — id_swap mutants are syntax garbage: `_ID_RE`
   apostrophe absorbs string-closing quotes (Python/JS), KEYWORDS misses
   raise/elif/except/assert/function/throw/…. Make apostrophe
   language-conditional (haskell/lean only), skip identifiers inside/adjacent
   to string literals, expand KEYWORDS per language.
2. **anomaly.py:118** [major] — `is_code_line` only rejects lines *starting*
   with comment markers ⇒ mutations land inside docstrings/block comments and
   trailing comments. Track block-comment state; skip non-code mutation sites.
3. **anomaly.py meta + anomaly_analysis.py:90 + make_report.py:122** [major]
   — record the **nominal** depth for each site (meta) and aggregate by it;
   current per-site exact byte offsets make every depth group a singleton, so
   the surprise-vs-depth statistic never pools.
4. **anomaly_analysis.py:94** [minor] — record orig/mutant span byte lengths;
   report length-adjusted delta (mutants can change span length, adding a
   mechanical ±BPB·Δbytes term).
5. **runner.py:91** [minor] — `read_manifest` dies on a torn trailing line
   (crash mid-append) ⇒ resume permanently blocked. Skip+warn on bad lines.
6. **ppl_dump.cpp:104** [minor] — variants-mode PREFIX decodes request logits
   for all tokens; only last is consumed (~25% wasted decode in anomaly).
   Set batch logits flags accordingly; **recompile** `/home/user/scratch/ppl-dump`.
7. **ablation.py:41/42/119** — Python stripper can emit invalid code
   (paren-wrapped annotations; UTF-8 col_offset vs str indices) and skips bare
   `x: int` AnnAssign (50/80 remain); Haskell regex misses operator sigs and
   name-on-own-line style (56% of `::` lines retained). Fix + re-parse check
   (ast.parse) before use; fall back to original file on failure.

Analysis-side (before trusting REPORT.md): aggregate.py:111 terminal-bin
distortion; crossover.py:56 multi-root handling; semantics.py:39 nested block
comments (Lean/Haskell nest!); semantics.py:73 newline-run token class
attribution; make_report.py:47 crossover uncertainty; fit_and_report.py:156/
178/79 silent drops + 1e12-boundary crossings reported as genuine.

A resumed workflow re-running the crashed `corpus-windows` finder (the one
dimension with zero findings — server error) may still be in flight:
`w4t9oy75r` / run `wf_4dffd72c-8a4`. Process its result when it lands —
corpus/window bugs would invalidate measured windows, so check first.

## Environment (rebuilt 3× — container restarts + CPU migration are real)

- One command rebuilds everything: `scripts/setup_env.sh /home/user/scratch`
  (venv, pinned corpora clones, llama.cpp @ b6100 **portable x86-64-v3 build**
  — never `-march=native`, a host migration SIGILLed the native build —
  Qwen2-0.5B from the SageMaker S3 mirror, GGUF Q8_0 conversion, ppl-dump,
  tests). Then `scripts/launch_runs.sh /home/user/scratch` (idempotent,
  pidfile-guarded) relaunches daemons.
- huggingface.co / zlib.net / ftp.gnu.org / modelscope / ollama registry are
  **blocked** by the network policy; PyPI, Hackage, crates.io, npm, S3 work.
  Public GitHub clones work via the session git proxy.
- `pgrep/pkill -f <name>` is a footgun here: wrapper shells embed command
  strings (a pkill once killed its own shell). Use the pidfiles.
- Corpus pins live in `results/corpora_manifest.tsv`; `fetch_corpora.sh`
  honors them (re-pinning verified: both pre-restart windows rebuild
  bit-identically).
- `/home/user` (repo + scratch) has survived all restarts so far; `/tmp` and
  task outputs sometimes do not. Copy anything precious into the repo or
  `/home/user/scratch` immediately.

## Method summary (for whoever reads results)

Frozen-LLM in-context scaling: per-token logprobs → bits per byte, attributed
to bytes-of-context-read, log-binned, averaged across windows per (language,
ordering); fit `L(c) = L_inf + A·c^(−α)` (window-bootstrap CIs); solve
pairwise crossovers; cross-checks = bug-injection surprise vs depth,
type-signature ablation, Q8-vs-fp32 sanity, semantics byte-class breakdown.
Deliberate deviations from the essay: 0.5B model (GLM-5.2-class infeasible
here), 16 k-token windows, 6 languages, topic-matched repo choices (sympy vs
mathlib, agda vs lean4) as the bias control.

## Verification status

- 34 unit tests green (`.venv/bin/python -m pytest tests/ -q`).
- Dumper smoke-tested end-to-end on this host (per-token TSV sane).
- 19 measured windows committed under `results/main/`.
