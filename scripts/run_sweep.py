#!/usr/bin/env python3
"""Main perplexity sweep: per-position losses over long code windows.

For every (language, ordering, window) job: assemble the window from the
corpus, run ppl-dump, checkpoint the gzipped TSV, and append a manifest row.
Fully resumable — completed jobs are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leanscale.corpus import corpus_stats, scan_language
from leanscale.runner import DumperConfig, append_manifest, read_manifest, run_full_window
from leanscale.windows import build_windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--languages", default="lean,python,haskell,rust,c,javascript")
    ap.add_argument("--sorted-windows", type=int, default=8)
    ap.add_argument("--shuffled-windows", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--budget", type=int, default=90_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    cfg = DumperConfig(
        binary=args.binary, model=args.model, n_ctx=args.ctx, n_threads=args.threads
    )
    manifest_path = os.path.join(args.results, "manifest.jsonl")
    done = {r["window_id"] for r in read_manifest(manifest_path)}
    langs = args.languages.split(",")

    jobs = []
    for lang in langs:
        records = scan_language(args.corpora, lang)
        stats = corpus_stats(records)
        print(f"[{lang}] {stats['n_files']} files, {stats['total_bytes']/1e6:.1f} MB", flush=True)
        for ordering, n in (("sorted", args.sorted_windows), ("shuffled", args.shuffled_windows)):
            windows = build_windows(records, lang, ordering, n, args.budget, seed=args.seed)
            jobs.extend(windows)

    # Interleave languages so partial results cover all languages evenly.
    jobs.sort(key=lambda w: (w.index, w.ordering, w.language))

    total = len(jobs)
    for k, w in enumerate(jobs):
        if w.window_id in done:
            print(f"[{k+1}/{total}] {w.window_id} cached", flush=True)
            continue
        out_path = os.path.join(args.results, "dumps", w.window_id + ".tsv.gz")
        t0 = time.time()
        info = run_full_window(cfg, w.text, out_path, args.scratch)
        append_manifest(
            manifest_path,
            {
                "window_id": w.window_id,
                "language": w.language,
                "ordering": w.ordering,
                "index": w.index,
                "n_files": len(w.files),
                "files": [f.display_path for f in w.files],
                "text_bytes": w.n_bytes,
                "ctx": args.ctx,
                "seed": args.seed,
                **info,
            },
        )
        print(f"[{k+1}/{total}] {w.window_id} done in {time.time()-t0:.0f}s", flush=True)
    print("SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
