#!/usr/bin/env python3
"""Bug-injection anomaly sweep.

For each language, take the same sorted windows as the main sweep; at several
byte depths per window, mutate one code line (comparison swap, sign flip,
off-by-one, identifier swap) and score original + mutants against the
identical prefix via ppl-dump variants mode (incremental prefix, KV rollback).
Resumable per (language, window).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leanscale.anomaly import build_variants_job
from leanscale.corpus import scan_language
from leanscale.runner import DumperConfig, append_manifest, read_manifest, run_variants_job
from leanscale.windows import build_windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--languages", default="lean,python,haskell,rust,c,javascript")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--depths", default="2000,8000,24000,45000")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--budget", type=int, default=90_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    cfg = DumperConfig(binary=args.binary, model=args.model, n_ctx=args.ctx, n_threads=args.threads)
    depths = [int(d) for d in args.depths.split(",")]
    manifest_path = os.path.join(args.results, "anomaly_manifest.jsonl")
    done = {r["job_id"] for r in read_manifest(manifest_path)}

    for lang in args.languages.split(","):
        records = scan_language(args.corpora, lang)
        windows = build_windows(
            records, lang, "sorted", args.windows, args.budget, seed=args.seed, allow_fewer=True
        )
        if len(windows) < args.windows:
            print(
                f"[{lang}] WARNING: corpus supports only {len(windows)}/{args.windows} "
                f"sorted windows; injecting into what exists",
                flush=True,
            )
        for w in windows:
            job_id = f"anomaly-{w.window_id}"
            if job_id in done:
                print(f"{job_id} cached", flush=True)
                continue
            built = build_variants_job(
                w.text, lang, depths, os.path.join(args.scratch, "anomaly"), job_id
            )
            if built is None:
                print(f"{job_id} no usable sites", flush=True)
                continue
            job_path, meta = built
            out_path = os.path.join(args.results, "anomaly", job_id + ".tsv")
            t0 = time.time()
            info = run_variants_job(cfg, job_path, out_path)
            append_manifest(
                manifest_path,
                {
                    "job_id": job_id,
                    "language": lang,
                    "window_id": w.window_id,
                    "depths": depths,
                    "meta": meta,
                    **info,
                },
            )
            print(f"{job_id} done in {time.time()-t0:.0f}s ({len(meta)} spans)", flush=True)
    print("ANOMALY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
