#!/usr/bin/env python3
"""Type-signature ablation sweep.

Rebuilds the same sorted windows as the main sweep but with type signatures
stripped (Python annotations via AST surgery; Haskell top-level `::`
signatures), and measures per-position losses on the stripped variants.
The annotated baselines come from the main sweep, so only the stripped
windows are run here. Resumable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leanscale.ablation import strip_haskell_signatures, strip_python_annotations
from leanscale.corpus import scan_language
from leanscale.languages import LANGUAGES, header_line
from leanscale.runner import DumperConfig, append_manifest, read_manifest, run_full_window
from leanscale.windows import build_windows

STRIPPERS = {
    "python": strip_python_annotations,
    "haskell": strip_haskell_signatures,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--languages", default="python,haskell")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--budget", type=int, default=90_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    cfg = DumperConfig(binary=args.binary, model=args.model, n_ctx=args.ctx, n_threads=args.threads)
    manifest_path = os.path.join(args.results, "ablation_manifest.jsonl")
    done = {r["window_id"] for r in read_manifest(manifest_path)}

    for lang in args.languages.split(","):
        stripper = STRIPPERS[lang]
        cfg_lang = LANGUAGES[lang]
        records = scan_language(args.corpora, lang)
        windows = build_windows(records, lang, "sorted", args.windows, args.budget, seed=args.seed)
        for w in windows:
            wid = f"{lang}-stripped-w{w.index}"
            if wid in done:
                print(f"{wid} cached", flush=True)
                continue
            parts = []
            n_changed = 0
            for rec in w.files:
                with open(rec.abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                stripped = stripper(content)
                if stripped != content:
                    n_changed += 1
                if not stripped.endswith("\n"):
                    stripped += "\n"
                parts.append(header_line(cfg_lang, rec.display_path))
                parts.append(stripped)
                parts.append("\n")
            text = "".join(parts)
            out_path = os.path.join(args.results, "dumps", wid + ".tsv.gz")
            t0 = time.time()
            info = run_full_window(cfg, text, out_path, args.scratch)
            append_manifest(
                manifest_path,
                {
                    "window_id": wid,
                    "language": lang,
                    "ordering": "stripped",
                    "index": w.index,
                    "n_files": len(w.files),
                    "n_files_changed": n_changed,
                    "text_bytes": len(text.encode("utf-8")),
                    "baseline_window_id": w.window_id,
                    "ctx": args.ctx,
                    **info,
                },
            )
            print(f"{wid} done in {time.time()-t0:.0f}s ({n_changed}/{len(w.files)} files changed)", flush=True)
    print("ABLATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
