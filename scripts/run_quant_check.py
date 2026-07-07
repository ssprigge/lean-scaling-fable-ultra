#!/usr/bin/env python3
"""Quantization sanity check: Q8_0 llama.cpp vs fp32 HF transformers.

Runs the original safetensors model (float32, CPU) over prefixes of a few
sweep windows and compares per-token logprobs against the Q8_0 ppl-dump
output for the same text. Writes a JSON summary with per-token deltas.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def hf_logprobs(model_dir: str, text: str, n_tokens: int) -> tuple[list[int], list[float]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    lps: list[float] = [math.nan]
    with torch.no_grad():
        x = torch.tensor([ids])
        out = model(x)
        logits = out.logits[0]  # (T, V)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for i in range(1, len(ids)):
            lps.append(float(logprobs[i - 1, ids[i]]))
    return ids, lps


def dump_logprobs(path: str) -> tuple[list[int], list[float]]:
    opener = gzip.open if path.endswith(".gz") else open
    toks, lps = [], []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("idx\t"):
                continue
            p = line.rstrip().split("\t")
            toks.append(int(p[1]))
            lps.append(float(p[3]))
    return toks, lps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="HF safetensors dir")
    ap.add_argument("--window-text", required=True, help="window text file")
    ap.add_argument("--q8-dump", required=True, help="ppl-dump TSV(.gz) for same text")
    ap.add_argument("--n-tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.window_text, encoding="utf-8") as f:
        text = f.read()
    ids_hf, lp_hf = hf_logprobs(args.model_dir, text, args.n_tokens)
    ids_q8, lp_q8 = dump_logprobs(args.q8_dump)
    n = min(len(ids_hf), len(ids_q8), args.n_tokens)

    mismatch = sum(1 for i in range(n) if ids_hf[i] != ids_q8[i])
    a = np.array(lp_hf[1:n])
    b = np.array(lp_q8[1:n])
    d = b - a
    summary = {
        "n_tokens_compared": n,
        "token_id_mismatches": mismatch,
        "mean_bits_fp32": float((-a / math.log(2)).mean()),
        "mean_bits_q8": float((-b / math.log(2)).mean()),
        "mean_abs_delta_bits": float(np.abs(d / math.log(2)).mean()),
        "p95_abs_delta_bits": float(np.percentile(np.abs(d / math.log(2)), 95)),
        "corr": float(np.corrcoef(a, b)[0, 1]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
