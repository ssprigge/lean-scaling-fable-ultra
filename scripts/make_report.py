#!/usr/bin/env python3
"""Assemble REPORT.md from results/analysis CSVs and figures.

Numbers are read from the analysis artifacts so the report always matches
the committed data; prose sections live in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def fmt_bytes(b: float | None) -> str:
    if b is None or not np.isfinite(b):
        return "—"
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b/div:.1f} {unit}"
    return f"{b:.0f} B"


def fits_table(fits: pd.DataFrame, ordering: str) -> str:
    rows = fits[(fits.ordering == ordering) & (fits.model == "floor")].sort_values("alpha", ascending=False)
    lines = [
        "| language | α (exponent) | 95% CI | L∞ (floor, b/B) | 95% CI | A (amplitude) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in rows.iterrows():
        a_ci = f"[{r.alpha_lo:.3f}, {r.alpha_hi:.3f}]" if "alpha_lo" in r and np.isfinite(r.get("alpha_lo", np.nan)) else "—"
        l_ci = f"[{r.L_inf_lo:.3f}, {r.L_inf_hi:.3f}]" if "L_inf_lo" in r and np.isfinite(r.get("L_inf_lo", np.nan)) else "—"
        lines.append(
            f"| {r.language} | **{r.alpha:.3f}** | {a_ci} | {r.L_inf:.3f} | {l_ci} | {r.A:.3f} |"
        )
    return "\n".join(lines)


def crossover_section(x: pd.DataFrame, ordering: str) -> str:
    sub = x[x.ordering == ordering]
    lines = []
    for _, r in sub.iterrows():
        if pd.isna(r.crossover_bytes):
            lines.append(
                f"- **{r.lang_a} vs {r.lang_b}**: no crossover — `{r.lower_at_small}` stays "
                f"more predictable over the whole extrapolated range."
            )
        else:
            ci = ""
            if "crossover_lo" in r and np.isfinite(r.get("crossover_lo", np.nan)):
                ci = f" (bootstrap 95% CI {fmt_bytes(r.crossover_lo)} – {fmt_bytes(r.crossover_hi)}; " \
                     f"{100*r.get('boot_frac_crossing', np.nan):.0f}% of replicates cross)"
            lines.append(
                f"- **{r.lang_a} vs {r.lang_b}**: `{r.lower_at_small}` is more predictable in small "
                f"codebases, `{r.lower_at_large}` wins beyond ≈ **{fmt_bytes(r.crossover_bytes)}** of context{ci}."
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--out", default="REPORT.md")
    args = ap.parse_args()
    A = args.analysis

    fits = pd.read_csv(os.path.join(A, "fits.csv"))
    curves = pd.read_csv(os.path.join(A, "curves.csv"))
    xovers = pd.read_csv(os.path.join(A, "crossovers.csv"))
    summary = json.load(open(os.path.join(A, "summary.json")))

    bpt = (
        curves.groupby("language")
        .apply(lambda g: g.n_bytes.sum(), include_groups=False)
        .to_dict()
    )

    # Aggregate BPB at smallest/largest common depths for the flavor table.
    def bpb_at(lang, ordering, lo=True):
        g = curves[(curves.language == lang) & (curves.ordering == ordering)].dropna(subset=["bpb"])
        if g.empty:
            return np.nan
        g = g.sort_values("ctx_bytes")
        return g.bpb.iloc[0] if lo else g.bpb.iloc[-1]

    langs = sorted(curves.language.unique())

    sections: list[str] = []
    sections.append(HEADER_MD)
    sections.append("## Fitted scaling laws (sorted file order, primary)\n")
    sections.append("L(c) = L∞ + A·c^(−α), c in bytes of context, weighted LSQ, bootstrap-over-windows CIs.\n")
    sections.append(fits_table(fits, "sorted"))
    sections.append("\n### Shuffled order (locality destroyed)\n")
    sections.append(fits_table(fits, "shuffled"))
    sections.append("\n## Crossovers (sorted order)\n")
    sections.append(crossover_section(xovers, "sorted"))

    start_end = [
        f"| {l} | {bpb_at(l,'sorted',True):.2f} | {bpb_at(l,'sorted',False):.2f} | "
        f"{bpb_at(l,'shuffled',False):.2f} |"
        for l in langs
    ]
    sections.append(
        "\n## Raw curve endpoints\n\n"
        "| language | BPB @ ~32 B ctx | BPB @ max ctx (sorted) | BPB @ max ctx (shuffled) |\n|---|---|---|---|\n"
        + "\n".join(start_end)
    )

    for extra, title in (
        ("anomaly.csv", "## Bug-injection surprise (Δbits, buggy − original line)"),
        ("ablation.csv", "## Type-signature ablation"),
        ("class_breakdown.csv", "## Byte-class breakdown"),
    ):
        p = os.path.join(A, extra)
        if os.path.exists(p):
            df = pd.read_csv(p)
            if extra == "anomaly.csv":
                agg = df.groupby(["language", "depth_bytes"]).delta_bits.agg(["mean", "count"]).reset_index()
                tbl = ["| language | depth (bytes) | mean Δbits | n |", "|---|---|---|---|"] + [
                    f"| {r.language} | {int(r.depth_bytes)} | {r['mean']:.2f} | {int(r['count'])} |"
                    for _, r in agg.iterrows()
                ]
                sections.append(title + "\n\n" + "\n".join(tbl))
            elif extra == "ablation.csv":
                tbl = ["| language | annotated BPB | stripped BPB | Δ |", "|---|---|---|---|"] + [
                    f"| {r.language} | {r.annotated_bpb:.4f} | {r.stripped_bpb:.4f} | {r.stripped_bpb - r.annotated_bpb:+.4f} |"
                    for _, r in df.iterrows()
                ]
                sections.append(title + "\n\n" + "\n".join(tbl))
            else:
                agg = df.groupby(["language", "class"]).agg(bpb=("bits", "sum")).reset_index()
                byt = df.groupby(["language", "class"]).agg(b=("bytes", "sum")).reset_index()
                agg["bpb"] = agg.bpb / byt.b.values
                agg["frac"] = byt.b.values
                piv = agg.pivot(index="language", columns="class", values="bpb")
                tbl = ["| language | " + " | ".join(f"{c} BPB" for c in piv.columns) + " |",
                       "|" + "---|" * (len(piv.columns) + 1)] + [
                    f"| {lang} | " + " | ".join(f"{piv.loc[lang, c]:.2f}" for c in piv.columns) + " |"
                    for lang in piv.index
                ]
                sections.append(title + "\n\n" + "\n".join(tbl))

    for qc in ("python", "lean"):
        p = os.path.join(A, f"quant_check_{qc}.json")
        if os.path.exists(p):
            q = json.load(open(p))
            sections.append(
                f"\n**Quantization sanity ({qc})**: mean |Δbits/token| Q8_0 vs fp32 = "
                f"{q['mean_abs_delta_bits']:.4f} (p95 {q['p95_abs_delta_bits']:.4f}), corr {q['corr']:.5f}, "
                f"{q['token_id_mismatches']} token mismatches over {q['n_tokens_compared']} tokens."
            )

    sections.append(FOOTER_MD.format(n_windows=summary.get("n_windows", "?")))
    with open(args.out, "w") as f:
        f.write("\n\n".join(sections) + "\n")
    print(f"wrote {args.out}")


HEADER_MD = """# Lean Software Scaling Laws — measurement results

*(Implementation of the measurement design in Gwern's
[lean-scaling essay](https://gwern.net/lean-scaling); see README.md for
method details and reproduction instructions. Placeholder — regenerated
with final prose by make_report.py; see REPORT.md committed in repo.)*"""

FOOTER_MD = """---
*{n_windows} windows measured. See results/figures/ for plots and
results/analysis/ for machine-readable outputs.*"""


if __name__ == "__main__":
    main()
