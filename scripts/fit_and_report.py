#!/usr/bin/env python3
"""Aggregate sweep dumps, fit scaling laws, locate crossovers, emit figures
and machine-readable results (results/analysis/*.json|csv + figures/*.png).

Can be run at any time during the sweep; it analyzes whatever windows have
completed. The written artifacts are consumed by REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leanscale.aggregate import (
    curve_from_windows,
    make_bins,
    per_window_curves,
    read_dump,
)
from leanscale.anomaly_analysis import collect_anomaly_results
from leanscale.crossover import bootstrap_crossover, find_crossover
from leanscale.fitting import bootstrap_fits, fit_curve, percentile_ci
from leanscale.plots import plot_ablation, plot_anomaly, plot_curves, plot_extrapolation
from leanscale.runner import read_manifest


def load_windows(results_dir: str, manifest: list[dict]):
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for row in manifest:
        path = os.path.join(results_dir, "dumps", row["window_id"] + ".tsv.gz")
        if not os.path.exists(path):
            continue
        w = read_dump(path, row["window_id"], row["language"], row["ordering"])
        groups[(row["language"], row["ordering"])].append(w)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--bin-lo", type=float, default=24.0)
    ap.add_argument("--bin-hi", type=float, default=131072.0)
    ap.add_argument("--n-bins", type=int, default=26)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    edges = make_bins(args.bin_lo, args.bin_hi, args.n_bins)

    manifest = read_manifest(os.path.join(args.results, "manifest.jsonl"))
    groups = load_windows(args.results, manifest)
    print(f"loaded {sum(len(v) for v in groups.values())} windows in {len(groups)} groups")

    # ---- curves + fits per (language, ordering) ----
    all_rows = []
    fits_floor: dict[str, dict] = defaultdict(dict)
    fits_pure: dict[str, dict] = defaultdict(dict)
    boots: dict[tuple[str, str], dict] = {}
    curves_by_ordering: dict[str, dict] = defaultdict(dict)
    c_data_max = 0.0

    for (lang, ordering), windows in sorted(groups.items()):
        curve = curve_from_windows(windows, edges)
        curves_by_ordering[ordering][lang] = (curve.centers, curve.bpb)
        m = np.isfinite(curve.bpb)
        if m.any():
            c_data_max = max(c_data_max, float(curve.centers[m].max()))
        for c, y, nb in zip(curve.centers, curve.bpb, curve.n_bytes):
            all_rows.append(
                {"language": lang, "ordering": ordering, "ctx_bytes": float(c),
                 "bpb": float(y), "n_bytes": float(nb), "n_windows": curve.n_windows}
            )
        ffit = fit_curve(curve.centers, curve.bpb, curve.n_bytes, model="floor")
        pfit = fit_curve(curve.centers, curve.bpb, curve.n_bytes, model="pure")
        fits_floor[ordering][lang] = ffit
        fits_pure[ordering][lang] = pfit
        B, N = per_window_curves(windows, edges)
        boots[(lang, ordering)] = bootstrap_fits(B, N, curve.centers, model="floor",
                                                 n_boot=args.n_boot, seed=7)

    pd.DataFrame(all_rows).to_csv(os.path.join(args.out, "curves.csv"), index=False)

    fit_rows = []
    for ordering in fits_floor:
        for lang, fit in fits_floor[ordering].items():
            if fit is None:
                continue
            bs = boots.get((lang, ordering), {})
            row = {"language": lang, "ordering": ordering, "model": "floor",
                   **fit.params, "aicc": fit.aicc, "n_points": fit.n_points}
            for p in ("L_inf", "A", "alpha"):
                if p in bs and len(bs[p]):
                    lo, hi = percentile_ci(bs[p])
                    row[f"{p}_lo"], row[f"{p}_hi"] = lo, hi
            fit_rows.append(row)
            pf = fits_pure[ordering].get(lang)
            if pf is not None:
                fit_rows.append({"language": lang, "ordering": ordering, "model": "pure",
                                 **pf.params, "aicc": pf.aicc, "n_points": pf.n_points})
    pd.DataFrame(fit_rows).to_csv(os.path.join(args.out, "fits.csv"), index=False)

    # ---- crossovers (floor model, per ordering) ----
    xrows = []
    for ordering, fits in fits_floor.items():
        langs = [l for l, f in fits.items() if f is not None]
        for i, a in enumerate(langs):
            for b in langs[i + 1:]:
                x = find_crossover(fits[a], fits[b], a, b, c_min=128.0)
                row = {"ordering": ordering, "lang_a": a, "lang_b": b,
                       "lower_at_small": x.lang_lo_small, "lower_at_large": x.lang_lo_large,
                       "crossover_bytes": x.c_cross}
                ba, bb = boots.get((a, ordering), {}), boots.get((b, ordering), {})
                if ba and bb and all(k in ba for k in ("L_inf", "A", "alpha")) and all(k in bb for k in ("L_inf", "A", "alpha")):
                    xs = bootstrap_crossover(ba, bb, "floor")
                    frac = float(np.mean(np.isfinite(xs)))
                    row["boot_frac_crossing"] = frac
                    if np.isfinite(xs).sum() >= 10:
                        lo, hi = np.nanpercentile(xs, [2.5, 97.5])
                        row["crossover_lo"], row["crossover_hi"] = float(lo), float(hi)
                xrows.append(row)
    pd.DataFrame(xrows).to_csv(os.path.join(args.out, "crossovers.csv"), index=False)

    # ---- figures ----
    for ordering, curves in curves_by_ordering.items():
        plot_curves(curves, fits_floor[ordering],
                    f"Bits/byte vs. context — {ordering} file order (Qwen2-0.5B, Q8_0)",
                    os.path.join(args.figures, f"curves_{ordering}.png"))
        plot_extrapolation({k: v for k, v in fits_floor[ordering].items()},
                           c_data_max,
                           os.path.join(args.figures, f"extrapolation_{ordering}.png"),
                           title=f"Extrapolated scaling laws — {ordering} order")

    # ---- anomaly results (if present) ----
    amanifest = read_manifest(os.path.join(args.results, "anomaly_manifest.jsonl"))
    if amanifest:
        adf = collect_anomaly_results(amanifest, args.results)
        if not adf.empty:
            adf.to_csv(os.path.join(args.out, "anomaly.csv"), index=False)
            plot_anomaly(adf, os.path.join(args.figures, "anomaly.png"))

    # ---- ablation results (if present) ----
    abmanifest = read_manifest(os.path.join(args.results, "ablation_manifest.jsonl"))
    if abmanifest:
        pairs: dict[str, dict] = defaultdict(dict)
        strows = []
        for lang in {r["language"] for r in abmanifest}:
            stripped = [read_dump(os.path.join(args.results, "dumps", r["window_id"] + ".tsv.gz"),
                                  r["window_id"], lang, "stripped")
                        for r in abmanifest if r["language"] == lang
                        and os.path.exists(os.path.join(args.results, "dumps", r["window_id"] + ".tsv.gz"))]
            baseline = groups.get((lang, "sorted"), [])
            base_ids = {r["baseline_window_id"] for r in abmanifest if r["language"] == lang}
            baseline = [w for w in baseline if w.window_id in base_ids]
            if not stripped or not baseline:
                continue
            sc = curve_from_windows(stripped, edges)
            bc = curve_from_windows(baseline, edges)
            pairs[lang]["stripped"] = (sc.centers, sc.bpb)
            pairs[lang]["annotated"] = (bc.centers, bc.bpb)
            tb_s = sum(float(np.nansum(w.bits)) for w in stripped)
            by_s = sum(w.total_bytes for w in stripped)
            tb_b = sum(float(np.nansum(w.bits)) for w in baseline)
            by_b = sum(w.total_bytes for w in baseline)
            strows.append({"language": lang,
                           "annotated_bits": tb_b, "annotated_bytes": by_b,
                           "stripped_bits": tb_s, "stripped_bytes": by_s,
                           "annotated_bpb": tb_b / by_b, "stripped_bpb": tb_s / by_s,
                           "annotated_total_kbits": tb_b / 1e3, "stripped_total_kbits": tb_s / 1e3})
        if pairs:
            plot_ablation(pairs, os.path.join(args.figures, "ablation.png"))
            pd.DataFrame(strows).to_csv(os.path.join(args.out, "ablation.csv"), index=False)

    # ---- headline json ----
    summary = {
        "n_windows": sum(len(v) for v in groups.values()),
        "groups": {f"{l}/{o}": len(v) for (l, o), v in groups.items()},
        "c_data_max_bytes": c_data_max,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
