"""Figures for the scaling-law report (matplotlib, no external styles)."""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colorblind-safe categorical palette (Okabe–Ito), fixed per language.
LANG_COLORS = {
    "lean": "#0072B2",
    "python": "#E69F00",
    "haskell": "#009E73",
    "rust": "#D55E00",
    "c": "#56B4E9",
    "javascript": "#CC79A7",
}


def _lang_color(lang: str) -> str:
    return LANG_COLORS.get(lang, "#555555")


def plot_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], fits: dict[str, object] | None,
                title: str, out_path: str, xlabel: str = "context (bytes of code read)") -> None:
    """curves: language -> (centers, bpb). fits: language -> Fit (optional)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    for lang, (c, y) in sorted(curves.items()):
        m = np.isfinite(y)
        ax.plot(c[m], y[m], "o", ms=4, color=_lang_color(lang), label=lang)
        if fits and lang in fits and fits[lang] is not None:
            cs = np.geomspace(c[m].min(), c[m].max(), 200)
            ax.plot(cs, fits[lang].predict(cs), "-", lw=1.4, color=_lang_color(lang), alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss (bits per byte)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_extrapolation(fits: dict[str, object], c_data_max: float, out_path: str,
                       c_max: float = 1e10, title: str = "Extrapolated scaling laws") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    cs = np.geomspace(64, c_max, 400)
    for lang, fit in sorted(fits.items()):
        if fit is None:
            continue
        ax.plot(cs, fit.predict(cs), "-", lw=1.6, color=_lang_color(lang), label=lang)
    ax.axvline(c_data_max, color="k", ls=":", lw=1)
    ax.text(c_data_max * 1.3, ax.get_ylim()[1] * 0.8, "measured range ends", rotation=90,
            va="top", fontsize=8, color="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("context (bytes of code)")
    ax.set_ylabel("fitted loss (bits per byte)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_anomaly(df, out_path: str, title: str = "Bug-injection surprise vs. context depth") -> None:
    """df: tidy frame with language, depth_bytes, delta_bits."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    for lang, grp in df.groupby("language"):
        agg = grp.groupby("depth_bytes")["delta_bits"].agg(["mean", "sem", "count"]).reset_index()
        ax.errorbar(
            agg["depth_bytes"], agg["mean"], yerr=1.96 * agg["sem"].fillna(0),
            marker="o", ms=4, lw=1.3, capsize=2, color=_lang_color(lang), label=f"{lang} (n={int(agg['count'].sum())})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("context before mutated line (bytes)")
    ax.set_ylabel("extra bits for buggy line vs. original")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ablation(curve_pairs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], out_path: str) -> None:
    """curve_pairs: language -> {"annotated": (c,y), "stripped": (c,y)}."""
    n = len(curve_pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.6), dpi=150, squeeze=False)
    for ax, (lang, pair) in zip(axes[0], sorted(curve_pairs.items())):
        for variant, style in (("annotated", "-o"), ("stripped", "--s")):
            if variant not in pair:
                continue
            c, y = pair[variant]
            m = np.isfinite(y)
            ax.plot(c[m], y[m], style, ms=3.5, lw=1.2, color=_lang_color(lang),
                    alpha=1.0 if variant == "annotated" else 0.55, label=variant)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{lang}: type signatures vs. stripped")
        ax.set_xlabel("context (bytes)")
        ax.set_ylabel("bits per byte")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
