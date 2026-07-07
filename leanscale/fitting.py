"""Scaling-law fits for BPB-vs-context curves.

Primary form (in-context scaling law with an irreducible floor):

    L(c) = L_inf + A * c^(-alpha)

with c = context depth in bytes, L = bits/byte. `alpha` is the *predictability
scaling exponent* the essay proposes as a language design metric: how fast
additional context keeps paying off. `L_inf` is the extrapolated loss floor,
`A + L_inf` the small-context constant. A pure power law (L_inf = 0) is also
fit and compared by AICc, since with finite windows the floor and exponent
are correlated and the floor may not be identifiable.

Uncertainty: bootstrap over windows (resample windows with replacement,
rebuild binned curves, refit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


def power_floor(c, L_inf, A, alpha):
    return L_inf + A * np.power(c, -alpha)


def power_pure(c, A, alpha):
    return A * np.power(c, -alpha)


@dataclass
class Fit:
    model: str                     # "floor" | "pure"
    params: dict[str, float]
    aicc: float
    rss: float
    n_points: int
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    def predict(self, c: np.ndarray) -> np.ndarray:
        p = self.params
        if self.model == "floor":
            return power_floor(c, p["L_inf"], p["A"], p["alpha"])
        return power_pure(c, p["A"], p["alpha"])


def _aicc(rss: float, n: int, k: int) -> float:
    if n <= k + 1:
        return np.inf
    aic = n * np.log(max(rss, 1e-300) / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def _clean(centers: np.ndarray, bpb: np.ndarray, weights: np.ndarray):
    m = np.isfinite(bpb) & (weights > 0) & (bpb > 0)
    return centers[m], bpb[m], weights[m]


def fit_curve(centers: np.ndarray, bpb: np.ndarray, weights: np.ndarray, model: str = "floor") -> Fit | None:
    """Weighted least-squares fit in linear BPB space.

    `weights` are per-bin byte counts; sigma ~ 1/sqrt(bytes) approximates the
    shrinking sampling noise of bigger bins.
    """
    c, y, w = _clean(centers, bpb, weights)
    if len(c) < 5:
        return None
    sigma = 1.0 / np.sqrt(w)

    best: Fit | None = None
    ymin, ymax = float(y.min()), float(y.max())
    for alpha0 in (0.05, 0.15, 0.3, 0.6):
        try:
            if model == "floor":
                p0 = (max(ymin * 0.7, 1e-3), max(ymax - ymin, 1e-3), alpha0)
                bounds = ([0.0, 1e-9, 1e-4], [ymax, 100.0, 2.0])
                popt, _ = curve_fit(power_floor, c, y, p0=p0, sigma=sigma, bounds=bounds, maxfev=20000)
                pred = power_floor(c, *popt)
                params = {"L_inf": popt[0], "A": popt[1], "alpha": popt[2]}
                k = 3
            else:
                p0 = (ymax, alpha0)
                bounds = ([1e-9, 1e-4], [100.0, 2.0])
                popt, _ = curve_fit(power_pure, c, y, p0=p0, sigma=sigma, bounds=bounds, maxfev=20000)
                pred = power_pure(c, *popt)
                params = {"A": popt[0], "alpha": popt[1]}
                k = 2
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum(((y - pred) / sigma) ** 2))
        cand = Fit(model=model, params={k_: float(v) for k_, v in params.items()},
                   aicc=_aicc(rss, len(c), k), rss=rss, n_points=len(c))
        if best is None or cand.rss < best.rss:
            best = cand
    return best


def bootstrap_fits(
    per_window_bits: np.ndarray,   # (W, B) bits per bin per window
    per_window_bytes: np.ndarray,  # (W, B)
    centers: np.ndarray,
    model: str = "floor",
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Bootstrap over windows; returns arrays of fitted params per replicate."""
    rng = np.random.default_rng(seed)
    W = per_window_bits.shape[0]
    out: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, W, size=W)
        bits = per_window_bits[idx].sum(axis=0)
        nbytes = per_window_bytes[idx].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bpb = np.where(nbytes > 0, bits / nbytes, np.nan)
        fit = fit_curve(centers, bpb, nbytes, model=model)
        if fit is None:
            continue
        for k, v in fit.params.items():
            out.setdefault(k, []).append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo = float(np.percentile(samples, 100 * (1 - level) / 2))
    hi = float(np.percentile(samples, 100 * (1 + level) / 2))
    return lo, hi
