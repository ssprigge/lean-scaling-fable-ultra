"""Crossover analysis: where does one language's BPB curve undercut another's?

Given two fitted curves L_i(c), L_j(c), find crossings of their difference in
log-context space, both inside the measured range and extrapolated far beyond
it (the essay's headline question: at what codebase size would Lean become
more absolutely predictable than Python?). Bootstrap replicates of the fits
propagate uncertainty into the crossover location.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitting import Fit


@dataclass
class Crossover:
    lang_lo_small: str      # language with lower BPB at c_min
    lang_lo_large: str      # language with lower BPB at c_max
    c_cross: float | None   # bytes; None if no crossing in (c_min, c_max)
    c_min: float
    c_max: float


def _curve_diff(fit_a: Fit, fit_b: Fit, cs: np.ndarray) -> np.ndarray:
    return fit_a.predict(cs) - fit_b.predict(cs)


def find_crossover(
    fit_a: Fit,
    fit_b: Fit,
    name_a: str,
    name_b: str,
    c_min: float = 128.0,
    c_max: float = 1e12,
    n_grid: int = 4000,
) -> Crossover:
    """Locate the first sign change of L_a - L_b on a log grid of context sizes.

    `c_max` defaults to 1 TB of code, i.e. far beyond plausibility; a crossing
    reported near it is an extrapolation artifact, which the caller should
    treat as 'effectively never'.
    """
    cs = np.geomspace(c_min, c_max, n_grid)
    d = _curve_diff(fit_a, fit_b, cs)
    lo_small = name_a if d[0] < 0 else name_b
    lo_large = name_a if d[-1] < 0 else name_b
    sign = np.sign(d)
    flips = np.nonzero(np.diff(sign) != 0)[0]
    if len(flips) == 0:
        return Crossover(lo_small, lo_large, None, c_min, c_max)
    k = flips[0]
    # Bisect in log space for a precise crossing.
    lo, hi = cs[k], cs[k + 1]
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if np.sign(_curve_diff(fit_a, fit_b, np.array([mid]))[0]) == sign[k]:
            lo = mid
        else:
            hi = mid
    return Crossover(lo_small, lo_large, float(np.sqrt(lo * hi)), c_min, c_max)


def bootstrap_crossover(
    boot_a: dict[str, np.ndarray],
    boot_b: dict[str, np.ndarray],
    model: str,
    c_min: float = 128.0,
    c_max: float = 1e12,
    seed: int = 0,
) -> np.ndarray:
    """Crossover locations across paired bootstrap replicates (NaN = none)."""
    rng = np.random.default_rng(seed)
    n = min(len(next(iter(boot_a.values()))), len(next(iter(boot_b.values()))))
    ia = rng.permutation(len(next(iter(boot_a.values()))))[:n]
    ib = rng.permutation(len(next(iter(boot_b.values()))))[:n]
    out = np.full(n, np.nan)
    for k in range(n):
        pa = {key: float(v[ia[k]]) for key, v in boot_a.items()}
        pb = {key: float(v[ib[k]]) for key, v in boot_b.items()}
        fa = Fit(model=model, params=pa, aicc=np.nan, rss=np.nan, n_points=0)
        fb = Fit(model=model, params=pb, aicc=np.nan, rss=np.nan, n_points=0)
        c = find_crossover(fa, fb, "a", "b", c_min=c_min, c_max=c_max)
        if c.c_cross is not None:
            out[k] = c.c_cross
    return out
