import numpy as np

from leanscale.crossover import find_crossover
from leanscale.fitting import Fit, bootstrap_fits, fit_curve, power_floor


def _synth_curve(L_inf, A, alpha, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    c = np.geomspace(32, 65536, 24)
    y = power_floor(c, L_inf, A, alpha) * (1 + noise * rng.standard_normal(len(c)))
    w = np.full(len(c), 1e5)
    return c, y, w


def test_fit_recovers_params():
    c, y, w = _synth_curve(0.8, 3.0, 0.35)
    fit = fit_curve(c, y, w, model="floor")
    assert fit is not None
    assert abs(fit.params["L_inf"] - 0.8) < 0.05
    assert abs(fit.params["A"] - 3.0) < 0.3
    assert abs(fit.params["alpha"] - 0.35) < 0.03


def test_fit_noisy_still_reasonable():
    c, y, w = _synth_curve(1.0, 2.0, 0.25, noise=0.02)
    fit = fit_curve(c, y, w, model="floor")
    assert fit is not None
    assert 0.1 < fit.params["alpha"] < 0.5


def test_pure_power_law_fit():
    c, y, w = _synth_curve(0.0, 4.0, 0.2)
    fit = fit_curve(c, y, w, model="pure")
    assert fit is not None
    assert abs(fit.params["alpha"] - 0.2) < 0.02


def test_crossover_found():
    # A: better at small c; B: better exponent, crosses over later.
    fa = Fit(model="floor", params={"L_inf": 1.0, "A": 2.0, "alpha": 0.30}, aicc=0, rss=0, n_points=0)
    fb = Fit(model="floor", params={"L_inf": 0.4, "A": 6.0, "alpha": 0.30}, aicc=0, rss=0, n_points=0)
    x = find_crossover(fa, fb, "a", "b", c_min=32)
    assert x.lang_lo_small == "a"
    assert x.lang_lo_large == "b"
    assert x.c_cross is not None
    # analytic crossing: L_inf difference 0.6 = 4 * c^-0.3 -> c = (4/0.6)^(1/0.3)
    expected = (4.0 / 0.6) ** (1 / 0.3)
    assert abs(np.log(x.c_cross) - np.log(expected)) < 0.05


def test_no_crossover():
    fa = Fit(model="floor", params={"L_inf": 0.5, "A": 2.0, "alpha": 0.3}, aicc=0, rss=0, n_points=0)
    fb = Fit(model="floor", params={"L_inf": 0.9, "A": 3.0, "alpha": 0.3}, aicc=0, rss=0, n_points=0)
    x = find_crossover(fa, fb, "a", "b")
    assert x.c_cross is None
    assert x.lang_lo_small == "a" and x.lang_lo_large == "a"


def test_bootstrap_shapes():
    rng = np.random.default_rng(0)
    c = np.geomspace(32, 65536, 20)
    W = 6
    base = power_floor(c, 0.9, 2.5, 0.3)
    bytes_pw = np.full((W, len(c)), 5e4)
    bits_pw = base[None, :] * bytes_pw * (1 + 0.01 * rng.standard_normal((W, len(c))))
    boots = bootstrap_fits(bits_pw, bytes_pw, c, model="floor", n_boot=25, seed=1)
    assert set(boots) == {"L_inf", "A", "alpha"}
    assert all(len(v) > 0 for v in boots.values())
