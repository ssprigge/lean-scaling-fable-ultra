import gzip
import math

import numpy as np

from leanscale.aggregate import curve_from_windows, make_bins, per_window_curves, read_dump

LN2 = math.log(2.0)


def _write_dump(path, rows, gz=False):
    opener = gzip.open if gz else open
    with opener(path, "wt") as f:
        f.write("# mode=full n_tokens=%d input_bytes=999\n" % len(rows))
        f.write("idx\ttoken_id\tn_bytes\tlogprob_nats\n")
        for i, (tok, nb, lp) in enumerate(rows):
            f.write(f"{i}\t{tok}\t{nb}\t{lp}\n")


def test_read_dump_basic(tmp_path):
    rows = [(10, 4, "nan"), (11, 3, -1.386294), (12, 5, -0.693147)]
    p = tmp_path / "w.tsv.gz"
    _write_dump(p, rows, gz=True)
    w = read_dump(str(p), "w", "python", "sorted")
    assert w.token_bytes.tolist() == [4, 3, 5]
    assert np.isnan(w.bits[0])
    assert abs(w.bits[1] - 2.0) < 1e-5  # -ln(0.25)/ln2 = 2 bits
    assert abs(w.bits[2] - 1.0) < 1e-5
    assert w.ctx_bytes.tolist() == [0.0, 4.0, 7.0]


def test_binning_bpb(tmp_path):
    # Constant 2 bits/token, 4 bytes/token -> 0.5 bits/byte at all depths.
    rows = [(1, 4, "nan")] + [(1, 4, -2 * LN2)] * 500
    p = tmp_path / "w.tsv"
    _write_dump(p, rows)
    w = read_dump(str(p), "w", "x", "sorted")
    edges = make_bins(24, 4096, 8)
    curve = curve_from_windows([w], edges)
    got = curve.bpb[np.isfinite(curve.bpb)]
    assert np.allclose(got, 0.5, atol=1e-6)


def test_per_window_stacks(tmp_path):
    rows = [(1, 4, "nan")] + [(1, 4, -1.0)] * 100
    p1, p2 = tmp_path / "a.tsv", tmp_path / "b.tsv"
    _write_dump(p1, rows)
    _write_dump(p2, rows)
    w1 = read_dump(str(p1), "a", "x", "sorted")
    w2 = read_dump(str(p2), "b", "x", "sorted")
    edges = make_bins(24, 2048, 6)
    B, N = per_window_curves([w1, w2], edges)
    assert B.shape == (2, 6)
    assert np.allclose(B[0], B[1])
    # combined curve equals single-window curve for identical windows
    c1 = curve_from_windows([w1], edges)
    c2 = curve_from_windows([w1, w2], edges)
    m = np.isfinite(c1.bpb)
    assert np.allclose(c1.bpb[m], c2.bpb[m])
