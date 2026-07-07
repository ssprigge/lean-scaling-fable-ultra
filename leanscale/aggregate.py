"""Aggregation of per-token losses into bits-per-byte vs. context curves.

The dumper emits, per window, one row per token: its UTF-8 byte length and
its log-probability in nats. We convert to bits and attribute each token's
bits to the *byte offset of context already read* (cumulative bytes of all
preceding tokens). Binning by log-spaced context depth and dividing summed
bits by summed bytes yields bits-per-byte (BPB) as a function of context —
normalizing away both tokenizer verbosity differences and per-line length
differences between languages, as the essay's design calls for.
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass

import numpy as np

LN2 = math.log(2.0)


@dataclass
class WindowLosses:
    window_id: str
    language: str
    ordering: str
    token_ids: np.ndarray      # int32, all tokens (position 0 included)
    token_bytes: np.ndarray    # int32
    bits: np.ndarray           # float64, NaN for position 0
    ctx_bytes: np.ndarray      # float64, bytes of context preceding each token
    ctx_tokens: np.ndarray     # int64

    @property
    def total_bytes(self) -> int:
        return int(self.token_bytes.sum())


def read_dump(path: str, window_id: str = "", language: str = "", ordering: str = "") -> WindowLosses:
    """Parse a `ppl-dump --mode full` TSV (optionally gzipped)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    idxs, toks, nbytes, lps = [], [], [], []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("idx\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            idxs.append(int(parts[0]))
            toks.append(int(parts[1]))
            nbytes.append(int(parts[2]))
            lps.append(float(parts[3]))
    token_ids = np.asarray(toks, dtype=np.int32)
    token_bytes = np.asarray(nbytes, dtype=np.int32)
    logprobs = np.asarray(lps, dtype=np.float64)
    order = np.argsort(np.asarray(idxs))
    token_ids, token_bytes, logprobs = token_ids[order], token_bytes[order], logprobs[order]
    bits = -logprobs / LN2
    cum = np.cumsum(token_bytes, dtype=np.int64)
    ctx_bytes = np.concatenate([[0], cum[:-1]]).astype(np.float64)
    ctx_tokens = np.arange(len(token_ids), dtype=np.int64)
    return WindowLosses(
        window_id=window_id,
        language=language,
        ordering=ordering,
        token_ids=token_ids,
        token_bytes=token_bytes,
        bits=bits,
        ctx_bytes=ctx_bytes,
        ctx_tokens=ctx_tokens,
    )


def make_bins(lo: float = 24.0, hi: float = 131072.0, n: int = 26) -> np.ndarray:
    """Log-spaced context-depth bin edges in bytes."""
    return np.geomspace(lo, hi, n + 1)


@dataclass
class BinnedCurve:
    """BPB curve for one group (e.g. one language+ordering)."""
    centers: np.ndarray        # geometric bin centers (context bytes)
    bpb: np.ndarray            # bits per byte in each bin (NaN if empty)
    n_bytes: np.ndarray        # bytes contributing to each bin
    n_windows: int


def bin_window(w: WindowLosses, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window sums: (bits_per_bin, bytes_per_bin). Position 0 excluded."""
    valid = ~np.isnan(w.bits)
    ctx = np.clip(w.ctx_bytes[valid], edges[0], None)
    bits = w.bits[valid]
    nbytes = w.token_bytes[valid].astype(np.float64)
    which = np.digitize(ctx, edges) - 1
    which = np.clip(which, 0, len(edges) - 2)
    bits_sum = np.bincount(which, weights=bits, minlength=len(edges) - 1)
    byte_sum = np.bincount(which, weights=nbytes, minlength=len(edges) - 1)
    return bits_sum, byte_sum


def curve_from_windows(windows: list[WindowLosses], edges: np.ndarray) -> BinnedCurve:
    bits_tot = np.zeros(len(edges) - 1)
    byte_tot = np.zeros(len(edges) - 1)
    for w in windows:
        b, n = bin_window(w, edges)
        bits_tot += b
        byte_tot += n
    with np.errstate(invalid="ignore", divide="ignore"):
        bpb = np.where(byte_tot > 0, bits_tot / byte_tot, np.nan)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return BinnedCurve(centers=centers, bpb=bpb, n_bytes=byte_tot, n_windows=len(windows))


def per_window_curves(windows: list[WindowLosses], edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stacked per-window (bits, bytes) bin sums, for bootstrap resampling."""
    B = np.stack([bin_window(w, edges)[0] for w in windows])
    N = np.stack([bin_window(w, edges)[1] for w in windows])
    return B, N
