"""Analysis of variants-mode dumps: bug-injection surprise vs. context depth."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

LN2 = math.log(2.0)


def read_variants_dump(path: str) -> pd.DataFrame:
    """Parse a ppl-dump variants TSV into one row per (span_id, token)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("span_id\t"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) != 5:
                continue
            rows.append(
                {
                    "span_id": p[0],
                    "tok_idx": int(p[1]),
                    "token_id": int(p[2]),
                    "n_bytes": int(p[3]),
                    "logprob_nats": float(p[4]),
                }
            )
    return pd.DataFrame(rows)


def span_bits(df: pd.DataFrame) -> pd.DataFrame:
    """Total bits to encode each span (sum over its tokens)."""
    g = df.groupby("span_id").agg(
        bits=("logprob_nats", lambda s: float(-(s.sum()) / LN2)),
        n_tokens=("tok_idx", "count"),
        n_bytes=("n_bytes", "sum"),
    )
    return g.reset_index()


@dataclass
class AnomalyResult:
    language: str
    window_id: str
    site: int
    depth_bytes: int
    kind: str
    bits_orig: float
    bits_mut: float

    @property
    def delta_bits(self) -> float:
        return self.bits_mut - self.bits_orig


def collect_anomaly_results(manifest_rows: list[dict], results_dir: str) -> pd.DataFrame:
    """Join variants dumps with their manifests into a tidy Δbits table."""
    import os

    out: list[dict] = []
    for row in manifest_rows:
        path = os.path.join(results_dir, "anomaly", row["job_id"] + ".tsv")
        if not os.path.exists(path):
            continue
        df = read_variants_dump(path)
        if df.empty:
            continue
        bits = span_bits(df).set_index("span_id")["bits"]
        meta = pd.DataFrame(row["meta"])
        for site, grp in meta.groupby("site"):
            orig = grp[grp.kind == "orig"]
            if orig.empty:
                continue
            oid = orig.iloc[0]["span_id"]
            if oid not in bits.index:
                continue
            for _, m in grp[grp.kind != "orig"].iterrows():
                if m["span_id"] not in bits.index:
                    continue
                out.append(
                    {
                        "language": row["language"],
                        "window_id": row["window_id"],
                        "site": int(site),
                        "depth_bytes": int(m["depth_bytes"]),
                        "kind": m["kind"],
                        "bits_orig": float(bits[oid]),
                        "bits_mut": float(bits[m["span_id"]]),
                        "delta_bits": float(bits[m["span_id"]] - bits[oid]),
                    }
                )
    return pd.DataFrame(out)
