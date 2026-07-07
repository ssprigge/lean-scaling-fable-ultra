"""Sweep orchestration: drive ppl-dump over windows with resumable checkpoints.

Each (language, ordering, window) job writes its window text to scratch, runs
the dumper subprocess, gzips the TSV into the results directory, and appends
a manifest line. Jobs whose output already exists are skipped, so an
interrupted sweep resumes for free.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass


@dataclass
class DumperConfig:
    binary: str
    model: str
    n_ctx: int = 16384
    n_batch: int = 512
    n_threads: int = 4


def run_full_window(
    cfg: DumperConfig,
    window_text: str,
    out_path: str,          # final gzipped TSV path
    scratch_dir: str,
) -> dict:
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = os.path.basename(out_path).replace(".tsv.gz", "")
    in_file = os.path.join(scratch_dir, base + ".txt")
    raw_out = os.path.join(scratch_dir, base + ".tsv")
    with open(in_file, "w", encoding="utf-8") as f:
        f.write(window_text)
    t0 = time.time()
    proc = subprocess.run(
        [
            cfg.binary, "--model", cfg.model, "--mode", "full",
            "--input", in_file, "--output", raw_out,
            "--ctx", str(cfg.n_ctx), "--batch", str(cfg.n_batch),
            "--threads", str(cfg.n_threads),
        ],
        capture_output=True,
        text=True,
    )
    wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"ppl-dump failed ({proc.returncode}): {proc.stderr[-2000:]}")
    with open(raw_out, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(raw_out)
    os.remove(in_file)
    return {"wall_s": round(wall, 1)}


def run_variants_job(cfg: DumperConfig, job_path: str, out_path: str) -> dict:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(
        [
            cfg.binary, "--model", cfg.model, "--mode", "variants",
            "--job", job_path, "--output", out_path,
            "--ctx", str(cfg.n_ctx), "--batch", str(cfg.n_batch),
            "--threads", str(cfg.n_threads),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ppl-dump variants failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return {"wall_s": round(time.time() - t0, 1)}


def append_manifest(manifest_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_manifest(manifest_path: str) -> list[dict]:
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        return [json.loads(line) for line in f if line.strip()]
