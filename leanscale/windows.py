"""Measurement-window assembly.

A *window* is a single long text built by concatenating corpus files (each
prefixed by a native-comment header naming the file) that the frozen LLM
reads left-to-right; per-position losses over the window trace out the
in-context scaling curve.

Orderings:
  * ``sorted``   — files in path-sorted order within each repo (module
    locality: neighboring files are related, approximating how a codebase is
    actually laid out and read);
  * ``shuffled`` — files in seeded-random order (destroys locality; measures
    the language/domain prior without cross-file structure).

Windows within one ordering are disjoint (no file reused), assembled by
chopping the ordered file list into consecutive byte-budgeted chunks starting
from a seeded offset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .corpus import FileRecord
from .languages import LANGUAGES, header_line


@dataclass
class Window:
    language: str
    ordering: str          # "sorted" | "shuffled"
    index: int
    files: list[FileRecord]
    text: str

    @property
    def n_bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def window_id(self) -> str:
        return f"{self.language}-{self.ordering}-w{self.index}"


def render_window_text(files: list[FileRecord], language: str) -> str:
    cfg = LANGUAGES[language]
    parts: list[str] = []
    for rec in files:
        parts.append(header_line(cfg, rec.display_path))
        with open(rec.abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.endswith("\n"):
            content += "\n"
        parts.append(content)
        parts.append("\n")
    return "".join(parts)


def order_files(records: list[FileRecord], ordering: str, seed: int) -> list[FileRecord]:
    if ordering == "sorted":
        ordered = sorted(records, key=lambda r: (r.repo, r.rel_path))
        # Rotate to a seeded starting file so different runs can sample
        # different corpus regions while preserving local sorted structure.
        rng = random.Random(seed)
        off = rng.randrange(len(ordered)) if ordered else 0
        return ordered[off:] + ordered[:off]
    if ordering == "shuffled":
        ordered = sorted(records, key=lambda r: (r.repo, r.rel_path))
        rng = random.Random(seed)
        rng.shuffle(ordered)
        return ordered
    raise ValueError(f"unknown ordering {ordering!r}")


def build_windows(
    records: list[FileRecord],
    language: str,
    ordering: str,
    n_windows: int,
    byte_budget: int,
    seed: int = 0,
) -> list[Window]:
    """Chop the ordered file list into `n_windows` disjoint windows of at
    least `byte_budget` bytes each (headers included)."""
    ordered = order_files(records, ordering, seed)
    windows: list[Window] = []
    i = 0
    for w in range(n_windows):
        chosen: list[FileRecord] = []
        acc = 0
        while i < len(ordered) and acc < byte_budget:
            rec = ordered[i]
            chosen.append(rec)
            acc += rec.n_bytes + 64  # rough header overhead
            i += 1
        if not chosen or acc < byte_budget * 0.9:
            break  # corpus exhausted
        windows.append(
            Window(
                language=language,
                ordering=ordering,
                index=w,
                files=chosen,
                text=render_window_text(chosen, language),
            )
        )
    if len(windows) < n_windows:
        raise ValueError(
            f"{language}/{ordering}: corpus supports only {len(windows)} of "
            f"{n_windows} requested disjoint windows of {byte_budget} bytes"
        )
    return windows
