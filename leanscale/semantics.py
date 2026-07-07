"""Semantic segmentation of window bytes: header / comment / blank / code.

The essay's *Semantics* cross-check asks which parts of a language are
predictable boilerplate vs. genuinely surprising code. License headers and
comments are highly predictable filler that could flatter a language's
curve, so we classify every byte of a window with a small per-language state
machine (line comments, block comments, Python docstrings) and re-derive
BPB-vs-context curves restricted to *code* bytes as a robustness check.

Strings are counted as code; the classifier is deliberately simple (a
comment marker inside a string literal misclassifies that line) — errors are
rare and language-symmetric at the corpus scale we operate on.
"""

from __future__ import annotations

import numpy as np

from .languages import LANGUAGES

HEADER, COMMENT, BLANK, CODE = 0, 1, 2, 3
CLASS_NAMES = {HEADER: "header", COMMENT: "comment", BLANK: "blank", CODE: "code"}


def classify_bytes(text: str, language: str) -> np.ndarray:
    """Per-byte class array (uint8) for the UTF-8 encoding of `text`."""
    cfg = LANGUAGES[language]
    line_c = cfg.comment_prefix
    block = cfg.block_comment
    out = np.empty(len(text.encode("utf-8")), dtype=np.uint8)
    pos = 0  # byte position
    in_block = False
    in_doc: str | None = None  # python triple-quote delimiter
    for raw in text.splitlines(keepends=True):
        nb = len(raw.encode("utf-8"))
        s = raw.strip()
        if in_block:
            out[pos:pos + nb] = COMMENT
            if block and block[1] in raw:
                in_block = False
        elif in_doc is not None:
            out[pos:pos + nb] = COMMENT
            if in_doc in raw:
                in_doc = None
        elif "==== File:" in raw and s.startswith((line_c, block[0] if block else line_c)):
            out[pos:pos + nb] = HEADER
        elif not s:
            out[pos:pos + nb] = BLANK
        elif s.startswith(line_c):
            out[pos:pos + nb] = COMMENT
        elif block and s.startswith(block[0]):
            out[pos:pos + nb] = COMMENT
            body = s[len(block[0]):]
            if block[1] not in body:
                in_block = True
        elif language == "python" and (s.startswith('"""') or s.startswith("'''")):
            q = s[:3]
            out[pos:pos + nb] = COMMENT
            if s.count(q) < 2:
                in_doc = q
        else:
            out[pos:pos + nb] = CODE
        pos += nb
    assert pos == len(out)
    return out


def token_classes(byte_classes: np.ndarray, token_bytes: np.ndarray) -> np.ndarray:
    """Class of each token = class of its first byte (tokens rarely straddle
    class boundaries since boundaries fall on newlines)."""
    starts = np.concatenate([[0], np.cumsum(token_bytes)[:-1]]).astype(np.int64)
    starts = np.clip(starts, 0, len(byte_classes) - 1)
    return byte_classes[starts]


def class_breakdown(bits: np.ndarray, token_bytes: np.ndarray, tclass: np.ndarray) -> dict:
    """Total bits and bytes per class (NaN-safe for position 0)."""
    out = {}
    valid = ~np.isnan(bits)
    for cls, name in CLASS_NAMES.items():
        m = (tclass == cls) & valid
        out[name] = {
            "bits": float(bits[m].sum()),
            "bytes": float(token_bytes[m].sum()),
            "bpb": float(bits[m].sum() / token_bytes[m].sum()) if token_bytes[m].sum() else float("nan"),
            "frac_bytes": float(token_bytes[m].sum() / token_bytes[valid].sum()) if token_bytes[valid].sum() else 0.0,
        }
    return out
