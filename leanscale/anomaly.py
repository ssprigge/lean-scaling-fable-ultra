"""Bug-injection anomaly-detection cross-check.

The essay's 'inverse scaling' probe: inject subtle, stylistically plausible
bugs into code at varying context depths and ask how *surprised* the frozen
model is (extra bits to encode the mutated line vs. the original line, given
the identical prefix). A model that genuinely uses codebase context should
grow more surprised at bugs the more context it has read; local plausibility
alone is differenced out by comparing against the original line at the same
position.

Mutators are deliberately language-agnostic surface edits that preserve
style: comparison-operator swaps, arithmetic sign flips, off-by-one constant
perturbations, and identifier swaps within a line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .languages import LANGUAGES

KEYWORDS = {
    "if", "then", "else", "for", "while", "return", "let", "in", "do", "def",
    "fn", "fun", "match", "case", "where", "import", "from", "with", "as",
    "type", "struct", "enum", "impl", "pub", "mut", "const", "static", "var",
    "class", "self", "this", "true", "false", "null", "none", "nil", "and",
    "or", "not", "theorem", "lemma", "by", "have", "show", "exact", "simp",
    "instance", "namespace", "end", "open", "universe", "variable", "async",
    "await", "yield", "module", "data", "newtype", "instance", "deriving",
}


@dataclass(frozen=True)
class Mutation:
    kind: str
    original: str
    mutated: str


def _swap_first(line: str, a: str, b: str) -> str | None:
    ia = line.find(a)
    ib = line.find(b)
    if ia >= 0 and (ib < 0 or ia <= ib):
        return line[:ia] + b + line[ia + len(a):]
    if ib >= 0:
        return line[:ib] + a + line[ib + len(b):]
    return None


def mutate_comparison(line: str) -> str | None:
    # Ordered so that longer operators are matched before their prefixes.
    for a, b in ((" <= ", " < "), (" >= ", " > "), (" == ", " != ")):
        m = _swap_first(line, a, b)
        if m is not None and m != line:
            return m
    return None


def mutate_arith(line: str) -> str | None:
    if "++" in line or "--" in line or "+=" in line or "-=" in line:
        return None
    m = _swap_first(line, " + ", " - ")
    return m if m is not None and m != line else None


_INT_RE = re.compile(r"(?<![\w.])([2-9]|[1-9][0-9]{1,3})(?![\w.])")


def mutate_off_by_one(line: str) -> str | None:
    m = _INT_RE.search(line)
    if not m:
        return None
    val = int(m.group(1))
    return line[: m.start(1)] + str(val + 1) + line[m.end(1):]


_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']{2,}")


def mutate_identifier_swap(line: str) -> str | None:
    ids = [m for m in _ID_RE.finditer(line) if m.group(0).lower() not in KEYWORDS]
    distinct: list[str] = []
    for m in ids:
        if m.group(0) not in distinct:
            distinct.append(m.group(0))
    if len(distinct) < 2:
        return None
    a, b = distinct[0], distinct[1]
    # Swap the first occurrence of each (a plausible wrong-variable bug).
    out, swapped_a, swapped_b = [], False, False
    i = 0
    result = ""
    for m in _ID_RE.finditer(line):
        result += line[i:m.start()]
        tok = m.group(0)
        if tok == a and not swapped_a:
            result += b
            swapped_a = True
        elif tok == b and not swapped_b:
            result += a
            swapped_b = True
        else:
            result += tok
        i = m.end()
    result += line[i:]
    return result if result != line else None


MUTATORS = {
    "cmp_swap": mutate_comparison,
    "arith_swap": mutate_arith,
    "off_by_one": mutate_off_by_one,
    "id_swap": mutate_identifier_swap,
}


def is_code_line(line: str, language: str) -> bool:
    """Reject blank lines, comments, and window headers as mutation targets."""
    cfg = LANGUAGES[language]
    s = line.strip()
    if len(s) < 16:
        return False
    if s.startswith(cfg.comment_prefix):
        return False
    if cfg.block_comment and (s.startswith(cfg.block_comment[0]) or s.startswith("*")):
        return False
    if "==== File:" in s:
        return False
    return True


def mutations_for_line(line: str) -> list[Mutation]:
    out = []
    for kind, fn in MUTATORS.items():
        m = fn(line)
        if m is not None and m != line:
            out.append(Mutation(kind=kind, original=line, mutated=m))
    return out


@dataclass
class AnomalySite:
    depth_bytes: int          # prefix length in bytes at the line start
    line: str
    mutations: list[Mutation]


def pick_sites(window_text: str, language: str, depths: list[int]) -> list[AnomalySite]:
    """For each requested depth, find the first mutable code line at or after
    that byte offset. Returned prefix depths are exact line-start offsets."""
    sites: list[AnomalySite] = []
    # Precompute line start offsets (in bytes; text is ASCII-dominated code,
    # but measure real UTF-8 lengths to be exact).
    lines = window_text.splitlines(keepends=True)
    offsets = []
    off = 0
    for ln in lines:
        offsets.append(off)
        off += len(ln.encode("utf-8"))
    used: set[int] = set()
    for depth in depths:
        found = None
        for i, (start, ln) in enumerate(zip(offsets, lines)):
            if start < depth or i in used:
                continue
            stripped = ln.rstrip("\n")
            if not is_code_line(stripped, language):
                continue
            muts = mutations_for_line(stripped)
            if not muts:
                continue
            found = (i, start, stripped, muts)
            break
        if found is None:
            continue
        i, start, stripped, muts = found
        used.add(i)
        sites.append(AnomalySite(depth_bytes=start, line=stripped, mutations=muts))
    return sites


def build_variants_job(
    window_text: str,
    language: str,
    depths: list[int],
    scratch_dir: str,
    job_name: str,
) -> tuple[str, list[dict]] | None:
    """Write a ppl-dump `variants` job: PREFIX chunks growing through the
    window, and at each site the original line plus its mutants as SPANs.

    Returns (job_file_path, metadata_rows) or None if no usable site.
    """
    import os

    sites = pick_sites(window_text, language, depths)
    if not sites:
        return None
    os.makedirs(scratch_dir, exist_ok=True)
    job_lines: list[str] = []
    meta: list[dict] = []
    prev = 0
    # Work in byte space so prefixes cut exactly at line starts.
    blob = window_text.encode("utf-8")
    for si, site in enumerate(sites):
        chunk = blob[prev:site.depth_bytes]
        if not chunk:
            continue
        pfile = os.path.join(scratch_dir, f"{job_name}_p{si}.txt")
        with open(pfile, "wb") as f:
            f.write(chunk)
        job_lines.append(f"PREFIX {pfile}")
        prev = site.depth_bytes

        spans = [("orig", site.line)] + [(m.kind, m.mutated) for m in site.mutations]
        for kind, text in spans:
            sid = f"s{si}_{kind}"
            sfile = os.path.join(scratch_dir, f"{job_name}_{sid}.txt")
            with open(sfile, "wb") as f:
                f.write((text + "\n").encode("utf-8"))
            job_lines.append(f"SPAN {sid} {sfile}")
            meta.append(
                {
                    "site": si,
                    "span_id": sid,
                    "kind": kind,
                    "depth_bytes": site.depth_bytes,
                    "line": site.line if kind == "orig" else text,
                }
            )
    if not meta:
        return None
    job_path = os.path.join(scratch_dir, f"{job_name}.job")
    with open(job_path, "w") as f:
        f.write("\n".join(job_lines) + "\n")
    return job_path, meta
