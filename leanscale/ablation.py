"""Type-signature ablation (the essay's 'do type signatures actually help?').

Python: surgically strip parameter annotations, return annotations, and
annotated-assignment types from source using AST node spans, preserving all
other formatting byte-for-byte. Measuring BPB curves on annotated vs.
stripped variants of the *same* files quantifies the global predictive value
of optional type signatures at each context depth.

Haskell: strip single-line top-level type signatures (`name :: Type`),
a cruder but serviceable analogue.
"""

from __future__ import annotations

import ast


def _spans_to_strip(tree: ast.AST) -> list[tuple[int, int, int, int, str]]:
    """Collect (lineno, col, end_lineno, end_col, kind) spans to delete.

    Kinds: 'ann' (': T' in args / AnnAssign), 'ret' ('-> T' return annotation).
    Line/col are 1-based lineno and 0-based col offsets as in the AST.
    """
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            all_args = (
                list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
                + ([args.vararg] if args.vararg else [])
                + ([args.kwarg] if args.kwarg else [])
            )
            for a in all_args:
                if a.annotation is not None:
                    spans.append(
                        (a.lineno, a.col_offset + len(a.arg),
                         a.annotation.end_lineno, a.annotation.end_col_offset, "ann")
                    )
            if node.returns is not None:
                r = node.returns
                spans.append((r.lineno, r.col_offset, r.end_lineno, r.end_col_offset, "ret"))
        elif isinstance(node, ast.AnnAssign) and node.value is not None and node.simple:
            t = node.target
            spans.append((t.end_lineno, t.end_col_offset, node.annotation.end_lineno,
                          node.annotation.end_col_offset, "ann"))
    return spans


def strip_python_annotations(source: str) -> str:
    """Return `source` with function/variable type annotations removed.

    Only annotations are touched; comments, spacing, and everything else are
    preserved. Files that fail to parse are returned unchanged.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    # Build absolute char offsets per line for span surgery.
    line_off = [0]
    for ln in lines:
        line_off.append(line_off[-1] + len(ln))

    def abs_off(lineno: int, col: int) -> int:
        return line_off[lineno - 1] + col

    cuts: list[tuple[int, int]] = []
    for lineno, col, end_lineno, end_col, kind in _spans_to_strip(tree):
        if kind == "ret":
            # Delete from the '->' immediately preceding the annotation to
            # the annotation's end.
            ann_start = abs_off(lineno, col)
            end = abs_off(end_lineno, end_col)
            arrow = source.rfind("->", 0, ann_start)
            if arrow < 0:
                continue
            cuts.append((arrow, end))
        else:
            # Delete ': T' — from the ':' after the target/arg name to the
            # annotation end.
            start = abs_off(lineno, col)
            end = abs_off(end_lineno, end_col)
            colon = source.find(":", start, end)
            if colon < 0:
                continue
            cuts.append((colon, end))
    if not cuts:
        return source
    # Merge overlapping cuts, apply right-to-left.
    cuts.sort()
    merged: list[list[int]] = []
    for s, e in cuts:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = source
    for s, e in reversed(merged):
        # Absorb whitespace immediately before the cut so removals like
        # "…) -> dict:" leave "…):" rather than "…) :".
        while s > 0 and out[s - 1] in (" ", "\t"):
            s -= 1
        out = out[:s] + out[e:]
    return out


def strip_haskell_signatures(source: str) -> str:
    """Remove single-line top-level type signatures like `foo :: A -> B`.

    Multi-line signatures (continuation lines indented after `::`) are also
    consumed greedily until a line that starts a new top-level definition.
    """
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    import re

    sig_re = re.compile(r"^([a-z_][A-Za-z0-9_']*(?:\s*,\s*[a-z_][A-Za-z0-9_']*)*)\s*::")
    while i < len(lines):
        m = sig_re.match(lines[i])
        if m:
            i += 1
            # consume indented continuation lines
            while i < len(lines) and lines[i][:1] in (" ", "\t") and lines[i].strip():
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)
