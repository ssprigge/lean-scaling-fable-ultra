import numpy as np

from leanscale.semantics import BLANK, CODE, COMMENT, HEADER, classify_bytes, token_classes


def test_classify_python():
    text = (
        "# ==== File: repo/mod.py ====\n"
        "# a comment\n"
        "\n"
        '"""docstring\n'
        "continues\n"
        '"""\n'
        "x = 1\n"
    )
    cls = classify_bytes(text, "python")
    lines = text.splitlines(keepends=True)
    offs = np.cumsum([0] + [len(l.encode()) for l in lines])
    assert cls[offs[0]] == HEADER
    assert cls[offs[1]] == COMMENT
    assert cls[offs[2]] == BLANK
    assert cls[offs[3]] == COMMENT and cls[offs[4]] == COMMENT and cls[offs[5]] == COMMENT
    assert cls[offs[6]] == CODE


def test_classify_c_block():
    text = "/* ==== File: r/a.c ==== */\n/* multi\nline */\nint x;\n"
    cls = classify_bytes(text, "c")
    lines = text.splitlines(keepends=True)
    offs = np.cumsum([0] + [len(l.encode()) for l in lines])
    assert cls[offs[0]] == HEADER
    assert cls[offs[1]] == COMMENT and cls[offs[2]] == COMMENT
    assert cls[offs[3]] == CODE


def test_classify_lean():
    text = "-- ==== File: m/A.lean ====\n-- doc\n/- block\nend -/\ntheorem t : 1 = 1 := rfl\n"
    cls = classify_bytes(text, "lean")
    lines = text.splitlines(keepends=True)
    offs = np.cumsum([0] + [len(l.encode()) for l in lines])
    assert cls[offs[0]] == HEADER
    assert cls[offs[1]] == COMMENT
    assert cls[offs[2]] == COMMENT and cls[offs[3]] == COMMENT
    assert cls[offs[4]] == CODE


def test_token_classes_alignment():
    text = "# c\nx=1\n"
    cls = classify_bytes(text, "python")
    # tokens: "# c\n" (4 bytes), "x=1\n" (4 bytes)
    tb = np.array([4, 4])
    tc = token_classes(cls, tb)
    assert tc[0] == COMMENT
    assert tc[1] == CODE


def test_unicode_byte_alignment():
    text = "# α β\nx = 'δ'\n"
    cls = classify_bytes(text, "python")
    assert len(cls) == len(text.encode("utf-8"))
    assert cls[0] == COMMENT
    assert cls[-2] == CODE
