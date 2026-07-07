import os

import pytest

from leanscale.corpus import FileRecord, scan_language
from leanscale.windows import build_windows, order_files, render_window_text


@pytest.fixture()
def fake_corpus(tmp_path):
    root = tmp_path / "python" / "repo1"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "vendor").mkdir()
    for i in range(30):
        (root / "pkg" / f"mod_{i:02d}.py").write_text(
            f"# module {i}\n" + f"def f_{i}(x):\n    return x + {i}\n" * 20
        )
    # Files that must be excluded:
    (root / "tests" / "test_x.py").write_text("def test():\n    assert 1 == 1\n" * 20)
    (root / "vendor" / "v.py").write_text("VENDORED = True\n" * 30)
    (root / "pkg" / "gen.py").write_text("# This file is auto-generated. DO NOT EDIT\n" + "x = 1\n" * 30)
    (root / "pkg" / "dup_a.py").write_text("DUP = 1\n" * 40)
    (root / "pkg" / "dup_b.py").write_text("DUP = 1\n" * 40)
    (root / "pkg" / "tiny.py").write_text("x=1\n")
    return tmp_path


def test_scan_filters(fake_corpus):
    recs = scan_language(fake_corpus, "python")
    paths = {r.rel_path for r in recs}
    assert not any("tests/" in p for p in paths)
    assert not any("vendor/" in p for p in paths)
    assert not any("gen.py" in p for p in paths)
    assert "pkg/tiny.py" not in paths
    # dedup: only one of the identical files survives
    assert sum(1 for p in paths if p.startswith("pkg/dup")) == 1
    assert len(recs) == 31


def test_windows_disjoint_and_budgeted(fake_corpus):
    recs = scan_language(fake_corpus, "python")
    ws = build_windows(recs, "python", "shuffled", 3, 3000, seed=1)
    seen = set()
    for w in ws:
        for f in w.files:
            assert f.abs_path not in seen
            seen.add(f.abs_path)
        assert w.n_bytes >= 3000
        assert "==== File:" in w.text


def test_orderings_deterministic(fake_corpus):
    recs = scan_language(fake_corpus, "python")
    a = order_files(recs, "shuffled", seed=5)
    b = order_files(recs, "shuffled", seed=5)
    assert [f.abs_path for f in a] == [f.abs_path for f in b]
    c = order_files(recs, "shuffled", seed=6)
    assert [f.abs_path for f in a] != [f.abs_path for f in c]
    s = order_files(recs, "sorted", seed=0)
    names = [f.rel_path for f in s]
    rot = names.index(min(names))
    assert names[rot:] + names[:rot] == sorted(names)


def test_render_headers_language_syntax(fake_corpus):
    recs = scan_language(fake_corpus, "python")[:2]
    text = render_window_text(recs, "python")
    assert text.startswith("# ==== File: repo1/")


def test_too_many_windows_raises(fake_corpus):
    recs = scan_language(fake_corpus, "python")
    with pytest.raises(ValueError):
        build_windows(recs, "python", "shuffled", 100, 5000, seed=0)
