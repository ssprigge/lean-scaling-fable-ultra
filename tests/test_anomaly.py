from leanscale.anomaly import (
    mutate_arith,
    mutate_comparison,
    mutate_identifier_swap,
    mutate_off_by_one,
    mutations_for_line,
    pick_sites,
)


def test_mutate_comparison():
    assert mutate_comparison("if a <= b:") == "if a < b:"
    assert mutate_comparison("if a < b:") == "if a <= b:"
    assert mutate_comparison("if a == b:") == "if a != b:"
    assert mutate_comparison("plain line") is None


def test_mutate_arith():
    assert mutate_arith("total = x + y") == "total = x - y"
    assert mutate_arith("i++") is None
    assert mutate_arith("x += 1") is None


def test_mutate_off_by_one():
    assert mutate_off_by_one("for i in range(10):") == "for i in range(11):"
    # 0/1 and identifier-adjacent digits are not mutated
    assert mutate_off_by_one("x = a1 + b0") is None
    assert mutate_off_by_one("v = 3.14") is None


def test_mutate_identifier_swap():
    out = mutate_identifier_swap("result = compute(alpha, beta)")
    assert out is not None
    assert "compute" in out  # keywords/short ids preserved, swap happened
    assert out != "result = compute(alpha, beta)"


def test_mutations_skip_comments():
    line = "# total = x + y and a < b"
    from leanscale.anomaly import is_code_line

    assert not is_code_line(line, "python")
    assert not is_code_line("-- foo <= bar baz quux", "lean")
    assert is_code_line("theorem add_comm : a + b = b + a := by", "lean")


def test_pick_sites_depths_monotone():
    text = "".join(
        f"value_{i} = compute_{i}(alpha_{i} + beta_{i})\n" for i in range(2000)
    )
    sites = pick_sites(text, "python", [1000, 20000, 50000])
    assert len(sites) == 3
    assert sites[0].depth_bytes >= 1000
    assert sites[1].depth_bytes >= 20000
    assert sites[2].depth_bytes >= 50000
    for s in sites:
        assert s.mutations
