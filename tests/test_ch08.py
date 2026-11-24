import pytest

from evidence.dataset import Item
from evidence.slices import (
    allocate,
    benjamini_hochberg,
    holm,
    release_gate,
    sample_by_counts,
)


def test_holm_and_bh_known_values():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}
    assert holm(p) == pytest.approx({"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.5})
    assert benjamini_hochberg(p) == pytest.approx(
        {"a": 0.04, "c": 0.0533333, "b": 0.0533333, "d": 0.5}
    )


def test_corrections_never_lower_a_p_value():
    p = {str(i): (i + 1) / 20 for i in range(10)}
    for adj in (holm(p), benjamini_hochberg(p)):
        assert all(adj[k] >= p[k] for k in p)


def test_allocate_respects_total_and_floor():
    counts = allocate(1000, {"a": 8.0, "b": 1.0, "c": 0.1}, minimum=80)
    assert sum(counts.values()) == 1000
    assert min(counts.values()) >= 80
    assert counts["a"] > counts["b"] > counts["c"]


def test_sample_by_counts():
    log = [Item(f"{c}{i}", c, "", "") for c in "xy" for i in range(50)]
    s = sample_by_counts(log, {"x": 10, "y": 5}, seed=0)
    assert sum(i.category == "x" for i in s) == 10 and len(s) == 15


def _pair(drop):
    items = [Item(f"b{i}", "billing", "", "") for i in range(400)]
    a = {it.id: 1 for it in items}
    b = {it.id: int(i >= drop) for i, it in enumerate(items)}
    return items, a, b


@pytest.mark.parametrize(
    "drop, rule, ships",
    [
        (0, "estimate", True),
        (0, "harm", True),
        (0, "shown", True),
        (60, "estimate", False),  # 15-point drop
        (60, "harm", False),
        (60, "shown", False),
    ],
)
def test_release_gate_rules(drop, rule, ships):
    items, a, b = _pair(drop)
    assert release_gate(a, b, items, {"billing": 0.05}, rule)["billing"] is ships


def test_release_gate_rejects_unknown_rule():
    items, a, b = _pair(0)
    with pytest.raises(ValueError):
        release_gate(a, b, items, {"billing": 0.05}, "vibes")
