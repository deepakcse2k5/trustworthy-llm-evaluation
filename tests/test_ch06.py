import json
import subprocess
import sys

import pytest

from evidence.compare import (
    compare,
    discordant,
    mcnemar_exact,
    paired_diff_interval,
    paired_stat_interval,
    unpaired_z,
    verdict,
)
from evidence.dataset import Item, save
from evidence.spec import overall


def test_discordant_ignores_ties():
    a = {"q1": 1, "q2": 1, "q3": 0, "q4": 0}
    b = {"q1": 1, "q2": 0, "q3": 1, "q4": 0}
    assert discordant(a, b, list(a)) == (1, 1)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(9, 0) == pytest.approx(2 / 2**9)
    assert mcnemar_exact(9, 9) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_paired_interval_brackets_the_difference():
    a = {f"q{i}": 1 for i in range(40)}
    b = {f"q{i}": int(i >= 8) for i in range(40)}
    lo, hi = paired_diff_interval(a, b, list(a), seed=1)
    assert lo < -0.2 < hi < 0


def test_unpaired_z_is_symmetric_and_flat_at_equal_rates():
    assert unpaired_z(42, 50, 33, 50) == pytest.approx(unpaired_z(33, 50, 42, 50))
    assert unpaired_z(30, 50, 30, 50) == 1.0


@pytest.mark.parametrize(
    "lo, hi, expected",
    [
        (0.03, 0.08, "better"),
        (-0.08, -0.03, "worse"),
        (-0.01, 0.015, "no change that matters"),
        (0.005, 0.06, "better, may be small"),
        (-0.06, -0.005, "worse, may be small"),
        (-0.04, 0.06, "inconclusive"),
    ],
)
def test_verdict(lo, hi, expected):
    assert verdict(lo, hi, 0.02) == expected


def _items_and_results():
    items = [Item(f"b{i}", "billing", "", "") for i in range(50)] + [
        Item(f"o{i}", "orders", "", "") for i in range(50)
    ]
    a = {it.id: 1 for it in items}
    b = {it.id: int(not (it.category == "billing" and int(it.id[1:]) < 20)) for it in items}
    return items, a, b


def test_compare_flags_the_category_that_dropped():
    items, a, b = _items_and_results()
    rows = {r.name: r for r in compare(a, b, items)}
    assert rows["billing"].verdict == "worse"
    assert rows["orders"].verdict == "no change that matters"
    assert rows["all"].n == 100


def test_paired_stat_interval_weights_categories():
    items, a, b = _items_and_results()
    est, lo, hi = paired_stat_interval(
        a, b, items, lambda r: overall(r, {"billing": 0.5, "orders": 0.5})
    )
    assert est == pytest.approx(-0.2)
    assert lo <= est <= hi


def test_command_line(tmp_path):
    items, a, b = _items_and_results()
    save(items, tmp_path / "items.jsonl")
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "evidence",
            "compare",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "--items",
            str(tmp_path / "items.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "billing" in out and "worse" in out
