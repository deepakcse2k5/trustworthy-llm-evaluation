import pytest

from evidence.compare import Comparison
from evidence.report import MemoInputs, describe_range, memo, one_in, points, recommend


def test_points_and_ranges():
    assert points(0.031) == "3 points higher"
    assert points(-0.01) == "1 point lower"
    assert points(0.004) == "no change"
    assert describe_range(-0.200, -0.097) == "between 10 and 20 points lower"
    assert describe_range(0.0, 0.064) == "between no change and 6 points higher"
    assert describe_range(-0.04, 0.04) == "between 4 points lower and 4 points higher"


@pytest.mark.parametrize("rate, text", [(0.0039, "about 1 in 260"), (0.07, "about 1 in 14")])
def test_one_in(rate, text):
    assert one_in(rate) == text


def _row(name, verdict, lo=-0.01, hi=0.01):
    return Comparison(name, 100, 0.8, 0.8, 1, 1, 1.0, lo, hi, verdict)


def _inputs(rows, cost=(0.0, -10.0, 10.0)):
    return MemoInputs("T", "D?", "v1", "v2", rows, cost, 0.03, ["m"], ["u"], ["n"])


def test_recommendation_blocks_on_a_worse_category_or_cost():
    ok = [_row("billing", "no change that matters"), _row("all", "inconclusive")]
    assert recommend(_inputs(ok)).startswith("Ship v2")
    bad = [_row("billing", "worse", -0.2, -0.1), _row("all", "better")]
    assert "billing" in recommend(_inputs(bad))
    assert "cost" in recommend(_inputs(ok, cost=(500.0, 100.0, 900.0)))


def test_memo_leads_with_the_decision():
    text = memo(_inputs([_row("refuse", "inconclusive"), _row("all", "inconclusive")]))
    lines = text.splitlines()
    assert lines[0] == "# T" and "Recommendation" in text.split("## What we found")[0]
    assert "| refusals |" in text
