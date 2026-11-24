import pytest

from evidence.dataset import Item
from evidence.intervals import bootstrap, stratified_bootstrap, wald, wilson


def test_wilson_known_values():
    lo, hi = wilson(42, 50)
    assert (round(lo, 3), round(hi, 3)) == (0.715, 0.917)
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(10, 10)  # a perfect score still gets a lower bound below 100%
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.722, abs=0.001)


def test_wald_collapses_at_extremes_and_wilson_does_not():
    assert wald(10, 10) == (1.0, 1.0)  # claims certainty from ten questions
    assert wilson(10, 10)[0] < 1.0


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    values = [1] * 42 + [0] * 8
    lo, hi = bootstrap(values, seed=3)
    assert (lo, hi) == bootstrap(values, seed=3)
    assert lo < 0.84 < hi


def test_stratified_bootstrap_weights_by_traffic():
    items = [Item(f"a{i}", "a", "", "") for i in range(20)] + [
        Item(f"b{i}", "b", "", "") for i in range(20)
    ]
    results = {f"a{i}": 1 for i in range(20)} | {f"b{i}": 0 for i in range(20)}
    assert stratified_bootstrap(results, items, {"a": 0.9, "b": 0.1}) == pytest.approx((0.9, 0.9))
