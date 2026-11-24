import pytest

from evidence.labels import (
    LabelItem,
    Rater,
    accuracy,
    adjudicate,
    fleiss_kappa,
    leave_one_out,
    majority,
    make_items,
)


def test_perfect_rater_and_perfect_agreement():
    items = make_items(200, seed=0)
    perfect = Rater("p", skill=1.0)
    labs = [perfect.label(items, clarity=1.0, seed=s) for s in range(3)]
    assert accuracy(labs[0], items) == 1.0
    assert fleiss_kappa(labs) == pytest.approx(1.0)


def test_fleiss_kappa_near_zero_for_coin_flips():
    items = make_items(2000, borderline=1.0, seed=1)
    coin = Rater("c", skill=0.95)
    labs = [coin.label(items, clarity=0.0, seed=s) for s in range(4)]
    assert abs(fleiss_kappa(labs)) < 0.05


def test_guidelines_help_on_borderline_answers():
    items = make_items(2000, borderline=1.0, seed=2)
    r = Rater("r", skill=0.95)
    assert accuracy(r.label(items, 0.9, seed=0), items) > accuracy(
        r.label(items, 0.1, seed=0), items
    )


def test_lenient_rater_passes_borderline_fails():
    items = [LabelItem(f"g{i}", 0, True) for i in range(500)]
    assert accuracy(Rater("e", 0.95, lenience=0.7).label(items, 1.0, seed=0), items) < 0.4


def test_majority_adjudicate_and_leave_one_out():
    a, b, c = {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}
    assert majority([a, b, c]) == {"x": 1, "y": 1}
    assert adjudicate(a, b, {"x": 0, "y": 0}) == {"x": 1, "y": 0}
    loo = leave_one_out({"a": a, "b": b, "c": c, "d": b})
    assert loo["b"] >= loo["a"]
