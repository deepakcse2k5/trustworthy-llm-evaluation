from dataclasses import replace

import numpy as np
import pytest

from evidence.judge import (
    Answer,
    BiasedJudge,
    calibration_report,
    human_prefers_first,
    longer_wins,
    make_answers,
    position_check,
    win_rate,
)


def _pairs(n=400):
    results = {f"q{i}": int(i % 5 != 0) for i in range(n)}
    old = make_answers(results, "v1", "A", mean_words=150, seed=1)
    new = make_answers(results, "v2", "A", mean_words=150, seed=2)
    return new, old


def test_answers_follow_the_truth():
    new, _ = _pairs()
    good = [a.quality for a in new.values() if a.good]
    bad = [a.quality for a in new.values() if not a.good]
    assert np.mean(good) > 0 > np.mean(bad)


def test_unbiased_judge_tracks_the_person():
    new, old = _pairs()
    fair = BiasedJudge("C", sharpness=50, position=0, length=0, self_pref=0)
    person = np.mean([human_prefers_first(new[q], old[q]) for q in new])
    assert win_rate(fair, new, old, seed=0, order="both") == pytest.approx(person, abs=0.03)


def test_position_bias_shows_up_and_both_orders_cancel_it():
    new, old = _pairs()
    biased = BiasedJudge("C", position=2.0, length=0, self_pref=0)
    first, _ = position_check(biased, new, old, seed=0)
    assert first > 0.65
    assert position_check(replace(biased, position=0.0), new, old, seed=0)[0] == pytest.approx(
        0.5, abs=0.05
    )
    assert win_rate(biased, new, old, 0, "new first") > win_rate(biased, new, old, 0, "old first")


def test_longer_wins_skips_ties():
    a = Answer("q", "v1", 1, 0.0, 100, "A")
    b = Answer("q", "v2", 1, 0.0, 200, "A")
    assert longer_wins([False, True], [a, a], [b, a]) == (1, 1)


def test_unknown_order_is_refused():
    new, old = _pairs(10)
    with pytest.raises(ValueError):
        win_rate(BiasedJudge("A"), new, old, seed=0, order="sideways")


def test_calibration_report_by_group():
    human = [1, 1, 1, 1, 0, 0, 0, 0]
    judge = [1, 1, 1, 0, 1, 0, 0, 0]
    groups = ["x", "x", "y", "y", "x", "x", "y", "y"]
    rows = {r.group: r for r in calibration_report(human, judge, groups)}
    assert rows["all"].sensitivity == 0.75 and rows["all"].specificity == 0.75
    assert rows["x"].sensitivity == 1.0 and rows["x"].specificity == 0.5
    assert rows["y"].sensitivity == 0.5 and rows["y"].specificity == 1.0
