import numpy as np
import pytest

from evidence.noise import (
    Judge,
    agreement,
    consistency,
    corrected_rate,
    judge_error_rates,
    question_bootstrap,
)


def test_consistency_counts_questions_not_runs():
    grid = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0]])
    assert consistency(grid) == {"always pass": 0.25, "always fail": 0.25, "changes": 0.5}


def test_question_bootstrap_is_reproducible_and_brackets_the_mean():
    grid = np.array([[1, 1, 0]] * 30 + [[0, 0, 0]] * 10)
    lo, hi = question_bootstrap(grid, seed=2)
    assert (lo, hi) == question_bootstrap(grid, seed=2)
    assert lo < grid.mean() < hi


def test_perfect_judge_reproduces_the_truth():
    truth = {str(i): i % 3 != 0 for i in range(60)}
    graded = Judge(1.0, 1.0).grade({k: int(v) for k, v in truth.items()}, seed=0)
    assert graded == {k: int(v) for k, v in truth.items()}


def test_agreement_and_error_rates():
    assert agreement([1, 0, 1, 0], [1, 0, 1, 0]) == (1.0, 1.0)
    human = [1, 1, 1, 1, 0, 0, 0, 0]
    judge = [1, 1, 1, 0, 1, 1, 0, 0]
    assert judge_error_rates(human, judge) == (0.75, 0.5)


def test_corrected_rate_inverts_the_judges_errors():
    true, sens, spec = 0.8, 0.97, 0.6
    observed = sens * true + (1 - spec) * (1 - true)
    assert corrected_rate(observed, sens, spec) == pytest.approx(true)
    with pytest.raises(ValueError):
        corrected_rate(0.5, 0.5, 0.5)
