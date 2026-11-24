import numpy as np
import pytest

from evidence.production import (
    Conversations,
    naive_diff,
    poisson_alarm_threshold,
    psi,
    simulate_ab,
    srm_pvalue,
    user_bootstrap_diff,
)


def test_srm():
    assert srm_pvalue(5000, 5000) == pytest.approx(1.0)
    assert srm_pvalue(10_067, 9_561) < 0.001


def test_user_bootstrap_matches_estimate_and_is_wider_than_naive_with_clusters():
    c = simulate_ab(3000, lift=0.0, seed=1)
    est, lo, hi = user_bootstrap_diff(c, seed=0)
    nd, nlo, nhi = naive_diff(c)
    assert est == pytest.approx(nd)
    assert hi - lo > nhi - nlo


def test_bootstrap_on_tiny_example():
    c = Conversations(
        user=np.array([0, 0, 1, 2, 2, 3]),
        arm=np.array([0, 0, 0, 1, 1, 1]),
        resolved=np.array([1, 0, 1, 1, 1, 0]),
        harmful=np.zeros(6, dtype=bool),
    )
    est, lo, hi = user_bootstrap_diff(c, seed=0, reps=200)
    assert est == pytest.approx(2 / 3 - 2 / 3)
    assert lo <= est <= hi


def test_logging_bug_shows_in_srm():
    c = simulate_ab(60_000, lift=0.0, seed=3, drop_low_users=0.04)
    n = [len(np.unique(c.user[c.arm == a])) for a in (0, 1)]
    assert srm_pvalue(*n) < 0.01


def test_psi():
    mix = {"a": 0.5, "b": 0.5}
    assert psi(mix, mix) == pytest.approx(0.0)
    assert psi(mix, {"a": 0.9, "b": 0.1}) > 0.25


def test_poisson_alarm_threshold():
    k = poisson_alarm_threshold(10, 0.001)
    rng = np.random.default_rng(0)
    assert np.mean(rng.poisson(10, 200_000) >= k) <= 0.0015
    assert np.mean(rng.poisson(10, 200_000) >= k - 1) > 0.001
