"""Evidence in production: online A/B tests, drift monitoring and alarms for rare guardrail
events. All traffic here is SIMULATED; every rate and effect size is an assumption for the
book's examples.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Conversations:
    """One row per conversation. user: who had it; arm: 0 control, 1 treatment."""

    user: np.ndarray
    arm: np.ndarray
    resolved: np.ndarray
    harmful: np.ndarray  # a guardrail event, such as promising a refund the policy forbids


def simulate_ab(
    n_users: int,
    lift: float,
    seed: int,
    base: float = 0.70,
    harm: tuple[float, float] = (0.004, 0.004),
    drop_low_users: float = 0.0,
) -> Conversations:
    """Users are randomised to an arm and keep it for every conversation they have.

    lift: change in log-odds of resolution in the treatment arm.
    drop_low_users: share of treatment-arm users with the lowest resolution propensity whose
    data never reaches the log (a logging bug in the new version)."""
    rng = np.random.default_rng(seed)
    logit = math.log(base / (1 - base))
    users, arms, res, bad = [], [], [], []
    for u in range(n_users):
        arm = int(rng.random() < 0.5)
        propensity = rng.normal(0.0, 1.0)  # some users' problems are simply harder
        if arm and drop_low_users and propensity < -1 and rng.random() < drop_low_users / 0.16:
            continue
        n = 1 + rng.poisson(math.exp(rng.normal(math.log(2), 0.8)))  # a few heavy users
        p = 1 / (1 + math.exp(-(logit + propensity + lift * arm)))
        users += [u] * n
        arms += [arm] * n
        res += list(rng.random(n) < p)
        bad += list(rng.random(n) < harm[arm])
    return Conversations(np.array(users), np.array(arms), np.array(res), np.array(bad))


def srm_pvalue(n_control: int, n_treatment: int, share: float = 0.5) -> float:
    """Sample ratio mismatch: p-value for the split of USERS between arms, against the share
    the randomiser was set to. A tiny p-value means the experiment is broken; stop and debug."""
    n = n_control + n_treatment
    expected = np.array([n * (1 - share), n * share])
    chi2 = float((((np.array([n_control, n_treatment]) - expected) ** 2) / expected).sum())
    return math.erfc(math.sqrt(chi2 / 2))


def naive_diff(c: Conversations, metric: str = "resolved") -> tuple[float, float, float]:
    """Difference in a per-conversation rate, treating every conversation as independent.
    Shown for comparison: users with many conversations make it too narrow."""
    y = getattr(c, metric)
    a, b = y[c.arm == 0], y[c.arm == 1]
    d = b.mean() - a.mean()
    se = math.sqrt(a.mean() * (1 - a.mean()) / len(a) + b.mean() * (1 - b.mean()) / len(b))
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se)


def user_bootstrap_diff(
    c: Conversations, metric: str = "resolved", reps: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Difference in a per-conversation rate, resampling USERS (the unit that was randomised)
    with all their conversations."""
    rng = np.random.default_rng(seed)
    y = getattr(c, metric).astype(float)
    out = []
    totals = {}
    for arm in (0, 1):
        m = c.arm == arm
        _, inv = np.unique(c.user[m], return_inverse=True)
        num = np.bincount(inv, weights=y[m])
        den = np.bincount(inv).astype(float)
        totals[arm] = (num, den)
    est = totals[1][0].sum() / totals[1][1].sum() - totals[0][0].sum() / totals[0][1].sum()
    for _ in range(reps):
        rates = []
        for arm in (0, 1):
            num, den = totals[arm]
            idx = rng.integers(0, len(num), len(num))
            rates.append(num[idx].sum() / den[idx].sum())
        out.append(rates[1] - rates[0])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def psi(expected: dict[str, float], observed: dict[str, float]) -> float:
    """Population stability index between two category mixes: 0 for identical; above about
    0.1 is a shift worth looking at, above about 0.25 a large one (common rules of thumb)."""
    total = 0.0
    for c in expected:
        e, o = max(expected[c], 1e-6), max(observed.get(c, 0.0), 1e-6)
        total += (o - e) * math.log(o / e)
    return total


def poisson_alarm_threshold(expected: float, false_alarm: float) -> int:
    """Smallest count k with P(count >= k) <= false_alarm when counts are Poisson(expected)."""
    k, term = 0, math.exp(-expected)
    cdf = term
    while 1 - cdf > false_alarm:
        k += 1
        term *= expected / k
        cdf += term
    return k + 1
