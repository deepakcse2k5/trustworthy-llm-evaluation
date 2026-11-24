"""Comparing two versions on the same questions: the paired test, an interval for the
difference, and a verdict that separates "statistically clear" from "large enough to matter"."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from evidence.dataset import Item
from evidence.spec import SPEC


def discordant(a: dict[str, int], b: dict[str, int], ids: list[str]) -> tuple[int, int]:
    """Questions only a passed, and questions only b passed. Questions both passed or both
    failed say nothing about which version is better."""
    only_a = sum(a[i] and not b[i] for i in ids)
    only_b = sum(b[i] and not a[i] for i in ids)
    return only_a, only_b


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided p-value: if the versions were equally good, each discordant question would be
    a coin flip. How surprising is a split this lopsided?"""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_diff_interval(
    a: dict[str, int], b: dict[str, int], ids: list[str], reps: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for rate(b) - rate(a): resample questions, keeping each question's pair."""
    rng = np.random.default_rng(seed)
    d = np.array([b[i] - a[i] for i in ids], dtype=float)
    means = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def unpaired_z(passed_a: int, n_a: int, passed_b: int, n_b: int) -> float:
    """Two-sided p-value from the usual two-proportion test, which ignores pairing. Shown for
    comparison: on shared questions it throws away the pairing."""
    pa, pb = passed_a / n_a, passed_b / n_b
    pool = (passed_a + passed_b) / (n_a + n_b)
    se = math.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = abs(pb - pa) / se
    return math.erfc(z / math.sqrt(2))


def verdict(lo: float, hi: float, margin: float) -> str:
    """Read an interval for b - a against the smallest change that matters (margin)."""
    if lo > margin:
        return "better"
    if hi < -margin:
        return "worse"
    if -margin <= lo and hi <= margin:
        return "no change that matters"
    if lo > 0:
        return "better, may be small"
    if hi < 0:
        return "worse, may be small"
    return "inconclusive"


@dataclass(frozen=True)
class Comparison:
    name: str
    n: int
    rate_a: float
    rate_b: float
    only_a: int
    only_b: int
    p: float
    lo: float
    hi: float
    verdict: str


def compare(
    a: dict[str, int], b: dict[str, int], items: list[Item], seed: int = 0
) -> list[Comparison]:
    """One row per category, judged against the spec's allowed drop, then one for all items."""
    groups = {c: [i.id for i in items if i.category == c] for c in sorted(SPEC)}
    groups["all"] = [i.id for i in items]
    rows = []
    for name, ids in groups.items():
        if not ids:
            continue
        only_a, only_b = discordant(a, b, ids)
        lo, hi = paired_diff_interval(a, b, ids, seed=seed)
        margin = SPEC[name].max_drop_points / 100 if name in SPEC else 0.02
        rows.append(
            Comparison(
                name,
                len(ids),
                sum(a[i] for i in ids) / len(ids),
                sum(b[i] for i in ids) / len(ids),
                only_a,
                only_b,
                mcnemar_exact(only_a, only_b),
                lo,
                hi,
                verdict(lo, hi, margin),
            )
        )
    return rows


def paired_stat_interval(
    a: dict[str, int],
    b: dict[str, int],
    items: list[Item],
    stat: Callable[[dict[str, float]], float],
    reps: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """stat(b) - stat(a), where stat turns category pass rates into one number (a weighted
    rate, a cost). Resamples questions with their pairs. Returns (estimate, low, high)."""
    rng = np.random.default_rng(seed)
    cats = sorted({i.category for i in items})
    code = np.array([cats.index(i.category) for i in items])
    xa = np.array([a[i.id] for i in items], dtype=float)
    xb = np.array([b[i.id] for i in items], dtype=float)

    def diff(idx: np.ndarray) -> float:
        ra, rb = {}, {}
        for k, c in enumerate(cats):
            m = code[idx] == k
            ra[c], rb[c] = xa[idx][m].mean(), xb[idx][m].mean()
        return stat(rb) - stat(ra)

    est = diff(np.arange(len(items)))
    boots = [diff(rng.integers(0, len(items), len(items))) for _ in range(reps)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def format_report(rows: list[Comparison], label_a: str = "A", label_b: str = "B") -> str:
    head = (
        f"{'':<9}{'n':>5}{label_a:>8}{label_b:>8}  only {label_a}  only {label_b}"
        f"{'p':>8}   {label_b} - {label_a} (95% interval)      verdict"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"{r.name:<9}{r.n:>5}{r.rate_a:>8.1%}{r.rate_b:>8.1%}{r.only_a:>8}{r.only_b:>8}"
            f"{r.p:>8.3f}   {r.rate_b - r.rate_a:+6.1%} ({r.lo:+.1%} to {r.hi:+.1%})"
            f"   {r.verdict}"
        )
    return "\n".join(lines)
