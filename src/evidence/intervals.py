"""How sure is a score? Confidence intervals for pass rates."""

import math
from collections import defaultdict

import numpy as np

from evidence.dataset import Item

Z95 = 1.959964


def wald(passed: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The textbook interval, p +/- z * sqrt(p(1-p)/n). Shown for comparison: it fails at
    small n and at rates near 0% or 100%."""
    p = passed / n
    half = z * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)


def wilson(passed: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval: the default for a single pass rate."""
    if n == 0:
        return 0.0, 1.0
    p = passed / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap(values: list[float], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap for a mean: resample the questions with replacement."""
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def stratified_bootstrap(
    results: dict[str, int],
    items: list[Item],
    mix: dict[str, float],
    reps: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Interval for a traffic-weighted rate from a stratified set: resample within each
    category, then weight the category rates by the traffic mix."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for it in items:
        groups[it.category].append(results[it.id])
    total = np.zeros(reps)
    for cat, vals in groups.items():
        x = np.asarray(vals, dtype=float)
        total += mix[cat] * x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1)
    lo, hi = np.percentile(total, [2.5, 97.5])
    return float(lo), float(hi)
