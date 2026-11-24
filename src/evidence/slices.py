"""Many metrics, many slices: corrections for many tests, comparing across different mixes,
and release gates that the evaluation set can actually check."""

import random
from collections import defaultdict

from evidence.compare import paired_diff_interval
from evidence.dataset import Item


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values. Compare them with 0.05 and the chance of ANY false alarm
    across all the tests stays at 5%."""
    order = sorted(pvalues, key=pvalues.get)
    m, out, running = len(order), {}, 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        out[name] = running
    return out


def benjamini_hochberg(pvalues: dict[str, float]) -> dict[str, float]:
    """BH-adjusted p-values (q-values). Compare them with 0.05 and, on average, at most 5% of
    the results you flag are false alarms. Less strict than Holm; for finding leads."""
    order = sorted(pvalues, key=pvalues.get, reverse=True)
    m, out, running = len(order), {}, 1.0
    for rank, name in enumerate(order):
        running = min(running, pvalues[name] * m / (m - rank))
        out[name] = running
    return out


def rates_by_category(results: dict[str, int], items: list[Item]) -> dict[str, float]:
    passed: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for it in items:
        passed[it.category] += results[it.id]
        total[it.category] += 1
    return {c: passed[c] / total[c] for c in total}


def allocate(total: int, weights: dict[str, float], minimum: int = 0) -> dict[str, int]:
    """Split `total` questions across categories in proportion to weights, with a floor."""
    counts = {c: minimum for c in weights}
    rest = total - minimum * len(weights)
    s = sum(weights.values())
    for c, w in weights.items():
        counts[c] += int(rest * w / s)
    leftover = total - sum(counts.values())
    for c in sorted(weights, key=weights.get, reverse=True)[:leftover]:
        counts[c] += 1
    return counts


def sample_by_counts(log: list[Item], counts: dict[str, int], seed: int) -> list[Item]:
    """A random sample of logged questions with a chosen number from each category."""
    rng = random.Random(seed)
    by_cat: dict[str, list[Item]] = defaultdict(list)
    for it in log:
        by_cat[it.category].append(it)
    return [it for c in sorted(counts) for it in rng.sample(by_cat[c], counts[c])]


def release_gate(
    a: dict[str, int],
    b: dict[str, int],
    items: list[Item],
    margins: dict[str, float],
    rule: str,
    seed: int = 0,
) -> dict[str, bool]:
    """category -> True if the new version b passes the gate in that category.

    rule "estimate":  pass unless the observed drop is larger than the margin
    rule "harm":      pass unless the whole interval for b - a is below -margin
    rule "shown":     pass only if the whole interval for b - a is above -margin
    """
    out = {}
    for c, margin in margins.items():
        ids = [i.id for i in items if i.category == c]
        drop = (sum(a[i] for i in ids) - sum(b[i] for i in ids)) / len(ids)
        lo, hi = paired_diff_interval(a, b, ids, reps=1000, seed=seed)
        if rule == "estimate":
            out[c] = drop <= margin
        elif rule == "harm":
            out[c] = hi >= -margin
        elif rule == "shown":
            out[c] = lo > -margin
        else:
            raise ValueError(rule)
    return out
