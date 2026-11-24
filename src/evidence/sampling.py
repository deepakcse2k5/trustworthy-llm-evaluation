"""Drawing an evaluation set from logs, and estimating the traffic-weighted pass rate from it."""

import random
from collections import Counter, defaultdict

from evidence.dataset import Item
from evidence.leakage import normalise


def random_sample(log: list[Item], n: int, seed: int) -> list[Item]:
    return random.Random(seed).sample(log, n)


def distinct_by_category(log: list[Item]) -> dict[str, list[Item]]:
    """Each distinct question once (ignoring case, punctuation and greetings), by category."""
    groups: dict[str, list[Item]] = defaultdict(list)
    seen: set[str] = set()
    for it in log:
        key = normalise(it.question)
        if key not in seen:
            seen.add(key)
            groups[it.category].append(it)
    return dict(groups)


def stratified_sample(strata: dict[str, list[Item]], per_category: int, seed: int) -> list[Item]:
    """The same number of distinct questions from every category, so small categories are
    measured too and no question appears twice."""
    rng = random.Random(seed)
    return [it for cat in sorted(strata) for it in rng.sample(strata[cat], per_category)]


def frequencies(log: list[Item]) -> Counter:
    """How often each distinct question appears in the log."""
    return Counter(normalise(it.question) for it in log)


def weighted_rate(
    results: dict[str, int],
    items: list[Item],
    mix: dict[str, float],
    freq: Counter | None = None,
) -> float:
    """Pass rate per category, weighted by each category's share of real traffic.

    With freq, each question also counts in proportion to how often customers ask it,
    so the estimate describes real traffic rather than the list of distinct questions.
    """
    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    for it in items:
        w = freq[normalise(it.question)] if freq else 1.0
        num[it.category] += w * results[it.id]
        den[it.category] += w
    return sum(mix[c] * num[c] / den[c] for c in den)
