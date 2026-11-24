"""Turn per-question results into scores: overall and by category."""

from collections import defaultdict

from evidence.dataset import Item


def pass_rate(results: dict[str, int], items: list[Item]) -> float:
    return sum(results[it.id] for it in items) / len(items)


def by_category(results: dict[str, int], items: list[Item]) -> dict[str, tuple[int, int]]:
    """category -> (passed, total)."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for it in items:
        out[it.category][0] += results[it.id]
        out[it.category][1] += 1
    return {c: (p, n) for c, (p, n) in out.items()}
