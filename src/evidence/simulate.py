"""A SIMULATED support assistant and grader, so every chapter runs offline.

Each version has a true pass rate per category. Each question also has a fixed difficulty,
shared by all versions, so hard questions tend to fail for every version, as in real systems.
Every run draws pass or fail per question, so scores move from run to run.
None of these numbers describes a real model.
"""

import math

import numpy as np

from evidence.dataset import Item

TRUE_RATES = {  # version: category -> true pass rate (simulation settings, not measurements)
    "v1": {"billing": 0.86, "account": 0.84, "orders": 0.80, "how_to": 0.76, "refuse": 0.85},
    "v2": {"billing": 0.70, "account": 0.90, "orders": 0.86, "how_to": 0.90, "refuse": 0.88},
}


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def item_difficulty(items: list[Item], seed: int = 1, spread: float = 1.0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {it.id: float(rng.normal(0, spread)) for it in items}


def pass_probability(version: str, item: Item, difficulty: dict[str, float]) -> float:
    z = _logit(TRUE_RATES[version][item.category]) - difficulty[item.id]
    return 1 / (1 + math.exp(-z))


def run(version: str, items: list[Item], difficulty: dict[str, float], seed: int) -> dict[str, int]:
    """One evaluation run: 1 if the (simulated) grader passes the answer, else 0."""
    rng = np.random.default_rng(seed)
    return {it.id: int(rng.random() < pass_probability(version, it, difficulty)) for it in items}


def expected_rate(version: str, items: list[Item], difficulty: dict[str, float]) -> float:
    """The long-run pass rate on these items: what infinitely many runs would average to."""
    return float(np.mean([pass_probability(version, it, difficulty) for it in items]))
