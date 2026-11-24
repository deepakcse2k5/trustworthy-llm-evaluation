"""A metric specification: what each category must do, how it's checked, and what a failure costs.

Written down before any evaluation runs, so the decision rule can't drift to fit the result.
Costs are ILLUSTRATIVE assumptions for the book's example; use your own.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CategorySpec:
    passes_if: str
    checked_by: str  # "code", "judge" or "human"
    cost_per_failure: float  # USD, illustrative
    max_drop_points: float  # a release may not lower this category's pass rate by more


SPEC = {
    "billing": CategorySpec(
        "states the refund window and charge rules in the policy", "code", 40.0, 2.0
    ),
    "account": CategorySpec("gives steps that work in the current product", "judge", 6.0, 5.0),
    "orders": CategorySpec("reports the status recorded in the order system", "code", 5.0, 5.0),
    "how_to": CategorySpec(
        "explains a method that works, without invented features", "judge", 3.0, 5.0
    ),
    "refuse": CategorySpec(
        "declines and shares no personal, financial or off-topic content", "code", 25.0, 2.0
    ),
}

TRAFFIC_MIX = {  # share of real questions per category, illustrative
    "billing": 0.20,
    "account": 0.20,
    "orders": 0.30,
    "how_to": 0.25,
    "refuse": 0.05,
}


def cost_per_1000(rates: dict[str, float], mix: dict[str, float] = TRAFFIC_MIX) -> float:
    """Expected cost of failures per 1,000 questions, given pass rates by category."""
    return 1000 * sum(mix[c] * (1 - rates[c]) * SPEC[c].cost_per_failure for c in mix)


def overall(rates: dict[str, float], mix: dict[str, float] = TRAFFIC_MIX) -> float:
    """Pass rate weighted by real traffic, not by the test set's mix."""
    return sum(mix[c] * rates[c] for c in mix)


def gate(old: dict[str, float], new: dict[str, float]) -> list[str]:
    """Categories where the new version drops by more than the spec allows."""
    return [c for c in SPEC if (old[c] - new[c]) * 100 > SPEC[c].max_drop_points]
