"""Human labels you can rely on: SIMULATED raters, agreement statistics, and ways to combine
labels. Rater skills and the share of borderline answers are settings for the book's
examples, not measurements of real people.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelItem:
    id: str
    good: int  # the answer an expert panel would reach after discussion
    borderline: bool  # reasonable people could disagree without clear guidelines


def make_items(n: int, pass_rate: float = 0.8, borderline: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        LabelItem(f"a{i:04d}", int(rng.random() < pass_rate), bool(rng.random() < borderline))
        for i in range(n)
    ]


@dataclass(frozen=True)
class Rater:
    """skill: chance of the right label on a clear answer. lenience: chance of passing a
    borderline answer that should fail, whatever the guidelines say (a habit, not confusion)."""

    name: str
    skill: float = 0.95
    lenience: float = 0.0

    def label(self, items: list[LabelItem], clarity: float, seed: int) -> dict[str, int]:
        """clarity (0 to 1): how well the guidelines settle borderline cases."""
        rng = np.random.default_rng(seed)
        out = {}
        for it in items:
            right = self.skill if not it.borderline else 0.5 + clarity * (self.skill - 0.5)
            if it.borderline and not it.good and rng.random() < self.lenience:
                out[it.id] = 1
            elif rng.random() < right:
                out[it.id] = it.good
            else:
                out[it.id] = 1 - it.good
        return out


def fleiss_kappa(labels: list[dict[str, int]]) -> float:
    """Agreement among several raters who all labelled the same items, beyond chance."""
    ids = list(labels[0])
    k = len(labels)
    ones = np.array([sum(lab[i] for lab in labels) for i in ids], dtype=float)
    per_item = (ones * (ones - 1) + (k - ones) * (k - ones - 1)) / (k * (k - 1))
    p1 = ones.sum() / (k * len(ids))
    chance = p1**2 + (1 - p1) ** 2
    return float((per_item.mean() - chance) / (1 - chance))


def majority(labels: list[dict[str, int]]) -> dict[str, int]:
    """Majority vote; use an odd number of raters."""
    return {i: int(Counter(lab[i] for lab in labels).most_common(1)[0][0]) for i in labels[0]}


def adjudicate(a: dict[str, int], b: dict[str, int], expert: dict[str, int]) -> dict[str, int]:
    """Two raters; where they disagree, an expert decides."""
    return {i: a[i] if a[i] == b[i] else expert[i] for i in a}


def leave_one_out(labels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Each rater's agreement with the majority of the OTHER raters: low values flag a rater
    who reads the guidelines differently."""
    out = {}
    for name, lab in labels.items():
        others = [v for n, v in labels.items() if n != name]
        ref = majority(others) if len(others) % 2 else majority(others[:-1])
        out[name] = float(np.mean([lab[i] == ref[i] for i in lab]))
    return out


def accuracy(labels: dict[str, int], items: list[LabelItem]) -> float:
    return float(np.mean([labels[it.id] == it.good for it in items]))
