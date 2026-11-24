"""A model as judge: SIMULATED answers, a SIMULATED judge with known biases, and the checks
that find those biases in a real judge.

Every bias strength below is a setting chosen for the book's examples, not a measurement of
any real model. Measure your own judge with the checks; don't assume these numbers.
"""

import math
from dataclasses import dataclass

import numpy as np

from evidence.intervals import wilson
from evidence.noise import agreement


@dataclass(frozen=True)
class Answer:
    qid: str
    version: str
    good: int  # what a careful person would say: 1 pass, 0 fail
    quality: float  # finer-grained quality, used for pairwise preferences
    words: int
    family: str  # which model family wrote it


def make_answers(
    results: dict[str, int], version: str, family: str, mean_words: float, seed: int
) -> dict[str, Answer]:
    """One answer per question. Quality follows pass or fail; length is unrelated to quality."""
    rng = np.random.default_rng(seed)
    out = {}
    for qid, good in results.items():
        q = rng.normal(1.0 if good else -1.0, 0.7)
        words = int(np.exp(rng.normal(math.log(mean_words), 0.35)))
        out[qid] = Answer(qid, version, good, float(q), words, family)
    return out


def human_prefers_first(a: Answer, b: Answer) -> bool:
    """The reference: a careful person prefers the better answer."""
    return a.quality > b.quality


@dataclass(frozen=True)
class BiasedJudge:
    """A simulated LLM judge. Each bias is a push, on the log-odds scale, toward:
    position: the answer shown first; length: the longer answer (per doubling of length);
    self_pref: answers from its own model family. sharpness: how much real quality counts."""

    family: str
    sharpness: float = 1.5
    position: float = 0.5
    length: float = 0.8
    self_pref: float = 0.6
    threshold: float = 0.0  # for pass/fail grading

    def _push(self, ans: Answer) -> float:
        return self.length * math.log2(ans.words / 150) + self.self_pref * (
            ans.family == self.family
        )

    def prefers_first(self, a: Answer, b: Answer, rng: np.random.Generator) -> bool:
        z = self.sharpness * (a.quality - b.quality) + self.position + self._push(a) - self._push(b)
        return rng.random() < 1 / (1 + math.exp(-z))

    def grade(self, ans: Answer, rng: np.random.Generator) -> int:
        z = self.sharpness * ans.quality + self._push(ans) - self.threshold
        return int(rng.random() < 1 / (1 + math.exp(-z)))


def win_rate(
    judge: BiasedJudge,
    new: dict[str, Answer],
    old: dict[str, Answer],
    seed: int,
    order: str = "new first",
) -> float:
    """Share of questions where the judge prefers the new answer.

    order: "new first", "old first", "random", or "both" (ask twice, swapping the order;
    a win counts 1, a split verdict 0.5)."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for qid in new:
        n, o = new[qid], old[qid]
        if order == "new first":
            total += judge.prefers_first(n, o, rng)
        elif order == "old first":
            total += not judge.prefers_first(o, n, rng)
        elif order == "random":
            total += (
                judge.prefers_first(n, o, rng)
                if rng.random() < 0.5
                else (not judge.prefers_first(o, n, rng))
            )
        elif order == "both":
            total += (judge.prefers_first(n, o, rng) + (not judge.prefers_first(o, n, rng))) / 2
        else:
            raise ValueError(order)
    return total / len(new)


def position_check(
    judge: BiasedJudge, new: dict[str, Answer], old: dict[str, Answer], seed: int
) -> tuple[float, float]:
    """Ask every pair in both orders. Returns (share of verdicts that went to the answer shown
    first, share of pairs whose verdict changed when the order was swapped). Without position
    bias the first is near 50%; the second is never zero, because judges are also noisy."""
    rng = np.random.default_rng(seed)
    first_wins = flips = 0
    for q in new:
        a = judge.prefers_first(new[q], old[q], rng)
        b = judge.prefers_first(old[q], new[q], rng)
        first_wins += a + b
        flips += a == b
    return first_wins / (2 * len(new)), flips / len(new)


def longer_wins(prefer: list[bool], firsts: list[Answer], seconds: list[Answer]) -> tuple[int, int]:
    """Among pairs of different length: (times the longer answer was preferred, pairs)."""
    wins = n = 0
    for p, a, b in zip(prefer, firsts, seconds, strict=True):
        if a.words == b.words:
            continue
        n += 1
        wins += p == (a.words > b.words)
    return wins, n


@dataclass(frozen=True)
class GroupCalibration:
    group: str
    n: int
    sensitivity: float
    sens_interval: tuple[float, float]
    specificity: float
    spec_interval: tuple[float, float]
    kappa: float


def calibration_report(
    human: list[int], judge: list[int], groups: list[str]
) -> list[GroupCalibration]:
    """The judge's error rates overall and within each group (length band, model family,
    category...), each with a Wilson interval, from answers a person also graded."""
    h, j, g = np.asarray(human), np.asarray(judge), np.asarray(groups)
    rows = []
    for name in ["all", *sorted(set(groups))]:
        m = np.ones(len(h), bool) if name == "all" else g == name
        hh, jj = h[m], j[m]
        good, bad = int((hh == 1).sum()), int((hh == 0).sum())
        tp, tn = int(((hh == 1) & (jj == 1)).sum()), int(((hh == 0) & (jj == 0)).sum())
        rows.append(
            GroupCalibration(
                name,
                int(m.sum()),
                tp / good if good else float("nan"),
                wilson(tp, good),
                tn / bad if bad else float("nan"),
                wilson(tn, bad),
                agreement(list(hh), list(jj))[1],
            )
        )
    return rows
