"""Noise you didn't plan for: run-to-run variation in the system, and errors in the grader.

The grader here is SIMULATED as an LLM judge with fixed error rates. Real judges have error
rates you must measure, which is what judge_error_rates is for.
"""

from dataclasses import dataclass

import numpy as np

from evidence.dataset import Item
from evidence.simulate import run


def repeated_runs(
    version: str, items: list[Item], difficulty: dict[str, float], k: int, seed: int
) -> np.ndarray:
    """k runs of the same questions: a 0/1 grid with one row per question, one column per run."""
    cols = [run(version, items, difficulty, seed=seed + j) for j in range(k)]
    return np.array([[c[it.id] for c in cols] for it in items])


def consistency(grid: np.ndarray) -> dict[str, float]:
    """Share of questions that passed every run, failed every run, or changed between runs."""
    per_q = grid.mean(axis=1)
    return {
        "always pass": float(np.mean(per_q == 1)),
        "always fail": float(np.mean(per_q == 0)),
        "changes": float(np.mean((per_q > 0) & (per_q < 1))),
    }


def question_bootstrap(grid: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Interval for the pass rate from repeated runs: resample whole questions (rows), keeping
    all of a question's runs together, because runs of one question are not independent."""
    rng = np.random.default_rng(seed)
    per_q = grid.mean(axis=1)
    means = per_q[rng.integers(0, len(per_q), size=(reps, len(per_q)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Judge:
    """A simulated grader. sensitivity: chance it passes a good answer.
    specificity: chance it fails a bad answer."""

    sensitivity: float
    specificity: float

    def grade(self, truth: dict[str, int], seed: int) -> dict[str, int]:
        rng = np.random.default_rng(seed)
        out = {}
        for qid, good in truth.items():
            u = rng.random()
            out[qid] = int(u < self.sensitivity) if good else int(u >= self.specificity)
        return out


def agreement(a: list[int], b: list[int]) -> tuple[float, float]:
    """Raw agreement and Cohen's kappa (agreement beyond what chance alone would give)."""
    x, y = np.asarray(a), np.asarray(b)
    observed = float(np.mean(x == y))
    chance = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    kappa = (observed - chance) / (1 - chance) if chance < 1 else 1.0
    return observed, kappa


def judge_error_rates(human: list[int], judge: list[int]) -> tuple[float, float]:
    """Estimate a judge's sensitivity and specificity from questions a person also graded."""
    h, j = np.asarray(human), np.asarray(judge)
    return float(j[h == 1].mean()), float(1 - j[h == 0].mean())


def corrected_rate(observed: float, sensitivity: float, specificity: float) -> float:
    """The true pass rate implied by a judge's pass rate and its error rates.

    observed = sens * true + (1 - spec) * (1 - true), solved for true."""
    denom = sensitivity + specificity - 1
    if denom <= 0:
        raise ValueError("judge is no better than chance; its scores can't be corrected")
    return min(1.0, max(0.0, (observed + specificity - 1) / denom))


def corrected_interval(
    judged: list[int],
    calib_human: list[int],
    calib_judge: list[int],
    reps: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap interval for the corrected rate. Resamples the evaluation set and the
    calibration set separately, so uncertainty about the judge's error rates is included."""
    rng = np.random.default_rng(seed)
    j = np.asarray(judged)
    h, c = np.asarray(calib_human), np.asarray(calib_judge)
    out = []
    for _ in range(reps):
        obs = j[rng.integers(0, len(j), len(j))].mean()
        idx = rng.integers(0, len(h), len(h))
        hh, cc = h[idx], c[idx]
        if hh.min() == hh.max():  # a resample with no good or no bad answers: skip it
            continue
        sens, spec = cc[hh == 1].mean(), 1 - cc[hh == 0].mean()
        if sens + spec > 1:
            out.append(corrected_rate(obs, sens, spec))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)
