"""Evaluating retrieval-augmented generation: retrieval metrics, end-to-end scores, and a
breakdown of where each failure happened. The retrievers and the generator are SIMULATED;
their settings are assumptions for the book's examples.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Retriever:
    """found: chance that the retriever finds a relevant document at all.
    sharpness: chance, at each rank, that a found document sits there rather than lower,
    so higher sharpness puts found documents nearer the top. Found documents that would
    land below rank 20 are lost."""

    name: str
    found: float
    sharpness: float

    def rank(self, relevant: list[str], n_docs: int, rng: np.random.Generator) -> list[str]:
        """The top 20 document ids, best first."""
        slots: list[str | None] = [None] * 20
        for doc in relevant:
            if rng.random() >= self.found:
                continue
            pos = int(rng.geometric(self.sharpness)) - 1
            while pos < 20 and slots[pos] is not None:
                pos += 1
            if pos < 20:
                slots[pos] = doc
        out = []
        for s in slots:
            out.append(s if s is not None else f"d{int(rng.integers(0, n_docs)):05d}")
        return out


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked: list[str], relevant: list[str], k: int) -> int:
    """1 if at least one relevant document is in the top k."""
    return int(bool(set(ranked[:k]) & set(relevant)))


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    """1 / position of the first relevant document (0 if none in the list)."""
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1 / i
    return 0.0


@dataclass(frozen=True)
class Generator:
    """Reads the top `context` documents. With a relevant one in view, answers correctly with
    chance `faithful`; without, it can still be right from memory with chance `memory`."""

    context: int = 3
    faithful: float = 0.9
    memory: float = 0.3

    def answer(
        self, ranked: list[str], relevant: list[str], rng: np.random.Generator
    ) -> tuple[int, int]:
        """(correct, grounded): grounded means a relevant document was in the context."""
        grounded = hit_at_k(ranked, relevant, self.context)
        p = self.faithful if grounded else self.memory
        return int(rng.random() < p), grounded


def failure_breakdown(correct: list[int], grounded: list[int]) -> dict[str, float]:
    """Where did each question end up? Shares of all questions."""
    c, g = np.asarray(correct), np.asarray(grounded)
    return {
        "correct, supported": float(np.mean((c == 1) & (g == 1))),
        "correct, unsupported": float(np.mean((c == 1) & (g == 0))),
        "wrong, retrieval missed": float(np.mean((c == 0) & (g == 0))),
        "wrong, despite the document": float(np.mean((c == 0) & (g == 1))),
    }
