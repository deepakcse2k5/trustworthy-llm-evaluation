"""Find evaluation questions that also appear, or nearly appear, in prompts or training data."""

import re


def normalise(text: str) -> str:
    text = re.sub(r"^(hi|hello),?\s+", "", text.lower().strip())
    text = re.sub(r"\s+thanks\.?$", "", text)
    return re.sub(r"[^a-z0-9#]+", " ", text).strip()


def _words(text: str) -> set[str]:
    return set(normalise(text).split())


def overlaps(eval_questions: list[str], other: list[str], threshold: float = 0.8) -> list[tuple]:
    """(eval question, other text, similarity) for pairs at or above the word-overlap threshold."""
    found = []
    other_sets = [(o, _words(o)) for o in other]
    for q in eval_questions:
        qw = _words(q)
        for o, ow in other_sets:
            sim = len(qw & ow) / len(qw | ow) if qw | ow else 0.0
            if sim >= threshold:
                found.append((q, o, round(sim, 2)))
    return found
