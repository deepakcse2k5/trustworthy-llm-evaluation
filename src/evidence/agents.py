"""Evaluating agents: task success, step-by-step rules, and reliability over repeated runs.
The agents are SIMULATED: each tries a refund task with a few tools, and its habits are
settings for the book's examples."""

from dataclasses import dataclass

import numpy as np

TOOLS = ("lookup_order", "verify_identity", "check_policy", "issue_refund", "reply")


@dataclass(frozen=True)
class Task:
    id: str
    refundable: bool  # what the policy says for this order
    difficulty: float  # higher is harder; shared by every agent and every run


@dataclass(frozen=True)
class Agent:
    """skip_verify / skip_policy: chance of leaving out that step.
    loop: chance, after each step, of repeating the last tool call.
    skill: how well it decides once it has the facts (log-odds scale)."""

    name: str
    skill: float
    skip_verify: float
    skip_policy: float
    loop: float

    def attempt(self, task: Task, rng: np.random.Generator) -> tuple[list[str], bool]:
        """Run once. Returns the tool calls made and whether the refund decision was right."""
        steps = ["lookup_order"]
        if rng.random() >= self.skip_verify:
            steps.append("verify_identity")
        checked = rng.random() >= self.skip_policy
        if checked:
            steps.append("check_policy")
        z = self.skill - task.difficulty - (0 if checked else 1.5)
        right = rng.random() < 1 / (1 + np.exp(-z))
        refunds = task.refundable == right  # a right decision refunds exactly the refundable
        if refunds:
            steps.append("issue_refund")
        steps.append("reply")
        out = []
        for s in steps:
            out.append(s)
            while rng.random() < self.loop and len(out) < 15:
                out.append(s)
        return out, bool(right)


RULES = {
    "verify before refund": lambda s: (
        "issue_refund" not in s
        or ("verify_identity" in s and s.index("verify_identity") < s.index("issue_refund"))
    ),
    "policy before refund": lambda s: (
        "issue_refund" not in s
        or ("check_policy" in s and s.index("check_policy") < s.index("issue_refund"))
    ),
    "at most 8 calls": lambda s: len(s) <= 8,
}


def check_trajectory(steps: list[str]) -> dict[str, bool]:
    """Which step-by-step rules does this trajectory keep?"""
    return {name: rule(steps) for name, rule in RULES.items()}


def pass_at_k(successes: np.ndarray) -> float:
    """Share of tasks solved in at least one of k runs (rows are tasks, columns runs)."""
    return float(np.mean(successes.any(axis=1)))


def pass_all_k(successes: np.ndarray) -> float:
    """Share of tasks solved in every one of k runs: what a customer relying on it gets."""
    return float(np.mean(successes.all(axis=1)))
