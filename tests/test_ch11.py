import numpy as np
import pytest

from evidence.agents import Agent, Task, check_trajectory, pass_all_k, pass_at_k
from evidence.rag import (
    Generator,
    Retriever,
    failure_breakdown,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics():
    ranked = ["x", "r1", "y", "r2"]
    assert recall_at_k(ranked, ["r1", "r2"], 2) == 0.5
    assert recall_at_k(ranked, ["r1", "r2"], 4) == 1.0
    assert hit_at_k(ranked, ["r2"], 3) == 0
    assert reciprocal_rank(ranked, ["r1", "r2"]) == 0.5
    assert reciprocal_rank(ranked, ["z"]) == 0.0


def test_sharper_retriever_ranks_higher():
    rel = [["d00001"]] * 500
    sharp, flat = Retriever("s", 1.0, 0.8), Retriever("f", 1.0, 0.2)
    mrr = [
        np.mean(
            [
                reciprocal_rank(r.rank(q, 100, np.random.default_rng(i)), q)
                for i, q in enumerate(rel)
            ]
        )
        for r in (sharp, flat)
    ]
    assert mrr[0] > mrr[1]


def test_generator_and_breakdown():
    g = Generator(context=1, faithful=1.0, memory=0.0)
    rng = np.random.default_rng(0)
    assert g.answer(["r"], ["r"], rng) == (1, 1)
    assert g.answer(["x", "r"], ["r"], rng) == (0, 0)
    parts = failure_breakdown([1, 1, 0, 0], [1, 0, 0, 1])
    assert parts == pytest.approx({k: 0.25 for k in parts})
    assert sum(parts.values()) == pytest.approx(1.0)


def test_trajectory_rules():
    good = ["lookup_order", "verify_identity", "check_policy", "issue_refund", "reply"]
    assert all(check_trajectory(good).values())
    skipped = ["lookup_order", "check_policy", "issue_refund", "reply"]
    assert check_trajectory(skipped)["verify before refund"] is False
    no_refund = ["lookup_order", "reply"]
    assert all(check_trajectory(no_refund).values())
    assert check_trajectory(["lookup_order"] * 9)["at most 8 calls"] is False


def test_agent_follows_its_habits():
    careful = Agent("c", skill=5, skip_verify=0.0, skip_policy=0.0, loop=0.0)
    steps, right = careful.attempt(Task("t", True, 0.0), np.random.default_rng(0))
    assert steps == ["lookup_order", "verify_identity", "check_policy", "issue_refund", "reply"]
    assert right


def test_pass_at_k_and_pass_all_k():
    s = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=bool)
    assert pass_at_k(s) == pytest.approx(2 / 3)
    assert pass_all_k(s) == pytest.approx(1 / 3)
