"""Chapter 11: evaluating a RAG pipeline and an agent, on SIMULATED components."""

import numpy as np

from evidence.agents import Agent, Task, check_trajectory, pass_all_k, pass_at_k
from evidence.compare import mcnemar_exact
from evidence.rag import (
    Generator,
    Retriever,
    failure_breakdown,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)

# 1. Retrieval metrics for two retrievers on the same 1,000 questions.
rng = np.random.default_rng(0)
N_DOCS = 50_000
questions = [
    [f"d{int(x):05d}" for x in rng.choice(N_DOCS, size=rng.integers(1, 3), replace=False)]
    for _ in range(1000)
]
retrievers = [Retriever("keyword", found=0.80, sharpness=0.6), Retriever("dense", 0.95, 0.3)]
ranked = {
    r.name: [
        r.rank(rel, N_DOCS, np.random.default_rng(1000 + i)) for i, rel in enumerate(questions)
    ]
    for r in retrievers
}
print("retrieval metrics, 1,000 questions:")
print(f"  {'retriever':<10}{'recall@20':>11}{'recall@3':>10}{'hit@3':>8}{'hit@10':>8}{'MRR':>7}")
for r in retrievers:
    lists = ranked[r.name]
    print(
        f"  {r.name:<10}"
        f"{np.mean([recall_at_k(x, q, 20) for x, q in zip(lists, questions, strict=True)]):>11.1%}"
        f"{np.mean([recall_at_k(x, q, 3) for x, q in zip(lists, questions, strict=True)]):>10.1%}"
        f"{np.mean([hit_at_k(x, q, 3) for x, q in zip(lists, questions, strict=True)]):>8.1%}"
        f"{np.mean([hit_at_k(x, q, 10) for x, q in zip(lists, questions, strict=True)]):>8.1%}"
        f"{np.mean([reciprocal_rank(x, q) for x, q in zip(lists, questions, strict=True)]):>7.2f}"
    )

# 2. End to end: the same retrievers feeding generators that read 3 or 10 documents.
generators = {"reads 3": Generator(context=3), "reads 10": Generator(context=10, faithful=0.85)}
print("\nend to end, share of questions answered correctly:")
results = {}
for gname, g in generators.items():
    for r in retrievers:
        out = [
            g.answer(x, q, np.random.default_rng(5000 + i))
            for i, (x, q) in enumerate(zip(ranked[r.name], questions, strict=True))
        ]
        results[(gname, r.name)] = out
        correct = [c for c, _ in out]
        print(f"  generator {gname:<9} + {r.name:<8} {np.mean(correct):.1%}")
    a = [c for c, _ in results[(gname, "keyword")]]
    b = [c for c, _ in results[(gname, "dense")]]
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    p = mcnemar_exact(only_a, only_b)
    print(f"    dense - keyword: {np.mean(b) - np.mean(a):+.1%}, paired p = {p:.3f}")

print("\nwhere the answers ended up (generator reads 3):")
for r in retrievers:
    c, g = zip(*results[("reads 3", r.name)], strict=True)
    parts = failure_breakdown(list(c), list(g))
    print(f"  {r.name}")
    for k, v in parts.items():
        print(f"    {k:<30}{v:>6.1%}")

# 3. Agents: a refund task, five runs of each of 300 tasks.
trng = np.random.default_rng(7)
tasks = [Task(f"t{i:03d}", bool(trng.random() < 0.6), float(trng.normal())) for i in range(300)]
agents = [
    Agent("careful", skill=2.5, skip_verify=0.02, skip_policy=0.05, loop=0.05),
    Agent("fast", skill=3.0, skip_verify=0.15, skip_policy=0.10, loop=0.02),
]
K = 5
print(f"\nagents, 300 refund tasks, {K} runs each:")
header = ["success", f"pass@{K}", f"pass^{K}", "calls", "no verify", "no policy", "wrong refund"]
print(f"  {'agent':<9}" + "".join(f"{h:>13}" for h in header))
for ag in agents:
    arng = np.random.default_rng(11)
    succ = np.zeros((len(tasks), K), dtype=bool)
    calls, unverified, unchecked, wrong_refunds, runs = [], 0, 0, 0, 0
    for i, t in enumerate(tasks):
        for k in range(K):
            steps, right = ag.attempt(t, arng)
            succ[i, k] = right
            rules = check_trajectory(steps)
            calls.append(len(steps))
            unverified += not rules["verify before refund"]
            unchecked += not rules["policy before refund"]
            wrong_refunds += ("issue_refund" in steps) and not t.refundable
            runs += 1
    row = [
        f"{succ.mean():.1%}",
        f"{pass_at_k(succ):.1%}",
        f"{pass_all_k(succ):.1%}",
        f"{np.mean(calls):.1f}",
        f"{unverified / runs:.1%}",
        f"{unchecked / runs:.1%}",
        f"{wrong_refunds / runs:.1%}",
    ]
    print(f"  {ag.name:<9}" + "".join(f"{x:>13}" for x in row))
