"""Chapter 1: three ways to decide between two versions, and why two of them mislead.
SIMULATED assistant and grader; see src/evidence/simulate.py."""

import numpy as np

from evidence.dataset import load, make_eval_set, save
from evidence.score import by_category, pass_rate
from evidence.simulate import expected_rate, item_difficulty, run

items = make_eval_set()
save(items, "data/eval_set.jsonl")
items = load("data/eval_set.jsonl")
diff = item_difficulty(items)
print(f"evaluation set: {len(items)} questions")

# 1. The demo: ten questions someone tries by hand, all from features they care about.
demo = [it for it in items if it.category == "how_to"][:6] + [
    it for it in items if it.category == "account"
][:4]
for v in ("v1", "v2"):
    r = run(v, demo, diff, seed=42)
    print(f"demo, {v}: {sum(r.values())}/10")

# 2. The full set, one run each.
print()
full = {v: run(v, items, diff, seed=7) for v in ("v1", "v2")}
for v in ("v1", "v2"):
    print(f"full set, {v}: {pass_rate(full[v], items):.1%}")
print(f"{'category':<10}{'v1':>10}{'v2':>10}")
c1, c2 = by_category(full["v1"], items), by_category(full["v2"], items)
for cat in c1:
    (p1, n), (p2, _) = c1[cat], c2[cat]
    print(f"{cat:<10}{p1:>5}/{n:<4}{p2:>5}/{n:<4}")

# 3. The same version, run again.
print()
scores = [pass_rate(run("v1", items, diff, seed=s), items) for s in range(100, 105)]
print("v1, five more runs:", ", ".join(f"{s:.1%}" for s in scores))

# 4. How often would each method pick v2?
rng = np.random.default_rng(0)
demo_pool = [it for it in items if it.category in ("how_to", "account")]
wins, trials = 0, 10_000
for t in range(trials):
    pick = list(rng.choice(len(demo_pool), size=10, replace=False))
    d = [demo_pool[i] for i in pick]
    a = sum(run("v1", d, diff, seed=2 * t).values())
    b = sum(run("v2", d, diff, seed=2 * t + 1).values())
    wins += b > a
print()
print(f"10-question demos where v2 looks better: {wins / trials:.0%} of {trials:,}")
billing = [it for it in items if it.category == "billing"]
for v in ("v1", "v2"):
    print(
        f"long-run rate, {v}: overall {expected_rate(v, items, diff):.1%}, "
        f"billing {expected_rate(v, billing, diff):.1%}"
    )
