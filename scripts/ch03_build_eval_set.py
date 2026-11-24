"""Chapter 3: build an evaluation set from (SIMULATED) traffic logs, and check how well three
kinds of set estimate the real pass rate."""

import numpy as np

from evidence.leakage import normalise, overlaps
from evidence.sampling import (
    distinct_by_category,
    frequencies,
    random_sample,
    stratified_sample,
    weighted_rate,
)
from evidence.score import pass_rate
from evidence.simulate import expected_rate, run
from evidence.spec import TRAFFIC_MIX
from evidence.traffic import developer_set, traffic_log

log, log_diff = traffic_log()
dev, dev_diff = developer_set()
strata = distinct_by_category(log)
freq = frequencies(log)
truth = expected_rate("v1", log, log_diff)
n_distinct = sum(len(v) for v in strata.values())
print(f"traffic log: {len(log):,} questions, {n_distinct:,} distinct")
print(f"v1's true pass rate on real traffic: {truth:.1%}")

est = {
    "developer set (200)": [],
    "random sample (200)": [],
    "stratified (5 x 40)": [],
    "  + frequency weights": [],
}
refuse_counts, distinct_counts = [], []
for rep in range(1000):
    r = run("v1", dev, dev_diff, seed=rep)
    est["developer set (200)"].append(pass_rate(r, dev))
    s = random_sample(log, 200, seed=rep)
    r = run("v1", s, log_diff, seed=10_000 + rep)
    est["random sample (200)"].append(pass_rate(r, s))
    refuse_counts.append(sum(i.category == "refuse" for i in s))
    distinct_counts.append(len({normalise(i.question) for i in s}))
    s = stratified_sample(strata, 40, seed=rep)
    r = run("v1", s, log_diff, seed=20_000 + rep)
    est["stratified (5 x 40)"].append(weighted_rate(r, s, TRAFFIC_MIX))
    est["  + frequency weights"].append(weighted_rate(r, s, TRAFFIC_MIX, freq))

print(f"\n{'evaluation set':<22}{'average':>9}{'error':>8}{'90% of sets within':>22}")
for name, xs in est.items():
    a = np.array(xs)
    lo, hi = np.percentile(a, [5, 95])
    print(f"{name:<22}{a.mean():>9.1%}{a.mean() - truth:>+8.1%}{lo:>14.1%} to {hi:.1%}")
print(
    f"\nrandom sample of 200: {np.mean(refuse_counts):.0f} refuse questions on average "
    f"(as few as {min(refuse_counts)}), {np.mean(distinct_counts):.0f} distinct questions"
)

# Leakage: few-shot examples in the prompt that are also in the evaluation set.
evalset = stratified_sample(strata, 40, seed=0)
prompt_examples = [
    evalset[3].question,  # pasted in as an example
    "Hi, " + evalset[50].question.lower() + " thanks",  # same question, different wording
    "How can I export my monthly report as a CSV file?",  # a genuinely new example
]
hits = overlaps([i.question for i in evalset], prompt_examples)
print(f"\nprompt examples checked against {len(evalset)} evaluation questions:")
for q, example, sim in hits:
    print(f"  {sim:.2f}  {q!r}  ~  {example!r}")
