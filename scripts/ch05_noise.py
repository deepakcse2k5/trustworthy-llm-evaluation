"""Chapter 5: run-to-run noise and grader noise, on SIMULATED data where the truth is known."""

import numpy as np

from evidence.dataset import load
from evidence.intervals import wilson
from evidence.noise import (
    Judge,
    agreement,
    consistency,
    corrected_interval,
    corrected_rate,
    judge_error_rates,
    question_bootstrap,
    repeated_runs,
)
from evidence.sampling import random_sample
from evidence.simulate import expected_rate, item_difficulty, run
from evidence.traffic import traffic_log

# 1. The same questions, five runs.
items = load("data/eval_set.jsonl")
diff = item_difficulty(items)
billing = [i for i in items if i.category == "billing"]
grid = repeated_runs("v1", items, diff, k=5, seed=100)
print("v1 on Chapter 1's 200 questions, five runs:")
print("  overall per run: " + "  ".join(f"{m:.1%}" for m in grid.mean(axis=0)))
print("  " + "   ".join(f"{k} {v:.0%}" for k, v in consistency(grid).items()))
print("\nbilling, v1 minus v2, per run (50 questions):")
gaps = []
for j in range(5):
    a = run("v1", billing, diff, seed=100 + j)
    b = run("v2", billing, diff, seed=100 + j)
    gaps.append((sum(a.values()) - sum(b.values())) / 50)
print("  " + "  ".join(f"{g:+.0%}" for g in gaps))

# 2. Coverage when every question is run five times.
log, log_diff = traffic_log()
truth = expected_rate("v1", log, log_diff)
hits = {"one run, Wilson": 0, "5 runs, Wilson on 1,000": 0, "5 runs, question bootstrap": 0}
width = {k: [] for k in hits}
for rep in range(500):
    s = random_sample(log, 200, seed=rep)
    g = repeated_runs("v1", s, log_diff, k=5, seed=60_000 + 10 * rep)
    for name, (lo, hi) in {
        "one run, Wilson": wilson(int(g[:, 0].sum()), 200),
        "5 runs, Wilson on 1,000": wilson(int(g.sum()), g.size),
        "5 runs, question bootstrap": question_bootstrap(g, seed=rep),
    }.items():
        hits[name] += lo <= truth <= hi
        width[name].append(hi - lo)
print(f"\n200 questions from the log, true rate {truth:.1%}, 500 evaluations:")
for name in hits:
    print(f"  {name:<28} covers {hits[name] / 5:.1f}%   width {np.mean(width[name]) * 100:.1f} pts")

# 3. An LLM judge that is too lenient.
judge = Judge(sensitivity=0.97, specificity=0.60)
s = random_sample(log, 300, seed=11)
true_labels = run("v1", s, log_diff, seed=12)
first, second = judge.grade(true_labels, seed=13), judge.grade(true_labels, seed=14)
ids = [it.id for it in s]
h, j1, j2 = ([d[i] for i in ids] for d in (true_labels, first, second))
print("\none evaluation, 300 questions, graded by a simulated judge:")
print(f"  true pass rate of these answers   {np.mean(h):.1%}")
print(f"  judge's pass rate                  {np.mean(j1):.1%}")
print("  judge vs itself:  agreement {:.0%}, kappa {:.2f}".format(*agreement(j1, j2)))
print("  judge vs person:  agreement {:.0%}, kappa {:.2f}".format(*agreement(h, j1)))
calib, rest = slice(0, 100), slice(100, 300)  # a person also grades the first 100
sens, spec = judge_error_rates(h[calib], j1[calib])
obs = float(np.mean(j1[rest]))
lo, hi = corrected_interval(j1[rest], h[calib], j1[calib], seed=1)
print(f"  from 100 graded by a person: sensitivity {sens:.0%}, specificity {spec:.0%}")
print(
    f"  other 200: judge says {obs:.1%}, corrected {corrected_rate(obs, sens, spec):.1%} "
    f"(95% interval {lo:.1%} to {hi:.1%})"
)

# 4. Coverage of judge-graded intervals: is the truth inside?
hits2 = {"judge rate, Wilson": 0, "corrected, bootstrap": 0}
width2 = {k: [] for k in hits2}
for rep in range(300):
    s = random_sample(log, 300, seed=1000 + rep)
    t = run("v1", s, log_diff, seed=2000 + rep)
    g = judge.grade(t, seed=3000 + rep)
    h = [t[it.id] for it in s]
    j = [g[it.id] for it in s]
    for name, (lo, hi) in {
        "judge rate, Wilson": wilson(sum(j[100:]), 200),
        "corrected, bootstrap": corrected_interval(j[100:], h[:100], j[:100], reps=1000, seed=rep),
    }.items():
        hits2[name] += lo <= truth <= hi
        width2[name].append(hi - lo)
print(f"\njudge-graded evaluations, true rate {truth:.1%}, 300 evaluations:")
for name in hits2:
    print(
        f"  {name:<22} covers {hits2[name] / 3:.1f}%   width {np.mean(width2[name]) * 100:.1f} pts"
    )
