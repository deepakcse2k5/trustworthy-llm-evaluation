"""Chapter 4: confidence intervals, checked against SIMULATED data where the truth is known."""

import numpy as np

from evidence.dataset import load
from evidence.intervals import bootstrap, stratified_bootstrap, wald, wilson
from evidence.sampling import distinct_by_category, random_sample, stratified_sample, weighted_rate
from evidence.score import by_category
from evidence.simulate import expected_rate, item_difficulty, run
from evidence.spec import TRAFFIC_MIX
from evidence.traffic import traffic_log

# 1. Coverage: how often does a "95%" interval contain the true rate?
rng = np.random.default_rng(1)
print("share of 95% intervals that contain the true rate (10,000 tests each)")
print(f"{'n':>5}{'true rate':>11}{'Wald':>8}{'Wilson':>8}{'bootstrap':>11}")
for n in (10, 40, 200):
    for p in (0.80, 0.95):
        hits = {"wald": 0, "wilson": 0, "boot": 0}
        for _ in range(10_000):
            k = int(rng.binomial(n, p))
            lo, hi = wald(k, n)
            hits["wald"] += lo <= p <= hi
            lo, hi = wilson(k, n)
            hits["wilson"] += lo <= p <= hi
            boot = rng.binomial(n, k / n, size=1000) / n  # bootstrap of a pass rate
            lo, hi = np.percentile(boot, [2.5, 97.5])
            hits["boot"] += lo <= p <= hi
        print(
            f"{n:>5}{p:>11.0%}{hits['wald'] / 1e4:>8.1%}{hits['wilson'] / 1e4:>8.1%}"
            f"{hits['boot'] / 1e4:>11.1%}"
        )

# 2. The same check on the support assistant: one run on a random sample of the log.
log, log_diff = traffic_log()
truth = expected_rate("v1", log, log_diff)
covered, widths = 0, []
for rep in range(1000):
    s = random_sample(log, 200, seed=rep)
    r = run("v1", s, log_diff, seed=50_000 + rep)
    lo, hi = wilson(sum(r.values()), len(s))
    covered += lo <= truth <= hi
    widths.append(hi - lo)
print(
    f"\nsupport assistant, 200 questions, one run: Wilson interval contains the true "
    f"{truth:.1%} in {covered / 10:.1f}% of 1,000 evaluations (width about "
    f"{np.mean(widths) * 100:.0f} points)"
)

# 3. One real evaluation, reported with intervals.
strata = distinct_by_category(log)
s = stratified_sample(strata, 40, seed=0)
r = run("v1", s, log_diff, seed=7)
print("\none evaluation of v1, stratified set of 200:")
for cat, (k, n) in sorted(by_category(r, s).items()):
    lo, hi = wilson(k, n)
    print(f"  {cat:<9}{k:>3}/{n}  {k / n:>6.1%}   95% interval {lo:.1%} to {hi:.1%}")
lo, hi = stratified_bootstrap(r, s, TRAFFIC_MIX)
print(
    f"  weighted overall {weighted_rate(r, s, TRAFFIC_MIX):.1%}   95% interval {lo:.1%} to {hi:.1%}"
)

# 4. Chapter 1's billing result, with intervals.
items = load("data/eval_set.jsonl")
diff = item_difficulty(items)
billing = [i for i in items if i.category == "billing"]
print("\nChapter 1 billing, 50 questions each:")
for v in ("v1", "v2"):
    k = sum(run(v, items, diff, seed=7)[i.id] for i in billing)
    lo, hi = wilson(k, 50)
    blo, bhi = bootstrap([run(v, items, diff, seed=7)[i.id] for i in billing])
    print(f"  {v}: {k}/50   Wilson {lo:.1%} to {hi:.1%}   bootstrap {blo:.1%} to {bhi:.1%}")
