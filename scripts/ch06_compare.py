"""Chapter 6: comparing two versions, on SIMULATED data where the true difference is known."""

import json
from pathlib import Path

from evidence.compare import (
    compare,
    discordant,
    format_report,
    mcnemar_exact,
    paired_diff_interval,
    paired_stat_interval,
    unpaired_z,
)
from evidence.dataset import load, save
from evidence.sampling import random_sample
from evidence.simulate import expected_rate, item_difficulty, run
from evidence.spec import cost_per_1000, overall
from evidence.traffic import traffic_log


def report(label, a, b, ids):
    only_a, only_b = discordant(a, b, ids)
    pa, pb = sum(a[i] for i in ids), sum(b[i] for i in ids)
    lo, hi = paired_diff_interval(a, b, ids)
    print(f"{label}: v1 {pa}/{len(ids)}, v2 {pb}/{len(ids)}")
    print(f"  only v1 passed {only_a}, only v2 passed {only_b}")
    print(
        f"  paired test p = {mcnemar_exact(only_a, only_b):.3f}   unpaired test p = "
        f"{unpaired_z(pa, len(ids), pb, len(ids)):.3f}"
    )
    print(f"  v2 - v1 = {(pb - pa) / len(ids):+.1%}, 95% interval {lo:+.1%} to {hi:+.1%}")


# 1. Chapter 1's comparison, done properly.
items = load("data/eval_set.jsonl")
diff = item_difficulty(items)
a, b = run("v1", items, diff, seed=7), run("v2", items, diff, seed=7)
report("Chapter 1 billing", a, b, [i.id for i in items if i.category == "billing"])
report("Chapter 1 overall", a, b, [i.id for i in items])

# 2. What pairing buys. Two kinds of system:
#    noisy:  each version's answer is drawn afresh (Chapter 5's run-to-run noise)
#    stable: both versions share each question's random draw, so an answer changes only
#            when the version change matters for that question
log, log_diff = traffic_log()
t1, t2 = expected_rate("v1", log, log_diff), expected_rate("v2", log, log_diff)
print(f"\ntrue rates on traffic: v1 {t1:.1%}, v2 {t2:.1%}, difference {t2 - t1:+.1%}")
print("share of 300 evaluations that detect v2 - v1 at p < 0.05")
print(f"{'':>10}{'noisy system':^20}{'stable system':^20}{'v1 vs v1':>10}")
print(
    f"{'questions':>10}{'paired':>10}{'unpaired':>10}{'paired':>10}{'unpaired':>10}{'(noisy)':>10}"
)
for n in (200, 400, 800, 1600):
    hits = [0, 0, 0, 0, 0]
    for rep in range(300):
        s = random_sample(log, n, seed=rep)
        ids = [i.id for i in s]
        r1 = run("v1", s, log_diff, seed=10_000 + rep)
        r2 = run("v2", s, log_diff, seed=20_000 + rep)
        r2s = run("v2", s, log_diff, seed=10_000 + rep)
        r1b = run("v1", s, log_diff, seed=30_000 + rep)
        k1 = sum(r1.values())
        hits[0] += mcnemar_exact(*discordant(r1, r2, ids)) < 0.05
        hits[1] += unpaired_z(k1, n, sum(r2.values()), n) < 0.05
        hits[2] += mcnemar_exact(*discordant(r1, r2s, ids)) < 0.05
        hits[3] += unpaired_z(k1, n, sum(r2s.values()), n) < 0.05
        hits[4] += mcnemar_exact(*discordant(r1, r1b, ids)) < 0.05
    print(f"{n:>10}" + "".join(f"{h / 300:>10.0%}" for h in hits))

# 3. Statistically clear is not the same as good: 1,000 logged questions, stable system.
s = random_sample(log, 1000, seed=21)
r1, r2 = run("v1", s, log_diff, seed=5), run("v2", s, log_diff, seed=5)
print("\nv1 vs v2 on 1,000 questions from the log:")
print(format_report(compare(r1, r2, s), "v1", "v2"))
est, lo, hi = paired_stat_interval(r1, r2, s, overall)
print(f"traffic-weighted pass rate, v2 - v1: {est:+.1%} (95% interval {lo:+.1%} to {hi:+.1%})")
est, lo, hi = paired_stat_interval(r1, r2, s, cost_per_1000)
print(f"cost of failures per 1,000 questions, v2 - v1: ${est:+,.0f} (${lo:+,.0f} to ${hi:+,.0f})")

# 4. The same comparison from files, as the command line runs it.
out = Path("data")
save(s, out / "ch06_items.jsonl")
(out / "ch06_v1.json").write_text(json.dumps(r1))
(out / "ch06_v2.json").write_text(json.dumps(r2))
print("\nwrote data/ch06_items.jsonl, data/ch06_v1.json, data/ch06_v2.json")
