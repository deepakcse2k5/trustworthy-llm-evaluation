"""Chapter 8: many tests, many slices, and release gates, on SIMULATED data."""

import numpy as np

from evidence.compare import discordant, mcnemar_exact
from evidence.sampling import random_sample
from evidence.simulate import TRUE_RATES, expected_rate, run
from evidence.slices import (
    allocate,
    benjamini_hochberg,
    holm,
    release_gate,
    sample_by_counts,
)
from evidence.spec import SPEC, TRAFFIC_MIX, overall
from evidence.traffic import traffic_log

# A third version for this chapter only: v1 improved by 4 points in every category.
TRUE_RATES["v3"] = {c: r + 0.04 for c, r in TRUE_RATES["v1"].items()}
# And a fourth: v1 with billing 4 points worse and everything else unchanged.
TRUE_RATES["v4"] = {**TRUE_RATES["v1"], "billing": TRUE_RATES["v1"]["billing"] - 0.04}

log, log_diff = traffic_log()


def slices(items):
    """Ten slices: each category, split by whether the question was typed casually
    (lower case, a greeting or a sign-off) or cleanly."""
    out = {}
    for it in items:
        q = it.question
        style = (
            "casual" if q[0].islower() or q.startswith("hi, ") or q.endswith(" thanks") else "clean"
        )
        out.setdefault(f"{it.category}/{style}", []).append(it.id)
    return out


def pvalues(a, b, groups):
    return {g: mcnemar_exact(*discordant(a, b, ids)) for g, ids in groups.items()}


# 1. v1 against itself: how often does at least one slice look "significant"?
counts = {"raw": [0, 0], "holm": [0, 0], "bh": [0, 0]}
for rep in range(1000):
    s = random_sample(log, 1000, seed=rep)
    r1, r1b = run("v1", s, log_diff, seed=10_000 + rep), run("v1", s, log_diff, seed=20_000 + rep)
    cats = {c: [i.id for i in s if i.category == c] for c in TRAFFIC_MIX}
    for k, groups in enumerate((cats, slices(s))):
        p = pvalues(r1, r1b, groups)
        counts["raw"][k] += min(p.values()) < 0.05
        counts["holm"][k] += min(holm(p).values()) < 0.05
        counts["bh"][k] += min(benjamini_hochberg(p).values()) < 0.05
print("v1 against itself, 1,000 evaluations of 1,000 questions: at least one false alarm")
print(f"{'':<22}{'5 categories':>14}{'10 slices':>11}")
for name, label in (("raw", "p < 0.05"), ("holm", "Holm p < 0.05"), ("bh", "BH q < 0.05")):
    print(f"  {label:<20}{counts[name][0] / 10:>13.1f}%{counts[name][1] / 10:>10.1f}%")

# 2. One real comparison, v1 against v2, ten slices.
s = random_sample(log, 1000, seed=77)
r1, r2 = run("v1", s, log_diff, seed=1), run("v2", s, log_diff, seed=2)
groups = slices(s)
p = pvalues(r1, r2, groups)
h, q = holm(p), benjamini_hochberg(p)
print("\nv1 vs v2, 1,000 questions, ten slices:")
print(f"  {'slice':<16}{'n':>5}{'v1':>7}{'v2':>7}{'p':>8}{'Holm':>8}{'BH':>8}")
for g in sorted(groups):
    ids = groups[g]
    ra, rb = np.mean([r1[i] for i in ids]), np.mean([r2[i] for i in ids])
    print(f"  {g:<16}{len(ids):>5}{ra:>7.0%}{rb:>7.0%}{p[g]:>8.3f}{h[g]:>8.3f}{q[g]:>8.3f}")

# 3. Simpson's paradox: last month's v1 against this month's v3, on different traffic.
#    True rates (no sampling noise), so the reversal can't be blamed on chance.
last_month = {"billing": 0.40, "account": 0.30, "how_to": 0.16, "orders": 0.08, "refuse": 0.06}
this_month = {"billing": 0.06, "account": 0.06, "how_to": 0.16, "orders": 0.54, "refuse": 0.18}
true = {
    v: {c: expected_rate(v, [i for i in log if i.category == c], log_diff) for c in TRAFFIC_MIX}
    for v in ("v1", "v3")
}
print("\nlast month v1, this month v3 (v3 is v1 plus 4 points everywhere, by design), true rates:")
print(f"  {'category':<10}{'v1':>8}{'v3':>8}{'last mix':>10}{'this mix':>10}")
for c in sorted(TRAFFIC_MIX):
    print(
        f"  {c:<10}{true['v1'][c]:>8.1%}{true['v3'][c]:>8.1%}"
        f"{last_month[c]:>10.0%}{this_month[c]:>10.0%}"
    )
print(
    f"  as reported: v1 last month {overall(true['v1'], last_month):.1%}, "
    f"v3 this month {overall(true['v3'], this_month):.1%}"
)
print(
    f"  on the same mix (Chapter 2's traffic): v1 {overall(true['v1']):.1%}, "
    f"v3 {overall(true['v3']):.1%}"
)

# 4. Release gates: how often does each rule let each version ship? (noisy setting)
margins = {c: SPEC[c].max_drop_points / 100 for c in SPEC}
risk = {c: TRAFFIC_MIX[c] * SPEC[c].cost_per_failure for c in SPEC}
designs = {
    "traffic-shaped": allocate(1000, TRAFFIC_MIX),
    "risk-allocated": allocate(1000, risk, minimum=80),
}
for name, d in designs.items():
    print(f"\n{name} set of 1,000: " + ", ".join(f"{c} {n}" for c, n in sorted(d.items())))
scenarios = {"no change": "v1", "v3 all +4": "v3", "v4 billing -4": "v4", "v2 billing -13": "v2"}
rules = ("estimate", "harm", "shown")
ships = {}  # (margin setting, design, rule, version) -> count
for name, d in designs.items():
    for version in scenarios.values():
        for rep in range(200):
            s = sample_by_counts(log, d, seed=3000 + rep)
            a = run("v1", s, log_diff, seed=40_000 + rep)
            b = run(version, s, log_diff, seed=50_000 + rep)
            for wide in (False, True):
                m = {c: (0.08 if wide else margins[c]) for c in margins}
                for rule in rules:
                    key = (wide, name, rule, version)
                    ships[key] = ships.get(key, 0) + all(
                        release_gate(a, b, s, m, rule, rep).values()
                    )
for wide in (False, True):
    title = "margins widened to 8 points" if wide else "spec margins (2 or 5 points)"
    print(f"\nshare of 200 evaluations where the new version ships, {title}:")
    print(f"  {'':<16}{'rule':<10}" + "".join(f"{k:>16}" for k in scenarios))
    for name in designs:
        for rule in rules:
            row = [ships[(wide, name, rule, v)] / 200 for v in scenarios.values()]
            print(f"  {name:<16}{rule:<10}" + "".join(f"{x:>16.0%}" for x in row))
