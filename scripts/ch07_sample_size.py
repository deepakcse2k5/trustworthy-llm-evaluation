"""Chapter 7: planning how many questions an evaluation needs, checked on SIMULATED data."""

import numpy as np

from evidence.compare import discordant, mcnemar_exact
from evidence.intervals import wilson
from evidence.power import (
    detectable_effect,
    discordant_share,
    n_for_precision,
    n_paired,
    n_unpaired,
)
from evidence.sampling import random_sample
from evidence.simulate import expected_rate, run
from evidence.traffic import traffic_log

# 1. Precision of one pass rate.
print("questions needed for a 95% interval of +/- h around a pass rate")
print(f"{'rate':>6}{'h = 5 pts':>11}{'3 pts':>8}{'2 pts':>8}{'1 pt':>8}")
for p in (0.80, 0.90, 0.95):
    ns = [n_for_precision(p, h) for h in (0.05, 0.03, 0.02, 0.01)]
    print(f"{p:>6.0%}" + "".join(f"{n:>11,}" if i == 0 else f"{n:>8,}" for i, n in enumerate(ns)))
n = n_for_precision(0.80, 0.03)
lo, hi = wilson(round(0.8 * n), n)
print(f"check: Wilson at {round(0.8 * n)}/{n} is {lo:.1%} to {hi:.1%}")

# 2. Planning a comparison from a pilot.
log, log_diff = traffic_log()
true_gap = expected_rate("v2", log, log_diff) - expected_rate("v1", log, log_diff)
pilot = random_sample(log, 200, seed=500)
ids = [i.id for i in pilot]
print(f"\ntrue difference v2 - v1 on traffic: {true_gap:+.1%}; plan to detect 3 points")
plans = {}
for name, seed_b in (("noisy", 502), ("stable", 501)):
    r1 = run("v1", pilot, log_diff, seed=501)
    r2 = run("v2", pilot, log_diff, seed=seed_b)
    psi = discordant_share(r1, r2, ids)
    k = sum(r1[i] != r2[i] for i in ids)
    upper = wilson(k, len(ids))[1]
    plans[name] = n_paired(psi, 0.03)
    plans[name + ", cautious"] = n_paired(upper, 0.03)
    print(
        f"  {name:<7} pilot of 200: {psi:.1%} differ (95% interval up to {upper:.1%}) -> "
        f"{plans[name]:,} questions, or {plans[name + ', cautious']:,} using the upper end"
    )
print(f"  unpaired, two separate samples: {n_unpaired(0.81, 0.84):,} questions per version")

big = random_sample(log, 5000, seed=99)
for name, seed_b in (("noisy", 2), ("stable", 1)):
    share = discordant_share(
        run("v1", big, log_diff, seed=1), run("v2", big, log_diff, seed=seed_b), [i.id for i in big]
    )
    print(f"  {name:<7} on 5,000 questions instead: {share:.1%} differ")

# 3. Does the plan deliver 80% power?
for name, n in plans.items():
    hits = 0
    for rep in range(300):
        s = random_sample(log, n, seed=700 + rep)
        q = [i.id for i in s]
        r1 = run("v1", s, log_diff, seed=10_000 + rep)
        r2 = run("v2", s, log_diff, seed=(10_000 if "stable" in name else 20_000) + rep)
        hits += mcnemar_exact(*discordant(r1, r2, q)) < 0.05
    print(
        f"  check: {name + ' plan':<22}{n:>6,} questions, detects the difference in {hits / 3:.0f}%"
    )

# 4. What can a fixed budget detect?
print("\nsmallest difference detected with 80% power")
print(f"{'questions':>10}{'8% differ':>12}{'25% differ':>12}")
for n in (200, 500, 1000, 2000, 5000):
    row = [detectable_effect(n, psi) for psi in (0.08, 0.25)]
    print(f"{n:>10,}" + "".join(f"{d * 100:>11.1f}p" for d in row))

# 5. Peeking: v1 against itself, checked every 50 questions up to 1,000.
looks = list(range(50, 1001, 50))
four = [looks.index(k) for k in (250, 500, 750, 1000)]
final, every, planned = [], [], []
for rep in range(2000):
    s = random_sample(log, 1000, seed=5000 + rep)
    q = [i.id for i in s]
    r1 = run("v1", s, log_diff, seed=100_000 + rep)
    r1b = run("v1", s, log_diff, seed=200_000 + rep)
    ps = [mcnemar_exact(*discordant(r1, r1b, q[:k])) for k in looks]
    final.append(ps[-1])
    every.append(min(ps))
    planned.append(min(ps[i] for i in four))
final, every, planned = np.array(final), np.array(every), np.array(planned)
threshold = float(np.quantile(planned, 0.05))
print("\nv1 against itself, 2,000 evaluations (no real difference): share declaring a winner")
print(f"  one check at 1,000 questions, p < 0.05            {np.mean(final < 0.05):.1%}")
print(f"  check every 50, stop at the first p < 0.05        {np.mean(every < 0.05):.1%}")
print(f"  check at 250, 500, 750, 1,000, p < 0.05 each      {np.mean(planned < 0.05):.1%}")
label = f"  same four checks, p <= {threshold:.4f} each"
print(f"{label:<52}{np.mean(planned <= threshold):.1%}  (threshold set by simulation)")

# 6. What the planned looks cost and save when there is a real difference.
print("\nv2 against v1, real difference, 500 evaluations of up to 1,000 questions:")
for name in ("noisy", "stable"):
    fixed = early = 0
    used = []
    for rep in range(500):
        s = random_sample(log, 1000, seed=9000 + rep)
        q = [i.id for i in s]
        r1 = run("v1", s, log_diff, seed=300_000 + rep)
        r2 = run("v2", s, log_diff, seed=(300_000 if name == "stable" else 400_000) + rep)
        fixed += mcnemar_exact(*discordant(r1, r2, q)) < 0.05
        for k in (250, 500, 750, 1000):
            if mcnemar_exact(*discordant(r1, r2, q[:k])) <= threshold:
                early += 1
                used.append(k)
                break
        else:
            used.append(1000)
    print(
        f"  {name:<7} one check: detects {fixed / 5:.0f}% using 1,000;   four planned checks: "
        f"detects {early / 5:.0f}% using {np.mean(used):,.0f} on average"
    )
