"""Chapter 12: an online A/B test, drift monitoring and rare-event alarms, on SIMULATED traffic."""

import math

import numpy as np

from evidence.intervals import wilson
from evidence.production import (
    naive_diff,
    poisson_alarm_threshold,
    psi,
    simulate_ab,
    srm_pvalue,
    user_bootstrap_diff,
)
from evidence.simulate import TRUE_RATES
from evidence.spec import TRAFFIC_MIX

LIFT = 0.10  # log-odds; the true effect in points is computed below
z = np.random.default_rng(0).normal(size=1_000_000)
l0 = math.log(0.7 / 0.3)
true_lift = float(np.mean(1 / (1 + np.exp(-(l0 + z + LIFT))) - 1 / (1 + np.exp(-(l0 + z)))))


def users_per_arm(c):
    return [len(np.unique(c.user[c.arm == a])) for a in (0, 1)]


# 1. A/A tests: no real difference. How often does each analysis claim one?
naive_fp = user_fp = 0
for rep in range(300):
    c = simulate_ab(4000, lift=0.0, seed=rep)
    _, lo, hi = naive_diff(c)
    naive_fp += not lo <= 0 <= hi
    _, lo, hi = user_bootstrap_diff(c, seed=rep)
    user_fp += not lo <= 0 <= hi
print("300 A/A tests, 4,000 users each, 95% intervals that exclude zero:")
print(f"  conversations treated as independent  {naive_fp / 3:.1f}%")
print(f"  users resampled with their conversations  {user_fp / 3:.1f}%")

# 2. A real test: v2's prompt against v1's, 20,000 users.
c = simulate_ab(20_000, lift=LIFT, seed=42, harm=(0.004, 0.007))
nu = users_per_arm(c)
print(f"\nA/B test, 20,000 users; true effect on resolution {true_lift:+.1%}")
print(f"  users: control {nu[0]:,}, treatment {nu[1]:,}; SRM p = {srm_pvalue(*nu):.2f}")
print(f"  conversations: control {np.sum(c.arm == 0):,}, treatment {np.sum(c.arm == 1):,}")
for metric, label in (("resolved", "resolution"), ("harmful", "harmful promises")):
    y0, y1 = getattr(c, metric)[c.arm == 0].mean(), getattr(c, metric)[c.arm == 1].mean()
    _, nlo, nhi = naive_diff(c, metric)
    est, lo, hi = user_bootstrap_diff(c, metric, seed=1)
    print(
        f"  {label:<17} control {y0:.2%}  treatment {y1:.2%}  difference {est:+.2%}\n"
        f"  {'':<17} naive interval {nlo:+.2%} to {nhi:+.2%}; by user {lo:+.2%} to {hi:+.2%}"
    )

# 3. The same test with a logging bug that loses 4% of treatment users, the hardest ones.
c = simulate_ab(20_000, lift=LIFT, seed=42, harm=(0.004, 0.007), drop_low_users=0.04)
nu = users_per_arm(c)
est, lo, hi = user_bootstrap_diff(c, seed=1)
print("\nsame test, treatment logging loses some users:")
print(f"  users: control {nu[0]:,}, treatment {nu[1]:,}; SRM p = {srm_pvalue(*nu):.6f}")
print(f"  resolution difference {est:+.2%} ({lo:+.2%} to {hi:+.2%}); true {true_lift:+.2%}")

# 4. Drift: 12 weeks, 300 judged conversations a week.
rng = np.random.default_rng(5)
base_rates = dict(TRUE_RATES["v1"])
shifted = {"billing": 0.10, "account": 0.15, "orders": 0.50, "how_to": 0.20, "refuse": 0.05}
weeks = []
for w in range(1, 13):
    t = min(max((w - 4) / 3, 0), 1)  # mix moves toward `shifted` over weeks 5 to 7
    mix = {k: (1 - t) * TRAFFIC_MIX[k] + t * shifted[k] for k in TRAFFIC_MIX}
    rates = dict(base_rates)
    if w >= 9:
        rates["billing"] -= 0.10  # the provider updates the model in week 9
    cats = rng.choice(list(mix), size=300, p=list(mix.values()))
    passed = rng.random(300) < np.array([rates[k] for k in cats])
    obs_mix = {k: float(np.mean(cats == k)) for k in mix}
    by_cat = {k: (int(passed[cats == k].sum()), int((cats == k).sum())) for k in mix}
    weeks.append((w, obs_mix, by_cat, float(passed.mean())))

base = weeks[:4]
pooled = {k: (sum(b[2][k][0] for b in base), sum(b[2][k][1] for b in base)) for k in TRAFFIC_MIX}
ref_rates = {k: p / n for k, (p, n) in pooled.items()}
ref_raw = float(np.mean([b[3] for b in base]))
limit = 3 * math.sqrt(ref_raw * (1 - ref_raw) / 300)  # 3 standard errors for one week
print("\nweekly monitoring, 300 judged conversations a week (weeks 1-4 are the baseline):")
print(f"  {'week':>4}{'mix PSI':>9}{'pass rate':>11}{'':>3}{'billing, last 4 weeks':>24}")
for i, (w, obs_mix, _by_cat, raw) in enumerate(weeks):
    recent = weeks[max(0, i - 3) : i + 1]
    bp = sum(r[2]["billing"][0] for r in recent)
    bn = sum(r[2]["billing"][1] for r in recent)
    flag_raw = "!" if abs(raw - ref_raw) > limit else ""
    flag_bill = "!" if wilson(bp, bn)[1] < ref_rates["billing"] else ""
    print(
        f"  {w:>4}{psi(TRAFFIC_MIX, obs_mix):>9.2f}{raw:>11.1%}{flag_raw:<3}"
        f"{bp / bn:>17.1%} (n={bn:>3}){flag_bill}"
    )
print(f"  alarm (!): pass rate more than {limit:.1%} from the baseline {ref_raw:.1%}, or billing's")
print(f"  4-week Wilson interval wholly below its baseline {ref_rates['billing']:.1%}")

# 5. A rare guardrail event: 0.2% of conversations, rising on day 31. 5,000 a day.
base_rate, per_day = 0.002, 5000
print("\nrare event alarm, baseline 0.2% of 5,000 conversations a day, rising on day 31:")
for window in (1, 7):
    expected = base_rate * per_day * window
    k = poisson_alarm_threshold(expected, 0.001)
    print(f"  {window}-day window: alarm at {k} or more events (expected {expected:.0f})")
    for factor in (2.0, 1.5):
        false_alarms, delays = 0, []
        for rep in range(500):
            r = np.random.default_rng(rep)
            rate = np.where(np.arange(1, 91) >= 31, factor * base_rate, base_rate)
            counts = np.convolve(r.poisson(rate * per_day), np.ones(window), "valid")
            fired = np.arange(window, 91)[counts >= k]
            false_alarms += int(np.sum(fired < 31))
            after = fired[fired >= 31]
            delays.append(after[0] - 30 if len(after) else 61)
        print(
            f"    rate x{factor}: median days to detect {np.median(delays):.0f}; "
            f"false alarms before the change {false_alarms / 500 * 365 / 30:.1f} a year"
        )
