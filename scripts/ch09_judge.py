"""Chapter 9: finding and removing a judge's biases, on SIMULATED answers and a SIMULATED judge."""

from dataclasses import replace

import numpy as np

from evidence.judge import (
    BiasedJudge,
    calibration_report,
    human_prefers_first,
    longer_wins,
    make_answers,
    position_check,
    win_rate,
)
from evidence.noise import corrected_rate, judge_error_rates
from evidence.sampling import random_sample
from evidence.simulate import run
from evidence.traffic import traffic_log

log, log_diff = traffic_log()
s = random_sample(log, 1000, seed=31)
old = make_answers(run("v1", s, log_diff, seed=1), "v1", "A", mean_words=120, seed=2)
new = make_answers(run("v2", s, log_diff, seed=3), "v2", "B", mean_words=240, seed=4)
judge = BiasedJudge(family="B")  # the judge comes from the same model family as v2
qids = list(new)

# 1. Head to head: which answer is better, v2's or v1's?
human = np.mean([human_prefers_first(new[q], old[q]) for q in qids])
print(f"v2 against v1, 1,000 questions: a person prefers v2's answer {human:.1%} of the time")
print("the judge prefers v2's answer:")
for order in ("new first", "old first", "random", "both"):
    print(f"  {order:<10} {win_rate(judge, new, old, seed=5, order=order):.1%}")

# 2. Which bias does how much? Switch each off in turn (order random).
print("\nthe same judge with one bias switched off (order random):")
variants = {
    "all biases": judge,
    "no position bias": replace(judge, position=0.0),
    "no length bias": replace(judge, length=0.0),
    "no self-preference": replace(judge, self_pref=0.0),
    "no biases": replace(judge, position=0.0, length=0.0, self_pref=0.0),
}
for name, j in variants.items():
    print(f"  {name:<20}{win_rate(j, new, old, seed=6, order='random'):.1%}")

# 3. Position: ask twice, swapping the order.
print("\nposition check, every pair asked in both orders:")
for name, j in (("judge", judge), ("without position bias", replace(judge, position=0.0))):
    first, flips = position_check(j, new, old, seed=7)
    print(f"  {name:<22} answer shown first wins {first:.1%}; verdict changes {flips:.1%}")

# 4. Length: how often does the longer answer win?
print("\nlength check, order random:")
rng = np.random.default_rng(8)
firsts, seconds = [], []
for q in qids:
    a, b = (new[q], old[q]) if rng.random() < 0.5 else (old[q], new[q])
    firsts.append(a)
    seconds.append(b)
for name, prefer in (
    ("person", [human_prefers_first(a, b) for a, b in zip(firsts, seconds, strict=True)]),
    ("judge", [judge.prefers_first(a, b, rng) for a, b in zip(firsts, seconds, strict=True)]),
):
    w, n = longer_wins(prefer, firsts, seconds)
    print(f"  longer answer preferred by the {name:<7}{w / n:.1%} of {n} pairs")
v2_longer = np.mean([new[q].words > old[q].words for q in qids])
print(f"  (v2's answer is the longer one in {v2_longer:.0%} of pairs)")

# 5. Self-preference: the same pairs, judged by models from three families (both orders).
print("\njudge from each model family, both orders:")
for fam in ("A", "B", "C"):
    j = replace(judge, family=fam)
    print(f"  family {fam}: prefers v2 {win_rate(j, new, old, seed=9, order='both'):.1%}")
fixed = replace(judge, family="C", length=0.2)
print(
    f"  family C with a length-neutral rubric: prefers v2 "
    f"{win_rate(fixed, new, old, seed=9, order='both'):.1%}   (person: {human:.1%})"
)

# 6. Pass/fail grading and a calibration report.
grader = replace(judge, threshold=-0.8)
rng = np.random.default_rng(10)
answers = [old[q] for q in qids] + [new[q] for q in qids]
graded = [grader.grade(a, rng) for a in answers]
calib = list(range(0, 150)) + list(range(1000, 1150))  # a person grades 150 of each version
h = [answers[i].good for i in calib]
jg = [graded[i] for i in calib]
print("\ncalibration on 300 answers graded by a person (150 per version):")
print(f"  {'group':<14}{'n':>5}{'sensitivity':>23}{'specificity':>23}{'kappa':>7}")
for label, groups in (
    ("version", [answers[i].version for i in calib]),
    ("length", ["long (>150)" if answers[i].words > 150 else "short" for i in calib]),
):
    for r in calibration_report(h, jg, groups):
        if label == "length" and r.group == "all":
            continue
        print(
            f"  {r.group:<14}{r.n:>5}{r.sensitivity:>8.0%} ({r.sens_interval[0]:.0%}-"
            f"{r.sens_interval[1]:.0%}){'':>4}{r.specificity:>8.0%} ({r.spec_interval[0]:.0%}-"
            f"{r.spec_interval[1]:.0%}){'':>4}{r.kappa:>7.2f}"
        )

print("\npass rates on the other 850 answers per version:")
pooled = judge_error_rates(h, jg)
for v, off in (("v1", 0), ("v2", 1000)):
    rest = range(off + 150, off + 1000)
    true = np.mean([answers[i].good for i in rest])
    raw = np.mean([graded[i] for i in rest])
    own = [i for i in calib if answers[i].version == v]
    sens, spec = judge_error_rates([answers[i].good for i in own], [graded[i] for i in own])
    own_fix = corrected_rate(raw, sens, spec)
    print(
        f"  {v}: true {true:.1%}   judge {raw:.1%}   corrected with pooled rates "
        f"{corrected_rate(raw, *pooled):.1%}   with {v}'s own rates {own_fix:.1%}"
    )
