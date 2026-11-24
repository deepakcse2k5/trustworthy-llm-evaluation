"""Chapter 10: how much to trust human labels, on SIMULATED raters and answers."""

import numpy as np

from evidence.labels import (
    LabelItem,
    Rater,
    accuracy,
    adjudicate,
    fleiss_kappa,
    leave_one_out,
    majority,
    make_items,
)
from evidence.noise import Judge, agreement, judge_error_rates

raters = [
    Rater("Asha", 0.95),
    Rater("Ben", 0.93),
    Rater("Chen", 0.96),
    Rater("Dana", 0.94),
    Rater("Eli", 0.95, lenience=0.7),  # passes "partly correct" answers out of habit
]

# 1. A pilot round: five raters, the same 100 answers, before and after revising guidelines.
pilot = make_items(100, seed=1)
for clarity, title in ((0.2, "first guidelines"), (0.8, "revised guidelines, with examples")):
    labs = {r.name: r.label(pilot, clarity, seed=10 + k) for k, r in enumerate(raters)}
    print(f"pilot, 100 answers, {title}:")
    print(f"  Fleiss' kappa, all five: {fleiss_kappa(list(labs.values())):.2f}")
    loo = leave_one_out(labs)
    for r in raters:
        others = [
            agreement(list(labs[r.name].values()), list(labs[o.name].values()))[1]
            for o in raters
            if o is not r
        ]
        print(
            f"  {r.name:<5} passes {np.mean(list(labs[r.name].values())):.0%}   "
            f"agrees with the others' majority {loo[r.name]:.0%}   mean kappa {np.mean(others):.2f}"
        )
    clear = [it.id for it in pilot if not it.borderline]
    border = [it.id for it in pilot if it.borderline]
    for name, ids in (("clear", clear), ("borderline", border)):
        sub = [{i: lab[i] for i in ids} for lab in labs.values()]
        print(f"  Fleiss' kappa on {len(ids)} {name} answers: {fleiss_kappa(sub):.2f}")
    print()

# 2. Gold questions: answers an expert panel has settled, mixed into each rater's queue.
mixed = make_items(30, seed=2)
targeted = [LabelItem(f"g{i:02d}", 0, True) for i in range(20)]  # borderline, should fail
print("gold questions, revised guidelines:")
print(f"  {'':<6}{'30 typical answers':>20}{'20 borderline fails':>22}")
for k, r in enumerate(raters):
    a = accuracy(r.label(mixed, 0.8, seed=20 + k), mixed)
    b = accuracy(r.label(targeted, 0.8, seed=25 + k), targeted)
    print(f"  {r.name:<6}{a:>20.0%}{b:>22.0%}")

# 3. Ways to label 1,000 answers, and what each costs and delivers.
items = make_items(1000, seed=3)
truth = np.mean([it.good for it in items])
team = raters[:4]  # Eli coached on borderline fails before joining the main round
expert = Rater("expert", 0.99)
judge = Judge(sensitivity=0.97, specificity=0.60)
graded = judge.grade({it.id: it.good for it in items}, seed=4)
print(f"\nlabelling 1,000 answers (true pass rate {truth:.1%}); judge's true specificity 60%")
print(f"  {'strategy':<32}{'cost':>6}{'accuracy':>10}{'pass rate':>11}{'judge spec':>12}")
for clarity in (0.2, 0.8):
    lab = [r.label(items, clarity, seed=30 + k) for k, r in enumerate(team)]
    ex = expert.label(items, 1.0, seed=40)
    strategies = {
        "one rater": (lab[0], 1),
        "two raters + expert on splits": (
            adjudicate(lab[0], lab[1], ex),
            2 + 3 * np.mean([lab[0][i] != lab[1][i] for i in lab[0]]),
        ),
        "three raters, majority": (majority(lab[:3]), 3),
        "expert only": (ex, 3),
    }
    print(f"  clarity {clarity}:")
    for name, (labels, cost) in strategies.items():
        h = [labels[it.id] for it in items]
        sens, spec = judge_error_rates(h, [graded[it.id] for it in items])
        print(
            f"    {name:<30}{cost:>6.2f}{accuracy(labels, items):>10.1%}"
            f"{np.mean(h):>11.1%}{spec:>12.0%}"
        )
