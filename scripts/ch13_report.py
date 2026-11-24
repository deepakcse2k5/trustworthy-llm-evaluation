"""Chapter 13: a decision memo generated from Chapter 6's comparison, on SIMULATED data."""

from pathlib import Path

from evidence.compare import compare, discordant, paired_stat_interval
from evidence.power import detectable_effect
from evidence.report import MemoInputs, describe_range, memo, one_in, points
from evidence.sampling import random_sample
from evidence.simulate import run
from evidence.spec import cost_per_1000
from evidence.traffic import traffic_log

# 1. Plain-language building blocks.
print("plain-language wording:")
for x in (0.031, -0.144, 0.004):
    print(f"  {x:+.1%} -> {points(x)}")
for lo, hi in ((-0.200, -0.097), (0.014, 0.048), (0.000, 0.064), (-0.04, 0.04)):
    print(f"  {lo:+.1%} to {hi:+.1%} -> {describe_range(lo, hi)}")
for rate in (0.0039, 0.0070, 0.188):
    print(f"  {rate:.2%} -> {one_in(rate)}")

# 2. Chapter 6's comparison, turned into a memo.
log, log_diff = traffic_log()
s = random_sample(log, 1000, seed=21)
r1, r2 = run("v1", s, log_diff, seed=5), run("v2", s, log_diff, seed=5)
ids = [i.id for i in s]
oa, ob = discordant(r1, r2, ids)
inputs = MemoInputs(
    title="Should the support assistant move to v2?",
    decision="Replace the current assistant (v1) with the new version (v2) next sprint?",
    old="v1",
    new="v2",
    rows=compare(r1, r2, s),
    cost=paired_stat_interval(r1, r2, s, cost_per_1000),
    detectable=detectable_effect(len(ids), (oa + ob) / len(ids)),
    method=[
        "1,000 customer questions sampled from last month's logs, in proportion to "
        "traffic; both versions answered the same questions.",
        "Answers checked by code for billing, orders and refusals, and by an LLM judge "
        "calibrated against two raters plus an expert for account and how-to.",
        "Likely ranges are 95% intervals from a paired bootstrap; margins come from the "
        "metric specification written before the evaluation.",
        "Costs per wrong answer are the specification's estimates: billing $40, refusals $25, "
        "account $6, orders $5, how-to $3.",
    ],
    next_steps=[
        "Read the 28 billing questions that v1 got right and v2 got wrong, and fix the cause.",
        "Re-test billing on a set sized to detect a 2-point drop; ship if it shows no "
        "evidence of a drop beyond 2 points.",
        "Then run a two-week A/B test with harmful promises as a guardrail (Chapter 12).",
    ],
    unknowns=[
        "Online behaviour: resolution and complaints need an A/B test with guardrails.",
        "Questions unlike last month's traffic, such as a new product launch.",
        "Model updates by the provider after this evaluation.",
    ],
)
text = memo(inputs)
Path("data/ch13_memo.md").write_text(text)
print("\n" + text)
