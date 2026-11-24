"""Reporting evidence: plain-language wording for results, and a decision memo generated from
a comparison. The wording rules are the chapter's recommendations; adapt them to your readers.
"""

from dataclasses import dataclass

from evidence.compare import Comparison

NAMES = {"how_to": "how-to", "refuse": "refusals"}  # code names to reader's names

PLAIN = {  # Chapter 6's verdicts, in words a reader outside the team can act on
    "better": "better, by more than the amount that matters",
    "worse": "worse, by more than the allowed margin",
    "no change that matters": "no change large enough to matter",
    "better, may be small": "better, though possibly by less than the amount that matters",
    "worse, may be small": "worse, though possibly by less than the allowed margin",
    "inconclusive": "not settled by this evaluation",
}


def points(x: float) -> str:
    """A rate difference in whole percentage points, with a sign word."""
    p = round(abs(x) * 100)
    if p == 0:
        return "no change"
    return f"{p} point{'s' if p != 1 else ''} {'higher' if x > 0 else 'lower'}"


def describe_range(lo: float, hi: float) -> str:
    """'between 10 and 20 points lower' rather than '-20.0% to -9.7%'."""
    a, b = round(lo * 100), round(hi * 100)
    if a == 0 and b == 0:
        return "no change either way"
    if a == 0:
        return f"between no change and {b} points higher"
    if b == 0:
        return f"between {-a} points lower and no change"
    if a < 0 < b:
        return f"between {-a} points lower and {b} points higher"
    word = "higher" if a > 0 else "lower"
    small, large = sorted((abs(a), abs(b)))
    return f"between {small} and {large} points {word}"


def one_in(rate: float) -> str:
    """A rate as a natural frequency: 0.007 -> 'about 1 in 140'."""
    if rate <= 0:
        return "none"
    n = 1 / rate
    step = 1 if n < 20 else 5 if n < 100 else 10 if n < 1000 else 100
    return f"about 1 in {round(n / step) * step:,}"


@dataclass(frozen=True)
class MemoInputs:
    title: str
    decision: str  # the question the memo answers, in the reader's words
    old: str
    new: str
    rows: list[Comparison]  # from compare(): categories, then "all"
    cost: tuple[float, float, float]  # cost difference per 1,000 questions: estimate, low, high
    detectable: float  # smallest difference this evaluation could detect (Chapter 7)
    method: list[str]  # one line per method note
    unknowns: list[str]  # what the evaluation can't tell you
    next_steps: list[str]  # what would change the recommendation, and what happens next


def recommend(inp: MemoInputs) -> str:
    """Ship only if no category is clearly worse than its margin and cost didn't clearly rise."""
    blocked = [r.name for r in inp.rows if r.name != "all" and r.verdict == "worse"]
    _, lo, _ = inp.cost
    if blocked:
        return f"Do not ship {inp.new} yet: {', '.join(blocked)} got worse beyond its margin."
    if lo > 0:
        return f"Do not ship {inp.new} yet: it raises the cost of failures."
    return f"Ship {inp.new}: no category got worse beyond its margin."


def memo(inp: MemoInputs) -> str:
    """A one-page decision memo in Markdown: decision first, evidence second, method last."""
    allq = next(r for r in inp.rows if r.name == "all")
    est, lo, hi = inp.cost
    lines = [
        f"# {inp.title}",
        "",
        f"**Decision needed:** {inp.decision}",
        "",
        f"**Recommendation:** {recommend(inp)}",
        "",
        "## What we found",
        "",
        f"- Overall, {inp.new} answered {allq.rate_b:.0%} of questions correctly, against "
        f"{allq.rate_a:.0%} for {inp.old}: {points(allq.rate_b - allq.rate_a)}, likely "
        f"{describe_range(allq.lo, allq.hi)}.",
        f"- But the cost of wrong answers would be about ${abs(est):,.0f} "
        f"{'more' if est > 0 else 'less'} per 1,000 questions (likely between "
        f"${abs(lo):,.0f} and ${abs(hi):,.0f} {'more' if lo > 0 else 'less' if hi < 0 else ''})"
        ".",
        "",
        "| Area | " + inp.old + " | " + inp.new + " | Change (likely range) | What it means |",
        "|---|---|---|---|---|",
    ]
    for r in inp.rows:
        if r.name == "all":
            continue
        lines.append(
            f"| {NAMES.get(r.name, r.name)} | {r.rate_a:.0%} | {r.rate_b:.0%} | "
            f"{describe_range(r.lo, r.hi)} | {PLAIN[r.verdict]} |"
        )
    lines += [
        "",
        "## What would change this, and what happens next",
        "",
        *[f"- {n}" for n in inp.next_steps],
        "",
        "## What this evaluation can't tell you",
        "",
        f"- It can reliably detect a change of about {inp.detectable * 100:.0f} points in the "
        "overall rate. Smaller changes may be real but would look like no change.",
        *[f"- {u}" for u in inp.unknowns],
        "",
        "## How we measured",
        "",
        *[f"- {m}" for m in inp.method],
        "",
    ]
    return "\n".join(lines)
