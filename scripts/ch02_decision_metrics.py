"""Chapter 2: start from the decision. Long-run rates come from the SIMULATED versions in Chapter 1;
failure costs and traffic mix are ILLUSTRATIVE."""

from evidence.dataset import load
from evidence.simulate import expected_rate, item_difficulty
from evidence.spec import SPEC, TRAFFIC_MIX, cost_per_1000, gate, overall

items = load("data/eval_set.jsonl")
diff = item_difficulty(items)
rates = {
    v: {c: expected_rate(v, [i for i in items if i.category == c], diff) for c in SPEC}
    for v in ("v1", "v2")
}
print(f"{'category':<10}{'share':>7}{'cost':>7}{'v1':>8}{'v2':>8}")
for c in SPEC:
    print(
        f"{c:<10}{TRAFFIC_MIX[c]:>7.0%}{SPEC[c].cost_per_failure:>7.0f}"
        f"{rates['v1'][c]:>8.1%}{rates['v2'][c]:>8.1%}"
    )
print()
for v in ("v1", "v2"):
    print(
        f"{v}: pass rate weighted by traffic {overall(rates[v]):.1%}, "
        f"cost of failures ${cost_per_1000(rates[v]):,.0f} per 1,000 questions"
    )
print("gate:", gate(rates["v1"], rates["v2"]) or "pass")
