from evidence.dataset import CATEGORIES, load, make_eval_set, save
from evidence.score import by_category, pass_rate
from evidence.simulate import TRUE_RATES, expected_rate, item_difficulty, run


def test_eval_set_is_reproducible_and_balanced(tmp_path):
    items = make_eval_set()
    assert items == make_eval_set()  # same seed, same set
    assert len(items) == 200
    counts = {c: sum(i.category == c for i in items) for c in CATEGORIES}
    assert counts == {c: n for c, (n, _) in CATEGORIES.items()}
    path = tmp_path / "set.jsonl"
    save(items, str(path))
    assert load(str(path)) == items


def test_scoring():
    items = make_eval_set()[:4]  # four billing questions
    results = {items[0].id: 1, items[1].id: 0, items[2].id: 1, items[3].id: 1}
    assert pass_rate(results, items) == 0.75
    assert by_category(results, items) == {"billing": (3, 4)}


def test_runs_repeat_with_a_seed_and_vary_without_one():
    items = make_eval_set()
    diff = item_difficulty(items)
    assert run("v1", items, diff, seed=1) == run("v1", items, diff, seed=1)
    scores = {pass_rate(run("v1", items, diff, seed=s), items) for s in range(20)}
    assert len(scores) > 1  # the same version scores differently on different runs


def test_long_run_rates_reflect_the_settings():
    items = make_eval_set()
    diff = item_difficulty(items)
    billing = [i for i in items if i.category == "billing"]
    assert expected_rate("v2", billing, diff) < expected_rate("v1", billing, diff)
    assert set(TRUE_RATES) == {"v1", "v2"}
