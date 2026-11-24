from collections import Counter

import pytest

from evidence.dataset import Item
from evidence.leakage import normalise, overlaps
from evidence.sampling import distinct_by_category, frequencies, stratified_sample, weighted_rate
from evidence.traffic import question_pool, traffic_log


def item(i, cat, q):
    return Item(f"t{i}", cat, q, "")


def test_normalise_ignores_greetings_case_and_punctuation():
    assert normalise("Hi, Where is order #10231? thanks") == normalise("where is order #10231")


def test_distinct_and_stratified_sampling():
    log = [
        item(0, "a", "Q one?"),
        item(1, "a", "q one"),
        item(2, "a", "Q two"),
        item(3, "b", "Q three"),
        item(4, "b", "Q four"),
    ]
    strata = distinct_by_category(log)
    assert [i.question for i in strata["a"]] == ["Q one?", "Q two"]
    s = stratified_sample(strata, 2, seed=0)
    assert Counter(i.category for i in s) == {"a": 2, "b": 2}
    assert len({normalise(i.question) for i in s}) == 4  # no question twice


def test_weighted_rate_by_traffic_and_frequency():
    items = [item(0, "a", "popular"), item(1, "a", "rare"), item(2, "b", "other")]
    results = {"t0": 1, "t1": 0, "t2": 0}
    mix = {"a": 0.5, "b": 0.5}
    assert weighted_rate(results, items, mix) == pytest.approx(0.25)
    freq = Counter({"popular": 9, "rare": 1, "other": 1})
    assert weighted_rate(results, items, mix, freq) == pytest.approx(0.45)


def test_frequencies_count_normalised_questions():
    log = [item(0, "a", "Q one?"), item(1, "a", "hi, q one thanks"), item(2, "a", "Q two")]
    assert frequencies(log) == Counter({"q one": 2, "q two": 1})


def test_overlaps_find_exact_and_reworded_copies_only():
    hits = overlaps(
        ["Where is order #10231?", "How do I export to PDF?"],
        ["Hi, where is order #10231 thanks", "Can I schedule a weekly report?"],
    )
    assert [(q, round(s, 2)) for q, _, s in hits] == [("Where is order #10231?", 1.0)]


def test_traffic_is_reproducible_and_pool_is_capped():
    a, da = traffic_log(n=500)
    b, db = traffic_log(n=500)
    assert a == b and da == db
    sizes = {c: len(v) for c, v in question_pool().items()}
    assert all(0 < n <= 150 for n in sizes.values())
