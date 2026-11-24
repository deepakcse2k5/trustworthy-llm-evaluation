import subprocess
import sys

import pytest

from evidence.intervals import wilson
from evidence.power import (
    detectable_effect,
    discordant_share,
    n_for_precision,
    n_paired,
    n_unpaired,
)


def test_precision_plan_delivers_the_width():
    n = n_for_precision(0.8, 0.03)
    assert n == 683
    lo, hi = wilson(round(0.8 * n), n)
    assert (hi - lo) / 2 == pytest.approx(0.03, abs=0.002)


def test_paired_needs_fewer_questions_when_fewer_differ():
    assert n_paired(0.08, 0.03) < n_paired(0.25, 0.03)
    assert n_paired(0.25, 0.03) < n_unpaired(0.81, 0.84)
    with pytest.raises(ValueError):
        n_paired(0.02, 0.03)


def test_detectable_effect_inverts_n_paired():
    d = detectable_effect(1000, 0.08)
    assert n_paired(0.08, d) <= 1000 < n_paired(0.08, d * 0.99)


def test_discordant_share():
    assert discordant_share({"a": 1, "b": 0}, {"a": 0, "b": 0}, ["a", "b"]) == 0.5


def test_size_command():
    out = subprocess.run(
        [sys.executable, "-m", "evidence", "size", "rate", "--rate", "0.8", "--half-width", "0.03"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.startswith("683 questions")
