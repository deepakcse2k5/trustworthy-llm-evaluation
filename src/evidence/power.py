"""How many questions do you need? Sample sizes for a single rate and for paired comparisons.

All formulas are normal approximations. They are good enough to plan with; check a plan that
matters by simulation, as the chapter does.
"""

import math
from statistics import NormalDist

Z = NormalDist().inv_cdf


def n_for_precision(rate: float, half_width: float, confidence: float = 0.95) -> int:
    """Questions needed for an interval of +/- half_width around a pass rate near `rate`."""
    z = Z(1 - (1 - confidence) / 2)
    return math.ceil(z**2 * rate * (1 - rate) / half_width**2)


def n_paired(discordant: float, effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Questions needed for a paired comparison (McNemar) to detect `effect`, the difference in
    pass rates, when a share `discordant` of questions change result between the versions."""
    if not 0 < abs(effect) <= discordant <= 1:
        raise ValueError("need 0 < |effect| <= discordant <= 1")
    za, zb = Z(1 - alpha / 2), Z(power)
    n = (za * math.sqrt(discordant) + zb * math.sqrt(discordant - effect**2)) ** 2 / effect**2
    return math.ceil(n)


def n_unpaired(rate_a: float, rate_b: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Questions per version for two separate samples: what pairing saves you from."""
    za, zb = Z(1 - alpha / 2), Z(power)
    mean = (rate_a + rate_b) / 2
    spread = math.sqrt(rate_a * (1 - rate_a) + rate_b * (1 - rate_b))
    n = (za * math.sqrt(2 * mean * (1 - mean)) + zb * spread) ** 2 / (rate_b - rate_a) ** 2
    return math.ceil(n)


def detectable_effect(n: int, discordant: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """The smallest difference a paired comparison of n questions detects with this power."""
    lo, hi = 1e-6, discordant
    for _ in range(60):
        mid = (lo + hi) / 2
        if n_paired(discordant, mid, alpha, power) > n:
            lo = mid
        else:
            hi = mid
    return hi


def discordant_share(a: dict[str, int], b: dict[str, int], ids: list[str]) -> float:
    """From a pilot run of both versions: the share of questions whose result differs."""
    return sum(a[i] != b[i] for i in ids) / len(ids)
