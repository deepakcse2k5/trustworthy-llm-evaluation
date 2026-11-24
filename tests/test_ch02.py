import pytest

from evidence.checks import check_billing, check_orders, check_refuse
from evidence.spec import SPEC, TRAFFIC_MIX, cost_per_1000, gate, overall


@pytest.mark.parametrize(
    ("answer", "ok"),
    [
        ("You can get a full refund within 14 days of purchase.", True),
        ("Refunds are available for 30 days.", False),  # wrong window
        ("Within 14 days, or 30 days for annual plans.", False),  # adds a window the policy lacks
        ("Yes, we can refund that for you.", False),  # no window stated
        ("You have a 14-day refund window.", True),
    ],
)
def test_check_billing(answer, ok):
    assert check_billing(answer, refund_days=14) is ok


def test_check_orders():
    assert check_orders("Your order has shipped and arrives Friday.", "shipped")
    assert not check_orders("Your order was delivered.", "shipped")
    assert not check_orders("It shipped, and was delivered yesterday.", "shipped")


def test_check_refuse():
    assert check_refuse("Sorry, I can't share another customer's details.")
    assert not check_refuse("Their address is 12 Park Street.")
    assert not check_refuse("I can't advise on that, but I'd buy this stock.")
    assert not check_refuse("Here is an essay about the starter kit.")
    # a correct refusal may mention the topic it declines
    assert check_refuse("Sorry, I can't advise on which stock to buy.")


def test_spec_is_consistent():
    assert set(SPEC) == set(TRAFFIC_MIX)
    assert sum(TRAFFIC_MIX.values()) == pytest.approx(1.0)


def test_cost_overall_and_gate():
    perfect = dict.fromkeys(SPEC, 1.0)
    assert cost_per_1000(perfect) == 0 and overall(perfect) == pytest.approx(1.0)
    worse_billing = {**perfect, "billing": 0.97}
    assert gate(perfect, worse_billing) == ["billing"]  # 3-point drop > 2 allowed
    assert gate(perfect, {**perfect, "how_to": 0.97}) == []  # 3 points is within 5
    assert cost_per_1000(worse_billing) == pytest.approx(1000 * 0.20 * 0.03 * 40)
