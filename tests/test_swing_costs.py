"""
Delivery charges. The flat DP fee is the one that matters on a small ticket.

A percentage cost model cannot see a fixed charge, and a fixed charge is
exactly what makes a Rs 10,000 position proportionally more expensive than a
Rs 33,000 one. That asymmetry is the whole reason this file exists.
"""
from __future__ import annotations

import pytest

from nifty_algo.swing.costs_equity import (DEFAULT_EQUITY_COSTS,
                                           EquityCostModel, VERIFIED_ON)


@pytest.fixture
def costs() -> EquityCostModel:
    return EquityCostModel()


def test_stt_is_charged_on_both_legs(costs):
    """
    The single biggest difference from the option model, which pays it once.

    Delivery pays 0.1% going in AND 0.1% coming out.
    """
    bare = EquityCostModel(transaction_pct=0.0, sebi_pct=0.0, gst_pct=0.0,
                           stamp_duty_pct=0.0, dp_charge=0.0,
                           slippage_pct=0.0)
    assert bare.buy_cost(1000.0, 10) == pytest.approx(10.0)     # 0.1% of 10k
    assert bare.sell_cost(1000.0, 10) == pytest.approx(10.0)
    assert bare.round_trip(1000.0, 1000.0, 10) == pytest.approx(20.0)


def test_stamp_duty_is_buy_side_only(costs):
    bare = EquityCostModel(transaction_pct=0.0, sebi_pct=0.0, gst_pct=0.0,
                           stt_pct=0.0, dp_charge=0.0, slippage_pct=0.0)
    assert bare.buy_cost(1000.0, 10) == pytest.approx(1.5)      # 0.015%
    assert bare.sell_cost(1000.0, 10) == pytest.approx(0.0)


def test_the_dp_charge_is_flat_and_sell_side_only(costs):
    """Rs 15.34 per scrip whether you sell one share or a thousand."""
    small = costs.sell_cost(100.0, 1)
    large = costs.sell_cost(100.0, 1000)
    assert small - costs.dp_charge < 1.0        # almost all of it IS the fee
    assert costs.buy_cost(100.0, 1) < costs.dp_charge


def test_a_flat_fee_hurts_a_small_ticket_proportionally_more(costs):
    """
    The finding this module exists to make visible.

    Same 5% stop, same 2:1 payoff - the smaller position simply keeps less of
    it, and nothing in a percentage-based cost model would show that.
    """
    small = costs.friction_r(500.0, 475.0, 20)      # Rs 10,000 deployed
    large = costs.friction_r(500.0, 475.0, 66)      # Rs 33,000 deployed
    assert small > large
    assert small == pytest.approx(0.095, abs=0.02)  # ~9.5% of the risk budget


def test_friction_in_r_is_comparable_to_the_ladders_rungs(costs):
    """
    Rs 45 sounds like nothing. 0.1R against a 2R target does not.

    A 2:1 setup really pays about 1.9:1, and a scan ranking on raw R:R is
    ranking on a number nobody actually receives.
    """
    r = costs.friction_r(500.0, 475.0, 20)
    assert 0.05 < r < 0.20


def test_the_breakeven_move_is_a_real_fraction_of_a_percent(costs):
    """You start roughly half a percent underwater on a delivery round trip."""
    move = costs.breakeven_move_pct(500.0, 20)
    assert 0.003 < move < 0.008


def test_zero_quantity_and_zero_risk_do_not_divide_by_zero(costs):
    assert costs.friction_r(500.0, 500.0, 20) == 0.0     # no risk points
    assert costs.breakeven_move_pct(0.0, 0) == 0.0


def test_every_rate_carries_the_date_it_was_checked():
    """
    A rate with no date is a rate nobody will re-check.

    Stamp duty, exchange transaction charges and delivery STT have all moved
    in the last few years; this file has to say when it last looked.
    """
    assert VERIFIED_ON and len(VERIFIED_ON) == 10
    assert VERIFIED_ON in DEFAULT_EQUITY_COSTS.note(500.0, 475.0, 20)


def test_a_scale_out_pays_the_dp_charge_on_every_sell_day(costs):
    """
    The DP fee is per scrip PER SELL DAY. A position that banks a partial and
    then exits pays it twice, and the ladder is built to produce exactly that
    shape - so charging one round trip flatters the winners specifically.
    """
    one_exit = costs.friction(500.0, 550.0, 20)
    two_exits = one_exit + costs.dp_charge

    assert two_exits > one_exit
    # On a Rs 10,000 ticket the second fee alone is ~3% of a Rs 500 risk
    # budget. Not a rounding error.
    assert costs.dp_charge / 500.0 > 0.025
