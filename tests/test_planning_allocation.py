"""
`planning/allocation.py` - the whole-of-wealth arithmetic.

The assertions that matter most are the refusals: an incomplete snapshot must
withhold every percentage and every "next rupee" suggestion, because a
rebalance computed against a denominator that could not be established is an
ACTION taken on a guess.
"""
from __future__ import annotations

import pytest

from nifty_algo.config import Config
from nifty_algo.factor.drawdown import DRAWDOWN_HAIRCUT
from nifty_algo.factor.sleeve import MEASURED_DRAWDOWN
from nifty_algo.planning import allocation as alloc
from nifty_algo.portfolio.aggregate import PortfolioSnapshot
from nifty_algo.portfolio.base import (CASH, EQUITY, ETF, FIXED_INCOME, GOLD,
                                       ConnectorResult, Position)


def _pos(symbol, value, market="india", asset_class=EQUITY):
    return Position(key=f"{market}:{symbol}", symbol=symbol, market=market,
                    quantity=1.0, average_price=0.0, last_price=value,
                    currency="INR", asset_class=asset_class, source="manual")


def _snap(*positions, complete=True):
    snap = PortfolioSnapshot(positions=list(positions))
    snap.value_inr = {p.key: p.value_native for p in positions}
    snap.results = [ConnectorResult.ok("manual", list(positions))]
    if not complete:
        snap.results.append(ConnectorResult.unavailable("kite", "token expired"))
    return snap


def _plan(**weights):
    cfg = Config()
    plan = cfg.plan
    plan.w_india_equity = weights.get("india", 60.0)
    plan.w_foreign_equity = weights.get("foreign", 20.0)
    plan.w_gold = weights.get("gold", 10.0)
    plan.w_fixed_income = weights.get("fixed", 5.0)
    plan.w_cash = weights.get("cash", 5.0)
    plan.rebalance_band_pp = 5.0
    return plan


# ---------------------------------------------------------------- buckets

def test_every_class_lands_in_the_right_bucket():
    assert alloc.bucket_of(_pos("HDFCBANK", 1)) == alloc.INDIA_EQUITY
    assert alloc.bucket_of(_pos("SPUS", 1, market="us", asset_class=ETF)) \
        == alloc.FOREIGN_EQUITY
    assert alloc.bucket_of(_pos("EPF", 1, asset_class=FIXED_INCOME)) \
        == alloc.BUCKET_FIXED
    assert alloc.bucket_of(_pos("SAVINGS", 1, asset_class=CASH)) \
        == alloc.BUCKET_CASH
    assert alloc.bucket_of(_pos("COINS", 1, asset_class=GOLD)) \
        == alloc.BUCKET_GOLD


def test_a_gold_etf_from_kite_is_gold_but_a_jeweller_is_not():
    """Kite labels every NSE line EQUITY. The hint is a list, not a substring
    match, precisely so a jeweller's share is not counted as gold."""
    assert alloc.bucket_of(_pos("GOLDBEES", 1)) == alloc.BUCKET_GOLD
    assert alloc.bucket_of(_pos("SGBJUN31I", 1)) == alloc.BUCKET_GOLD
    assert alloc.bucket_of(_pos("GOLDIAM", 1)) == alloc.INDIA_EQUITY


# ---------------------------------------------------------------- build

def test_current_against_target():
    a = alloc.build(_snap(_pos("A", 70_000), _pos("CASH", 30_000,
                                                  asset_class=CASH)),
                    _plan(india=60, foreign=0, gold=0, fixed=0, cash=40))
    assert a.net_worth_inr == 100_000
    india = a.row(alloc.INDIA_EQUITY)
    assert india.current_pct == pytest.approx(70.0)
    assert india.drift_pp == pytest.approx(10.0)
    assert india.status == "over"
    assert a.row(alloc.BUCKET_CASH).status == "under"


def test_an_incomplete_snapshot_withholds_every_percentage():
    a = alloc.build(_snap(_pos("A", 70_000), complete=False), _plan())
    assert a.net_worth_inr is None
    assert all(r.current_pct is None for r in a.rows)
    assert all(r.status == "unknown" for r in a.rows)
    assert a.total_inr == 70_000           # the rupees read are still facts
    assert alloc.plan_contribution(a, 10_000) is None


def test_targets_that_do_not_sum_to_100_block_the_plan():
    a = alloc.build(_snap(_pos("A", 1_000)), _plan(india=50))   # sums to 90
    assert not a.targets_set
    assert any("not 100%" in n for n in a.notes)
    assert alloc.plan_contribution(a, 10_000) is None


def test_unset_targets_are_zero_and_not_a_split():
    """Zero means 'not set' - the default never chooses an allocation."""
    assert alloc.targets_total(Config().plan) == 0.0
    assert not alloc.targets_valid(Config().plan)


# ---------------------------------------------------------------- new money

def test_new_money_fills_the_largest_gap_first_and_sells_nothing():
    plan = _plan(india=50, foreign=50, gold=0, fixed=0, cash=0)
    a = alloc.build(_snap(_pos("A", 80_000),
                          _pos("SPUS", 20_000, market="us", asset_class=ETF)),
                    plan)
    out = alloc.plan_contribution(a, 10_000)
    assert out.buy[alloc.FOREIGN_EQUITY] == pytest.approx(10_000)
    assert out.buy[alloc.INDIA_EQUITY] == pytest.approx(0.0)
    # 90k/110k India is still 31.8pp over a 50% target: name the trim.
    assert alloc.INDIA_EQUITY in out.trim
    assert sum(out.buy.values()) == pytest.approx(10_000)


def test_money_beyond_every_gap_is_split_by_target():
    plan = _plan(india=50, foreign=50, gold=0, fixed=0, cash=0)
    a = alloc.build(_snap(_pos("A", 50_000),
                          _pos("SPUS", 40_000, market="us", asset_class=ETF)),
                    plan)
    out = alloc.plan_contribution(a, 110_000)
    # after: 200k -> 100k each. India needs 50k, foreign 60k: exact.
    assert out.buy[alloc.INDIA_EQUITY] == pytest.approx(50_000)
    assert out.buy[alloc.FOREIGN_EQUITY] == pytest.approx(60_000)
    assert not out.trim


def test_contributions_always_sum_to_the_amount():
    plan = _plan()
    a = alloc.build(_snap(_pos("A", 12_345), _pos("G", 999, asset_class=GOLD),
                          _pos("C", 5_000, asset_class=CASH)), plan)
    for amount in (0.0, 1.0, 777.0, 50_000.0, 5e6):
        out = alloc.plan_contribution(a, amount)
        assert sum(out.buy.values()) == pytest.approx(amount)
        assert all(v >= 0 for v in out.buy.values())


# ---------------------------------------------------------------- emergency

def test_the_emergency_fund_is_cash_against_months_of_expenses():
    plan = _plan()
    plan.monthly_expenses_inr = 50_000
    plan.emergency_months = 6
    a = alloc.build(_snap(_pos("A", 1e6), _pos("C", 200_000, asset_class=CASH)),
                    plan)
    check = alloc.emergency(a, plan)
    assert check.months_covered == pytest.approx(4.0)
    assert not check.ok
    assert check.shortfall_inr == pytest.approx(100_000)


def test_no_expenses_means_no_emergency_verdict():
    a = alloc.build(_snap(_pos("A", 1)), _plan())
    assert alloc.emergency(a, _plan()) is None


# ---------------------------------------------------------------- sleeve

def test_the_sleeve_share_is_F2bs_arithmetic():
    """Same constants as `sizing_report`, so the page and F2b cannot drift."""
    planning = min(0.95, MEASURED_DRAWDOWN * DRAWDOWN_HAIRCUT)
    assert alloc.planning_drawdown() == pytest.approx(planning)
    assert alloc.sleeve_share(0.15) == pytest.approx(0.15 / planning)
    # The figure CLAUDE.md quotes: tolerate 15% -> about 17% of net worth.
    assert round(alloc.sleeve_share(0.15), 2) == 0.17


def test_no_net_worth_means_no_ceiling():
    assert alloc.sleeve_cap(None, 0.15) is None
    assert alloc.sleeve_cap(0.0, 0.15) is None
    assert alloc.sleeve_cap(1_000_000, 0.15) == pytest.approx(
        1_000_000 * alloc.sleeve_share(0.15))
