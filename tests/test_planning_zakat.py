"""
`planning/zakat.py` - zakat on the whole of your wealth, and purification.

The refusals come first. An unchosen ruling, an unknown metal price or an
unread account must withhold the amount due, because each one, defaulted
quietly, is an amount someone would pay.
"""
from __future__ import annotations

import pytest

from nifty_algo.config import Config
from nifty_algo.planning import zakat as z
from nifty_algo.portfolio.aggregate import PortfolioSnapshot
from nifty_algo.portfolio.base import (CASH, EQUITY, FIXED_INCOME,
                                       ConnectorResult, Position)


def _pos(symbol, value, asset_class=EQUITY, name="", source="manual"):
    return Position(key=f"india:{symbol}", symbol=symbol, market="india",
                    quantity=1.0, average_price=0.0, last_price=value,
                    currency="INR", asset_class=asset_class, source=source,
                    name=name)


def _snap(*positions, complete=True):
    snap = PortfolioSnapshot(positions=list(positions))
    snap.value_inr = {p.key: p.value_native for p in positions}
    snap.results = [ConnectorResult.ok("manual", list(positions))]
    if not complete:
        snap.results.append(ConnectorResult.unavailable("kite", "expired"))
    return snap


def _plan(**kw):
    plan = Config().plan
    plan.zakat_nisab_basis = kw.get("basis", z.BASIS_SILVER)
    plan.silver_price_inr_per_g = kw.get("silver", 100.0)
    plan.gold_price_inr_per_g = kw.get("gold", 0.0)
    plan.zakat_equity_method = kw.get("method", z.METHOD_MARKET)
    plan.zakat_equity_proxy_pct = kw.get("proxy", 0.0)
    plan.zakat_include_retirement = kw.get("retire", False)
    return plan


WEALTH = (_pos("INFY", 400_000), _pos("SAVINGS", 100_000, CASH),
          _pos("GOLDBEES", 50_000), _pos("EPF", 300_000, FIXED_INCOME))


# ---------------------------------------------------------------- refusals

def test_a_fresh_config_states_no_amount_and_says_why():
    """Every ruling defaults to unset - the code does not pick a madhhab."""
    r = z.compute(_snap(*WEALTH), Config().plan)
    assert r.due_inr is None
    assert any("nisab basis" in m for m in r.missing)
    assert any("shares and funds" in m for m in r.missing)


def test_no_price_no_nisab():
    r = z.compute(_snap(*WEALTH), _plan(silver=0.0))
    assert r.nisab_inr is None and r.due_inr is None
    assert any("price" in m for m in r.missing)


def test_the_gold_basis_reads_the_gold_price_not_the_silver_one():
    assert z.nisab_inr(_plan(basis=z.BASIS_GOLD, gold=0.0)) is None
    assert z.nisab_inr(_plan(basis=z.BASIS_GOLD, gold=7_000.0)) \
        == pytest.approx(z.NISAB_GOLD_G * 7_000.0)


def test_an_incomplete_snapshot_withholds_the_amount():
    r = z.compute(_snap(*WEALTH, complete=False), _plan())
    assert r.due_inr is None
    assert r.lines                       # the breakdown still renders


def test_the_assets_method_needs_its_proxy():
    r = z.compute(_snap(*WEALTH), _plan(method=z.METHOD_ASSETS, proxy=0.0))
    assert r.due_inr is None


# ---------------------------------------------------------------- arithmetic

def test_market_value_counts_everything_but_retirement():
    r = z.compute(_snap(*WEALTH), _plan())
    # INFY + cash + gold; EPF excluded until accessible.
    assert r.base_inr == pytest.approx(550_000)
    assert r.due_inr == pytest.approx(550_000 * 0.025)


def test_retirement_money_can_be_included():
    r = z.compute(_snap(*WEALTH), _plan(retire=True))
    assert r.base_inr == pytest.approx(850_000)


def test_the_assets_method_applies_the_proxy_to_investments_only():
    r = z.compute(_snap(*WEALTH), _plan(method=z.METHOD_ASSETS, proxy=25.0))
    # 25% of INFY, all of cash and gold.
    assert r.base_inr == pytest.approx(100_000 + 100_000 + 50_000)


def test_below_nisab_owes_nothing_rather_than_withholding():
    r = z.compute(_snap(_pos("SAVINGS", 10_000, CASH)), _plan(silver=100.0))
    assert r.nisab_inr == pytest.approx(61_236)
    assert r.above_nisab is False
    assert r.due_inr == 0.0


def test_a_deposit_that_is_not_retirement_is_counted():
    r = z.compute(_snap(_pos("FD-HDFC", 200_000, FIXED_INCOME,
                             name="Fixed deposit")), _plan())
    assert r.base_inr == pytest.approx(200_000)


# ---------------------------------------------------------------- purification

def test_dividends_without_a_ratio_are_unknown_not_zero():
    plan = _plan()
    plan.interest_received_inr = 1_200.0
    plan.dividends_received_inr = 5_000.0
    pur = z.purification(_snap(*WEALTH), plan)
    assert pur.dividend_inr is None and pur.total_inr is None
    plan.dividend_purification_pct = 4.0
    pur = z.purification(_snap(*WEALTH), plan)
    assert pur.total_inr == pytest.approx(1_200 + 200)


def test_interest_sources_are_named():
    ibkr_cash = Position(key="us:IBKR-USD", symbol="IBKR-USD", market="us",
                         quantity=1.0, average_price=0.0, last_price=5_000,
                         currency="USD", asset_class=CASH, source="ibkr")
    pur = z.purification(_snap(_pos("EPF", 1, FIXED_INCOME),
                               _pos("SGBJUN31I", 1), _pos("LIQUIDBEES", 1),
                               ibkr_cash), _plan())
    text = " ".join(pur.sources)
    for needle in ("EPF", "SGBJUN31I", "LIQUIDBEES", "IBKR"):
        assert needle in text
