"""
`planning/compliance.py` - is what you actually own halal?

The assertions that matter most are the refusals. An unverified holding is
never counted as halal, a missing balance sheet is never a pass, and an
incomplete snapshot withholds every share. Each of those, got wrong, reads as
a clean bill of health.
"""
from __future__ import annotations

import pytest

from nifty_algo.planning import compliance as c
from nifty_algo.portfolio.aggregate import PortfolioSnapshot
from nifty_algo.portfolio.base import (CASH, EQUITY, ETF, FIXED_INCOME, GOLD,
                                       MUTUAL_FUND, ConnectorResult, Position)

FUNDS_CSV = """# comment
symbol,name_contains,name,status,standard,domicile,purification_pct,note
SPUS,,SP Funds S&P 500 Sharia,compliant,AAOIFI,US,,
,TATA ETHICAL,Tata Ethical Fund,compliant,Shariah board,IN,2.5,
BADROW,,Broken,maybe,,,,
"""


def _pos(symbol, value=100.0, market="india", asset_class=EQUITY, name=""):
    return Position(key=f"{market}:{symbol}", symbol=symbol, market=market,
                    quantity=1.0, average_price=0.0, last_price=value,
                    currency="INR", asset_class=asset_class, source="manual",
                    name=name)


@pytest.fixture
def funds(tmp_path):
    path = tmp_path / "halal_funds.csv"
    path.write_text(FUNDS_CSV, encoding="utf-8")
    return c.load_fund_list(path)


def _status(position, funds, **kw):
    return c.classify(position, funds, **kw).status


# ---------------------------------------------------------------- the list

def test_the_fund_list_matches_by_symbol_and_by_name(funds):
    assert _status(_pos("SPUS", market="us", asset_class=ETF), funds) \
        == c.COMPLIANT
    cas = _pos("INF277K01Z77", asset_class=MUTUAL_FUND,
               name="Tata Ethical Fund - Direct Plan - Growth")
    assert _status(cas, funds) == c.COMPLIANT
    assert funds.by_name[0].purification_pct == 2.5


def test_a_bad_row_is_a_warning_not_a_silent_drop(funds):
    assert "BADROW" not in funds.by_symbol
    assert any("BADROW" in w or "row 5" in w or "maybe" in w
               for w in funds.warnings)


def test_a_missing_fund_list_says_so(tmp_path):
    fl = c.load_fund_list(tmp_path / "nope.csv")
    assert fl.warnings and not fl.by_symbol


# ---------------------------------------------------------------- classes

def test_interest_bearing_lines_are_not_halal(funds):
    assert _status(_pos("EPF", asset_class=FIXED_INCOME), funds) \
        == c.NON_COMPLIANT
    assert _status(_pos("LIQUIDBEES"), funds) == c.NON_COMPLIANT


def test_cash_is_halal_to_hold(funds):
    assert _status(_pos("SAVINGS", asset_class=CASH), funds) == c.COMPLIANT


def test_metal_etfs_are_contested_until_the_holder_rules(funds):
    for p in (_pos("GOLDBEES"), _pos("SILVERBEES"),
              _pos("COINS", asset_class=GOLD)):
        assert _status(p, funds) == c.CONTESTED
        assert _status(p, funds, accept_metal_etfs=True) == c.COMPLIANT


def test_an_sgb_stays_contested_whatever_the_metal_ruling(funds):
    """Its coupon is interest, whatever view is taken of the gold."""
    sgb = _pos("SGBJUN31I")
    assert _status(sgb, funds) == c.CONTESTED
    assert _status(sgb, funds, accept_metal_etfs=True) == c.CONTESTED


def test_a_conventional_fund_is_not_halal(funds):
    assert _status(_pos("NIFTYBEES", asset_class=ETF,
                        name="Nippon India ETF Nifty 50 BeES"), funds) \
        == c.NON_COMPLIANT


def test_a_shariah_sounding_fund_off_the_list_is_unverified_not_halal(funds):
    p = _pos("INF000X", asset_class=MUTUAL_FUND,
             name="Some Shariah Opportunities Fund")
    assert _status(p, funds) == c.UNVERIFIED


def test_a_typed_total_cannot_be_screened(funds):
    p = _pos("DIRECT_US", market="us")
    assert _status(p, funds, screen_equity=lambda _p: (True, "x", "ratio")) \
        == c.UNVERIFIED


# ---------------------------------------------------------------- shares

def test_a_share_without_a_balance_sheet_is_unverified_never_a_pass(funds):
    assert _status(_pos("NEWCO"), funds) == c.UNVERIFIED
    assert _status(_pos("NEWCO"), funds, screen_equity=lambda _p: None) \
        == c.UNVERIFIED


def test_a_share_follows_the_screen(funds):
    ok = lambda _p: (True, "activity permissible", "ratio")      # noqa: E731
    bad = lambda _p: (False, "conventional banking", "activity")  # noqa: E731
    nodata = lambda _p: (False, "cannot verify", "no_data")        # noqa: E731
    assert _status(_pos("INFY"), funds, screen_equity=ok) == c.COMPLIANT
    assert _status(_pos("HDFCBANK"), funds, screen_equity=bad) \
        == c.NON_COMPLIANT
    assert _status(_pos("X"), funds, screen_equity=nodata) == c.UNVERIFIED
    v = c.classify(_pos("Y"), funds,
                   screen_equity=lambda _p: (True, "ruled", "override"))
    assert v.basis == "override"


# ---------------------------------------------------------------- summary

def _snap(*positions, complete=True):
    snap = PortfolioSnapshot(positions=list(positions))
    snap.value_inr = {p.key: p.value_native for p in positions}
    snap.results = [ConnectorResult.ok("manual", list(positions))]
    if not complete:
        snap.results.append(ConnectorResult.unavailable("kite", "expired"))
    return snap


def test_shares_are_by_value_and_unverified_is_not_halal(funds):
    snap = _snap(_pos("SPUS", 600, market="us", asset_class=ETF),
                 _pos("NEWCO", 300),
                 _pos("EPF", 100, asset_class=FIXED_INCOME))
    s = c.assess(snap, funds)
    assert s.share(c.COMPLIANT) == pytest.approx(0.6)
    assert s.share(c.UNVERIFIED) == pytest.approx(0.3)
    assert s.share(c.NON_COMPLIANT) == pytest.approx(0.1)
    assert s.needs_attention


def test_an_incomplete_snapshot_withholds_every_share(funds):
    s = c.assess(_snap(_pos("SPUS", 600, market="us", asset_class=ETF),
                       complete=False), funds)
    for status in c.STATUSES:
        assert s.share(status) is None
    # Rupee totals are facts about what WAS read, and still render.
    assert s.total(c.COMPLIANT) == 600


def test_the_committed_fund_list_loads_cleanly():
    """The shipped file is source: it must parse with no warnings."""
    fl = c.load_fund_list()
    assert not fl.warnings, fl.warnings
    assert {"SPUS", "HLAL", "ISDW"} <= set(fl.by_symbol)
