"""
The typed foreign balances on US / LRS, as the snapshot sees them.

Both regressions here understated foreign wealth silently: the direct-shares
box fed the estate meter and nothing else, and an Irish UCITS typed by hand was
keyed differently from the same fund read off the IBKR statement.
"""
from __future__ import annotations

from nifty_algo.portfolio.base import EQUITY
from nifty_algo.ui import page_portfolio


def test_direct_us_shares_reach_the_snapshot():
    typed = page_portfolio._typed_positions({"SPUS": 1_000.0,
                                             "DIRECT_US": 2_500.0})
    by_key = {p.key: p for p in typed}
    assert "us:DIRECT_US" in by_key
    assert by_key["us:DIRECT_US"].value_native == 2_500.0
    assert by_key["us:DIRECT_US"].asset_class == EQUITY
    assert by_key["us:DIRECT_US"].currency == "USD"


def test_an_irish_ucits_is_keyed_by_its_lse_listing():
    """The IBKR statement keys ISDW `uk:` (see test_ibkr_flex); a typed
    balance must use the same key or one fund has two names."""
    typed = page_portfolio._typed_positions({"ISDW": 900.0})
    assert [p.key for p in typed] == ["uk:ISDW"]


def test_zero_balances_are_not_positions():
    assert page_portfolio._typed_positions({"SPUS": 0.0,
                                            "DIRECT_US": 0.0}) == []
