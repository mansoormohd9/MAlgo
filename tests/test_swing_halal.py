"""
The halal screen.

This is the file that matters most in the swing subsystem, because every other
gate costs you a trade when it is wrong and this one costs you the reason you
asked for the feature. The two properties under test are:

  1. it fails CLOSED - absent data never reads as compliance
  2. the override file always wins, in both directions

Everything else is boundary arithmetic on three ratios.
"""
from __future__ import annotations

import pytest

from nifty_algo.config import Config
from nifty_algo.swing import halal
from nifty_algo.swing.fundamentals import Fundamentals
from nifty_algo.swing.universe import Stock


@pytest.fixture
def cfg() -> Config:
    return Config()


def stock(symbol="ACME", sector="Information Technology",
          industry="Computers - Software & Consulting") -> Stock:
    return Stock(symbol, f"{symbol} Industries", sector, industry, f"{symbol}.NS")


def funds(assets=1000.0, debt=100.0, cash=100.0, receivables=100.0,
          **kw) -> Fundamentals:
    return Fundamentals(symbol="ACME", total_assets=assets, total_debt=debt,
                        cash_and_investments=cash, receivables=receivables,
                        balance_sheet_date="2026-03-31", **kw)


# ---------------------------------------------------------------- activity

@pytest.mark.parametrize("industry, expected", [
    ("Private Sector Bank", "conventional banking (interest-based)"),
    ("Public Sector Bank", "conventional banking (interest-based)"),
    ("Non Banking Financial Company (NBFC)", "interest-based lending (NBFC)"),
    ("Life Insurance", "conventional insurance"),
    ("General Insurance", "conventional insurance"),
    ("Breweries & Distilleries", "alcohol"),
    ("Hotels & Resorts", "hotel bar/banquet revenue not separable"),
    ("Investment Company", "conventional investment holding"),
])
def test_prohibited_industries_are_caught(industry, expected):
    assert halal.activity_failure(stock(industry=industry)) == expected


def test_industry_is_read_before_sector():
    """
    ITC's sector is FMCG, which is permissible, and its industry names
    tobacco, which is not. Reading only the sector would pass it.
    """
    itc = stock("ITC", sector="Fast Moving Consumer Goods",
                industry="Diversified FMCG and Cigarettes & Tobacco Products")
    assert halal.activity_failure(itc) == "tobacco"


def test_permissible_business_is_not_flagged():
    assert halal.activity_failure(stock()) is None
    assert halal.activity_failure(
        stock(sector="Healthcare", industry="Pharmaceuticals")) is None


def test_activity_failure_short_circuits_the_ratios(cfg):
    """A bank with a pristine balance sheet is still a bank."""
    bank = stock("HDFCBANK", "Financial Services", "Private Sector Bank")
    verdict = halal.screen(bank, funds(debt=0, cash=0, receivables=0), cfg)
    assert verdict.eligible is False
    assert verdict.source == "activity"


# ---------------------------------------------------------------- ratios

@pytest.mark.parametrize("debt, passes", [
    (329.0, True),      # 32.9%
    (330.0, True),      # exactly 33.0% - the limit is inclusive
    (331.0, False),     # 33.1%
])
def test_debt_to_assets_boundary(cfg, debt, passes):
    verdict = halal.screen(stock(), funds(debt=debt), cfg)
    assert verdict.eligible is passes


@pytest.mark.parametrize("cash, passes", [(330.0, True), (331.0, False)])
def test_cash_to_assets_boundary(cfg, cash, passes):
    assert halal.screen(stock(), funds(cash=cash), cfg).eligible is passes


@pytest.mark.parametrize("receivables, passes", [(490.0, True), (491.0, False)])
def test_receivables_boundary(cfg, receivables, passes):
    assert halal.screen(stock(), funds(receivables=receivables),
                        cfg).eligible is passes


def test_denominator_is_assets_not_market_cap(cfg):
    """
    The verdict must not move when only the share price moves. This is the
    whole reason total assets was chosen over market cap.
    """
    a = halal.screen(stock(), funds(market_cap=1e9), cfg)
    b = halal.screen(stock(), funds(market_cap=1e11), cfg)
    assert a.eligible == b.eligible is True


def test_every_failing_ratio_is_named(cfg):
    verdict = halal.screen(stock(), funds(debt=900, cash=900, receivables=900),
                           cfg)
    assert verdict.eligible is False
    assert len(verdict.failures) == 3


def test_passing_verdict_carries_its_working(cfg):
    verdict = halal.screen(stock(), funds(), cfg)
    assert verdict.eligible is True
    assert set(verdict.checks) == {"debt_to_assets", "cash_to_assets",
                                   "receivables_to_assets"}
    assert verdict.checks["debt_to_assets"]["value"] == pytest.approx(0.10)


# ---------------------------------------------------------------- fail closed

@pytest.mark.parametrize("broken", [
    None,
    Fundamentals(symbol="ACME"),                                  # nothing at all
    funds(debt=None),                                             # one line missing
    funds(assets=0.0),                                            # unusable denominator
    Fundamentals(symbol="ACME", error="Yahoo returned no balance sheet"),
])
def test_missing_data_fails_closed(cfg, broken):
    """
    Absent data is a failure to VERIFY, never a pass. The two errors do not
    cost the same: wrongly excluding one halal stock costs a candidate,
    wrongly including a haram one defeats the point of the screen.
    """
    verdict = halal.screen(stock(), broken, cfg)
    assert verdict.eligible is False
    assert any("insufficient data" in f for f in verdict.failures)
    # ...and it is marked as unverifiable, NOT as a failed ratio. A renamed
    # ticker must never be presented as a ruling that the company is haram.
    assert verdict.source == halal.SOURCE_NO_DATA
    assert verdict.unverifiable is True


def test_a_real_ratio_failure_is_not_marked_unverifiable(cfg):
    verdict = halal.screen(stock(), funds(debt=900), cfg)
    assert verdict.eligible is False
    assert verdict.source == halal.SOURCE_RATIO
    assert verdict.unverifiable is False


def test_missing_data_can_be_allowed_when_explicitly_configured(cfg):
    cfg.swing.halal.exclude_on_missing_data = False
    assert halal.screen(stock(), None, cfg).eligible is True


def test_haram_revenue_is_never_claimed_as_verified(cfg):
    """The one test this module cannot perform must never look performed."""
    assert halal.screen(stock(), funds(), cfg).haram_revenue_verified is False


# ---------------------------------------------------------------- overrides

def test_override_can_force_compliant(cfg):
    """A bank you have personally cleared still gets through - your call."""
    bank = stock("SOMEBANK", "Financial Services", "Private Sector Bank")
    ov = {"SOMEBANK": {"verdict": "compliant", "note": "checked accounts",
                       "reviewed_on": "2026-08-01"}}
    verdict = halal.screen(bank, funds(), cfg, ov)
    assert verdict.eligible is True
    assert verdict.source == "override"
    assert "checked accounts" in verdict.reason


def test_override_can_force_non_compliant(cfg):
    ov = {"ACME": {"verdict": "non_compliant", "note": "interest income too high",
                   "reviewed_on": ""}}
    verdict = halal.screen(stock(), funds(), cfg, ov)
    assert verdict.eligible is False
    assert verdict.source == "override"


def test_unrecognised_override_verdict_falls_through_rather_than_guessing(cfg):
    ov = {"ACME": {"verdict": "maybe", "note": "", "reviewed_on": ""}}
    verdict = halal.screen(stock(), funds(), cfg, ov)
    assert verdict.source == "ratio"
    assert verdict.eligible is True


# ---------------------------------------------------------------- the file

def test_override_file_roundtrip(tmp_path):
    path = tmp_path / "halal_overrides.csv"
    halal.save_overrides(path, [
        {"symbol": "wipro", "verdict": "compliant", "note": "checked FY25",
         "reviewed_on": "2026-04-11"},
    ])
    loaded, warnings = halal.load_overrides(path)
    assert warnings == []
    assert loaded["WIPRO"]["verdict"] == "compliant"


def test_bad_row_warns_and_is_skipped_without_killing_the_file(tmp_path):
    """
    A ruling you meant to apply and which vanished silently is worse than no
    ruling, so the bad row has to be reported rather than dropped quietly.
    """
    path = tmp_path / "halal_overrides.csv"
    path.write_text(
        "symbol,verdict,note,reviewed_on\n"
        "GOODROW,compliant,fine,2026-01-01\n"
        "BADROW,probably,typo,2026-01-01\n",
        encoding="utf-8")
    loaded, warnings = halal.load_overrides(path)
    assert "GOODROW" in loaded
    assert "BADROW" not in loaded
    assert len(warnings) == 1 and "BADROW" in warnings[0]


def test_missing_override_file_is_not_an_error(tmp_path):
    loaded, warnings = halal.load_overrides(tmp_path / "nope.csv")
    assert loaded == {} and warnings == []


def test_comments_in_the_override_file_are_ignored(tmp_path):
    path = tmp_path / "o.csv"
    path.write_text("# a comment\nsymbol,verdict,note,reviewed_on\n"
                    "# another\nACME,compliant,ok,2026-01-01\n",
                    encoding="utf-8")
    loaded, warnings = halal.load_overrides(path)
    assert list(loaded) == ["ACME"] and warnings == []


# ---------------------------------------------------------------- shipped data

def test_the_committed_universe_screens_the_obvious_cases():
    """
    Guards the data file, not the code: a sector/industry label edited into
    something vague would silently stop excluding banks.
    """
    from nifty_algo.swing.universe import load_universe
    by_symbol = {s.symbol: s for s in load_universe("data/nifty100.csv")}

    for symbol in ("HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                   "BAJFINANCE", "SHRIRAMFIN", "LICI", "SBILIFE", "ICICIGI",
                   "ITC", "UNITDSPR", "INDHOTEL"):
        assert halal.activity_failure(by_symbol[symbol]) is not None, symbol

    for symbol in ("RELIANCE", "TCS", "INFY", "SUNPHARMA", "TITAN", "LT"):
        assert halal.activity_failure(by_symbol[symbol]) is None, symbol
