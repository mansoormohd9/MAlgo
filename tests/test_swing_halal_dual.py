"""
Two Shariah standards, and the GICS activity vocabulary.

The two properties that matter here:

  1. `eligible` follows the PRIMARY standard and nothing else. The second
     standard is computed and reported, never allowed to gate. If that ever
     slips, changing what is displayed would change what is tradeable.
  2. the GICS table is not a transliteration of the NSE one. Two entries that
     are correct against NSE labels are wrong against Yahoo's, and this file
     pins both.
"""
from __future__ import annotations

import pytest

from nifty_algo.config import Config
from nifty_algo.swing import halal, halal_taxonomy
from nifty_algo.swing import markets as markets_mod
from nifty_algo.swing.fundamentals import Fundamentals
from nifty_algo.swing.universe import Stock


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def us(cfg):
    return markets_mod.get(cfg, markets_mod.US)


def stock(symbol="ACME", sector="Technology", industry="Software - Infrastructure"):
    return Stock(symbol, f"{symbol} Inc", sector, industry, symbol)


def funds(assets=1000.0, debt=100.0, cash=100.0, receivables=100.0,
          market_cap=5000.0, **kw) -> Fundamentals:
    return Fundamentals(symbol="ACME", total_assets=assets, total_debt=debt,
                        cash_and_investments=cash, receivables=receivables,
                        market_cap=market_cap,
                        balance_sheet_date="2026-06-30", **kw)


# ------------------------------------------------------- the GICS vocabulary

@pytest.mark.parametrize("industry, expected_fragment", [
    ("Banks - Diversified", "banking"),
    ("Banks—Regional", "banking"),
    ("Credit Services", "lending"),
    ("Capital Markets", "brokerage"),
    ("Asset Management", "asset management"),
    ("Insurance - Life", "insurance"),
    ("REIT - Mortgage", "mortgage"),
    ("Beverages - Brewers", "alcohol"),
    ("Beverages - Wineries & Distilleries", "alcohol"),
    ("Tobacco", "tobacco"),
    ("Gambling", "gambling"),
    ("Resorts & Casinos", "gambling"),
    ("Lodging", "hotel"),
])
def test_gics_labels_are_caught(cfg, us, industry, expected_fragment):
    hit = halal.activity_failure(stock(industry=industry), market=us, cfg=cfg)
    assert hit is not None, industry
    assert expected_fragment in hit.lower()


def test_video_games_are_not_gambling(cfg, us):
    """
    The NSE table matches "gaming" because on NSE that means casinos. In
    Yahoo's vocabulary it also matches "Electronic Gaming & Multimedia", which
    is video games - not gambling, and not excluded by any mainstream index.
    """
    assert halal.activity_failure(
        stock(industry="Electronic Gaming & Multimedia"), market=us,
        cfg=cfg) is None


def test_ordinary_operating_businesses_pass_the_activity_screen(cfg, us):
    for industry in ("Semiconductors", "Drug Manufacturers - General",
                     "Oil & Gas Integrated", "Specialty Retail"):
        assert halal.activity_failure(
            stock(industry=industry), market=us, cfg=cfg) is None, industry


def test_the_indian_table_is_unchanged(cfg):
    """India's verdicts must not move because a second market was added."""
    india = markets_mod.get(cfg, markets_mod.INDIA)
    assert halal.PROHIBITED_ACTIVITIES is halal_taxonomy.NSE_ACTIVITIES
    hit = halal.activity_failure(
        stock(industry="Non Banking Financial Company (NBFC)"),
        market=india, cfg=cfg)
    assert hit == "interest-based lending (NBFC)"


def test_the_default_vocabulary_is_still_india(cfg):
    """Every pre-existing caller passes no market and must keep its behaviour."""
    assert halal.activity_failure(
        stock(industry="Private Sector Bank")) == \
        "conventional banking (interest-based)"


def test_an_unknown_taxonomy_raises_rather_than_matching_nothing():
    """
    Screening US stocks against NSE's vocabulary would match nothing at all
    and report every bank on the S&P 500 as activity-permissible.
    """
    with pytest.raises(KeyError):
        halal_taxonomy.table_for("gics-v2")


# ---------------------------------------------------------------- toggles

def test_defence_is_a_toggle_not_a_ruling(cfg, us):
    """SPUS excludes aerospace & defence; HLAL does not. Neither is a bug."""
    aerospace = stock(industry="Aerospace & Defense")

    cfg.swing.halal.exclude_defence = True
    assert halal.activity_failure(aerospace, market=us, cfg=cfg) is not None

    cfg.swing.halal.exclude_defence = False
    assert halal.activity_failure(aerospace, market=us, cfg=cfg) is None


def test_exchanges_and_data_is_a_toggle(cfg, us):
    exchange = stock(industry="Financial Data & Stock Exchanges")

    cfg.swing.halal.exclude_exchanges_and_data = True
    assert halal.activity_failure(exchange, market=us, cfg=cfg) is not None

    cfg.swing.halal.exclude_exchanges_and_data = False
    assert halal.activity_failure(exchange, market=us, cfg=cfg) is None


def test_toggles_do_not_apply_to_the_indian_table(cfg):
    india = markets_mod.get(cfg, markets_mod.INDIA)
    assert halal_taxonomy.toggled_for(india.taxonomy, cfg.swing.halal) == ()


# ---------------------------------------------------- the two methodologies

def test_both_standards_are_computed(cfg, us):
    v = halal.screen(stock(), funds(), cfg, market=us)
    assert set(v.verdicts) == {halal.METHOD_FTSE, halal.METHOD_AAOIFI}
    assert v.primary_method == halal.METHOD_FTSE


def test_eligibility_follows_the_primary_standard_only(cfg, us):
    """
    Debt is 40% of assets (fails FTSE at 33%) but 8% of market cap (passes
    AAOIFI at 30%). The stock is out, because FTSE is primary - and the fact
    that AAOIFI would have kept it is recorded, not acted on.
    """
    f = funds(assets=1000.0, debt=400.0, market_cap=5000.0)
    v = halal.screen(stock(), f, cfg, market=us)

    assert v.eligible is False
    assert v.verdicts[halal.METHOD_FTSE].passed is False
    assert v.verdicts[halal.METHOD_AAOIFI].passed is True
    assert v.disagreement is True


def test_switching_the_primary_switches_the_verdict(cfg, us):
    f = funds(assets=1000.0, debt=400.0, market_cap=5000.0)
    cfg.swing.halal.primary_method = halal.METHOD_AAOIFI
    v = halal.screen(stock(), f, cfg, market=us)
    assert v.eligible is True
    assert v.disagreement is True


def test_agreement_is_not_reported_as_disagreement(cfg, us):
    v = halal.screen(stock(), funds(), cfg, market=us)
    assert v.eligible is True
    assert v.disagreement is False


def test_a_missing_market_cap_does_not_gate_when_ftse_is_primary(cfg, us):
    """
    AAOIFI cannot be computed without a market cap. That must not take out a
    stock the PRIMARY standard cleared - the second opinion is information,
    not a veto.
    """
    v = halal.screen(stock(), funds(market_cap=None), cfg, market=us)
    assert v.eligible is True
    assert v.verdicts[halal.METHOD_AAOIFI].available is False
    assert v.disagreement is False


def test_a_missing_denominator_for_the_primary_is_a_no_data_verdict(cfg, us):
    """
    Fail closed, and distinguishably: "could not check" is a different
    statement from "failed the check".
    """
    cfg.swing.halal.primary_method = halal.METHOD_AAOIFI
    v = halal.screen(stock(), funds(market_cap=None), cfg, market=us)
    assert v.eligible is False
    assert v.source == halal.SOURCE_NO_DATA
    assert v.unverifiable is True


def test_the_override_still_beats_both_standards(cfg, us):
    ov = {"ACME": {"verdict": halal.VERDICT_NON_COMPLIANT,
                   "note": "checked the annual report", "reviewed_on": "2026-08-01"}}
    v = halal.screen(stock(), funds(), cfg, ov, market=us)
    assert v.eligible is False
    assert v.source == halal.SOURCE_OVERRIDE


def test_the_primary_checks_are_the_ones_on_the_verdict(cfg, us):
    """The page renders `checks`; it must show the standard that decided."""
    v = halal.screen(stock(), funds(), cfg, market=us)
    assert set(v.checks) == {"debt_to_assets", "cash_to_assets",
                             "receivables_to_assets"}
    assert v.checks["debt_to_assets"]["value"] == pytest.approx(0.10)
    assert "total assets" in v.checks["debt_to_assets"]["label"].lower()


def test_the_market_is_recorded_on_the_verdict(cfg, us):
    assert halal.screen(stock(), funds(), cfg, market=us).market == "us"


def test_haram_revenue_is_still_never_verified(cfg, us):
    assert halal.screen(stock(), funds(), cfg,
                        market=us).haram_revenue_verified is False


# ------------------------------------------------- the unclassified fail-open

def test_an_unclassified_stock_cannot_be_screened(cfg, us):
    """
    A REAL HOLE, NOT A HYPOTHETICAL. Yahoo returns no classification for UK
    closed-end investment trusts, so FCIT and Scottish Mortgage passed a
    screen that correctly excludes III and Pershing Square — which it happens
    to label "Asset Management". With nothing for the table to match, the
    activity screen reported "permissible", which is a fail-OPEN in a module
    whose entire contract is the opposite.
    """
    v = halal.screen(stock(sector="Unclassified", industry="Unclassified"),
                     funds(), cfg, market=us)
    assert v.eligible is False
    assert v.source == halal.SOURCE_NO_DATA
    assert v.unverifiable is True


@pytest.mark.parametrize("sector, industry", [
    ("", ""),
    ("Unclassified", ""),
    (None, None),
    ("n/a", "unknown"),
])
def test_every_flavour_of_missing_classification_fails_closed(cfg, us, sector,
                                                              industry):
    v = halal.screen(stock(sector=sector, industry=industry), funds(), cfg,
                     market=us)
    assert v.eligible is False and v.unverifiable is True


def test_one_classified_field_is_enough_to_screen(cfg, us):
    """A known sector with a blank industry is still screenable."""
    v = halal.screen(stock(sector="Technology", industry=""), funds(), cfg,
                     market=us)
    assert v.eligible is True


def test_an_unclassified_stock_can_still_be_overridden(cfg, us):
    """Your ruling beats a failure to classify, in both directions."""
    ov = {"ACME": {"verdict": halal.VERDICT_COMPLIANT,
                   "note": "checked it myself", "reviewed_on": "2026-08-21"}}
    v = halal.screen(stock(sector="Unclassified", industry="Unclassified"),
                     funds(), cfg, ov, market=us)
    assert v.eligible is True and v.source == halal.SOURCE_OVERRIDE
