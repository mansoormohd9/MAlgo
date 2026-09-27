"""
`planning/goals.py`, `planning/tax.py` and the stale-balance check.

Same refusal-first shape as the rest of `planning/`: a projection without a
chosen return or a net worth is withheld, a line without a cost basis has an
unknown gain rather than a zero one, and an undated balance is never judged
fresh.
"""
from __future__ import annotations

from datetime import date

import pytest

from nifty_algo.planning import allocation as alloc
from nifty_algo.planning import goals as g
from nifty_algo.planning import tax
from nifty_algo.portfolio.aggregate import PortfolioSnapshot, _combine
from nifty_algo.portfolio.base import (EQUITY, ETF, FIXED_INCOME, MUTUAL_FUND,
                                       ConnectorResult, Position)
from nifty_algo.portfolio.manual import _to_position


# ---------------------------------------------------------------- goals

def test_the_projection_is_withheld_until_it_can_be_stated():
    goal = [g.Goal("House", 1_000_000, 2030)]
    assert g.project(None, 10_000, 5.0, goal, 2026) is None
    assert g.project(500_000, 10_000, 0.0, goal, 2026) is None
    assert g.project(500_000, 10_000, 5.0, [], 2026) is None


def test_contributions_alone_at_a_near_zero_return():
    """At ~0% real, one year of ₹10,000 a month is ₹1.2 lakh."""
    proj = g.project(0.0, 10_000, 1e-9, [g.Goal("X", 1e9, 2026)], 2026)
    assert proj.checks[0].projected_inr == pytest.approx(120_000, rel=1e-6)


def test_a_spend_goal_leaves_the_pot_and_a_corpus_does_not():
    spend = g.project(1_000_000, 0, 1e-9, [g.Goal("House", 400_000, 2026),
                                            g.Goal("Later", 1, 2027)], 2026)
    assert spend.checks[1].projected_inr == pytest.approx(600_000, rel=1e-6)
    keep = g.project(1_000_000, 0, 1e-9,
                     [g.Goal("Corpus", 400_000, 2026, spend=False),
                      g.Goal("Later", 1, 2027)], 2026)
    assert keep.checks[1].projected_inr == pytest.approx(1_000_000, rel=1e-6)


def test_a_shortfall_names_the_monthly_amount_that_closes_it():
    proj = g.project(0.0, 0.0, 1e-9, [g.Goal("Hajj", 120_000, 2026)], 2026)
    check = proj.checks[0]
    assert not check.on_track
    assert check.short_inr == pytest.approx(120_000)
    assert check.extra_monthly_inr == pytest.approx(10_000, rel=1e-6)


def test_goals_round_trip_and_bad_rows_are_warned(tmp_path):
    path = tmp_path / "goals.csv"
    goals = [g.Goal("House", 2_500_000, 2031),
             g.Goal("Retire", 50_000_000, 2054, spend=False)]
    g.save(goals, path)
    back, warnings = g.load(path)
    assert back == goals and not warnings
    path.write_text(path.read_text() + "Broken,,2030,true\n")
    _, warnings = g.load(path)
    assert warnings


def test_a_blank_editor_row_is_not_a_goal():
    assert g.from_row({"name": "", "target_inr": float("nan"),
                       "year": ""}) is None
    assert g.from_row({"name": "X", "target_inr": float("nan"),
                       "year": 2030}) is None


# ---------------------------------------------------------------- tax

def _pos(symbol, value, cost, market="india", asset_class=EQUITY, name=""):
    return Position(key=f"{market}:{symbol}", symbol=symbol, market=market,
                    quantity=1.0, average_price=cost, last_price=value,
                    currency="INR", asset_class=asset_class, name=name)


def _snap(*positions):
    snap = PortfolioSnapshot(positions=list(positions))
    snap.value_inr = {p.key: p.value_native for p in positions}
    snap.results = [ConnectorResult.ok("manual", list(positions))]
    return snap


def test_harvest_counts_only_equity_oriented_indian_gains():
    view = tax.harvest_view(_snap(
        _pos("INFY", 150_000, 100_000),                 # +50k counts
        _pos("TCS", 80_000, 100_000),                   # a loss adds nothing
        _pos("GOLDBEES", 200_000, 100_000),             # gold: excluded
        _pos("MON100", 200_000, 100_000),               # overseas: excluded
        _pos("SPUS", 200_000, 100_000, market="us", asset_class=ETF),
        _pos("EPF", 200_000, 100_000, asset_class=FIXED_INCOME),
        _pos("FUND", 90_000, 0.0, asset_class=MUTUAL_FUND)))  # no cost
    assert view.gain_inr == pytest.approx(50_000)
    assert view.lines_without_cost == 1
    assert view.tax_saved_inr == pytest.approx(50_000 * tax.LTCG_RATE)


def test_the_harvest_is_capped_at_the_exemption():
    view = tax.harvest_view(_snap(_pos("INFY", 1_000_000, 100_000)))
    assert view.harvestable_inr == tax.LTCG_EXEMPTION_INR


def test_the_financial_year_ends_in_march():
    assert tax.fy_end(date(2026, 9, 27)) == date(2027, 3, 31)
    assert tax.fy_end(date(2027, 2, 1)) == date(2027, 3, 31)
    assert tax.harvest_season(date(2027, 2, 1))
    assert not tax.harvest_season(date(2026, 9, 27))


# ---------------------------------------------------------------- staleness

def test_only_dated_lines_older_than_the_limit_are_stale():
    today = date(2026, 9, 27)
    old = Position(key="india:EPF", symbol="EPF", market="india",
                   quantity=1, average_price=0, last_price=1, currency="INR",
                   as_of="2026-01-01")
    fresh = Position(key="india:PPF", symbol="PPF", market="india",
                     quantity=1, average_price=0, last_price=1,
                     currency="INR", as_of="2026-09-01")
    undated = Position(key="india:FD", symbol="FD", market="india",
                       quantity=1, average_price=0, last_price=1,
                       currency="INR")
    got = alloc.stale_lines([old, fresh, undated], today)
    assert [p.symbol for p, _ in got] == ["EPF"]


def test_the_manual_csv_carries_as_of():
    pos, why = _to_position({"symbol": "EPF", "currency": "INR",
                             "value": "300000", "asset_class": "fixed_income",
                             "as_of": "2026-03-31"})
    assert pos is not None, why
    assert pos.as_of == "2026-03-31"


def test_a_combined_line_is_as_old_as_its_stalest_part():
    a = Position(key="india:X", symbol="X", market="india", quantity=1,
                 average_price=1, last_price=1, currency="INR",
                 as_of="2026-06-01")
    b = Position(key="india:X", symbol="X", market="india", quantity=1,
                 average_price=1, last_price=1, currency="INR",
                 as_of="2026-01-01")
    live = Position(key="india:X", symbol="X", market="india", quantity=1,
                    average_price=1, last_price=1, currency="INR")
    assert _combine(a, b).as_of == "2026-01-01"
    assert _combine(live, a).as_of == "2026-06-01"
