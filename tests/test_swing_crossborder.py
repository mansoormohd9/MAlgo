"""
ETF overlap, and the cross-border arithmetic.

The overlap check answers "is this actually a new position?" and the estate
meter answers "has the US-situs total crossed $60,000?". Both are properties
of the whole portfolio that no individual ticket can show, and both are
arithmetic - which is exactly why they are testable and worth testing.
"""
from __future__ import annotations

import pytest

from nifty_algo.config import Config
from nifty_algo.swing import crossborder, holdings
from nifty_algo.swing import markets as markets_mod


@pytest.fixture
def cfg() -> Config:
    return Config()


class Held:
    def __init__(self, label, value_usd, us_situs):
        self.label, self.value_usd, self.us_situs = label, value_usd, us_situs


# ---------------------------------------------------------------- parsing

def _row(ticker, weight, name="Thing Inc", flag="", date="08/21/2026"):
    return {"Date": date, "StockTicker": ticker, "SecurityName": name,
            "Weightings": weight, "MoneyMarketFlag": flag}


def test_a_normal_row_parses():
    out, dropped = holdings._parse("SPUS", [_row("NVDA", "13.62%", "NVIDIA Corp")])
    assert dropped == 0
    assert out[0].symbol == "NVDA"
    assert out[0].weight == pytest.approx(0.1362)
    assert out[0].as_of == "2026-08-21"


def test_an_administrator_placeholder_is_dropped():
    """
    Both issuers publish rows like `2602335D` at a zero price. Left in, they
    become universe entries no price feed can ever resolve.
    """
    out, dropped = holdings._parse("HLAL", [_row("2602335D", "0.00%")])
    assert out == [] and dropped == 1


def test_a_zero_weight_is_not_a_position():
    out, dropped = holdings._parse("SPUS", [_row("XYZ", "0.00%")])
    assert out == [] and dropped == 1


def test_a_money_market_sweep_is_dropped():
    out, dropped = holdings._parse("SPUS", [_row("MMF", "2.00%", flag="Y")])
    assert out == [] and dropped == 1


def test_a_duplicate_ticker_is_counted_once():
    out, dropped = holdings._parse(
        "SPUS", [_row("AAPL", "5.00%"), _row("AAPL", "5.00%")])
    assert len(out) == 1 and dropped == 1


def test_a_class_suffix_ticker_survives():
    out, _ = holdings._parse("SPUS", [_row("BRK.B", "1.00%")])
    assert out and out[0].symbol == "BRK.B"


# ---------------------------------------------------------------- round trip

def test_holdings_round_trip(tmp_path):
    path = tmp_path / "etf_holdings.csv"
    holdings.save_holdings(path, [
        holdings.Holding("SPUS", "NVDA", "NVIDIA Corp", 0.1362, "2026-08-21"),
        holdings.Holding("HLAL", "NVDA", "NVIDIA Corp", 0.0812, "2026-08-21"),
        holdings.Holding("SPUS", "AAPL", "Apple Inc", 0.1186, "2026-08-21"),
    ])
    book = holdings.load_holdings(path)
    assert set(book) == {"NVDA", "AAPL"}
    assert len(book["NVDA"]) == 2
    assert holdings.as_of(book) == "2026-08-21"
    assert holdings.universe_symbols(book) == ["AAPL", "NVDA"]


def test_a_missing_holdings_file_is_empty_not_fatal(tmp_path):
    """The overlap badge is an enhancement; losing it must not lose the scan."""
    assert holdings.load_holdings(tmp_path / "nope.csv") == {}


# ---------------------------------------------------------------- overlap

def test_a_held_name_reports_every_fund_that_holds_it():
    book = {"NVDA": [holdings.Holding("SPUS", "NVDA", "NVIDIA", 0.136),
                     holdings.Holding("HLAL", "NVDA", "NVIDIA", 0.081)]}
    o = holdings.overlap_for("NVDA", book)
    assert o.any
    assert "SPUS" in o.note() and "HLAL" in o.note()
    assert o.total_weight == pytest.approx(0.217)


def test_an_unheld_name_is_stated_as_genuinely_new():
    o = holdings.overlap_for("TSLA", {"NVDA": []})
    assert not o.any
    assert "genuinely a new position" in o.note()


def test_balances_turn_a_percentage_into_an_amount():
    book = {"NVDA": [holdings.Holding("SPUS", "NVDA", "NVIDIA", 0.10)]}
    o = holdings.overlap_for("NVDA", book, {"SPUS": 500_000.0})
    assert o.value_inr == pytest.approx(50_000.0)
    assert "50,000" in o.note()


def test_no_balances_means_no_invented_amount():
    book = {"NVDA": [holdings.Holding("SPUS", "NVDA", "NVIDIA", 0.10)]}
    o = holdings.overlap_for("NVDA", book)
    assert o.value_inr == 0.0
    assert "exposure already" not in o.note()


def test_the_lookup_is_case_insensitive():
    book = {"NVDA": [holdings.Holding("SPUS", "NVDA", "NVIDIA", 0.10)]}
    assert holdings.overlap_for("nvda", book).any


def test_heavy_is_a_threshold_not_a_rejection():
    book = {"X": [holdings.Holding("SPUS", "X", "X", 0.05)]}
    o = holdings.overlap_for("X", book)
    assert o.heavy(0.03) is True
    assert o.heavy(0.08) is False


# ---------------------------------------------------------------- TCS

def test_no_tcs_inside_the_allowance():
    r = crossborder.tcs_on(500_000.0)
    assert r.tcs_inr == 0.0
    assert "No TCS" in r.note()


def test_tcs_applies_only_to_the_excess():
    r = crossborder.tcs_on(1_500_000.0)
    assert r.taxable_inr == pytest.approx(500_000.0)
    assert r.tcs_inr == pytest.approx(100_000.0)


def test_exactly_at_the_threshold_is_not_taxed():
    assert crossborder.tcs_on(crossborder.TCS_FREE_ALLOWANCE_INR).tcs_inr == 0.0


def test_the_allowance_is_cumulative_across_the_year():
    """It is per person per year, not per transaction or per bank."""
    r = crossborder.tcs_on(500_000.0, already_remitted_inr=800_000.0)
    assert r.taxable_inr == pytest.approx(300_000.0)
    assert r.tcs_inr == pytest.approx(60_000.0)


def test_tcs_is_described_as_recoverable_not_as_a_cost():
    note = crossborder.tcs_on(2_000_000.0).note()
    assert "NOT a cost" in note and "cash-flow" in note


# ---------------------------------------------------------------- estate tax

def test_us_situs_assets_count_and_irish_ucits_do_not():
    """
    The whole point. A UCITS holding the same American companies is not a
    US-situs asset and there is no look-through to what it owns.
    """
    e = crossborder.estate_exposure([
        Held("SPUS", 30_000.0, True),
        Held("HLAL", 20_000.0, True),
        Held("ISDW", 90_000.0, False),
    ])
    assert e.us_situs_usd == pytest.approx(50_000.0)
    assert e.non_us_situs_usd == pytest.approx(90_000.0)
    assert e.breached is False
    assert e.headroom_usd == pytest.approx(10_000.0)


def test_direct_us_shares_are_us_situs_exactly_as_an_etf_is():
    e = crossborder.estate_exposure([
        Held("SPUS", 40_000.0, True),
        Held("Direct US shares", 30_000.0, True),
    ])
    assert e.breached is True
    assert e.over_exemption_usd == pytest.approx(10_000.0)


def test_the_breach_note_names_the_missing_treaty():
    e = crossborder.estate_exposure([Held("SPUS", 200_000.0, True)])
    note = e.note()
    assert "no estate tax treaty" in note
    assert "Ireland-domiciled" in note


def test_indicative_tax_uses_the_top_rate_on_the_excess_only():
    e = crossborder.estate_exposure([Held("SPUS", 160_000.0, True)])
    assert e.over_exemption_usd == pytest.approx(100_000.0)
    assert e.indicative_tax_usd == pytest.approx(
        100_000.0 * crossborder.US_ESTATE_TOP_RATE)


def test_an_empty_portfolio_is_not_a_breach():
    e = crossborder.estate_exposure([])
    assert e.breached is False and e.indicative_tax_usd == 0.0


# ---------------------------------------------------------------- SDRT

def test_uk_shares_pay_stamp_duty(cfg):
    uk = markets_mod.get(cfg, markets_mod.UK)
    costs = crossborder.ticket_costs(uk, 10_000.0)
    assert costs.sdrt == pytest.approx(50.0)
    assert costs.pct_of_deployed == pytest.approx(0.005)


def test_an_lse_listed_irish_ucits_pays_no_stamp_duty(cfg):
    """SDRT follows the security, not the exchange."""
    uk = markets_mod.get(cfg, markets_mod.UK)
    assert crossborder.ticket_costs(uk, 10_000.0,
                                    uk_incorporated=False).total == 0.0


def test_us_tickets_have_no_stamp_duty(cfg):
    us = markets_mod.get(cfg, markets_mod.US)
    assert crossborder.ticket_costs(us, 10_000.0).total == 0.0


# ---------------------------------------------------------------- narrative

def test_withholding_differs_by_domicile():
    assert "25%" in crossborder.withholding_note("US")
    assert "15%" in crossborder.withholding_note("IE")
    assert "ACCUMULATING" in crossborder.withholding_note("IE")


def test_a_swing_trade_is_never_described_as_long_term():
    note = crossborder.cgt_note()
    assert "short-term by a wide margin" in note


def test_schedule_fa_is_flagged_as_a_calendar_year():
    assert "CALENDAR" in crossborder.SCHEDULE_FA_NOTE


def test_every_rate_carries_a_verification_date():
    assert crossborder.VERIFIED_ON in crossborder.DISCLAIMER
    assert "not advice" in crossborder.DISCLAIMER.lower()


def test_a_partial_refresh_keeps_the_other_funds_rows(tmp_path, monkeypatch):
    """
    Writing only what was fetched would silently delete the fund whose CDN was
    down, and the overlap check would then report "not held by any of your
    funds" for names you very much hold — reassurance manufactured by an
    outage.
    """
    path = tmp_path / "etf_holdings.csv"
    holdings.save_holdings(path, [
        holdings.Holding("SPUS", "NVDA", "NVIDIA", 0.136, "2026-08-01"),
        holdings.Holding("HLAL", "AAPL", "Apple", 0.115, "2026-08-01"),
    ])

    def only_spus_works(etf, timeout=30):
        if etf.upper() == "SPUS":
            return [holdings.Holding("SPUS", "MSFT", "Microsoft", 0.09,
                                     "2026-08-21")], "SPUS: 1 holding"
        return [], "HLAL refresh failed (boom)."

    monkeypatch.setattr(holdings, "fetch_holdings", only_spus_works)
    ok, detail = holdings.refresh_all(path)

    assert ok
    book = holdings.load_holdings(path)
    assert "MSFT" in book                       # the refreshed fund updated
    assert "NVDA" not in book                   # ...and replaced its own rows
    assert "AAPL" in book                       # the failed fund kept its book
    assert "NOT refreshed" in detail


def test_a_total_failure_leaves_the_file_alone(tmp_path, monkeypatch):
    path = tmp_path / "etf_holdings.csv"
    holdings.save_holdings(path, [
        holdings.Holding("SPUS", "NVDA", "NVIDIA", 0.136, "2026-08-01")])

    monkeypatch.setattr(holdings, "fetch_holdings",
                        lambda etf, timeout=30: ([], f"{etf} down"))
    ok, detail = holdings.refresh_all(path)

    assert not ok and "unchanged" in detail
    assert "NVDA" in holdings.load_holdings(path)
