"""
The scan, pointed at a foreign market.

Three claims:

  1. a foreign market with no trusted FX rate, or an unfunded capital pool,
     STANDS DOWN - and that is reported as a distinct state from "nothing
     qualified today", because they mean opposite things
  2. sizing crosses the currency boundary explicitly, so a $200 stock is not
     sized as though dollars were rupees
  3. a ticket carries both currencies, because the risk budget is denominated
     in rupees even when the trade is not
"""
from __future__ import annotations

from datetime import date

import pytest
from conftest import daily_bars, trending

from nifty_algo.config import Config
from nifty_algo.swing import fx as fx_mod
from nifty_algo.swing import markets as markets_mod
from nifty_algo.swing import news as news_mod
from nifty_algo.swing import scanner
from nifty_algo.swing.fundamentals import Fundamentals
from nifty_algo.swing.prices import PriceSet
from nifty_algo.swing.universe import Stock

TODAY = date(2026, 8, 20)
USDINR = 88.0

QUALIFYING_SEEDS = (2, 3, 5, 6, 7, 8, 10, 11, 12, 15, 17)


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.capital.foreign_capital_inr = 800_000.0
    for m in c.swing.markets.values():
        m.min_price = 1.0
        m.min_avg_turnover = 0.0
    return c


def make_stock(symbol, sector="Technology",
               industry="Software - Infrastructure") -> Stock:
    return Stock(symbol, f"{symbol} Inc", sector, industry, symbol)


def clean_fundamentals(symbol) -> Fundamentals:
    return Fundamentals(symbol=symbol, total_assets=1000.0, total_debt=100.0,
                        cash_and_investments=100.0, receivables=100.0,
                        market_cap=5e11, balance_sheet_date="2026-06-30",
                        currency="USD",
                        fetched_at="2026-08-19T09:00:00")


@pytest.fixture
def world(monkeypatch):
    state = {"stocks": [], "bars": {}, "rate": USDINR, "fx_fails": False}

    def fake_prices(tickers, cfg, market, benchmark=None, force_refresh=False,
                    divisors=None, progress=None):
        return PriceSet(
            bars={s: state["bars"][s] for s in tickers if s in state["bars"]},
            benchmark=daily_bars(trending(seed=99)),
            missing=[s for s in tickers if s not in state["bars"]])

    def fake_fundamentals(stocks, cfg, market, force_refresh=False,
                          progress=None):
        return {s.symbol: clean_fundamentals(s.symbol) for s in stocks}

    def fake_rate(market, cfg, force_refresh=False):
        if market.is_home:
            return fx_mod.Rate("INR", 1.0, __import__("datetime").datetime.now())
        if state["fx_fails"]:
            raise fx_mod.FxUnavailable(
                "could not fetch USDINR=X ... this market stands down for now.")
        return fx_mod.Rate(market.currency, state["rate"],
                           __import__("datetime").datetime.now())

    monkeypatch.setattr(scanner.prices_mod, "load_prices", fake_prices)
    monkeypatch.setattr(scanner.fundamentals_mod, "load_fundamentals",
                        fake_fundamentals)
    monkeypatch.setattr(scanner.fundamentals_mod, "cache_age_days",
                        lambda cfg: 1)
    monkeypatch.setattr(scanner.fx_mod, "rate_for_market", fake_rate)
    monkeypatch.setattr(
        scanner.news_mod, "fetch_for",
        lambda stocks, cfg, progress=None: {
            s.symbol: news_mod.NewsResult(s.symbol, available=True,
                                          note="stubbed") for s in stocks})
    return state


def populate(world, symbols, sector="Technology", price=200.0):
    """US-priced series: a $200 stock, not a Rs 200 one."""
    for i, symbol in enumerate(symbols):
        seed = QUALIFYING_SEEDS[i % len(QUALIFYING_SEEDS)]
        world["stocks"].append(make_stock(symbol, sector))
        world["bars"][symbol] = daily_bars(
            trending(seed=seed, start_price=price))
    return world["stocks"]


def run(cfg, world, key="us", **kw):
    market = markets_mod.get(cfg, key)
    return scanner.scan(cfg, market=market, universe=world["stocks"],
                        today=TODAY, skip_news=True, **kw)


# ---------------------------------------------------------------- stand down

def test_no_fx_rate_stands_the_market_down(cfg, world):
    populate(world, ["AAA", "BBB"])
    world["fx_fails"] = True
    result = run(cfg, world)

    assert result.stood_down
    assert result.picks == []
    assert any("stands down" in w for w in result.warnings)


def test_standing_down_is_not_the_same_as_nothing_qualifying(cfg, world):
    """
    "No picks" and "the scan never ran" look identical on a page that does not
    distinguish them, and mean opposite things about the market.
    """
    populate(world, ["AAA"])
    world["fx_fails"] = True
    stood_down = run(cfg, world)

    world["fx_fails"] = False
    world["stocks"].clear()
    world["stocks"].append(make_stock("ZZZ"))
    world["bars"]["ZZZ"] = daily_bars([100.0] * 120)   # flat: no setup
    quiet = run(cfg, world)

    assert stood_down.stood_down and not quiet.stood_down
    assert stood_down.picks == quiet.picks == []


def test_an_unfunded_foreign_pool_stands_the_market_down(cfg, world):
    populate(world, ["AAA"])
    cfg.capital.foreign_capital_inr = 0.0
    result = run(cfg, world)

    assert result.stood_down
    assert "foreign_capital_inr" in result.stood_down


def test_no_fx_failure_can_stand_india_down(cfg, world):
    """The domestic book must be unaffected by a foreign rate outage."""
    populate(world, ["AAA", "BBB"], price=200.0)
    world["fx_fails"] = True
    result = run(cfg, world, key="india")
    assert not result.stood_down


# ---------------------------------------------------------------- sizing

def test_the_budget_is_converted_before_it_is_divided(cfg, world):
    """
    THE BUG THIS EXISTS FOR. The budget is rupees, the stop is dollars.
    Without the conversion the size comes out ~88x too large.
    """
    populate(world, ["AAA"])
    result = run(cfg, world)
    if not result.picks:
        pytest.skip("no qualifying setup in the synthetic series")

    pick = result.picks[0]
    budget_inr = cfg.capital.risk_inr("foreign")
    budget_usd = budget_inr / USDINR

    assert pick.risk_amount <= budget_usd + 1e-6
    assert pick.risk_amount * USDINR == pytest.approx(pick.risk_inr, rel=1e-9)
    # And the naive, unconverted answer would have been far larger.
    assert pick.quantity < budget_inr / pick.setup.risk_points


def test_a_ticket_carries_both_currencies(cfg, world):
    populate(world, ["AAA"])
    result = run(cfg, world)
    if not result.picks:
        pytest.skip("no qualifying setup in the synthetic series")

    pick = result.picks[0]
    assert pick.currency == "USD" and pick.currency_symbol == "$"
    assert pick.market == "us"
    assert pick.fx_inr_per_unit == pytest.approx(USDINR)
    assert pick.deployed_inr == pytest.approx(pick.deployed * USDINR)
    assert pick.reward_inr == pytest.approx(pick.reward_amount * USDINR)


def test_the_risk_in_rupees_matches_the_rupee_budget(cfg, world):
    """
    The whole point of two pools and one formula: whatever currency the trade
    is in, the rupee risk is the rupee budget.
    """
    populate(world, ["AAA"])
    result = run(cfg, world)
    if not result.picks:
        pytest.skip("no qualifying setup in the synthetic series")
    assert result.picks[0].risk_inr <= cfg.capital.risk_inr("foreign") + 1e-6


def test_us_sizes_fractionally_and_uk_does_not(cfg, world):
    populate(world, ["AAA"])
    us = run(cfg, world)
    uk = run(cfg, world, key="uk")

    for pick in uk.picks:
        assert pick.quantity == float(int(pick.quantity)), \
            "the LSE order book does not fill fractions"
    for pick in us.picks:
        assert pick.quantity > 0


def test_a_rate_change_moves_the_size_and_nothing_else(cfg, world):
    populate(world, ["AAA"])
    a = run(cfg, world)
    world["rate"] = USDINR * 2
    b = run(cfg, world)
    if not (a.picks and b.picks):
        pytest.skip("no qualifying setup in the synthetic series")

    # Half the dollars for the same rupees, so half the shares...
    assert b.picks[0].quantity == pytest.approx(a.picks[0].quantity / 2, rel=0.02)
    # ...but the trade idea itself is untouched.
    assert b.picks[0].setup.entry == pytest.approx(a.picks[0].setup.entry)
    assert b.picks[0].setup.stop == pytest.approx(a.picks[0].setup.stop)


# ---------------------------------------------------------------- records

def test_the_record_carries_the_market_and_both_key_eras(cfg, world):
    """
    The journal is append-only. Records written before this change spell the
    fields `rupee_risk`/`rupee_reward`, so both are emitted and the rupee ones
    always hold rupees.
    """
    populate(world, ["AAA"])
    result = run(cfg, world)
    if not result.picks:
        pytest.skip("no qualifying setup in the synthetic series")

    record = result.picks[0].to_record()
    assert record["market"] == "us" and record["currency"] == "USD"
    # `to_record` rounds to 2dp for the journal, so compare at that precision.
    assert record["risk_amount"] == pytest.approx(
        result.picks[0].risk_amount, abs=0.01)
    assert record["rupee_risk"] == pytest.approx(
        result.picks[0].risk_inr, abs=0.01)
    assert record["fx_inr_per_unit"] == pytest.approx(USDINR)


def test_every_symbol_is_still_accounted_for(cfg, world):
    populate(world, [f"SYM{i}" for i in range(10)])
    result = run(cfg, world)
    assert result.accounts_for_everything()


def test_the_market_is_recorded_on_the_result(cfg, world):
    populate(world, ["AAA"])
    result = run(cfg, world)
    assert result.market == "us"
    assert "US" in result.market_label
    assert result.fx_note      # foreign scans state the rate they used


def test_india_reports_no_fx_note(cfg, world):
    populate(world, ["AAA"])
    assert run(cfg, world, key="india").fx_note == ""
