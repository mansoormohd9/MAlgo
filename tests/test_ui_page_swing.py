"""
The Daily picks page, rendered with picks actually present.

WHY THIS FILE EXISTS. `nifty_algo/ui/` renders and decides nothing, so it has
no unit tests - the logic all lives in headless modules that are tested
directly. That is right, and it left a hole: a page that raises `NameError` on
a line only reached when a pick is on screen is not testable from any headless
suite, because `scanner.scan()` never imports Streamlit.

That hole was not hypothetical. `_pick_card` referenced a module constant that
had been replaced by a per-market helper, and it shipped, because the only UI
check that had ever run rendered the page in its EMPTY state - where `render()`
returns early at `if result is None:`, a hundred lines above the fault.

So these tests do the one thing that catches that class of bug: put a real
`ScanResult` into session state and render the page for real. They assert
almost nothing about the output. `not at.exception` is the whole point -
everything below the early return has to actually execute.

SEEDING BARS IS LOAD-BEARING. The chart lookup at the bottom of `_pick_card` is
where the bug was, and it only runs when bars exist for the symbol. A test that
seeds picks but not bars skips the exact line that broke.

These are slow (~5s per app boot, four boots). That is the price of executing
the render path; if it ever needs cutting, mark them slow rather than deleting
them.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from conftest import daily_bars, seed_offline_broker, trending
from streamlit.testing.v1 import AppTest

from nifty_algo.config import Config
from nifty_algo.swing import fx as fx_mod
from nifty_algo.swing import markets as markets_mod
from nifty_algo.swing import news as news_mod
from nifty_algo.swing import scanner
from nifty_algo.swing.fundamentals import Fundamentals
from nifty_algo.swing.prices import PriceSet
from nifty_algo.swing.universe import Stock

#: `AppTest.from_file` resolves a relative path against the file that CALLS
#: it, which is this one - so a bare "app.py" looks for tests/app.py.
APP = str(Path(__file__).resolve().parent.parent / "app.py")

TODAY = date(2026, 8, 20)
USDINR = 88.0
GBPINR = 130.0

#: Seeds whose random walk actually produces a setup clearing the 2:1 floor.
QUALIFYING_SEEDS = (2, 3, 5, 6, 7, 8, 10, 11, 12, 15, 17)

#: Real tickers, because `data/etf_holdings.csv` is real: these are in SPUS and
#: HLAL, so the overlap badge on the pick card renders its "already held"
#: branch rather than its "genuinely new" one.
US_SYMBOLS = ("NVDA", "AAPL", "MSFT")
UK_SYMBOLS = ("SHEL", "AZN", "GSK")
INDIA_SYMBOLS = ("INFY", "TCS", "WIPRO")

#: A name the funds hold that fails the ratio screen, so the excluded table
#: renders its "the funds disagree with you" banner.
CONTESTED = "ABBV"


def _stock(symbol, sector="Technology",
           industry="Software - Infrastructure") -> Stock:
    return Stock(symbol, f"{symbol} Inc", sector, industry, symbol)


def _funds(symbol, debt=100.0) -> Fundamentals:
    return Fundamentals(symbol=symbol, total_assets=1000.0, total_debt=debt,
                        cash_and_investments=100.0, receivables=100.0,
                        market_cap=5e11, balance_sheet_date="2026-06-30",
                        fetched_at="2026-08-19T09:00:00")


def _build_result(monkeypatch, market_key: str, symbols, price: float):
    """
    A genuine `ScanResult`, produced by the real `scan()` over stubbed
    boundaries.

    Hand-assembling a `SwingPick` would test a shape the scanner never
    actually emits, which is precisely the kind of gap this file exists to
    close.
    """
    cfg = Config()
    cfg.capital.swing_capital_inr = 100_000.0
    cfg.capital.foreign_capital_inr = 800_000.0
    for m in cfg.swing.markets.values():
        m.min_price = 1.0
        m.min_avg_turnover = 0.0

    market = markets_mod.get(cfg, market_key)
    rate = 1.0 if market.is_home else (USDINR if market_key == "us" else GBPINR)

    bars = {}
    stocks = []
    for i, symbol in enumerate(symbols):
        stocks.append(_stock(symbol))
        bars[symbol] = daily_bars(
            trending(seed=QUALIFYING_SEEDS[i % len(QUALIFYING_SEEDS)],
                     start_price=price))
    # One name that fails the ratio screen, to populate `excluded_halal`.
    stocks.append(_stock(CONTESTED))
    bars[CONTESTED] = daily_bars(trending(seed=4, start_price=price))

    def fake_prices(tickers, cfg_, market_, benchmark=None, force_refresh=False,
                    divisors=None, progress=None):
        return PriceSet(
            bars={s: bars[s] for s in tickers if s in bars},
            benchmark=daily_bars(trending(seed=99)),
            missing=[s for s in tickers if s not in bars])

    def fake_fundamentals(stocks_, cfg_, market_, force_refresh=False,
                          progress=None):
        return {s.symbol: _funds(s.symbol,
                                 debt=900.0 if s.symbol == CONTESTED else 100.0)
                for s in stocks_}

    def fake_rate(market_, cfg_, force_refresh=False):
        if market_.is_home:
            return fx_mod.Rate("INR", 1.0, datetime.now())
        return fx_mod.Rate(market_.currency, rate, datetime.now())

    monkeypatch.setattr(scanner.prices_mod, "load_prices", fake_prices)
    monkeypatch.setattr(scanner.fundamentals_mod, "load_fundamentals",
                        fake_fundamentals)
    monkeypatch.setattr(scanner.fundamentals_mod, "cache_age_days",
                        lambda cfg_: 1)
    monkeypatch.setattr(scanner.fx_mod, "rate_for_market", fake_rate)
    monkeypatch.setattr(
        scanner.news_mod, "fetch_for",
        lambda stocks_, cfg_, progress=None: {
            s.symbol: news_mod.NewsResult(s.symbol, available=True,
                                          note="stubbed") for s in stocks_})

    result = scanner.scan(cfg, market=market, universe=stocks, today=TODAY,
                          skip_news=True)
    return cfg, result, bars


def _render(cfg, market_key, result, bars):
    """Boot the app with the result already in session state, open the page."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["cfg"] = cfg
    at.session_state["swing_market"] = market_key
    at.session_state[f"swing_result_{market_key}"] = result
    # The Arm panel on every pick card asks the broker for free cash, so
    # without this the render path dials Zerodha. See conftest.
    seed_offline_broker(at, cfg)
    # Bars are what make the chart branch run - see the module docstring.
    at.session_state[f"swing_bars_{market_key}"] = bars
    at.run()
    assert not at.exception, _first(at)
    at.sidebar.radio[0].set_value("Daily picks").run()
    return at


def _first(at) -> str:
    return "; ".join(str(e.value)[:400] for e in (at.exception or []))


# ---------------------------------------------------------------- the markets

@pytest.mark.parametrize("market_key, symbols, price", [
    ("us", US_SYMBOLS, 200.0),
    ("uk", UK_SYMBOLS, 25.0),
    ("india", INDIA_SYMBOLS, 1500.0),
])
def test_the_page_renders_with_picks_on_screen(monkeypatch, market_key,
                                               symbols, price):
    """
    THE REGRESSION THIS FILE EXISTS FOR.

    Every panel below `render()`'s early return has to execute: the pick card,
    the chart lookup, the overlap badge, the two-standard comparison, the
    excluded table and - for a foreign market - the cross-border panel.
    """
    cfg, result, bars = _build_result(monkeypatch, market_key, symbols, price)
    assert result.picks, "the fixture must produce picks or this tests nothing"

    at = _render(cfg, market_key, result, bars)

    assert not at.exception, _first(at)
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == "Daily picks"


def test_a_foreign_ticket_shows_both_currencies(monkeypatch):
    """The rupee line is rendered inside the card's HTML, so it needs the
    render path to have run rather than just the scan."""
    cfg, result, bars = _build_result(monkeypatch, "us", US_SYMBOLS, 200.0)
    at = _render(cfg, "us", result, bars)

    assert not at.exception, _first(at)
    body = " ".join(m.value for m in at.markdown)
    assert "In rupees" in body


def test_the_india_page_shows_no_cross_border_panel(monkeypatch):
    """
    None of it applies to a domestic trade, and the home-market branches are
    as unexercised as the foreign ones were.
    """
    cfg, result, bars = _build_result(monkeypatch, "india", INDIA_SYMBOLS,
                                      1500.0)
    at = _render(cfg, "india", result, bars)

    assert not at.exception, _first(at)
    body = " ".join(m.value for m in at.markdown)
    assert "In rupees:" not in body


def test_a_stood_down_market_renders_its_banner(monkeypatch):
    """
    The other never-executed path. "The scan did not run" and "nothing
    qualified" are different statements and the page has to survive making
    the first one.
    """
    cfg, result, bars = _build_result(monkeypatch, "us", US_SYMBOLS, 200.0)
    result.stood_down = ("US sizes off the foreign capital pool, which is set "
                         "to ₹0.")
    result.picks = []

    at = _render(cfg, "us", result, bars)

    assert not at.exception, _first(at)
    assert at.title[0].value == "Daily picks"
