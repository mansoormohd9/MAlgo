"""
The market registry, and the two silent-wrongness bugs it exists to prevent.

Both of the failures under test here produce a plausible number rather than an
exception, which is what makes them worth a test file of their own:

  1. a benchmark cached under one market being served to another, so relative
     strength is measured against the wrong index and nothing says so
  2. an LSE pence quote treated as pounds, so every price floor, turnover
     figure and position size in the UK book is out by a factor of 100

Neither would fail loudly. Both would just be wrong.
"""
from __future__ import annotations

import pandas as pd
import pytest
from conftest import daily_bars, trending

from nifty_algo.config import Config
from nifty_algo.swing import markets as markets_mod
from nifty_algo.swing import prices as prices_mod


@pytest.fixture
def cfg() -> Config:
    return Config()


# ---------------------------------------------------------------- registry

def test_every_registered_market_is_complete(cfg):
    for key in markets_mod.keys(cfg):
        m = markets_mod.get(cfg, key)
        assert m.key == key
        assert m.universe_csv and m.benchmark_ticker and m.currency
        assert m.symbol and m.turnover_divisor > 0
        assert m.taxonomy in (markets_mod.TAXONOMY_NSE, markets_mod.TAXONOMY_GICS)
        assert m.capital_pool in (markets_mod.POOL_HOME,
                                  markets_mod.POOL_SWING_IN,
                                  markets_mod.POOL_FOREIGN)
        # Every registered pool must be one CapitalConfig can actually price,
        # or the market is scannable and unsizeable at the same time.
        cfg.capital.capital_inr(m.capital_pool)


def test_an_unknown_market_raises_rather_than_defaulting(cfg):
    """
    Falling back to India on a typo would scan the wrong exchange and produce
    a page full of entirely plausible, entirely wrong tickets.
    """
    with pytest.raises(markets_mod.UnknownMarket):
        markets_mod.get(cfg, "nasdaq")


def test_only_india_is_a_home_market(cfg):
    assert markets_mod.get(cfg, markets_mod.INDIA).is_home
    assert not markets_mod.get(cfg, markets_mod.US).is_home
    assert not markets_mod.get(cfg, markets_mod.UK).is_home


def test_each_config_owns_its_markets():
    """A threshold moved in one test must not leak into the next."""
    a, b = Config(), Config()
    a.swing.markets[markets_mod.US].min_price = 999.0
    assert b.swing.markets[markets_mod.US].min_price != 999.0


def test_qualified_keys_cannot_collide_across_markets(cfg):
    india = markets_mod.get(cfg, markets_mod.INDIA)
    us = markets_mod.get(cfg, markets_mod.US)
    assert india.qualified("INFY") != us.qualified("INFY")


# ------------------------------------------------- the benchmark cache bug

def test_the_benchmark_key_carries_its_ticker():
    assert prices_mod.benchmark_key("^NSEI") != prices_mod.benchmark_key("^GSPC")


def test_each_market_caches_to_its_own_file():
    assert prices_mod.cache_name("india") != prices_mod.cache_name("us")


def test_a_us_cache_cannot_satisfy_an_india_scan(cfg, tmp_path, monkeypatch):
    """
    THE REGRESSION THIS MODULE EXISTS FOR.

    Under the old single-file cache with a fixed `__BENCHMARK__` key, scanning
    the US and then India found that key present, accepted the cache, and
    computed India's relative strength against the S&P 500. No error, no
    warning, a wrong answer on every row.
    """
    cfg.swing.cache_dir = str(tmp_path)
    india = markets_mod.get(cfg, markets_mod.INDIA)
    us = markets_mod.get(cfg, markets_mod.US)

    nifty = daily_bars(trending(seed=1))
    spx = daily_bars(trending(seed=2))
    downloads: list[list[str]] = []

    def fake_download(tickers, period_days):
        downloads.append(list(tickers))
        frames = {}
        for t in tickers:
            frames[t] = spx if t == us.benchmark_ticker else nifty
        return pd.concat(frames, axis=1)

    monkeypatch.setattr(prices_mod, "_download", fake_download)

    prices_mod.load_prices({"AAPL": "AAPL"}, cfg, us)
    india_set = prices_mod.load_prices({"INFY": "INFY.NS"}, cfg, india)

    # It went back to the network for India rather than reusing the US file...
    assert any(india.benchmark_ticker in batch for batch in downloads)
    # ...and the benchmark it ended up with is India's, not the S&P's.
    assert india_set.benchmark is not None
    assert india_set.benchmark["close"].iloc[-1] == pytest.approx(
        nifty["close"].iloc[-1])


# ------------------------------------------------------------ the pence bug

def test_a_pence_quote_is_divided_once():
    raw = daily_bars([2500.0, 2510.0, 2490.0])
    out = prices_mod._extract(raw, "SHEL.L", divisor=100.0)
    assert out is not None
    assert out["close"].iloc[0] == pytest.approx(25.00)
    assert out["high"].iloc[0] == pytest.approx(raw["high"].iloc[0] / 100.0)


def test_volume_is_never_divided():
    """
    Turnover is close x volume. Dividing both would undo the correction and
    leave the liquidity figure exactly as wrong as before.
    """
    raw = daily_bars([2500.0, 2510.0], volume=1_234_000.0)
    out = prices_mod._extract(raw, "SHEL.L", divisor=100.0)
    assert out["volume"].iloc[0] == pytest.approx(1_234_000.0)


def test_a_divisor_of_one_changes_nothing():
    raw = daily_bars([100.0, 101.0])
    out = prices_mod._extract(raw, "INFY.NS", divisor=1.0)
    assert out["close"].iloc[0] == pytest.approx(raw["close"].iloc[0])


def test_the_benchmark_is_never_divided(cfg, tmp_path, monkeypatch):
    """
    An index is quoted in points, not in the minor unit of its constituents.
    Dividing ^FTSE by 100 would cancel out of a ratio and hide here, so it is
    asserted directly.
    """
    cfg.swing.cache_dir = str(tmp_path)
    uk = markets_mod.get(cfg, markets_mod.UK)
    assert uk.price_divisor == 100.0

    bars = daily_bars([8000.0, 8010.0, 8020.0])

    def fake_download(tickers, period_days):
        return pd.concat({t: bars for t in tickers}, axis=1)

    monkeypatch.setattr(prices_mod, "_download", fake_download)
    out = prices_mod.load_prices({"SHEL": "SHEL.L"}, cfg, uk)

    assert out.benchmark["close"].iloc[-1] == pytest.approx(8020.0)
    assert out.bars["SHEL"]["close"].iloc[-1] == pytest.approx(80.20)


def test_a_per_symbol_divisor_overrides_the_market(cfg, tmp_path, monkeypatch):
    """
    Not every LSE line is in pence. Where Yahoo says GBP, the market default
    must lose to the fact.
    """
    cfg.swing.cache_dir = str(tmp_path)
    uk = markets_mod.get(cfg, markets_mod.UK)
    bars = daily_bars([2500.0, 2510.0])

    monkeypatch.setattr(
        prices_mod, "_download",
        lambda tickers, period_days: pd.concat({t: bars for t in tickers}, axis=1))

    out = prices_mod.load_prices({"XYZ": "XYZ.L"}, cfg, uk,
                                 divisors={"XYZ": 1.0})
    assert out.bars["XYZ"]["close"].iloc[-1] == pytest.approx(2510.0)


# ---------------------------------------------------------------- turnover

def test_turnover_is_expressed_in_the_markets_own_unit():
    bars = daily_bars([100.0] * 25, volume=1_000_000.0)
    crore = prices_mod.avg_turnover(bars, 1e7)
    millions = prices_mod.avg_turnover(bars, 1e6)
    assert crore == pytest.approx(10.0)
    assert millions == pytest.approx(100.0)


def test_turnover_crore_still_means_crore():
    bars = daily_bars([100.0] * 25, volume=1_000_000.0)
    assert prices_mod.turnover_crore(bars) == pytest.approx(
        prices_mod.avg_turnover(bars, 1e7))
