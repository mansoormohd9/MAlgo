"""Tests for the pure signal primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_algo import signals as sig

from conftest import flat_bars, make_bars, append_bar


def test_vwap_resets_each_session():
    """
    A VWAP that runs across days is meaningless intraday. This asserts each
    session's first bar starts its own average rather than continuing the
    previous day's.
    """
    day1 = pd.date_range("2026-03-10 09:15", periods=10, freq="5min")
    day2 = pd.date_range("2026-03-11 09:15", periods=10, freq="5min")
    idx = day1.append(day2)

    df = pd.DataFrame({
        "open": [100.0] * 10 + [200.0] * 10,
        "high": [100.0] * 10 + [200.0] * 10,
        "low": [100.0] * 10 + [200.0] * 10,
        "close": [100.0] * 10 + [200.0] * 10,
        "volume": [1000.0] * 20,
    }, index=idx)

    vw = sig.vwap(df)
    assert vw.iloc[9] == pytest.approx(100.0)
    # If VWAP leaked across the boundary the day-2 opener would be pulled
    # toward 100; a clean reset puts it exactly at 200.
    assert vw.iloc[10] == pytest.approx(200.0)
    assert vw.iloc[-1] == pytest.approx(200.0)


def test_vwap_is_volume_weighted_not_a_mean():
    idx = pd.date_range("2026-03-10 09:15", periods=2, freq="5min")
    df = pd.DataFrame({
        "open": [100.0, 200.0], "high": [100.0, 200.0],
        "low": [100.0, 200.0], "close": [100.0, 200.0],
        "volume": [1.0, 9.0],
    }, index=idx)
    # Weighted toward the heavy second bar: (100*1 + 200*9) / 10 = 190
    assert sig.vwap(df).iloc[-1] == pytest.approx(190.0)


def test_narrowest_range_n_finds_the_compressed_bar():
    rows = [{"open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}
            for _ in range(6)]
    rows.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
    df = make_bars(rows)
    nr = sig.narrowest_range_n(df, n=7)
    assert bool(nr.iloc[-1]), "the 2-point bar should be the narrowest of 7"
    assert not bool(nr.iloc[-2])


def test_sweep_reclaim_detects_bullish_sweep():
    """Wick well below the level, close back above it."""
    df = flat_bars(n=5, price=26_000)
    level = 25_990.0
    df = append_bar(df, open_=26_000, high=26_005, low=25_960,
                    close=26_002, volume=100_000)
    assert sig.sweep_reclaim(df, level, current_atr=50.0,
                             min_wick_atr=0.35) == "bullish"


def test_sweep_reclaim_detects_bearish_sweep():
    df = flat_bars(n=5, price=26_000)
    level = 26_010.0
    df = append_bar(df, open_=26_000, high=26_040, low=25_995,
                    close=26_005, volume=100_000)
    assert sig.sweep_reclaim(df, level, current_atr=50.0,
                             min_wick_atr=0.35) == "bearish"


def test_sweep_reclaim_rejects_a_shallow_wick():
    """A 2-point poke past a level is noise, not a stop hunt."""
    df = flat_bars(n=5, price=26_000)
    df = append_bar(df, open_=26_000, high=26_005, low=25_988,
                    close=26_002, volume=100_000)
    assert sig.sweep_reclaim(df, 25_990.0, current_atr=50.0,
                             min_wick_atr=0.35) is None


def test_sweep_reclaim_rejects_a_close_that_stayed_outside():
    """Wicked below AND closed below is a breakdown, not a reclaim."""
    df = flat_bars(n=5, price=26_000)
    df = append_bar(df, open_=26_000, high=26_002, low=25_950,
                    close=25_960, volume=100_000)
    assert sig.sweep_reclaim(df, 25_990.0, current_atr=50.0) is None


def test_gap_metrics_signs():
    gap, gap_atr = sig.gap_metrics(26_100.0, 26_000.0, 50.0)
    assert gap == pytest.approx(100.0)
    assert gap_atr == pytest.approx(2.0)

    gap, gap_atr = sig.gap_metrics(25_900.0, 26_000.0, 50.0)
    assert gap == pytest.approx(-100.0)
    assert gap_atr == pytest.approx(-2.0)


def test_gap_metrics_handles_zero_atr():
    gap, gap_atr = sig.gap_metrics(26_100.0, 26_000.0, 0.0)
    assert gap == pytest.approx(100.0)
    assert gap_atr == 0.0


def test_trend_state_directions():
    up = make_bars([{"open": 100 + i, "high": 101 + i, "low": 99 + i,
                     "close": 100.8 + i, "volume": 1000} for i in range(40)])
    assert sig.trend_state(up, 9, 21).iloc[-1] == 1

    down = make_bars([{"open": 200 - i, "high": 201 - i, "low": 199 - i,
                       "close": 199.2 - i, "volume": 1000} for i in range(40)])
    assert sig.trend_state(down, 9, 21).iloc[-1] == -1


def test_consecutive_closes_counts_direction():
    up = make_bars([{"open": 100, "high": 102, "low": 99,
                     "close": 100 + i, "volume": 1000} for i in range(10)])
    assert sig.consecutive_closes(up, direction=1, lookback=5) == 5
    assert sig.consecutive_closes(up, direction=-1, lookback=5) == 0


def test_vwap_bands_bracket_the_vwap():
    df = flat_bars(n=30, seed=5)
    lower, vw, upper = sig.vwap_bands(df, sigma=1.0)
    tail = slice(-10, None)
    assert (lower.iloc[tail] <= vw.iloc[tail]).all()
    assert (upper.iloc[tail] >= vw.iloc[tail]).all()
