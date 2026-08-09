"""
The look-ahead bias tests.

These are the most important tests in the suite. Every other failure produces
a wrong answer you can see; look-ahead bias produces a *flattering* answer you
cannot, and it invalidates every backtest number silently.

The specific hazard is `signals.find_pivots()`. It uses a CENTRED rolling
window - a bar is a swing high if it exceeds `lookback` bars on both sides.
Live, that fact is unknowable until `lookback` more bars have printed. If the
backtester hands a strategy the whole frame, the strategy can build a level
from a pivot that had not been confirmed yet and "trade" its breakout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_algo import signals as sig
from nifty_algo.data.csv_feed import BarReplayer

from conftest import flat_bars


def test_replayer_window_never_contains_future_bars():
    """The window handed to a strategy ends at the decision bar. Full stop."""
    df = flat_bars(n=80)
    replayer = BarReplayer(df, warmup=30)

    for i in range(30, len(df)):
        window = replayer.window_at(i)
        assert window.index[-1] == df.index[i], (
            f"window at {i} ends at {window.index[-1]}, expected {df.index[i]}")
        assert len(window) == min(i + 1, replayer.max_window)
        assert window.index.max() <= df.index[i], "future bar leaked into the window"


def test_replayer_iteration_is_strictly_causal():
    df = flat_bars(n=60)
    seen = []
    for window in BarReplayer(df, warmup=30):
        seen.append(window.index[-1])
        assert (window.index <= window.index[-1]).all()
    assert seen == list(df.index[30:])


def test_pivot_is_not_visible_before_it_is_confirmable():
    """
    Build a frame with one unmistakable swing high, then assert the replayer
    hides it until `lookback` bars have printed past it.

    This is the exact bug the BarReplayer exists to prevent: if this test
    fails, every backtest in the project is measuring a system that can see
    the future.
    """
    lookback = 5
    n = 41
    peak_idx = 20

    rows = []
    for i in range(n):
        base = 26_000.0 + (50.0 if i == peak_idx else 0.0)
        rows.append({"open": base, "high": base + 2, "low": base - 2,
                     "close": base, "volume": 100_000})
    df = pd.DataFrame(rows, index=pd.date_range("2026-03-10 09:15",
                                                periods=n, freq="5min"))

    # With the FULL frame the pivot is detectable - this is what a naive
    # backtester would see, and it is the wrong answer.
    is_high_full, _ = sig.find_pivots(df, lookback)
    assert bool(is_high_full.iloc[peak_idx]), "fixture failed to build a pivot"

    replayer = BarReplayer(df, warmup=10)

    # On the bar the peak forms, and for the next lookback-1 bars, it must NOT
    # be detectable - the confirming bars have not printed.
    for i in range(peak_idx, peak_idx + lookback):
        window = replayer.window_at(i)
        is_high, _ = sig.find_pivots(window, lookback)
        assert not bool(is_high.reindex(window.index).fillna(False).loc[df.index[peak_idx]]), (
            f"pivot at bar {peak_idx} was visible at bar {i} — "
            f"{peak_idx + lookback - i} confirming bars had not printed yet. "
            f"This is look-ahead bias.")

    # Once lookback bars have passed, it is legitimately knowable.
    window = replayer.window_at(peak_idx + lookback)
    is_high, _ = sig.find_pivots(window, lookback)
    assert bool(is_high.loc[df.index[peak_idx]]), (
        "pivot should be confirmable once lookback bars have printed")


def test_build_levels_respects_the_window_it_is_given():
    """A level built from a window can only reflect that window's bars."""
    df = flat_bars(n=120, seed=11)
    early = BarReplayer(df, warmup=40).window_at(50)
    late = BarReplayer(df, warmup=40).window_at(110)

    early_levels = sig.build_levels(early, lookback=5, min_touches=2)
    for lv in early_levels:
        assert early["low"].min() - 50 <= lv.price <= early["high"].max() + 50, (
            "a level fell outside the price range of the window that produced it")

    # More history can only add information, never remove the window boundary.
    assert len(late) > len(early)
