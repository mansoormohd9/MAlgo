"""
The market-regime filter, the setup switch, and the breakeven override.

These are the three levers `swing/experiment.py` sweeps. Each one is real
config that the LIVE scanner honours too - a backtest-only switch would let
the sweep measure a book that cannot be traded, which is invariant 1 in
another costume.

The look-ahead test here matters as much as the arithmetic ones: a regime
filter read off the benchmark's LAST bar rather than the decision bar would
know on Monday what the index closed at on Friday, and would improve every
result while doing it.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.swing import market_regime as regime_mod
from nifty_algo.swing import setup as setup_mod


def _bench(closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range(datetime(2024, 1, 1), periods=len(closes))
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c,
                         "volume": np.zeros(len(c))}, index=idx)


# ---------------------------------------------------------------- the filter

def test_off_by_default_lets_every_day_through():
    """
    Zero means no filter, and no filter must mean no change.

    Every result recorded before this module existed was produced with the
    gate absent; if `0` blocked anything, those numbers would silently stop
    being comparable to new ones.
    """
    assert Config().swing.regime_ma_days == 0
    reading = regime_mod.benchmark_state(_bench([100.0] * 300), 0)
    assert reading.ok
    assert not reading.enabled


def test_above_the_average_passes_and_below_it_stands_aside():
    rising = _bench(list(np.linspace(100.0, 200.0, 300)))
    falling = _bench(list(np.linspace(200.0, 100.0, 300)))

    up = regime_mod.benchmark_state(rising, 200)
    down = regime_mod.benchmark_state(falling, 200)

    assert up.ok and "above" in up.reason
    assert not down.ok and "below" in down.reason
    assert "stands aside" in down.reason
    assert up.moving_average < up.close
    assert down.moving_average > down.close


def test_a_missing_or_short_benchmark_blocks_rather_than_passes():
    """
    Fail closed, like `fx.py` and the halal screen.

    "The market is fine" is not something a filter may conclude from data it
    does not have - and a filter that stops filtering on the days its feed
    breaks is worse than no filter, because it looks like one.
    """
    assert not regime_mod.benchmark_state(None, 200).ok
    assert not regime_mod.benchmark_state(_bench([100.0] * 50), 200).ok
    assert "need 200" in regime_mod.benchmark_state(_bench([100.0] * 50), 200).reason


def test_the_reading_uses_only_bars_up_to_the_decision_day():
    """
    LOOK-AHEAD. The index rises for a year and then collapses; asked about a
    day before the collapse, the filter must not know about it.
    """
    closes = list(np.linspace(100.0, 200.0, 300)) + [80.0] * 20
    bench = _bench(closes)
    before = bench.index[299]

    assert regime_mod.benchmark_state(bench, 200, as_of=before).ok
    assert not regime_mod.benchmark_state(bench, 200).ok      # last bar: after


# ---------------------------------------------------------------- the switch

def test_enabled_setups_selects_which_archetypes_may_fire():
    """
    The keys in the tuple are the keys `detect` can return, and disabling all
    of them produces no setup rather than an exception.
    """
    cfg = Config()
    assert {k for k, _ in setup_mod.BUILDERS} == set(cfg.swing.enabled_setups)

    rng = np.random.default_rng(7)
    n = 200
    c = 100 * np.cumprod(1 + rng.normal(0.001, 0.012, n))
    df = pd.DataFrame({
        "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
    }, index=pd.bdate_range(datetime(2024, 1, 2), periods=n))

    cfg.swing.enabled_setups = ()
    found, why = setup_mod.detect("AAA", df, cfg)
    assert found is None and why


# ------------------------------------------------------------- the override

def test_the_swing_breakeven_rung_can_be_moved_without_touching_the_option_book():
    """
    None inherits, a number overrides, and the intraday book never moves.

    +1R on a 5-minute bar and +1R on a daily bar are different distances
    relative to the noise each has to survive, which is the same reason
    `trail_atr_multiple` is already overridden here.
    """
    from nifty_algo.swing.book import swing_trade

    cfg = Config()
    assert cfg.swing.breakeven_at_r is None
    assert swing_trade(cfg).breakeven_at_r == cfg.trade.breakeven_at_r

    cfg.swing.breakeven_at_r = 1.5
    assert swing_trade(cfg).breakeven_at_r == pytest.approx(1.5)
    assert cfg.trade.breakeven_at_r == pytest.approx(1.0)   # untouched
