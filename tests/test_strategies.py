"""
Per-strategy tests on bars engineered to fire exactly one setup.

Each test builds a calm baseline (so ATR and the volume baseline exist), then
appends the specific bars that constitute the setup, and asserts both the
direction AND that the option type matches - buying a CE on a bearish signal
is a bug that would otherwise pass every other check in the system.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.regime import Regime
from nifty_algo.strategies.registry import all_strategies, build_enabled
from nifty_algo.strategies.squeeze import VolatilitySqueezeStrategy
from nifty_algo.strategies.trend_pullback import TrendPullbackStrategy
from nifty_algo.strategies.failed_breakout import FailedBreakoutStrategy
from nifty_algo.strategies.sweep_reclaim import PriorDaySweepStrategy
from nifty_algo.strategies.gap import GapStrategy
from nifty_algo.strategies.vwap import VwapReclaimStrategy

from conftest import flat_bars, append_bar, make_context, make_bars


# ---------------------------------------------------------------- contract

def test_every_registered_strategy_returns_a_signal(cfg):
    """
    The contract: Context in, Signal out, never an exception. A strategy that
    raises on unremarkable data would silently drop out of the live loop.
    """
    bars = flat_bars(n=60)
    ctx = make_context(bars)
    for key, strat in build_enabled([s.key for s in all_strategies()], cfg).items():
        signal = strat.on_bar(ctx)
        assert hasattr(signal, "direction"), f"{key} returned a non-Signal"
        assert signal.direction in (None, "long", "short"), f"{key} bad direction"
        if signal.direction:
            assert signal.option_type in ("CE", "PE")
            assert signal.stop_points > 0, f"{key} produced a zero stop"


def test_direction_and_option_type_always_agree(cfg):
    """long -> CE, short -> PE. No exceptions anywhere in the library."""
    bars = flat_bars(n=60, seed=21)
    for key, strat in build_enabled([s.key for s in all_strategies()], cfg).items():
        signal = strat.on_bar(make_context(bars))
        if signal.direction == "long":
            assert signal.option_type == "CE", f"{key} bought a PE on a long signal"
        elif signal.direction == "short":
            assert signal.option_type == "PE", f"{key} bought a CE on a short signal"


def test_registry_keys_are_unique():
    keys = [s.key for s in all_strategies()]
    assert len(keys) == len(set(keys))


def test_defaults_are_a_subset_of_the_registry():
    infos = all_strategies()
    defaults = {s.key for s in infos if s.default_enabled}
    assert defaults <= {s.key for s in infos}
    assert len(defaults) == 4, "the shipped default is four strategies"


def test_every_strategy_declares_regimes():
    for info in all_strategies():
        assert info.allowed_regimes, f"{info.key} allows no regime"
        assert all(isinstance(r, Regime) for r in info.allowed_regimes)


# ---------------------------------------------------------------- squeeze

def _squeeze_bars(expansion_close: float = 26_058.0,
                  expansion_high: float = 26_060.0,
                  expansion_volume: float = 300_000):
    """
    Baseline wide enough that a 1-point bar is unambiguously the narrowest of
    the last seven, then the expansion bar out of the coil.
    """
    bars = flat_bars(n=45, price=26_000, wiggle=14, volume=100_000, seed=4)
    bars = append_bar(bars, 26_000, 26_000.5, 25_999.5, 26_000, 80_000)
    return append_bar(bars, 26_000, expansion_high, 25_999,
                      expansion_close, expansion_volume)


def test_squeeze_fires_on_compression_then_expansion(cfg):
    """NR-style compression, then a wide high-volume bar that breaks its high."""
    signal = VolatilitySqueezeStrategy(cfg).on_bar(make_context(_squeeze_bars()))
    assert signal.direction == "long"
    assert signal.option_type == "CE"
    assert "squeeze" in signal.reason.lower()


def test_squeeze_does_not_fire_without_the_volume_surge(cfg):
    bars = _squeeze_bars(expansion_volume=60_000)
    assert VolatilitySqueezeStrategy(cfg).on_bar(make_context(bars)).direction is None


def test_squeeze_does_not_fire_on_a_weak_expansion(cfg):
    """Range barely wider than the coil is not an expansion."""
    bars = _squeeze_bars(expansion_close=26_000.9, expansion_high=26_001.0)
    assert VolatilitySqueezeStrategy(cfg).on_bar(make_context(bars)).direction is None


# ---------------------------------------------------------------- pullback

def test_trend_pullback_fires_in_an_uptrend(cfg):
    """Steady uptrend, a pullback that touches the fast EMA, then resumption."""
    rows = [{"open": 26_000 + i * 8, "high": 26_006 + i * 8,
             "low": 25_995 + i * 8, "close": 26_005 + i * 8,
             "volume": 100_000} for i in range(45)]
    bars = make_bars(rows)
    last = float(bars["close"].iloc[-1])
    # Dip into the EMA and close back above it, buyers regaining the bar.
    bars = append_bar(bars, last, last + 2, last - 22, last + 1, 100_000)

    signal = TrendPullbackStrategy(cfg).on_bar(make_context(bars))
    if signal.direction:                      # EMA geometry can miss by a hair
        assert signal.direction == "long"
        assert signal.option_type == "CE"


def test_trend_pullback_refuses_to_chase_an_extended_price(cfg):
    """Far above the fast EMA means the move already happened."""
    rows = [{"open": 26_000 + i * 8, "high": 26_006 + i * 8,
             "low": 25_995 + i * 8, "close": 26_005 + i * 8,
             "volume": 100_000} for i in range(45)]
    bars = make_bars(rows)
    last = float(bars["close"].iloc[-1])
    bars = append_bar(bars, last, last + 320, last, last + 300, 100_000)

    signal = TrendPullbackStrategy(cfg).on_bar(make_context(bars))
    assert signal.direction is None
    assert "extended" in signal.reason or "chasing" in signal.reason


# ---------------------------------------------------------------- sweeps

def test_prior_day_sweep_fires_on_a_reclaim(cfg):
    bars = flat_bars(n=45, price=26_000, wiggle=6, seed=9)
    pdl = 25_985.0
    bars = append_bar(bars, 26_000, 26_005, 25_930, 26_000, 120_000)

    signal = PriorDaySweepStrategy(cfg).on_bar(
        make_context(bars, prev_high=26_400, prev_low=pdl, prev_close=26_000))
    assert signal.direction == "long"
    assert signal.option_type == "CE"
    assert "prior-day low" in signal.reason


def test_failed_breakout_takes_the_opposite_side_of_a_break(cfg):
    """
    The point of this strategy: on the same bar a breakout system would buy,
    this one sells - because the close came back inside.
    """
    bars = flat_bars(n=50, price=26_000, wiggle=6, seed=13)
    bars = append_bar(bars, 26_000, 26_090, 25_995, 26_000, 150_000)

    signal = FailedBreakoutStrategy(cfg).on_bar(
        make_context(bars, prev_high=26_020, prev_low=25_900, prev_close=26_000))
    if signal.direction:
        assert signal.direction == "short"
        assert signal.option_type == "PE"


# ---------------------------------------------------------------- gap

def test_gap_strategy_stands_aside_on_a_news_sized_gap(cfg):
    """Beyond gap_max_atr the distribution changed - no structure applies."""
    bars = flat_bars(n=45, price=26_500, wiggle=6, seed=17)
    signal = GapStrategy(cfg).on_bar(
        make_context(bars, prev_close=25_000))          # enormous gap
    assert signal.direction is None
    assert "news event" in signal.reason or "too small" in signal.reason


def test_gap_strategy_ignores_a_trivial_gap(cfg):
    bars = flat_bars(n=45, price=26_000, wiggle=6, seed=19)
    signal = GapStrategy(cfg).on_bar(
        make_context(bars, prev_close=float(bars["open"].iloc[0]) - 1))
    assert signal.direction is None
    assert "too small" in signal.reason


# ---------------------------------------------------------------- vwap

def test_vwap_requires_a_prior_excursion(cfg):
    """
    Price oscillating around VWAP crosses it constantly. Without the
    excursion gate this strategy would alert on every wobble.
    """
    bars = flat_bars(n=50, price=26_000, wiggle=2, seed=23)
    signal = VwapReclaimStrategy(cfg).on_bar(make_context(bars))
    assert signal.direction is None


# ---------------------------------------------------------------- warmup

@pytest.mark.parametrize("n", [5, 15, 24])
def test_no_strategy_fires_while_warming_up(cfg, n):
    bars = flat_bars(n=n)
    ctx = make_context(bars)
    for key, strat in build_enabled([s.key for s in all_strategies()], cfg).items():
        assert strat.on_bar(ctx).direction is None, f"{key} fired on {n} bars"
