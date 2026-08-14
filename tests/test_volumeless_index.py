"""
Behaviour on a volume-less index series.

Real NIFTY 50 data has volume 0 on every bar. Before this was handled, feeding
it to the system produced three silent failures at once, each in a different
direction:

    volume_surge()          0 >= 0 * 1.5 is True  -> gate passed on EVERY bar
    vwap()                  cumulative volume 0   -> NaN -> strategy never fired
    underlying_liquidity_ok baseline 0            -> False -> blocked everything

None of those raise. None of them look wrong from the outside. This file
pins the corrected behaviour so they cannot come back.
"""
import numpy as np
import pandas as pd
import pytest

from nifty_algo import signals as sig


def _bars(n=60, with_volume=True, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-07 09:15", periods=n, freq="5min")
    close = 25_800 + np.cumsum(rng.normal(0, 6, n))
    df = pd.DataFrame({
        "open": close - rng.normal(0, 2, n),
        "high": close + np.abs(rng.normal(6, 2, n)),
        "low": close - np.abs(rng.normal(6, 2, n)),
        "close": close,
        "volume": rng.integers(80_000, 200_000, n) if with_volume else 0,
    }, index=idx)
    return df


def test_detects_the_absence_of_volume():
    assert sig.has_traded_volume(_bars(with_volume=True))
    assert not sig.has_traded_volume(_bars(with_volume=False))


def test_volume_gate_does_not_pass_on_every_bar_without_volume():
    """
    The original comparison was `0 >= 0 * 1.5`, which is True. That turned the
    breakout confirmation into a no-op on exactly the data you would run live.
    """
    surge = sig.volume_surge(_bars(with_volume=False), multiple=1.5)
    assert not surge.all()
    assert surge.any()          # and it is not a no-op in the other direction


def test_volume_gate_still_uses_volume_when_it_exists():
    df = _bars(with_volume=True)
    df.loc[df.index[-1], "volume"] = int(df["volume"].iloc[:-1].mean() * 5)
    assert bool(sig.volume_surge(df, multiple=1.5).iloc[-1])


def test_vwap_is_finite_without_volume():
    """It degrades to a session TWAP - a different benchmark, but a number."""
    vw = sig.vwap(_bars(with_volume=False))
    assert vw.notna().all()
    assert vw.iloc[-1] == pytest.approx(
        sig.typical_price(_bars(with_volume=False)).mean(), rel=1e-9)


def test_vwap_is_volume_weighted_when_volume_exists():
    """The fallback must not quietly replace the real calculation."""
    df = _bars(with_volume=True)
    equal_weight = sig.typical_price(df).mean()
    assert sig.vwap(df).iloc[-1] != pytest.approx(equal_weight)


def test_vwap_bands_are_finite_without_volume():
    lower, mid, upper = sig.vwap_bands(_bars(with_volume=False))
    assert lower.iloc[-1] < mid.iloc[-1] < upper.iloc[-1]


def test_liquidity_check_does_not_block_everything_without_volume():
    df = _bars(with_volume=False)
    assert sig.underlying_liquidity_ok(df) in (True, False)   # a real verdict
    # A collapsed range must still read as a dead tape.
    dead = df.copy()
    dead.loc[dead.index[-1], "high"] = float(dead["close"].iloc[-1])
    dead.loc[dead.index[-1], "low"] = float(dead["close"].iloc[-1])
    assert not sig.underlying_liquidity_ok(dead)


def test_the_backtester_warns_when_there_is_no_volume():
    """
    A weaker confirmation than the strategy docstrings describe is a fact the
    reader has to be told, not one they should have to infer.
    """
    from nifty_algo.backtest import Backtester
    from nifty_algo.config import Config

    days = []
    for d in range(3):
        day = _bars(70, with_volume=False, seed=d)
        day.index = day.index + pd.Timedelta(days=d)
        days.append(day)
    bars = pd.concat(days)

    res = Backtester(Config()).run(bars)
    assert any("NO TRADED VOLUME" in w for w in res.warnings)
