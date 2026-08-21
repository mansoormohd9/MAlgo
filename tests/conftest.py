"""Shared fixtures: bar builders that let a test engineer an exact setup."""
from __future__ import annotations
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.strategy import Context


BASE_DAY = datetime(2026, 3, 10, 9, 15)


def make_bars(rows: list[dict], start: datetime = BASE_DAY,
              minutes: int = 5) -> pd.DataFrame:
    """Build a frame from explicit OHLCV dicts."""
    idx = [start + timedelta(minutes=minutes * i) for i in range(len(rows))]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def flat_bars(n: int = 40, price: float = 26_000.0, wiggle: float = 8.0,
              volume: float = 100_000, seed: int = 3,
              start: datetime = BASE_DAY) -> pd.DataFrame:
    """
    A calm baseline that establishes ATR and volume history without firing
    anything. Tests append their setup bars to this.
    """
    rng = np.random.default_rng(seed)
    rows = []
    p = price
    for _ in range(n):
        step = rng.normal(0, wiggle * 0.35)
        o, c = p, p + step
        h = max(o, c) + abs(rng.normal(0, wiggle * 0.3))
        l = min(o, c) - abs(rng.normal(0, wiggle * 0.3))
        rows.append({"open": o, "high": h, "low": l, "close": c,
                     "volume": volume})
        p = c
    return make_bars(rows, start=start)


def append_bar(df: pd.DataFrame, open_: float, high: float, low: float,
               close: float, volume: float, minutes: int = 5) -> pd.DataFrame:
    nxt = df.index[-1] + timedelta(minutes=minutes)
    row = pd.DataFrame([{"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}],
                       index=pd.DatetimeIndex([nxt]))
    return pd.concat([df, row])


def make_context(bars: pd.DataFrame, prev_high: float = 0.0,
                 prev_low: float = 0.0, prev_close: float = 0.0,
                 is_expiry: bool = False) -> Context:
    return Context(
        bars=bars,
        now=bars.index[-1].time(),
        prev_day_high=prev_high or float(bars["high"].max()) + 500,
        prev_day_low=prev_low or float(bars["low"].min()) - 500,
        prev_day_close=prev_close or float(bars["close"].iloc[0]),
        is_expiry_day=is_expiry,
    )


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def calm() -> pd.DataFrame:
    return flat_bars()


# ---------------------------------------------------------------- daily bars
#
# The swing scanner reads DAILY bars, not 5-minute ones, so it needs its own
# builders. Same idea as the intraday ones above: a series a test can reason
# about, rather than a fixture file nobody can check by eye.

DAILY_START = datetime(2026, 1, 5)


def daily_bars(closes, volume: float = 1_000_000.0,
               start: datetime = DAILY_START) -> pd.DataFrame:
    """Daily OHLCV from a close series, with plausible wicks around it."""
    idx = [start + timedelta(days=i) for i in range(len(closes))]
    rows, prev = [], closes[0]
    for c in closes:
        o = prev
        rows.append({"open": o, "high": max(o, c) * 1.004,
                     "low": min(o, c) * 0.996, "close": c, "volume": volume})
        prev = c
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def trending(n: int = 140, drift: float = 0.0035, vol: float = 0.010,
             seed: int = 7, start_price: float = 100.0) -> list[float]:
    """A geometric random walk with drift - an uptrend you can seed."""
    rng = np.random.default_rng(seed)
    return list(start_price * np.exp(np.cumsum(rng.normal(drift, vol, n))))
