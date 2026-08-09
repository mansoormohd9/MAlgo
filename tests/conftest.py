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
