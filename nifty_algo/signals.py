"""
Signal layer: turns the patterns you trade by eye into numbers.

Every function here is pure - bars in, values out, no I/O, no broker calls.
That is what makes them testable and what lets backtest and live share code.

All functions operate on the UNDERLYING (Nifty spot or futures), never on
option premiums. Options are the execution vehicle, not the signal source.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------- volatility ----------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - your unit of 'normal' movement."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------------- candle anatomy ----------------

def body_to_range(df: pd.DataFrame) -> pd.Series:
    """
    Conviction measure. Near 1.0 = decisive candle. Near 0.0 = indecision.
    This one ratio expresses both your 'doji' and your 'pressure' ideas.
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["open"]).abs() / rng


def close_position_in_range(df: pd.DataFrame) -> pd.Series:
    """
    Where the candle closed within its own range: 1.0 = at the high
    (buyers won the bar), 0.0 = at the low (sellers won).
    This is your 'buy/sell pressure' made numeric.
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["low"]) / rng


def is_doji(df: pd.DataFrame, max_body_ratio: float = 0.15) -> pd.Series:
    return body_to_range(df) <= max_body_ratio


def volume_surge(df: pd.DataFrame, lookback: int = 20,
                 multiple: float = 1.5) -> pd.Series:
    baseline = df["volume"].rolling(lookback).mean()
    return df["volume"] >= baseline * multiple


# ---------------- support / resistance ----------------

@dataclass
class Level:
    price: float
    touches: int
    kind: str          # "support" or "resistance"

    def distance_atr(self, price: float, current_atr: float) -> float:
        if current_atr <= 0:
            return float("inf")
        return abs(price - self.price) / current_atr


def find_pivots(df: pd.DataFrame, lookback: int = 5) -> tuple[pd.Series, pd.Series]:
    """
    A swing high is a bar whose high exceeds `lookback` bars on both sides.
    Note the forward-looking window: when running live you can only confirm
    a pivot `lookback` bars after it formed. The backtester must respect
    that delay or you have look-ahead bias.
    """
    highs, lows = df["high"], df["low"]
    win = 2 * lookback + 1
    is_high = highs == highs.rolling(win, center=True).max()
    is_low = lows == lows.rolling(win, center=True).min()
    return is_high.fillna(False), is_low.fillna(False)


def cluster_levels(prices: Sequence[float], tolerance: float,
                   kind: str, min_touches: int = 2) -> list[Level]:
    """
    Group nearby pivots into a single zone. A level touched four times is
    a very different object from one touched once - touches are the signal.
    """
    if not len(prices):
        return []
    ordered = sorted(prices)
    clusters: list[list[float]] = [[ordered[0]]]
    for p in ordered[1:]:
        if p - clusters[-1][-1] <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [
        Level(price=float(np.mean(c)), touches=len(c), kind=kind)
        for c in clusters if len(c) >= min_touches
    ]


def build_levels(df: pd.DataFrame, lookback: int = 5,
                 cluster_atr_frac: float = 0.3,
                 min_touches: int = 2) -> list[Level]:
    is_high, is_low = find_pivots(df, lookback)
    current_atr = float(atr(df).iloc[-1])
    tol = current_atr * cluster_atr_frac
    res = cluster_levels(df.loc[is_high, "high"].tolist(), tol, "resistance", min_touches)
    sup = cluster_levels(df.loc[is_low, "low"].tolist(), tol, "support", min_touches)
    return res + sup


# ---------------- trendline: your 'line traversing upside or downside' ----------------

@dataclass
class Trendline:
    slope: float           # points per bar
    intercept: float
    r_squared: float
    kind: str              # "rising_support" or "falling_resistance"

    def value_at(self, bar_index: float) -> float:
        return self.slope * bar_index + self.intercept


def fit_trendline(df: pd.DataFrame, lookback: int = 5,
                  kind: str = "rising_support",
                  min_points: int = 3) -> Trendline | None:
    """
    Least-squares fit through recent swing lows (rising support) or swing
    highs (falling resistance). r_squared is your confidence gate - a line
    through scattered points is not a trendline, it is wishful thinking.
    """
    is_high, is_low = find_pivots(df, lookback)
    mask = is_low if kind == "rising_support" else is_high
    col = "low" if kind == "rising_support" else "high"

    pts = df.loc[mask, col]
    if len(pts) < min_points:
        return None

    x = np.array([df.index.get_loc(i) for i in pts.index], dtype=float)
    y = pts.to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if kind == "rising_support" and slope <= 0:
        return None
    if kind == "falling_resistance" and slope >= 0:
        return None

    return Trendline(float(slope), float(intercept), r2, kind)


# ---------------- liquidity ----------------

def underlying_liquidity_ok(df: pd.DataFrame, lookback: int = 20,
                            min_ratio: float = 0.5) -> bool:
    """Refuse to trade in a dead tape - thin volume means bad option fills."""
    if len(df) < lookback + 1:
        return False
    recent = float(df["volume"].iloc[-1])
    baseline = float(df["volume"].iloc[-lookback:].mean())
    return baseline > 0 and recent / baseline >= min_ratio


# ---------------- opening range ----------------

def opening_range(df: pd.DataFrame, minutes: int = 15,
                  bar_minutes: int = 5) -> tuple[float, float] | None:
    bars = max(1, minutes // bar_minutes)
    if len(df) < bars:
        return None
    window = df.iloc[:bars]
    return float(window["high"].max()), float(window["low"].min())


# ---------------- VWAP: the benchmark institutions actually trade against ----------------

def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-anchored VWAP. Resets at each new calendar day in the index, so a
    multi-day frame gives you one VWAP per session rather than a running
    average across days (which would be meaningless intraday).
    """
    tp = typical_price(df)
    pv = tp * df["volume"]
    day = pd.Series(df.index.date, index=df.index)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def vwap_bands(df: pd.DataFrame, sigma: float = 1.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    VWAP with volume-weighted standard-deviation bands. Returns (lower, vwap, upper).
    Price outside a band is stretched; price returning inside is a reversion cue.
    """
    vw = vwap(df)
    tp = typical_price(df)
    day = pd.Series(df.index.date, index=df.index)
    sq = ((tp - vw) ** 2) * df["volume"]
    cum_sq = sq.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    dev = np.sqrt(cum_sq / cum_vol)
    return vw - dev * sigma, vw, vw + dev * sigma


# ---------------- trend ----------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def trend_state(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.Series:
    """+1 = up, -1 = down, 0 = no trend. Fast/slow EMA separation with slope agreement."""
    ef, es = ema(df["close"], fast), ema(df["close"], slow)
    up = (ef > es) & (ef.diff() > 0)
    down = (ef < es) & (ef.diff() < 0)
    return pd.Series(np.where(up, 1, np.where(down, -1, 0)), index=df.index)


def consecutive_closes(df: pd.DataFrame, direction: int, lookback: int = 5) -> int:
    """
    How many of the last `lookback` bars closed in `direction` (+1 up, -1 down).
    A crude but robust thrust measure - no smoothing lag.
    """
    if len(df) < 2:
        return 0
    delta = df["close"].diff().tail(lookback)
    return int((delta > 0).sum() if direction > 0 else (delta < 0).sum())


# ---------------- volatility compression: the option buyer's edge ----------------

def narrowest_range_n(df: pd.DataFrame, n: int = 7) -> pd.Series:
    """
    True where the current bar's range is the narrowest of the last n bars (NR7).
    Compression precedes expansion. This is the one setup that gets a premium
    buyer LONG volatility before it expands rather than after - everything else
    in this system buys once the move is already underway and IV has repriced.
    """
    rng = df["high"] - df["low"]
    return rng == rng.rolling(n).min()


def range_percentile(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Where the current bar's range sits in its own recent distribution, 0..1."""
    rng = df["high"] - df["low"]
    return rng.rolling(lookback).rank(pct=True)


# ---------------- liquidity sweeps: where stop hunts live ----------------

def sweep_reclaim(df: pd.DataFrame, level: float, current_atr: float,
                  min_wick_atr: float = 0.35,
                  max_close_back_atr: float = 0.20) -> Optional[str]:
    """
    Detect a failed breakout on the last bar: price wicked decisively beyond a
    level, then closed back inside it.

    Returns "bullish" (swept below support, reclaimed), "bearish" (swept above
    resistance, rejected), or None.

    This is the inverse of a breakout, and deliberately so - it is the trade
    that pays on the chop days that punish LevelBreakStrategy.
    """
    if current_atr <= 0 or len(df) < 1:
        return None
    last = df.iloc[-1]
    high, low, close = float(last["high"]), float(last["low"]), float(last["close"])

    swept_below = (level - low) >= min_wick_atr * current_atr
    reclaimed = (close - level) >= -max_close_back_atr * current_atr and close > level
    if swept_below and reclaimed:
        return "bullish"

    swept_above = (high - level) >= min_wick_atr * current_atr
    rejected = (level - close) >= -max_close_back_atr * current_atr and close < level
    if swept_above and rejected:
        return "bearish"

    return None


# ---------------- gaps ----------------

def gap_metrics(session_open: float, prev_close: float,
                current_atr: float) -> tuple[float, float]:
    """
    Returns (gap_points, gap_in_atr). Sign follows direction: positive = gap up.
    A gap beyond ~3 ATR is news, not structure - the caller should stand aside.
    """
    gap = session_open - prev_close
    if current_atr <= 0:
        return gap, 0.0
    return gap, gap / current_atr
