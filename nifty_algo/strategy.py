"""
Strategy layer.

THE ONE ARCHITECTURAL RULE: this file is imported unchanged by both the
backtester and the live runner. A strategy only ever sees a Context and
only ever returns a Signal. It never touches a broker, a clock, or a file.
If you ever find yourself writing `if backtest:` in here, stop.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from .config import Config, DEFAULT
from . import signals as sig


@dataclass
class Context:
    """Everything a strategy is allowed to know at decision time."""
    bars: pd.DataFrame          # underlying OHLCV up to and including now
    now: time
    prev_day_high: float
    prev_day_low: float
    prev_day_close: float
    is_expiry_day: bool = False


@dataclass
class Signal:
    direction: Optional[str]    # "long", "short", or None
    option_type: Optional[str]  # "CE" for long, "PE" for short
    stop_points: float = 0.0    # stop distance on the UNDERLYING
    reason: str = ""
    confidence: float = 0.0


NO_SIGNAL = Signal(None, None, reason="no setup")


class Strategy(ABC):
    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg

    @abstractmethod
    def on_bar(self, ctx: Context) -> Signal:
        ...


class LevelBreakStrategy(Strategy):
    """
    Strategy #1 - 'Level break with pressure confirmation'.

    This is your discretionary process, written down:

      1. Build support/resistance from swing pivots + prior-day levels
         + the opening range.
      2. Wait for price to CLOSE beyond a level by a buffer (not just wick
         through it - wicks are where stop hunts live).
      3. Demand confirmation: volume surge AND a conviction candle
         (body/range high, close near the extreme in the trade direction).
      4. Stop distance = 1 x ATR on the underlying. The risk engine then
         converts that into a maximum delta and picks the strike.

    Variant B (doji reversal at a level) is included but off by default -
    trade one idea at a time, or you cannot attribute your results.
    """

    def __init__(self, cfg: Config = DEFAULT, enable_doji_reversal: bool = False):
        super().__init__(cfg)
        self.enable_doji_reversal = enable_doji_reversal

    def on_bar(self, ctx: Context) -> Signal:
        s = self.cfg.signal
        df = ctx.bars

        if len(df) < max(s.atr_period, 25):
            return Signal(None, None, reason="warming up")

        if not sig.underlying_liquidity_ok(df):
            return Signal(None, None, reason="thin liquidity")

        current_atr = float(sig.atr(df, s.atr_period).iloc[-1])
        if current_atr <= 0:
            return Signal(None, None, reason="zero atr")

        last = df.iloc[-1]
        close = float(last["close"])
        buffer = current_atr * s.breakout_buffer_atr_frac

        levels = self._all_levels(ctx, current_atr)
        if not levels:
            return Signal(None, None, reason="no levels")

        conviction = float(sig.body_to_range(df).iloc[-1])
        pressure = float(sig.close_position_in_range(df).iloc[-1])
        vol_ok = bool(sig.volume_surge(df, multiple=s.volume_surge_multiple).iloc[-1])

        stop_points = current_atr * s.atr_stop_multiple

        # ---- upside break of resistance ----
        broken_res = [lv for lv in levels
                      if lv.kind == "resistance" and close > lv.price + buffer]
        if broken_res:
            strongest = max(broken_res, key=lambda lv: lv.touches)
            if vol_ok and conviction >= s.min_body_to_range and pressure >= 0.65:
                return Signal(
                    direction="long", option_type="CE",
                    stop_points=stop_points,
                    reason=f"broke resistance {strongest.price:.1f} "
                           f"({strongest.touches} touches), vol surge, "
                           f"body {conviction:.2f}, pressure {pressure:.2f}",
                    confidence=min(1.0, strongest.touches / 4),
                )

        # ---- downside break of support ----
        broken_sup = [lv for lv in levels
                      if lv.kind == "support" and close < lv.price - buffer]
        if broken_sup:
            strongest = max(broken_sup, key=lambda lv: lv.touches)
            if vol_ok and conviction >= s.min_body_to_range and pressure <= 0.35:
                return Signal(
                    direction="short", option_type="PE",
                    stop_points=stop_points,
                    reason=f"broke support {strongest.price:.1f} "
                           f"({strongest.touches} touches), vol surge, "
                           f"body {conviction:.2f}, pressure {pressure:.2f}",
                    confidence=min(1.0, strongest.touches / 4),
                )

        if self.enable_doji_reversal:
            return self._doji_reversal(ctx, levels, current_atr, stop_points)

        return NO_SIGNAL

    # ---------------- helpers ----------------

    def _all_levels(self, ctx: Context, current_atr: float) -> list[sig.Level]:
        s = self.cfg.signal
        levels = sig.build_levels(
            ctx.bars,
            lookback=s.pivot_lookback,
            cluster_atr_frac=s.level_cluster_atr_frac,
            min_touches=s.min_level_touches,
        )
        # Prior-day levels count as pre-established zones (2 touches).
        for price, kind in [(ctx.prev_day_high, "resistance"),
                            (ctx.prev_day_low, "support"),
                            (ctx.prev_day_close, "support")]:
            if price and price > 0:
                levels.append(sig.Level(float(price), 2, kind))

        orange = sig.opening_range(
            ctx.bars,
            minutes=self.cfg.session.opening_range_minutes,
            bar_minutes=self.cfg.session.bar_interval_minutes,
        )
        if orange:
            hi, lo = orange
            levels.append(sig.Level(hi, 2, "resistance"))
            levels.append(sig.Level(lo, 2, "support"))
        return levels

    def _doji_reversal(self, ctx: Context, levels: list[sig.Level],
                       current_atr: float, stop_points: float) -> Signal:
        """
        Variant B: indecision candle sitting ON a level, then the next bar
        closes beyond that doji's range in the direction of the level.
        Your 'doji with buy and sell pressure' idea, made mechanical.
        """
        s = self.cfg.signal
        df = ctx.bars
        if len(df) < 2:
            return NO_SIGNAL

        prev, last = df.iloc[-2], df.iloc[-1]
        prev_body = float(sig.body_to_range(df).iloc[-2])
        if prev_body > s.max_doji_body_to_range:
            return NO_SIGNAL

        prev_mid = (float(prev["high"]) + float(prev["low"])) / 2
        near = [lv for lv in levels
                if lv.distance_atr(prev_mid, current_atr) <= 0.4]
        if not near:
            return NO_SIGNAL

        level = max(near, key=lambda lv: lv.touches)
        close = float(last["close"])

        if level.kind == "support" and close > float(prev["high"]):
            return Signal("long", "CE", stop_points,
                          f"doji rejection at support {level.price:.1f}, "
                          f"confirmed by close above doji high",
                          confidence=min(1.0, level.touches / 4))
        if level.kind == "resistance" and close < float(prev["low"]):
            return Signal("short", "PE", stop_points,
                          f"doji rejection at resistance {level.price:.1f}, "
                          f"confirmed by close below doji low",
                          confidence=min(1.0, level.touches / 4))
        return NO_SIGNAL
