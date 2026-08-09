"""
Strategy #4 - VWAP reclaim / rejection.

VWAP is the benchmark institutions are measured against, which makes it the
one intraday level with a genuine structural reason to matter: a desk filling
a large order all day is judged on whether it beat VWAP, so price repeatedly
gets defended there.

Two symmetric setups:

  Reclaim  - price spent time meaningfully BELOW VWAP, then closes back above
             it with volume. Sellers lost control of the benchmark. Buy CE.
  Rejection - price pushed above VWAP, failed, closes back below. Buy PE.

The `vwap_min_distance_atr` gate matters more than it looks. Price oscillating
within a few points of VWAP will cross it a dozen times a session; without
requiring that price was genuinely away first, this fires constantly and every
one of those trades pays full friction for nothing.
"""
from __future__ import annotations

import pandas as pd

from ..regime import Regime
from ..strategy import Signal, Context, NO_SIGNAL
from .. import signals as sig
from .base import BaseStrategy


class VwapReclaimStrategy(BaseStrategy):
    key = "vwap_reclaim"
    label = "VWAP reclaim / reject"
    description = ("Price returns across session VWAP after being meaningfully "
                   "away from it. The benchmark institutions actually trade against.")
    allowed_regimes = (Regime.EXPANSION, Regime.GAP_DAY)
    default_enabled = True
    min_bars = 25

    def on_bar(self, ctx: Context) -> Signal:
        current_atr, abort = self._preflight(ctx)
        if abort:
            return self._reject(abort)

        st = self.cfg.strategy
        df = ctx.bars
        vw = sig.vwap(df)
        if pd.isna(vw.iloc[-1]):
            return self._reject("no vwap")

        vwap_now = float(vw.iloc[-1])
        close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        prev_vwap = float(vw.iloc[-2])

        stop_points = self._stop_points(current_atr)
        margin = current_atr * st.vwap_reclaim_atr_frac
        pressure = self._pressure(df)
        conviction = self._conviction(df)
        vol_ok = bool(sig.volume_surge(
            df, multiple=self.cfg.signal.volume_surge_multiple).iloc[-1])

        # How far from VWAP was price over the recent window? Without this,
        # every wobble across the line is a "signal".
        lookback = min(10, len(df) - 1)
        excursion = (df["close"].iloc[-lookback:] - vw.iloc[-lookback:]) / current_atr
        was_below = float(excursion.min()) <= -st.vwap_min_distance_atr
        was_above = float(excursion.max()) >= st.vwap_min_distance_atr

        crossed_up = prev_close <= prev_vwap and close > vwap_now + margin
        crossed_down = prev_close >= prev_vwap and close < vwap_now - margin

        if crossed_up and was_below and pressure >= 0.60 and vol_ok:
            return Signal(
                direction="long", option_type="CE",
                stop_points=stop_points,
                reason=(f"reclaimed VWAP {vwap_now:.1f} after trading "
                        f"{abs(float(excursion.min())):.2f} ATR below it, "
                        f"pressure {pressure:.2f}"),
                confidence=round(min(1.0, 0.5 + 0.5 * conviction), 3),
            )

        if crossed_down and was_above and pressure <= 0.40 and vol_ok:
            return Signal(
                direction="short", option_type="PE",
                stop_points=stop_points,
                reason=(f"rejected from VWAP {vwap_now:.1f} after trading "
                        f"{float(excursion.max()):.2f} ATR above it, "
                        f"pressure {pressure:.2f}"),
                confidence=round(min(1.0, 0.5 + 0.5 * conviction), 3),
            )

        return NO_SIGNAL
