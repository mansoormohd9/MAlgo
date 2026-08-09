"""
Strategy #9 - Prior-day high/low sweep and reclaim.

A specialisation of the failed-breakout idea, anchored only to the prior day's
high and low.

Why give PDH/PDL their own strategy rather than folding them into the generic
sweep: those two prices are visible on every chart, in every broker terminal,
and to every algo. The stop cluster beyond them is an order of magnitude
larger than the one beyond an intraday swing pivot. Sweeps there are more
deliberate and the reversals are cleaner.

`prev_day_high` and `prev_day_low` already arrive on Context - no new data
plumbing needed.
"""
from __future__ import annotations

from ..regime import Regime
from ..strategy import Signal, Context, NO_SIGNAL
from .. import signals as sig
from .base import BaseStrategy


class PriorDaySweepStrategy(BaseStrategy):
    key = "pd_sweep"
    label = "PDH/PDL sweep reclaim"
    description = ("Sweep of the prior day's high or low, then reclaim. The largest and "
                   "most visible stop cluster on the chart.")
    allowed_regimes = (Regime.RANGE, Regime.EXPANSION, Regime.GAP_DAY)
    default_enabled = False
    min_bars = 25

    def on_bar(self, ctx: Context) -> Signal:
        current_atr, abort = self._preflight(ctx)
        if abort:
            return self._reject(abort)

        st = self.cfg.strategy
        df = ctx.bars
        stop_points = self._stop_points(current_atr)
        last = df.iloc[-1]

        candidates = [
            ("prior-day high", ctx.prev_day_high),
            ("prior-day low", ctx.prev_day_low),
        ]

        for name, level in candidates:
            if not level or level <= 0:
                continue
            verdict = sig.sweep_reclaim(
                df, float(level), current_atr,
                min_wick_atr=st.sweep_min_wick_atr,
                max_close_back_atr=st.sweep_max_close_back_atr,
            )
            if verdict is None:
                continue

            if verdict == "bullish":
                wick = float(level) - float(last["low"])
                return Signal(
                    direction="long", option_type="CE",
                    stop_points=stop_points,
                    reason=(f"swept {name} {level:.1f} by {wick:.1f} pts and reclaimed - "
                            f"the day's most-watched stop cluster taken"),
                    confidence=0.75,
                )

            wick = float(last["high"]) - float(level)
            return Signal(
                direction="short", option_type="PE",
                stop_points=stop_points,
                reason=(f"swept {name} {level:.1f} by {wick:.1f} pts and rejected - "
                        f"the day's most-watched stop cluster taken"),
                confidence=0.75,
            )

        return NO_SIGNAL
