"""
Strategy #6 - Failed breakout / liquidity sweep.

The deliberate inverse of LevelBreakStrategy.

Resting stop orders cluster just beyond obvious levels. Price is routinely
pushed through those levels to fill against that liquidity, then returns.
When that happens, everyone who bought the breakout is instantly offside and
must exit - and their exits are fuel for the move the other way.

Mechanically: the last bar WICKED decisively beyond a level (min_wick_atr)
but CLOSED back inside it. Trade against the direction of the sweep.

Run this alongside the level-break strategy and understand what you are doing:
they take opposite sides of the same event. That is intentional - the regime
gate decides which one is allowed to speak. Breakouts on expansion days,
sweeps on range days. Running both ungated on the same day is how you pay
friction twice to end up flat.
"""
from __future__ import annotations

from ..regime import Regime
from ..strategy import Signal, Context, NO_SIGNAL
from .. import signals as sig
from .base import BaseStrategy


class FailedBreakoutStrategy(BaseStrategy):
    key = "failed_breakout"
    label = "Failed breakout / sweep"
    description = ("Wick beyond a level, close back inside. Takes the opposite side "
                   "of the level-break strategy - regime gate decides which speaks.")
    allowed_regimes = (Regime.RANGE, Regime.EXPANSION)
    default_enabled = False
    min_bars = 30

    def on_bar(self, ctx: Context) -> Signal:
        current_atr, abort = self._preflight(ctx)
        if abort:
            return self._reject(abort)

        st = self.cfg.strategy
        df = ctx.bars
        stop_points = self._stop_points(current_atr)

        levels = self._levels(ctx)
        if not levels:
            return self._reject("no levels")

        # Strongest level first - a 4-touch level being swept is a far better
        # trade than a 2-touch one, because more stops were resting there.
        for level in sorted(levels, key=lambda lv: lv.touches, reverse=True):
            verdict = sig.sweep_reclaim(
                df, level.price, current_atr,
                min_wick_atr=st.sweep_min_wick_atr,
                max_close_back_atr=st.sweep_max_close_back_atr,
            )
            if verdict is None:
                continue

            confidence = round(min(1.0, level.touches / 4.0), 3)
            last = df.iloc[-1]

            if verdict == "bullish":
                wick = level.price - float(last["low"])
                return Signal(
                    direction="long", option_type="CE",
                    stop_points=stop_points,
                    reason=(f"swept {level.price:.1f} ({level.touches} touches) by "
                            f"{wick:.1f} pts to the downside and reclaimed it - "
                            f"stops taken, sellers trapped"),
                    confidence=confidence,
                )

            wick = float(last["high"]) - level.price
            return Signal(
                direction="short", option_type="PE",
                stop_points=stop_points,
                reason=(f"swept {level.price:.1f} ({level.touches} touches) by "
                        f"{wick:.1f} pts to the upside and rejected - "
                        f"stops taken, buyers trapped"),
                confidence=confidence,
            )

        return NO_SIGNAL

    def _levels(self, ctx: Context) -> list[sig.Level]:
        """Swing levels plus the pre-established prior-day and OR levels."""
        s = self.cfg.signal
        levels = sig.build_levels(
            ctx.bars,
            lookback=s.pivot_lookback,
            cluster_atr_frac=s.level_cluster_atr_frac,
            min_touches=s.min_level_touches,
        )
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
