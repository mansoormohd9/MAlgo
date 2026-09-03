"""
Strategy #5 - Opening range breakout with retest.

The opening range (first 15 minutes) is the day's first agreed value area.
Breaking it is information; breaking it and then HOLDING it on a retest is
tradeable information.

The retest requirement is the whole point, and it is what makes this different
from LevelBreakStrategy firing on the same OR level. A naked OR break is the
single most front-run pattern in intraday trading - everyone sees the same
15-minute high. Waiting for price to come back, touch the level, and hold it
filters the sweeps that reverse immediately, at the cost of a worse entry.

For an option buyer that trade-off is correct: you cannot afford to pay
friction on breaks that fail inside two bars.
"""
from __future__ import annotations

from ..regime import Regime
from ..strategy import Signal, Context, NO_SIGNAL
from .. import signals as sig
from .base import BaseStrategy


class OpeningRangeRetestStrategy(BaseStrategy):
    key = "orb_retest"
    label = "ORB retest"
    description = ("Opening-range break, then a retest that holds. The retest is "
                   "what separates this from a naked - and heavily front-run - ORB.")
    allowed_regimes = (Regime.EXPANSION, Regime.GAP_DAY)
    default_enabled = False
    min_bars = 25

    def on_bar(self, ctx: Context) -> Signal:
        current_atr, abort = self._preflight(ctx)
        if abort:
            return self._reject(abort)

        st = self.cfg.strategy
        s = self.cfg.session
        df = ctx.bars

        orange = sig.opening_range(df, s.opening_range_minutes, s.bar_interval_minutes)
        if not orange:
            return self._reject("no opening range")
        or_high, or_low = orange

        # Anchored to today, not to the frame: with warm-up sessions
        # prepended, `df.iloc[or_bars:]` would be "everything after the third
        # bar of the warm-up window", i.e. almost the whole frame.
        or_bars = max(1, s.opening_range_minutes // s.bar_interval_minutes)
        after = sig.last_session(df).iloc[or_bars:]
        if len(after) < 3:
            return self._reject("too soon after opening range")

        close = float(df["close"].iloc[-1])
        low = float(df["low"].iloc[-1])
        high = float(df["high"].iloc[-1])
        stop_points = self._stop_points(current_atr)
        tol = current_atr * st.orb_retest_tolerance_atr
        pressure = self._pressure(df)
        conviction = self._conviction(df)

        # --- upside: broke OR high earlier, now retested it and held ---
        broke_up_at = self._first_break_index(after, or_high, up=True)
        if broke_up_at is not None:
            bars_since = len(after) - 1 - broke_up_at
            if 1 <= bars_since <= st.orb_retest_max_bars:
                retested = low <= or_high + tol
                held = close > or_high
                if retested and held and pressure >= 0.60:
                    return Signal(
                        direction="long", option_type="CE",
                        stop_points=stop_points,
                        reason=(f"ORB high {or_high:.1f} broken {bars_since} bars ago, "
                                f"retested to {low:.1f} and held, "
                                f"close {close:.1f}, pressure {pressure:.2f}"),
                        confidence=self._confidence(bars_since, conviction, st),
                    )

        # --- downside: broke OR low earlier, now retested it and held ---
        broke_dn_at = self._first_break_index(after, or_low, up=False)
        if broke_dn_at is not None:
            bars_since = len(after) - 1 - broke_dn_at
            if 1 <= bars_since <= st.orb_retest_max_bars:
                retested = high >= or_low - tol
                held = close < or_low
                if retested and held and pressure <= 0.40:
                    return Signal(
                        direction="short", option_type="PE",
                        stop_points=stop_points,
                        reason=(f"ORB low {or_low:.1f} broken {bars_since} bars ago, "
                                f"retested to {high:.1f} and held, "
                                f"close {close:.1f}, pressure {pressure:.2f}"),
                        confidence=self._confidence(bars_since, conviction, st),
                    )

        return NO_SIGNAL

    @staticmethod
    def _first_break_index(after, level: float, up: bool) -> int | None:
        """Index within `after` of the first close beyond the level, else None."""
        mask = after["close"] > level if up else after["close"] < level
        if not mask.any():
            return None
        return int(mask.values.argmax())

    @staticmethod
    def _confidence(bars_since: int, conviction: float, st) -> float:
        """A prompt retest is stronger than one that took the whole window."""
        promptness = 1.0 - (bars_since - 1) / max(st.orb_retest_max_bars, 1)
        return round(min(1.0, 0.5 * promptness + 0.5 * conviction), 3)
