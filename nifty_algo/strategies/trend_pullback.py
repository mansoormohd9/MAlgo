"""
Strategy #7 - Trend pullback continuation.

Every other strategy here needs a level, a line, or a range to break. On a
strong trend day all of those broke hours ago and price just keeps going -
so those strategies go quiet exactly when the day is most tradeable.

This one fills that hole. Establish trend with EMA 9/21 separation and slope
agreement, then wait for a pullback INTO the fast EMA and enter when pressure
resumes in the trend direction.

The `pullback_max_distance_atr` gate is doing real work: it forbids chasing.
If price is far above the fast EMA it is extended, and buying an option there
means paying elevated premium for a move that has already happened.
"""
from __future__ import annotations

from ..regime import Regime
from ..strategy import Signal, Context, NO_SIGNAL
from .. import signals as sig
from .base import BaseStrategy


class TrendPullbackStrategy(BaseStrategy):
    key = "trend_pullback"
    label = "Trend pullback"
    description = ("EMA 9/21 trend, entry on a pullback to the fast EMA with pressure "
                   "resuming. Covers trend days after every level has already broken.")
    allowed_regimes = (Regime.EXPANSION, Regime.GAP_DAY)
    default_enabled = False
    min_bars = 30

    def on_bar(self, ctx: Context) -> Signal:
        current_atr, abort = self._preflight(ctx)
        if abort:
            return self._reject(abort)

        st = self.cfg.strategy
        df = ctx.bars

        trend = sig.trend_state(df, st.ema_fast, st.ema_slow)
        direction = int(trend.iloc[-1])
        if direction == 0:
            return self._reject("no trend")

        # Demand the trend has actually persisted, not just flickered on this bar.
        recent = trend.iloc[-st.pullback_min_trend_bars:]
        if int((recent == direction).sum()) < st.pullback_min_trend_bars:
            return self._reject("trend not established")

        fast = sig.ema(df["close"], st.ema_fast)
        fast_now = float(fast.iloc[-1])
        close = float(df["close"].iloc[-1])
        low = float(df["low"].iloc[-1])
        high = float(df["high"].iloc[-1])

        distance_atr = abs(close - fast_now) / current_atr
        if distance_atr > st.pullback_max_distance_atr:
            return self._reject(f"extended {distance_atr:.2f} ATR from EMA - no chasing")

        stop_points = self._stop_points(current_atr)
        pressure = self._pressure(df)
        conviction = self._conviction(df)
        thrust = sig.consecutive_closes(df, direction, lookback=st.pullback_min_trend_bars)

        if direction > 0:
            touched = low <= fast_now + current_atr * st.pullback_max_distance_atr
            if touched and close > fast_now and pressure >= 0.60:
                return Signal(
                    direction="long", option_type="CE",
                    stop_points=stop_points,
                    reason=(f"uptrend (EMA{st.ema_fast}>{st.ema_slow}, {thrust}/"
                            f"{st.pullback_min_trend_bars} up closes), pulled back to "
                            f"EMA {fast_now:.1f} and resumed, pressure {pressure:.2f}"),
                    confidence=self._confidence(thrust, conviction, st),
                )

        if direction < 0:
            touched = high >= fast_now - current_atr * st.pullback_max_distance_atr
            if touched and close < fast_now and pressure <= 0.40:
                return Signal(
                    direction="short", option_type="PE",
                    stop_points=stop_points,
                    reason=(f"downtrend (EMA{st.ema_fast}<{st.ema_slow}, {thrust}/"
                            f"{st.pullback_min_trend_bars} down closes), pulled back to "
                            f"EMA {fast_now:.1f} and resumed, pressure {pressure:.2f}"),
                    confidence=self._confidence(thrust, conviction, st),
                )

        return NO_SIGNAL

    @staticmethod
    def _confidence(thrust: int, conviction: float, st) -> float:
        strength = thrust / max(st.pullback_min_trend_bars, 1)
        return round(min(1.0, 0.6 * strength + 0.4 * conviction), 3)
