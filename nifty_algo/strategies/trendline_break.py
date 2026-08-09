"""
Strategy #3 - Trendline break.

This is your "a line traversing upside or downside" signal, and it activates
`signals.fit_trendline()`, which was written but called by nothing.

The logic: fit a least-squares line through recent swing lows (rising support)
or swing highs (falling resistance). Gate it on R-squared - a line through
scattered points is not a trendline, it is wishful thinking, and the R2 floor
is the only thing separating the two. Then trade the BREAK of that line, not
the bounce off it: a rising support line that fails is trend exhaustion, and
exhaustion moves are fast, which is what a premium buyer needs.

Note the direction. Breaking a RISING SUPPORT line is bearish (buy PE).
Breaking a FALLING RESISTANCE line is bullish (buy CE). Getting this backwards
is the most common way to lose money with trendlines.
"""
from __future__ import annotations

from ..regime import Regime
from ..strategy import Signal, Context, NO_SIGNAL
from .. import signals as sig
from .base import BaseStrategy


class TrendlineBreakStrategy(BaseStrategy):
    key = "trendline_break"
    label = "Trendline break"
    description = ("R²-gated least-squares line through swing points; trade the "
                   "break, not the bounce. Activates fit_trendline().")
    allowed_regimes = (Regime.EXPANSION, Regime.GAP_DAY)
    default_enabled = True
    min_bars = 30

    def on_bar(self, ctx: Context) -> Signal:
        current_atr, abort = self._preflight(ctx)
        if abort:
            return self._reject(abort)

        st = self.cfg.strategy
        df = ctx.bars
        close = float(df["close"].iloc[-1])
        buffer = current_atr * st.trendline_break_atr_frac
        stop_points = self._stop_points(current_atr)
        bar_idx = len(df) - 1

        conviction = self._conviction(df)
        pressure = self._pressure(df)
        vol_ok = bool(sig.volume_surge(
            df, multiple=self.cfg.signal.volume_surge_multiple).iloc[-1])

        # --- rising support breaks DOWN -> bearish -> PE ---
        rising = sig.fit_trendline(
            df, lookback=self.cfg.signal.pivot_lookback,
            kind="rising_support", min_points=st.trendline_min_points)
        if rising and rising.r_squared >= st.trendline_min_r2:
            line_price = rising.value_at(bar_idx)
            if close < line_price - buffer and pressure <= 0.35 and vol_ok:
                return Signal(
                    direction="short", option_type="PE",
                    stop_points=stop_points,
                    reason=(f"broke rising support trendline at {line_price:.1f} "
                            f"(R² {rising.r_squared:.2f}, slope "
                            f"{rising.slope:+.2f} pts/bar), pressure {pressure:.2f}"),
                    confidence=self._confidence(rising.r_squared, conviction),
                )

        # --- falling resistance breaks UP -> bullish -> CE ---
        falling = sig.fit_trendline(
            df, lookback=self.cfg.signal.pivot_lookback,
            kind="falling_resistance", min_points=st.trendline_min_points)
        if falling and falling.r_squared >= st.trendline_min_r2:
            line_price = falling.value_at(bar_idx)
            if close > line_price + buffer and pressure >= 0.65 and vol_ok:
                return Signal(
                    direction="long", option_type="CE",
                    stop_points=stop_points,
                    reason=(f"broke falling resistance trendline at {line_price:.1f} "
                            f"(R² {falling.r_squared:.2f}, slope "
                            f"{falling.slope:+.2f} pts/bar), pressure {pressure:.2f}"),
                    confidence=self._confidence(falling.r_squared, conviction),
                )

        return NO_SIGNAL

    def _confidence(self, r2: float, conviction: float) -> float:
        """A tight line broken by a decisive candle is the high-quality case."""
        return round(min(1.0, 0.6 * r2 + 0.4 * conviction), 3)
