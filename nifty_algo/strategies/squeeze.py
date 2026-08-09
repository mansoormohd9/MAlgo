"""
Strategy #8 - Volatility squeeze expansion.

The most important strategy in this file for a PREMIUM BUYER, and the reason
is structural rather than technical.

Every other setup here enters after a move has started - after a level breaks,
after a trendline snaps, after VWAP is reclaimed. By then implied volatility
has already repriced upward, so you are buying the option at its most
expensive, and you need the move to continue just to overcome what you paid.

This one enters at the opposite point of that cycle. Range compression (NR7:
the narrowest bar of the last seven) means realised volatility has collapsed,
which drags IV and therefore premium down with it. Volatility is mean-reverting
and range-bound behaviour resolves into expansion. Buying a cheap option
immediately before expansion means you are long vega going into a vega
increase, instead of short-changed by it.

The cost of that edge: compression resolves in EITHER direction, so you must
wait for the expansion bar to declare one. That first expansion bar is your
entry, and it is why the confirmation demands both a volume surge and a
conviction candle - without them you are guessing at the direction of a coin
flip and paying friction for the privilege.
"""
from __future__ import annotations

from ..regime import Regime
from ..strategy import Signal, Context, NO_SIGNAL
from .. import signals as sig
from .base import BaseStrategy


class VolatilitySqueezeStrategy(BaseStrategy):
    key = "squeeze"
    label = "Volatility squeeze"
    description = ("NR7 range compression resolving into expansion. The only setup here "
                   "that buys premium BEFORE IV reprices upward rather than after.")
    # Allowed in RANGE too - a squeeze is precisely how a range day ends.
    allowed_regimes = (Regime.EXPANSION, Regime.RANGE, Regime.GAP_DAY)
    default_enabled = True
    min_bars = 30

    def on_bar(self, ctx: Context) -> Signal:
        current_atr, abort = self._preflight(ctx)
        if abort:
            return self._reject(abort)

        st = self.cfg.strategy
        df = ctx.bars
        if len(df) < st.squeeze_lookback + 2:
            return self._reject("warming up")

        # The PRIOR bar must be the compressed one; the CURRENT bar is the
        # expansion out of it. Checking compression on the current bar would
        # mean entering into a narrow bar, which is backwards.
        compressed = bool(sig.narrowest_range_n(df, st.squeeze_lookback).iloc[-2])
        pct = sig.range_percentile(df, lookback=20).iloc[-2]
        tight = compressed or (pct is not None and float(pct) <= st.squeeze_range_percentile)
        if not tight:
            return self._reject("no compression")

        prev = df.iloc[-2]
        last = df.iloc[-1]
        prev_high, prev_low = float(prev["high"]), float(prev["low"])
        close = float(last["close"])
        buffer = current_atr * st.squeeze_expansion_atr_frac

        # The expansion bar must be genuinely bigger than the coiled one.
        prev_range = prev_high - prev_low
        this_range = float(last["high"]) - float(last["low"])
        if prev_range > 0 and this_range < prev_range * 1.5:
            return self._reject("expansion too weak")

        stop_points = self._stop_points(current_atr)
        pressure = self._pressure(df)
        conviction = self._conviction(df)
        vol_ok = bool(sig.volume_surge(
            df, multiple=self.cfg.signal.volume_surge_multiple).iloc[-1])

        if not vol_ok or conviction < self.cfg.signal.min_body_to_range:
            return self._reject("expansion unconfirmed")

        expansion_ratio = this_range / prev_range if prev_range > 0 else 0.0

        if close > prev_high + buffer and pressure >= 0.65:
            return Signal(
                direction="long", option_type="CE",
                stop_points=stop_points,
                reason=(f"squeeze ({st.squeeze_lookback}-bar compression) resolved up "
                        f"through {prev_high:.1f}, range expanded {expansion_ratio:.1f}x, "
                        f"pressure {pressure:.2f} - long vega into expansion"),
                confidence=self._confidence(expansion_ratio, conviction),
            )

        if close < prev_low - buffer and pressure <= 0.35:
            return Signal(
                direction="short", option_type="PE",
                stop_points=stop_points,
                reason=(f"squeeze ({st.squeeze_lookback}-bar compression) resolved down "
                        f"through {prev_low:.1f}, range expanded {expansion_ratio:.1f}x, "
                        f"pressure {pressure:.2f} - long vega into expansion"),
                confidence=self._confidence(expansion_ratio, conviction),
            )

        return NO_SIGNAL

    @staticmethod
    def _confidence(expansion_ratio: float, conviction: float) -> float:
        scaled = min(expansion_ratio / 3.0, 1.0)
        return round(min(1.0, 0.5 * scaled + 0.5 * conviction), 3)
