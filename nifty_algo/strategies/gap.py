"""
Strategy #10 - Gap playbook.

Nifty gaps often, because it prices in overnight US and Asian moves at 09:15.
A gap is not one setup but two mutually exclusive ones, and the whole skill is
telling them apart:

  GAP FADE        - a modest gap into no particular structure tends to fill
                    back toward the prior close. Trade toward prev_day_close.
  GAP-AND-GO      - price holds above (below) the opening range after gapping
                    up (down). The gap was real repricing, not an overreaction.
                    Trade with the gap.

The discriminator used here is the opening range: fail back into it and the gap
is being rejected, hold outside it and it is being accepted.

Both tails are excluded on purpose. Below `gap_min_atr` it is not a gap, just
an open. Above `gap_max_atr` it is a news event - the distribution has changed,
IV has already exploded so premiums are terrible, and no amount of price
structure tells you what happens next. Stand aside.
"""
from __future__ import annotations

from ..regime import Regime
from ..strategy import Signal, Context, NO_SIGNAL
from .. import signals as sig
from .base import BaseStrategy


class GapStrategy(BaseStrategy):
    key = "gap"
    label = "Gap fade / gap-and-go"
    description = ("Gap beyond 0.5 ATR: fade back toward prior close, or continue if the "
                   "opening range holds. Stands aside on news-sized gaps.")
    allowed_regimes = (Regime.GAP_DAY,)
    default_enabled = False
    min_bars = 25

    def on_bar(self, ctx: Context) -> Signal:
        current_atr, abort = self._preflight(ctx)
        if abort:
            return self._reject(abort)

        st = self.cfg.strategy
        s = self.cfg.session
        df = ctx.bars

        if not ctx.prev_day_close or ctx.prev_day_close <= 0:
            return self._reject("no prior close")

        open_px = sig.session_open(df)
        if open_px is None:
            return self._reject("no session open")
        gap_pts, gap_atr = sig.gap_metrics(open_px, float(ctx.prev_day_close),
                                           current_atr)

        if abs(gap_atr) < st.gap_min_atr:
            return self._reject(f"gap {gap_atr:+.2f} ATR too small")
        if abs(gap_atr) > st.gap_max_atr:
            return self._reject(f"gap {gap_atr:+.2f} ATR is a news event - standing aside")

        orange = sig.opening_range(df, s.opening_range_minutes, s.bar_interval_minutes)
        if not orange:
            return self._reject("no opening range")
        or_high, or_low = orange

        close = float(df["close"].iloc[-1])
        stop_points = self._stop_points(current_atr)
        pressure = self._pressure(df)
        conviction = self._conviction(df)
        vol_ok = bool(sig.volume_surge(
            df, multiple=self.cfg.signal.volume_surge_multiple).iloc[-1])

        gapped_up = gap_atr > 0

        # ---- gap-and-go: opening range holds in the gap's direction ----
        if gapped_up and close > or_high and pressure >= 0.65 and vol_ok:
            return Signal(
                direction="long", option_type="CE",
                stop_points=stop_points,
                reason=(f"gap-and-go: gapped up {gap_pts:.0f} pts ({gap_atr:+.2f} ATR), "
                        f"held above OR high {or_high:.1f} - repricing accepted"),
                confidence=round(min(1.0, 0.5 + 0.5 * conviction), 3),
            )
        if not gapped_up and close < or_low and pressure <= 0.35 and vol_ok:
            return Signal(
                direction="short", option_type="PE",
                stop_points=stop_points,
                reason=(f"gap-and-go: gapped down {gap_pts:.0f} pts ({gap_atr:+.2f} ATR), "
                        f"held below OR low {or_low:.1f} - repricing accepted"),
                confidence=round(min(1.0, 0.5 + 0.5 * conviction), 3),
            )

        # ---- gap fade: price falls back INTO the opening range ----
        if gapped_up and close < or_low and pressure <= 0.40:
            return Signal(
                direction="short", option_type="PE",
                stop_points=stop_points,
                reason=(f"gap fade: gapped up {gap_pts:.0f} pts ({gap_atr:+.2f} ATR) but "
                        f"lost OR low {or_low:.1f} - fading toward prior close "
                        f"{ctx.prev_day_close:.1f}"),
                confidence=round(min(1.0, 0.4 + 0.6 * conviction), 3),
            )
        if not gapped_up and close > or_high and pressure >= 0.60:
            return Signal(
                direction="long", option_type="CE",
                stop_points=stop_points,
                reason=(f"gap fade: gapped down {gap_pts:.0f} pts ({gap_atr:+.2f} ATR) but "
                        f"reclaimed OR high {or_high:.1f} - fading toward prior close "
                        f"{ctx.prev_day_close:.1f}"),
                confidence=round(min(1.0, 0.4 + 0.6 * conviction), 3),
            )

        return NO_SIGNAL
