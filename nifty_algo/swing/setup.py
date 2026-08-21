"""
Turning a daily chart into a ticket: entry, stop, target.

Every indicator here comes from `nifty_algo.signals`. Nothing is
reimplemented. Those functions are pure `bars in, values out` and carry no
assumption about bar size, so the same `atr`, `build_levels`, `trend_state`
and `sweep_reclaim` that read 5-minute NIFTY bars read daily stock bars
unchanged - and any fix to them fixes both books at once.

TWO THINGS THAT ARE DIFFERENT ON DAILY BARS

  Volume is real. The index carries none, so `volume_surge` degrades to a
  range proxy there (see the comment block in signals.py). Stocks have
  genuine traded volume, so here it is the actual participation test.

  EMAs are 20/50, not 9/21. A 9-period EMA on daily bars is a two-week
  average; at a multi-day holding horizon that is noise dressed as trend.

LOOK-AHEAD: `find_pivots` uses a CENTRED window, so the most recent
`pivot_lookback` bars can never be confirmed as pivots. That is a feature -
it means a level is only a level once the market has had time to respect it -
but it also means every level used here was formed at least five sessions ago
and is knowable today. Nothing in this module reads a bar after the signal
bar. `tests/test_swing_setup.py` asserts it by truncating the frame and
checking the ticket does not move.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..signals import (atr, build_levels, close_position_in_range,
                       consecutive_closes, ema, narrowest_range_n,
                       range_percentile, sweep_reclaim, trend_state,
                       volume_surge)

LONG = "long"


@dataclass
class SwingSetup:
    """A fully specified long candidate, before risk sizing."""
    symbol: str
    key: str                       # breakout | pullback | squeeze | reclaim
    label: str
    entry: float
    stop: float
    target: float
    trigger_note: str              # what has to happen for you to be filled
    quality: float                 # 0..1, how good the pattern is
    reasons: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def risk_points(self) -> float:
        return self.entry - self.stop

    @property
    def reward_points(self) -> float:
        return self.target - self.entry

    @property
    def reward_risk(self) -> float:
        return self.reward_points / self.risk_points if self.risk_points > 0 else 0.0

    @property
    def stop_pct(self) -> float:
        return self.risk_points / self.entry if self.entry > 0 else 0.0

    @property
    def target_pct(self) -> float:
        return self.reward_points / self.entry if self.entry > 0 else 0.0


def min_bars(cfg) -> int:
    """Bars needed before any of this means anything."""
    return cfg.swing.ema_slow + cfg.swing.atr_period + 10


def detect(symbol: str, df: pd.DataFrame, cfg) -> tuple[SwingSetup | None, str]:
    """
    Best long setup on the last bar of `df`, or (None, why-not).

    Returns the reason as well as the setup because "nothing fired" and
    "four things fired and every target was too close" are different facts
    about the day, and only one of them means the scanner is working.
    """
    if len(df) < min_bars(cfg):
        return None, f"only {len(df)} daily bars, need {min_bars(cfg)}"

    ctx = _context(df, cfg)
    if ctx["atr"] <= 0:
        return None, "ATR is zero - no usable volatility estimate"

    candidates: list[SwingSetup] = []
    notes: list[str] = []
    for builder in (_breakout, _pullback, _squeeze, _reclaim):
        found, note = builder(symbol, df, ctx, cfg)
        if found:
            candidates.append(found)
        elif note:
            notes.append(note)

    if not candidates:
        return None, "; ".join(notes) if notes else "no setup on the last bar"

    # Several archetypes can describe the same bar. Take the best-quality one
    # and record that the others agreed - confluence is worth saying out loud.
    candidates.sort(key=lambda s: s.quality, reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        others = ", ".join(c.label for c in candidates[1:])
        best.reasons.append(f"Also reads as: {others}.")
        best.detail["confluence"] = [c.key for c in candidates[1:]]
    return best, ""


# ---------------- shared context, computed once ----------------

def _context(df: pd.DataFrame, cfg) -> dict:
    swing = cfg.swing
    a = float(atr(df, swing.atr_period).iloc[-1])
    close = float(df["close"].iloc[-1])
    fast = ema(df["close"], swing.ema_fast)
    slow = ema(df["close"], swing.ema_slow)
    levels = build_levels(df, swing.pivot_lookback,
                          swing.level_cluster_atr_frac,
                          swing.min_level_touches)
    trend = trend_state(df, swing.ema_fast, swing.ema_slow)
    surge = volume_surge(df, 20, swing.volume_surge_multiple)

    return {
        "atr": a,
        "close": close,
        "ema_fast": float(fast.iloc[-1]),
        "ema_slow": float(slow.iloc[-1]),
        "ema_fast_rising": bool(fast.diff().iloc[-1] > 0),
        "ema_slow_rising": bool(slow.diff().iloc[-1] > 0),
        "trend": int(trend.iloc[-1]),
        "levels": levels,
        "resistances": sorted((l for l in levels if l.kind == "resistance"),
                              key=lambda l: l.price),
        "supports": sorted((l for l in levels if l.kind == "support"),
                           key=lambda l: l.price, reverse=True),
        "volume_surge": bool(surge.iloc[-1]),
        "close_position": float(close_position_in_range(df).iloc[-1] or 0.0),
        "up_closes": consecutive_closes(df, +1, 5),
        "nr7": bool(narrowest_range_n(df, swing.squeeze_lookback).iloc[-1]),
        "range_pct": float(range_percentile(df, 20).iloc[-1] or 1.0),
        "recent_high": float(df["high"].iloc[-swing.stop_structure_bars:].max()),
        "recent_low": float(df["low"].iloc[-swing.stop_structure_bars:].min()),
    }


# ---------------- the four archetypes ----------------

def _breakout(symbol, df, ctx, cfg) -> tuple[SwingSetup | None, str]:
    """
    Price pressing a resistance that has held at least twice, in an uptrend,
    with participation. Entry is a stop-buy above the level, so an approach
    that stalls never fills.
    """
    swing = cfg.swing
    a, close = ctx["atr"], ctx["close"]

    if ctx["ema_fast"] <= ctx["ema_slow"]:
        return None, ""

    overhead = [l for l in ctx["resistances"] if l.price > close]
    if overhead:
        level = overhead[0]
        trigger = level.price
        touches = level.touches
        basis = f"resistance at {level.price:,.1f} touched {touches}x"
    else:
        # Nothing left overhead: the stock is at the top of its own range.
        # The trigger becomes the recent high itself.
        trigger = ctx["recent_high"]
        touches = 1
        basis = f"{swing.stop_structure_bars}-day high at {trigger:,.1f}, no resistance overhead"

    entry = trigger + swing.breakout_buffer_atr_frac * a
    if entry - close > swing.max_entry_distance_atr * a:
        return None, (f"breakout trigger {(entry - close) / a:.1f} ATR away - "
                      f"too far to be today's trade")

    resolved = _resolve(entry, df, ctx, cfg)
    if resolved is None:
        return None, "breakout: no target far enough above entry"
    stop, target, target_basis = resolved

    quality = _quality(ctx, base=0.45, touches=touches)
    reasons = [
        f"Uptrend intact - the {swing.ema_fast}-day EMA ({ctx['ema_fast']:,.1f}) "
        f"is above the {swing.ema_slow}-day ({ctx['ema_slow']:,.1f}).",
        f"Pressing {basis}.",
        f"Entry is a stop-buy {swing.breakout_buffer_atr_frac:.2f} ATR above the "
        f"level, so a failed approach never fills you.",
        target_basis,
    ]
    if ctx["volume_surge"]:
        reasons.append("Volume on the last bar ran above its 20-day average - "
                       "real participation, not a drift.")

    return SwingSetup(
        symbol, "breakout", "Level breakout", entry, stop, target,
        trigger_note=f"buy stop at {entry:,.2f}, valid while price holds above "
                     f"{ctx['ema_fast']:,.1f}",
        quality=quality, reasons=reasons,
        detail={"trigger_level": trigger, "touches": touches,
                "atr": a, "basis": basis},
    ), ""


def _pullback(symbol, df, ctx, cfg) -> tuple[SwingSetup | None, str]:
    """
    An established uptrend that has come back to its fast EMA and stopped
    going down. The cheapest entry of the four, and the one that needs the
    trend to be real rather than recent.
    """
    swing = cfg.swing
    a, close = ctx["atr"], ctx["close"]

    if not (ctx["ema_fast"] > ctx["ema_slow"] and ctx["ema_slow_rising"]):
        return None, ""

    distance = abs(close - ctx["ema_fast"]) / a if a > 0 else 99.0
    if distance > swing.pullback_max_distance_atr:
        return None, ""
    if close < ctx["ema_slow"]:
        return None, "pullback went through the slow EMA - that is a trend change"
    if ctx["close_position"] < 0.5:
        return None, "pullback bar closed in its lower half - sellers still in control"

    entry = float(df["high"].iloc[-1]) + swing.breakout_buffer_atr_frac * a
    if entry - close > swing.max_entry_distance_atr * a:
        return None, ""

    resolved = _resolve(entry, df, ctx, cfg)
    if resolved is None:
        return None, "pullback: no target far enough above entry"
    stop, target, target_basis = resolved

    quality = _quality(ctx, base=0.50, touches=2)
    reasons = [
        f"Uptrend with both EMAs rising; price has pulled back to within "
        f"{distance:.2f} ATR of the {swing.ema_fast}-day EMA rather than "
        f"breaking it.",
        f"The pullback bar closed in the top "
        f"{ctx['close_position']:.0%} of its range - buyers took the bar back.",
        f"Entry is above that bar's high, so you are paid to wait for the "
        f"trend to resume rather than catching a falling knife.",
        target_basis,
    ]
    return SwingSetup(
        symbol, "pullback", "Trend pullback", entry, stop, target,
        trigger_note=f"buy stop at {entry:,.2f} (above yesterday's high)",
        quality=quality, reasons=reasons,
        detail={"distance_to_ema_atr": distance, "atr": a},
    ), ""


def _squeeze(symbol, df, ctx, cfg) -> tuple[SwingSetup | None, str]:
    """
    Compression before expansion. The one setup that gets you positioned
    before the move rather than after it, which is also why it fires least
    often and is worth the least on its own.
    """
    swing = cfg.swing
    a, close = ctx["atr"], ctx["close"]

    if ctx["trend"] < 0:
        return None, ""
    if not (ctx["nr7"] or ctx["range_pct"] <= 0.25):
        return None, ""

    trigger = ctx["recent_high"]
    entry = trigger + swing.breakout_buffer_atr_frac * a
    if entry - close > swing.max_entry_distance_atr * a:
        return None, "squeeze: the range is wider than one ATR, not a squeeze"

    resolved = _resolve(entry, df, ctx, cfg)
    if resolved is None:
        return None, "squeeze: no target far enough above entry"
    stop, target, target_basis = resolved

    quality = _quality(ctx, base=0.40, touches=1)
    reasons = [
        f"Range has compressed - last bar is "
        + ("the narrowest of the last "
           f"{swing.squeeze_lookback} " if ctx["nr7"] else
           f"in the bottom {ctx['range_pct']:.0%} of its 20-day range ")
        + "sessions. Compression precedes expansion.",
        f"Trend is not down, so the expansion has a direction to favour.",
        f"Entry sits above the {swing.stop_structure_bars}-day high at "
        f"{trigger:,.1f} - the squeeze has to actually break to pay you.",
        target_basis,
    ]
    return SwingSetup(
        symbol, "squeeze", "Volatility squeeze", entry, stop, target,
        trigger_note=f"buy stop at {entry:,.2f} on expansion out of the range",
        quality=quality, reasons=reasons,
        detail={"nr7": ctx["nr7"], "range_percentile": ctx["range_pct"], "atr": a},
    ), ""


def _reclaim(symbol, df, ctx, cfg) -> tuple[SwingSetup | None, str]:
    """
    A support level was swept and immediately reclaimed - the stop-hunt trade.
    Pays on exactly the chop that punishes the breakout setup.
    """
    swing = cfg.swing
    a, close = ctx["atr"], ctx["close"]

    if not ctx["supports"]:
        return None, ""
    support = ctx["supports"][0]

    verdict = sweep_reclaim(df, support.price, a,
                            swing.sweep_min_wick_atr,
                            swing.sweep_max_close_back_atr)
    if verdict != "bullish":
        return None, ""
    if close < ctx["ema_slow"]:
        return None, "reclaim below the slow EMA - that is a downtrend, not a sweep"

    entry = float(df["high"].iloc[-1]) + swing.breakout_buffer_atr_frac * a
    if entry - close > swing.max_entry_distance_atr * a:
        return None, ""

    resolved = _resolve(entry, df, ctx, cfg)
    if resolved is None:
        return None, "reclaim: no target far enough above entry"
    stop, target, target_basis = resolved

    quality = _quality(ctx, base=0.45, touches=support.touches)
    reasons = [
        f"Support at {support.price:,.1f} (touched {support.touches}x) was "
        f"wicked through and reclaimed on the same bar - the sellers who "
        f"broke it could not hold it.",
        f"Price is still above the {swing.ema_slow}-day EMA, so this is a "
        f"shakeout inside a trend rather than a breakdown.",
        f"Stop goes under the sweep low, which is exactly where the idea is wrong.",
        target_basis,
    ]
    return SwingSetup(
        symbol, "reclaim", "Sweep and reclaim", entry, stop, target,
        trigger_note=f"buy stop at {entry:,.2f} (above the reclaim bar)",
        quality=quality, reasons=reasons,
        detail={"support": support.price, "touches": support.touches, "atr": a},
    ), ""


# ---------------- stop, target, quality ----------------

def _resolve(entry: float, df: pd.DataFrame, ctx: dict,
             cfg) -> tuple[float, float, str] | None:
    """
    Place the stop at structure and find the first real target above entry.

    Returns (stop, target, one-line explanation) or None when there is no
    target far enough away to be worth the risk - which is a rejection the
    caller should report, not paper over by inventing a level.
    """
    swing = cfg.swing
    a = ctx["atr"]

    # --- stop: structure, clamped into the ATR band ---
    structural = ctx["recent_low"] - 0.10 * a
    tightest = entry - swing.swing_atr_stop_min_multiple * a
    widest = entry - swing.swing_atr_stop_multiple * a
    stop = max(widest, min(structural, tightest))
    if stop <= 0 or stop >= entry:
        return None

    # --- target: the first resistance that is actually far enough away ---
    floor = entry + swing.target_min_atr * a
    ceiling = entry + swing.target_max_atr * a

    overhead = [l for l in ctx["resistances"] if l.price >= floor]
    if overhead:
        level = overhead[0]
        basis = (f"Target is the next resistance at {level.price:,.1f}, touched "
                 f"{level.touches}x - where the move realistically stalls, not "
                 f"an arbitrary multiple.")
        return stop, *_cap(level.price, ceiling, basis, swing, a, entry)

    # Nothing overhead. Use the measured move: a base that took N sessions to
    # build tends to project its own height once it breaks.
    window = df.iloc[-swing.measured_move_bars:]
    height = float(window["high"].max() - window["low"].min())
    target = entry + height
    if target < floor:
        return None
    basis = (f"No resistance left overhead, so the target is a measured move - "
             f"the {swing.measured_move_bars}-day base is {height:,.1f} points "
             f"tall.")
    return stop, *_cap(target, ceiling, basis, swing, a, entry)


def _cap(target: float, ceiling: float, basis: str, swing, a: float,
         entry: float) -> tuple[float, str]:
    """
    Hold the target inside what the holding window can actually deliver.

    A resistance eight ATR overhead is a real level and a fictional target:
    over roughly ten sessions a stock covers about sqrt(10) ATR, so a trade
    aimed that far will time out long before it arrives, while the R:R it
    advertises does the ranking a favour it has not earned. When the honest
    level is beyond reach the ticket shows the reachable number and says so -
    the level itself stays in the sentence, because it is still where you
    would trail towards if the move keeps going.
    """
    if target <= ceiling:
        return target, basis
    return ceiling, (
        f"{basis} That level is {(target - entry) / a:.1f} ATR away, further "
        f"than this holding window realistically reaches, so the target shown "
        f"is capped at {swing.target_max_atr:.0f} ATR ({ceiling:,.1f}). Treat "
        f"the level above as where to trail, not where to expect a fill."
    )


def _quality(ctx: dict, base: float, touches: int) -> float:
    """
    How good the pattern is, 0..1, before any of the ranking metrics.

    Deliberately crude and additive: this feeds one of six weighted score
    components, and a precise-looking number built from six fudge factors
    would be a false precision that survives into the ranking.
    """
    score = base
    if ctx["trend"] > 0:
        score += 0.15
    if ctx["ema_fast_rising"] and ctx["ema_slow_rising"]:
        score += 0.10
    if ctx["volume_surge"]:
        score += 0.10
    score += min(0.10, 0.03 * max(0, touches - 1))
    score += 0.05 * min(1.0, ctx["close_position"])
    return round(min(1.0, score), 3)
