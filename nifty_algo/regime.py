"""
Day-regime classification.

This is not a strategy. It is the filter that decides which strategies are
allowed to speak today.

The reasoning: a long option is a long-volatility, negative-theta position.
On a range day the underlying goes nowhere, IV bleeds, and theta collects
against you every bar you hold - so a breakout system fires repeatedly into
chop and pays the friction each time. On an expansion day the same system is
the only thing that works.

Classifying the day BEFORE letting a strategy fire is the highest-leverage
filter in this whole system, and it costs one ATR comparison.

Pure functions and one dataclass. No I/O.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .config import Config, DEFAULT
from . import signals as sig


class Regime(str, Enum):
    UNKNOWN = "unknown"          # not enough bars yet - nothing may fire
    EXPANSION = "expansion"      # wide opening range, directional - breakouts work
    RANGE = "range"              # narrow, mean-reverting - theta eats buyers
    GAP_DAY = "gap_day"          # opened away from prior close - own playbook


@dataclass
class RegimeReading:
    regime: Regime
    or_width_atr: float          # opening-range width in ATR units
    vwap_slope_atr: float        # VWAP drift over the lookback, in ATR units
    gap_atr: float               # signed gap size in ATR units
    detail: str

    @property
    def is_directional(self) -> bool:
        return self.regime in (Regime.EXPANSION, Regime.GAP_DAY)


def classify(bars: pd.DataFrame, prev_day_close: float,
             cfg: Config = DEFAULT) -> RegimeReading:
    """
    Classify the current session from the bars seen so far.

    Deliberately uses only information available at the current bar, so the
    live engine and the backtester get identical readings.
    """
    r = cfg.regime
    s = cfg.session

    if len(bars) < max(cfg.signal.atr_period, 10):
        return RegimeReading(Regime.UNKNOWN, 0.0, 0.0, 0.0, "warming up")

    current_atr = float(sig.atr(bars, cfg.signal.atr_period).iloc[-1])
    if current_atr <= 0:
        return RegimeReading(Regime.UNKNOWN, 0.0, 0.0, 0.0, "zero atr")

    # --- gap ---
    session_open = float(bars["open"].iloc[0])
    _, gap_atr = sig.gap_metrics(session_open, prev_day_close, current_atr)

    # --- opening range width ---
    orange = sig.opening_range(bars, s.opening_range_minutes, s.bar_interval_minutes)
    or_width_atr = 0.0
    if orange:
        hi, lo = orange
        or_width_atr = (hi - lo) / current_atr

    # --- VWAP drift: is the session benchmark actually going somewhere? ---
    vw = sig.vwap(bars)
    lb = min(r.vwap_slope_lookback, len(vw) - 1)
    vwap_slope_atr = 0.0
    if lb > 0 and pd.notna(vw.iloc[-1]) and pd.notna(vw.iloc[-1 - lb]):
        vwap_slope_atr = (float(vw.iloc[-1]) - float(vw.iloc[-1 - lb])) / current_atr

    # --- decide ---
    if abs(gap_atr) >= r.gap_atr_multiple:
        return RegimeReading(
            Regime.GAP_DAY, or_width_atr, vwap_slope_atr, gap_atr,
            f"gap {gap_atr:+.2f} ATR from prior close",
        )

    if or_width_atr >= r.expansion_or_atr_multiple or abs(vwap_slope_atr) >= 1.0:
        return RegimeReading(
            Regime.EXPANSION, or_width_atr, vwap_slope_atr, gap_atr,
            f"OR width {or_width_atr:.2f} ATR, VWAP drift {vwap_slope_atr:+.2f} ATR",
        )

    if or_width_atr <= r.range_or_atr_multiple and abs(vwap_slope_atr) < 0.5:
        return RegimeReading(
            Regime.RANGE, or_width_atr, vwap_slope_atr, gap_atr,
            f"OR width {or_width_atr:.2f} ATR, flat VWAP - theta favours sellers",
        )

    # Between the two thresholds: no strong claim either way. Treat as
    # expansion-capable but say so, rather than pretending to certainty.
    return RegimeReading(
        Regime.EXPANSION, or_width_atr, vwap_slope_atr, gap_atr,
        f"indeterminate (OR {or_width_atr:.2f} ATR) - treated as expansion",
    )


def is_allowed(reading: RegimeReading, allowed: tuple[Regime, ...],
               cfg: Config = DEFAULT) -> bool:
    """Gate check used by the engine before running each strategy."""
    if not cfg.regime.enforce_regime_gate:
        return True
    if reading.regime is Regime.UNKNOWN:
        return False
    return reading.regime in allowed
