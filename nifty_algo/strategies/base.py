"""
Shared base for the added strategies.

Adds three pieces of metadata the registry, UI, and engine all read, without
changing the Strategy/Signal contract in ../strategy.py:

  key              stable identifier used in config, journal, and dedupe keys
  allowed_regimes  which day types this setup may fire in (see ../regime.py)
  default_enabled  ships on, or ships off

`default_enabled` is False for most of these on purpose. The README's advice
is to trade one idea at a time - run eight strategies at once and you cannot
attribute a losing week to any of them.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config, DEFAULT
from ..regime import Regime
from ..strategy import Strategy, Signal, Context, NO_SIGNAL
from .. import signals as sig


class BaseStrategy(Strategy):
    key: str = "base"
    label: str = "Base"
    description: str = ""
    allowed_regimes: tuple[Regime, ...] = (Regime.EXPANSION,)
    default_enabled: bool = False

    #: bars needed before this strategy will look at anything
    min_bars: int = 25

    def __init__(self, cfg: Config = DEFAULT):
        super().__init__(cfg)

    # ---------------- shared preconditions ----------------

    def _preflight(self, ctx: Context) -> tuple[float, str | None]:
        """
        Checks every strategy needs. Returns (atr, reason_to_abort).

        Kept here rather than duplicated eight times - and deliberately
        identical to the checks LevelBreakStrategy already makes, so the
        strategies are comparable to each other in a backtest.
        """
        df = ctx.bars
        if len(df) < max(self.cfg.signal.atr_period, self.min_bars):
            return 0.0, "warming up"
        if not sig.underlying_liquidity_ok(df):
            return 0.0, "thin liquidity"
        current_atr = float(sig.atr(df, self.cfg.signal.atr_period).iloc[-1])
        if current_atr <= 0 or pd.isna(current_atr):
            return 0.0, "zero atr"
        return current_atr, None

    def _stop_points(self, current_atr: float) -> float:
        return current_atr * self.cfg.signal.atr_stop_multiple

    def _reject(self, reason: str) -> Signal:
        return Signal(None, None, reason=reason)

    @staticmethod
    def _pressure(df: pd.DataFrame) -> float:
        v = sig.close_position_in_range(df).iloc[-1]
        return 0.5 if pd.isna(v) else float(v)

    @staticmethod
    def _conviction(df: pd.DataFrame) -> float:
        v = sig.body_to_range(df).iloc[-1]
        return 0.0 if pd.isna(v) else float(v)


__all__ = ["BaseStrategy", "Signal", "Context", "NO_SIGNAL", "Regime"]
