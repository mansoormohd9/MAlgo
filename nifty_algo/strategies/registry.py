"""
The single source of truth for "which strategies exist".

The UI, the live engine, and the backtester all enumerate strategies from
here, so a strategy can never be live-tradeable but un-backtestable, or
tunable in the UI but absent from the engine.

The two original strategies in ../strategy.py are wrapped rather than edited.
`LevelBreakStrategy` is untouched; the adapters below only attach the metadata
(key, label, regimes, default) that the registry needs. That keeps the file
the README calls out as architecturally sacred exactly as it was.
"""
from __future__ import annotations
from dataclasses import dataclass

from ..config import Config, DEFAULT
from ..regime import Regime
from ..strategy import LevelBreakStrategy, Strategy

from .trendline_break import TrendlineBreakStrategy
from .vwap import VwapReclaimStrategy
from .orb import OpeningRangeRetestStrategy
from .failed_breakout import FailedBreakoutStrategy
from .trend_pullback import TrendPullbackStrategy
from .squeeze import VolatilitySqueezeStrategy
from .sweep_reclaim import PriorDaySweepStrategy
from .gap import GapStrategy


# ---------------- adapters for the two original strategies ----------------

class LevelBreak(LevelBreakStrategy):
    """Your original strategy #1, metadata attached, logic untouched."""
    key = "level_break"
    label = "Level break"
    description = ("Close beyond a clustered support/resistance level with a volume "
                   "surge and a conviction candle. The original strategy.")
    allowed_regimes = (Regime.EXPANSION, Regime.GAP_DAY)
    default_enabled = True

    def __init__(self, cfg: Config = DEFAULT):
        super().__init__(cfg, enable_doji_reversal=False)


class DojiReversal(LevelBreakStrategy):
    """Your original variant B, exposed as a strategy in its own right."""
    key = "doji_reversal"
    label = "Doji reversal at level"
    description = ("Indecision candle sitting on a level, confirmed by the next bar "
                   "closing beyond the doji's range. Original variant B.")
    allowed_regimes = (Regime.RANGE, Regime.EXPANSION)
    default_enabled = False

    def __init__(self, cfg: Config = DEFAULT):
        super().__init__(cfg, enable_doji_reversal=True)


# ---------------- the registry ----------------

STRATEGY_CLASSES: tuple[type, ...] = (
    LevelBreak,
    TrendlineBreakStrategy,
    VwapReclaimStrategy,
    VolatilitySqueezeStrategy,
    OpeningRangeRetestStrategy,
    FailedBreakoutStrategy,
    TrendPullbackStrategy,
    PriorDaySweepStrategy,
    GapStrategy,
    DojiReversal,
)


@dataclass
class StrategyInfo:
    key: str
    label: str
    description: str
    allowed_regimes: tuple[Regime, ...]
    default_enabled: bool
    cls: type

    @property
    def regimes_text(self) -> str:
        return ", ".join(r.value for r in self.allowed_regimes)


def all_strategies() -> list[StrategyInfo]:
    """Metadata for every registered strategy, in display order."""
    return [
        StrategyInfo(
            key=c.key,
            label=c.label,
            description=c.description,
            allowed_regimes=c.allowed_regimes,
            default_enabled=c.default_enabled,
            cls=c,
        )
        for c in STRATEGY_CLASSES
    ]


def default_enabled_keys() -> list[str]:
    return [s.key for s in all_strategies() if s.default_enabled]


def build_enabled(keys: list[str] | None = None,
                  cfg: Config = DEFAULT) -> dict[str, Strategy]:
    """
    Instantiate the requested strategies. `None` means "the shipped defaults".

    Defaults are level break, trendline break, VWAP, and squeeze - four setups
    that fire on genuinely different conditions. Turning all ten on is
    available and inadvisable: overlapping signals on the same bar burn your
    three daily entries before 11:00 and make a losing week unattributable.
    """
    if keys is None:
        keys = default_enabled_keys()
    wanted = set(keys)
    return {
        info.key: info.cls(cfg)
        for info in all_strategies()
        if info.key in wanted
    }


def get_info(key: str) -> StrategyInfo | None:
    for info in all_strategies():
        if info.key == key:
            return info
    return None
