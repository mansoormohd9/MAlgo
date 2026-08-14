"""
The alert payload and the Notifier contract.

TradeAlert is what the whole system exists to produce: a fully-specified,
actionable instruction with the strike already chosen and the target and stop
already computed, so that at the moment it arrives there is nothing left to
decide and nothing to calculate under time pressure.

It carries `feed_latency_note` deliberately. A delayed-feed alert that reaches
your phone without that warning attached is worse than no alert - you would
act on a 15-minute-old level with no reminder that it is stale. The warning
travels with the message, not on a settings page.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class AlertKind(str, Enum):
    ENTRY = "entry"                  # a tradeable setup
    MANAGE = "manage"                # stop moved: breakeven shift or a trail
    PARTIAL_EXIT = "partial_exit"    # runner banked a lot at +2R
    EXIT = "exit"                    # a position closed
    CLOSE_ALL = "close_all"          # a day rule fired while positions were open
    FORCE_EXIT = "force_exit"        # 15:10 - flat before close, non-negotiable #2
    HALT = "halt"                    # session governor tripped
    KILL_SWITCH = "kill_switch"      # data gap or unhandled error
    TEST = "test"                    # channel test from the Settings page


# Kinds that describe an OPEN POSITION rather than a proposal. These must never
# be suppressed by the dedupe window: missing a duplicate entry alert costs you
# an opportunity, missing an exit alert costs you the trade.
POSITION_KINDS = frozenset({
    AlertKind.MANAGE, AlertKind.PARTIAL_EXIT, AlertKind.EXIT,
    AlertKind.CLOSE_ALL,
})


@dataclass
class TradeAlert:
    kind: AlertKind
    timestamp: datetime

    # --- what to trade ---
    strategy_key: str = ""
    strategy_label: str = ""
    direction: Optional[str] = None       # "long" | "short"
    option_type: Optional[str] = None     # "CE" | "PE"
    strike: Optional[int] = None
    expiry: Optional[str] = None

    # --- the numbers you act on ---
    entry_premium: float = 0.0
    target_premium: float = 0.0
    stop_premium: float = 0.0
    quantity: int = 0
    lots: int = 0
    rupee_risk: float = 0.0
    rupee_reward: float = 0.0
    delta: float = 0.0
    underlying_stop_points: float = 0.0
    underlying_price: float = 0.0

    # --- why, and how much to trust it ---
    reason: str = ""
    confidence: float = 0.0
    regime: str = ""

    # --- provenance ---
    feed_name: str = ""
    feed_latency_note: str = ""
    chain_source: str = ""
    chain_note: str = ""
    message: str = ""                     # for non-entry kinds

    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------- identity ----------------

    @property
    def dedupe_key(self) -> str:
        """
        What makes two alerts "the same alert".

        Bar timestamp is in the key because a 5-minute bar is re-evaluated on
        every 15-second UI refresh - roughly twenty times. Without this you
        would receive the same Telegram message twenty times per bar, and the
        one that actually mattered would be lost in the noise.
        """
        if self.kind is not AlertKind.ENTRY:
            return f"{self.kind.value}|{self.timestamp:%Y-%m-%d %H:%M}"
        return (f"{self.kind.value}|{self.strategy_key}|{self.direction}|"
                f"{self.strike}|{self.timestamp:%Y-%m-%d %H:%M}")

    @property
    def reward_risk(self) -> float:
        risk = self.entry_premium - self.stop_premium
        reward = self.target_premium - self.entry_premium
        return reward / risk if risk > 0 else 0.0

    # ---------------- rendering ----------------

    def title(self) -> str:
        if self.kind is AlertKind.ENTRY:
            return (f"{self.direction.upper()} {self.strike}{self.option_type} "
                    f"— {self.strategy_label}")
        return f"{self.kind.value.replace('_', ' ').upper()}"

    def as_text(self) -> str:
        """Plain text for Telegram, email, and desktop toast."""
        if self.kind is not AlertKind.ENTRY:
            return f"[{self.kind.value.upper()}] {self.message}"

        lines = [
            f"{self.direction.upper()}  {self.strike}{self.option_type}"
            f"{('  exp ' + self.expiry) if self.expiry else ''}",
            f"Strategy : {self.strategy_label}  (conf {self.confidence:.2f})",
            f"Regime   : {self.regime}",
            "",
            f"ENTRY    : {self.entry_premium:.2f}",
            f"TARGET   : {self.target_premium:.2f}   (+Rs {self.rupee_reward:,.0f})",
            f"STOP     : {self.stop_premium:.2f}   (-Rs {self.rupee_risk:,.0f})",
            f"QTY      : {self.quantity}  ({self.lots} lot)   R:R {self.reward_risk:.2f}",
            "",
            f"Underlying {self.underlying_price:,.1f}, "
            f"stop {self.underlying_stop_points:.0f} pts, delta {self.delta:.2f}",
            f"Why      : {self.reason}",
        ]
        if self.feed_latency_note:
            lines += ["", f"FEED: {self.feed_latency_note}"]
        if self.chain_note:
            lines += [f"CHAIN: {self.chain_note}"]
        lines += ["", "Alert only — this system never places an order."]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


class Notifier(ABC):
    """A delivery channel. Must never raise - a dead channel cannot stop the engine."""
    name: str = "base"

    @property
    @abstractmethod
    def configured(self) -> bool:
        """False when credentials are missing. The UI shows this on Settings."""

    @abstractmethod
    def send(self, alert: TradeAlert) -> tuple[bool, str]:
        """Return (delivered, detail). Detail explains any failure."""
