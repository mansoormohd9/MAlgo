"""
The risk engine.

Architectural rule: the strategy PROPOSES, risk DISPOSES. No order reaches
the broker without passing through RiskEngine.approve(). This is the layer
that enforces your session governors and, critically, chooses the strike.

Reuse this module unchanged in backtest and in live. If backtest and live
risk logic ever diverge, your backtest is measuring a different system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, time
from enum import Enum
from typing import Optional

from .config import Config, DEFAULT


class HaltReason(str, Enum):
    NONE = "none"
    SESSION_TARGET_HIT = "session_target_hit"
    SESSION_STOP_HIT = "session_stop_hit"
    MAX_ENTRIES_REACHED = "max_entries_reached"
    OUTSIDE_ENTRY_WINDOW = "outside_entry_window"
    EXPIRY_DAY_BLOCKED = "expiry_day_blocked"
    EVENT_BLACKOUT = "event_blackout"
    KILL_SWITCH = "kill_switch"


class RejectReason(str, Enum):
    NONE = "none"
    NO_VIABLE_STRIKE = "no_viable_strike"
    ILLIQUID_CONTRACT = "illiquid_contract"
    PREMIUM_OUT_OF_RANGE = "premium_out_of_range"
    LOT_COST_EXCEEDS_FREE_CAPITAL = "lot_cost_exceeds_free_capital"
    CORRELATION_LIMIT = "correlation_limit"
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"


@dataclass
class OptionQuote:
    strike: int
    option_type: str          # "CE" or "PE"
    premium: float
    delta: float
    bid: float
    ask: float
    open_interest: int

    @property
    def spread_pct(self) -> float:
        if self.premium <= 0:
            return 1.0
        return (self.ask - self.bid) / self.premium


@dataclass
class ApprovedOrder:
    quote: OptionQuote
    lots: int
    quantity: int
    entry_premium: float
    premium_stop: float        # absolute premium level
    premium_target: float
    underlying_stop_points: float
    rupee_risk: float
    rupee_reward: float


@dataclass
class RejectedOrder:
    reason: RejectReason
    detail: str = ""


@dataclass
class OpenPosition:
    quote: OptionQuote
    quantity: int
    entry_premium: float
    premium_stop: float
    premium_target: float
    direction: str             # "long_underlying" or "short_underlying"


@dataclass
class SessionState:
    """Resets every trading day."""
    trading_day: Optional[date] = None
    realised_pnl: float = 0.0
    entries_taken: int = 0
    halted: bool = False
    halt_reason: HaltReason = HaltReason.NONE
    open_positions: list[OpenPosition] = field(default_factory=list)

    def reset(self, day: date) -> None:
        self.trading_day = day
        self.realised_pnl = 0.0
        self.entries_taken = 0
        self.halted = False
        self.halt_reason = HaltReason.NONE
        self.open_positions = []


class RiskEngine:
    def __init__(self, cfg: Config = DEFAULT, starting_capital: float | None = None):
        self.cfg = cfg
        self.capital = starting_capital or cfg.capital.starting_capital
        self.session = SessionState()
        self._kill_switch = False

    # ---------------- session governors ----------------

    def start_day(self, day: date) -> None:
        self.session.reset(day)

    def _session_target_rupees(self) -> float:
        return self.capital * self.cfg.capital.session_target_pct

    def _session_stop_rupees(self) -> float:
        return -abs(self.capital * self.cfg.capital.session_stop_pct)

    def check_halt(self, now: time, is_expiry_day: bool = False,
                   in_event_blackout: bool = False) -> HaltReason:
        """
        Evaluate all stop conditions. 'Whichever reaches first' - once any
        one trips, the session is done for new entries.
        """
        s = self.session
        if self._kill_switch:
            return self._halt(HaltReason.KILL_SWITCH)
        if s.halted:
            return s.halt_reason

        if s.realised_pnl >= self._session_target_rupees():
            return self._halt(HaltReason.SESSION_TARGET_HIT)
        if s.realised_pnl <= self._session_stop_rupees():
            return self._halt(HaltReason.SESSION_STOP_HIT)
        if s.entries_taken >= self.cfg.capital.max_entries_per_session:
            return self._halt(HaltReason.MAX_ENTRIES_REACHED)

        # Non-latching conditions - these block entry but do not end the day
        if is_expiry_day and not self.cfg.session.trade_on_expiry_day:
            return HaltReason.EXPIRY_DAY_BLOCKED
        if in_event_blackout:
            return HaltReason.EVENT_BLACKOUT
        if not (self.cfg.session.entry_start <= now <= self.cfg.session.entry_cutoff):
            return HaltReason.OUTSIDE_ENTRY_WINDOW

        return HaltReason.NONE

    def _halt(self, reason: HaltReason) -> HaltReason:
        self.session.halted = True
        self.session.halt_reason = reason
        return reason

    def trip_kill_switch(self) -> None:
        """Call this on any unhandled exception, data gap, or broker error."""
        self._kill_switch = True
        self._halt(HaltReason.KILL_SWITCH)

    # ---------------- strike selection ----------------

    def required_max_delta(self, underlying_stop_points: float) -> float:
        """
        THE CORE CONSTRAINT.

        Your rupee risk and the lot size together cap how much premium you
        can afford to lose per unit. Delta converts an underlying move into
        a premium move. So:

            max_premium_loss_per_unit = risk_rupees / lot_size
            required_delta <= max_premium_loss_per_unit / stop_points

        The stop chooses the strike, not the other way round.
        """
        lot = self.cfg.instrument.lot_size
        risk = self.cfg.capital.risk_per_trade_rupees
        max_premium_loss_per_unit = risk / lot
        if underlying_stop_points <= 0:
            return 0.0
        return max_premium_loss_per_unit / underlying_stop_points

    def select_strike(self, chain: list[OptionQuote],
                      underlying_stop_points: float,
                      option_type: str) -> Optional[OptionQuote]:
        """
        Pick the highest-delta contract that still fits the risk budget and
        passes liquidity gates. Highest delta = best directional capture
        per rupee of theta paid.
        """
        inst = self.cfg.instrument
        sig = self.cfg.signal
        ceiling = min(self.required_max_delta(underlying_stop_points), sig.max_delta)

        candidates = [
            q for q in chain
            if q.option_type == option_type
            and sig.min_delta <= abs(q.delta) <= ceiling
            and inst.min_premium <= q.premium <= inst.max_premium
            and q.spread_pct <= inst.max_spread_pct_of_premium
            and q.open_interest >= inst.min_open_interest
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda q: abs(q.delta))

    # ---------------- approval ----------------

    def approve(self, chain: list[OptionQuote], option_type: str,
                underlying_stop_points: float,
                free_capital: float) -> ApprovedOrder | RejectedOrder:
        quote = self.select_strike(chain, underlying_stop_points, option_type)
        if quote is None:
            ceiling = self.required_max_delta(underlying_stop_points)
            return RejectedOrder(
                RejectReason.NO_VIABLE_STRIKE,
                f"needed delta <= {ceiling:.3f} with premium "
                f"{self.cfg.instrument.min_premium}-{self.cfg.instrument.max_premium} "
                f"and spread <= {self.cfg.instrument.max_spread_pct_of_premium:.2%}"
            )

        lot = self.cfg.instrument.lot_size
        lots = 1                                # capital supports exactly one
        quantity = lots * lot
        lot_cost = quote.premium * quantity

        if lot_cost > free_capital:
            return RejectedOrder(
                RejectReason.LOT_COST_EXCEEDS_FREE_CAPITAL,
                f"lot cost Rs {lot_cost:,.0f} > free capital Rs {free_capital:,.0f}"
            )

        # Correlation guard: three same-direction Nifty options is one trade
        # at 3x size, not three trades. Cap same-direction concurrency.
        same_dir = sum(1 for p in self.session.open_positions
                       if p.quote.option_type == option_type)
        if same_dir >= 2:
            return RejectedOrder(
                RejectReason.CORRELATION_LIMIT,
                f"already {same_dir} open {option_type} positions"
            )

        risk_rupees = self.cfg.capital.risk_per_trade_rupees
        reward_rupees = self.cfg.capital.reward_per_trade_rupees

        premium_loss_per_unit = risk_rupees / quantity
        premium_gain_per_unit = reward_rupees / quantity

        premium_stop = max(quote.premium - premium_loss_per_unit, 0.05)
        premium_target = quote.premium + premium_gain_per_unit

        return ApprovedOrder(
            quote=quote,
            lots=lots,
            quantity=quantity,
            entry_premium=quote.premium,
            premium_stop=round(premium_stop / self.cfg.instrument.tick_size)
                         * self.cfg.instrument.tick_size,
            premium_target=round(premium_target / self.cfg.instrument.tick_size)
                           * self.cfg.instrument.tick_size,
            underlying_stop_points=underlying_stop_points,
            rupee_risk=risk_rupees,
            rupee_reward=reward_rupees,
        )

    # ---------------- bookkeeping ----------------

    def register_entry(self, order: ApprovedOrder, direction: str) -> None:
        self.session.entries_taken += 1
        self.session.open_positions.append(OpenPosition(
            quote=order.quote,
            quantity=order.quantity,
            entry_premium=order.entry_premium,
            premium_stop=order.premium_stop,
            premium_target=order.premium_target,
            direction=direction,
        ))

    def register_exit(self, position: OpenPosition, net_pnl: float) -> None:
        self.session.realised_pnl += net_pnl
        self.capital += net_pnl
        if position in self.session.open_positions:
            self.session.open_positions.remove(position)

    def summary(self) -> dict:
        c = self.cfg.capital
        return {
            "capital": round(self.capital, 2),
            "session_pnl": round(self.session.realised_pnl, 2),
            "entries_taken": self.session.entries_taken,
            "max_entries": c.max_entries_per_session,
            "risk_per_trade": round(c.risk_per_trade_rupees, 2),
            "reward_per_trade": round(c.reward_per_trade_rupees, 2),
            "reward_risk_ratio": round(c.reward_risk_ratio, 2),
            "halted": self.session.halted,
            "halt_reason": self.session.halt_reason.value,
        }
