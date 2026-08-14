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
from .governor import SessionGovernor, GovernorReason, GovernorVerdict


class HaltReason(str, Enum):
    NONE = "none"
    SESSION_TARGET_HIT = "session_target_hit"
    SESSION_STOP_HIT = "session_stop_hit"
    GIVE_BACK_FLOOR_HIT = "give_back_floor_hit"
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
    # Implied volatility. Flat and assumed on a synthetic chain; solved per
    # strike from the real premium on a broker chain, which is the only way
    # the volatility SKEW becomes visible. Optional because a stale or crossed
    # quote cannot be inverted, and a guessed IV is worse than an absent one.
    iv: Optional[float] = None
    volume: int = 0

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
    runner_enabled: bool = True
    sizing_note: str = ""      # why this many lots - surfaced in the alert


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
        # The day rules live in governor.py; this holds one and delegates, so
        # there is a single implementation of the ratchet for the engine, the
        # backtester and the daily brief to share.
        self.governor = SessionGovernor(cfg=cfg, capital=self.capital)
        self._kill_switch = False

    # ---------------- session governors ----------------

    def start_day(self, day: date) -> None:
        self.session.reset(day)
        self.governor.start_day(self.capital)

    def _session_target_rupees(self) -> float:
        return self.governor.target

    def _session_stop_rupees(self) -> float:
        """
        The RATCHETING floor, not a constant -5%.

        Kept under the original name because engine, UI and tests all read it;
        what changed is that it now trails the day's peak realised P&L. See
        governor.SessionGovernor.floor.
        """
        return self.governor.floor

    def check_halt(self, now: time, is_expiry_day: bool = False,
                   in_event_blackout: bool = False) -> HaltReason:
        """
        Evaluate all stop conditions. 'Whichever reaches first' - once any
        one trips, the session is done for new entries.

        NOTE: this gates ENTRIES. When the target or the floor trips there may
        still be an open position, and blocking entries does nothing about it -
        `governor.evaluate()` returns CLOSE_ALL for that case and the engine
        acts on it after every realised exit.
        """
        s = self.session
        if self._kill_switch:
            return self._halt(HaltReason.KILL_SWITCH)
        if s.halted:
            return s.halt_reason

        verdict = self.governor.evaluate()
        if verdict.reason is GovernorReason.SESSION_TARGET_HIT:
            return self._halt(HaltReason.SESSION_TARGET_HIT)
        if verdict.reason is GovernorReason.GIVE_BACK_FLOOR_HIT:
            return self._halt(HaltReason.GIVE_BACK_FLOOR_HIT)
        if verdict.reason is GovernorReason.MAX_ENTRIES_REACHED:
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

    def required_max_delta(self, underlying_stop_points: float,
                           lots: int = 1) -> float:
        """
        THE CORE CONSTRAINT.

        Your rupee risk and the position size together cap how much premium you
        can afford to lose per unit. Delta converts an underlying move into
        a premium move. So:

            max_premium_loss_per_unit = risk_rupees / (lot_size x lots)
            required_delta <= max_premium_loss_per_unit / stop_points

        The stop chooses the strike, not the other way round.

        Note `lots` in the denominator: doubling the size does NOT double the
        risk, it halves the delta you may buy. Risk per trade is fixed at
        1.667% of capital and everything else bends around it.
        """
        qty = self.cfg.instrument.lot_size * max(lots, 1)
        risk = self.cfg.capital.risk_per_trade_rupees
        max_premium_loss_per_unit = risk / qty
        if underlying_stop_points <= 0:
            return 0.0
        return max_premium_loss_per_unit / underlying_stop_points

    def select_strike(self, chain: list[OptionQuote],
                      underlying_stop_points: float,
                      option_type: str, lots: int = 1) -> Optional[OptionQuote]:
        """
        Pick the highest-delta contract that still fits the risk budget and
        passes liquidity gates. Highest delta = best directional capture
        per rupee of theta paid.
        """
        candidates = self.viable_strikes(chain, underlying_stop_points,
                                         option_type, lots)
        if not candidates:
            return None
        return max(candidates, key=lambda q: abs(q.delta))

    def viable_strikes(self, chain: list[OptionQuote],
                       underlying_stop_points: float, option_type: str,
                       lots: int = 1) -> list[OptionQuote]:
        """Every quote that passes all gates. The daily brief renders these."""
        inst = self.cfg.instrument
        sig = self.cfg.signal
        ceiling = min(self.required_max_delta(underlying_stop_points, lots),
                      sig.max_delta)
        return [
            q for q in chain
            if q.option_type == option_type
            and sig.min_delta <= abs(q.delta) <= ceiling
            and inst.min_premium <= q.premium <= inst.max_premium
            and q.spread_pct <= inst.max_spread_pct_of_premium
            and q.open_interest >= inst.min_open_interest
        ]

    def lot_ladder(self) -> list[int]:
        """Sizes to try, largest first. One definition, three callers."""
        t = self.cfg.trade
        sizes = [t.preferred_lots, t.min_lots] if t.enable_runner else [t.min_lots]
        return sorted({max(int(n), 1) for n in sizes}, reverse=True)

    def lots_for(self, underlying_stop_points: float) -> int:
        """
        The largest size whose delta ceiling still admits a strike IN PRINCIPLE.

        `approve()` needs the real chain to decide; the backtester and the daily
        brief need an answer without one. This is the shared rule so the three
        cannot drift - a backtest that sizes trades the live engine would refuse
        is measuring a different system.

        UPPER BOUND, NOT A PROMISE. This only checks that the delta ceiling
        clears `min_delta`; it cannot know whether a strike actually EXISTS in
        that band, because strikes are discrete (50 points apart) and the band
        gets very narrow as the ceiling approaches the floor. Around a 63-64
        point stop the two-lot band is roughly 0.200-0.203 wide and may contain
        nothing at all, in which case `approve()` sizes down to one lot.

        So `approve().lots <= lots_for()` always, and the gap is confined to a
        narrow range of stop widths. Where it bites, the backtester assumes the
        runner was available when live it would not have been - mildly
        optimistic, bounded, and asserted in tests/test_kite_and_chain.py.
        """
        ladder = self.lot_ladder()
        if underlying_stop_points <= 0:
            return ladder[-1]
        for lots in ladder:
            if self.required_max_delta(underlying_stop_points, lots) >= \
                    self.cfg.signal.min_delta:
                return lots
        return ladder[-1]

    def gate_failures(self, q: OptionQuote, underlying_stop_points: float,
                      lots: int = 1) -> list[str]:
        """
        Which gates a quote fails, and by how much.

        Exists for the daily brief: a chain table that only shows the winner
        teaches you nothing about why the other twenty-four strikes lost.
        """
        inst = self.cfg.instrument
        sig = self.cfg.signal
        ceiling = min(self.required_max_delta(underlying_stop_points, lots),
                      sig.max_delta)
        d = abs(q.delta)
        out: list[str] = []
        if d > ceiling:
            out.append(f"delta {d:.3f} > ceiling {ceiling:.3f}")
        elif d < sig.min_delta:
            out.append(f"delta {d:.3f} < floor {sig.min_delta:.2f}")
        if q.premium < inst.min_premium:
            out.append(f"premium {q.premium:.1f} < {inst.min_premium:.0f}")
        elif q.premium > inst.max_premium:
            out.append(f"premium {q.premium:.1f} > {inst.max_premium:.0f}")
        if q.spread_pct > inst.max_spread_pct_of_premium:
            out.append(f"spread {q.spread_pct:.2%} > "
                       f"{inst.max_spread_pct_of_premium:.2%}")
        if q.open_interest < inst.min_open_interest:
            out.append(f"OI {q.open_interest:,} < {inst.min_open_interest:,}")
        return out

    # ---------------- approval ----------------

    def approve(self, chain: list[OptionQuote], option_type: str,
                underlying_stop_points: float,
                free_capital: float) -> ApprovedOrder | RejectedOrder:
        """
        Size, strike, stop and target - or a reason why not.

        SIZING. The runner in `positions.py` banks one lot at +2R and trails
        the rest, and NSE requires order quantities in exact multiples of the
        lot size. A single 65-quantity NIFTY lot cannot be halved, so the
        runner needs at least two lots. This tries `preferred_lots` first and
        falls back, because two lots is not always reachable: risk per trade is
        fixed, so doubling the size HALVES the delta ceiling, and on a wide-stop
        day that ceiling drops under `min_delta` and no strike survives.

        Whichever size wins, `sizing_note` records why, and `runner_enabled`
        tells the caller whether a partial exit is even possible. A trade that
        silently became a plain 2:1 because only one lot fit is a trade you
        would otherwise misread all day.
        """
        t = self.cfg.trade
        lot = self.cfg.instrument.lot_size
        ladder = self.lot_ladder()

        quote = None
        lots = 0
        blocked_by_capital: Optional[str] = None

        for candidate_lots in ladder:
            q = self.select_strike(chain, underlying_stop_points, option_type,
                                   candidate_lots)
            if q is None:
                continue
            if q.premium * lot * candidate_lots > free_capital:
                blocked_by_capital = (
                    f"{candidate_lots} lot(s) of {q.strike}{option_type} at "
                    f"Rs {q.premium:.1f} costs "
                    f"Rs {q.premium * lot * candidate_lots:,.0f} > free capital "
                    f"Rs {free_capital:,.0f}"
                )
                continue
            quote, lots = q, candidate_lots
            break

        if quote is None:
            if blocked_by_capital:
                return RejectedOrder(
                    RejectReason.LOT_COST_EXCEEDS_FREE_CAPITAL, blocked_by_capital)
            ceiling = self.required_max_delta(underlying_stop_points, ladder[-1])
            return RejectedOrder(
                RejectReason.NO_VIABLE_STRIKE,
                f"needed delta <= {ceiling:.3f} at {ladder[-1]} lot(s) with premium "
                f"{self.cfg.instrument.min_premium}-{self.cfg.instrument.max_premium} "
                f"and spread <= {self.cfg.instrument.max_spread_pct_of_premium:.2%}"
            )

        quantity = lots * lot

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

        runner = t.enable_runner and lots > t.partial_exit_lots
        if runner:
            note = (f"{lots} lots ({quantity} qty): banks {t.partial_exit_lots} "
                    f"lot at +{t.partial_exit_at_r:.0f}R, "
                    f"{lots - t.partial_exit_lots} runs on a trail")
        elif t.enable_runner:
            note = (f"{lots} lot ({quantity} qty): RUNNER DISABLED - a partial "
                    f"exit needs more than {t.partial_exit_lots} lot and NSE "
                    f"will not fill part of one. Plain 2:1 target.")
        else:
            note = f"{lots} lot(s) ({quantity} qty): runner off in config"

        tick = self.cfg.instrument.tick_size
        return ApprovedOrder(
            quote=quote,
            lots=lots,
            quantity=quantity,
            entry_premium=quote.premium,
            premium_stop=round(premium_stop / tick) * tick,
            premium_target=round(premium_target / tick) * tick,
            underlying_stop_points=underlying_stop_points,
            rupee_risk=risk_rupees,
            rupee_reward=reward_rupees,
            runner_enabled=runner,
            sizing_note=note,
        )

    # ---------------- bookkeeping ----------------

    def register_entry(self, order: ApprovedOrder, direction: str) -> None:
        self.session.entries_taken += 1
        self.governor.register_entry()
        self.session.open_positions.append(OpenPosition(
            quote=order.quote,
            quantity=order.quantity,
            entry_premium=order.entry_premium,
            premium_stop=order.premium_stop,
            premium_target=order.premium_target,
            direction=direction,
        ))

    def register_exit(self, position: Optional[OpenPosition],
                      net_pnl: float) -> GovernorVerdict:
        """
        Book a realised P&L and re-run the day rules.

        Returns the governor's verdict because this is the moment that matters:
        a partial fill can lift the peak, tighten the floor, or reach the day
        target outright while the rest of the position is still open. Callers
        must act on a CLOSE_ALL here rather than waiting for the next entry
        attempt, which may never come.

        `position` may be None for a partial exit that does not close the
        tracked position.
        """
        self.session.realised_pnl += net_pnl
        self.capital += net_pnl
        if position is not None and position in self.session.open_positions:
            self.session.open_positions.remove(position)

        verdict = self.governor.register_exit(net_pnl)
        if verdict.reason is GovernorReason.SESSION_TARGET_HIT:
            self._halt(HaltReason.SESSION_TARGET_HIT)
        elif verdict.reason is GovernorReason.GIVE_BACK_FLOOR_HIT:
            self._halt(HaltReason.GIVE_BACK_FLOOR_HIT)
        return verdict

    def summary(self) -> dict:
        c = self.cfg.capital
        g = self.governor
        return {
            "capital": round(self.capital, 2),
            "session_pnl": round(self.session.realised_pnl, 2),
            "session_peak": round(g.peak_realised_pnl, 2),
            "day_target": round(g.target, 2),
            "day_floor": round(g.floor, 2),
            "day_floor_pct": f"{abs(g.floor_pct_of_capital):.2%}",
            "risk_remaining": round(g.risk_remaining, 2),
            "entries_taken": self.session.entries_taken,
            "max_entries": c.max_entries_per_session,
            "risk_per_trade": round(c.risk_per_trade_rupees, 2),
            "reward_per_trade": round(c.reward_per_trade_rupees, 2),
            "reward_risk_ratio": round(c.reward_risk_ratio, 2),
            "halted": self.session.halted,
            "halt_reason": self.session.halt_reason.value,
        }
