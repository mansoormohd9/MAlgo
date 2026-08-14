"""
Trade management: what happens after the entry.

The ladder you asked for:

    entry           stop at -1R, target at +2R
    +1R reached     stop to breakeven - the trade can no longer lose
    +2R reached     release one lot (banks ~Rs 3,333); the rest becomes a runner
    runner          stop trails 1 ATR behind the underlying, RATCHET ONLY
    stop touched    exit whatever remains
    15:10           force exit, unconditional

WHY THIS IS WRITTEN IN R AND NOTHING ELSE

`ExitLadder` below knows nothing about premiums, strikes, lots or rupees. It
takes an R-multiple and returns a decision. That is deliberate, and it is the
single most important structural choice in this module.

The live engine measures R from the option premium. The backtester measures R
from the underlying. Those are different numbers from different sources - but
because `RiskEngine.approve()` sizes the position so that
`stop_points x delta x quantity == risk_per_trade_rupees`, one R means the same
thing in both. Feeding both into ONE state machine is what keeps the promise in
risk.py's docstring: if backtest and live logic diverge, the backtest is
measuring a different system.

PESSIMISM. Inside a single bar you cannot know the path. `advance()` therefore
tests the stop against the bar's ADVERSE extreme, using the stop level as it
stood when the bar opened, BEFORE it considers any promotion from the bar's
favourable extreme. A bar that touches both the trailing stop and the next
promotion is scored as a stop. This mirrors the tie-break already used in
backtest.py and is the difference between a believable equity curve and a
flattering one.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional

from .config import Config, DEFAULT
from .risk import ApprovedOrder, OptionQuote


# R is computed by dividing prices, so a trigger sitting exactly on a rung
# lands a few ulps either side of it. Without a tolerance, a premium priced to
# be exactly +2R misses the partial by 1e-16 and the position runs on with no
# profit banked. The rungs are whole numbers of R; 1e-9 cannot blur them.
_EPS = 1e-9


class LadderMode(str, Enum):
    INITIAL = "initial"        # original stop, nothing triggered yet
    BREAKEVEN = "breakeven"    # +1R seen, stop at entry
    TRAIL = "trail"            # partial banked, remainder trailing


class ExitKind(str, Enum):
    STOP_TO_BREAKEVEN = "stop_to_breakeven"
    PARTIAL_EXIT = "partial_exit"
    TRAIL_UPDATE = "trail_update"
    STOPPED_OUT = "stopped_out"
    TARGET_EXIT = "target_exit"       # 2R reached with the runner disabled
    FORCE_EXIT = "force_exit"
    CLOSE_ALL = "close_all"           # a session governor ended the day


@dataclass
class LadderDecision:
    """What the ladder wants done, in R and lots. No prices, by design."""
    kind: Optional[ExitKind] = None
    exit_lots: int = 0
    exit_r: float = 0.0
    new_stop_r: Optional[float] = None
    detail: str = ""

    @property
    def is_exit(self) -> bool:
        return self.kind in (ExitKind.STOPPED_OUT, ExitKind.TARGET_EXIT,
                             ExitKind.PARTIAL_EXIT, ExitKind.FORCE_EXIT,
                             ExitKind.CLOSE_ALL)


@dataclass
class LadderState:
    lots_total: int
    lots_remaining: int
    stop_r: float = -1.0
    mode: LadderMode = LadderMode.INITIAL
    peak_r: float = 0.0
    partial_done: bool = False

    @property
    def closed(self) -> bool:
        return self.lots_remaining <= 0


class ExitLadder:
    """
    The rules, as a pure function of R. Shared by the live engine and the
    backtester so there is exactly one implementation to be wrong.
    """

    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg

    def new_state(self, lots: int) -> LadderState:
        return LadderState(lots_total=lots, lots_remaining=lots)

    def runner_enabled(self, lots: int) -> bool:
        """
        A runner needs at least two lots.

        NSE requires order quantities in exact multiples of the lot size, so
        you cannot sell part of one NIFTY lot - 65 is 65. With a single lot
        the 2R event is a full exit, not a partial, and that is a real
        behavioural difference the caller must be told about rather than
        discover from its fills.
        """
        return (self.cfg.trade.enable_runner
                and lots > self.cfg.trade.partial_exit_lots)

    def advance(self, st: LadderState, mark_r: float,
                best_r: Optional[float] = None,
                worst_r: Optional[float] = None,
                trail_distance_r: float = 0.0) -> list[LadderDecision]:
        """
        Advance one bar (or one live tick). Returns EVERY transition the bar
        triggered, in the order it triggered them.

        A list, not a single decision, because one bar can cross more than one
        rung. A gap or a fast 5-minute candle can carry a trade from 0R through
        breakeven and past +2R before you see another price. Emitting only the
        first transition would leave the partial un-banked until the next bar -
        by which time the move may already have given back.

        mark_r  where R stands now
        best_r  most favourable R touched this bar   (defaults to mark_r)
        worst_r most adverse R touched this bar      (defaults to mark_r)
        trail_distance_r  how far behind, in R, the trailing stop should sit;
                          this is `atr * trail_atr_multiple / stop_points`
        """
        if st.closed:
            return []
        best = mark_r if best_r is None else best_r
        worst = mark_r if worst_r is None else worst_r

        # --- the stop, at the level it held when the bar OPENED ---
        # Tested before any promotion, so a favourable extreme later in the
        # same bar cannot rescue a bar that had already stopped out.
        if worst <= st.stop_r + _EPS:
            lots = st.lots_remaining
            st.lots_remaining = 0
            where = ("breakeven" if abs(st.stop_r) < _EPS
                     else f"{st.stop_r:+.2f}R")
            return [LadderDecision(
                kind=ExitKind.STOPPED_OUT, exit_lots=lots, exit_r=st.stop_r,
                detail=f"stop hit at {where}",
            )]

        st.peak_r = max(st.peak_r, best)

        out: list[LadderDecision] = []
        # Bounded: there are only three rungs, so this can never spin.
        for _ in range(4):
            d = self._next_rung(st, best, trail_distance_r)
            if d is None:
                break
            out.append(d)
            if st.closed:
                break
        return out

    def _next_rung(self, st: LadderState, best: float,
                   trail_distance_r: float) -> Optional[LadderDecision]:
        """One transition, or None when the ladder has settled for this bar."""
        t = self.cfg.trade

        # --- breakeven shift at +1R ---
        if st.mode is LadderMode.INITIAL and best >= t.breakeven_at_r - _EPS:
            st.mode = LadderMode.BREAKEVEN
            st.stop_r = max(st.stop_r, 0.0)
            return LadderDecision(
                kind=ExitKind.STOP_TO_BREAKEVEN, new_stop_r=st.stop_r,
                detail=f"+{t.breakeven_at_r:.0f}R reached, stop to breakeven",
            )

        # --- partial / target at +2R ---
        if not st.partial_done and best >= t.partial_exit_at_r - _EPS:
            st.partial_done = True
            if self.runner_enabled(st.lots_total):
                st.lots_remaining -= t.partial_exit_lots
                st.mode = LadderMode.TRAIL
                st.stop_r = max(st.stop_r, 0.0)
                return LadderDecision(
                    kind=ExitKind.PARTIAL_EXIT,
                    exit_lots=t.partial_exit_lots,
                    exit_r=t.partial_exit_at_r,
                    new_stop_r=st.stop_r,
                    detail=(f"banked {t.partial_exit_lots} lot at "
                            f"+{t.partial_exit_at_r:.0f}R; "
                            f"{st.lots_remaining} lot(s) now trailing"),
                )
            lots = st.lots_remaining
            st.lots_remaining = 0
            return LadderDecision(
                kind=ExitKind.TARGET_EXIT, exit_lots=lots,
                exit_r=t.partial_exit_at_r,
                detail=(f"+{t.partial_exit_at_r:.0f}R target, full exit "
                        f"(single lot - cannot split a 65-qty contract)"),
            )

        # --- trail the runner ---
        if st.mode is LadderMode.TRAIL and trail_distance_r > 0:
            candidate = st.peak_r - trail_distance_r
            if candidate > st.stop_r + _EPS:
                st.stop_r = candidate       # ratchet only; never loosens
                return LadderDecision(
                    kind=ExitKind.TRAIL_UPDATE, new_stop_r=st.stop_r,
                    detail=(f"trail to {st.stop_r:+.2f}R "
                            f"({trail_distance_r:.2f}R behind peak)"),
                )

        return None

    def force_exit(self, st: LadderState, mark_r: float,
                   reason: str = "15:10 force exit") -> LadderDecision:
        if st.closed:
            return LadderDecision()
        lots = st.lots_remaining
        st.lots_remaining = 0
        return LadderDecision(kind=ExitKind.FORCE_EXIT, exit_lots=lots,
                              exit_r=mark_r, detail=reason)

    def close_all(self, st: LadderState, mark_r: float,
                  reason: str) -> LadderDecision:
        if st.closed:
            return LadderDecision()
        lots = st.lots_remaining
        st.lots_remaining = 0
        return LadderDecision(kind=ExitKind.CLOSE_ALL, exit_lots=lots,
                              exit_r=mark_r, detail=reason)


# ---------------------------------------------------------------- live side

@dataclass
class ManagedPosition:
    """A live position, plus everything needed to convert premium <-> R."""
    quote: OptionQuote
    strategy_key: str
    direction: str                 # "long" / "short" on the UNDERLYING
    entry_premium: float
    entry_underlying: float
    lot_size: int
    stop_points: float             # 1R, in underlying points
    risk_rupees: float             # 1R, in rupees, for the FULL position
    state: LadderState
    opened_at: Optional[datetime] = None
    tradingsymbol: str = ""
    realised_pnl: float = 0.0
    exits: list[dict] = field(default_factory=list)

    @property
    def premium_per_r(self) -> float:
        """
        Premium movement worth 1R.

        Equals `risk_rupees / original_quantity`, which is exactly how
        `RiskEngine.approve()` derived `premium_stop` - so the ladder's -1R and
        that stop are the same price, not two numbers that happen to be close.
        """
        qty = self.lot_size * self.state.lots_total
        return self.risk_rupees / qty if qty else 0.0

    @property
    def quantity_remaining(self) -> int:
        return self.lot_size * self.state.lots_remaining

    def r_of(self, premium: float) -> float:
        per_r = self.premium_per_r
        return (premium - self.entry_premium) / per_r if per_r else 0.0

    def premium_of(self, r: float) -> float:
        return self.entry_premium + r * self.premium_per_r

    @property
    def stop_premium(self) -> float:
        return round(self.premium_of(self.state.stop_r) /
                     0.05) * 0.05           # NSE tick

    def pnl_for(self, lots: int, exit_premium: float) -> float:
        """Gross premium P&L. Costs are applied by the caller via CostModel."""
        return (exit_premium - self.entry_premium) * self.lot_size * lots


@dataclass
class ExitAction:
    """What the manager decided, in prices a human or a broker can act on."""
    position: ManagedPosition
    kind: ExitKind
    lots: int
    premium: float
    quantity: int
    gross_pnl: float
    new_stop_premium: Optional[float]
    detail: str


class PositionManager:
    """
    Owns the open positions and drives the ladder for each of them.

    Called once per bar by the engine, BEFORE the strategy loop - managing an
    open position always outranks looking for a new one.
    """

    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg
        self.ladder = ExitLadder(cfg)
        self.positions: list[ManagedPosition] = []

    # ---------------- lifecycle ----------------

    def open(self, order: ApprovedOrder, direction: str, strategy_key: str,
             entry_underlying: float, opened_at: Optional[datetime] = None,
             tradingsymbol: str = "") -> ManagedPosition:
        pos = ManagedPosition(
            quote=order.quote,
            strategy_key=strategy_key,
            direction=direction,
            entry_premium=order.entry_premium,
            entry_underlying=entry_underlying,
            lot_size=order.quantity // max(order.lots, 1),
            stop_points=order.underlying_stop_points,
            risk_rupees=order.rupee_risk,
            state=self.ladder.new_state(order.lots),
            opened_at=opened_at,
            tradingsymbol=tradingsymbol,
        )
        self.positions.append(pos)
        return pos

    def clear(self) -> None:
        self.positions = []

    @property
    def open_lots(self) -> int:
        return sum(p.state.lots_remaining for p in self.positions)

    # ---------------- per-bar ----------------

    def update(self, premiums: dict[str, float], atr: float,
               now: Optional[time] = None) -> list[ExitAction]:
        """
        Advance every open position.

        `premiums` maps a position's tradingsymbol (or strike+type key) to its
        current premium. A position with no quote is SKIPPED rather than marked
        at a stale price - acting on a price you did not receive is how a
        trailing stop fires on a gap that never happened.
        """
        actions: list[ExitAction] = []
        for pos in list(self.positions):
            if pos.state.closed:
                continue
            premium = premiums.get(self._key(pos))
            if premium is None:
                continue

            if now is not None and now >= self.cfg.session.force_exit:
                d = self.ladder.force_exit(pos.state, pos.r_of(premium))
                if d.kind is not None:
                    actions.append(self._to_action(pos, d, premium))
                continue

            trail_r = (atr * self.cfg.trade.trail_atr_multiple / pos.stop_points
                       if pos.stop_points > 0 else 0.0)
            for d in self.ladder.advance(pos.state, mark_r=pos.r_of(premium),
                                         trail_distance_r=trail_r):
                actions.append(self._to_action(pos, d, premium))

        self.positions = [p for p in self.positions if not p.state.closed]
        return actions

    def close_all(self, premiums: dict[str, float], reason: str) -> list[ExitAction]:
        """Flatten everything - a session governor ended the day."""
        actions: list[ExitAction] = []
        for pos in list(self.positions):
            premium = premiums.get(self._key(pos), pos.entry_premium)
            d = self.ladder.close_all(pos.state, pos.r_of(premium), reason)
            if d.kind is not None:
                actions.append(self._to_action(pos, d, premium))
        self.positions = [p for p in self.positions if not p.state.closed]
        return actions

    # ---------------- helpers ----------------

    @staticmethod
    def _key(pos: ManagedPosition) -> str:
        return pos.tradingsymbol or f"{pos.quote.strike}{pos.quote.option_type}"

    def _to_action(self, pos: ManagedPosition, d: LadderDecision,
                   premium: float) -> ExitAction:
        # An exit fills at the stop LEVEL, not at the mark, whenever the stop
        # is what triggered it - otherwise a bar that blew through the stop
        # would be scored at its worst price and flatter nothing.
        fill = premium
        if d.kind in (ExitKind.STOPPED_OUT, ExitKind.PARTIAL_EXIT,
                      ExitKind.TARGET_EXIT):
            fill = pos.premium_of(d.exit_r)

        gross = pos.pnl_for(d.exit_lots, fill) if d.exit_lots else 0.0
        if d.exit_lots:
            pos.realised_pnl += gross
            pos.exits.append({"kind": d.kind.value, "lots": d.exit_lots,
                              "premium": round(fill, 2), "gross": round(gross, 2)})

        return ExitAction(
            position=pos,
            kind=d.kind,
            lots=d.exit_lots,
            premium=round(fill, 2),
            quantity=pos.lot_size * d.exit_lots,
            gross_pnl=gross,
            new_stop_premium=(pos.premium_of(d.new_stop_r)
                              if d.new_stop_r is not None else None),
            detail=d.detail,
        )
