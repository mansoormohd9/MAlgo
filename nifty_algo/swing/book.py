"""
What you actually did, as opposed to what the scanner said.

[tracker.py](tracker.py) answers "was the ranking any good?" by replaying every
journalled pick against later bars, including the ones you skipped. That is a
question about the SCANNER and it stays useful. This module answers a different
one - "where is my money, and what is protecting it?" - and no amount of replay
can produce that, because a replay cannot know you skipped two of the three
picks, were filled 0.4% worse than the ticket said, and sold half by hand.

THE LEDGER IS EVENT-SOURCED THROUGH THE EXISTING JOURNAL. No new mutable file.
Every state change appends one line; `Book.load()` replays them. That is the
same append-only discipline the rest of the repo runs on, and for the same
reason: a position file you can rewrite is one that will eventually be
rewritten to agree with your memory of a trade.

THE BROKER IS THE SOURCE OF TRUTH, ALWAYS. `reconcile()` compares the ledger
against Zerodha's holdings, GTTs and orders, and every disagreement is
SURFACED, never silently resolved. A book that quietly "fixes" itself to match
is a book that will one day quietly fix itself to match the wrong thing. The
two failure modes that matter most - a position with no live stop, and a stop
that cannot execute because today's TPIN authorisation is missing - are
reported as CRITICAL and are meant to be impossible to scroll past.

ONE OCO PER POSITION, AND WHY THE +2R PARTIAL IS NOT AT THE BROKER. Zerodha's
two-leg GTT sells one quantity at one of two triggers; it cannot express "bank
half here, run the rest to there". So the resting OCO carries the STOP and the
structural TARGET for the whole remaining position - the two legs that must
survive the laptop being shut - and the +2R partial is taken by the app on its
daily run. That split follows the risk: the stop prevents a loss and must never
depend on the app being open; the partial banks a profit and can wait a day.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from typing import Optional

from ..config import Config, DEFAULT
from ..positions import ExitLadder, LadderMode, LadderState

# ---------------------------------------------------------------- events

EVENT_ARMED = "swing_armed"
EVENT_REARMED = "swing_rearmed"
EVENT_FILLED = "swing_filled"
EVENT_EXIT_ARMED = "swing_exit_armed"
EVENT_LADDER = "swing_ladder"
EVENT_EXIT = "swing_exit"
EVENT_CLOSED = "swing_closed"
EVENT_CANCELLED = "swing_cancelled"
EVENT_EXIT_FAILED = "swing_exit_failed"

#: Every event this module writes, for the Journal page's help table and for
#: anyone grepping a day's file.
EVENTS = (EVENT_ARMED, EVENT_REARMED, EVENT_FILLED, EVENT_EXIT_ARMED,
          EVENT_LADDER, EVENT_EXIT, EVENT_CLOSED, EVENT_CANCELLED,
          EVENT_EXIT_FAILED)


class TicketState(str, Enum):
    ARMED = "armed"          # buy GTT resting at Zerodha, not yet triggered
    OPEN = "open"            # filled; exit OCO should be resting
    CLOSED = "closed"
    CANCELLED = "cancelled"  # expired, dropped out of the scan, or you cancelled

    @property
    def is_live(self) -> bool:
        return self in (TicketState.ARMED, TicketState.OPEN)


# ---------------------------------------------------------------- the ticket

@dataclass
class SwingTicket:
    """One thing you committed money to, and everything managing it."""
    key: str
    market: str
    symbol: str
    name: str = ""
    sector: str = ""
    setup: str = ""

    # --- the plan, as the scan stated it ---
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    quantity: float = 0.0
    risk_inr: float = 0.0
    reward_inr: float = 0.0
    deployed_inr: float = 0.0
    scanned_on: Optional[date] = None
    valid_until: Optional[date] = None

    # --- what happened ---
    state: TicketState = TicketState.ARMED
    buy_gtt_id: Optional[str] = None
    exit_gtt_id: Optional[str] = None
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    opened_on: Optional[date] = None
    closed_on: Optional[date] = None
    realised_inr: float = 0.0
    exits: list[dict] = field(default_factory=list)
    ladder: Optional[LadderState] = None
    #: The last session whose bar was fed to the ladder. A daily bar must be
    #: consumed EXACTLY once: re-feeding it re-tests its low against a stop
    #: the first pass had already ratcheted upward, which stops the position
    #: out on a move that never happened - and, worse, sells it for real.
    laddered_on: Optional[date] = None
    note: str = ""

    # ---------- the R conversion, mirroring ManagedPosition ----------

    @property
    def entry_reference(self) -> float:
        """
        The price 0R is measured from.

        The ACTUAL fill once there is one, not the ticket's entry. A fill 0.4%
        above the trigger is 0.4% of risk you really took; measuring R off the
        price you hoped for would report a stop-out as -0.92R and quietly
        flatter every statistic downstream of it.
        """
        return self.avg_fill_price or self.entry

    @property
    def risk_points(self) -> float:
        """1R in rupees per share. Derived from the real entry, per above."""
        return max(self.entry_reference - self.stop, 0.0)

    @property
    def is_degenerate(self) -> bool:
        """
        Filled at or below the stop, so there is no 1R to measure in.

        A gap straight through the trigger can do this. Everything downstream
        then quietly breaks in the worst possible way: `r_of()` returns 0.0
        for every price, so the ladder never promotes and never stops, and
        `stop_price` collapses onto the fill. The position would sit there
        with a stop 30% below the market and a ladder that can never fire.
        It has to be surfaced and closed, not computed around.
        """
        return (self.state is TicketState.OPEN
                and self.filled_qty > 0 and self.risk_points <= 0)

    def r_of(self, price: float) -> float:
        rp = self.risk_points
        return (price - self.entry_reference) / rp if rp > 0 else 0.0

    def price_of(self, r: float) -> float:
        return self.entry_reference + r * self.risk_points

    @property
    def stop_price(self) -> float:
        """Where the stop sits NOW - the number that goes into the GTT."""
        if self.ladder is None:
            return self.stop
        return round(self.price_of(self.ladder.stop_r), 2)

    @property
    def open_quantity(self) -> float:
        if self.ladder is None:
            return self.filled_qty or self.quantity
        return float(self.ladder.lots_remaining)

    @property
    def open_risk_r(self) -> float:
        """
        R still at risk on this ticket, floored at zero.

        Once the stop is at or above breakeven the position can no longer lose,
        so it contributes nothing to portfolio heat - which is the whole point
        of moving it there.
        """
        if self.ladder is None or self.state is not TicketState.OPEN:
            return 0.0
        per_unit = max(-self.ladder.stop_r, 0.0)
        total = float(self.ladder.lots_total) or 1.0
        return per_unit * (self.ladder.lots_remaining / total)

    @property
    def realised_r(self) -> float:
        """
        Realised P&L in R. 1R is the rupee risk this ticket actually took on,
        which is `risk_points * filled_qty` - measured off the fill, not off
        the ticket's hoped-for entry.
        """
        denominator = self.risk_points * self.filled_qty
        return self.realised_inr / denominator if denominator > 0 else 0.0

    def unrealised_inr(self, last_price: float) -> float:
        if self.state is not TicketState.OPEN or not last_price:
            return 0.0
        return (last_price - self.entry_reference) * self.open_quantity

    @property
    def rung(self) -> str:
        """Which step of the ladder this position has reached, in English."""
        if self.ladder is None:
            return "-"
        if self.ladder.mode is LadderMode.TRAIL:
            return "trailing"
        if self.ladder.mode is LadderMode.BREAKEVEN:
            return "breakeven"
        return "initial"

    def days_held(self, today: Optional[date] = None) -> Optional[int]:
        if self.opened_on is None:
            return None
        return ((today or date.today()) - self.opened_on).days

    def to_record(self) -> dict:
        return {
            "key": self.key, "market": self.market, "symbol": self.symbol,
            "name": self.name, "sector": self.sector, "setup": self.setup,
            "entry": round(self.entry, 2), "stop": round(self.stop, 2),
            "target": round(self.target, 2), "quantity": self.quantity,
            "risk_inr": round(self.risk_inr, 2),
            "reward_inr": round(self.reward_inr, 2),
            "deployed_inr": round(self.deployed_inr, 2),
            "scanned_on": _iso(self.scanned_on),
            "valid_until": _iso(self.valid_until),
        }


# ---------------------------------------------------------------- findings

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

#: Finding kinds, named so the UI and the tests agree on the strings.
F_UNPROTECTED = "unprotected"
F_NO_AUTHORISATION = "no_authorisation"
F_GTT_VANISHED = "gtt_vanished"
F_FILLED = "filled"
F_STOP_DRIFTED = "stop_drifted"
F_QTY_MISMATCH = "quantity_mismatch"
F_UNTRACKED_HOLDING = "untracked_holding"
F_ORPHAN_GTT = "orphan_gtt"
F_EXPIRED = "expired"
F_CLOSED_ELSEWHERE = "closed_elsewhere"


@dataclass
class Finding:
    kind: str
    severity: str
    symbol: str
    message: str
    action: str = ""


@dataclass
class ReconcileReport:
    findings: list[Finding] = field(default_factory=list)
    checked: int = 0
    broker_reachable: bool = True

    def add(self, kind, severity, symbol, message, action="") -> None:
        self.findings.append(Finding(kind, severity, symbol, message, action))

    def of(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def critical(self) -> list[Finding]:
        return self.of(SEVERITY_CRITICAL)

    @property
    def clean(self) -> bool:
        return not self.findings

    def headline(self) -> str:
        if not self.broker_reachable:
            return ("The broker could not be reached, so nothing was verified. "
                    "This is NOT a clean bill of health.")
        if self.clean:
            return f"{self.checked} ticket(s) agree with Zerodha."
        crit = len(self.critical)
        rest = len(self.findings) - crit
        if crit:
            return f"{crit} CRITICAL finding(s) and {rest} other(s)."
        return f"{len(self.findings)} finding(s), none critical."


# ---------------------------------------------------------------- the book

class Book:
    """
    The open ledger. Built by replay; mutated only by writing an event.
    """

    def __init__(self, cfg: Config = DEFAULT, journal=None):
        self.cfg = cfg
        self.journal = journal
        self.tickets: dict[str, SwingTicket] = {}
        self.ladder = ExitLadder(cfg, trade=swing_trade(cfg))

    # ---------------- replay ----------------

    @classmethod
    def load(cls, journal, cfg: Config = DEFAULT,
             market: Optional[str] = None) -> "Book":
        book = cls(cfg, journal)
        for record in journal.read_all():
            book._apply(record, market)
        return book

    def _apply(self, record: dict, market: Optional[str]) -> None:
        event = record.get("event")
        if event not in EVENTS:
            return
        key = record.get("key")
        if not key:
            return

        if event == EVENT_ARMED:
            if market is not None and record.get("market") != market:
                return
            self.tickets[key] = _ticket_from(record)
            return

        ticket = self.tickets.get(key)
        if ticket is None:
            # An event for a ticket we filtered out by market, or a journal
            # truncated by retention. Silently ignoring it is right; there is
            # nothing to update and nothing has been lost.
            return

        if event == EVENT_REARMED:
            ticket.entry = _f(record.get("entry"), ticket.entry)
            ticket.stop = _f(record.get("stop"), ticket.stop)
            ticket.target = _f(record.get("target"), ticket.target)
            ticket.quantity = _f(record.get("quantity"), ticket.quantity)
            ticket.valid_until = _date(record.get("valid_until")) or ticket.valid_until
            if record.get("buy_gtt_id"):
                ticket.buy_gtt_id = str(record["buy_gtt_id"])
        elif event == EVENT_FILLED:
            ticket.state = TicketState.OPEN
            ticket.filled_qty = _f(record.get("filled_qty"), ticket.quantity)
            ticket.avg_fill_price = _f(record.get("avg_fill_price"), ticket.entry)
            ticket.opened_on = _date(record.get("opened_on"))
            ticket.ladder = _ladder_from(record.get("ladder"))
        elif event == EVENT_EXIT_ARMED:
            ticket.exit_gtt_id = _opt_str(record.get("exit_gtt_id"))
        elif event == EVENT_LADDER:
            ticket.ladder = _ladder_from(record.get("ladder")) or ticket.ladder
            ticket.laddered_on = _date(record.get("on")) or ticket.laddered_on
        elif event == EVENT_EXIT:
            ticket.realised_inr = _f(record.get("realised_inr"), ticket.realised_inr)
            ticket.exits.append({
                "kind": record.get("kind"), "quantity": record.get("quantity"),
                "price": record.get("price"), "r": record.get("r"),
            })
            ticket.ladder = _ladder_from(record.get("ladder")) or ticket.ladder
            ticket.laddered_on = _date(record.get("on")) or ticket.laddered_on
        elif event == EVENT_CLOSED:
            ticket.state = TicketState.CLOSED
            ticket.closed_on = _date(record.get("closed_on"))
            ticket.realised_inr = _f(record.get("realised_inr"), ticket.realised_inr)
            ticket.note = str(record.get("reason") or ticket.note)
        elif event == EVENT_EXIT_FAILED:
            ticket.state = TicketState.OPEN
            ticket.closed_on = None
            ticket.ladder = _ladder_from(record.get("ladder")) or ticket.ladder
            ticket.note = str(record.get("reason") or ticket.note)
        elif event == EVENT_CANCELLED:
            ticket.state = TicketState.CANCELLED
            ticket.closed_on = _date(record.get("closed_on"))
            ticket.note = str(record.get("reason") or ticket.note)

    # ---------------- views ----------------

    def by_state(self, *states: TicketState) -> list[SwingTicket]:
        return [t for t in self.tickets.values() if t.state in states]

    @property
    def open(self) -> list[SwingTicket]:
        return self.by_state(TicketState.OPEN)

    @property
    def armed(self) -> list[SwingTicket]:
        return self.by_state(TicketState.ARMED)

    @property
    def closed(self) -> list[SwingTicket]:
        return self.by_state(TicketState.CLOSED)

    def for_symbol(self, symbol: str) -> Optional[SwingTicket]:
        """The one LIVE ticket in a name, if any. Duplicates are refused."""
        sym = symbol.upper()
        for t in self.tickets.values():
            if t.symbol.upper() == sym and t.state.is_live:
                return t
        return None

    @property
    def open_risk_r(self) -> float:
        """
        Total R at risk across everything open - portfolio heat.

        `top_n` caps how many tickets ONE scan proposes. Nothing capped how
        many could be open at once, so three scans on three days could put
        nine positions on and risk 9R against a day stop built for 3.
        """
        return sum(t.open_risk_r for t in self.open)

    @property
    def committed_inr(self) -> float:
        """Cash an armed buy would consume if every trigger fired today."""
        return sum(t.entry * t.quantity for t in self.armed)

    # ---------------- mutation ----------------

    def _write(self, event: str, payload: dict, day: Optional[date] = None) -> None:
        if self.journal is not None:
            self.journal.write(event, payload, day=day)

    def arm(self, pick, buy_gtt_id: str, today: Optional[date] = None) -> SwingTicket:
        """Record that a buy GTT is now resting at Zerodha for this pick."""
        today = today or date.today()
        ticket = _ticket_from_pick(pick, today)
        ticket.buy_gtt_id = str(buy_gtt_id)
        self.tickets[ticket.key] = ticket
        self._write(EVENT_ARMED,
                    {**ticket.to_record(), "buy_gtt_id": ticket.buy_gtt_id},
                    day=today)
        return ticket

    def rearm(self, ticket: SwingTicket, pick, buy_gtt_id: Optional[str] = None,
              today: Optional[date] = None) -> None:
        """Re-price an unfilled entry onto a fresh scan's numbers."""
        today = today or date.today()
        ticket.entry = float(pick.setup.entry)
        ticket.stop = float(pick.setup.stop)
        ticket.target = float(pick.setup.target)
        ticket.quantity = float(pick.quantity)
        ticket.valid_until = pick.valid_until
        if buy_gtt_id:
            ticket.buy_gtt_id = str(buy_gtt_id)
        self._write(EVENT_REARMED, {
            "key": ticket.key, "symbol": ticket.symbol,
            "entry": round(ticket.entry, 2), "stop": round(ticket.stop, 2),
            "target": round(ticket.target, 2), "quantity": ticket.quantity,
            "valid_until": _iso(ticket.valid_until),
            "buy_gtt_id": ticket.buy_gtt_id,
        }, day=today)

    def record_fill(self, ticket: SwingTicket, quantity: float, price: float,
                    on: Optional[date] = None) -> None:
        """
        The buy GTT triggered and filled. Start managing.

        The ladder is created here rather than at arm time because it is
        denominated in R, and R is not defined until there is a fill price to
        measure it from.
        """
        on = on or date.today()
        ticket.state = TicketState.OPEN
        ticket.filled_qty = float(quantity)
        ticket.avg_fill_price = float(price)
        ticket.opened_on = on
        ticket.ladder = self.ladder.new_state(int(quantity))
        self._write(EVENT_FILLED, {
            "key": ticket.key, "symbol": ticket.symbol,
            "filled_qty": ticket.filled_qty,
            "avg_fill_price": round(ticket.avg_fill_price, 2),
            "planned_entry": round(ticket.entry, 2),
            # Slippage, recorded on every single fill. After twenty trades
            # this is a measured number rather than a modelled one - which is
            # the Phase 4 discipline the option book's roadmap asks for.
            "slippage_pct": round(
                (ticket.avg_fill_price - ticket.entry) / ticket.entry * 100.0, 4
            ) if ticket.entry else 0.0,
            "opened_on": _iso(on),
            "ladder": _ladder_dict(ticket.ladder),
        }, day=on)

    def record_exit_gtt(self, ticket: SwingTicket, exit_gtt_id: Optional[str],
                        today: Optional[date] = None) -> None:
        ticket.exit_gtt_id = _opt_str(exit_gtt_id)
        self._write(EVENT_EXIT_ARMED, {
            "key": ticket.key, "symbol": ticket.symbol,
            "exit_gtt_id": ticket.exit_gtt_id,
            "stop": ticket.stop_price, "target": round(ticket.target, 2),
            "quantity": ticket.open_quantity,
        }, day=today or date.today())

    def advance(self, ticket: SwingTicket, high: float, low: float,
                close: float, atr: float,
                today: Optional[date] = None) -> list:
        """
        Run one daily bar through the ladder. Returns the decisions it made.

        Bar extremes are passed, so the ladder tests the stop against the
        adverse extreme at its PRE-BAR level before considering any promotion
        from the favourable one. A day that touched both the trailing stop and
        the next rung is scored as a stop. Same pessimism as everywhere else
        in this repo, and the reason the outcome is believable.
        """
        if ticket.ladder is None or ticket.state is not TicketState.OPEN:
            return []
        today = today or date.today()
        # Keyed on the BAR's date, which is what `today` carries here - not
        # on the date the run happened. On a weekend, a holiday, or against an
        # unrefreshed cache, the newest available bar is Friday's; running
        # three times over a weekend would otherwise re-test Friday's low
        # against the stop Friday itself had already ratcheted, and sell on a
        # move that never happened.
        if ticket.laddered_on is not None and today <= ticket.laddered_on:
            return []           # this bar has already been consumed
        ticket.laddered_on = today
        rp = ticket.risk_points
        trail_r = (atr * self.cfg.swing.trail_atr_multiple / rp) if rp > 0 else 0.0

        decisions = self.ladder.advance(
            ticket.ladder,
            mark_r=ticket.r_of(close),
            best_r=ticket.r_of(high),
            worst_r=ticket.r_of(low),
            trail_distance_r=trail_r,
        )
        for d in decisions:
            self._book_decision(ticket, d, today)
        return decisions

    def _book_decision(self, ticket: SwingTicket, d, today: date) -> None:
        if d.exit_lots:
            fill = ticket.price_of(d.exit_r)
            gross = (fill - ticket.entry_reference) * d.exit_lots
            ticket.realised_inr += gross
            ticket.exits.append({"kind": d.kind.value, "quantity": d.exit_lots,
                                 "price": round(fill, 2), "r": round(d.exit_r, 3)})
            self._write(EVENT_EXIT, {
                "key": ticket.key, "symbol": ticket.symbol,
                "kind": d.kind.value, "quantity": d.exit_lots,
                "price": round(fill, 2), "r": round(d.exit_r, 3),
                "gross_inr": round(gross, 2),
                "realised_inr": round(ticket.realised_inr, 2),
                "detail": d.detail,
                "on": _iso(today),
                "ladder": _ladder_dict(ticket.ladder),
            }, day=today)
            if ticket.ladder.closed:
                self.close(ticket, d.detail, today)
        else:
            self._write(EVENT_LADDER, {
                "key": ticket.key, "symbol": ticket.symbol,
                "kind": d.kind.value if d.kind else "",
                "stop_price": ticket.stop_price,
                "detail": d.detail,
                "on": _iso(today),
                "ladder": _ladder_dict(ticket.ladder),
            }, day=today)

    def snapshot(self, ticket: SwingTicket) -> dict:
        """Enough of a ticket to put it back if the broker refuses an exit."""
        return {
            "ladder": _ladder_dict(ticket.ladder),
            "state": ticket.state.value,
            "realised_inr": ticket.realised_inr,
            "exits": len(ticket.exits),
            "laddered_on": _iso(ticket.laddered_on),
        }

    def revert_exit(self, ticket: SwingTicket, snap: dict, reason: str,
                    on: Optional[date] = None) -> None:
        """
        Put a ticket back the way it was, because the sell did not happen.

        `book.advance()` records what the RUNGS say; the broker is what makes
        it true. When a sell is refused - which without DDPI is the ordinary
        case, not the exotic one - a ledger showing a closed position while
        the shares are still in the account is worse than useless: it is the
        one state in which nobody goes looking for a stop.

        A compensating event rather than an edit, because the journal is
        append-only and the failed attempt is itself worth recording.
        """
        on = on or date.today()
        ticket.ladder = _ladder_from(snap.get("ladder")) or ticket.ladder
        ticket.state = TicketState(snap.get("state", TicketState.OPEN.value))
        ticket.realised_inr = _f(snap.get("realised_inr"))
        del ticket.exits[int(snap.get("exits", 0)):]
        ticket.closed_on = None
        self._write(EVENT_EXIT_FAILED, {
            "key": ticket.key, "symbol": ticket.symbol,
            "reason": reason, "on": _iso(on),
            "ladder": _ladder_dict(ticket.ladder),
        }, day=on)

    def close(self, ticket: SwingTicket, reason: str,
              on: Optional[date] = None) -> None:
        on = on or date.today()
        ticket.state = TicketState.CLOSED
        ticket.closed_on = on
        ticket.note = reason
        self._write(EVENT_CLOSED, {
            "key": ticket.key, "symbol": ticket.symbol,
            "closed_on": _iso(on), "reason": reason,
            "realised_inr": round(ticket.realised_inr, 2),
            "r_multiple": round(ticket.realised_r, 3),
        }, day=on)

    def cancel(self, ticket: SwingTicket, reason: str,
               on: Optional[date] = None) -> None:
        on = on or date.today()
        ticket.state = TicketState.CANCELLED
        ticket.closed_on = on
        ticket.note = reason
        self._write(EVENT_CANCELLED, {
            "key": ticket.key, "symbol": ticket.symbol,
            "closed_on": _iso(on), "reason": reason,
        }, day=on)


# --------------------------------------------------------------- performance

@dataclass
class Performance:
    """Realised results of trades you actually took. Requirement 4's answer."""
    trades: int = 0
    wins: int = 0
    total_r: float = 0.0
    realised_inr: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0
    open_positions: int = 0
    open_risk_r: float = 0.0
    unrealised_inr: float = 0.0
    avg_slippage_pct: Optional[float] = None

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def expectancy_r(self) -> float:
        return self.total_r / self.trades if self.trades else 0.0

    def headline(self, breakeven_win_rate: float = 1 / 3) -> str:
        if not self.trades:
            return (f"{self.open_positions} open, nothing closed yet - too "
                    f"early to say whether this is working.")
        verdict = ("above" if self.win_rate > breakeven_win_rate else "below")
        return (f"{self.trades} closed · {self.wins} won ({self.win_rate:.0%}, "
                f"{verdict} the {breakeven_win_rate:.0%} breakeven) · "
                f"expectancy {self.expectancy_r:+.2f}R · "
                f"net ₹{self.realised_inr:+,.0f}")


def performance(book: Book, last_prices: Optional[dict] = None) -> Performance:
    last_prices = last_prices or {}
    p = Performance()
    rs = []
    for t in book.closed:
        r = t.realised_r
        rs.append(r)
        p.trades += 1
        p.wins += 1 if r > 0 else 0
        p.total_r += r
        p.realised_inr += t.realised_inr
    if rs:
        p.best_r, p.worst_r = max(rs), min(rs)
    for t in book.open:
        p.open_positions += 1
        p.realised_inr += t.realised_inr        # a banked partial is realised
        p.unrealised_inr += t.unrealised_inr(last_prices.get(t.symbol, 0.0))
    p.open_risk_r = book.open_risk_r
    return p


# --------------------------------------------------------------- reconcile

def reconcile(book: Book, broker, today: Optional[date] = None,
              broker_reachable: bool = True, holdings_in=None,
              gtts_in=None) -> ReconcileReport:
    """
    Compare the ledger against Zerodha. Report every disagreement.

    Nothing here mutates the book. Deciding what to DO about a finding is the
    caller's job - and, for anything that costs money, yours. This function's
    only responsibility is that no disagreement goes unmentioned.
    """
    today = today or date.today()
    report = ReconcileReport(broker_reachable=broker_reachable)

    # Reuse what the caller already fetched where it can. Re-reading would
    # double the API calls AND give a second chance to fail - at which point
    # every open position reads as "sold outside this app" and every stop as
    # vanished, a page of alarming and entirely wrong findings.
    if not broker_reachable:
        holdings, gtts = {}, []
    else:
        holdings = {h.symbol: h for h in
                    (broker.holdings() if holdings_in is None else holdings_in)}
        gtts = broker.gtts() if gtts_in is None else list(gtts_in)
    by_id = {g.trigger_id: g for g in gtts}
    active_exit_for = {g.symbol: g for g in gtts if g.is_active and g.is_exit}

    live = book.by_state(TicketState.ARMED, TicketState.OPEN)
    report.checked = len(live)

    # --- the authorisation gate, once, before anything else ---
    state, why = broker.protection_state(today)
    if live and state != "protected":
        severity = (SEVERITY_CRITICAL if state == "unprotected"
                    else SEVERITY_WARNING)
        report.add(F_NO_AUTHORISATION, severity, "",
                   why,
                   action=("Authorise holdings in Kite now, or enable DDPI "
                           "once and never do this again."))

    if not broker_reachable:
        return report

    for t in live:
        held = holdings.get(t.symbol.upper())
        owned = held.total_quantity if held else 0.0

        if t.state is TicketState.ARMED:
            if owned > 0:
                report.add(
                    F_FILLED, SEVERITY_INFO, t.symbol,
                    f"{t.symbol} is in your holdings ({owned:g} shares at "
                    f"₹{held.average_price:,.2f}) but the book still says "
                    f"armed - the buy GTT triggered since the last run.",
                    action="Record the fill so the exit can be armed.")
            elif t.buy_gtt_id and t.buy_gtt_id not in by_id:
                report.add(
                    F_GTT_VANISHED, SEVERITY_WARNING, t.symbol,
                    f"The buy trigger for {t.symbol} is no longer at Zerodha "
                    f"and nothing was filled. A GTT is cancelled rather than "
                    f"retried when its limit does not fill, and corporate "
                    f"actions delete them outright.",
                    action="Re-arm it, or cancel the ticket.")
            elif t.valid_until and today > t.valid_until:
                report.add(
                    F_EXPIRED, SEVERITY_INFO, t.symbol,
                    f"{t.symbol} never reached ₹{t.entry:,.2f} inside its "
                    f"{t.valid_until:%d %b} window. The chart it was read off "
                    f"is now stale.",
                    action="Let the next scan re-price it, or cancel.")
            continue

        # --- OPEN ---
        if owned <= 0:
            report.add(
                F_CLOSED_ELSEWHERE, SEVERITY_WARNING, t.symbol,
                f"The book holds {t.open_quantity:g} {t.symbol} but Zerodha "
                f"reports none. It was sold outside this app, or the OCO "
                f"fired since the last run.",
                action="Close the ticket with the real exit price.")
            continue

        if abs(owned - t.open_quantity) > 1e-6:
            report.add(
                F_QTY_MISMATCH, SEVERITY_WARNING, t.symbol,
                f"{t.symbol}: the book says {t.open_quantity:g} shares, "
                f"Zerodha says {owned:g}. A partial fill or a manual part-sale.",
                action="Reconcile the quantity before the stop is moved again.")

        resting = by_id.get(t.exit_gtt_id or "") or active_exit_for.get(t.symbol.upper())
        if resting is None or not resting.is_active:
            report.add(
                F_UNPROTECTED, SEVERITY_CRITICAL, t.symbol,
                f"{t.symbol} is OPEN with no active exit trigger at Zerodha. "
                f"There is nothing between this position and a gap down.",
                action="Arm the exit OCO now.")
            continue

        want_stop = t.stop_price
        have_stop = resting.stop
        if have_stop is not None and abs(have_stop - want_stop) > 0.05:
            direction = "below" if have_stop < want_stop else "above"
            report.add(
                F_STOP_DRIFTED, SEVERITY_WARNING, t.symbol,
                f"{t.symbol}: the resting stop is ₹{have_stop:,.2f}, "
                f"{direction} where the ladder says it should be "
                f"(₹{want_stop:,.2f}).",
                action="Modify the OCO to match the ladder.")

    # --- things at the broker that the book knows nothing about ---
    tracked = {t.symbol.upper() for t in live}
    for symbol, held in holdings.items():
        if symbol not in tracked and held.total_quantity > 0:
            report.add(
                F_UNTRACKED_HOLDING, SEVERITY_INFO, symbol,
                f"You hold {held.total_quantity:g} {symbol} that this book did "
                f"not put on. It is not being managed here and has no stop "
                f"from this app.",
                action="Ignore it, or adopt it into the book.")

    known_ids = {t.exit_gtt_id for t in live} | {t.buy_gtt_id for t in live}
    for g in gtts:
        if g.is_active and g.trigger_id not in known_ids and g.symbol in tracked:
            report.add(
                F_ORPHAN_GTT, SEVERITY_WARNING, g.symbol,
                f"An active trigger on {g.symbol} at Zerodha is not the one "
                f"this book placed. Two triggers on one holding can sell it "
                f"twice.",
                action="Delete whichever one you did not mean to keep.")

    return report


# --------------------------------------------------------------- helpers

def swing_trade(cfg: Config):
    """
    The swing book's trade-management settings, from `SwingConfig`.

    Built by `replace()` off the intraday dataclass rather than declared as a
    second one, so a rung added to the ladder cannot exist for one book and
    not the other.
    """
    return replace(
        cfg.trade,
        trail_atr_multiple=cfg.swing.trail_atr_multiple,
        partial_exit_fraction=cfg.swing.partial_exit_fraction,
    )


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _opt_str(value) -> Optional[str]:
    return str(value) if value else None


def _iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def _date(value) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _ladder_dict(state: Optional[LadderState]) -> Optional[dict]:
    if state is None:
        return None
    return {
        "lots_total": state.lots_total,
        "lots_remaining": state.lots_remaining,
        "stop_r": round(state.stop_r, 6),
        "mode": state.mode.value,
        "peak_r": round(state.peak_r, 6),
        "partial_done": state.partial_done,
    }


def _ladder_from(raw) -> Optional[LadderState]:
    if not isinstance(raw, dict):
        return None
    return LadderState(
        lots_total=int(raw.get("lots_total", 0)),
        lots_remaining=int(raw.get("lots_remaining", 0)),
        stop_r=_f(raw.get("stop_r"), -1.0),
        mode=LadderMode(raw.get("mode", LadderMode.INITIAL.value)),
        peak_r=_f(raw.get("peak_r")),
        partial_done=bool(raw.get("partial_done")),
    )


def _ticket_from(record: dict) -> SwingTicket:
    return SwingTicket(
        key=str(record["key"]),
        market=str(record.get("market") or "india"),
        symbol=str(record.get("symbol") or "").upper(),
        name=str(record.get("name") or ""),
        sector=str(record.get("sector") or ""),
        setup=str(record.get("setup") or ""),
        entry=_f(record.get("entry")), stop=_f(record.get("stop")),
        target=_f(record.get("target")), quantity=_f(record.get("quantity")),
        risk_inr=_f(record.get("risk_inr")),
        reward_inr=_f(record.get("reward_inr")),
        deployed_inr=_f(record.get("deployed_inr")),
        scanned_on=_date(record.get("scanned_on")),
        valid_until=_date(record.get("valid_until")),
        buy_gtt_id=_opt_str(record.get("buy_gtt_id")),
        state=TicketState.ARMED,
    )


def ticket_key(market: str, symbol: str, scanned_on: date) -> str:
    """
    `{market}:{SYMBOL}:{date}` - unique per proposal.

    Market-qualified for the reason everything else in this package is: bare
    tickers collide across exchanges, and a collision is not a crash, it is
    one company's stop applied to another's position.
    """
    return f"{market}:{symbol.upper()}:{scanned_on.isoformat()}"


def _ticket_from_pick(pick, today: date) -> SwingTicket:
    return SwingTicket(
        key=ticket_key(pick.market, pick.symbol, pick.scanned_on or today),
        market=pick.market, symbol=pick.symbol.upper(),
        name=getattr(pick, "name", ""), sector=getattr(pick, "sector", ""),
        setup=pick.setup.key,
        entry=float(pick.setup.entry), stop=float(pick.setup.stop),
        target=float(pick.setup.target), quantity=float(pick.quantity),
        risk_inr=float(pick.risk_inr), reward_inr=float(pick.reward_inr),
        deployed_inr=float(pick.deployed_inr),
        scanned_on=pick.scanned_on or today, valid_until=pick.valid_until,
        state=TicketState.ARMED,
    )
