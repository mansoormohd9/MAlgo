"""
The once-a-day run: reconcile, advance the ladder, move the stops.

HEADLESS, LIKE `engine.py`. Nothing here imports Streamlit. The Trade book
page is a viewer over this, exactly as `app.py` is a viewer over the option
engine - so what the page does and what a script would do cannot diverge.

THE ORDER IS THE POINT, and it is the same rule `engine.run_once()` follows:

    1. reconcile against the broker      what is actually true
    2. record fills                      money that moved without you
    3. arm exits for anything unguarded  the stop comes before the profit
    4. advance the ladder on today's bar
    5. push ratcheted stops to the broker
    6. re-price or retire unfilled entries

Position management outranks new opportunities, and *protection* outranks
management. Step 3 sits above step 4 deliberately: a position discovered
without a stop gets one before anything else is considered, because the cost
of the two errors is not symmetric.

WHAT THIS WILL NOT DO. It never opens a position. Arming an entry is a
deliberate click on a specific ticket - `arm_pick()` below, called by the page
- for the same reason `engine.confirm_entry()` exists on the option side.
Exits, once you are in, are automatic, because a stop that needs a human to
press a button is not a stop.

AND WHAT IT CANNOT DO. Without DDPI, none of the sells it arranges can execute
unless you authorised holdings in Kite that morning. Every action below is
still worth taking - the GTT is in place for the day you do authorise - but
`Outcome.protection` carries the real state and the caller must show it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from ..config import Config, DEFAULT
from .book import (Book, ReconcileReport, TicketState, SwingTicket,
                   reconcile)
from .costs_equity import DEFAULT_EQUITY_COSTS, EquityCostModel


@dataclass
class Action:
    """One thing the run did, or refused to do."""
    kind: str
    symbol: str
    detail: str
    ok: bool = True


@dataclass
class Outcome:
    report: Optional[ReconcileReport] = None
    actions: list[Action] = field(default_factory=list)
    protection: tuple[str, str] = ("unverified", "")
    free_cash: Optional[float] = None
    ran_on: Optional[date] = None

    def did(self, kind: str, symbol: str, detail: str, ok: bool = True) -> None:
        self.actions.append(Action(kind, symbol, detail, ok))

    @property
    def failures(self) -> list[Action]:
        return [a for a in self.actions if not a.ok]

    def headline(self) -> str:
        if not self.actions:
            return "Nothing needed doing."
        bad = len(self.failures)
        if bad:
            return (f"{len(self.actions)} action(s), {bad} of which the broker "
                    f"refused.")
        return f"{len(self.actions)} action(s), all accepted."


def run(book: Book, broker, bars: dict[str, pd.DataFrame],
        cfg: Config = DEFAULT, today: Optional[date] = None,
        costs: EquityCostModel = DEFAULT_EQUITY_COSTS,
        broker_reachable: bool = True) -> Outcome:
    """
    One daily pass. Safe to call twice - every step is idempotent.

    `bars` is `{symbol: daily DataFrame}`, the map the scan already downloaded,
    so this costs no extra price requests.
    """
    today = today or date.today()
    out = Outcome(ran_on=today)
    out.protection = broker.protection_state(today)

    # Snapshotted BEFORE the first read, because `reconcile` reads too. A
    # failed holdings call returns an empty list, and reconcile would then
    # report every open position as "sold outside this app" and every stop as
    # missing - a page full of alarming, entirely wrong findings.
    # `free_cash` is read FIRST and outside the guard: a margins call can
    # fail on its own - it is the one read that needs a live token even in
    # dry run - and losing the whole run, including arming stops, over an
    # unknown cash balance would be the wrong trade entirely.
    out.free_cash = broker.free_cash() if broker_reachable else None

    before_reads = getattr(broker, "read_failures", 0)
    holdings_raw = broker.holdings() if broker_reachable else []
    gtts_raw = broker.gtts() if broker_reachable else []
    reads_failed = getattr(broker, "read_failures", 0) > before_reads

    out.report = reconcile(book, broker, today,
                           broker_reachable and not reads_failed,
                           holdings_in=holdings_raw, gtts_in=gtts_raw)

    if not broker_reachable:
        return out
    if reads_failed:
        out.did("aborted", "",
                "A broker read failed, so what is actually held and resting "
                "at Zerodha is unknown. Nothing was changed - re-run once the "
                "connection is back.", ok=False)
        return out

    # ---- 2. fills that happened while you were not looking ----
    holdings = {h.symbol: h for h in holdings_raw}
    gtts = {g.trigger_id: g for g in gtts_raw}

    for t in list(book.armed):
        held = holdings.get(t.symbol.upper())
        if held is None or held.total_quantity <= 0:
            continue
        # THE BUY TRIGGER MUST BE GONE. Shares in this name are not proof the
        # ticket filled - you may simply already have owned them, from a trade
        # this book never made. Adopting those would record a fabricated fill
        # at their average cost and then arm a sell OCO over the whole
        # holding, i.e. put a stop on somebody else's position.
        #
        # A GTT is consumed when it fires, so a trigger still sitting active
        # at Zerodha means this ticket has NOT filled. `reconcile` reports the
        # stray holding separately as `untracked_holding`.
        resting = gtts.get(t.buy_gtt_id or "")
        if resting is not None and resting.is_active:
            continue
        # ...and a GONE trigger is not proof either: a GTT can be cancelled,
        # expired, or deleted by a corporate action without ever firing. Pair
        # it with an actually COMPLETED buy. Without that corroboration, a
        # cancelled trigger plus shares you happened to already own
        # fabricates a fill at their average cost and then arms a sell OCO
        # over a position this book never opened.
        if not _bought(broker, t.symbol):
            continue
        filled = min(held.total_quantity, t.quantity)
        book.record_fill(t, quantity=filled,
                         price=held.average_price, on=today)
        out.did("filled", t.symbol,
                f"{filled:g} shares at ₹{held.average_price:,.2f} "
                f"(ticket said ₹{t.entry:,.2f})")

    # ---- 3. PROTECT ANYTHING UNGUARDED, before anything else ----
    for t in list(book.open):
        # A fill at or below the stop leaves no 1R to measure in, so the
        # ladder can never promote and never fire. Close it rather than carry
        # a position whose management is silently inert.
        if t.is_degenerate:
            book.close(t, f"filled at ₹{t.entry_reference:,.2f}, at or below "
                          f"the ₹{t.stop:,.2f} stop - no risk unit to manage "
                          f"in", on=today)
            out.did("degenerate", t.symbol,
                    "filled through the stop - closed in the book; sell it "
                    "manually if you still hold it", ok=False)

    resting_exit_for = {g.symbol: g for g in gtts_raw
                        if g.is_active and g.is_exit}
    for t in book.open:
        # Match on the stored id OR on the symbol. The id alone is not
        # enough: if the book lost it - a torn write, a manually placed GTT,
        # an id that failed to come back from a modify - this would arm a
        # SECOND sell trigger over the same holding, and two OCOs can sell it
        # twice.
        known = gtts.get(t.exit_gtt_id or "")
        if known is not None and known.is_active:
            continue
        found = resting_exit_for.get(t.symbol.upper())
        if found is not None:
            book.record_exit_gtt(t, found.trigger_id, today)
            out.did("exit_adopted", t.symbol,
                    f"an exit trigger was already resting at Zerodha "
                    f"({found.trigger_id}) - adopted rather than duplicated")
            continue
        last = _last_close(bars, t.symbol) or t.entry_reference
        gtt_id = broker.place_exit_gtt(t.symbol, t.open_quantity,
                                       t.stop_price, t.target, last)
        book.record_exit_gtt(t, gtt_id, today)
        out.did("exit_armed", t.symbol,
                f"stop ₹{t.stop_price:,.2f} / target ₹{t.target:,.2f} "
                f"on {t.open_quantity:g} shares",
                ok=gtt_id is not None)

    # ---- 4-5. advance the ladder, then push any ratchet to the broker ----
    for t in list(book.open):
        dated = _bar_for(bars, t.symbol, today)
        if dated is None:
            continue
        bar_date, bar = dated
        before = t.stop_price
        snap = book.snapshot(t)
        # The BAR's date, not the run's - see `Book.advance`. Three runs over
        # a weekend must not re-consume Friday's bar three times.
        decisions = book.advance(t, high=float(bar["high"]),
                                 low=float(bar["low"]),
                                 close=float(bar["close"]),
                                 atr=_atr(bars.get(t.symbol), cfg),
                                 today=bar_date)
        # A LADDER DECISION IS NOT A FILL. `book.advance` records what the
        # rungs say happened; something still has to SELL. The resting OCO
        # cannot do it - it holds one quantity at the stop and one at the
        # structural target, and can express neither "bank half at +2R" nor a
        # trailing stop that has since moved. So every decision carrying
        # `exit_lots` becomes a real sell here.
        #
        # Getting this wrong is not a small bug: the first version booked the
        # P&L, shrank the ticket and deleted the OCO while the shares were
        # still sitting in the account, unstopped, showing as a closed winner.
        refused = False
        for d in decisions:
            out.did("ladder", t.symbol, d.detail)
            if not d.exit_lots:
                continue
            held = holdings.get(t.symbol.upper())
            still_held = held.total_quantity if held else 0.0
            if still_held <= 0:
                # The resting OCO already did it - this is the same exit seen
                # from the other side, not a second one.
                out.did("exit_confirmed", t.symbol,
                        f"{d.detail} - Zerodha already sold it")
                continue
            order_id = broker.place_market_exit(
                t.symbol, min(d.exit_lots, still_held), float(bar["close"]))
            if order_id is None:
                refused = True
            out.did("sold", t.symbol,
                    f"{d.exit_lots:g} shares - {d.detail}",
                    ok=order_id is not None)

        if refused:
            # THE no-DDPI case, and the one this module exists for. The rungs
            # fired but the broker would not sell, so the shares are still
            # there. Put the ticket back and LEAVE THE RESTING OCO ALONE - a
            # ledger showing a closed position while you still hold it is the
            # one state in which nobody goes looking for a stop.
            book.revert_exit(t, snap,
                             "the broker refused the sell - the position is "
                             "still open and the resting trigger was left in "
                             "place", on=today)
            out.did("exit_reverted", t.symbol,
                    f"You still hold {t.open_quantity:g} shares. The resting "
                    f"stop was NOT removed. Without DDPI, this is what a "
                    f"missing TPIN authorisation looks like.", ok=False)
            continue

        if t.state is not TicketState.OPEN:
            # Retire the resting OCO so a stale trigger cannot fire against
            # shares that are no longer there. Safe now: whatever the ladder
            # closed has either been sold above or was already gone.
            if t.exit_gtt_id:
                ok = broker.delete_gtt(t.exit_gtt_id) is not None
                out.did("exit_cleared", t.symbol,
                        "position closed - resting trigger deleted", ok=ok)
            continue

        if abs(t.stop_price - before) > 0.05 or _qty_changed(decisions):
            last = float(bar["close"])
            new_id = broker.modify_exit_gtt(
                t.exit_gtt_id or "", t.symbol, t.open_quantity,
                t.stop_price, t.target, last)
            out.did("stop_moved", t.symbol,
                    f"₹{before:,.2f} → ₹{t.stop_price:,.2f} "
                    f"({t.rung})", ok=new_id is not None)
            if new_id:
                book.record_exit_gtt(t, new_id, today)

    # ---- 6. unfilled entries that have gone stale ----
    for t in list(book.armed):
        if t.valid_until and today > t.valid_until:
            ok = True
            if t.buy_gtt_id:
                ok = broker.delete_gtt(t.buy_gtt_id) is not None
            book.cancel(t, f"never reached ₹{t.entry:,.2f} by "
                           f"{t.valid_until:%d %b}", on=today)
            out.did("expired", t.symbol,
                    "entry window closed - trigger removed", ok=ok)

    return out


# --------------------------------------------------------------- arming

@dataclass
class ArmCheck:
    """Whether a pick may be armed, and every reason it may not."""
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blockers


def can_arm(pick, book: Book, broker, cfg: Config = DEFAULT,
            free_cash: Optional[float] = None,
            costs: EquityCostModel = DEFAULT_EQUITY_COSTS) -> ArmCheck:
    """
    The pre-flight checks, run before a single rupee is committed.

    Every one of these is something that otherwise fails LATER - at the moment
    the trigger fires, days from now, with no one watching. A GTT reserves no
    margin and validates nothing at placement, so this is the only place these
    questions get asked while you can still act on the answer.
    """
    check = ArmCheck()
    market_pool = cfg.swing.markets[pick.market].capital_pool

    if book.for_symbol(pick.symbol) is not None:
        check.blockers.append(
            f"You already have a live ticket in {pick.symbol}. Two positions "
            f"in one name is one bet at twice the size.")

    heat = book.open_risk_r
    if heat + 1.0 > cfg.swing.max_open_risk_r + 1e-9:
        check.blockers.append(
            f"Open risk is already {heat:.1f}R. One more would put "
            f"{heat + 1:.1f}R at risk against a {cfg.swing.max_open_risk_r:.0f}R "
            f"cap - which is the day stop the governors are built around.")

    if len(book.open) + len(book.armed) >= cfg.swing.top_n:
        check.blockers.append(
            f"{cfg.swing.top_n} tickets are already live. That is the same "
            f"number as `max_entries_per_session`, deliberately.")

    wanted = pick.setup.entry * pick.quantity
    committed = book.committed_inr + sum(t.entry_reference * t.open_quantity
                                         for t in book.open)
    capital = cfg.capital.capital_inr(market_pool)
    if committed + wanted > capital + 1e-6:
        check.blockers.append(
            f"₹{wanted:,.0f} would take total deployment to "
            f"₹{committed + wanted:,.0f} against a ₹{capital:,.0f} pot.")

    if free_cash is not None and free_cash < wanted:
        check.blockers.append(
            f"Zerodha reports ₹{free_cash:,.0f} available and this needs "
            f"₹{wanted:,.0f}. A GTT blocks no margin, so it would be accepted "
            f"now and REJECTED when it triggers.")
    elif free_cash is None:
        check.warnings.append(
            "Free cash could not be read from Zerodha, so it has not been "
            "checked. A trigger that fires without funds is simply rejected.")

    state, why = broker.protection_state()
    if state != "protected":
        check.warnings.append(why)

    friction_r = costs.friction_r(pick.setup.entry, pick.setup.stop,
                                  pick.quantity)
    if friction_r > 0.15:
        check.warnings.append(
            f"Charges are {friction_r:.2f}R on this ticket - {friction_r:.0%} "
            f"of the risk budget before the trade has done anything. The flat "
            f"₹{costs.dp_charge:.2f} DP fee does not shrink with the position.")

    return check


def arm_pick(pick, book: Book, broker, cfg: Config = DEFAULT,
             today: Optional[date] = None,
             last_price: Optional[float] = None) -> tuple[bool, str]:
    """
    Place the resting BUY trigger and record the ticket.

    THE TICKET IS ONLY RECORDED IF THE BROKER ACCEPTED IT. `confirm_entry()`
    on the option side does the opposite - a failed `place_entry` still opens
    a tracked position, and the engine then manages a trade that does not
    exist. Not repeating that here.
    """
    today = today or date.today()
    trigger = float(pick.setup.entry)
    gtt_id = broker.place_buy_gtt(pick.symbol, trigger, pick.quantity,
                                  last_price or pick.last_close or trigger)
    if gtt_id is None:
        return False, (f"Zerodha did not accept the trigger for "
                       f"{pick.symbol}. Nothing was recorded.")
    book.arm(pick, buy_gtt_id=gtt_id, today=today)
    return True, (f"{pick.symbol} armed: buy {pick.quantity:g} if it trades "
                  f"through ₹{trigger:,.2f}. Zerodha watches it from here.")


def close_now(ticket: SwingTicket, book: Book, broker, last_price: float,
              today: Optional[date] = None) -> tuple[bool, str]:
    """Exit a position by hand, and retire whatever was resting for it."""
    today = today or date.today()
    if ticket.exit_gtt_id:
        broker.delete_gtt(ticket.exit_gtt_id)
    order_id = broker.place_market_exit(ticket.symbol, ticket.open_quantity,
                                        last_price)
    if order_id is None:
        return False, f"The sell for {ticket.symbol} was not accepted."
    gross = (last_price - ticket.entry_reference) * ticket.open_quantity
    ticket.realised_inr += gross
    book.close(ticket, f"closed by hand at ~₹{last_price:,.2f}", on=today)
    return True, f"{ticket.symbol} sold ({ticket.open_quantity:g} shares)."


# --------------------------------------------------------------- helpers

def _bar_for(bars: dict, symbol: str, day: date):
    """`(bar_date, bar)` for the newest session at or before `day`, or None."""
    df = bars.get(symbol)
    if df is None or df.empty:
        return None
    frame = df.loc[df.index.date <= day]
    if frame.empty:
        return None
    return frame.index[-1].date(), frame.iloc[-1]


def _bought(broker, symbol: str) -> bool:
    """
    Did a BUY in this symbol actually complete?

    The corroboration a fill needs. Reads the day's order book rather than
    inferring from a trigger's absence, because "the GTT is gone" covers
    fired, cancelled, expired and deleted-by-a-corporate-action alike - and
    only the first of those is a fill.
    """
    sym = symbol.upper()
    for o in broker.orders():
        if (str(o.get("tradingsymbol", "")).upper() == sym
                and str(o.get("transaction_type", "")).upper() == "BUY"
                and str(o.get("status", "")).upper() == "COMPLETE"):
            return True
    return False


def _last_close(bars: dict, symbol: str) -> Optional[float]:
    df = bars.get(symbol)
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


def _atr(df: Optional[pd.DataFrame], cfg: Config) -> float:
    if df is None or len(df) < cfg.swing.atr_period + 1:
        return 0.0
    from ..signals import atr as atr_fn
    value = atr_fn(df, cfg.swing.atr_period)
    try:
        value = float(value.iloc[-1]) if hasattr(value, "iloc") else float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(value) else value


def _qty_changed(decisions) -> bool:
    return any(d.exit_lots for d in decisions)
