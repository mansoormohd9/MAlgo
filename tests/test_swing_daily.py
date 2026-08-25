"""
The once-a-day run, and the pre-flight checks before a rupee is committed.

Two things here are worth more than the rest.

`can_arm` is where every "this fails LATER" question gets asked while you can
still act on the answer. A GTT reserves no margin and validates nothing at
placement, so without these checks the failure surfaces days from now, at the
moment the trigger fires, with nobody watching.

`run`'s ORDERING is the other. Protection comes before management: a position
discovered without a stop gets one before the ladder is even consulted,
because the cost of the two errors is not symmetric.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from conftest import daily_bars, trending

from nifty_algo.config import Config
from nifty_algo.journal import Journal
from nifty_algo.swing import daily as daily_mod
from nifty_algo.swing.book import Book, TicketState
from nifty_algo.broker.kite_equity import GttView, HoldingView

TODAY = date(2026, 8, 24)


# ---------------------------------------------------------------- fakes

class _Setup:
    def __init__(self, entry, stop, target):
        self.entry, self.stop, self.target = entry, stop, target
        self.key = "breakout"


class _Pick:
    def __init__(self, symbol="INFY", entry=1000.0, stop=950.0,
                 target=1150.0, qty=20, market="india"):
        self.symbol, self.name = symbol, f"{symbol} Ltd"
        self.sector, self.market = "Information Technology", market
        self.setup = _Setup(entry, stop, target)
        self.quantity = qty
        self.risk_inr = (entry - stop) * qty
        self.reward_inr = (target - entry) * qty
        self.deployed_inr = entry * qty
        self.last_close = entry * 0.99
        self.scanned_on = TODAY
        self.valid_until = TODAY + timedelta(days=5)


class _Broker:
    """Records every write and answers reads from what it was seeded with."""

    def __init__(self, holdings=(), gtts=(), cash=1_000_000.0,
                 protection=("protected", "ok"), accept=True,
                 read_failures=0, orders=None):
        #: A gone GTT is not proof of a fill - it may have been cancelled or
        #: expired. `daily.run` corroborates against a COMPLETED buy, so a
        #: holding here has to come with one by default.
        self._orders = ([{"tradingsymbol": h.symbol, "transaction_type": "BUY",
                          "status": "COMPLETE"} for h in holdings]
                        if orders is None else list(orders))
        self._holdings = list(holdings)
        self._gtts = list(gtts)
        self._cash = cash
        self._protection = protection
        self.accept = accept
        self.calls: list[tuple] = []
        self._next = 0
        #: Mirrors `BrokerTransport.read_failures`. `daily.run` compares it
        #: before and after its reads, because an empty holdings/GTT list
        #: means both "nothing there" and "the request failed".
        self.read_failures = read_failures

    # reads
    def holdings(self):
        return self._holdings

    def gtts(self):
        return self._gtts

    def orders(self):
        return self._orders

    def free_cash(self):
        return self._cash

    def protection_state(self, day=None):
        return self._protection

    # writes
    def _id(self, what, *args):
        self.calls.append((what, *args))
        if not self.accept:
            return None
        self._next += 1
        return f"{what}-{self._next}"

    def place_buy_gtt(self, symbol, trigger, qty, last):
        return self._id("buy", symbol, trigger, qty)

    def place_exit_gtt(self, symbol, qty, stop, target, last):
        return self._id("exit", symbol, qty, stop, target)

    def modify_exit_gtt(self, tid, symbol, qty, stop, target, last):
        return self._id("modify", symbol, qty, stop, target)

    def delete_gtt(self, tid):
        return self._id("delete", tid)

    def place_market_exit(self, symbol, qty, last):
        return self._id("sell", symbol, qty)

    def did(self, what):
        return [c for c in self.calls if c[0] == what]


def holding(symbol="INFY", qty=20.0, t1=0.0, avg=1000.0):
    return HoldingView(symbol=symbol, exchange="NSE", quantity=qty,
                       t1_quantity=t1, average_price=avg, last_price=avg,
                       pnl=0.0)


def exit_gtt(tid="G1", symbol="INFY", stop=950.0, target=1150.0, qty=20.0):
    return GttView(trigger_id=tid, symbol=symbol, exchange="NSE",
                   status="active", trigger_type="two-leg",
                   trigger_values=[stop, target], quantity=qty,
                   transaction_type="SELL")


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.capital.swing_capital_inr = 100_000.0
    return c


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal")


@pytest.fixture
def book(cfg, journal) -> Book:
    return Book(cfg, journal)


def bars_for(symbol="INFY", high=1010.0, low=990.0, close=1005.0):
    """One frame whose LAST bar is exactly the one we want managed."""
    df = daily_bars(trending(n=90, seed=3, start_price=1000.0))
    df = df.copy()
    df.iloc[-1, df.columns.get_loc("high")] = high
    df.iloc[-1, df.columns.get_loc("low")] = low
    df.iloc[-1, df.columns.get_loc("close")] = close
    df.index = pd.DatetimeIndex(
        pd.bdate_range(end=pd.Timestamp(TODAY), periods=len(df)))
    return {symbol: df}


# ---------------------------------------------------------------- can_arm

def test_a_name_you_already_hold_cannot_be_armed_again(cfg, book):
    """Two positions in one name is one bet at twice the size."""
    book.arm(_Pick("INFY"), buy_gtt_id="B1", today=TODAY)
    check = daily_mod.can_arm(_Pick("INFY"), book, _Broker(), cfg,
                              free_cash=1_000_000.0)

    assert not check.ok
    assert any("already have a live ticket" in b for b in check.blockers)


def test_arming_beyond_the_heat_cap_is_refused(cfg, book):
    """
    Total open risk, not per-trade risk.

    `top_n` caps what ONE scan proposes. Nothing stopped three scans on three
    days putting nine positions on and risking 9R against a day stop built
    for 3.
    """
    for i, sym in enumerate(("AAA", "BBB", "CCC")):
        t = book.arm(_Pick(sym), buy_gtt_id=f"B{i}", today=TODAY)
        book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    assert book.open_risk_r == pytest.approx(3.0)

    check = daily_mod.can_arm(_Pick("DDD"), book, _Broker(), cfg,
                              free_cash=1_000_000.0)

    assert not check.ok
    assert any("Open risk" in b for b in check.blockers)


def test_arming_more_than_the_pot_can_pay_for_is_refused(cfg, book):
    cfg.capital.swing_capital_inr = 15_000.0
    check = daily_mod.can_arm(_Pick("INFY", entry=1000.0, qty=20), book,
                              _Broker(), cfg, free_cash=1_000_000.0)

    assert not check.ok
    assert any("pot" in b for b in check.blockers)


def test_insufficient_broker_cash_is_refused_with_the_reason(cfg, book):
    """
    A GTT blocks no margin. Without this check the order is accepted now and
    REJECTED days later when it triggers - at the exact moment the setup was
    working.
    """
    check = daily_mod.can_arm(_Pick("INFY", entry=1000.0, qty=20), book,
                              _Broker(), cfg, free_cash=5_000.0)

    assert not check.ok
    assert any("blocks no margin" in b for b in check.blockers)


def test_an_unreadable_balance_warns_but_does_not_block(cfg, book):
    """
    None is "not checked", not "no money". Blocking on it would make an
    unauthenticated session look like an empty account.
    """
    check = daily_mod.can_arm(_Pick(), book, _Broker(), cfg, free_cash=None)

    assert check.ok
    assert any("could not be read" in w for w in check.warnings)


def test_a_missing_authorisation_warns_rather_than_blocking(cfg, book):
    """
    The BUY is unaffected by TPIN - only sells need it. Blocking the entry
    would be the wrong lever; telling you the exit is not yet protected is
    the right one.
    """
    broker = _Broker(protection=("unprotected", "no authorisation today"))
    check = daily_mod.can_arm(_Pick(), book, broker, cfg,
                              free_cash=1_000_000.0)

    assert check.ok
    assert any("authorisation" in w for w in check.warnings)


def test_a_clean_pick_passes(cfg, book):
    check = daily_mod.can_arm(_Pick(), book, _Broker(), cfg,
                              free_cash=1_000_000.0)
    assert check.ok and not check.blockers


# ---------------------------------------------------------------- arm_pick

def test_a_refused_order_records_no_ticket(cfg, book):
    """
    The option book does the opposite: a failed `place_entry` still opens a
    tracked position, and the engine then manages a trade that does not
    exist. Not repeating that here.
    """
    broker = _Broker(accept=False)
    ok, msg = daily_mod.arm_pick(_Pick(), book, broker, cfg, today=TODAY)

    assert not ok
    assert "Nothing was recorded" in msg
    assert book.tickets == {}


def test_an_accepted_order_records_the_ticket_with_its_trigger_id(cfg, book):
    broker = _Broker()
    ok, _ = daily_mod.arm_pick(_Pick(), book, broker, cfg, today=TODAY)

    assert ok
    ticket = book.for_symbol("INFY")
    assert ticket.state is TicketState.ARMED
    assert ticket.buy_gtt_id == "buy-1"


# ---------------------------------------------------------------- the run

def test_a_fill_that_happened_overnight_is_recorded(cfg, book):
    book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    broker = _Broker(holdings=[holding(qty=20.0, avg=1004.0)])

    out = daily_mod.run(book, broker, bars_for(), cfg, today=TODAY)

    ticket = book.for_symbol("INFY")
    assert ticket.state is TicketState.OPEN
    assert ticket.avg_fill_price == pytest.approx(1004.0)
    assert any(a.kind == "filled" for a in out.actions)


def test_an_unguarded_position_gets_a_stop_before_anything_else(cfg, book):
    """
    THE ordering rule. Protection outranks management.

    A position found with no resting stop is armed first, before the ladder
    is consulted at all - the cost of the two errors is not symmetric.
    """
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    broker = _Broker(holdings=[holding()], gtts=[])      # nothing resting

    out = daily_mod.run(book, broker, bars_for(), cfg, today=TODAY)

    assert broker.did("exit")
    assert broker.calls[0][0] == "exit"                 # FIRST, not later
    assert any(a.kind == "exit_armed" for a in out.actions)


def test_a_ratcheted_stop_is_pushed_to_the_broker(cfg, book):
    """The trailing stop only exists once it reaches Zerodha."""
    t = book.arm(_Pick(entry=1000.0, stop=950.0), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt()])

    # A bar reaching +1R moves the stop to breakeven.
    out = daily_mod.run(book, broker, bars_for(high=1060.0, low=1005.0,
                                               close=1055.0),
                        cfg, today=TODAY)

    assert t.ladder.stop_r == pytest.approx(0.0)
    assert broker.did("modify")
    assert any(a.kind == "stop_moved" for a in out.actions)


def test_a_closed_position_has_its_resting_trigger_removed(cfg, book):
    """
    A stale trigger on shares you no longer own can fire into a short.
    """
    t = book.arm(_Pick(entry=1000.0, stop=950.0, qty=1), buy_gtt_id="B1",
                 today=TODAY)
    book.record_fill(t, quantity=1, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[holding(qty=1.0)], gtts=[exit_gtt(qty=1.0)])

    # One share cannot split, so +2R is a full exit.
    daily_mod.run(book, broker, bars_for(high=1110.0, low=1005.0,
                                         close=1105.0), cfg, today=TODAY)

    assert t.state is TicketState.CLOSED
    assert broker.did("delete")


def test_an_expired_entry_is_cancelled_and_its_trigger_deleted(cfg, book):
    """
    A GTT lives for a YEAR. The setup was read off a chart that is now stale,
    so leaving it standing means firing in March on a thesis from August.
    """
    book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    broker = _Broker(holdings=[], gtts=[])

    out = daily_mod.run(book, broker, bars_for(), cfg,
                        today=TODAY + timedelta(days=9))

    ticket = list(book.tickets.values())[0]
    assert ticket.state is TicketState.CANCELLED
    assert broker.did("delete")
    assert any(a.kind == "expired" for a in out.actions)


def test_the_run_is_idempotent(cfg, book):
    """Safe to press twice - the second pass must not re-arm anything."""
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt()])
    bars = bars_for()

    daily_mod.run(book, broker, bars, cfg, today=TODAY)
    before = len(broker.calls)
    daily_mod.run(book, broker, bars, cfg, today=TODAY)

    assert len(broker.calls) == before


def test_an_unreachable_broker_does_nothing_and_says_so(cfg, book):
    """
    Acting on facts you could not read is how a stop gets moved on a price
    that never printed.
    """
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    broker = _Broker()

    out = daily_mod.run(book, broker, bars_for(), cfg, today=TODAY,
                        broker_reachable=False)

    assert broker.calls == []
    assert not out.report.broker_reachable


def test_a_refused_broker_call_is_marked_as_a_failure(cfg, book):
    """
    An action the broker rejected must not read as an action that happened.
    """
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    broker = _Broker(holdings=[holding()], gtts=[], accept=False)

    out = daily_mod.run(book, broker, bars_for(), cfg, today=TODAY)

    assert out.failures
    assert "refused" in out.headline()


# ---------------------------------------------------------------- close_now

def test_closing_by_hand_removes_the_resting_trigger_first(cfg, book):
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker()

    ok, _ = daily_mod.close_now(t, book, broker, last_price=1080.0,
                                today=TODAY)

    assert ok
    assert [c[0] for c in broker.calls] == ["delete", "sell"]
    assert t.state is TicketState.CLOSED
    assert t.realised_inr == pytest.approx(1600.0)      # 20 x Rs 80


def test_a_failed_broker_read_aborts_instead_of_double_arming(cfg, book):
    """
    An empty GTT list means BOTH "nothing is resting" and "the request
    failed", and treating the second as the first places a SECOND stop on a
    position that already has one. Two OCOs on one holding can sell it twice.
    """
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)

    class _Flaky(_Broker):
        def gtts(self):
            self.read_failures += 1      # the request failed
            return []

    broker = _Flaky(holdings=[holding()], gtts=[exit_gtt()])
    out = daily_mod.run(book, broker, bars_for(), cfg, today=TODAY)

    assert not broker.did("exit")                 # nothing was armed twice
    assert any(a.kind == "aborted" for a in out.actions)


def test_a_ladder_partial_actually_sells(cfg, book):
    """
    THE bug this test exists for.

    `book.advance` records what the rungs say; something still has to SELL.
    The resting OCO cannot express "bank half at +2R" - it holds one quantity
    at the stop and one at the target. Without an explicit sell the first
    version booked the profit, shrank the ticket and left the shares sitting
    in the account showing as a banked win.
    """
    t = book.arm(_Pick(entry=1000.0, stop=950.0, target=1400.0, qty=20),
                 buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[holding(qty=20.0)], gtts=[exit_gtt()])

    # +2R is Rs 1,100 - well short of the Rs 1,400 structural target, so the
    # OCO could not possibly have done this.
    out = daily_mod.run(book, broker, bars_for(high=1120.0, low=1005.0,
                                               close=1115.0),
                        cfg, today=TODAY)

    assert t.ladder.lots_remaining == 10          # half banked
    sold = broker.did("sell")
    assert sold and sold[0][2] == 10              # ...and half actually SOLD
    assert any(a.kind == "sold" for a in out.actions)


def test_an_exit_the_broker_already_made_is_not_sold_twice(cfg, book):
    """
    If the resting OCO fired, the shares are gone. Selling again would open
    a short in a cash account.
    """
    t = book.arm(_Pick(entry=1000.0, stop=950.0, qty=1), buy_gtt_id="B1",
                 today=TODAY)
    book.record_fill(t, quantity=1, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[], gtts=[exit_gtt(qty=1.0)])   # already sold

    out = daily_mod.run(book, broker, bars_for(high=1110.0, low=1005.0,
                                               close=1105.0), cfg, today=TODAY)

    assert not broker.did("sell")
    assert any(a.kind == "exit_confirmed" for a in out.actions)


# ------------------------------------------------- regressions worth naming

def test_running_twice_on_the_same_bar_does_not_stop_the_position_out(cfg, book):
    """
    A daily bar must be consumed EXACTLY once.

    The first pass ratchets the stop up; re-feeding the same bar then tests
    that bar's low against the NEW stop and stops out on a move that never
    happened - and, because ladder exits now place real orders, sells for
    real. The idempotency claim in `run`'s docstring has to be true.
    """
    t = book.arm(_Pick(entry=1000.0, stop=950.0, target=1400.0, qty=20),
                 buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt()])
    # Reaches +1R (stop to breakeven at 1000) but the same bar's low is 1005.
    bars = bars_for(high=1060.0, low=1005.0, close=1055.0)

    daily_mod.run(book, broker, bars, cfg, today=TODAY)
    assert t.state is TicketState.OPEN
    assert t.ladder.stop_r == pytest.approx(0.0)

    daily_mod.run(book, broker, bars, cfg, today=TODAY)

    assert t.state is TicketState.OPEN          # NOT stopped out
    assert not broker.did("sell")


def test_a_bar_is_still_consumed_after_a_reload(cfg, journal):
    """
    `laddered_on` has to survive journal replay, or reopening the app
    re-opens the same hole.
    """
    book = Book(cfg, journal)
    t = book.arm(_Pick(entry=1000.0, stop=950.0, target=1400.0, qty=20),
                 buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.advance(t, high=1060.0, low=1005.0, close=1055.0, atr=12.0,
                 today=TODAY)

    reloaded = Book.load(journal, cfg)
    r = reloaded.tickets[t.key]

    assert r.laddered_on == TODAY
    assert reloaded.advance(r, high=1060.0, low=1005.0, close=1055.0,
                            atr=12.0, today=TODAY) == []


def test_shares_you_already_owned_are_not_adopted_as_a_fill(cfg, book):
    """
    Holding a name is not proof this ticket filled - you may simply have
    owned it already. Adopting it would record a fabricated fill at that
    holding's average cost and then arm a sell OCO over somebody else's
    position.

    A GTT is consumed when it fires, so a trigger still active means no fill.
    """
    book.arm(_Pick("INFY"), buy_gtt_id="B1", today=TODAY)
    resting_buy = GttView(trigger_id="B1", symbol="INFY", exchange="NSE",
                          status="active", trigger_type="single",
                          trigger_values=[1000.0], quantity=20,
                          transaction_type="BUY")
    broker = _Broker(holdings=[holding(qty=100.0, avg=700.0)],
                     gtts=[resting_buy], orders=[])

    out = daily_mod.run(book, broker, bars_for(), cfg, today=TODAY)

    assert book.for_symbol("INFY").state is TicketState.ARMED
    assert not broker.did("exit")
    assert not any(a.kind == "filled" for a in out.actions)


def test_a_fill_below_the_stop_is_closed_rather_than_carried(cfg, book):
    """
    A gap through the trigger can fill below the stop. There is then no 1R to
    measure in: `r_of()` returns 0.0 for every price, so the ladder can never
    promote and never fire, and the stop collapses onto the fill. A position
    whose management is silently inert has to be surfaced.
    """
    t = book.arm(_Pick(entry=1000.0, stop=950.0), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=940.0, on=TODAY)   # below the stop
    assert t.is_degenerate

    out = daily_mod.run(book, broker=_Broker(holdings=[holding(avg=940.0)]),
                        bars=bars_for(), cfg=cfg, today=TODAY)

    assert t.state is TicketState.CLOSED
    assert any(a.kind == "degenerate" and not a.ok for a in out.actions)


def test_a_failed_read_stops_reconcile_reporting_false_alarms(cfg, book):
    """
    An empty holdings list from a FAILED call would have reconcile report
    every open position as "sold outside this app" and every stop as missing
    - a page full of alarming, entirely wrong findings.
    """
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)

    class _Flaky(_Broker):
        def holdings(self):
            self.read_failures += 1
            return []

    out = daily_mod.run(book, _Flaky(), bars_for(), cfg, today=TODAY)

    assert not out.report.broker_reachable
    assert any(a.kind == "aborted" for a in out.actions)
    assert not out.report.of("critical")


def test_a_refused_sell_leaves_the_position_open_and_the_stop_in_place(cfg, book):
    """
    THE no-DDPI case, and the one this module exists for.

    The rungs fire, the broker refuses the sell, and the shares are still
    there. A ledger showing a closed position while you still hold it is the
    one state in which nobody goes looking for a stop - so the ticket is put
    back and the resting OCO is left alone.
    """
    t = book.arm(_Pick(entry=1000.0, stop=950.0, target=1400.0, qty=20),
                 buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt()], accept=False)

    out = daily_mod.run(book, broker, bars_for(high=1120.0, low=1005.0,
                                               close=1115.0),
                        cfg, today=TODAY)

    assert t.state is TicketState.OPEN            # NOT closed
    assert t.ladder.lots_remaining == 20          # nothing was banked
    assert t.realised_inr == 0.0
    assert not broker.did("delete")               # the stop was left alone
    assert any(a.kind == "exit_reverted" and not a.ok for a in out.actions)


def test_a_revert_survives_a_reload(cfg, journal):
    """The compensating event has to replay, or a restart re-loses it."""
    book = Book(cfg, journal)
    t = book.arm(_Pick(entry=1000.0, stop=950.0, target=1400.0, qty=20),
                 buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt()], accept=False)
    daily_mod.run(book, broker, bars_for(high=1120.0, low=1005.0,
                                         close=1115.0), cfg, today=TODAY)

    r = Book.load(journal, cfg).tickets[t.key]
    assert r.state is TicketState.OPEN
    assert r.ladder.lots_remaining == 20


def test_a_stale_bar_is_not_re_consumed_over_a_weekend(cfg, book):
    """
    On a Saturday the newest bar is still Friday's. Keying the guard on the
    RUN date rather than the BAR date would re-test Friday's low against the
    stop Friday itself had ratcheted, and sell on a move that never happened.
    """
    t = book.arm(_Pick(entry=1000.0, stop=950.0, target=1400.0, qty=20),
                 buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    book.record_exit_gtt(t, "G1", today=TODAY)
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt()])
    bars = bars_for(high=1060.0, low=1005.0, close=1055.0)

    daily_mod.run(book, broker, bars, cfg, today=TODAY)
    sells_before = len(broker.did("sell"))
    # Two more days pass with no new bar - the cache still ends on TODAY.
    daily_mod.run(book, broker, bars, cfg, today=TODAY + timedelta(days=1))
    daily_mod.run(book, broker, bars, cfg, today=TODAY + timedelta(days=2))

    assert t.state is TicketState.OPEN
    assert len(broker.did("sell")) == sells_before


def test_an_exit_trigger_already_resting_is_adopted_not_duplicated(cfg, book):
    """
    Two OCOs on one holding can sell it twice. If the book lost the id, match
    on the symbol rather than arming a second trigger.
    """
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)
    t.exit_gtt_id = None                       # the id was lost
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt(tid="G9")])

    out = daily_mod.run(book, broker, bars_for(), cfg, today=TODAY)

    assert not broker.did("exit")
    assert t.exit_gtt_id == "G9"
    assert any(a.kind == "exit_adopted" for a in out.actions)


def test_an_unreadable_balance_does_not_abort_the_whole_run(cfg, book):
    """
    A margins call needs a live token even in dry run. Losing the run -
    including arming stops - over an unknown cash balance would be the wrong
    trade entirely.
    """
    t = book.arm(_Pick(), buy_gtt_id="B1", today=TODAY)
    book.record_fill(t, quantity=20, price=1000.0, on=TODAY)

    class _NoCash(_Broker):
        def free_cash(self):
            self.read_failures += 1
            return None

    broker = _NoCash(holdings=[holding()], gtts=[])
    out = daily_mod.run(book, broker, bars_for(), cfg, today=TODAY)

    assert broker.did("exit")                  # the stop was still armed
    assert not any(a.kind == "aborted" for a in out.actions)
