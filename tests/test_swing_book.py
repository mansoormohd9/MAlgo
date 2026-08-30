"""
The position book: the ladder on shares, journal replay, and reconciliation.

Reconciliation is the part that earns its tests. Every finding it can produce
is a real thing that has happened to somebody with a real position on, and the
two CRITICAL ones - a position with no live stop, and a stop that cannot
execute for want of a TPIN authorisation - are the difference between a
managed trade and an unmanaged one.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from nifty_algo.config import Config
from nifty_algo.journal import Journal
from nifty_algo.positions import ExitKind
from nifty_algo.swing import book as book_mod
from nifty_algo.swing.book import (Book, SEVERITY_CRITICAL, TicketState,
                                   performance, reconcile, ticket_key)
from nifty_algo.broker.kite_equity import GttView, HoldingView

TODAY = date(2026, 8, 24)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.capital.swing_capital_inr = 100_000.0
    return c


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal")


class _Setup:
    def __init__(self, entry, stop, target, key="breakout"):
        self.entry, self.stop, self.target, self.key = entry, stop, target, key


class _Pick:
    """The smallest thing that quacks like a SwingPick for `Book.arm`."""

    def __init__(self, symbol="INFY", entry=1000.0, stop=950.0, target=1150.0,
                 quantity=20, scanned_on=TODAY, market="india"):
        self.symbol = symbol
        self.name = f"{symbol} Ltd"
        self.sector = "Information Technology"
        self.market = market
        self.setup = _Setup(entry, stop, target)
        self.quantity = quantity
        self.risk_inr = (entry - stop) * quantity
        self.reward_inr = (target - entry) * quantity
        self.deployed_inr = entry * quantity
        self.scanned_on = scanned_on
        self.valid_until = scanned_on + timedelta(days=5)


class _Broker:
    """A stand-in Zerodha. Reconciliation reads exactly these four things."""

    def __init__(self, holdings=(), gtts=(), protection=("protected", "ok")):
        self._holdings = list(holdings)
        self._gtts = list(gtts)
        self._protection = protection

    def holdings(self):
        return self._holdings

    def gtts(self):
        return self._gtts

    def protection_state(self, day=None):
        return self._protection


def holding(symbol="INFY", qty=20.0, t1=0.0, avg=1000.0, ltp=1000.0):
    return HoldingView(symbol=symbol, exchange="NSE", quantity=qty,
                       t1_quantity=t1, average_price=avg, last_price=ltp,
                       pnl=(ltp - avg) * qty)


def exit_gtt(trigger_id="G1", symbol="INFY", stop=950.0, target=1150.0,
             qty=20.0, status="active"):
    return GttView(trigger_id=trigger_id, symbol=symbol, exchange="NSE",
                   status=status, trigger_type="two-leg",
                   trigger_values=[stop, target], quantity=qty,
                   transaction_type="SELL")


def armed_ticket(book, pick=None, gtt="B1"):
    return book.arm(pick or _Pick(), buy_gtt_id=gtt, today=TODAY)


def open_ticket(book, pick=None, fill=1000.0, gtt="G1"):
    t = armed_ticket(book, pick)
    book.record_fill(t, quantity=pick.quantity if pick else 20, price=fill,
                     on=TODAY)
    book.record_exit_gtt(t, gtt, today=TODAY)
    return t


# ---------------------------------------------------------------- the ticket

def test_r_is_measured_from_the_actual_fill_not_the_ticket(cfg, journal):
    """
    A fill worse than the trigger is risk you really took.

    Measuring R off the hoped-for entry would score a stop-out at better than
    -1R and flatter every statistic built on top of it.
    """
    b = Book(cfg, journal)
    t = open_ticket(b, _Pick(entry=1000.0, stop=950.0), fill=1004.0)

    assert t.entry_reference == 1004.0
    assert t.risk_points == pytest.approx(54.0)      # 1004 - 950, not 50
    assert t.r_of(950.0) == pytest.approx(-1.0)
    assert t.r_of(1004.0) == pytest.approx(0.0)


def test_the_stop_price_follows_the_ladder(cfg, journal):
    b = Book(cfg, journal)
    t = open_ticket(b, _Pick(entry=1000.0, stop=950.0), fill=1000.0)

    assert t.stop_price == pytest.approx(950.0)      # -1R
    t.ladder.stop_r = 0.0
    assert t.stop_price == pytest.approx(1000.0)     # breakeven
    t.ladder.stop_r = 1.5
    assert t.stop_price == pytest.approx(1075.0)     # +1.5R, locked in


def test_open_risk_is_zero_once_the_stop_is_at_breakeven(cfg, journal):
    """That is the entire point of moving it there, and heat must reflect it."""
    b = Book(cfg, journal)
    t = open_ticket(b)

    assert t.open_risk_r == pytest.approx(1.0)
    t.ladder.stop_r = 0.0
    assert t.open_risk_r == 0.0


# ---------------------------------------------------------------- the ladder

def test_the_users_rule_start_to_finish(cfg, journal):
    """
    Rs 10,000 in, Rs 500 risk, Rs 1,000 target - the rule as stated.

    +1R moves the stop to breakeven, +2R banks half, and the remainder trails.
    """
    b = Book(cfg, journal)
    # 20 shares at Rs 500, stop Rs 25 away -> 1R = Rs 500 on Rs 10,000.
    pick = _Pick(entry=500.0, stop=475.0, target=600.0, quantity=20)
    t = open_ticket(b, pick, fill=500.0)

    assert t.deployed_inr == pytest.approx(10_000.0)
    assert t.risk_points * t.filled_qty == pytest.approx(500.0)

    # Day one: +1R (Rs 525). Stop to breakeven - the trade cannot lose now.
    b.advance(t, high=525.0, low=505.0, close=520.0, atr=12.0, today=TODAY)
    assert t.ladder.stop_r == pytest.approx(0.0)
    assert t.stop_price == pytest.approx(500.0)
    assert t.open_risk_r == 0.0

    # Day two: +2R (Rs 550). Bank half; Rs 1,000 of profit is realised.
    b.advance(t, high=550.0, low=530.0, close=545.0, atr=12.0,
              today=TODAY + timedelta(days=1))
    assert t.ladder.lots_remaining == 10
    assert t.realised_inr == pytest.approx(500.0)     # 10 shares x Rs 50
    assert any(e["kind"] == ExitKind.PARTIAL_EXIT.value for e in t.exits)

    # Day three: it keeps going. The stop ratchets up and never back down.
    before = t.ladder.stop_r
    b.advance(t, high=600.0, low=548.0, close=595.0, atr=12.0,
              today=TODAY + timedelta(days=2))
    assert t.ladder.stop_r > before
    b.advance(t, high=596.0, low=560.0, close=570.0, atr=12.0,
              today=TODAY + timedelta(days=3))
    assert t.ladder.stop_r >= before


def test_a_bar_covering_both_the_stop_and_the_next_rung_is_a_stop(cfg, journal):
    """
    Pessimism, same as everywhere else here.

    Daily data cannot say which extreme came first. Assuming the good half of
    an unknowable coin flip is how a record flatters itself.
    """
    b = Book(cfg, journal)
    t = open_ticket(b, _Pick(entry=1000.0, stop=950.0, quantity=20))

    b.advance(t, high=1120.0, low=945.0, close=1000.0, atr=20.0, today=TODAY)

    assert t.state is TicketState.CLOSED
    assert t.realised_r == pytest.approx(-1.0)
    assert t.ladder.lots_remaining == 0


def test_a_single_share_ticket_cannot_split_and_takes_a_full_exit(cfg, journal):
    b = Book(cfg, journal)
    t = open_ticket(b, _Pick(entry=1000.0, stop=950.0, quantity=1))

    b.advance(t, high=1100.0, low=1005.0, close=1090.0, atr=20.0, today=TODAY)

    assert t.state is TicketState.CLOSED
    assert [e["kind"] for e in t.exits] == [ExitKind.TARGET_EXIT.value]


# ---------------------------------------------------------------- replay

def test_the_book_rebuilds_itself_from_the_journal(cfg, journal):
    """
    Event-sourced, so the ledger cannot drift from what was written down.

    Nothing is persisted except the events; a book that loaded from a mutable
    file could disagree with the journal, and then neither would be evidence.
    """
    b = Book(cfg, journal)
    t = open_ticket(b, _Pick(entry=500.0, stop=475.0, quantity=20), fill=502.0)
    b.advance(t, high=555.0, low=505.0, close=550.0, atr=10.0, today=TODAY)

    reloaded = Book.load(journal, cfg)
    r = reloaded.tickets[t.key]

    assert r.state is t.state
    assert r.avg_fill_price == pytest.approx(502.0)
    assert r.ladder.lots_remaining == t.ladder.lots_remaining
    assert r.ladder.stop_r == pytest.approx(t.ladder.stop_r)
    assert r.ladder.mode is t.ladder.mode
    assert r.realised_inr == pytest.approx(t.realised_inr)


def test_replay_can_be_filtered_to_one_market(cfg, journal):
    b = Book(cfg, journal)
    armed_ticket(b, _Pick(symbol="INFY", market="india"), gtt="B1")
    armed_ticket(b, _Pick(symbol="AAPL", market="us"), gtt="B2")

    india = Book.load(journal, cfg, market="india")
    assert {t.symbol for t in india.tickets.values()} == {"INFY"}


def test_slippage_is_recorded_on_every_fill(cfg, journal):
    """After twenty trades this is measured, not modelled."""
    b = Book(cfg, journal)
    open_ticket(b, _Pick(entry=1000.0), fill=1004.0)

    fills = [r for r in journal.read_day(TODAY)
             if r["event"] == book_mod.EVENT_FILLED]
    assert fills and fills[0]["slippage_pct"] == pytest.approx(0.4)


# ---------------------------------------------------------------- heat

def test_portfolio_heat_sums_open_risk_across_positions(cfg, journal):
    b = Book(cfg, journal)
    open_ticket(b, _Pick(symbol="AAA"), gtt="G1")
    open_ticket(b, _Pick(symbol="BBB"), gtt="G2")
    third = open_ticket(b, _Pick(symbol="CCC"), gtt="G3")

    assert b.open_risk_r == pytest.approx(3.0)

    third.ladder.stop_r = 0.0        # moved to breakeven
    assert b.open_risk_r == pytest.approx(2.0)


def test_one_live_ticket_per_symbol_is_findable(cfg, journal):
    """The duplicate-name guard needs this to be unambiguous."""
    b = Book(cfg, journal)
    t = armed_ticket(b, _Pick(symbol="INFY"))

    assert b.for_symbol("infy") is t
    b.cancel(t, "dropped out of the scan", on=TODAY)
    assert b.for_symbol("INFY") is None


# ---------------------------------------------------------------- reconcile

def test_an_open_position_with_no_resting_exit_is_critical(cfg, journal):
    b = Book(cfg, journal)
    open_ticket(b)
    broker = _Broker(holdings=[holding()], gtts=[])       # the GTT is gone

    report = reconcile(b, broker, today=TODAY)

    kinds = {f.kind for f in report.critical}
    assert book_mod.F_UNPROTECTED in kinds
    assert not report.clean


def test_a_missing_tpin_authorisation_is_critical(cfg, journal):
    """
    The dangerous case: the GTT is there and looks fine, and will be rejected.

    Without DDPI the sell needs an authorisation that expires nightly. Kite
    still shows the trigger as active, so nothing about the broker's own
    screen tells you the stop is decorative.
    """
    b = Book(cfg, journal)
    open_ticket(b)
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt()],
                     protection=("unprotected", "no authorisation today"))

    report = reconcile(b, broker, today=TODAY)

    assert book_mod.F_NO_AUTHORISATION in {f.kind for f in report.critical}


def test_an_authorised_but_unverifiable_day_warns_rather_than_passing(cfg, journal):
    b = Book(cfg, journal)
    open_ticket(b)
    broker = _Broker(holdings=[holding()], gtts=[exit_gtt()],
                     protection=("unverified", "cannot be confirmed"))

    report = reconcile(b, broker, today=TODAY)
    findings = [f for f in report.findings
                if f.kind == book_mod.F_NO_AUTHORISATION]

    assert findings and findings[0].severity == book_mod.SEVERITY_WARNING


def test_a_fill_that_happened_since_the_last_run_is_detected(cfg, journal):
    b = Book(cfg, journal)
    armed_ticket(b)
    broker = _Broker(holdings=[holding(qty=20.0)])

    report = reconcile(b, broker, today=TODAY)

    assert book_mod.F_FILLED in {f.kind for f in report.findings}


def test_t1_shares_count_as_a_fill(cfg, journal):
    """
    The day after a buy, Kite reports the shares under `t1_quantity`.

    Reading only `quantity` shows zero, which is indistinguishable from "the
    buy never happened" - and would have the reconciler tear down the stop on
    a live position.
    """
    b = Book(cfg, journal)
    armed_ticket(b)
    broker = _Broker(holdings=[holding(qty=0.0, t1=20.0)])

    report = reconcile(b, broker, today=TODAY)

    assert book_mod.F_FILLED in {f.kind for f in report.findings}


def test_a_vanished_buy_trigger_is_reported(cfg, journal):
    """Corporate actions delete GTTs, and an unfilled one is not retried."""
    b = Book(cfg, journal)
    armed_ticket(b, gtt="B1")
    broker = _Broker(holdings=[], gtts=[])

    report = reconcile(b, broker, today=TODAY)

    assert book_mod.F_GTT_VANISHED in {f.kind for f in report.findings}


def test_a_stop_that_drifted_from_the_ladder_is_reported(cfg, journal):
    b = Book(cfg, journal)
    t = open_ticket(b, _Pick(entry=1000.0, stop=950.0, quantity=20))
    t.ladder.stop_r = 0.0                       # the ladder moved to breakeven
    broker = _Broker(holdings=[holding()],
                     gtts=[exit_gtt(stop=950.0)])   # Zerodha still has the old one

    report = reconcile(b, broker, today=TODAY)

    assert book_mod.F_STOP_DRIFTED in {f.kind for f in report.findings}


def test_a_quantity_mismatch_is_reported(cfg, journal):
    b = Book(cfg, journal)
    open_ticket(b, _Pick(quantity=20))
    broker = _Broker(holdings=[holding(qty=12.0)], gtts=[exit_gtt()])

    report = reconcile(b, broker, today=TODAY)

    assert book_mod.F_QTY_MISMATCH in {f.kind for f in report.findings}


def test_a_position_sold_outside_the_app_is_reported(cfg, journal):
    b = Book(cfg, journal)
    open_ticket(b)
    broker = _Broker(holdings=[], gtts=[])

    report = reconcile(b, broker, today=TODAY)

    assert book_mod.F_CLOSED_ELSEWHERE in {f.kind for f in report.findings}


def test_a_holding_the_book_never_bought_is_surfaced(cfg, journal):
    b = Book(cfg, journal)
    broker = _Broker(holdings=[holding(symbol="TCS", qty=5.0)])

    report = reconcile(b, broker, today=TODAY)

    assert book_mod.F_UNTRACKED_HOLDING in {f.kind for f in report.findings}


def test_an_expired_unfilled_ticket_is_surfaced(cfg, journal):
    b = Book(cfg, journal)
    armed_ticket(b)
    resting_buy = GttView(trigger_id="B1", symbol="INFY", exchange="NSE",
                          status="active", trigger_type="single",
                          trigger_values=[1000.0], quantity=20,
                          transaction_type="BUY")
    broker = _Broker(holdings=[], gtts=[resting_buy])

    report = reconcile(b, broker, today=TODAY + timedelta(days=9))

    assert book_mod.F_EXPIRED in {f.kind for f in report.findings}


def test_an_unreachable_broker_is_not_a_clean_bill_of_health(cfg, journal):
    """
    "Nothing disagreed" and "nothing was checked" must never read the same.

    This is the same distinction `news.available` draws, and for the same
    reason: silence is not evidence.
    """
    b = Book(cfg, journal)
    open_ticket(b)
    broker = _Broker(protection=("protected", "ok"))

    report = reconcile(b, broker, today=TODAY, broker_reachable=False)

    assert not report.broker_reachable
    assert "NOT a clean bill of health" in report.headline()


def test_a_matching_book_and_broker_produce_no_findings(cfg, journal):
    b = Book(cfg, journal)
    open_ticket(b, _Pick(entry=1000.0, stop=950.0, target=1150.0, quantity=20),
                gtt="G1")
    broker = _Broker(holdings=[holding()],
                     gtts=[exit_gtt(trigger_id="G1", stop=950.0, target=1150.0)])

    report = reconcile(b, broker, today=TODAY)

    assert report.clean, [f.message for f in report.findings]


# ---------------------------------------------------------------- performance

def test_performance_counts_only_trades_that_were_actually_taken(cfg, journal):
    """
    The distinction from `tracker.py`, which replays every pick including the
    ones you skipped. This one can only see money that moved.
    """
    b = Book(cfg, journal)
    # One share, so +2R is a full exit rather than a partial with a runner.
    won = open_ticket(b, _Pick(symbol="AAA", entry=1000.0, stop=950.0,
                               target=1150.0, quantity=1), gtt="G1")
    b.advance(won, high=1160.0, low=1005.0, close=1155.0, atr=20.0, today=TODAY)
    assert won.state is TicketState.CLOSED
    lost = open_ticket(b, _Pick(symbol="BBB", entry=1000.0, stop=950.0,
                                quantity=20), gtt="G2")
    b.advance(lost, high=1010.0, low=940.0, close=945.0, atr=20.0, today=TODAY)
    armed_ticket(b, _Pick(symbol="CCC"), gtt="B3")      # never filled

    p = performance(b)

    assert p.trades == 2                    # the armed one is not a trade
    assert p.wins == 1
    assert p.win_rate == pytest.approx(0.5)
    assert p.best_r == pytest.approx(2.0)
    assert p.worst_r == pytest.approx(-1.0)
    # 2R won, 1R lost -> +0.5R a trade, comfortably above the 33% breakeven
    # a 2:1 ratio implies.
    assert p.expectancy_r == pytest.approx(0.5)

    # TWO closed trades cannot measure a payoff, so the headline falls back to
    # the DESIGN figure and says so rather than quoting a two-trade average as
    # if it were the book's real breakeven.
    line = p.headline()
    assert "above the 33% design breakeven" in line
    assert "not a measurement" in line


def test_the_yardstick_switches_to_the_realised_payoff_once_there_are_enough(
        cfg, journal):
    """
    The number a book is judged against comes from the trades it took.

    A ladder that shifts to breakeven at +1R does not produce 2:1 trades, so
    `1/(1+R:R)` is the intention and not the yardstick. Below
    MIN_TRADES_FOR_REALISED_BREAKEVEN there is no measurable payoff and the
    design figure is shown, labelled; at or above it, the realised one is.
    """
    from nifty_algo.swing.book import (MIN_TRADES_FOR_REALISED_BREAKEVEN,
                                       Performance)

    # Three wins of +2R against seven losses of -0.5R: the ladder's shape.
    # Realised breakeven is 0.5 / (2.0 + 0.5) = 20%, and 30% clears it - the
    # same trades against the 33% design figure would read as failing.
    p = Performance(trades=10, wins=3, total_r=2.5,
                    avg_win_r=2.0, avg_loss_r=0.5)
    assert p.trades >= MIN_TRADES_FOR_REALISED_BREAKEVEN
    rate, basis = p.yardstick(1 / 3)
    assert basis == "realised"
    assert rate == pytest.approx(0.2)
    assert "above the 20% realised breakeven" in p.headline(1 / 3)

    # One trade short, the same payoff is not yet evidence of a payoff.
    q = Performance(trades=MIN_TRADES_FOR_REALISED_BREAKEVEN - 1, wins=3,
                    total_r=2.5, avg_win_r=2.0, avg_loss_r=0.5)
    assert q.yardstick(1 / 3) == (1 / 3, "design")
