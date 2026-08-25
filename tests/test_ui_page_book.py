"""
The Trade book page, rendered with positions actually on screen.

Same reasoning as [test_ui_page_swing.py](test_ui_page_swing.py): `nifty_algo/ui/`
decides nothing, so its logic is tested headlessly - which leaves a hole
exactly where a page raises on a line only reached once there is something to
draw. An empty Trade book returns early from almost every panel, so a test
that renders it empty proves close to nothing.

These seed a real `Book` - built by replaying real journal events, not by
hand-assembling dataclasses - and then render. `not at.exception` is most of
the point.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from conftest import daily_bars, seed_offline_broker, trending
from streamlit.testing.v1 import AppTest

from nifty_algo.config import Config
from nifty_algo.journal import Journal
from nifty_algo.swing.book import Book

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TODAY = date(2026, 8, 24)


class _Setup:
    def __init__(self, entry, stop, target):
        self.entry, self.stop, self.target, self.key = entry, stop, target, "breakout"


class _Pick:
    def __init__(self, symbol, entry=1000.0, stop=950.0, target=1150.0, qty=20):
        self.symbol, self.name = symbol, f"{symbol} Ltd"
        self.sector, self.market = "Information Technology", "india"
        self.setup = _Setup(entry, stop, target)
        self.quantity = qty
        self.risk_inr = (entry - stop) * qty
        self.reward_inr = (target - entry) * qty
        self.deployed_inr = entry * qty
        self.scanned_on = TODAY
        self.valid_until = TODAY + timedelta(days=5)


def _world(tmp_path, *, with_open=True, with_armed=True, with_closed=True):
    """A journal holding real events, and the bars the page reads prices from."""
    cfg = Config()
    cfg.capital.swing_capital_inr = 100_000.0
    journal = Journal(tmp_path / "journal")
    book = Book(cfg, journal)

    bars = {}
    if with_open:
        t = book.arm(_Pick("INFY"), buy_gtt_id="B1", today=TODAY)
        book.record_fill(t, quantity=20, price=1004.0, on=TODAY)
        book.record_exit_gtt(t, "G1", today=TODAY)
        bars["INFY"] = daily_bars(trending(seed=3, start_price=1000.0))
    if with_armed:
        book.arm(_Pick("TCS", entry=3500.0, stop=3350.0, target=3900.0, qty=6),
                 buy_gtt_id="B2", today=TODAY)
        bars["TCS"] = daily_bars(trending(seed=5, start_price=3400.0))
    if with_closed:
        w = book.arm(_Pick("WIPRO", entry=500.0, stop=475.0, target=600.0,
                           qty=1), buy_gtt_id="B3", today=TODAY)
        book.record_fill(w, quantity=1, price=500.0, on=TODAY)
        book.record_exit_gtt(w, "G3", today=TODAY)
        # One clean bar to +2R: a single share cannot split, so this closes.
        book.advance(w, high=560.0, low=505.0, close=555.0, atr=10.0,
                     today=TODAY + timedelta(days=1))
        bars["WIPRO"] = daily_bars(trending(seed=7, start_price=520.0))

    return cfg, journal, bars


def _render(cfg, journal, bars):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["cfg"] = cfg
    at.session_state["journal"] = journal
    at.session_state["swing_bars_india"] = bars
    # Never a real broker. See `conftest.seed_offline_broker` - without this
    # these tests read the repo owner's live Zerodha account and started
    # failing the day their balance went negative.
    seed_offline_broker(at, cfg, journal)
    at.run()
    assert not at.exception, _why(at)
    at.sidebar.radio[0].set_value("Trade book").run()
    return at


def _why(at) -> str:
    return "; ".join(str(e.value)[:400] for e in (at.exception or []))


def _body(at) -> str:
    return " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------- rendering

def test_the_page_renders_with_open_armed_and_closed_tickets(tmp_path):
    """
    Every panel below the checklist has to execute: open positions, the armed
    table, the performance block and the closed-trade table.
    """
    cfg, journal, bars = _world(tmp_path)
    at = _render(cfg, journal, bars)

    assert not at.exception, _why(at)
    body = _body(at)
    assert "Open positions" in body
    assert "Armed, waiting to trigger" in body

    # `st.error` is a WIDGET here, not a test failure - the checklist uses it
    # deliberately for "no token" and "not authorised". Asserting `not
    # at.error` would fail on the page working correctly, so pin the content
    # instead: nothing unexpected may appear in red.
    expected = {"No valid token.", "Not authorised today."}
    unexpected = {e.value for e in at.error} - expected
    assert not unexpected, unexpected


def test_an_empty_book_still_renders(tmp_path):
    cfg, journal, bars = _world(tmp_path, with_open=False, with_armed=False,
                                with_closed=False)
    at = _render(cfg, journal, bars)

    assert not at.exception, _why(at)


def test_the_unprotected_warning_appears_with_a_live_position(tmp_path):
    """
    THE banner. Without DDPI and without today's authorisation, a resting
    stop is decorative - and Kite shows it as active either way, so the app
    is the only thing that can say so.
    """
    cfg, journal, bars = _world(tmp_path)
    assert not cfg.equity_broker.ddpi_active          # the shipped default
    at = _render(cfg, journal, bars)

    rendered = " ".join(m.value for m in at.markdown)
    assert "no CDSL authorisation" in rendered or "REJECTED" in rendered


def test_ddpi_removes_the_warning(tmp_path):
    cfg, journal, bars = _world(tmp_path)
    cfg.equity_broker.ddpi_active = True
    at = _render(cfg, journal, bars)

    assert not at.exception, _why(at)
    assert "no CDSL authorisation" not in _body(at)


def test_the_dry_run_mode_is_stated_on_the_page(tmp_path):
    """
    The one thing standing between this page and real money should never
    have to be inferred.
    """
    cfg, journal, bars = _world(tmp_path)
    at = _render(cfg, journal, bars)

    assert "DRY RUN" in _body(at)


def test_open_risk_is_reported_against_the_cap(tmp_path):
    """Portfolio heat, which nothing in the repo showed before."""
    cfg, journal, bars = _world(tmp_path, with_armed=False, with_closed=False)
    at = _render(cfg, journal, bars)

    captions = " ".join(c.value for c in at.caption)
    assert "Open risk" in captions and "cap" in captions
