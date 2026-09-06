"""
The Monthly sleeve page, rendered with a book actually on screen.

Same reasoning as `test_ui_page_swing.py`: `nifty_algo/ui/` decides nothing, so
its logic lives in `factor/sleeve.py` and is tested headlessly - which leaves a
hole exactly where a page raises on a line only reached once there is something
to draw. An empty sleeve returns early from almost every panel, so rendering it
empty proves close to nothing.

The assertions beyond `not at.exception` are about the two things this page is
allowed to get wrong only once: it must never present a provisional diff as a
decision, and it must never show the good decade without the bad one.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import sign_in
from streamlit.testing.v1 import AppTest

from nifty_algo.factor import sleeve as sl

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _series(start_price, n, drift, seed, start=date(2020, 1, 1)):
    rng = np.random.default_rng(seed)
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    close = start_price * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, 1e6)},
        index=pd.to_datetime(days))


def _world():
    return {f"WIN{i:02d}": _series(100.0 + i, 700, 0.0012 - i * 0.0003, i + 1)
            for i in range(12)}


@pytest.fixture
def page(monkeypatch):
    """The page with bars stubbed, holdings unread, and the pot funded."""
    world = _world()
    monkeypatch.setattr(sl, "load_bars", lambda cfg: (world, None))

    at = AppTest.from_file(APP, default_timeout=180)
    sign_in(at)
    at.run()
    at.sidebar.radio[0].set_value("Monthly sleeve").run()
    return at


def test_the_page_renders_before_any_scan(page):
    assert not page.exception
    assert any("Run scan" in b.label for b in page.button)


def _text(at) -> str:
    """Everything the page actually rendered, as one searchable blob."""
    parts = [str(e.value) for e in at.markdown]
    parts += [str(getattr(e, "value", "")) for e in at.caption]
    parts += [str(getattr(e, "value", "")) for e in at.info]
    parts += [str(getattr(e, "value", "")) for e in at.warning]
    for df in at.dataframe:
        parts.append(df.value.to_string())
    return " ".join(parts)


def test_both_windows_are_on_screen_before_anything_else(page):
    """
    A console that shows +18.79% without +0.60pp beside it is an
    advertisement. The record renders whether or not a scan has been run.
    """
    body = _text(page)
    assert "+7.94pp" in body
    assert "+0.60pp" in body
    assert "78.5%" in body


def test_a_scan_renders_a_book_and_stays_provisional(page):
    """
    The fixture's bars end well before today, so the page is never on a
    rebalance date - which is precisely the state it must refuse to arm.
    """
    page.number_input[0].set_value(500_000.0).run()
    next(b for b in page.button if "Run scan" in b.label).click().run()
    assert not page.exception

    scan = page.session_state["factor_sleeve_scan"]
    assert scan.picks
    assert scan.is_rebalance_day is False

    actions = page.session_state["factor_sleeve_actions"]
    assert actions and all(a.provisional for a in actions)
    assert "rovisional" in _text(page)


def test_a_provisional_call_offers_no_download(page):
    """
    The checklist is the artefact you act on, so it may not exist on a day the
    book is not supposed to trade. A page that let you export a provisional
    list has handed you a monthly book to trade daily.
    """
    page.number_input[0].set_value(500_000.0).run()
    next(b for b in page.button if "Run scan" in b.label).click().run()
    assert not page.exception
    assert not [b for b in page.button if "checklist" in b.label.lower()]


def test_an_unfunded_pot_renders_and_says_so(page):
    page.number_input[0].set_value(0.0).run()
    next(b for b in page.button if "Run scan" in b.label).click().run()
    assert not page.exception
    scan = page.session_state["factor_sleeve_scan"]
    assert scan.funded is False
    assert "pot is zero" in _text(page).lower()


def test_the_page_never_offers_a_stop_loss_control(page):
    """
    F2 measured a per-name stop across two widths, two bases and two decades.
    Offering the control anyway would let the page quietly contradict the
    result it is built on.
    """
    page.number_input[0].set_value(500_000.0).run()
    next(b for b in page.button if "Run scan" in b.label).click().run()
    assert not page.exception
    labels = " ".join(getattr(w, "label", "") or "" for w in
                      list(page.number_input) + list(page.slider) +
                      list(page.toggle) + list(page.selectbox)).lower()
    assert "stop" not in labels
