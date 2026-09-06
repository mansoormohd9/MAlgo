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


@pytest.fixture(autouse=True)
def _restore_config():
    """
    Put the shared config back after every test in this file.

    `ui.state.get_config()` returns the module-level `DEFAULT` itself, so a
    control on this page writes to the object every other test reads. Leaving
    `universe="nifty500"` behind would make some later test fail for a reason
    it has nothing to do with - which is precisely the failure mode
    `test_auth.py` already inflicts on `test_portfolio_connectors.py` through
    `os.environ`. One leak of that kind in the suite is one too many.
    """
    from nifty_algo.config import DEFAULT
    f, c = DEFAULT.factor, DEFAULT.capital
    saved = (f.universe, f.halal_screened, f.regime_ma_days,
             c.factor_capital_inr)
    yield
    (f.universe, f.halal_screened, f.regime_ma_days,
     c.factor_capital_inr) = saved


@pytest.fixture(autouse=True)
def _no_real_settings_writes(monkeypatch):
    """
    A page test must never touch `data/settings.json`.

    `_run` persists the configuration that produced a scan, which is right for
    a console opened once a month and catastrophic in a test: the first run of
    these tests wrote `factor_universe: nifty500` and a Rs 5,00,000 pot into
    the real file, and every later `get_config()` then applied them - which is
    also how three of these tests started failing for a reason that had
    nothing to do with what they assert.

    `settings_store.save` binds `DEFAULT_PATH` as a default argument at import,
    so patching the module constant would not redirect it. The seam that works
    is the page's own reference.
    """
    from nifty_algo.ui import page_sleeve
    monkeypatch.setattr(page_sleeve, "save_settings", lambda: None)


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


def test_the_record_follows_the_selected_universe(page):
    """
    THE POINT OF THIS PAGE'S RECORD. Selecting a restricted universe must
    change the figures on screen to that universe's own, or the page is
    quoting a book nobody is trading - the failure it was built to prevent,
    in a new costume.
    """
    unrestricted = _text(page)
    assert "+18.79%" in unrestricted

    page.selectbox[0].set_value("nifty500").run()
    restricted = _text(page)
    assert "+17.60%" in restricted          # the size-ranked control's number
    assert "+18.79%" not in restricted      # never the other universe's


def test_the_inflated_figure_appears_once_and_is_labelled(page):
    """
    +31.29% is not a caveat on the expectation, it is the wrong number to plan
    with. It is on the page so a reader who runs the backtest can see why it
    disagrees - in the caption, never in the table, and never alone.
    """
    page.selectbox[0].set_value("nifty500").run()
    captions = " ".join(str(getattr(c, "value", "")) for c in page.caption)
    tables = " ".join(df.value.to_string() for df in page.dataframe)

    assert "+31.29%" in captions
    assert "+31.29%" not in tables
    assert "look-ahead" in captions
    assert _text(page).count("+31.29%") == 1


def test_a_nifty100_mandate_is_told_not_to_run_it(page):
    """
    Against its control a Nifty 100 restriction costs 10pp and lands below the
    index. The page has to say that where the number is, not in a doc.
    """
    page.selectbox[0].set_value("nifty100").run()
    body = _text(page)
    assert "+7.94%" in body
    assert "DO NOT RUN THE SLEEVE HERE" in body


def test_an_unmeasured_universe_refuses_rather_than_borrowing(page,
                                                              monkeypatch):
    """
    Adding a universe without measuring it must make the page say so. The
    alternative - silently showing another universe's numbers - is the exact
    thing the per-universe record exists to stop.
    """
    monkeypatch.setitem(sl.RECORDS, "nifty500", None)
    monkeypatch.delitem(sl.RECORDS, "nifty500")
    page.selectbox[0].set_value("nifty500").run()
    body = _text(page)
    assert "No backtest describes" in body
    assert "+18.79%" not in body
    assert "+17.60%" not in body


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


def test_the_universe_selector_reaches_the_scan(page, monkeypatch):
    """
    The selector is the control the mandate runs through, so it has to arrive
    at `eligible_at` rather than merely render. Stubbing the resolver proves
    the page's choice is what the scan restricts on - and that a restricted
    book actually comes back smaller.
    """
    from nifty_algo.factor import restriction as restr

    seen = {}

    def _fake(cfg, key, bars, universe, root="."):
        seen["key"] = key
        return restr.static({"WIN04", "WIN05", "WIN06"}), f"stubbed {key}"

    monkeypatch.setattr(restr, "resolver", _fake)
    page.selectbox[0].set_value("nifty500").run()
    page.number_input[0].set_value(500_000.0).run()
    next(b for b in page.button if "Run scan" in b.label).click().run()
    assert not page.exception

    assert seen["key"] == "nifty500"
    scan = page.session_state["factor_sleeve_scan"]
    assert scan.universe_key == "nifty500"
    assert {p.symbol for p in scan.picks} <= {"WIN04", "WIN05", "WIN06"}
    assert "stubbed" in _text(page)


def test_the_micro_cap_warning_is_suppressed_once_restricted(page, monkeypatch):
    """
    "17 of 20 picks sit outside the Nifty 500" is the right warning for the
    unrestricted book and a lie once a universe key is on - the picks are
    inside it by construction.
    """
    from nifty_algo.factor import restriction as restr
    monkeypatch.setattr(
        restr, "resolver",
        lambda cfg, key, bars, uni, root=".": (
            restr.static({"WIN04", "WIN05", "WIN06"}), "stubbed"))
    page.selectbox[0].set_value("nifty500").run()
    page.number_input[0].set_value(500_000.0).run()
    next(b for b in page.button if "Run scan" in b.label).click().run()
    assert not page.exception
    assert "sit outside the Nifty 500" not in _text(page)
