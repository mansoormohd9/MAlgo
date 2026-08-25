"""
The Settings page — the capital boxes in particular.

WHY THIS FILE EXISTS. `page_settings.py` had no coverage at all, and the first
number anyone typed into the swing-capital box took the whole app down with a
ZeroDivisionError. The page-walk smoke check rendered Settings at its defaults,
where the pot is 0 and `_pot_note` returns at its guard one line ABOVE the
division. The only path never exercised was the one a user takes first.

The bug itself was a source-of-truth split: `_capital()` defers writing the
typed value onto `cfg` until Save is pressed, but `_pot_note()` was still
reading the SAVED pot - so the guard tested one number and the arithmetic
divided by another. That is why the tests below assert on the note's CONTENT
and not merely on "it did not raise": a note quoting risk against the wrong
balance is the same bug in its non-crashing form, and would have shipped
again.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import seed_offline_broker
from streamlit.testing.v1 import AppTest

from nifty_algo import settings_store
from nifty_algo.config import Config
from nifty_algo.ui import page_settings

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _fresh() -> Config:
    """A config with nothing saved - the state a first-time user is in."""
    cfg = Config()
    assert cfg.capital.swing_capital_inr == 0.0
    return cfg


def _open(cfg) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["cfg"] = cfg
    seed_offline_broker(at, cfg)      # never a real broker - see conftest
    at.run()
    assert not at.exception, _why(at)
    at.sidebar.radio[0].set_value("Settings").run()
    assert not at.exception, _why(at)
    return at


def _why(at) -> str:
    return "; ".join(f"{type(e.value).__name__}: {e.value}"
                     for e in (at.exception or []))


def _captions(at) -> str:
    return " ".join(c.value for c in at.caption)


def _note(at) -> str:
    """Just the pot note, not the sidebar - both quote a risk-per-trade."""
    return " ".join(c.value for c in at.caption
                    if "risk per trade" in c.value)


def _table(at) -> str:
    """The stop/deployed/tickets table the note renders beneath itself."""
    return " ".join(m.value for m in at.markdown if "Deployed per" in m.value)


# ---------------------------------------------------------------- the crash

def test_typing_a_pot_with_nothing_saved_does_not_crash():
    """
    THE regression. Open Settings on a fresh config and type a number.

    `swing` (typed) was 30,000 so the `swing <= 0` guard passed, while `risk`
    came from the SAVED pot and was therefore 0 - making `per_position` 0 and
    the next line a division by zero.
    """
    at = _open(_fresh())

    at.number_input(key="cap_swing").set_value(30_000.0).run()

    assert not at.exception, _why(at)


def test_the_note_prices_the_pot_you_typed_not_the_one_you_saved():
    """
    The non-crashing half of the same bug, and the reason it was worth a test
    rather than a one-line guard.

    Save Rs 1,00,000, then type Rs 30,000: reading `cfg` would quote Rs 1,667
    of risk - 1.67% of the SAVED pot - against a Rs 30,000 balance.
    """
    cfg = _fresh()
    cfg.capital.swing_capital_inr = 100_000.0        # already saved
    at = _open(cfg)

    at.number_input(key="cap_swing").set_value(30_000.0).run()

    assert not at.exception, _why(at)
    # Scoped to the note. The SIDEBAR also prints Rs 1,667 - that one is the
    # option book's risk off its own Rs 1,00,000 pot, and is correct there.
    note = _note(at)
    assert "₹500 of risk per trade" in note          # 1.67% of 30,000
    assert "₹1,667" not in note                      # ...not of 100,000


def test_the_note_reproduces_the_worked_example():
    """
    Rs 30,000 -> Rs 500 risk -> about Rs 10,000 per position, three of them.

    The arithmetic the whole swing book was sized around, surfaced where you
    choose the pot.
    """
    at = _open(_fresh())
    at.number_input(key="cap_swing").set_value(30_000.0).run()

    assert "₹500 of risk per trade" in _note(at)
    # A 5% stop on Rs 30,000: Rs 10,000 a position, three of them.
    assert "₹10,000" in _table(at) and "all 3" in _table(at)


def test_the_ticket_count_depends_on_the_stop_and_not_on_the_pot():
    """
    The finding that rewrote this note.

    Cash per position is `risk / stop%`, and risk is itself a fixed fraction
    of the pot - so the number of positions a pot funds is `stop% / risk%`
    and the pot CANCELS OUT. Rs 30,000 and Rs 5,00,000 both fund 3.0
    positions at a 5% stop and 1.8 at a 3% stop.

    The original note asked "is the pot big enough for `top_n`?", which reads
    as a question about the pot and is not one.
    """
    small = _open(_fresh())
    small.number_input(key="cap_swing").set_value(30_000.0).run()
    large = _open(_fresh())
    large.number_input(key="cap_swing").set_value(500_000.0).run()

    for at in (small, large):
        assert not at.exception, _why(at)
        table = _table(at)
        assert "1.8 of 3" in table          # 3% stop, whatever the pot
        assert "all 3" in table             # 5% stop, whatever the pot
        assert "refused for want of cash" in table

    # ...and the RUPEE figures do scale with the pot.
    assert "₹10,000" in _table(small)      # 5% stop on Rs 30,000
    assert "₹166,667" in _table(large)     # 5% stop on Rs 5,00,000


# ---------------------------------------------------------------- the guard

def test_a_zero_risk_percentage_does_not_divide_by_zero():
    """
    `risk_per_trade_pct` is `session_stop_pct / max_entries_per_session`. Set
    the stop to zero and every ticket sizes to nothing - a settings page is a
    poor place to discover that through a traceback.

    Writing this found a SECOND divide-by-zero, in `reward_risk_ratio`, which
    `app.py` prints in the sidebar on every render - so one bad config value
    took down every page rather than just this one. Both are guarded now.
    """
    cfg = _fresh()
    cfg.capital.session_stop_pct = 0.0
    from nifty_algo.ui.theme import get_palette

    page_settings._pot_note(cfg, 30_000.0, get_palette())   # must not raise
    assert cfg.capital.reward_risk_ratio == 0.0             # not a crash


def test_pot_note_is_safe_for_any_pot_size():
    """
    Called directly, so this stays fast and covers the numeric edges rather
    than the render path.
    """
    cfg = _fresh()
    palette = __import__("nifty_algo.ui.theme", fromlist=["get_palette"]).get_palette()
    for pot in (0.0, 0.01, 1.0, 999.0, 12_000.0, 30_000.0, 1e9):
        page_settings._pot_note(cfg, pot, palette)   # must not raise


# ---------------------------------------------------------------- persistence

def test_typing_does_not_persist_until_save_is_pressed(monkeypatch, tmp_path):
    """
    The behaviour the deferred assignment exists for: `save_settings()`
    serialises the WHOLE config, so any other button that saves would
    otherwise commit a number you were halfway through typing.
    """
    saves: list = []
    monkeypatch.setattr(page_settings, "save_settings",
                        lambda: saves.append(True))

    cfg = _fresh()
    at = _open(cfg)
    at.number_input(key="cap_swing").set_value(30_000.0).run()

    assert cfg.capital.swing_capital_inr == 0.0      # not committed
    assert not saves
    assert "Unsaved changes" in _captions(at)


def test_pressing_save_commits_all_three_pots(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(page_settings, "save_settings",
                        lambda: settings_store.save(cfg, path))

    cfg = _fresh()
    at = _open(cfg)
    at.number_input(key="cap_option").set_value(250_000.0).run()
    at.number_input(key="cap_swing").set_value(30_000.0).run()
    at.number_input(key="cap_foreign").set_value(800_000.0).run()
    at.button(key="save_capital").click().run()

    assert not at.exception, _why(at)
    assert cfg.capital.swing_capital_inr == 30_000.0

    restored = Config()
    settings_store.apply_to(restored, path)
    assert restored.capital.starting_capital == 250_000.0
    assert restored.capital.swing_capital_inr == 30_000.0
    assert restored.capital.foreign_capital_inr == 800_000.0


# ---------------------------------------------------------------- the page

def test_the_page_renders_at_its_defaults():
    """The happy path the smoke check already covered, kept explicit."""
    at = _open(_fresh())

    assert not at.exception, _why(at)
    assert at.title[0].value == "Settings"
    # An unfunded pot must say it will stand the scan down, not stay silent.
    assert "stand" in " ".join(m.value for m in at.markdown).lower()


def test_the_live_order_switch_is_off_and_needs_typed_confirmation():
    """
    The one thing between this app and real money should not be a stray click.
    """
    cfg = _fresh()
    at = _open(cfg)

    assert cfg.equity_broker.dry_run is True
    assert not [b for b in at.button if b.key == "enable_live"]

    at.text_input(key="go_live_confirm").set_value("GO LIVE").run()
    assert [b for b in at.button if b.key == "enable_live"]
    assert cfg.equity_broker.dry_run is True         # still not flipped
