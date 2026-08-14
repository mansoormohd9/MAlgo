"""
The ticket has to say which of three things it is.

"A strategy fired, take this" and "nothing fired, but here is what the engine
would buy" are different claims that used to render identically - a strike, a
premium and a target, with nothing distinguishing a signal from a hypothetical.
That ambiguity is the whole reason this component exists, so the three states
are worth a test rather than an eyeball.
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _render(with_alert: bool, permitted: bool = True):
    """Render one ticket in isolation and hand back its markdown."""
    def script():
        import streamlit as st
        from datetime import date, datetime

        from nifty_algo.alerts.base import AlertKind, TradeAlert
        from nifty_algo.brief import build_chain_view
        from nifty_algo.config import Config
        from nifty_algo.data.chain import ChainProvider
        from nifty_algo.risk import RiskEngine
        from nifty_algo.ui.entry_radar import entry_ticket
        from nifty_algo.ui.theme import get_palette

        cfg = Config()
        view = build_chain_view(
            26_000.0, "CE", 30.0, cfg,
            chain_provider=ChainProvider(cfg), risk=RiskEngine(cfg),
            today=date(2026, 3, 10),
            entry_permitted=st.session_state.permitted,
            halt_reason="" if st.session_state.permitted else "outside_entry_window",
        )
        alert = None
        if st.session_state.with_alert:
            alert = TradeAlert(
                kind=AlertKind.ENTRY, timestamp=datetime(2026, 3, 10, 10, 35),
                strategy_key="level_break", strategy_label="Level break",
                direction="long", option_type="CE", confidence=0.72,
            )
        entry_ticket(view, get_palette(), cfg, live_alert=alert)

    at = AppTest.from_function(script, default_timeout=120)
    at.session_state.with_alert = with_alert
    at.session_state.permitted = permitted
    at.run()
    assert not at.exception, [repr(e.value) for e in at.exception]
    return "\n".join(m.value for m in at.markdown)


def test_a_fired_signal_is_labelled_actionable():
    out = _render(with_alert=True)
    assert "ACTIONABLE" in out
    assert "Level break fired at 10:35" in out
    assert "confidence 0.72" in out
    assert "Place order" in out, "it must say where the button is"
    assert "REFERENCE ONLY" not in out


def test_no_signal_is_labelled_reference_only():
    out = _render(with_alert=False)
    assert "REFERENCE ONLY" in out
    assert "No strategy has fired" in out
    assert "ACTIONABLE" not in out


def test_a_blocked_session_outranks_a_fired_signal():
    """
    A parked order from earlier does not become enterable because it is still
    on screen. If the day rules have closed, the ticket says so and nothing
    else claims otherwise.
    """
    out = _render(with_alert=True, permitted=False)
    assert "BLOCKED" in out
    assert "outside entry window" in out
    assert "ACTIONABLE" not in out


def test_the_ticket_states_the_full_order_not_just_a_strike():
    out = _render(with_alert=False)
    for field in ("Entry", "Target", "Stop", "Quantity", "Risk", "Reward",
                  "Outlay", "Delta", "Spread", "Open interest"):
        assert field in out, f"{field} missing from the ticket"
    assert "BUY 2 lot(s) of CE strike" in out
    assert "before you are square" in out, "costs must be stated, not implied"
