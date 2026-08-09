"""Shared rendering pieces: banners, the session header, and the alert card."""
from __future__ import annotations
from datetime import datetime

import streamlit as st

from .theme import Palette
from ..alerts.base import TradeAlert, AlertKind


def banner(text: str, accent: str, icon: str = "") -> None:
    """
    A status strip. Always ships an icon plus words - status colour never
    carries the meaning on its own.
    """
    st.markdown(
        f'<div class="banner" style="--banner-accent:{accent}">'
        f'{icon + " " if icon else ""}{text}</div>',
        unsafe_allow_html=True,
    )


def feed_banner(engine, p: Palette) -> None:
    """
    Permanent provenance strip. This is not decoration: acting on a
    15-minute-delayed price with a 1-ATR stop can mean the level you are
    looking at was breached and reversed before you saw it.
    """
    if engine is None:
        return
    feed = engine.feed
    if feed.is_delayed:
        banner(f"<b>{feed.name.upper()} — {feed.latency_note}</b>",
               p.critical, "⚠")
    else:
        banner(f"<b>{feed.name.upper()}</b> — {feed.latency_note}",
               p.series_1, "ℹ")


def session_header(engine, p: Palette) -> None:
    """Where the session stands against the three governors."""
    risk = engine.risk
    cap = engine.cfg.capital
    s = risk.session

    target = risk.capital * cap.session_target_pct
    stop = -abs(risk.capital * cap.session_stop_pct)
    pnl = s.realised_pnl

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital", f"₹{risk.capital:,.0f}")
    c2.metric("Session P&L", f"₹{pnl:+,.0f}",
              f"target ₹{target:,.0f} / stop ₹{stop:,.0f}",
              delta_color="off")
    c3.metric("Entries used", f"{s.entries_taken} / {cap.max_entries_per_session}")
    c4.metric("Risk / trade", f"₹{cap.risk_per_trade_rupees:,.0f}",
              f"R:R {cap.reward_risk_ratio:.1f}:1", delta_color="off")

    regime = engine.state.regime
    c5.metric("Regime", regime.regime.value if regime else "—",
              regime.detail[:28] if regime else "", delta_color="off")

    if engine.state.kill_switch:
        banner(f"<b>KILL SWITCH TRIPPED</b> — no alerts until re-armed. "
               f"{engine.state.last_error or ''}", p.critical, "⛔")
    elif s.halted:
        banner(f"<b>Session halted</b> — "
               f"{s.halt_reason.value.replace('_', ' ')}. "
               f"No further entries today.", p.warning, "⏸")
    elif engine.state.halt_reason != "none":
        banner(f"Entries blocked — "
               f"{engine.state.halt_reason.replace('_', ' ')}.", p.warning, "⏸")


def alert_card(alert: TradeAlert, p: Palette, key: str,
               on_paper_fill=None) -> None:
    """
    One tradeable setup, fully specified.

    Everything needed to act is on the card, so nothing has to be calculated
    under time pressure: strike, entry, target, stop, quantity, and the rupee
    risk and reward those imply.
    """
    if alert.kind is not AlertKind.ENTRY:
        accent = {AlertKind.KILL_SWITCH: p.critical,
                  AlertKind.FORCE_EXIT: p.warning,
                  AlertKind.HALT: p.warning}.get(alert.kind, p.series_1)
        banner(f"<b>{alert.title()}</b> — {alert.message}", accent, "●")
        return

    accent = p.good if alert.direction == "long" else p.critical
    arrow = "▲" if alert.direction == "long" else "▼"
    rupee = "₹"
    dot = "·"
    dash = "—"

    st.markdown(f"""
<div class="alert-card" style="--alert-accent:{accent}">
  <h4>{arrow} {alert.direction.upper()} {alert.strike}{alert.option_type}
      <span class="pill">{alert.strategy_label}</span>
      <span class="pill">conf {alert.confidence:.2f}</span>
      <span class="pill">{alert.regime}</span></h4>
  <div class="alert-grid">
    <div><span class="lbl">Entry</span><span class="val">{alert.entry_premium:,.2f}</span></div>
    <div><span class="lbl">Target</span><span class="val" style="color:{p.good}">{alert.target_premium:,.2f}</span></div>
    <div><span class="lbl">Stop</span><span class="val" style="color:{p.critical}">{alert.stop_premium:,.2f}</span></div>
    <div><span class="lbl">Qty</span><span class="val">{alert.quantity} ({alert.lots} lot)</span></div>
    <div><span class="lbl">Risk</span><span class="val">{rupee}{alert.rupee_risk:,.0f}</span></div>
    <div><span class="lbl">Reward</span><span class="val">{rupee}{alert.rupee_reward:,.0f}</span></div>
    <div><span class="lbl">R:R</span><span class="val">{alert.reward_risk:.2f}</span></div>
    <div><span class="lbl">Delta</span><span class="val">{alert.delta:.2f}</span></div>
    <div><span class="lbl">Underlying</span><span class="val">{alert.underlying_price:,.1f}</span></div>
    <div><span class="lbl">U/L stop</span><span class="val">{alert.underlying_stop_points:.0f} pts</span></div>
  </div>
  <div class="alert-why"><b>Why:</b> {alert.reason}</div>
  <div class="alert-why">{alert.timestamp:%Y-%m-%d %H:%M} &nbsp;{dot}&nbsp;
       expiry {alert.expiry or dash} &nbsp;{dot}&nbsp; chain: {alert.chain_source}</div>
</div>
""", unsafe_allow_html=True)

    if alert.chain_note:
        banner(alert.chain_note, p.warning, "⚠")
    if alert.feed_latency_note:
        banner(alert.feed_latency_note, p.critical, "⚠")

    if on_paper_fill:
        c1, c2, _ = st.columns([1, 1, 3])
        fill = c1.number_input("Paper fill premium",
                               value=float(alert.entry_premium),
                               step=0.05, key=f"fill_{key}",
                               label_visibility="collapsed")
        if c2.button("Log paper fill", key=f"btn_{key}", width="stretch"):
            on_paper_fill(alert, fill)
            st.success("Logged to the journal.")


def evaluations_table(evaluations: list[dict]) -> None:
    """
    Why each strategy did or did not fire this bar.

    Worth as much as the alerts themselves. "Nothing fired" and "everything
    fired but risk rejected every strike" look identical from the outside and
    mean completely different things.
    """
    if not evaluations:
        st.caption("No strategies evaluated on this bar.")
        return
    import pandas as pd
    df = pd.DataFrame(evaluations)
    df.columns = [c.title() for c in df.columns]
    st.dataframe(df, width="stretch", hide_index=True)
