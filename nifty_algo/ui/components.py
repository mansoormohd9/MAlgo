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
    gov = risk.governor
    s = risk.session

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital", f"₹{risk.capital:,.0f}")
    c2.metric("Session P&L", f"₹{gov.realised_pnl:+,.0f}",
              f"peak ₹{gov.peak_realised_pnl:+,.0f} · "
              f"target ₹{gov.target:,.0f}", delta_color="off")
    # The floor moves during the day, so it gets its own tile rather than
    # sitting in a caption where a change would go unnoticed.
    ratcheted = gov.floor > gov.base_floor
    c3.metric("Day floor" + (" ↑" if ratcheted else ""),
              f"₹{gov.floor:+,.0f}",
              f"{abs(gov.floor_pct_of_capital):.1%} of capital · "
              f"₹{gov.risk_remaining:,.0f} left", delta_color="off")
    c4.metric("Entries used", f"{s.entries_taken} / {cap.max_entries_per_session}",
              f"₹{cap.risk_per_trade_rupees:,.0f} risk each",
              delta_color="off")

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
               on_paper_fill=None, on_confirm=None) -> None:
    """
    One tradeable setup, fully specified.

    Everything needed to act is on the card, so nothing has to be calculated
    under time pressure: strike, entry, target, stop, quantity, and the rupee
    risk and reward those imply.
    """
    if alert.kind is not AlertKind.ENTRY:
        accent = {AlertKind.KILL_SWITCH: p.critical,
                  AlertKind.CLOSE_ALL: p.critical,
                  AlertKind.FORCE_EXIT: p.warning,
                  AlertKind.HALT: p.warning,
                  AlertKind.PARTIAL_EXIT: p.good,
                  AlertKind.EXIT: p.series_1,
                  AlertKind.MANAGE: p.series_1}.get(alert.kind, p.series_1)
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

    sizing = alert.extra.get("sizing_note")
    if sizing:
        # A trade that silently fell back to one lot is a trade whose runner
        # will never fire. That has to be visible on the card, not inferred.
        banner(sizing, p.good if alert.extra.get("runner_enabled") else p.warning,
               "▣" if alert.extra.get("runner_enabled") else "⚠")

    cols = st.columns([1, 1, 1, 2])
    if on_paper_fill:
        fill = cols[0].number_input("Paper fill premium",
                                    value=float(alert.entry_premium),
                                    step=0.05, key=f"fill_{key}",
                                    label_visibility="collapsed")
        if cols[1].button("Log paper fill", key=f"btn_{key}", width="stretch"):
            on_paper_fill(alert, fill)
            st.success("Logged to the journal.")

    if on_confirm:
        if cols[2].button("Place order", key=f"confirm_{key}", type="primary",
                          width="stretch"):
            on_confirm(alert)


def positions_panel(engine, p: Palette) -> None:
    """
    Open positions and where their stops currently sit.

    The stop shown here is the LIVE one - it moves to breakeven at +1R and
    then trails - so this table is the answer to "where am I actually out?",
    which the original entry alert stops being able to answer the moment the
    ladder advances.
    """
    positions = engine.state.open_positions
    st.subheader(f"Open positions ({len(positions)})")

    if not positions:
        st.caption("Flat. Entries are placed from an alert card's "
                   "**Place order** button — nothing is entered automatically.")
        return

    import pandas as pd
    rows = []
    for pos in positions:
        rows.append({
            "Contract": f"{pos.quote.strike}{pos.quote.option_type}",
            "Strategy": pos.strategy_key,
            "Lots left": f"{pos.state.lots_remaining} / {pos.state.lots_total}",
            "Qty": pos.quantity_remaining,
            "Entry": f"{pos.entry_premium:,.2f}",
            "Stop": f"{pos.stop_premium:,.2f}",
            "Stop (R)": f"{pos.state.stop_r:+.2f}",
            "Mode": pos.state.mode.value,
            "Peak (R)": f"{pos.state.peak_r:+.2f}",
            "Banked": f"₹{pos.realised_pnl:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    at_breakeven = sum(1 for x in positions if x.state.stop_r >= 0)
    if at_breakeven:
        banner(f"{at_breakeven} position(s) have their stop at breakeven or "
               f"better — those cannot lose money from here.", p.good, "▣")


def broker_banner(engine, p: Palette) -> None:
    """
    Whether orders are real. Permanent, like the feed banner, and for the same
    reason: this is not something to hold in your head.
    """
    if engine.broker is None:
        banner("<b>ALERTS ONLY</b> — no broker attached. Positions are tracked "
               "and managed, but no order is ever sent.", p.series_1, "ℹ")
    elif engine.cfg.broker.dry_run:
        banner("<b>DRY RUN</b> — orders are logged, not sent. "
               "Set <code>BrokerConfig.dry_run = False</code> to trade live.",
               p.warning, "⚠")
    else:
        banner("<b>LIVE ORDERS</b> — confirmed entries place real trades with "
               "real money.", p.critical, "⛔")


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
