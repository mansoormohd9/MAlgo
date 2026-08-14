"""
Live alerts — the landing page.

Everything that changes on a tick lives in `_body`, which is run inside a
`st.fragment` by `refresh.live()`. That is what lets auto-refresh repaint the
alerts, the radar and the chart without re-executing the sidebar, the page
radio and the whole script every few seconds.
"""
from __future__ import annotations

import streamlit as st

from . import refresh
from .theme import get_palette
from .charts import price_chart, build_overlays
from .components import (feed_banner, session_header, alert_card,
                         evaluations_table, banner, positions_panel,
                         broker_banner)
from .entry_radar import entry_radar
from .state import (get_engine, get_config, get_journal, play_alert_sound,
                    engine_error, sync_ui_alerts, unannounced_alerts,
                    sound_armed)


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Live alerts")

    engine = get_engine()
    if engine is None:
        banner(f"<b>Feed unavailable.</b> {engine_error()}", p.critical, "⛔")
        st.info("Fix the feed on the **Settings** page, or generate sample data:\n\n"
                "```\npython -m nifty_algo.data.sample\n```")
        return

    refresh.live(_body, engine, cfg, p)


def _body(engine, cfg, p) -> None:
    refresh.mark_tick()

    feed_banner(engine, p)
    broker_banner(engine, p)

    _controls(engine, cfg, p)

    # --- run a pass ---
    auto = refresh.interval() is not None
    should_run = auto or st.session_state.pop("force_run", False) \
        or engine.state.last_run is None
    if should_run and not engine.state.kill_switch:
        engine.run_once()

    state = engine.state
    session_header(engine, p)

    if state.last_error:
        banner(f"<b>Engine error:</b> {state.last_error}", p.critical, "⛔")

    st.caption(
        f"Last evaluated {state.last_run:%H:%M:%S} · "
        f"underlying {state.underlying_price:,.1f} · "
        f"strategies: {', '.join(engine.strategies) or 'none enabled'}"
        if state.last_run else "Not evaluated yet."
    )

    # --- open positions, before alerts: managing money at risk comes first ---
    positions_panel(engine, p)

    # --- what could be entered right now ---
    entry_radar(engine, cfg, p)

    _alerts(engine, p)
    _chart(engine, cfg, p)

    st.subheader("Why nothing fired")
    st.caption("Each enabled strategy's verdict for the current bar. "
               "A run of *risk rejected* means the setups were real but no "
               "strike fit the budget — a different problem from no setups.")
    evaluations_table(state.evaluations)


# ---------------------------------------------------------------- controls

def _controls(engine, cfg, p) -> None:
    """Evaluate on demand, and arm the notifications that need arming."""
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.3, 3])

    if c1.button("Evaluate now", type="primary", width="stretch"):
        st.session_state.force_run = True

    armed = sound_armed()
    if c2.button("🔔 Sound: on" if armed else "🔕 Enable sound",
                 width="stretch",
                 help="Browsers refuse to play audio until you have clicked "
                      "something on the page. This button is that click."):
        st.session_state.sound_armed = not armed
        st.rerun()

    if c3.button("Test chime", width="stretch", disabled=not armed,
                 help="Verify the chime now rather than discovering it is "
                      "blocked during a live signal."):
        play_alert_sound()

    desktop = c4.toggle(
        "Desktop toast", value=cfg.alerts.enable_desktop,
        help="An OS notification, which is the only path that still reaches "
             "you with this tab in the background.")
    cfg.alerts.enable_desktop = desktop

    if engine.state.kill_switch:
        if st.button("Re-arm kill switch", type="secondary"):
            engine.reset_kill_switch()
            st.rerun()


# ---------------------------------------------------------------- alerts

def _alerts(engine, p) -> None:
    """
    The alert feed, plus the three ways a new alert announces itself.

    The feed is session-scoped rather than read straight off the engine so it
    survives a feed reconnect, and so alerts pushed through the in-app channel
    (the Settings test button) land in the same place as the engine's own.
    """
    st.subheader("Alerts")

    feed = sync_ui_alerts(engine)
    fresh = unannounced_alerts()

    for alert in fresh:
        st.toast(f"**{alert.title()}**", icon=":material/notifications_active:")
    if fresh and sound_armed():
        play_alert_sound()

    if not feed:
        st.info("No alerts yet. Every strategy's verdict for the current bar is "
                "in **Why nothing fired** below — that table is the useful one "
                "on a quiet day.")
        return

    journal = get_journal()

    def log_fill(alert, premium):
        journal.paper_fill(alert.to_dict(), premium,
                           note="manual paper fill from UI")

    def confirm(alert):
        pos = engine.confirm_entry(alert)
        if pos is None:
            st.warning(
                "Not entered. The alert is stale, already confirmed, or a "
                "day rule has closed the session since it fired. Nothing "
                "was sent to the broker."
            )
        else:
            st.success(
                f"Entered {pos.state.lots_total} lot(s) of "
                f"{pos.quote.strike}{pos.quote.option_type}. Exits, the "
                f"breakeven shift and the trail are now automatic."
            )
            st.rerun()

    confirmable = {o["key"] for o in engine.state.pending_orders}
    for i, alert in enumerate(reversed(feed[-8:])):
        alert_card(
            alert, p, key=f"{i}_{alert.dedupe_key}",
            on_paper_fill=log_fill,
            on_confirm=confirm if alert.dedupe_key in confirmable else None,
        )


# ---------------------------------------------------------------- chart

def _chart(engine, cfg, p) -> None:
    state = engine.state
    st.subheader("Chart")
    if state.bars is None or state.bars.empty:
        st.caption("No bars loaded.")
        return

    session = engine.feed.session_slice(state.bars)
    if session.empty:
        session = state.bars.tail(120)
    levels, trendlines, vwap_series = build_overlays(session, cfg)
    latest = state.alerts[-1] if state.alerts else None
    st.plotly_chart(
        price_chart(session, p, levels, vwap_series, trendlines, latest,
                    title=f"{cfg.data.symbol} · {cfg.data.interval_minutes}m · "
                          f"{session.index[-1]:%Y-%m-%d}"),
        width="stretch",
    )
    with st.expander("Table view — the bars behind the chart"):
        st.dataframe(session.tail(60).iloc[::-1], width="stretch")
    if levels:
        with st.expander(f"Levels in play ({len(levels)})"):
            import pandas as pd
            st.dataframe(pd.DataFrame(
                [{"Price": f"{lv.price:,.1f}", "Kind": lv.kind,
                  "Touches": lv.touches} for lv in levels[:20]]),
                width="stretch", hide_index=True)
