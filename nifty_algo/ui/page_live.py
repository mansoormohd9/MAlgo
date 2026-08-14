"""Live alerts page."""
from __future__ import annotations
from datetime import datetime

import streamlit as st

from .theme import get_palette
from .charts import price_chart, build_overlays
from .components import (feed_banner, session_header, alert_card,
                         evaluations_table, banner, positions_panel,
                         broker_banner)
from .state import get_engine, get_config, get_journal, play_alert_sound, engine_error


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

    feed_banner(engine, p)
    broker_banner(engine, p)

    # --- controls ---
    c1, c2, c3 = st.columns([1.4, 1.4, 3])
    auto = c1.toggle("Auto-refresh", value=st.session_state.get("auto_refresh", False),
                     key="auto_refresh")
    if c2.button("Evaluate now", type="primary", width="stretch"):
        st.session_state.force_run = True

    if auto:
        interval = cfg.data.poll_seconds * 1000
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=interval, key="live_autorefresh")
        except ImportError:
            # Not a hard dependency - the manual button covers the same ground.
            c3.caption(f"`pip install streamlit-autorefresh` for {cfg.data.poll_seconds}s "
                       f"polling in-browser. Meanwhile use *Evaluate now*.")

    if engine.state.kill_switch:
        if st.button("Re-arm kill switch", type="secondary"):
            engine.reset_kill_switch()
            st.rerun()

    # --- run a pass ---
    should_run = auto or st.session_state.pop("force_run", False) \
        or engine.state.last_run is None
    prev_alerts = len(engine.state.alerts)
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

    # --- alerts ---
    st.subheader("Alerts")
    if len(state.alerts) > prev_alerts:
        play_alert_sound()

    if not state.alerts:
        st.info("No alerts yet. Every strategy's verdict for the current bar is "
                "in **Why nothing fired** below — that table is the useful one "
                "on a quiet day.")
    else:
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

        confirmable = {o["key"] for o in state.pending_orders}
        for i, alert in enumerate(reversed(state.alerts[-8:])):
            alert_card(
                alert, p, key=f"{i}_{alert.dedupe_key}",
                on_paper_fill=log_fill,
                on_confirm=confirm if alert.dedupe_key in confirmable else None,
            )

    # --- chart ---
    st.subheader("Chart")
    if state.bars is not None and not state.bars.empty:
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
    else:
        st.caption("No bars loaded.")

    # --- verdicts ---
    st.subheader("Why nothing fired")
    st.caption("Each enabled strategy's verdict for the current bar. "
               "A run of *risk rejected* means the setups were real but no "
               "strike fit the budget — a different problem from no setups.")
    evaluations_table(state.evaluations)
