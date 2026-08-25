"""Journal page - the append-only record."""
from __future__ import annotations
import json

import pandas as pd
import streamlit as st

from .theme import get_palette
from .state import get_journal


EVENT_HELP = {
    "signal": "A strategy proposed a trade.",
    "rejected": "The risk engine refused it — strike, liquidity, or budget.",
    "alert_sent": "An alert was dispatched to the enabled channels.",
    "alert_suppressed": "A duplicate or cooling-down alert was withheld.",
    "halt": "A session governor blocked further entries.",
    "kill_switch": "The engine stopped on a data gap or unhandled error.",
    "paper_fill": "You logged a simulated entry.",
    "paper_exit": "You logged a simulated exit.",
    "session_start": "A new trading day began.",
    # --- the swing book ---
    "swing_scan": "A daily scan ran. Holds every pick it proposed.",
    "swing_armed": "A buy trigger was placed at Zerodha for a pick.",
    "swing_rearmed": "An unfilled entry was re-priced onto a fresh scan.",
    "swing_filled": "A buy trigger fired. Carries the fill price AND the "
                    "slippage against the ticket.",
    "swing_exit_armed": "A stop/target OCO was placed or replaced.",
    "swing_ladder": "The exit ladder moved - breakeven shift or a trail.",
    "swing_exit": "Part or all of a position was released by the ladder.",
    "swing_closed": "A position finished. Carries the realised R.",
    "swing_cancelled": "A ticket was retired without ever being filled.",
    "holdings_authorised": "You recorded a CDSL TPIN authorisation. Valid "
                           "for that trading day only.",
    # --- orders, either book ---
    "order_dry_run": "The exact payload an order WOULD have sent.",
    "order_placed": "An order or trigger the broker accepted.",
    "order_failed": "The broker refused it, or was unreachable.",
}


def render() -> None:
    p = get_palette()
    st.title("Journal")
    st.caption(
        "Append-only, one file per trading day, never rewritten. Its value is "
        "that it records what you knew at the time — a log you can edit is a log "
        "that eventually agrees with your memory instead of the facts."
    )

    journal = get_journal()
    days = journal.available_days()
    if not days:
        st.info("No journal entries yet. They appear once the engine evaluates a "
                "bar — run a pass on the **Live** page.")
        return

    c1, c2 = st.columns([1, 2])
    day = c1.selectbox("Day", days, format_func=lambda d: f"{d:%Y-%m-%d}")
    records = journal.read_day(day)
    if not records:
        st.info("That day's file is empty.")
        return

    kinds = sorted({r.get("event", "?") for r in records})
    chosen = c2.multiselect("Events", kinds, default=kinds)

    filtered = [r for r in records if r.get("event") in chosen]
    st.caption(f"{len(filtered)} of {len(records)} records · "
               f"`{journal.path_for(day)}`")

    # --- counts, so rejections are as visible as alerts ---
    counts = pd.Series([r.get("event") for r in records]).value_counts()
    cols = st.columns(min(len(counts), 6))
    for col, (event, n) in zip(cols, counts.items()):
        col.metric(event.replace("_", " ").title(), int(n),
                   help=EVENT_HELP.get(event, ""))

    if int(counts.get("rejected", 0)) > 0:
        st.caption(
            f"**{int(counts['rejected'])} setups were rejected by the risk engine "
            f"on this day.** That number matters as much as the alerts — if it "
            f"stays high, your stop width and the delta ceiling it implies are "
            f"fighting each other."
        )

    # --- flat table ---
    rows = []
    for r in filtered:
        row = {"Time": r.get("ts", "")[11:19], "Event": r.get("event", "")}
        for k in ("strategy", "reason", "detail", "regime"):
            if k in r:
                row[k.title()] = r[k]
        if "alert" in r and isinstance(r["alert"], dict):
            a = r["alert"]
            row["Strategy"] = a.get("strategy_label", row.get("Strategy", ""))
            if a.get("strike"):
                row["Trade"] = (f"{a.get('direction', '')} {a.get('strike')}"
                                f"{a.get('option_type', '')}")
                row["Entry"] = a.get("entry_premium")
                row["Target"] = a.get("target_premium")
                row["Stop"] = a.get("stop_premium")
        if "fill_premium" in r:
            row["Fill"] = r["fill_premium"]
        if "net_pnl" in r:
            row["Net ₹"] = r["net_pnl"]
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", df.to_csv(index=False),
                       f"journal_{day:%Y-%m-%d}.csv", "text/csv",
                       width="stretch")
    c2.download_button(
        "Download raw JSONL",
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        f"journal_{day:%Y-%m-%d}.jsonl", "application/x-ndjson",
        width="stretch")

    with st.expander("Raw records"):
        st.json(filtered[-40:])
