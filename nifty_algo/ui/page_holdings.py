"""
Holdings: every line you own, from every account that answered - and which
accounts did not.

Split out of the old Portfolio page, whose connector strip sat halfway down a
page about US estate tax. "What do I own" is the question every step reads
first, so it gets its own page and the SAME cached snapshot the steps use
(`state.get_snapshot`), so this table and the allocation can never disagree.

WHICH ACCOUNTS ANSWERED IS AT THE TOP, above the positions, because every
figure below is a figure about what was READ. A failed read renders as
"NO" with its reason rather than as a shorter table, since understating an
account is the direction that reads as reassurance.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .components import banner
from .state import get_config, get_snapshot
from .theme import get_palette
from ..planning import allocation as alloc


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Holdings")

    c1, _ = st.columns([1, 4])
    snapshot = get_snapshot(refresh=c1.button("Re-read", key="holdings_refresh"))

    st.dataframe(pd.DataFrame([{
        "Account": r.source,
        "Answered": "yes" if r.available else "NO",
        "Holdings": len(r.positions),
        "Detail": r.note,
    } for r in snapshot.results]), width="stretch", hide_index=True)

    if not snapshot.complete:
        banner(
            "<b>Not every account answered, so percentages are withheld.</b> "
            "A share computed against a denominator we could not establish "
            "reads exactly like one that was. " + " ".join(snapshot.caveats()),
            p.warning, "⚠")
    else:
        banner(f"Every connected account answered. {snapshot.note()}",
               p.good, "▣")

    if not snapshot.positions:
        st.caption("Nothing recorded yet — connect an account on "
                   "**1 · Connect**.")
        return

    st.dataframe(pd.DataFrame([{
        "Symbol": pos.symbol,
        "Name": pos.name,
        "Class": alloc.LABELS[alloc.bucket_of(pos)],
        "Qty": pos.quantity,
        "Last": pos.last_price,
        "Currency": pos.currency,
        "Value (₹)": round(snapshot.value_inr.get(pos.key, 0.0)),
        "Weight": snapshot.weight(pos.key),
        "Account": pos.source,
    } for pos in sorted(snapshot.positions,
                        key=lambda x: -snapshot.value_inr.get(x.key, 0.0))]),
        width="stretch", hide_index=True,
        column_config={
            "Weight": st.column_config.NumberColumn(format="percent"),
            "Value (₹)": st.column_config.NumberColumn(format="localized"),
        })

    st.caption(
        f"Connected: `{'`, `'.join(cfg.portfolio.connectors)}`. An account "
        f"you connect that cannot answer makes this incomplete; one you have "
        f"not connected is never asked.")
