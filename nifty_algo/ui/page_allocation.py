"""
Step 3 - Allocation: where the whole of your money stands against the plan.

The first page in this console that looks at net worth rather than at a book.
It renders `planning/allocation.py` and decides nothing: drift, the emergency
fund, where the next month's money should go, and - only when new money cannot
close a gap - which class to trim.

WITHHELD, NOT GUESSED. When any connected account failed to answer, every
percentage on this page is withheld (the `PortfolioSnapshot.weight()` rule).
Absolute rupees still show; they are facts about what was read.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .components import banner
from .state import get_config, get_snapshot
from .theme import get_palette
from .. import onboarding
from ..planning import allocation as alloc


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Allocation")
    st.caption("Step 3 of 5. Your whole portfolio against the split you set "
               "in step 2.")

    c1, _ = st.columns([1, 4])
    snapshot = get_snapshot(refresh=c1.button("Re-read holdings",
                                              key="alloc_refresh"))
    a = alloc.build(snapshot, cfg.plan)
    for note in a.notes:
        banner(note, p.warning, "⚠")

    _headline(a, cfg, p)
    _table(a)
    _emergency(a, cfg, p)
    _next_money(a, cfg, p)
    _halal_note(cfg, p)


def _headline(a, cfg, p) -> None:
    c1, c2, c3 = st.columns(3)
    if a.net_worth_inr is not None:
        c1.metric("Net worth (connected)", f"₹{a.net_worth_inr:,.0f}")
    else:
        c1.metric("Read so far (incomplete)", f"₹{a.total_inr:,.0f}")
    c2.metric("Classes outside the band",
              "—" if not (a.complete and a.targets_set)
              else str(len(a.out_of_band)))
    cap = alloc.sleeve_cap(a.net_worth_inr, cfg.plan.tolerated_drawdown_pct)
    c3.metric("Sleeve ceiling",
              "—" if cap is None else f"₹{cap:,.0f}",
              help="Largest monthly-sleeve pot your tolerated fall supports. "
                   "Set the pot on Money & goals.")


def _table(a) -> None:
    rows = []
    for r in a.rows:
        rows.append({
            "Class": r.label,
            "Value (₹)": round(r.value_inr),
            "Current %": r.current_pct,
            "Target %": r.target_pct if a.targets_set else None,
            "Drift (pp)": (round(r.drift_pp, 1)
                           if a.targets_set and r.drift_pp is not None
                           else None),
            "Status": r.status if a.targets_set else "targets not set",
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Current %": st.column_config.ProgressColumn(
                "Current %", min_value=0, max_value=100, format="%.1f%%"),
            "Target %": st.column_config.ProgressColumn(
                "Target %", min_value=0, max_value=100, format="%.0f%%"),
            "Value (₹)": st.column_config.NumberColumn(format="localized"),
        })
    st.caption(
        "Indian equity counts Kite holdings, Indian ETFs and equity mutual "
        "funds; gold ETFs are recognised by symbol (record anything missed as "
        "`gold`). The monthly sleeve's holdings sit inside Indian equity.")


def _emergency(a, cfg, p) -> None:
    st.subheader("Emergency fund")
    check = alloc.emergency(a, cfg.plan)
    if check is None:
        st.caption(f"Enter monthly expenses on **{onboarding.PLAN}** to check "
                   f"this.")
        return
    if check.ok:
        banner(f"Cash covers <b>{check.months_covered:.1f} months</b> of "
               f"expenses, against the {cfg.plan.emergency_months:.0f} you "
               f"set.", p.good, "▣")
    else:
        banner(f"Cash covers <b>{check.months_covered:.1f} months</b>, short "
               f"of {cfg.plan.emergency_months:.0f} by "
               f"<b>₹{check.shortfall_inr:,.0f}</b>. Fill this before "
               f"investing new money anywhere else — an emergency paid for by "
               f"selling equity in a crash is the drawdown made permanent.",
               p.critical, "⛔")


def _next_money(a, cfg, p) -> None:
    st.subheader("Where the next rupee goes")
    amount = st.number_input(
        "New money to place (₹)", min_value=0.0, step=5_000.0,
        value=float(cfg.plan.monthly_investment_inr), key="alloc_amount",
        help="Defaults to your monthly investment from step 2. Changing it "
             "here is a what-if and is not saved.")
    plan = alloc.plan_contribution(a, amount)
    if plan is None:
        st.caption("Needs every account read and a target split that sums to "
                   "100%.")
        return
    st.caption(plan.reason)
    st.dataframe(pd.DataFrame([
        {"Class": alloc.LABELS[b], "Put in (₹)": round(v)}
        for b, v in plan.buy.items() if v >= 1
    ]), hide_index=True, width="stretch")
    if plan.trim:
        st.markdown("**Still over the band after the new money:**")
        for b, v in plan.trim.items():
            st.markdown(f"- {alloc.LABELS[b]}: trim about ₹{v:,.0f}")
        st.caption("Selling realises gains. In India, listed equity held under "
                   "12 months is taxed as STCG at 20%, over 12 months as LTCG "
                   "at 12.5% above ₹1.25 lakh a year. Prefer waiting for more "
                   "new money unless the drift is large.")


def _halal_note(cfg, p) -> None:
    if cfg.plan.halal_only and cfg.plan.w_fixed_income > 0:
        banner("Your plan is halal-only and holds a fixed-income target. EPF, "
               "PPF, FDs and bond funds pay interest; if you hold them by "
               "obligation (EPF is mandatory for most salaried jobs) consider "
               "recording them but setting their target to what you cannot "
               "avoid, and holding the stable leg in gold or cash.",
               p.warning, "⚖")
