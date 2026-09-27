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

from datetime import date

import pandas as pd
import streamlit as st

from .components import banner
from .state import get_compliance, get_config, get_snapshot
from .theme import get_palette
from .. import onboarding
from ..planning import allocation as alloc
from ..planning import compliance
from ..planning import tax as tax_mod


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Allocation")
    st.caption("Step 3 of 6. Your whole portfolio against the split you set "
               "in step 2.")

    c1, _ = st.columns([1, 4])
    snapshot = get_snapshot(refresh=c1.button("Re-read holdings",
                                              key="alloc_refresh"))
    a = alloc.build(snapshot, cfg.plan)
    for note in a.notes:
        banner(note, p.warning, "⚠")
    _stale(snapshot, p)

    _headline(a, cfg, p)
    _compliance_strip(get_compliance(snapshot), cfg, p)
    _table(a)
    _emergency(a, cfg, p)
    _next_money(a, cfg, p)
    _tax(snapshot, p)
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
        "How holdings are classified: Indian equity is Kite shares, Indian "
        "ETFs and equity/hybrid mutual funds. Gold & silver ETFs and SGBs are "
        "recognised by symbol, metal funds by name. Liquid/overnight ETFs "
        "(LIQUIDBEES) count as cash. NSE-listed international ETFs (MON100, "
        "MAFANG…) and fund-of-funds investing abroad count as foreign equity. "
        "Anything misfiled can be recorded by hand with the right "
        "`asset_class`. The monthly sleeve's holdings sit inside Indian "
        "equity.")


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


def _compliance_strip(summary, cfg, p) -> None:
    """
    One line: how much of the whole is halal. The detail lives on Holdings.

    Withheld with the rest of the page's percentages when an account failed,
    for the same reason - "94% halal" against a partial book reads exactly
    like one against the whole.
    """
    if summary is None or not summary.rows:
        return
    parts = []
    for status in compliance.STATUSES:
        share = summary.share(status)
        if share is None:
            continue
        if share > 0 or status == compliance.COMPLIANT:
            parts.append(f"{compliance.ICONS[status]} {share:.0%} "
                         f"{compliance.LABELS[status].lower()}")
    if not parts:
        st.caption("Halal status withheld until every account answers.")
        return
    bad = summary.share(compliance.NON_COMPLIANT) or 0.0
    colour = (p.critical if (bad > 0 and cfg.plan.halal_only)
              else p.warning if summary.needs_attention else p.good)
    banner(" · ".join(parts) + " — by value. Per-line reasons on "
           "<b>Holdings</b>.", colour, "☪")


def _halal_note(cfg, p) -> None:
    if cfg.plan.halal_only and cfg.plan.w_fixed_income > 0:
        banner("Your plan is halal-only and holds a stable-leg target. EPF, "
               "PPF, FDs and bond funds pay interest. Sukuk funds (SPSK on "
               "the foreign side) are the halal version of this leg. If you "
               "hold EPF by obligation (it is mandatory for most salaried "
               "jobs), record it, set the target to what you cannot avoid, "
               "and purify the interest on the Zakat page.",
               p.warning, "⚖")


def _stale(snapshot, p) -> None:
    """Balances that describe a date long past - shown, never silently used."""
    old = alloc.stale_lines(snapshot.positions, date.today())
    if not old:
        return
    names = ", ".join(f"{pos.symbol} ({age // 30} months)"
                      for pos, age in old[:4])
    banner(f"<b>{len(old)} balance(s) are more than "
           f"{alloc.STALE_DAYS // 30} months old</b> — {names}. Update them "
           f"on <b>{onboarding.CONNECT}</b> (or import a fresh CAS); the "
           f"split below treats them as today's.", p.warning, "⏳")


def _tax(snapshot, p) -> None:
    """
    The yearly LTCG exemption - the tax you can legally not pay.

    Shown all year, loudest in January-March when an unused exemption is
    about to expire. An UPPER BOUND, and labelled as one: Kite gives an
    average price, not lots, so short-term lots are inside the figure.
    """
    st.subheader("Tax-free gains you can book this year")
    view = tax_mod.harvest_view(snapshot)
    today = date.today()
    c1, c2, c3 = st.columns(3)
    c1.metric("Yearly LTCG exemption",
              f"₹{tax_mod.LTCG_EXEMPTION_INR:,.0f}",
              help="Section 112A, equity-oriented holdings only. Does not "
                   "carry forward.")
    c2.metric("Unrealised gain (upper bound)", f"₹{view.gain_inr:,.0f}")
    c3.metric("Tax saved by harvesting", f"₹{view.tax_saved_inr:,.0f}",
              help=f"Up to the exemption, at {tax_mod.LTCG_RATE:.1%}.")
    if view.harvestable_inr > 0 and tax_mod.harvest_season(today):
        banner(f"The financial year ends <b>{tax_mod.fy_end(today):%d %b}</b>. "
               f"Selling lots held over 12 months to book up to "
               f"₹{view.harvestable_inr:,.0f} of gain, then buying back, "
               f"resets their cost at no tax.", p.good, "₹")
    st.caption(
        "Only lots held **over 12 months** qualify. The holding period is "
        "not visible here (Kite reports an average price, not lots), so check "
        "your tradebook before selling. Gold ETFs, debt funds and overseas "
        "fund-of-funds are excluded - the exemption is for equity-oriented "
        "holdings. "
        + (f"{view.lines_without_cost} line(s) have no cost basis, so their "
           f"gain is unknown, not zero. " if view.lines_without_cost else "")
        + f"Rates as of {tax_mod.VERIFIED_ON}.")
