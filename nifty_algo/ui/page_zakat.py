"""
Step 6 - Zakat & purification: what the year's wealth owes, and what to give
away.

Renders `planning/zakat.py` and decides nothing. Every ruling it depends on -
nisab basis, how shares are counted, retirement money - is chosen on Money &
goals, the one editor of any money number; this page reads them and says
plainly which are still missing rather than stating an amount built on a
default.

WITHHELD, NOT GUESSED. With an account unread, a metal price unknown or a
ruling unchosen, the amount due is withheld and the reason named. The
per-line breakdown still renders - it is a fact about what was read.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .components import banner
from .state import get_config, get_snapshot
from .theme import get_palette
from .. import onboarding
from ..planning import zakat


def render() -> None:
    p = get_palette()
    cfg = get_config()
    plan = cfg.plan
    st.title("Zakat & purification")
    st.caption(f"Step 6 of 6. Computed from every account you connected, "
               f"using the rulings you chose on **{onboarding.PLAN}**. "
               f"Arithmetic, not a fatwa - confirm your method with a scholar "
               f"you trust.")

    c1, _ = st.columns([1, 4])
    snapshot = get_snapshot(refresh=c1.button("Re-read holdings",
                                              key="zakat_refresh"))
    result = zakat.compute(snapshot, plan)

    _headline(result, p)
    for why in result.missing:
        banner(f"Amount withheld: {why}.", p.warning, "⚠")
    _hawl(plan)
    _breakdown(result)
    _purification(snapshot, plan, p)


def _headline(result, p) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Zakatable wealth", f"₹{result.base_inr:,.0f}",
              help="Each holding's value times the share of it that is "
                   "zakatable under your rulings - see the table.")
    c2.metric("Nisab", "—" if result.nisab_inr is None
              else f"₹{result.nisab_inr:,.0f}")
    due = result.due_inr
    c3.metric("Zakat due (2.5%)", "—" if due is None else f"₹{due:,.0f}")
    if due is not None:
        if result.above_nisab:
            banner(f"Your zakatable wealth is above the nisab, so "
                   f"<b>₹{due:,.0f}</b> is due on it.", p.good, "☪")
        else:
            banner("Your zakatable wealth is below the nisab - no zakat is "
                   "due on it this year.", p.good, "☪")


def _hawl(plan) -> None:
    if not plan.zakat_date:
        st.caption(f"No zakat date set - add your hawl on "
                   f"**{onboarding.PLAN}** to see how far away it is.")
        return
    try:
        anchor = date.fromisoformat(plan.zakat_date)
    except ValueError:
        return
    today = date.today()
    nxt = anchor.replace(year=today.year)
    if nxt < today:
        nxt = nxt.replace(year=today.year + 1)
    days = (nxt - today).days
    st.caption(
        f"Your zakat date is **{nxt:%d %b %Y}** ({days} days away). These "
        f"figures are today's holdings, not that day's - re-read on the day. "
        f"The Islamic year is about 11 days shorter than the Gregorian one; "
        f"if you keep a Gregorian date, many scholars adjust the rate to "
        f"about 2.577%.")


def _breakdown(result) -> None:
    st.subheader("How each holding is counted")
    if not result.lines:
        st.caption(f"Nothing read yet - connect an account on "
                   f"**{onboarding.CONNECT}**.")
        return
    frame = pd.DataFrame([{
        "Holding": line.symbol,
        "Name": line.name,
        "Value (₹)": round(line.value_inr),
        "Counted": line.zakatable_pct,
        "Zakatable (₹)": round(line.zakatable_inr),
        "Rule": line.rule,
    } for line in result.lines])
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "Counted": st.column_config.NumberColumn(format="percent"),
        "Value (₹)": st.column_config.NumberColumn(format="localized"),
        "Zakatable (₹)": st.column_config.NumberColumn(format="localized"),
    })
    st.download_button(
        "Download the statement (CSV)", frame.to_csv(index=False),
        file_name=f"zakat_{date.today().isoformat()}.csv", mime="text/csv",
        key="zakat_download", disabled=bool(result.missing))
    st.caption("Debts due within the year (an EMI, a card balance) are "
               "deductible in many opinions and are not subtracted here - "
               "take them off the zakatable figure yourself if you follow "
               "that view.")


def _purification(snapshot, plan, p) -> None:
    st.subheader("Purification")
    pur = zakat.purification(snapshot, plan)
    c1, c2, c3 = st.columns(3)
    c1.metric("Interest to give away", f"₹{pur.interest_inr:,.0f}")
    c2.metric("Impure dividend share",
              "unknown" if pur.dividend_inr is None
              else f"₹{pur.dividend_inr:,.0f}")
    c3.metric("Total to purify", "—" if pur.total_inr is None
              else f"₹{pur.total_inr:,.0f}")
    if pur.dividend_inr is None:
        banner("Dividends are entered but the impure share is 0% - that is "
               "reported as <b>unknown</b>, not as nothing owed. Shariah "
               "funds publish the figure yearly.", p.warning, "⚠")
    if pur.sources:
        st.markdown("**Where interest comes from in what you hold:**")
        st.markdown("\n".join(f"- {s}" for s in pur.sources))
    st.caption("Purification is given away without expecting reward, and is "
               "separate from zakat - it does not count towards it.")
