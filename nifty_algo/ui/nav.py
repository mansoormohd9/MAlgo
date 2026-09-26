"""
The sidebar: the steps in order, where you are in them, and the research lab.

WHY THE NAVIGATION IS STEPS AND NOT BOOKS. The console used to list one page
per trading book - Live alerts first, because the option book was built first
- so the landing page was an intraday option scanner for a user who does not
trade options, and the order you had to do things in (log in, set the pots,
then scan) was written down nowhere. The main list is now the order money
flows: connect the accounts, say what the money is for, see where it stands,
fund the books.

THE RESEARCH LAB IS HIDDEN, NOT DELETED. The option, swing and intraday books
were each measured against a null and lost (or stayed closed); their pages
remain behind one persisted toggle so the record is still reachable and every
test that pins them still runs.

The one radio is keyed (`nav_page`) so a "next step" button can move it, and
lab labels are unprefixed so existing page tests navigate unchanged once the
lab is on.
"""
from __future__ import annotations

import streamlit as st

from .. import onboarding
from .state import get_config, get_kite_session, peek_snapshot, save_settings

NAV_KEY = "nav_page"


def main_pages() -> list[str]:
    return [onboarding.CONNECT, onboarding.PLAN, onboarding.ALLOCATION,
            onboarding.SLEEVE, onboarding.FOREIGN,
            "Holdings", "Research", "Journal", "Settings"]


LAB_PAGES = ["Daily picks", "Trade book", "Live alerts", "Daily brief",
             "Strategies", "Backtest"]


def options(cfg) -> list[str]:
    return main_pages() + (LAB_PAGES if cfg.ui.show_research_lab else [])


def _kite_authenticated() -> bool:
    try:
        session = get_kite_session()
        return bool(session is not None and session.authenticated)
    except Exception:
        return False


def current_steps(cfg) -> list[onboarding.Step]:
    snap = peek_snapshot()
    return onboarding.steps(
        cfg, kite_authenticated=_kite_authenticated(),
        holdings_count=None if snap is None else len(snap.positions))


def _go(page: str) -> None:
    st.session_state[NAV_KEY] = page


def sidebar(cfg) -> str:
    """Render the sidebar and return the chosen page label."""
    steps = current_steps(cfg)
    opts = options(cfg)

    # First load of the session lands on the first unfinished step - or on
    # Allocation once setup is done. A stored choice that no longer exists
    # (the lab was switched off while on a lab page) falls back the same way.
    if st.session_state.get(NAV_KEY) not in opts:
        nxt = onboarding.next_step(steps)
        st.session_state[NAV_KEY] = (nxt.page if nxt is not None
                                     else onboarding.ALLOCATION)

    st.markdown("### Nifty Algo")
    st.caption("Your money, step by step")
    choice = st.radio("Page", opts, key=NAV_KEY,
                      label_visibility="collapsed")

    st.divider()
    _checklist(steps)
    _pots(cfg)
    _lab_toggle(cfg)
    return choice


def _checklist(steps) -> None:
    lines = []
    for s in steps:
        mark = "✅" if s.done else ("◻️" if s.optional else "⬜")
        lines.append(f"{mark} **{s.page}**  \n<span style='font-size:.78rem'>"
                     f"{s.detail}</span>")
    st.markdown("  \n".join(lines), unsafe_allow_html=True)
    nxt = onboarding.next_step(steps)
    if nxt is not None and st.session_state.get(NAV_KEY) != nxt.page:
        st.button(f"Next: {nxt.page}", key="nav_next", on_click=_go,
                  args=(nxt.page,), width="stretch")


def _pots(cfg) -> None:
    cap, eq = cfg.capital, cfg.equity_broker
    st.divider()
    lines = [f"**Sleeve** ₹{cap.factor_capital_inr:,.0f}",
             f"**Foreign / LRS** ₹{cap.foreign_capital_inr:,.0f}"]
    if cfg.ui.show_research_lab:
        lines += [f"**Swing** ₹{cap.swing_capital_inr:,.0f}",
                  f"**Options** ₹{cap.starting_capital:,.0f}"]
    st.caption("  \n".join(lines) + "  \nEdit on **2 · Money & goals**.")
    st.caption(("Kite: logged in today" if _kite_authenticated()
                else "Kite: not logged in") + "  \n"
               + ("Orders: DRY RUN" if eq.dry_run
                  else "Orders: LIVE (swing book)"))


def _lab_toggle(cfg) -> None:
    on = st.toggle(
        "Show research lab", value=bool(cfg.ui.show_research_lab),
        key="show_lab",
        help="The option, swing and intraday books. Each was measured "
             "against a random null and lost, or is closed - kept for the "
             "record, out of the way.")
    if bool(on) != bool(cfg.ui.show_research_lab):
        cfg.ui.show_research_lab = bool(on)
        save_settings()
        st.rerun()


def lab_footer(cfg, p) -> None:
    """The option-book notes, shown only when the lab is open."""
    if not cfg.ui.show_research_lab:
        return
    st.divider()
    st.markdown(
        f"<span style='color:{p.muted};font-size:.78rem;line-height:1.5'>"
        f"<b>Research lab</b><br>Option book "
        f"{'DRY RUN' if cfg.broker.dry_run else 'LIVE'}. Entries need your "
        f"click; exits are automatic.<br>"
        f"<code>python -m nifty_algo.run_live --telegram</code> never places "
        f"an order.<br><br>Not investment advice. SEBI data shows over 90% "
        f"of retail F&O traders lose money.</span>",
        unsafe_allow_html=True)
