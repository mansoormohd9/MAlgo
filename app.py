"""
Nifty Algo — your money, step by step.

    streamlit run app.py

The main flow is five steps (Connect -> Money & goals -> Allocation -> Monthly
sleeve -> US / LRS), built in `nifty_algo/ui/nav.py`. The intraday option
book, the swing book and their tools sit behind "Show research lab".

For the option book, this app is a VIEWER over `nifty_algo.engine.TradingEngine`. Every decision is
made in the engine, which is headless and importable, so the alerts you see
here are produced by exactly the same code as
`python -m nifty_algo.run_live`. The UI decides nothing.

ENTRIES REQUIRE A CLICK. An alert never becomes a position on its own: you
press **Place order** on the card, which calls `engine.confirm_entry()`. Once
you are in, exits are automatic - the breakeven shift, the partial at +2R and
the trailing stop all run without you, because a stop that needs a human to
press a button is not a stop.

`BrokerConfig.dry_run` defaults to True, so even a confirmed entry only logs
the payload until you deliberately turn it off.

THE LOGIN GATE IS AT MODULE LEVEL, NOT INSIDE `main()`. `auth.require_login()`
sits above the `page_*` imports and calls `st.stop()`, so an unauthenticated
session never imports a page module and therefore never builds the engine, the
feed, the journal, the broker or the auto-refresh fragment. Inside `main()` it
would still be correct, and every one of those would already have been
constructed for anybody who loaded the URL. See `nifty_algo/ui/auth.py`.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# THE WORKING DIRECTORY IS ANCHORED HERE, AND IT MUST HAPPEN BEFORE ANYTHING
# READS A FILE.
#
# Every data path in `config.py` is relative - `data/cache`, `data/nifty100.csv`,
# `data/settings.json`, `journal/` - and so is `load_dotenv()`, which searches
# upward from the CWD for `.env`. Relative to the CWD means relative to however
# you launched the app, so `streamlit run` from any other directory made the
# Monthly sleeve report "No factor cache at data/cache/factor_daily_india.parquet"
# with a 62 MB cache present, and told you to spend ~35 minutes re-fetching it.
# The same launch would silently find no `.env`, hence no `APP_PASSWORD`.
#
# One chdir fixes every one of those at once, which is why it is here rather
# than threaded through twenty call sites. `app.py` lives at the repo root, so
# its own location is the anchor - the one thing that does not depend on the
# launch. Libraries do not get to do this; a process entry point does, and this
# file is the process. `nifty_algo/paths.py` is the library-side answer for
# readers that must work no matter who launched them.
os.chdir(Path(__file__).resolve().parent)

load_dotenv()

st.set_page_config(
    page_title="Nifty Algo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from nifty_algo.ui import auth                           # noqa: E402

auth.bridge_secrets()   # st.secrets -> os.environ, so every os.getenv reader works
auth.require_login()    # NOTHING below this line runs until you are signed in

from nifty_algo.ui.theme import CSS, get_palette          # noqa: E402
from nifty_algo.ui import (page_live, page_brief, page_swing,        # noqa: E402
                           page_sleeve, page_book, page_portfolio,
                           page_research, page_strategies, page_backtest,
                           page_journal, page_settings, page_connect,
                           page_plan, page_allocation, page_holdings,
                           page_zakat, nav)
from nifty_algo.ui.state import get_config                # noqa: E402
from nifty_algo.ui.components import banner               # noqa: E402
from nifty_algo.ui import refresh                         # noqa: E402
from nifty_algo import onboarding                         # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

PAGES = {
    onboarding.CONNECT: page_connect.render,
    onboarding.PLAN: page_plan.render,
    onboarding.ALLOCATION: page_allocation.render,
    onboarding.SLEEVE: page_sleeve.render,
    onboarding.FOREIGN: page_portfolio.render,
    onboarding.ZAKAT: page_zakat.render,
    "Holdings": page_holdings.render,
    "Research": page_research.render,
    "Journal": page_journal.render,
    "Settings": page_settings.render,
    # research lab - listed only while the toggle is on (see nav.py)
    "Daily picks": page_swing.render,
    "Trade book": page_book.render,
    "Live alerts": page_live.render,
    "Daily brief": page_brief.render,
    "Strategies": page_strategies.render,
    "Backtest": page_backtest.render,
}


def main() -> None:
    cfg = get_config()
    p = get_palette()

    with st.sidebar:
        choice = nav.sidebar(cfg)
        if cfg.ui.show_research_lab:
            st.divider()
            # Polling drives the option engine only, which lives in the lab.
            refresh.sidebar_controls(cfg)
        auth.logout_control()
        nav.lab_footer(cfg, p)

    _protection_banner(cfg, p)
    _halal_banner(cfg, p)
    PAGES[choice]()


def _halal_banner(cfg, p) -> None:
    """
    On EVERY page, while a halal-only holder owns something not halal.

    Same reasoning as the protection banner: a warning confined to Holdings
    is one you see only when you were already looking. It reads the snapshot
    already in the session and never triggers a broker read, and it is
    wrapped the same way - a notice that can crash the app gets disabled.
    """
    if not cfg.plan.halal_only:
        return
    try:
        from nifty_algo.planning import compliance
        from nifty_algo.ui.state import get_compliance

        summary = get_compliance()
        if summary is None:
            return
        bad = summary.of(compliance.NON_COMPLIANT)
        if not bad:
            return
        names = ", ".join(r.position.symbol for r in bad[:4])
        more = f" and {len(bad) - 4} more" if len(bad) > 4 else ""
        banner(f"<b>{len(bad)} holding(s) are not halal</b> — {names}{more}. "
               f"Your plan is halal-only. See <b>Holdings</b> for the reason "
               f"on each line.", p.critical, "⛔")
    except Exception:
        pass


def _protection_banner(cfg, p) -> None:
    """
    On EVERY page, whenever a live position has no working stop today.

    Without DDPI a stop GTT is rejected on any day you have not authorised
    holdings - and Kite still shows it as active, so nothing on the broker's
    own screen tells you. A warning confined to the Trade book page would be
    a warning you see only when you were already looking.

    Wrapped in a bare `except` deliberately: this is a safety notice, and a
    safety notice that can crash the app is one you will end up disabling.
    """
    try:
        from nifty_algo.broker import kite_equity as eq_mod
        from nifty_algo.swing.book import TicketState
        from nifty_algo.ui.state import get_book, get_equity_broker

        book = get_book("india")
        live = book.by_state(TicketState.ARMED, TicketState.OPEN)
        if not live:
            return
        state, _ = get_equity_broker().protection_state()
        if state != eq_mod.UNPROTECTED:
            return
        banner(
            f"<b>{len(live)} live ticket(s) and no CDSL authorisation "
            f"recorded today.</b> Any stop that triggers will be REJECTED — "
            f"it will still show as active in Kite. Authorise holdings, or "
            f"enable DDPI once on Money & goals.",
            p.critical, "⛔")
    except Exception:
        pass


if __name__ == "__main__":
    main()
