"""
Nifty intraday option-buying — alert console.

    streamlit run app.py

This app is a VIEWER over `nifty_algo.engine.TradingEngine`. Every decision is
made in the engine, which is headless and importable, so the alerts you see
here are produced by exactly the same code as
`python -m nifty_algo.run_live`. The UI decides nothing.

ALERT ONLY. There is no broker order path in this application.
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Nifty Algo — Alert Console",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from nifty_algo.ui.theme import CSS, get_palette          # noqa: E402
from nifty_algo.ui import (page_live, page_strategies,     # noqa: E402
                           page_backtest, page_journal, page_settings)
from nifty_algo.ui.state import get_config                # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

PAGES = {
    "Live alerts": page_live.render,
    "Strategies": page_strategies.render,
    "Backtest": page_backtest.render,
    "Journal": page_journal.render,
    "Settings": page_settings.render,
}


def main() -> None:
    cfg = get_config()
    p = get_palette()

    with st.sidebar:
        st.markdown("### Nifty Algo")
        st.caption("Intraday option-buying alert console")
        choice = st.radio("Page", list(PAGES), label_visibility="collapsed")

        st.divider()
        st.caption(
            f"**Capital** ₹{cfg.capital.starting_capital:,.0f}  \n"
            f"**Target / stop** +{cfg.capital.session_target_pct:.0%} / "
            f"−{cfg.capital.session_stop_pct:.0%}  \n"
            f"**Max entries** {cfg.capital.max_entries_per_session}  \n"
            f"**Risk / trade** ₹{cfg.capital.risk_per_trade_rupees:,.0f} "
            f"at {cfg.capital.reward_risk_ratio:.0f}:1"
        )
        st.divider()
        st.markdown(
            f"<span style='color:{p.muted};font-size:.78rem;line-height:1.5'>"
            f"<b>Alert only — never places an order.</b><br>"
            f"For alerts with the browser closed, run:<br>"
            f"<code>python -m nifty_algo.run_live --telegram</code><br><br>"
            f"Not investment advice. SEBI data shows over 90% of retail F&O "
            f"traders lose money.</span>",
            unsafe_allow_html=True,
        )

    PAGES[choice]()


if __name__ == "__main__":
    main()
