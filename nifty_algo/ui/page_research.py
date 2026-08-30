"""
Research: the fact packs, and where the holdings behind them came from.

A VIEWER, LIKE EVERY OTHER PAGE HERE. `research/macro.py` and
`research/risk_report.py` are headless and are the same code the CLI runs, so
what this page shows and what a Claude Code skill reads cannot diverge. This
module decides nothing.

THE CONNECTOR STRIP IS THE MOST IMPORTANT THING ON THE PAGE, and it is at the
top for that reason. Every percentage below it is withheld when a source could
not be read - `PortfolioSnapshot.weight()` returns None rather than dividing
by a denominator it does not have - and a reader who does not know WHY the
weights are blank will assume the page is broken rather than that the account
is half-read.

THE JSON BUTTON IS NOT A DEBUGGING AID. The pack is what a skill consumes; the
briefing is written from it and from nothing else. Being able to copy it out
of the page is how you get the same briefing here that you would get in the
terminal.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..research import macro as macro_mod
from ..research import risk_report as risk_mod
from ..swing import markets as markets_mod
from .components import banner
from .state import get_config
from .theme import get_palette

REPORTS = {
    "Macro impact": (macro_mod.REPORT, macro_mod.build),
    "Portfolio risk": (risk_mod.REPORT, risk_mod.build),
}


def render() -> None:
    p = get_palette()
    cfg = get_config()

    st.title("Research")
    st.caption("Deterministic fact packs. Every number carries its source and "
               "whether it could be established at all - the prose is written "
               "from these, never the other way round.")

    c1, c2, c3 = st.columns([2, 2, 1])
    choice = c1.selectbox("Briefing", list(REPORTS))
    market_key = c2.selectbox("Market", markets_mod.keys(cfg),
                              format_func=lambda k: markets_mod.get(cfg, k).label)
    refresh = c3.checkbox("Refresh series", value=False,
                          help="Re-download the macro series instead of using "
                               "the few-hour cache.")

    if not st.button("Build the pack", type="primary"):
        _how_to(cfg, p)
        return

    report_key, build = REPORTS[choice]
    with st.status(f"Building the {choice.lower()} pack...", expanded=True) as s:
        pack = build(cfg, market_key=market_key, force_refresh=refresh,
                     progress=lambda m: s.write(m))
        s.update(label=f"{choice} pack ready", state="complete")

    _pack(pack, p)


def _how_to(cfg, p) -> None:
    st.info(
        "Nothing has been built yet. Press **Build the pack** above, or run "
        "the same thing headless:\n\n"
        "```\npython -m nifty_algo.research macro --market india\n"
        "python -m nifty_algo.research risk --market india --json\n```")
    st.markdown(
        f"**Connectors enabled:** `{'`, `'.join(cfg.portfolio.connectors)}` "
        f"(`PortfolioConfig.connectors`). A connector in that list that cannot "
        f"answer makes the snapshot incomplete and every portfolio percentage "
        f"is then withheld rather than computed against a partial book. One "
        f"that is *not* in the list is never asked and never counted.")
    st.caption(
        "Holdings no broker API reports - your ETFs, another broker's account "
        f"- go in `{cfg.portfolio.manual_path}`. It is gitignored, like every "
        "other file of numbers about your account.")


def _pack(pack, p) -> None:
    if pack.stood_down:
        banner(f"<b>Stood down.</b> {pack.stood_down}", p.critical, "\u26d4")
        return

    if pack.caveats:
        st.subheader("Read these first")
        for c in pack.caveats:
            banner(c, p.warning, "\u26a0")

    for section in pack.sections:
        _section(section, p)

    _closing(pack, p)


def _section(section, p) -> None:
    st.subheader(section.name)

    available = [f for f in section.facts if f.available]
    if available:
        cols = st.columns(min(len(available), 4))
        for col, fact in zip(cols * 4, available):
            col.metric(fact.label, fact.display())
        for fact in available:
            if fact.note:
                st.caption(f"**{fact.label}** — {fact.note}")

    for fact in section.facts:
        if not fact.available:
            # Never a blank cell and never a zero: an absent fact is stated.
            banner(f"<b>{fact.label}: unavailable.</b> {fact.note}",
                   p.muted, "\u2205")

    if section.rows:
        frame = pd.DataFrame(section.rows)
        st.dataframe(frame, width="stretch", hide_index=True)

    if section.note:
        st.caption(section.note)

    for question in section.judgment:
        banner(f"<b>Judgment, not measurement.</b> {question}",
               p.series_2, "◆")


def _closing(pack, p) -> None:
    missing = pack.unavailable()
    with st.expander(f"{len(missing)} fact(s) this run could not establish"):
        if missing:
            st.dataframe(pd.DataFrame(missing)[["section", "label", "note"]],
                         width="stretch", hide_index=True)
        st.caption(
            "A briefing written from this pack must name every line here as "
            "unavailable rather than skip it. An omitted absence reads as a "
            "clean result.")

    with st.expander("The pack as JSON — what a skill reads"):
        st.caption(
            "Copy this, or run the CLI. The `unavailable` and "
            "`judgment_required` lists are at the top level so the two rules "
            "(cite nothing that is not here; name everything missing) can be "
            "checked without walking the tree.")
        st.code(pack.to_json(), language="json")
