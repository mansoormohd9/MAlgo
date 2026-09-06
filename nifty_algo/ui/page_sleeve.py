"""
The monthly factor sleeve, as a page.

RENDERS AND DECIDES NOTHING. `factor/sleeve.py` produces the scan, the diff and
the sizing; this file turns them into widgets. That is invariant #4 applied to
the fifth book, and it is why `python -m nifty_algo.factor.sleeve` prints the
same month's decision with no Streamlit in the process.

THE PAGE IS BUILT AROUND A REFUSAL. The sleeve rebalances once a month, and the
single most expensive thing this page could do is make a monthly book feel like
a daily one. So the checklist is rendered read-only and stamped PROVISIONAL on
every day that is not the rebalance date, and the two-window record sits above
the picks rather than in a footnote - a console that shows +18.79% without
+0.60pp beside it is a machine for forgetting the second number.
"""
from __future__ import annotations

import html
from datetime import date

import pandas as pd
import streamlit as st

from ..factor import sleeve as sl
from ..factor.drawdown import DRAWDOWN_HAIRCUT
from ..swing import halal as halal_mod
from .components import banner
from .state import get_config
from .theme import get_palette

_SCAN_KEY = "factor_sleeve_scan"
_ACTIONS_KEY = "factor_sleeve_actions"
_HOLDINGS_KEY = "factor_sleeve_holdings"


def render() -> None:
    p = get_palette()
    cfg = get_config()

    st.title("Monthly sleeve")
    st.caption(
        f"Cross-sectional momentum · top {cfg.factor.top_n} NSE names · "
        f"rebalanced monthly · formation {cfg.factor.formation} · "
        f"band {cfg.factor.band}"
    )

    _record(p)
    _controls(cfg, p)

    scan = st.session_state.get(_SCAN_KEY)
    if scan is None:
        st.info(
            "Press **Run scan** to rank the universe as of today. It reads the "
            "cached daily bars, so it takes seconds — unless the halal screen "
            "is on, which downloads balance sheets for the shortlist the first "
            "time and then caches them."
        )
        return

    _cadence(scan, p)
    _context(scan, p)
    _sizing(cfg, scan, p)
    st.subheader(f"The book the sleeve wants ({len(scan.picks)})")
    _picks_table(scan)
    _concentration(scan, p)
    for pick in scan.picks:
        _pick_card(pick, scan, p)
    _call(scan, p)
    _screened_out(scan, p)
    _caveats(scan, p)


# ------------------------------------------------------------- the record

def _record(p) -> None:
    """
    Both windows, above everything. Never one of them.

    F1 measured +18.79% on 2016-2026 and cleared its gate at p = 0.002. F2's
    kill test on the never-used 2005-2016 window returned +0.60pp over the
    index. Showing the first without the second would make this page an
    advertisement.
    """
    rows = [dict(zip(sl.HEADLINE[0], row)) for row in sl.HEADLINE[1:]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    banner(
        "<b>Read both rows.</b> The upper one is the window the strategy was "
        "built and measured on; the lower one is a decade it had never seen, "
        "where it beat the index by 0.60pp and fell 78.5% taking 81 months to "
        "recover. Survivorship runs one way and flatters both.",
        p.warning, "⚠")


# ------------------------------------------------------------- the controls

def _controls(cfg, p) -> None:
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1.2])

    pot = c1.number_input(
        "Factor pot (₹)", min_value=0.0, step=25_000.0,
        value=float(cfg.capital.factor_capital_inr),
        help="A zero pot sizes every ticket to zero — the scan still runs and "
             "reports, it just cannot size anything.")
    cfg.capital.factor_capital_inr = float(pot)

    screened = c2.toggle(
        "Halal screen", value=bool(cfg.factor.halal_screened),
        help="Screens DOWN the ranking: the highest-ranked names that pass, "
             "looking no further than the shortlist. The recorded returns were "
             "measured WITHOUT it.")
    cfg.factor.halal_screened = bool(screened)

    regime = c3.toggle(
        "Show regime", value=bool(cfg.factor.regime_ma_days),
        help="Reports whether the Nifty is above its 50-day average. Shown as "
             "a fact; it never changes the picks.")
    cfg.factor.regime_ma_days = 50 if regime else 0

    with_news = c4.toggle(
        "Fetch news", value=False,
        help="Headlines for the picks. Context for you — it cannot reorder "
             "the ranking.")

    if c1.button("Run scan", type="primary", width="stretch"):
        _run(cfg, with_news)


def _run(cfg, with_news: bool) -> None:
    prog = st.progress(0.0, text="loading bars")
    try:
        bars, bench = sl.load_bars(cfg)
    except FileNotFoundError as e:
        prog.empty()
        st.error(str(e))
        return

    holdings = None
    try:
        from ..portfolio import aggregate
        prog.progress(0.3, text="reading holdings")
        holdings = aggregate.load(cfg)
    except Exception as e:                                 # pragma: no cover
        st.warning(f"Holdings could not be read ({e}). Every position will "
                   f"show as new — do not act on that.")

    prog.progress(0.5, text="ranking the universe")
    scan = sl.scan(cfg, bars, bench, today=date.today(), holdings=holdings,
                   with_news=with_news,
                   progress=lambda m: prog.progress(0.7, text=m))
    st.session_state[_SCAN_KEY] = scan
    st.session_state[_HOLDINGS_KEY] = holdings
    st.session_state[_ACTIONS_KEY] = sl.decide(scan, holdings)
    prog.empty()


# ------------------------------------------------------------- the cadence

def _cadence(scan, p) -> None:
    """The gate that keeps a monthly book monthly."""
    if scan.next_rebalance is None:
        banner("<b>No rebalance date.</b> The price cache holds no sessions. "
               "Refresh it before acting on anything here.", p.critical, "⛔")
    elif scan.is_rebalance_day:
        banner(f"<b>Rebalance day — {scan.next_rebalance}.</b> The checklist "
               f"below is live. This is the one day a month this book trades.",
               p.series_1, "◆")
    elif scan.sessions_to_rebalance is None:
        banner(f"<b>Provisional.</b> The next rebalance is about "
               f"{scan.next_rebalance}, estimated from the calendar because "
               f"the bar cache ends before it. Refresh the cache, and do not "
               f"trade this list today.", p.warning, "⏳")
    else:
        banner(f"<b>Provisional — {scan.sessions_to_rebalance} sessions until "
               f"{scan.next_rebalance}.</b> This is what the sleeve would want "
               f"if it rebalanced today, and acting on it is how a monthly "
               f"book turns into a daily one. Turnover is this sleeve's "
               f"largest measured cost.", p.warning, "⏳")


def _context(scan, p) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Universe", f"{scan.universe_size:,}",
              help="NSE names in the bar cache")
    c2.metric("Eligible today", f"{scan.eligible:,}",
              help="passed price, history and turnover gates")
    c3.metric("As of", str(scan.as_of))

    if scan.regime.known:
        accent = p.series_1 if scan.regime.above else p.warning
        banner(f"<b>Regime.</b> {html.escape(scan.regime.note)} "
               f"<i>Shown as a fact — it does not change the picks.</i>",
               accent, "◐")

    if not scan.holdings_available:
        banner(f"<b>{html.escape(scan.holdings_note)}.</b> Every position "
               f"below may show as new. A failed read is not an empty "
               f"account, and treating it as one would turn your whole book "
               f"into fresh buys.", p.critical, "⛔")

    st.caption(scan.membership_note)


# -------------------------------------------------------------- the sizing

def _sizing(cfg, scan, p) -> None:
    with st.expander("How much, and what fall you must be able to sit through",
                     expanded=not scan.funded):
        if not scan.funded:
            banner("<b>The pot is zero</b>, so every ticket sizes to zero. "
                   "Set it above. The scan still ranks and screens — it just "
                   "cannot size.", p.warning, "▣")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Pot", f"₹{scan.pot_inr:,.0f}")
            c2.metric("Per name", f"₹{scan.ticket_inr:,.0f}")
            c3.metric("Names", str(scan.top_n))

        planning = min(0.95, sl.MEASURED_DRAWDOWN * DRAWDOWN_HAIRCUT)
        rows = [{"you can sit through": f"{tol:.0%}",
                 "sleeve share of net worth": f"{tol / planning:.0%}"}
                for tol in (0.10, 0.15, 0.20, 0.25)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"Planning drawdown {planning:.0%} — the measured "
            f"{sl.MEASURED_DRAWDOWN:.1%} on month-end marks, haircut ×"
            f"{DRAWDOWN_HAIRCUT:.2f} because the true intra-month trough is "
            f"deeper and one realisation is not a distribution. Recovery took "
            f"{sl.MEASURED_RECOVERY_MONTHS} months."
        )
        banner(
            "<b>There is no stop-loss control here, on purpose.</b> F2 "
            "measured a per-name stop at two widths, two bases and across two "
            "decades. In a book that re-ranks and re-buys every month a stop "
            "is a delay rather than an exit — the name comes straight back at "
            "the next rebalance and you have paid two round trips. Control the "
            "drawdown with the allocation above instead.",
            p.series_1, "✋")


# --------------------------------------------------------------- the picks

def _picks_table(scan) -> None:
    if not scan.picks:
        st.info("Nothing ranked. Check the bar cache is current.")
        return
    rows = []
    for p_ in scan.picks:
        rows.append({
            "#": p_.rank,
            "symbol": p_.symbol,
            "mom 12-1": f"{p_.momentum_12_1:+.0%}",
            "vol 12m": f"{p_.vol_12m:.0%}",
            "off high": f"{p_.from_52w_high:.1%}",
            "ADV ₹cr": f"{p_.turnover_inr / 1e7:,.1f}",
            "index band": p_.index_band,
            "halal": ("—" if p_.halal is None
                      else ("pass" if p_.halal_ok else "FAIL")),
            "held": int(p_.held_qty) if p_.is_held else 0,
            "P&L": "—" if p_.pnl_pct is None else f"{p_.pnl_pct:+.1%}",
            "buy": p_.target_qty,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _concentration(scan, p) -> None:
    """
    Where these names actually live, and what that costs.

    The unscreened sleeve at `band="all"` reaches deep into the micro-cap
    tail. That is what was backtested, and it is also where the fills are
    worst - F1 measured the liquid band at +15.5% against +18.8% for all. A
    ticket you cannot fill is a return you did not get.
    """
    outside = sum(1 for x in scan.picks if x.index_band.startswith("Outside"))
    if not scan.picks or not outside:
        return
    banner(
        f"<b>{outside} of {len(scan.picks)} picks sit outside the Nifty "
        f"500.</b> That is what <code>band={html.escape(scan.band)}</code> "
        f"selects and it is what was backtested — but it is also where the "
        f"fills are worst. F1 measured the liquid band at +15.5% against "
        f"+18.8% for all. Check the ADV column against your ticket size "
        f"before you assume a fill.",
        p.warning, "◑")


def _pick_card(pick, scan, p) -> None:
    held = f" · holding {int(pick.held_qty)}" if pick.is_held else ""
    title = (f"{pick.rank}. {pick.symbol} · {pick.momentum_12_1:+.0%} "
             f"12-1 · {pick.index_band}{held}")
    with st.expander(title):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"₹{pick.price:,.2f}")
        c2.metric("Vol 3m / 12m", f"{pick.vol_3m:.0%} / {pick.vol_12m:.0%}")
        c3.metric("ADV", f"₹{pick.turnover_inr / 1e7:,.1f} cr")
        c4.metric("Target", f"{pick.target_qty} sh"
                            if pick.target_qty else "unfunded")
        st.caption(f"liquidity band: {pick.liquidity_band} · "
                   f"{pick.from_52w_high:.1%} off its 52-week high")
        _halal_panel(pick, p)
        _news_panel(pick, p)


def _halal_panel(pick, p) -> None:
    v = pick.halal
    if v is None:
        st.caption("Halal screen not run for this scan.")
        return
    accent = p.series_1 if v.eligible else p.critical
    banner(f"<b>{'Passes' if v.eligible else 'FAILS'} the screen.</b> "
           f"{html.escape(v.summary)}", accent, "☾")
    if getattr(v, "disagreement", False):
        banner("<b>The two methodologies disagree.</b> FTSE/Yasaar divides by "
               "total assets and AAOIFI/S&P by market cap; SPUS and HLAL "
               "follow different ones. Which standard you follow decides this "
               "name.", p.warning, "⚖")
    checks = getattr(v, "checks", {}) or {}
    if checks:
        st.dataframe(pd.DataFrame([
            {"test": c.get("label", k),
             "value": ("—" if c.get("value") != c.get("value")
                       else f"{c.get('value', 0):.1%}"),
             "limit": f"{c.get('limit', 0):.0%}",
             "passed": "yes" if c.get("passed") else "no"}
            for k, c in checks.items()]), hide_index=True, width="stretch")
    st.caption(halal_mod.DISCLAIMER)


def _news_panel(pick, p) -> None:
    n = pick.news
    if n is None:
        return
    if not getattr(n, "available", False):
        banner(f"<b>News unavailable.</b> {html.escape(getattr(n, 'note', ''))}"
               f" That is not the same as nothing having been published.",
               p.warning, "◌")
        return
    items = getattr(n, "items", []) or []
    if not items:
        st.caption("Nothing notable published in the lookback window.")
        return
    for item in items[:5]:
        st.markdown(f"- [{html.escape(str(getattr(item, 'title', '')))}]"
                    f"({getattr(item, 'link', '#')}) · "
                    f"{getattr(item, 'source', '')}")
    st.caption("News is context for you. It did not reorder the ranking.")


# ---------------------------------------------------------------- the call

def _call(scan, p) -> None:
    actions = st.session_state.get(_ACTIONS_KEY) or []
    trades = [a for a in actions if a.is_trade]
    st.subheader("The call")
    if not trades:
        st.info("Nothing to do — the book already matches the ranking, or the "
                "pot is unset.")
        return

    provisional = trades[0].provisional
    if provisional:
        banner("<b>Provisional — do not trade this today.</b> It becomes a "
               "checklist on the rebalance date shown above.", p.warning, "⏳")

    rows = [{"action": a.kind, "symbol": a.symbol, "qty": a.qty,
             "approx ₹": f"{a.value_inr:,.0f}", "why": a.reason}
            for a in trades]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if not provisional:
        st.download_button(
            "Download the checklist",
            pd.DataFrame(rows).to_csv(index=False).encode(),
            file_name=f"sleeve_{scan.as_of}.csv", mime="text/csv")
    banner("<b>Nothing here places an order.</b> This console is advisory: "
           "you execute it yourself in Kite. Going live on a third book is a "
           "separate decision from building a page that describes it.",
           p.series_1, "✋")


def _screened_out(scan, p) -> None:
    if not scan.screened_out:
        return
    with st.expander(f"Screened out on the way down the ranking "
                     f"({len(scan.screened_out)})"):
        st.dataframe(
            pd.DataFrame([{"symbol": s, "why": w}
                          for s, w in scan.screened_out]),
            hide_index=True, width="stretch")
        if scan.shortlist_short:
            banner(f"<b>The shortlist ran out.</b> Fewer than {scan.top_n} "
                   f"names passed within the shortlist, so the book is "
                   f"lighter than designed. Reaching further down the ranking "
                   f"to fill it would stop being a momentum book.",
                   p.warning, "▣")


def _caveats(scan, p) -> None:
    with st.expander("What this cannot see"):
        for c in scan.caveats:
            st.markdown(f"- {c}")
        if scan.rejections:
            st.caption("Why the rest of the universe was not ranked:")
            st.dataframe(
                pd.DataFrame([{"reason": k, "names": v}
                              for k, v in sorted(scan.rejections.items(),
                                                 key=lambda kv: -kv[1])]),
                hide_index=True, width="stretch")
