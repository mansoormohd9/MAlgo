"""
Daily picks: the halal-screened Nifty 100 swing scanner, rendered.

Like every other page here, this one decides nothing. `nifty_algo.swing.scanner`
produces the whole result - picks, exclusions and the rejection ledger - and
this module only lays it out.

WHY THE SCAN IS A BUTTON AND NOT AUTOMATIC

Streamlit re-executes the entire script on every interaction, and the sidebar
auto-refresh fires as often as every five seconds. A scan is a hundred-symbol
download plus fifteen news fetches. Running it on script execution would mean
re-downloading the market every time you scrolled, and would get the feeds to
rate-limit within a minute. So the scan runs when you ask for it, the result
lives in session state, and the page never calls `refresh.live()`.

That is also the honest cadence for what this is: daily bars change once a
day. There is nothing here that a five-second refresh could tell you.
"""
from __future__ import annotations

import html
from datetime import date

import pandas as pd
import streamlit as st

from . import charts
from .components import banner
from .state import get_config, get_journal
from .theme import get_palette
from ..swing import halal as halal_mod
from ..swing import prices as prices_mod
from ..swing import scanner as scanner_mod
from ..swing import tracker as tracker_mod
from ..swing import universe as universe_mod

RESULT_KEY = "swing_result"
BARS_KEY = "swing_bars"

STAGE_LABELS = {
    scanner_mod.STAGE_NO_DATA: "No price data",
    scanner_mod.STAGE_TRADEABILITY: "Not tradeable",
    scanner_mod.STAGE_SETUP: "No setup on the last bar",
    scanner_mod.STAGE_REWARD_RISK: "Reward:risk below the floor",
    scanner_mod.STAGE_EARNINGS: "Earnings blackout",
    scanner_mod.STAGE_NEWS: "Vetoed by news",
    scanner_mod.STAGE_SIZING: "Cannot be sized",
    scanner_mod.STAGE_SECTOR: "Out-ranked or sector-capped",
}


def render() -> None:
    p = get_palette()
    cfg = get_config()

    st.title("Daily picks")
    st.caption(
        f"Nifty 100 · halal-screened · cash equity LONG swing · "
        f"top {cfg.swing.top_n} · sized to "
        f"₹{cfg.capital.risk_per_trade_rupees:,.0f} risk at "
        f"{cfg.capital.reward_risk_ratio:.0f}:1 minimum"
    )

    banner(f"<b>Screen, not certification.</b> {halal_mod.DISCLAIMER}",
           p.warning, "☾")
    banner(
        "<b>End-of-day data.</b> These read yesterday's close and give you "
        "stop-triggered entries for today. Prices come from Yahoo, which is "
        "fine for daily bars and is not a live quote — check the price before "
        "you place anything. Not investment advice.",
        p.series_1, "ℹ")

    _controls(cfg, p)

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("Press **Run scan** to sweep the Nifty 100. The first run of "
                "the day downloads balance sheets for the halal screen and "
                "takes a few minutes; later runs reuse the cache and take "
                "seconds.")
        _overrides_editor(cfg, p)
        return

    _freshness(result, cfg, p)
    _tracker(cfg, p)

    st.subheader(f"Today's {len(result.picks)} "
                 f"{'pick' if len(result.picks) == 1 else 'picks'}")
    if not result.picks:
        banner("<b>Nothing qualified today.</b> That is a result, not a "
               "failure — the ledger below accounts for every symbol. A day "
               "with no 2:1 setup is a day to do nothing.", p.warning, "○")
    for i, pick in enumerate(result.picks):
        _pick_card(pick, cfg, p, index=i)

    if result.capital_note:
        accent = (p.warning if "more than your" in result.capital_note
                  else p.series_1)
        banner(result.capital_note, accent, "▣")

    _excluded(result, p)
    _ledger(result, p)
    _overrides_editor(cfg, p)


# ---------------------------------------------------------------- controls

def _controls(cfg, p) -> None:
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.4])

    run = c1.button("Run scan", type="primary", width="stretch")
    fresh_prices = c2.checkbox(
        "Fresh prices", value=False,
        help="Ignore the cached daily bars and download again. The cache is "
             f"{cfg.swing.price_cache_hours}h; daily bars only change once a "
             f"day, so you rarely need this.")
    fresh_fundamentals = c3.checkbox(
        "Refresh balance sheets", value=False,
        help="Re-download the fundamentals behind the halal screen. Slow — "
             "one request per symbol — and they only move once a quarter.")
    skip_news = c4.checkbox(
        "Skip news", value=False,
        help="Run on price alone. News is then EXCLUDED from the ranking "
             "rather than scored as neutral, and no candidate can be vetoed.")

    if c4.button("Refresh universe from NSE", width="stretch"):
        ok, detail = universe_mod.refresh_from_nse(cfg.swing.universe_csv)
        banner(detail, p.good if ok else p.warning, "▣" if ok else "⚠")

    if run:
        _run(cfg, fresh_prices, fresh_fundamentals, skip_news)


def _run(cfg, fresh_prices: bool, fresh_fundamentals: bool,
         skip_news: bool) -> None:
    bar = st.progress(0.0, text="starting")

    def progress(done: int, total: int, label: str) -> None:
        fraction = (done / total) if total else 0.0
        bar.progress(min(1.0, max(0.0, fraction)), text=label)

    try:
        result = scanner_mod.scan(
            cfg, force_prices=fresh_prices,
            force_fundamentals=fresh_fundamentals,
            skip_news=skip_news, progress=progress)
    except Exception as e:
        bar.empty()
        st.error(f"The scan could not complete: {e}")
        return
    bar.empty()

    st.session_state[RESULT_KEY] = result
    # Keep the bars so the charts and the tracker cost no further downloads.
    st.session_state[BARS_KEY] = _reload_bars(cfg, result)

    try:
        tracker_mod.record_scan(get_journal(), result)
    except Exception as e:
        st.warning(f"The picks were produced but not journalled: {e}. "
                   f"The follow-up panel will not see today's scan.")
    st.rerun()


def _reload_bars(cfg, result) -> dict[str, pd.DataFrame]:
    """
    The daily bars, from the cache the scan just wrote.

    Re-read rather than threaded out of the scanner because the tracker needs
    bars for symbols that are no longer in the universe or no longer eligible
    - a pick made three weeks ago still has to be followed to its stop.
    """
    try:
        stocks = universe_mod.load_universe(cfg.swing.universe_csv)
        tickers = {s.symbol: s.yf_ticker for s in stocks}
        return prices_mod.load_prices(tickers, cfg).bars
    except Exception:
        return {}


# ---------------------------------------------------------------- freshness

def _freshness(result, cfg, p) -> None:
    age = result.fundamentals_age_days
    if age is None:
        fundamentals = "balance sheets not cached"
    elif age > cfg.swing.fundamentals_cache_days:
        fundamentals = f"balance sheets {age} days old — stale"
    else:
        fundamentals = f"balance sheets {age} days old"

    st.caption(
        f"Scanned {result.scanned_on:%d %b %Y} · {result.prices_note} · "
        f"{fundamentals} · universe {result.universe_size}, "
        f"halal-eligible {result.eligible_size} · {result.news_note}"
    )
    for w in result.warnings:
        banner(w, p.warning, "⚠")
    if not result.accounts_for_everything():
        banner("<b>The ledger does not add up.</b> Some symbols are neither "
               "picked, excluded nor rejected — treat this scan as incomplete.",
               p.critical, "⛔")


# ---------------------------------------------------------------- tracker

def _tracker(cfg, p) -> None:
    bars = st.session_state.get(BARS_KEY) or {}
    try:
        summary = tracker_mod.open_picks(get_journal(), bars)
    except Exception as e:
        st.caption(f"Could not read past picks: {e}")
        return

    if not summary.outcomes:
        return

    with st.expander(f"Past picks — {summary.headline()}", expanded=True):
        rows = []
        for o in summary.outcomes:
            rows.append({
                "Scanned": f"{o.scanned_on:%d %b}",
                "Symbol": o.symbol,
                "Setup": o.setup,
                "Entry": f"{o.entry:,.2f}",
                "Stop": f"{o.stop:,.2f}",
                "Target": f"{o.target:,.2f}",
                "Outcome": o.outcome.replace("_", " "),
                "R": "—" if o.r_multiple is None else f"{o.r_multiple:+.2f}",
                "P&L": "—" if o.rupees is None else f"₹{o.rupees:+,.0f}",
                "Note": o.note,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "A bar whose range covered both the stop and the target is "
            "recorded as **ambiguous** and counted as a loss — daily data "
            "cannot say which was touched first, and assuming the good half "
            "of that coin flip is how a record flatters itself."
        )


# ---------------------------------------------------------------- pick card

def _pick_card(pick, cfg, p, index: int) -> None:
    s = pick.setup
    accent = p.good
    name = html.escape(pick.name)
    sector = html.escape(pick.sector)

    st.markdown(f"""
<div class="alert-card" style="--alert-accent:{accent}">
  <h4>▲ {html.escape(pick.symbol)} — {name}
      <span class="pill">{html.escape(s.label)}</span>
      <span class="pill">{sector}</span>
      <span class="pill">score {pick.score:.3f}</span></h4>
  <div class="alert-grid">
    <div><span class="lbl">Entry</span><span class="val">{s.entry:,.2f}</span></div>
    <div><span class="lbl">Stop</span><span class="val" style="color:{p.critical}">{s.stop:,.2f}</span></div>
    <div><span class="lbl">Target</span><span class="val" style="color:{p.good}">{s.target:,.2f}</span></div>
    <div><span class="lbl">Qty</span><span class="val">{pick.quantity:,}</span></div>
    <div><span class="lbl">Risk</span><span class="val">₹{pick.rupee_risk:,.0f}</span></div>
    <div><span class="lbl">Reward</span><span class="val">₹{pick.rupee_reward:,.0f}</span></div>
    <div><span class="lbl">R:R</span><span class="val">{pick.reward_risk:.2f}</span></div>
    <div><span class="lbl">Deployed</span><span class="val">₹{pick.deployed:,.0f}</span></div>
    <div><span class="lbl">Stop is</span><span class="val">{s.stop_pct:.2%}</span></div>
    <div><span class="lbl">Target is</span><span class="val">{s.target_pct:.2%}</span></div>
  </div>
  <div class="alert-why"><b>Trigger:</b> {html.escape(s.trigger_note)}</div>
  <div class="alert-why">Last close {pick.last_close:,.2f} &nbsp;·&nbsp;
       valid to {pick.valid_until:%d %b} &nbsp;·&nbsp;
       {html.escape(pick.direction)} only</div>
</div>
""", unsafe_allow_html=True)

    if pick.capital_note:
        banner(pick.capital_note, p.warning, "⚠")

    st.markdown("**Why this one**")
    for line in pick.why():
        st.markdown(f"- {line}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.expander("Score breakdown"):
            _score_table(pick)
    with c2:
        with st.expander("Metrics"):
            _metrics_table(pick)
    with c3:
        with st.expander("News"):
            _news_panel(pick, p)
    with c4:
        with st.expander("Halal screen"):
            _halal_panel(pick, p)

    bars = (st.session_state.get(BARS_KEY) or {}).get(pick.symbol)
    if bars is not None and not bars.empty:
        with st.expander("Chart", expanded=(index == 0)):
            st.plotly_chart(charts.swing_chart(bars, p, pick),
                            width="stretch",
                            key=f"swing_chart_{pick.symbol}")
    st.divider()


def _score_table(pick) -> None:
    rows = []
    for name, part in pick.score_parts.items():
        raw = part.get("raw")
        rows.append({
            "Component": name.replace("_", " ").title(),
            "Raw": "—" if raw is None else f"{raw:.3f}",
            "Weight": f"{part.get('weight', 0):.0%}",
            "Contribution": f"{part.get('contribution', 0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Total {pick.score:.4f}. Every component is shown because a "
               f"ranking you cannot audit is a ranking you cannot argue with.")
    if any(part.get("raw") is None for part in pick.score_parts.values()):
        st.caption("A component with no raw value was unavailable; its weight "
                   "was redistributed over the others rather than scored as "
                   "neutral.")


def _metrics_table(pick) -> None:
    m = pick.metrics
    rows = [
        ("Relative strength, 1 month", _pct(m.get("rs_short"))),
        ("Relative strength, 3 months", _pct(m.get("rs_long"))),
        ("Position in 52-week range", _pct(m.get("pos_52w"), signed=False)),
        ("ATR (14) as % of price", _pct(m.get("atr_pct"), signed=False)),
        ("Volume vs 20-day average", f"{m.get('volume_ratio', 0):.2f}x"),
        ("20-day average turnover", f"₹{m.get('turnover_cr', 0):,.0f} cr"),
        ("Days to results", _earnings(m.get("days_to_earnings"))),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]),
                 width="stretch", hide_index=True)


def _news_panel(pick, p) -> None:
    n = pick.news
    if n is None:
        st.caption("No news read.")
        return
    if not n.available:
        st.caption(f"**News unavailable** — {n.note}. It was excluded from "
                   f"the ranking, not scored as neutral.")
        return

    st.caption(f"{n.summary()} · window {len(n.items)} headlines")
    if not n.items:
        st.caption("Nothing published about this company in the window.")
        return
    for item in n.items[:12]:
        when = f"{item.published:%d %b %H:%M}" if item.published else "undated"
        matched = f" — matched {', '.join(item.matches)}" if item.matches else ""
        link = item.link or ""
        title = html.escape(item.title)
        st.markdown(
            f"- [{title}]({link})  \n"
            f"  <span style='color:{p.muted};font-size:.78rem'>"
            f"{html.escape(item.source)} · {when} · score {item.score:+.2f}"
            f"{html.escape(matched)}</span>",
            unsafe_allow_html=True)
    st.caption("Headlines that matched no phrase are shown but do not vote — "
               "twenty routine items should not dilute one real upgrade.")


def _halal_panel(pick, p) -> None:
    v = pick.halal
    if v is None:
        st.caption("Not screened.")
        return

    st.markdown(f"**Verdict:** eligible — {html.escape(v.summary)}")
    st.caption(f"Decided by: {v.source}"
               + (f" · balance sheet {v.balance_sheet_date}"
                  if v.balance_sheet_date else ""))

    if v.checks:
        rows = []
        for check in v.checks.values():
            value = check.get("value")
            rows.append({
                "Test": check["label"],
                "Value": "not reported" if value != value else f"{value:.1%}",
                "Limit": f"{check['limit']:.0%}",
                "Result": "pass" if check["passed"] else "FAIL",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown(
        f"<span style='color:{p.warning}'>**Haram revenue share: not "
        f"verified.**</span> The ≤5% non-compliant-income test and the "
        f"purification amount need the annual report and are not derivable "
        f"from any free data source. This screen cannot do them.",
        unsafe_allow_html=True)


def _pct(value, signed: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value:+.2%}" if signed else f"{value:.1%}"


def _earnings(days) -> str:
    if days is None:
        return "unknown — Yahoo has no date"
    if days < 0:
        return f"{abs(days)} days ago"
    return f"in {days} days"


# ---------------------------------------------------------------- exclusions

def _excluded(result, p) -> None:
    excluded = result.excluded_halal
    if not excluded:
        return

    # These two are NOT the same statement and must not share a table. One
    # says the company failed the screen; the other says the screen could not
    # be run — a renamed ticker or a Yahoo outage. Presenting a data problem
    # as a compliance ruling on a named company would be a real misstatement.
    failed = [x for x in excluded if not x.verdict.unverifiable]
    unverifiable = [x for x in excluded if x.verdict.unverifiable]

    def table(rows):
        st.dataframe(pd.DataFrame([{
            "Symbol": x.symbol,
            "Company": x.name,
            "Sector": x.sector,
            "Decided by": x.verdict.source.replace("_", " "),
            "Reason": x.verdict.reason,
        } for x in rows]), width="stretch", hide_index=True)

    with st.expander(f"Excluded by the halal screen ({len(failed)} of "
                     f"{result.universe_size})"):
        if failed:
            table(failed)
        st.caption(
            "These failed the activity screen or a balance-sheet ratio. "
            "Disagree with any row by adding it to the override file below — "
            "your ruling beats both."
        )

    if unverifiable:
        with st.expander(f"Could not be screened ({len(unverifiable)}) — "
                         f"a data problem, not a verdict"):
            table(unverifiable)
            st.caption(
                "**These are not rulings that the company is non-compliant.** "
                "The balance sheet could not be read at all — usually a ticker "
                "Yahoo has renamed or dropped. They are excluded because "
                "absent data is a failure to verify, never a pass. Fix the "
                "`yf_ticker` column in `data/nifty100.csv`, or add an override "
                "if you have checked the accounts yourself."
            )


def _ledger(result, p) -> None:
    if not result.rejections:
        return
    counts = result.stage_counts()
    total = sum(counts.values())
    with st.expander(f"Why the other {total} did not make it"):
        summary = pd.DataFrame(
            [{"Stage": STAGE_LABELS.get(k, k), "Symbols": v}
             for k, v in sorted(counts.items(), key=lambda kv: -kv[1])])
        st.dataframe(summary, width="stretch", hide_index=True)

        rows = [{"Symbol": r["symbol"],
                 "Stage": STAGE_LABELS.get(r["stage"], r["stage"]),
                 "Reason": r["reason"]} for r in result.rejections]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "\"Nothing fired\" and \"everything fired and the risk gates "
            "refused all of it\" look identical from the outside and mean "
            "opposite things. This table is the difference."
        )


# ---------------------------------------------------------------- overrides

def _overrides_editor(cfg, p) -> None:
    path = cfg.swing.halal.overrides_csv
    with st.expander("Halal overrides — your final word"):
        st.caption(
            f"Rows here beat both the activity screen and the computed "
            f"ratios, in either direction. Stored in `{path}`. "
            f"`verdict` must be exactly `compliant` or `non_compliant`; "
            f"anything else is ignored and reported as a warning on the next "
            f"scan."
        )
        existing, warnings = halal_mod.load_overrides(path)
        for w in warnings:
            banner(w, p.warning, "⚠")

        rows = [{"symbol": k, "verdict": v["verdict"], "note": v["note"],
                 "reviewed_on": v["reviewed_on"]} for k, v in existing.items()]
        if not rows:
            rows = [{"symbol": "", "verdict": "", "note": "", "reviewed_on": ""}]

        edited = st.data_editor(
            pd.DataFrame(rows), num_rows="dynamic", width="stretch",
            key="halal_overrides_editor",
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", required=False),
                "verdict": st.column_config.SelectboxColumn(
                    "Verdict", options=[halal_mod.VERDICT_COMPLIANT,
                                        halal_mod.VERDICT_NON_COMPLIANT]),
                "note": st.column_config.TextColumn("Why", width="large"),
                "reviewed_on": st.column_config.TextColumn("Reviewed (YYYY-MM-DD)"),
            })

        if st.button("Save overrides"):
            records = [r for r in edited.to_dict("records")
                       if str(r.get("symbol") or "").strip()]
            try:
                halal_mod.save_overrides(path, records)
            except Exception as e:
                st.error(f"Could not write {path}: {e}")
                return
            st.success(f"Saved {len(records)} override(s). Run the scan again "
                       f"for them to take effect.")
