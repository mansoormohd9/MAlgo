"""
Daily picks: the halal-screened swing scanner, rendered, for any market.

ONE PAGE, THREE MARKETS, THREE SEPARATE RESULTS. Each market keeps its own
entry in session state, so switching the selector shows what that market last
produced rather than discarding a scan that took minutes to run. Every panel
below reads the currency and the thresholds off the result's market - nothing
here knows what a rupee is.

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
from .state import (get_book, get_config, get_equity_broker,
                    get_journal)
from .theme import get_palette
from ..swing import crossborder as crossborder_mod
from ..swing import halal as halal_mod
from ..swing import holdings as holdings_mod
from ..swing import markets as markets_mod
from ..swing import prices as prices_mod
from ..swing import scanner as scanner_mod
from ..swing import tracker as tracker_mod
from ..swing import daily as daily_mod
from ..swing.costs_equity import DEFAULT_EQUITY_COSTS
from ..swing import universe as universe_mod

MARKET_KEY = "swing_market"
HOLDINGS_CSV = "data/etf_holdings.csv"


def _result_key(market_key: str) -> str:
    return f"swing_result_{market_key}"


def _bars_key(market_key: str) -> str:
    return f"swing_bars_{market_key}"

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

    market = _market_selector(cfg)
    budget = cfg.capital.risk_inr(market.capital_pool)
    st.caption(
        f"{market.label} · halal-screened · cash equity LONG swing · "
        f"top {cfg.swing.top_n} · sized to ₹{budget:,.0f} risk at "
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

    _controls(cfg, market, p)

    result = st.session_state.get(_result_key(market.key))
    if result is None:
        st.info(f"Press **Run scan** to sweep {market.label}. The first run "
                f"downloads balance sheets for the halal screen and takes a "
                f"few minutes; later runs reuse the cache and take seconds.")
        _overrides_editor(cfg, p)
        return

    if result.stood_down:
        # NOT "nothing qualified". The scan did not run, and conflating the
        # two would report a configuration problem as a market condition.
        banner(f"<b>{html.escape(market.label)} stood down — no scan was "
               f"run.</b> {html.escape(result.stood_down)}", p.critical, "⛔")
        _overrides_editor(cfg, p)
        return

    _freshness(result, market, cfg, p)
    _tracker(cfg, market, p)

    st.subheader(f"Today's {len(result.picks)} "
                 f"{'pick' if len(result.picks) == 1 else 'picks'}")
    if not result.picks:
        banner("<b>Nothing qualified today.</b> That is a result, not a "
               "failure — the ledger below accounts for every symbol. A day "
               "with no 2:1 setup is a day to do nothing.", p.warning, "○")
    overlap_book = holdings_mod.load_holdings(HOLDINGS_CSV)
    for i, pick in enumerate(result.picks):
        _pick_card(pick, cfg, market, p, index=i, overlap_book=overlap_book)

    if result.capital_note:
        accent = (p.warning if "without margin" in result.capital_note
                  else p.series_1)
        banner(result.capital_note, accent, "▣")

    if not market.is_home:
        _crossborder_panel(result, market, p)

    _excluded(result, p, market)
    _ledger(result, p)
    _overrides_editor(cfg, p)


# ---------------------------------------------------------------- market

def _market_selector(cfg):
    keys = markets_mod.keys(cfg)
    labels = {k: cfg.swing.markets[k].label for k in keys}
    current = st.session_state.get(MARKET_KEY, cfg.swing.default_market)
    if current not in keys:
        current = keys[0]

    chosen = st.segmented_control(
        "Market", keys, default=current,
        format_func=lambda k: labels[k], key="swing_market_control")
    # `segmented_control` returns None if the user clicks the active option,
    # which would otherwise silently reset the page to the default market.
    chosen = chosen or current
    st.session_state[MARKET_KEY] = chosen
    return markets_mod.get(cfg, chosen)


# ---------------------------------------------------------------- controls

def _controls(cfg, market, p) -> None:
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

    # The universe refresh differs by market because the sources do. India
    # pulls the index constituents from NSE; the US universe IS the union of
    # the two Shariah ETFs, so refreshing it means refreshing their books.
    if market.key == markets_mod.INDIA:
        if c4.button("Refresh universe from NSE", width="stretch"):
            ok, detail = universe_mod.refresh_from_nse(market.universe_csv)
            banner(detail, p.good if ok else p.warning, "▣" if ok else "⚠")
    elif market.key == markets_mod.US:
        if c4.button("Refresh ETF holdings", width="stretch"):
            ok, detail = holdings_mod.refresh_all(HOLDINGS_CSV)
            banner(detail, p.good if ok else p.warning, "▣" if ok else "⚠")
            st.caption("This updates the overlap check immediately. To rebuild "
                       "the universe itself from the new books, run "
                       "`python scripts/build_universe.py --market us`.")
    else:
        c4.caption("The FTSE 100 list is rebuilt with "
                   "`python scripts/build_universe.py --market uk`.")

    if run:
        _run(cfg, market, fresh_prices, fresh_fundamentals, skip_news)


def _run(cfg, market, fresh_prices: bool, fresh_fundamentals: bool,
         skip_news: bool) -> None:
    bar = st.progress(0.0, text="starting")

    def progress(done: int, total: int, label: str) -> None:
        fraction = (done / total) if total else 0.0
        bar.progress(min(1.0, max(0.0, fraction)), text=label)

    try:
        result = scanner_mod.scan(
            cfg, market=market, force_prices=fresh_prices,
            force_fundamentals=fresh_fundamentals,
            skip_news=skip_news, progress=progress)
    except Exception as e:
        bar.empty()
        st.error(f"The scan could not complete: {e}")
        return
    bar.empty()

    st.session_state[_result_key(market.key)] = result
    # Keep the bars so the charts and the tracker cost no further downloads.
    st.session_state[_bars_key(market.key)] = _reload_bars(cfg, market)

    if result.stood_down:
        st.rerun()

    try:
        tracker_mod.record_scan(get_journal(), result)
    except Exception as e:
        st.warning(f"The picks were produced but not journalled: {e}. "
                   f"The follow-up panel will not see today's scan.")
    st.rerun()


def _reload_bars(cfg, market) -> dict[str, pd.DataFrame]:
    """
    The daily bars, from the cache the scan just wrote.

    Re-read rather than threaded out of the scanner because the tracker needs
    bars for symbols that are no longer in the universe or no longer eligible
    - a pick made three weeks ago still has to be followed to its stop.
    """
    try:
        stocks = universe_mod.load_universe(market.universe_csv)
        tickers = {s.symbol: s.yf_ticker for s in stocks}
        return prices_mod.load_prices(tickers, cfg, market).bars
    except Exception:
        return {}


# ---------------------------------------------------------------- freshness

def _freshness(result, market, cfg, p) -> None:
    age = result.fundamentals_age_days
    if age is None:
        fundamentals = "balance sheets not cached"
    elif age > cfg.swing.fundamentals_cache_days:
        fundamentals = f"balance sheets {age} days old — stale"
    else:
        fundamentals = f"balance sheets {age} days old"

    fx = f" · FX {result.fx_note}" if result.fx_note else ""
    st.caption(
        f"Scanned {result.scanned_on:%d %b %Y} · {result.prices_note} · "
        f"{fundamentals} · universe {result.universe_size}, "
        f"halal-eligible {result.eligible_size} · {result.news_note}{fx}"
    )
    for w in result.warnings:
        banner(w, p.warning, "⚠")
    if not result.accounts_for_everything():
        banner("<b>The ledger does not add up.</b> Some symbols are neither "
               "picked, excluded nor rejected — treat this scan as incomplete.",
               p.critical, "⛔")


# ---------------------------------------------------------------- tracker

def _tracker(cfg, market, p) -> None:
    bars = st.session_state.get(_bars_key(market.key)) or {}
    try:
        summary = tracker_mod.open_picks(get_journal(), bars,
                                         market=market.key)
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
                "P&L": o.money(o.amount),
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

def _pick_card(pick, cfg, market, p, index: int, overlap_book=None) -> None:
    s = pick.setup
    accent = p.good
    name = html.escape(pick.name)
    sector = html.escape(pick.sector)

    # The risk budget is denominated in rupees even when the trade is not, so
    # a foreign ticket carries both. Omitted entirely for India, where showing
    # the same number twice would just be noise.
    inr_line = ""
    if pick.currency != "INR":
        inr_line = (
            f'<div class="alert-why">In rupees: risk '
            f'₹{pick.risk_inr:,.0f} &nbsp;·&nbsp; reward '
            f'₹{pick.reward_inr:,.0f} &nbsp;·&nbsp; deployed '
            f'₹{pick.deployed_inr:,.0f} &nbsp;·&nbsp; at '
            f'₹{pick.fx_inr_per_unit:,.2f}/{html.escape(pick.currency)}</div>')

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
    <div><span class="lbl">Qty</span><span class="val">{pick.qty_text()}</span></div>
    <div><span class="lbl">Risk</span><span class="val">{pick.money(pick.risk_amount)}</span></div>
    <div><span class="lbl">Reward</span><span class="val">{pick.money(pick.reward_amount)}</span></div>
    <div><span class="lbl">R:R</span><span class="val">{pick.reward_risk:.2f}</span></div>
    <div><span class="lbl">Deployed</span><span class="val">{pick.money(pick.deployed)}</span></div>
    <div><span class="lbl">Stop is</span><span class="val">{s.stop_pct:.2%}</span></div>
    <div><span class="lbl">Target is</span><span class="val">{s.target_pct:.2%}</span></div>
  </div>
  <div class="alert-why"><b>Trigger:</b> {html.escape(s.trigger_note)}</div>
  <div class="alert-why">Last close {pick.last_close:,.2f} &nbsp;·&nbsp;
       valid to {pick.valid_until:%d %b} &nbsp;·&nbsp;
       {html.escape(pick.direction)} only</div>
  {inr_line}
</div>
""", unsafe_allow_html=True)

    if pick.capital_note:
        banner(pick.capital_note, p.warning, "⚠")

    _overlap_banner(pick, market, p, overlap_book)

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

    bars = (st.session_state.get(_bars_key(market.key)) or {}).get(pick.symbol)
    if bars is not None and not bars.empty:
        with st.expander("Chart", expanded=(index == 0)):
            # The market is in the key because a keyed chart carries selection
            # state across reruns, and a symbol listed on two exchanges would
            # otherwise share one key when you switch markets.
            st.plotly_chart(charts.swing_chart(bars, p, pick),
                            width="stretch",
                            key=f"swing_chart_{market.key}_{pick.symbol}")

    _arm_panel(pick, cfg, market, p, index)
    st.divider()


def _arm_panel(pick, cfg, market, p, index: int) -> None:
    """
    One click arms a resting BUY trigger at Zerodha. It does not buy now.

    Every entry in this book sits ABOVE the last close, because each setup
    demands the stock prove itself before you pay for it. So the order has to
    WAIT - and Zerodha does the waiting, not this app. That is what makes a
    once-a-day routine sufficient: between the click and the fill, nothing of
    yours needs to be running.

    Only India for now. The US and UK books are IBKR-shaped (fractional
    shares, a different broker entirely) and arming them through Kite would
    place an order on an exchange that does not list them.
    """
    if market.key != markets_mod.INDIA:
        st.caption(
            f"Order placement is wired for India only. {market.label} tickets "
            f"are for you to place with whichever broker holds the foreign "
            f"pool."
        )
        return

    broker = get_equity_broker()
    book = get_book(market.key)
    existing = book.for_symbol(pick.symbol)

    st.markdown("**Place this trade**")
    st.caption(DEFAULT_EQUITY_COSTS.note(pick.setup.entry, pick.setup.stop,
                                         pick.quantity))

    if existing is not None:
        st.info(
            f"Already {existing.state.value} in {pick.symbol} — see the "
            f"**Trade book** page. Two positions in one name is one bet at "
            f"twice the size."
        )
        return

    check = daily_mod.can_arm(pick, book, broker, cfg,
                             free_cash=broker.free_cash())
    for w in check.warnings:
        st.warning(w)
    for b in check.blockers:
        st.error(b)

    if not check.ok:
        return

    label = ("Arm buy trigger (dry run)" if broker.dry_run
             else f"ARM: buy {pick.qty_text()} {pick.symbol} at "
                  f"{pick.setup.entry:,.2f}")
    if st.button(label, key=f"arm_{market.key}_{pick.symbol}",
                 type="primary", width="stretch"):
        ok, msg = daily_mod.arm_pick(pick, book, broker, cfg,
                                     last_price=pick.last_close)
        if not ok:
            # Deliberately NO rerun. A rerun repaints the page and throws the
            # message away, so a rejected order looks exactly like a click
            # that did nothing.
            st.error(msg)
            return
        get_book(market.key, rebuild=True)
        st.success(msg)
        st.rerun()

    st.caption(
        f"Zerodha holds the trigger until {pick.valid_until:%d %b}. It fires "
        f"with your laptop shut. The stop is armed on the **Trade book** page "
        f"once it fills — a buy trigger cannot chain into a sell trigger."
    )


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
        ("20-day average turnover",
         f"{m.get('currency_symbol', '₹')}{m.get('turnover', 0):,.1f} "
         f"{m.get('turnover_unit', 'cr')}"),
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

    _method_comparison(v, p)

    st.markdown(
        f"<span style='color:{p.warning}'>**Haram revenue share: not "
        f"verified.**</span> The ≤5% non-compliant-income test and the "
        f"purification amount need the annual report and are not derivable "
        f"from any free data source. This screen cannot do them.",
        unsafe_allow_html=True)


def _method_comparison(v, p) -> None:
    """
    Both Shariah standards, side by side, and whether they agree.

    Worth its own panel rather than a footnote: this is exactly the split
    between the two US ETFs in the portfolio - HLAL tracks FTSE/Yasaar, which
    divides by total assets, and SPUS tracks AAOIFI/S&P, which divides by
    market cap. When they disagree, the honest statement is that the answer
    depends on which standard you follow, not that the company is clearly one
    thing or the other.
    """
    others = v.other_methods()
    if not others:
        return

    if v.disagreement:
        banner(
            "<b>The two Shariah standards disagree on this name.</b> "
            "Eligibility above follows your primary standard. The other one "
            "reaches the opposite conclusion, which is a fact about the "
            "methodologies rather than about the setup — the same split "
            "separates HLAL (FTSE) from SPUS (AAOIFI).",
            p.warning, "⚖")

    rows = []
    for mv in v.verdicts.values():
        rows.append({
            "Standard": mv.label,
            "Denominator": mv.denominator_label,
            "Result": ("not computable" if not mv.available
                       else "passes" if mv.passed else "FAILS"),
            "Primary": "yes" if mv.method == v.primary_method else "",
            "Detail": "; ".join(mv.failures) if mv.failures else "",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------- overlap

def _overlap_banner(pick, market, p, overlap_book) -> None:
    """
    What you already hold of this name through the Shariah ETFs.

    A BADGE, NEVER A REJECTION. Concentration is a decision and it is yours;
    what this refuses to do is let it be made silently. Suppressed for India,
    where none of the funds have any exposure and the line would be noise.
    """
    if market.is_home or not overlap_book:
        return

    balances = _fund_balances_inr()
    overlap = holdings_mod.overlap_for(pick.symbol, overlap_book, balances)
    if not overlap.any:
        st.caption(f"○ {overlap.note()}")
        return

    heavy = overlap.heavy(0.03)
    banner(html.escape(overlap.note()),
           p.warning if heavy else p.series_1, "◍")


def _fund_balances_inr() -> dict[str, float]:
    """
    What you hold of each fund, in rupees, from the sidebar inputs.

    Empty until you enter them - and an empty map turns the overlap line from
    an amount back into a percentage rather than inventing a balance.
    """
    return {
        etf: float(st.session_state.get(f"fund_balance_{etf}") or 0.0)
        for etf in holdings_mod.SOURCES
    }


# ---------------------------------------------------------------- crossborder

def _crossborder_panel(result, market, p) -> None:
    """
    The frictions that do not show up on a chart.

    Arithmetic with citations, not advice - see `crossborder.py`. Shown only
    for foreign markets, because none of it applies to a domestic trade.
    """
    deployed = sum(pick.deployed for pick in result.picks)
    deployed_inr = sum(pick.deployed_inr for pick in result.picks)

    with st.expander("Cross-border costs and exposure", expanded=False):
        st.caption(crossborder_mod.DISCLAIMER)

        if market.key == markets_mod.UK and deployed:
            costs = crossborder_mod.ticket_costs(market, deployed)
            for label, amount, note in costs.lines:
                st.markdown(f"**{label}: {market.money(amount, 2)}** — {note}")

        already = float(st.session_state.get("lrs_used_inr") or 0.0)
        tcs = crossborder_mod.tcs_on(deployed_inr, already)
        st.markdown(f"**Remitting ₹{deployed_inr:,.0f} for all "
                    f"{len(result.picks)} tickets.** {tcs.note()}")

        st.markdown(f"**Withholding.** "
                    f"{crossborder_mod.withholding_note('US' if market.key == markets_mod.US else 'IE')}")
        st.markdown(f"**Capital gains.** {crossborder_mod.cgt_note()}")
        st.markdown(f"**Reporting.** {crossborder_mod.SCHEDULE_FA_NOTE}")

        st.caption("Sources: " + " · ".join(
            f"[{label}]({url})" for label, url in
            crossborder_mod.SOURCES.values()))


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

def _excluded(result, p, market=None) -> None:
    excluded = result.excluded_halal
    if not excluded:
        return

    # For a universe that IS the constituents of two professionally screened
    # funds, "we excluded a name the funds hold" is the single most
    # informative row on this page. It is either a bug in our taxonomy or a
    # real methodology difference, and both are worth knowing.
    book = holdings_mod.load_holdings(HOLDINGS_CSV) if market and not market.is_home else {}

    # These two are NOT the same statement and must not share a table. One
    # says the company failed the screen; the other says the screen could not
    # be run — a renamed ticker or a Yahoo outage. Presenting a data problem
    # as a compliance ruling on a named company would be a real misstatement.
    failed = [x for x in excluded if not x.verdict.unverifiable]
    unverifiable = [x for x in excluded if x.verdict.unverifiable]

    def table(rows):
        records = []
        for x in rows:
            held = book.get(x.symbol.upper(), [])
            other = [mv for mv in x.verdict.verdicts.values()
                     if mv.method != x.verdict.primary_method and mv.available]
            record = {
                "Symbol": x.symbol,
                "Company": x.name,
                "Sector": x.sector,
                "Decided by": x.verdict.source.replace("_", " "),
                "Reason": x.verdict.reason,
            }
            if book:
                record["Held by"] = ", ".join(h.etf for h in held) or "—"
                record["Other standard"] = (
                    "passes" if any(mv.passed for mv in other)
                    else "fails" if other else "—")
            records.append(record)
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)

    with st.expander(f"Excluded by the halal screen ({len(failed)} of "
                     f"{result.universe_size})"):
        if failed:
            table(failed)
        st.caption(
            "These failed the activity screen or a balance-sheet ratio. "
            "Disagree with any row by adding it to the override file below — "
            "your ruling beats both."
        )
        if book:
            contested = [x for x in failed if book.get(x.symbol.upper())]
            if contested:
                banner(
                    f"<b>{len(contested)} of these are held by SPUS or HLAL.</b> "
                    f"Your universe is those funds' constituents, so every one "
                    f"of these is a disagreement between this screen and a "
                    f"professional Shariah board. The most common cause is the "
                    f"debt test: Yahoo reports Total Debt including capitalised "
                    f"operating leases, while FTSE/Yasaar screens interest-"
                    f"bearing debt only, so lease-heavy businesses fail here and "
                    f"pass there. Check the <i>Other standard</i> column — a "
                    f"name that passes AAOIFI is one SPUS itself holds on that "
                    f"basis. Treat each as a case for the override file, not as "
                    f"a settled ruling.",
                    p.warning, "⚖")

    if unverifiable:
        with st.expander(f"Could not be screened ({len(unverifiable)}) — "
                         f"a data problem, not a verdict"):
            table(unverifiable)
            st.caption(
                "**These are not rulings that the company is non-compliant.** "
                "The balance sheet could not be read at all — usually a ticker "
                "Yahoo has renamed or dropped. They are excluded because "
                "absent data is a failure to verify, never a pass. Fix the "
                "`yf_ticker` column in the market's universe file, or add an "
                "override if you have checked the accounts yourself."
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
