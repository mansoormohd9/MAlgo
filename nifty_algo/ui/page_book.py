"""
Trade book — what you actually own, and what is protecting it.

THE DIFFERENCE FROM DAILY PICKS. That page shows what the scanner *said*, and
its tracker replays every pick including the ones you skipped, to answer "was
the ranking any good?". This page shows what you *did*: real fills, real
quantities, the real stop, and the money that actually moved. Both are worth
having and neither substitutes for the other.

THE CHECKLIST IS AT THE TOP FOR A REASON. Without DDPI on your Zerodha
account, every delivery sell needs a CDSL TPIN authorisation that expires
nightly - so a stop armed on Monday is REJECTED on Wednesday unless you
authorised that morning. Kite still shows the trigger as active. Nothing on
the broker's own screen tells you the stop is decorative, which is why it has
to be the first thing on this one.

This page renders and decides nothing. Every decision is in `swing/daily.py`,
headless, so a script and this page cannot disagree.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .components import banner
from .state import get_book, get_config, get_equity_broker
from .theme import get_palette
from ..broker import kite_equity as eq_mod
from ..swing import book as book_mod
from ..swing import daily as daily_mod
from ..swing import markets as markets_mod
from ..swing.costs_equity import DEFAULT_EQUITY_COSTS

#: Bars are shared with the Daily picks page, which caches them per market
#: after each scan. Reading them here costs no network call.
def _bars_key(market_key: str) -> str:
    return f"swing_bars_{market_key}"


def render() -> None:
    cfg = get_config()
    p = get_palette()
    broker = get_equity_broker()

    st.subheader("Trade book")
    st.caption(
        "Positions this app put on, what they are worth, and where the stop "
        "actually sits. Distinct from Daily picks, which shows what the "
        "scanner proposed."
    )

    market = markets_mod.get(cfg, markets_mod.INDIA)
    book = get_book(market.key)
    bars = st.session_state.get(_bars_key(market.key), {})

    _mode_banner(cfg, broker, p)
    _checklist(cfg, broker, book, p)
    _daily_run(cfg, book, broker, bars, market, p)

    st.divider()
    _open_positions(book, bars, cfg, broker, p)
    _armed(book, bars, p)
    _findings(p)
    st.divider()
    _performance(book, bars, cfg, p)


# ---------------------------------------------------------------- top strip

def _mode_banner(cfg, broker, p) -> None:
    if broker.dry_run:
        banner("<b>DRY RUN</b> — payloads are journalled, nothing reaches "
               "Zerodha. Turn this off on the Settings page when you are "
               "ready.", p.series_1, "ℹ")
    else:
        banner("<b>LIVE</b> — orders placed from this page spend real money.",
               p.critical, "⚠")


def _checklist(cfg, broker, book, p) -> None:
    """
    The morning ritual, in the order it has to happen.

    Three things, and the app can verify only the first. It says so rather
    than implying the other two are fine.
    """
    st.markdown("#### Before anything else")
    live = book.by_state(book_mod.TicketState.ARMED, book_mod.TicketState.OPEN)
    c1, c2, c3 = st.columns(3)

    # --- 1. Kite session ---
    session = broker.session
    authed = bool(getattr(session, "authenticated", False))
    with c1:
        st.markdown("**1 · Kite session**")
        if authed:
            st.success("Token valid for today.")
        else:
            st.error("No valid token.")
            st.caption(
                "Kite issues one per login and it dies overnight — there is "
                "no refresh token."
            )
            _login_panel(session)

    # --- 2. holdings authorisation ---
    state, why = broker.protection_state()
    with c2:
        st.markdown("**2 · Holdings authorised**")
        if state == eq_mod.PROTECTED:
            st.success("DDPI is active — sells execute without you.")
        elif state == eq_mod.UNVERIFIED:
            st.warning("Marked done for today. Expires tonight.")
        else:
            st.error("Not authorised today.")
        st.caption(why)
        if state != eq_mod.PROTECTED:
            if st.button("I have authorised holdings in Kite",
                         key="mark_authorised", width="stretch"):
                broker.record_authorisation()
                st.rerun()
            st.caption(
                "Enabling **DDPI** once in Zerodha Console removes this step "
                "permanently. It is a free e-sign."
            )

    # --- 3. funds ---
    with c3:
        st.markdown("**3 · Funds**")
        cash = broker.free_cash()
        committed = book.committed_inr
        if cash is None:
            st.warning("Could not read the balance.")
            st.caption("A trigger that fires without funds is rejected.")
        else:
            st.metric("Available", f"₹{cash:,.0f}")
            if committed > cash:
                st.error(
                    f"₹{committed:,.0f} of armed triggers could fire against "
                    f"₹{cash:,.0f}. A GTT reserves no margin."
                )
            elif committed:
                st.caption(f"₹{committed:,.0f} of armed triggers pending.")

    # The banner that must be impossible to scroll past.
    if live and state == eq_mod.UNPROTECTED:
        banner(
            f"<b>{len(live)} live ticket(s) and no authorisation recorded "
            f"today.</b> Any stop that triggers will be rejected by CDSL — "
            f"it will still show as active in Kite.",
            p.critical, "⛔")


def _login_panel(session) -> None:
    """The daily login, inline. It was CLI-only before this page existed."""
    try:
        url = session.login_url()
    except Exception as e:
        st.caption(f"Cannot build a login URL: {e}")
        return
    st.link_button("Open Kite login", url, width="stretch")
    pasted = st.text_input(
        "Paste the redirect URL (or just the request_token)",
        key="kite_request_token",
        help="After logging in, Kite redirects to your registered URL with "
             "?request_token=... in it. Paste the whole address.",
    )
    if pasted and st.button("Exchange for a token", key="kite_exchange"):
        from ..broker.kite_login import extract_request_token
        try:
            session.exchange(extract_request_token(pasted))
        except Exception as e:
            st.error(f"{e}")
            return
        st.session_state.pop("equity_broker", None)
        st.success("Authenticated.")
        st.rerun()


# ---------------------------------------------------------------- the run

def _daily_run(cfg, book, broker, bars, market, p) -> None:
    st.markdown("#### The daily run")
    c1, c2 = st.columns([1, 3])
    with c1:
        go = st.button("Run reconcile & manage", type="primary",
                       width="stretch")
    with c2:
        st.caption(
            "Checks the book against Zerodha, records anything that filled, "
            "arms a stop for anything unguarded, advances the ladder on "
            "today's bar and pushes any ratchet to the broker. Safe to run "
            "twice — every step is idempotent."
        )

    if go:
        if not bars and book.open:
            # Warning and then running anyway would arm exit GTTs off a stale
            # `entry_reference`, advance no ladder at all, and report "all
            # accepted" - the worst of both.
            st.error(
                "No daily bars in memory, and there are open positions. "
                "Run the scan on **Daily picks** first — this page manages "
                "stops off the bars it already downloaded, and without them "
                "it would price them off your entry instead of the market."
            )
            return
        with st.spinner("Reconciling..."):
            outcome = daily_mod.run(
                book, broker, bars, cfg,
                broker_reachable=bool(getattr(broker.session,
                                              "authenticated", False))
                or broker.dry_run)
        st.session_state.book_outcome = outcome
        get_book(market.key, rebuild=True)
        st.rerun()

    outcome = st.session_state.get("book_outcome")
    if outcome is None:
        return
    st.caption(f"Last run {outcome.ran_on:%d %b} · {outcome.headline()}")
    if outcome.actions:
        st.dataframe(
            pd.DataFrame([{
                "": "✓" if a.ok else "✗",
                "What": a.kind.replace("_", " "),
                "Symbol": a.symbol,
                "Detail": a.detail,
            } for a in outcome.actions]),
            hide_index=True, width="stretch",
        )


# ---------------------------------------------------------------- positions

def _open_positions(book, bars, cfg, broker, p) -> None:
    st.markdown("#### Open positions")
    positions = book.open
    if not positions:
        st.info("Nothing open. Arm a ticket from **Daily picks**.")
        return

    today = date.today()
    rows = []
    for t in positions:
        last = _last(bars, t.symbol) or t.entry_reference
        rows.append({
            "Symbol": t.symbol,
            "Qty": f"{t.open_quantity:g}",
            "Cost": f"₹{t.entry_reference:,.2f}",
            "Last": f"₹{last:,.2f}",
            "P&L": f"₹{t.unrealised_inr(last):+,.0f}",
            "R now": f"{t.r_of(last):+.2f}",
            "Stop": f"₹{t.stop_price:,.2f}",
            "To stop": f"{(last - t.stop_price) / last:.1%}" if last else "—",
            "Target": f"₹{t.target:,.2f}",
            "Rung": t.rung,
            "Days": t.days_held(today),
            "Banked": f"₹{t.realised_inr:+,.0f}" if t.realised_inr else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    heat = book.open_risk_r
    cap = cfg.swing.max_open_risk_r
    st.caption(
        f"**Open risk {heat:.2f}R of a {cap:.0f}R cap.** That is what you "
        f"lose if every open position stops out at once — and {cap:.0f}R is "
        f"the day stop the governors are built around. A position whose stop "
        f"has reached breakeven contributes nothing to this."
    )

    with st.expander("Close a position by hand"):
        st.caption(
            "Deletes the resting trigger and sells at a protective limit. "
            "For the case the ladder cannot express: you have decided to be out."
        )
        pick = st.selectbox("Position", [t.symbol for t in positions],
                            key="close_which")
        target = next(t for t in positions if t.symbol == pick)
        last = _last(bars, target.symbol) or target.entry_reference
        if st.button(f"Sell {target.open_quantity:g} {pick}",
                     key="close_now"):
            ok, msg = daily_mod.close_now(target, book, broker, last)
            if not ok:
                st.error(msg)      # no rerun - see page_swing._arm_panel
            else:
                get_book(markets_mod.INDIA, rebuild=True)
                st.rerun()


def _armed(book, bars, p) -> None:
    armed = book.armed
    if not armed:
        return
    st.markdown("#### Armed, waiting to trigger")
    rows = []
    for t in armed:
        last = _last(bars, t.symbol)
        away = ((t.entry - last) / last) if last else None
        rows.append({
            "Symbol": t.symbol,
            "Buy at": f"₹{t.entry:,.2f}",
            "Last": f"₹{last:,.2f}" if last else "—",
            "Away": f"{away:+.1%}" if away is not None else "—",
            "Qty": f"{t.quantity:g}",
            "Cost if it fires": f"₹{t.entry * t.quantity:,.0f}",
            "Valid to": f"{t.valid_until:%d %b}" if t.valid_until else "—",
            "Trigger": t.buy_gtt_id or "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Zerodha is watching these, not this app. They fire whether or not "
        "anything of yours is running."
    )


def _findings(p) -> None:
    outcome = st.session_state.get("book_outcome")
    report = getattr(outcome, "report", None)
    if report is None:
        return

    st.markdown("#### Reconciliation")
    if not report.broker_reachable:
        banner("<b>The broker could not be reached, so nothing was verified.</b> "
               "This is not a clean bill of health.", p.critical, "⚠")
        return
    if report.clean:
        banner(f"<b>{report.headline()}</b>", p.good, "✓")
        return

    banner(f"<b>{report.headline()}</b>",
           p.critical if report.critical else p.warning, "⚠")
    st.dataframe(
        pd.DataFrame([{
            "Severity": f.severity.upper(),
            "Symbol": f.symbol or "—",
            "What": f.message,
            "Do": f.action,
        } for f in report.findings]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "The broker is the source of truth. Nothing here is resolved "
        "silently — a book that quietly corrects itself to match will one "
        "day quietly correct itself to match the wrong thing."
    )


# ---------------------------------------------------------------- results

def _performance(book, bars, cfg, p) -> None:
    st.markdown("#### How the trades you took have performed")
    last_prices = {t.symbol: _last(bars, t.symbol) or 0.0 for t in book.open}
    perf = book_mod.performance(book, last_prices)
    # The DESIGN figure, passed as a fallback only: `headline` uses the payoff
    # this book actually realised once enough trades have closed to have one,
    # and says which of the two it is showing.
    design_breakeven = 1.0 / (1.0 + cfg.capital.reward_risk_ratio)

    st.caption(perf.headline(design_breakeven))
    if not perf.trades and not perf.open_positions:
        return

    c = st.columns(5)
    c[0].metric("Closed", perf.trades)
    yard, basis = perf.yardstick(design_breakeven)
    c[1].metric("Win rate", f"{perf.win_rate:.0%}",
                help=(f"Breakeven is {yard:.0%} ({basis}). The realised figure "
                      f"comes from the payoff these trades actually produced; "
                      f"the design figure is what a "
                      f"{cfg.capital.reward_risk_ratio:.0f}:1 ratio would imply "
                      f"if every trade ran to target."))
    c[2].metric("Expectancy", f"{perf.expectancy_r:+.2f}R")
    c[3].metric("Realised", f"₹{perf.realised_inr:+,.0f}")
    c[4].metric("Unrealised", f"₹{perf.unrealised_inr:+,.0f}")

    closed = book.closed
    if not closed:
        return
    rows = [{
        "Closed": f"{t.closed_on:%d %b}" if t.closed_on else "—",
        "Symbol": t.symbol,
        "Setup": t.setup,
        "In": f"₹{t.entry_reference:,.2f}",
        "R": f"{t.realised_r:+.2f}",
        "P&L": f"₹{t.realised_inr:+,.0f}",
        "Held": t.days_held(t.closed_on) if t.closed_on else "—",
        "Why": t.note,
    } for t in sorted(closed, key=lambda x: x.closed_on or date.min,
                      reverse=True)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    curve = [t.realised_r for t in sorted(closed,
                                          key=lambda x: x.closed_on or date.min)]
    if len(curve) > 1:
        st.line_chart(pd.Series(curve).cumsum(), height=180)
        st.caption("Cumulative R, in the order trades closed.")

    st.caption(
        f"**These figures are GROSS.** Delivery charges are not deducted "
        f"here - a round trip on a ₹10,000 ticket costs about "
        f"₹{DEFAULT_EQUITY_COSTS.friction(500.0, 500.0, 20):,.0f}, roughly a "
        f"tenth of a ₹500 risk budget, and the flat DP fee does not shrink "
        f"with the position. The swing **backtest** nets them off, so subtract "
        f"about 0.1R a trade before comparing the two."
    )


def _last(bars: dict, symbol: str):
    df = bars.get(symbol)
    if df is None or getattr(df, "empty", True):
        return None
    return float(df["close"].iloc[-1])
