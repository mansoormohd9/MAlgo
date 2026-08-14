"""
Daily brief page: the day's frame, and the option chain scored by the gates.

Everything rendered here comes from `nifty_algo.brief`, which is also the CLI
(`python -m nifty_algo.brief`). One implementation, two surfaces.
"""
from __future__ import annotations
from datetime import date

import streamlit as st

from . import refresh
from .theme import get_palette
from .components import banner
from .entry_radar import entry_gate, entry_ticket
from .state import get_config, get_engine, engine_error, cached_chain_view
from ..brief import build_preopen, review
from .. import signals as sig


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Daily brief")

    engine = get_engine()
    if engine is None:
        banner(f"<b>Feed unavailable.</b> {engine_error()}", p.critical, "⛔")
        return

    tab_pre, tab_chain, tab_review = st.tabs(
        ["Pre-open", "Option chain", "Review a day"])

    with tab_pre:
        _preopen(engine, cfg, p)

    with tab_chain:
        refresh.live(_chain, engine, cfg, p)

    with tab_review:
        _review(p)


# ---------------------------------------------------------------- pre-open

def _preopen(engine, cfg, p) -> None:
    try:
        pre = build_preopen(cfg, engine.feed, engine.chain_provider)
    except Exception as e:
        st.error(f"Could not build the pre-open brief: {e}")
        return

    st.caption(f"{pre.day:%A %d %B %Y} · feed **{pre.feed_name}**")
    if pre.feed_note:
        banner(pre.feed_note, p.critical, "⚠")
    if pre.expiry_blocks_trading:
        banner("<b>EXPIRY DAY</b> — entries are blocked. Theta on expiry day is "
               "brutal for buyers, which is why "
               "<code>trade_on_expiry_day</code> defaults to off.",
               p.warning, "⏸")
    if not pre.has_volume:
        banner("This is an <b>index series with no traded volume</b>. The "
               "participation gates fall back to range expansion and VWAP "
               "degrades to a session TWAP. Same claim, different measurement.",
               p.warning, "⚠")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prior close", f"{pre.prior_close:,.1f}",
              f"H {pre.prior_high:,.0f} / L {pre.prior_low:,.0f}",
              delta_color="off")
    c2.metric("ATR(14)", f"{pre.atr:,.1f} pts",
              "your 1R on the underlying", delta_color="off")
    c3.metric("Gap", f"{pre.gap_points:+,.1f}",
              f"{pre.gap_atr:+.2f} ATR", delta_color="off")
    c4.metric("India VIX", f"{pre.vix:.2f}" if pre.vix else "—",
              f"{pre.vix / 100:.1%} IV" if pre.vix else "run scripts/fetch_vix.py",
              delta_color="off")

    st.subheader("Today's budget")
    b = pre.budget
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Day target", f"₹{b['target']:,.0f}", f"+{b['target_pct']:.0%}",
              delta_color="off")
    d2.metric("Day floor", f"₹{b['floor']:,.0f}",
              f"{abs(b['floor_pct']):.0%} — ratchets up as you bank",
              delta_color="off")
    d3.metric("Per trade", f"₹{b['risk']:,.0f} risk",
              f"₹{b['reward']:,.0f} reward at {b['rr']:.0f}:1", delta_color="off")
    d4.metric("Entries", b["entries"],
              f"breakeven costs {b['breakeven_move']:.2f} pts", delta_color="off")

    if pre.expiry:
        st.caption(f"Expiry {pre.expiry:%d %b} ({pre.days_to_expiry} days). "
                   f"Source: {'broker instrument dump' if engine.chain_provider.expiry_is_verified else 'weekday guess — VERIFY'}.")


# ---------------------------------------------------------------- the chain

def _chain(engine, cfg, p) -> None:
    refresh.mark_tick()

    st.caption(
        "**Nothing on this tab is a signal.** It answers a different question: "
        "if a CE or a PE setup appeared right now, at the stop width below, "
        "which strike would the engine buy and what would that cost you. "
        "Both sides are always shown, because the chain does not have an "
        "opinion about direction — the strategies do, and they speak on the "
        "**Live alerts** page."
    )
    st.caption(
        "Gate verdicts come from `RiskEngine.gate_failures()` and every "
        "entry/target/stop from `RiskEngine.approve()` — the same calls the "
        "live engine makes, not a parallel calculation that could drift."
    )

    bars = engine.state.bars
    if bars is None or bars.empty:
        st.info("Run an evaluation on the **Live alerts** page first.")
        return

    spot = float(bars["close"].iloc[-1])
    atr_series = sig.atr(bars, cfg.signal.atr_period)
    default_stop = float(atr_series.iloc[-1]) if not atr_series.empty else 30.0

    c1, c2 = st.columns([1, 3])
    stop_points = c1.number_input(
        "Stop width (underlying points)", min_value=1.0, max_value=300.0,
        value=round(max(default_stop, 1.0), 1), step=1.0,
        help="1R. Wider stops force a lower delta ceiling, and past a point "
             "no strike survives at two lots — which disables the runner.")

    permitted, gate_detail = entry_gate(engine)
    if not permitted:
        banner(f"<b>Entries are blocked</b> — {gate_detail}. The tickets below "
               f"are priced correctly and cannot be acted on today.",
               p.warning, "⏸")

    day = bars.index[-1].date()
    session = engine.risk.session
    open_lots = sum(pos.state.lots_remaining for pos in engine.state.open_positions)

    for option_type in ("CE", "PE"):
        st.subheader(f"{option_type} — spot {spot:,.1f}")
        try:
            view = cached_chain_view(
                spot, option_type, stop_points, day,
                session.entries_taken, open_lots, permitted,
                "" if permitted else gate_detail,
                cfg, engine.chain_provider, engine.risk,
            )
        except Exception as e:
            st.error(f"Could not build the {option_type} chain: {e}")
            continue

        entry_ticket(view, p, cfg)

        verified = engine.chain_provider.expiry_is_verified
        st.caption(
            f"expiry {view.expiry:%d %b} ({view.days_to_expiry}d, "
            f"{'broker instrument dump' if verified else 'weekday guess — VERIFY'})"
            f" · source **{view.source}** · {view.lots} lot(s) → delta ceiling "
            f"{view.delta_ceiling:.3f}")
        if view.is_synthetic:
            banner(view.note, p.warning, "⚠")

        st.dataframe(
            view.to_frame(), width="stretch", hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn(format="%d"),
                "Action": st.column_config.TextColumn(
                    width="medium",
                    help="What buying this row would mean."),
                "Spread%": st.column_config.NumberColumn(format="percent"),
                "IV": st.column_config.NumberColumn(format="percent"),
                "OI": st.column_config.NumberColumn(format="localized"),
                "Entry": st.column_config.NumberColumn(format="%.2f"),
                "Target": st.column_config.NumberColumn(format="%.2f"),
                "Stop": st.column_config.NumberColumn(format="%.2f"),
                "Risk": st.column_config.NumberColumn("Risk ₹", format="%d"),
                "Reward": st.column_config.NumberColumn("Reward ₹", format="%d"),
                "Outlay": st.column_config.NumberColumn("Outlay ₹", format="%d"),
                "Why not": st.column_config.TextColumn(
                    width="large",
                    help="The gate this strike fails, and by how much. Blank "
                         "means it is buyable."),
            })

        buyable = sum(1 for r in view.rows if r.viable)
        st.caption(
            f"{buyable} of {len(view.rows)} strikes are actually buyable — "
            f"every row with an Entry is a trade you could take, at the size "
            f"and risk shown."
            + ("  These are synthetic quotes whose OI and spread were "
               "fabricated inside the gates they are tested against, so "
               "'passes' here means very little."
               if view.is_synthetic else ""))


# ---------------------------------------------------------------- review

def _review(p) -> None:
    day = st.date_input("Trading day", value=date.today())
    lines = review(day if isinstance(day, date) else date.today())
    st.code("\n".join(lines), language=None)
