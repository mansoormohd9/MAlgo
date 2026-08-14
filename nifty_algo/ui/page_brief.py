"""
Daily brief page: the day's frame, and the option chain scored by the gates.

Everything rendered here comes from `nifty_algo.brief`, which is also the CLI
(`python -m nifty_algo.brief`). One implementation, two surfaces.
"""
from __future__ import annotations
from datetime import date

import pandas as pd
import streamlit as st

from .theme import get_palette
from .components import banner
from .state import get_config, get_engine, engine_error
from ..brief import build_chain_view, build_preopen, review
from ..data.chain import ChainProvider
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
        _chain(engine, cfg, p)

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
                   f"Source: {'broker instrument dump' if engine.chain_provider._broker_chain else 'weekday guess — VERIFY'}.")


# ---------------------------------------------------------------- the chain

def _chain(engine, cfg, p) -> None:
    st.caption(
        "Every strike near the money, and **which gate each one fails**. The "
        "verdicts come from `RiskEngine.gate_failures()` and the entry/target/"
        "stop from `RiskEngine.approve()` — the same calls the live engine "
        "makes, not a parallel calculation that could drift."
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

    for option_type in ("CE", "PE"):
        st.subheader(f"{option_type} — spot {spot:,.1f}")
        try:
            view = build_chain_view(spot, option_type, stop_points, cfg,
                                    chain_provider=engine.chain_provider,
                                    risk=engine.risk)
        except Exception as e:
            st.error(f"Could not build the {option_type} chain: {e}")
            continue

        st.caption(f"expiry {view.expiry:%d %b} · source **{view.source}** · "
                   f"{view.lots} lot(s) → delta ceiling {view.delta_ceiling:.3f}")
        if view.is_synthetic:
            banner(view.note, p.warning, "⚠")

        frame = view.to_frame()
        st.dataframe(
            frame.style.map(
                lambda v: (f"color:{p.good};font-weight:600"
                           if v == "SELECTED" else
                           (f"color:{p.muted}" if v != "ok" else "")),
                subset=["Verdict"]),
            width="stretch", hide_index=True)

        failed = sum(1 for r in view.rows if r.gates)
        st.caption(f"{len(view.rows) - failed} of {len(view.rows)} strikes pass "
                   f"every gate.")

        from ..risk import ApprovedOrder
        if isinstance(view.decision, ApprovedOrder):
            d = view.decision
            banner(f"<b>If entered now:</b> BUY {d.lots} lot(s) × "
                   f"{d.quote.strike}{d.quote.option_type} = {d.quantity} qty · "
                   f"entry {d.entry_premium:.2f} · target {d.premium_target:.2f} · "
                   f"stop {d.premium_stop:.2f} · {d.sizing_note}",
                   p.good if d.runner_enabled else p.warning, "▣")
        else:
            banner(f"<b>No tradeable strike:</b> {view.decision.reason.value} — "
                   f"{view.decision.detail}", p.warning, "⏸")


# ---------------------------------------------------------------- review

def _review(p) -> None:
    day = st.date_input("Trading day", value=date.today())
    lines = review(day if isinstance(day, date) else date.today())
    st.code("\n".join(lines), language=None)
