"""
The entry radar: what you could get into, right now, said out loud.

The engine already computed all of this. `EngineState.pending_orders` has
carried strike, lots, entry, stop and target since the day it was written, and
the only thing the UI ever did with it was build a set of keys to decide
whether to draw a button. `build_chain_view()` has scored every near-money
strike against the real risk gates since the daily brief existed, and it was
reachable only from a tab on another page.

So this module renders almost nothing new. What it adds is the sentence the
old UI never said: whether what you are looking at is a trade to take or a
price to know about. The Daily brief showed a CE table and a PE table with an
"If entered now" banner under each and no direction, no signal, and no mention
of the session clock - three different things that all look identical when the
only output is a number.

Every ticket therefore leads with one of three states, in words:

  ACTIONABLE      a strategy fired, the order is parked, the button is live
  REFERENCE ONLY  nothing fired; this is what the engine WOULD pick
  BLOCKED         a day rule is closed; nothing here is enterable at all

`entry_ticket()` is shared with the Daily brief so the two surfaces cannot
drift into describing the same chain differently.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .components import banner
from .theme import Palette
from .state import cached_chain_view
from .. import signals as sig
from ..brief import ChainView
from ..costs import DEFAULT_COSTS
from ..regime import Regime


# ---------------------------------------------------------------- the gate

def entry_gate(engine) -> tuple[bool, str]:
    """
    Can you enter anything at all right now, and if not, why not.

    Reads `engine.state`, never `RiskEngine.check_halt()`: check_halt() calls
    `_halt()` for the latching conditions, so asking it a question from a
    render pass would end the trading day as a side effect of drawing a panel.

    This is a real gap in `approve()`, which knows about strikes and budgets
    and nothing about the clock. Ask it at 15:20 on expiry day and it will hand
    back a perfectly sized order for a session in which you may not trade.
    """
    state = engine.state
    session = engine.risk.session
    cap = engine.cfg.capital

    if state.kill_switch:
        return False, "kill switch tripped — re-arm it before anything else"
    if session.halted:
        return False, f"{session.halt_reason.value.replace('_', ' ')} — done for the day"
    if state.halt_reason and state.halt_reason != "none":
        return False, state.halt_reason.replace("_", " ")

    left = cap.max_entries_per_session - session.entries_taken
    if left <= 0:
        return False, "no entries left today"
    return True, (f"{left} of {cap.max_entries_per_session} entries left · "
                  f"₹{cap.risk_per_trade_rupees:,.0f} risk each")


def _gate_strip(permitted: bool, detail: str, p: Palette) -> None:
    if permitted:
        banner(f"<b>ENTRIES OPEN</b> — {detail}", p.good, "▣")
    else:
        banner(f"<b>ENTRIES BLOCKED</b> — {detail}. Everything below is "
               f"reference only; nothing here can be entered.", p.warning, "⏸")


# ---------------------------------------------------------------- the ticket

def entry_ticket(view: ChainView, p: Palette, cfg, *,
                 bias: str = "", live_alert=None) -> None:
    """
    One side of the chain, stated as an instruction rather than a table.

    `live_alert` is the TradeAlert behind a parked order for this side, when
    a strategy has actually fired. Its presence is the entire difference
    between "do this" and "this is what doing it would look like".
    """
    d = view.approved

    if live_alert is not None and view.entry_permitted:
        label, accent, icon = "ACTIONABLE", p.good, "▲"
        explanation = (f"{live_alert.strategy_label} fired at "
                       f"{live_alert.timestamp:%H:%M} "
                       f"(confidence {live_alert.confidence:.2f}). The order is "
                       f"parked — use **Place order** on its card below.")
    else:
        label, explanation = view.status()
        accent = p.warning if label == "BLOCKED" else p.muted
        if label == "REFERENCE ONLY":
            accent = p.series_1

    st.markdown(
        f"<div class='alert-card' style='--alert-accent:{accent}'>"
        f"<h4>{view.option_type} "
        f"<span class='pill'>{label}</span>"
        f"{f'<span class=pill>{bias}</span>' if bias else ''}</h4>"
        f"<div class='alert-why'>{explanation}</div>"
        f"<div style='margin:.55rem 0;font-size:1.02rem;font-weight:600'>"
        f"{view.headline()}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if d is None:
        _nothing_to_buy(view, p)
        return

    qty = d.quantity
    outlay = d.entry_premium * qty
    capital = cfg.capital.starting_capital
    entry_cost = DEFAULT_COSTS.entry_friction(d.entry_premium, qty)
    round_trip = DEFAULT_COSTS.round_trip(d.entry_premium, d.premium_target, qty)
    breakeven = DEFAULT_COSTS.breakeven_move(d.entry_premium, qty)
    q = d.quote
    rr = (d.rupee_reward / d.rupee_risk) if d.rupee_risk else 0.0

    st.markdown(f"""
<div class="alert-grid">
  <div><span class="lbl">Entry</span><span class="val">{d.entry_premium:,.2f}</span></div>
  <div><span class="lbl">Target</span><span class="val" style="color:{p.good}">{d.premium_target:,.2f}</span></div>
  <div><span class="lbl">Stop</span><span class="val" style="color:{p.critical}">{d.premium_stop:,.2f}</span></div>
  <div><span class="lbl">Quantity</span><span class="val">{qty} ({d.lots} lot)</span></div>
  <div><span class="lbl">Reward</span><span class="val" style="color:{p.good}">+₹{d.rupee_reward:,.0f}</span></div>
  <div><span class="lbl">Risk</span><span class="val" style="color:{p.critical}">−₹{d.rupee_risk:,.0f}</span></div>
  <div><span class="lbl">R:R</span><span class="val">{rr:.2f}</span></div>
  <div><span class="lbl">Outlay</span><span class="val">₹{outlay:,.0f}</span></div>
  <div><span class="lbl">Delta</span><span class="val">{abs(q.delta):.3f}</span></div>
  <div><span class="lbl">Spread</span><span class="val">{q.spread_pct:.2%}</span></div>
  <div><span class="lbl">Open interest</span><span class="val">{q.open_interest:,}</span></div>
  <div><span class="lbl">IV</span><span class="val">{f'{q.iv:.1%}' if q.iv else '—'}</span></div>
</div>
<div class="alert-why">
  Bid {q.bid:,.2f} / ask {q.ask:,.2f} &nbsp;·&nbsp;
  underlying stop {view.stop_points:.0f} pts &nbsp;·&nbsp;
  outlay is {outlay / capital:.0%} of ₹{capital:,.0f} capital &nbsp;·&nbsp;
  costs ₹{entry_cost:,.0f} in, ₹{round_trip:,.0f} round trip, so the premium
  must move +{breakeven:.2f} before you are square.
</div>
<div class="alert-why">{d.sizing_note}</div>
""", unsafe_allow_html=True)

    if not d.runner_enabled:
        banner("Sized down to one lot, so there is <b>no runner</b> — this "
               "trade is all-out at the target with no trail behind it.",
               p.warning, "⚠")

    _runner_ups(view, p)


def _nothing_to_buy(view: ChainView, p: Palette) -> None:
    """
    Why there is no trade here, and what would have to change.

    An enum name on its own ("premium_out_of_range") tells you a rule fired
    but not which strike came closest or by how much it missed, which is the
    only part that helps you judge whether the gate or the market is wrong.
    """
    nearest = min(view.rows, key=lambda r: abs(abs(r.delta) - view.delta_ceiling),
                  default=None)
    extra = ""
    if nearest is not None and nearest.gates:
        extra = (f" Closest was {nearest.strike}{nearest.option_type} at "
                 f"{nearest.premium:,.2f} — it fails on "
                 f"{'; '.join(nearest.gates)}.")
    banner(f"<b>Nothing to buy on this side.</b>{extra}", p.warning, "⛔")


def _runner_ups(view: ChainView, p: Palette) -> None:
    ups = view.runner_ups
    if not ups:
        st.caption("No other strike on this side clears every gate — the pick "
                   "is the only tradeable contract, not the best of several.")
        return
    with st.expander(f"Also tradeable on this side ({len(ups)})"):
        st.caption("Passed on for less delta. Highest delta wins because it is "
                   "the most directional capture per rupee of theta paid.")
        st.dataframe(pd.DataFrame([{
            "Strike": r.strike,
            "Premium": round(r.premium, 2),
            "Delta": round(abs(r.delta), 3),
            "Lots": r.lots,
            "Entry": round(r.entry, 2),
            "Target": round(r.target, 2),
            "Stop": round(r.stop, 2),
            "Risk ₹": round(r.rupee_risk or 0),
            "Reward ₹": round(r.rupee_reward or 0),
        } for r in ups]), width="stretch", hide_index=True)


# ---------------------------------------------------------------- the radar

def _bias(engine) -> tuple[str, str]:
    """
    Which side the day's reading leans towards, as a label only.

    Both sides always render. A regime is a filter on which strategies may
    speak, not a forecast, and hiding the PE ticket because VWAP happens to be
    drifting up would be dressing up a weak signal as a decision.
    """
    reading = engine.state.regime
    if reading is None or reading.regime is Regime.UNKNOWN:
        return "", ""
    if not reading.is_directional or abs(reading.vwap_slope_atr) < 0.25:
        return "", ""
    if reading.vwap_slope_atr > 0:
        return "CE", f"favoured — {reading.regime.value}, VWAP rising"
    return "PE", f"favoured — {reading.regime.value}, VWAP falling"


def entry_radar(engine, cfg, p: Palette) -> None:
    """Gate strip, then a CE and a PE ticket side by side."""
    st.subheader("Can I get in right now?")

    permitted, detail = entry_gate(engine)
    _gate_strip(permitted, detail, p)

    bars = engine.state.bars
    if bars is None or bars.empty:
        st.info("No bars loaded yet — press **Evaluate now** above, or switch "
                "auto-refresh on in the sidebar.")
        return

    spot = float(bars["close"].iloc[-1])
    atr_series = sig.atr(bars, cfg.signal.atr_period)
    stop_points = float(atr_series.iloc[-1]) if not atr_series.empty else 30.0
    stop_points = max(stop_points * cfg.signal.atr_stop_multiple, 1.0)

    favoured_side, bias_note = _bias(engine)
    pending = {(o["option_type"], o["strike"]): o
               for o in engine.state.pending_orders}
    alerts_by_key = {a.dedupe_key: a for a in engine.state.alerts}

    st.caption(
        f"Spot {spot:,.1f} · 1R stop {stop_points:.0f} pts "
        f"({cfg.signal.atr_stop_multiple:g} × ATR{cfg.signal.atr_period}) · "
        f"the stop chooses the strike, so a wider stop forces a lower delta."
    )

    day = bars.index[-1].date()
    session = engine.risk.session
    open_lots = sum(pos.state.lots_remaining for pos in engine.state.open_positions)

    cols = st.columns(2)
    for col, option_type in zip(cols, ("CE", "PE")):
        with col:
            try:
                view = cached_chain_view(
                    spot, option_type, stop_points, day,
                    session.entries_taken, open_lots, permitted,
                    "" if permitted else detail,
                    cfg, engine.chain_provider, engine.risk,
                )
            except Exception as e:
                banner(f"Could not build the {option_type} chain: {e}",
                       p.critical, "⛔")
                continue

            bias = bias_note if option_type == favoured_side else (
                "not favoured by the current reading" if favoured_side else "")

            live_alert = None
            parked = pending.get((option_type, view.approved.quote.strike)) \
                if view.approved else None
            if parked:
                live_alert = alerts_by_key.get(parked["key"])

            entry_ticket(view, p, cfg, bias=bias, live_alert=live_alert)
            _provenance(view, engine, p)


def _provenance(view: ChainView, engine, p: Palette) -> None:
    """
    Where these numbers came from. Not decoration.

    Two of these are quietly true today and invisible in the UI: on the CSV
    provider the feed caches the file and never re-reads it, so spot does not
    move however often you refresh; and the synthetic chain fabricates OI and
    a spread INSIDE the gates it is about to be tested against, which makes
    "every strike passes" a statement about the fixture, not the market.
    """
    verified = engine.chain_provider.expiry_is_verified
    st.caption(
        f"expiry {view.expiry:%d %b} ({view.days_to_expiry}d, "
        f"{'from the broker instrument dump' if verified else 'weekday guess — VERIFY'})"
        f" · chain **{view.source}** · {view.lots} lot(s) → delta ceiling "
        f"{view.delta_ceiling:.3f}"
    )
    if view.is_synthetic:
        banner(view.note, p.warning, "⚠")
    if getattr(engine.feed, "name", "") == "csv":
        st.caption("CSV feed: the file is cached on first read, so refreshing "
                   "will not move this price. Replay only.")
