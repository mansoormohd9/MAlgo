"""Strategy selection and parameter tuning."""
from __future__ import annotations

import streamlit as st

from .theme import get_palette
from .components import banner
from .state import get_config, get_engine
from ..strategies.registry import all_strategies, default_enabled_keys


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Strategies")

    st.caption(
        "Ten setups share one contract: a Context in, a Signal out. None of them "
        "can reach an alert without passing `RiskEngine.approve()` first."
    )

    banner(
        "<b>Turning everything on is available and inadvisable.</b> You get three "
        "entries a day. Ten strategies will spend them before 11:00 on whichever "
        "setups happen to fire first, and a losing week becomes unattributable. "
        "The shipped default is four setups that fire on genuinely different "
        "conditions.", p.warning, "⚠")

    infos = all_strategies()
    current = set(st.session_state.get("enabled_strategies", default_enabled_keys()))

    st.subheader("Enabled")
    chosen: list[str] = []
    for info in infos:
        c1, c2 = st.columns([1, 6])
        on = c1.toggle(" ", value=info.key in current, key=f"strat_{info.key}",
                       label_visibility="collapsed")
        if on:
            chosen.append(info.key)
        star = " ⭐" if info.default_enabled else ""
        c2.markdown(
            f"**{info.label}**{star}  \n"
            f"<span style='color:{p.muted};font-size:.85rem'>{info.description}</span>  \n"
            f"<span class='pill'>regimes: {info.regimes_text}</span>",
            unsafe_allow_html=True)

    if chosen != sorted(current, key=lambda k: [i.key for i in infos].index(k)):
        st.session_state.enabled_strategies = chosen
        engine = get_engine()
        if engine:
            engine.set_strategies(chosen)

    st.caption(f"{len(chosen)} enabled. ⭐ marks the shipped defaults.")

    # ---------------- regime gate ----------------
    st.subheader("Regime gate")
    st.caption(
        "A long option is a long-volatility, negative-theta position. In a range "
        "the underlying goes nowhere, IV bleeds, and theta collects against you "
        "every bar — so a breakout system fires into chop and pays friction each "
        "time. This gate is one ATR comparison and it is the highest-value filter "
        "in the system."
    )
    r = cfg.regime
    c1, c2, c3 = st.columns(3)
    r.enforce_regime_gate = c1.toggle("Enforce regime gate", value=r.enforce_regime_gate)
    r.expansion_or_atr_multiple = c2.slider(
        "Expansion threshold (OR width ÷ ATR)", 0.6, 2.5,
        float(r.expansion_or_atr_multiple), 0.1)
    r.range_or_atr_multiple = c3.slider(
        "Range threshold (OR width ÷ ATR)", 0.2, 1.2,
        float(r.range_or_atr_multiple), 0.1)
    if not r.enforce_regime_gate:
        banner("Regime gate is OFF — every enabled strategy may fire on every "
               "day type, including range days.", p.warning, "⚠")

    # ---------------- parameters ----------------
    st.subheader("Parameters")
    st.caption("Everything here lives in `config.py` so a backtest can sweep it. "
               "Changes apply to the next evaluation.")

    t1, t2, t3 = st.tabs(["Shared signal", "Strategy-specific", "Session & risk"])

    with t1:
        s = cfg.signal
        c1, c2 = st.columns(2)
        s.atr_period = c1.slider("ATR period", 5, 40, s.atr_period)
        s.atr_stop_multiple = c2.slider("Stop = ATR ×", 0.5, 3.0,
                                        float(s.atr_stop_multiple), 0.1)
        s.pivot_lookback = c1.slider("Pivot lookback (bars each side)", 2, 12,
                                     s.pivot_lookback)
        s.min_level_touches = c2.slider("Min level touches", 1, 5, s.min_level_touches)
        s.volume_surge_multiple = c1.slider("Volume surge ×", 1.0, 3.0,
                                            float(s.volume_surge_multiple), 0.1)
        s.min_body_to_range = c2.slider("Conviction candle (body ÷ range)", 0.1, 0.9,
                                        float(s.min_body_to_range), 0.05)
        c1.caption(f"Delta window {s.min_delta:.2f}–{s.max_delta:.2f}. "
                   f"A wider stop forces a lower-delta strike — the stop chooses "
                   f"the strike, never the reverse.")

    with t2:
        st_cfg = cfg.strategy
        c1, c2 = st.columns(2)
        st_cfg.min_confidence = c1.slider("Minimum confidence to alert", 0.0, 1.0,
                                          float(st_cfg.min_confidence), 0.05)
        st_cfg.trendline_min_r2 = c2.slider("Trendline min R²", 0.3, 0.99,
                                            float(st_cfg.trendline_min_r2), 0.01)
        st_cfg.vwap_min_distance_atr = c1.slider("VWAP: min prior excursion (ATR)",
                                                 0.0, 1.5,
                                                 float(st_cfg.vwap_min_distance_atr), 0.05)
        st_cfg.sweep_min_wick_atr = c2.slider("Sweep: min wick beyond level (ATR)",
                                              0.1, 1.0,
                                              float(st_cfg.sweep_min_wick_atr), 0.05)
        st_cfg.squeeze_lookback = c1.slider("Squeeze: NR-n lookback", 4, 15,
                                            st_cfg.squeeze_lookback)
        st_cfg.orb_retest_max_bars = c2.slider("ORB: retest window (bars)", 2, 15,
                                               st_cfg.orb_retest_max_bars)
        st_cfg.ema_fast = c1.slider("Pullback: fast EMA", 3, 20, st_cfg.ema_fast)
        st_cfg.ema_slow = c2.slider("Pullback: slow EMA", 10, 60, st_cfg.ema_slow)
        st_cfg.gap_min_atr = c1.slider("Gap: minimum (ATR)", 0.2, 2.0,
                                       float(st_cfg.gap_min_atr), 0.1)
        st_cfg.gap_max_atr = c2.slider("Gap: stand-aside above (ATR)", 1.0, 6.0,
                                       float(st_cfg.gap_max_atr), 0.25)

    with t3:
        cap, sess = cfg.capital, cfg.session
        c1, c2 = st.columns(2)
        cap.starting_capital = c1.number_input("Starting capital (₹)",
                                               value=float(cap.starting_capital),
                                               step=10_000.0)
        cap.max_entries_per_session = c2.slider("Max entries per session", 1, 6,
                                                cap.max_entries_per_session)
        cap.session_target_pct = c1.slider("Session target (%)", 0.02, 0.30,
                                           float(cap.session_target_pct), 0.01)
        cap.session_stop_pct = c2.slider("Session stop (%)", 0.01, 0.15,
                                         float(cap.session_stop_pct), 0.01)
        sess.trade_on_expiry_day = c1.toggle("Trade on expiry day",
                                             value=sess.trade_on_expiry_day)
        if sess.trade_on_expiry_day:
            banner("Expiry-day theta is brutal for buyers — premium can bleed "
                   "faster than the underlying moves in your favour.",
                   p.warning, "⚠")

        st.markdown("**Derived — these stay consistent by construction**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk / trade", f"₹{cap.risk_per_trade_rupees:,.0f}",
                  f"{cap.risk_per_trade_pct:.2%}", delta_color="off")
        m2.metric("Reward / trade", f"₹{cap.reward_per_trade_rupees:,.0f}",
                  f"{cap.reward_per_trade_pct:.2%}", delta_color="off")
        m3.metric("Reward : risk", f"{cap.reward_risk_ratio:.2f} : 1",
                  f"breakeven {1 / (1 + cap.reward_risk_ratio):.1%}",
                  delta_color="off")
        st.caption(
            f"Risk per trade is derived as session stop ÷ max entries, so "
            f"{cap.max_entries_per_session} consecutive losses land exactly on "
            f"the session stop. That is the design — the two rules are the same "
            f"constraint written twice."
        )
