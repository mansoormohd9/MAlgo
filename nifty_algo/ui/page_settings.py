"""Settings: data feed, notification channels, and credential status."""
from __future__ import annotations
import os

import streamlit as st

from .theme import get_palette
from .components import banner
from .state import get_config, get_engine
from ..alerts.channels import build_test_alert
from ..data.factory import PROVIDERS


FEED_NOTES = {
    "csv": "Historical file replayed bar by bar. No live prices — this is how you "
           "validate signals and alert routing without a broker account.",
    "yfinance": "Free, no credentials, DELAYED ~15 minutes. Good enough to prove the "
                "pipeline works; not good enough to trade against a 1-ATR stop.",
    "fyers": "Real-time broker feed. Structure is complete but the auth flow is "
             "untested here — the access token expires daily.",
    "dhan": "Real-time broker feed. Structure is complete but the auth flow is "
            "untested here.",
}


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Settings")

    banner("<b>This app never places an order.</b> It computes the strike, size, "
           "target, and stop, then tells you. There is no broker order path "
           "anywhere in the package — placing orders would need a SEBI Algo-ID "
           "and a whitelisted static IP, and the README's non-negotiable #4 says "
           "not before forty paper sessions.", p.series_1, "ℹ")

    # ---------------- data feed ----------------
    st.subheader("Data feed")
    c1, c2 = st.columns([1, 2])
    provider = c1.selectbox("Provider", PROVIDERS,
                            index=PROVIDERS.index(cfg.data.provider))
    c2.caption(FEED_NOTES.get(provider, ""))

    if provider == "csv":
        cfg.data.csv_path = st.text_input("CSV / Parquet path", value=cfg.data.csv_path)
        st.caption("No file yet? `python -m nifty_algo.data.sample` writes a "
                   "fabricated 8-session sample. It exercises the pipeline; its "
                   "backtest numbers mean nothing.")
    elif provider == "yfinance":
        cfg.data.yfinance_ticker = st.text_input("Ticker",
                                                 value=cfg.data.yfinance_ticker)

    c1, c2, c3 = st.columns(3)
    cfg.data.interval_minutes = c1.selectbox(
        "Bar interval (min)", [1, 3, 5, 15, 30, 60],
        index=[1, 3, 5, 15, 30, 60].index(cfg.data.interval_minutes)
        if cfg.data.interval_minutes in [1, 3, 5, 15, 30, 60] else 2)
    cfg.data.lookback_days = c2.slider("Lookback (days)", 1, 60,
                                       cfg.data.lookback_days)
    cfg.data.poll_seconds = c3.slider("Poll interval (s)", 5, 300,
                                      cfg.data.poll_seconds, 5)

    if provider != cfg.data.provider:
        cfg.data.provider = provider
        if st.button("Apply feed change (resets session risk state)",
                     type="primary"):
            get_engine(rebuild=True)
            st.rerun()
    else:
        if st.button("Reconnect feed"):
            get_engine(rebuild=True)
            st.rerun()

    engine = get_engine()
    if engine:
        st.success(f"Connected: **{engine.feed.name}** — {engine.feed.latency_note}")
    else:
        st.error(f"Feed unavailable: {st.session_state.get('engine_error')}")

    # ---------------- channels ----------------
    st.subheader("Notification channels")
    st.caption(
        "In-app works with zero setup but only while this tab is open. For alerts "
        "that reach you away from the desk, configure Telegram and run the "
        "headless loop: `python -m nifty_algo.run_live --provider yfinance --telegram`"
    )

    a = cfg.alerts
    c1, c2, c3, c4 = st.columns(4)
    a.enable_inapp = c1.toggle("In-app + sound", value=a.enable_inapp)
    a.enable_telegram = c2.toggle("Telegram", value=a.enable_telegram)
    a.enable_desktop = c3.toggle("Desktop toast", value=a.enable_desktop)
    a.enable_email = c4.toggle("Email", value=a.enable_email)

    if engine:
        st.markdown("**Status and channel test**")
        for status, col in zip(engine.dispatcher.channel_status(), st.columns(4)):
            name = status["name"]
            with col:
                if status["configured"]:
                    st.markdown(f"**{name}** &nbsp;<span class='pill'>ready</span>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"**{name}** &nbsp;<span class='pill'>not configured</span>",
                                unsafe_allow_html=True)
                if st.button("Send test", key=f"test_{name}",
                             width="stretch"):
                    notifier = next(n for n in engine.dispatcher.notifiers
                                    if n.name == name)
                    ok, detail = notifier.send(build_test_alert())
                    (st.success if ok else st.error)(detail)

    # ---------------- suppression ----------------
    st.subheader("Alert suppression")
    st.caption(
        f"A {cfg.data.interval_minutes}-minute bar is re-evaluated on every "
        f"{cfg.data.poll_seconds}-second refresh — roughly "
        f"{max(1, cfg.data.interval_minutes * 60 // max(cfg.data.poll_seconds, 1))} "
        f"times. Without de-duplication one breakout would send that many "
        f"messages on every channel."
    )
    c1, c2, c3 = st.columns(3)
    a.dedupe_window_minutes = c1.slider("Dedupe window (min)", 1, 120,
                                        a.dedupe_window_minutes)
    a.per_strategy_cooldown_minutes = c2.slider("Per-strategy cooldown (min)", 0, 90,
                                                a.per_strategy_cooldown_minutes)
    a.force_exit_reminder = c3.toggle(
        f"{cfg.session.force_exit:%H:%M} force-exit reminder",
        value=a.force_exit_reminder,
        help="Non-negotiable #2: flat before close, never carry an option overnight.")

    if engine and st.button("Clear suppression history"):
        engine.dispatcher.reset_suppression()
        st.success("Cleared — the next evaluation may re-alert setups already sent.")

    # ---------------- credentials ----------------
    st.subheader("Credentials")
    st.caption("Read from `.env`. Values are never displayed — only whether each "
               "key is present. Copy `.env.example` to `.env` to fill them in.")

    groups = {
        "Telegram": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"],
        "Email (SMTP)": ["SMTP_HOST", "SMTP_PORT", "SMTP_USER",
                         "SMTP_PASSWORD", "SMTP_TO"],
        "Fyers": ["FYERS_APP_ID", "FYERS_ACCESS_TOKEN"],
        "Dhan": ["DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN"],
    }
    for group, keys in groups.items():
        with st.expander(group):
            for k in keys:
                present = bool(os.getenv(k, "").strip())
                st.markdown(f"{'✅' if present else '⬜'} `{k}` — "
                            f"{'set' if present else 'not set'}")

    st.divider()
    st.caption("`.env` and `journal/` are in `.gitignore`. Broker credentials and "
               "a trade log should never reach a remote.")
