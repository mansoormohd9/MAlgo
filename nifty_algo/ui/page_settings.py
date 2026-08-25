"""Settings: data feed, notification channels, and credential status."""
from __future__ import annotations
import os

import streamlit as st

from .theme import get_palette
from .components import banner
from .state import (get_config, get_engine, get_equity_broker,
                    save_settings, settings_notes)
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

    _order_status_banner(cfg, p)
    for note in settings_notes():
        banner(note, p.warning, "⚠")

    _capital(cfg, p)
    _orders(cfg, p)

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
        "Kite (Zerodha)": ["KITE_API_KEY", "KITE_API_SECRET"],
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


# ---------------------------------------------------------------- capital

def _order_status_banner(cfg, p) -> None:
    """
    What this app can and cannot spend, stated at the top.

    It used to say "this app never places an order", which was true and is
    not any more. A safety notice that has quietly stopped being accurate is
    worse than no notice at all, because it is the one people stop reading.
    """
    eq = cfg.equity_broker
    if eq.dry_run:
        banner(
            "<b>Dry run.</b> The Indian swing book can place cash-equity "
            "orders through Kite, but every payload is journalled instead of "
            "sent. The intraday option book is separately dry-run and is not "
            "wired to this switch.", p.series_1, "ℹ")
    else:
        banner(
            "<b>LIVE.</b> Arming a ticket on Daily picks places a real order "
            "at Zerodha. SEBI's retail algo framework requires a broker "
            "Algo-ID and a whitelisted static IP for automated placement — "
            "check your own obligations with Zerodha.", p.critical, "⚠")


def _capital(cfg, p) -> None:
    st.subheader("Capital")
    st.caption(
        "Three separate pots, one formula. Per-trade risk is always "
        f"`session stop ÷ max entries` = "
        f"{cfg.capital.risk_per_trade_pct:.2%} of whichever pot is paying — "
        "never a second number you set independently, because that is how a "
        "day stop stops meaning three losses."
    )

    c1, c2, c3 = st.columns(3)
    option = c1.number_input(
        "Intraday options (₹)", min_value=0.0, step=10_000.0,
        value=float(cfg.capital.starting_capital), key="cap_option",
        help="The NIFTY option book. Untouched by the swing book.")
    swing = c2.number_input(
        "Indian swing (₹)", min_value=0.0, step=5_000.0,
        value=float(cfg.capital.swing_capital_inr), key="cap_swing",
        help="Cash equity held for days. ₹0 stands the Indian scan down "
             "rather than borrowing the option account's balance.")
    foreign = c3.number_input(
        "Foreign / LRS (₹)", min_value=0.0, step=50_000.0,
        value=float(cfg.capital.foreign_capital_inr), key="cap_foreign",
        help="Remitted under LRS, in another broker. Sizes the US and UK "
             "scans.")

    # NOT assigned to `cfg` yet. `save_settings()` serialises the whole
    # config, so any other button that saves - the DDPI checkbox, the
    # Portfolio page's foreign-capital box - would otherwise persist a number
    # you were only halfway through typing here.
    #
    # WHICH MEANS `swing` IS THE SOURCE OF TRUTH BELOW, NOT `cfg`. Deferring
    # the assignment without updating `_pot_note` is precisely how this page
    # came to divide by zero on the first number anyone typed into it: the
    # guard tested the typed pot and the arithmetic read the saved one.
    _pot_note(cfg, swing, p)

    changed = (option != cfg.capital.starting_capital
               or swing != cfg.capital.swing_capital_inr
               or foreign != cfg.capital.foreign_capital_inr)
    if changed:
        st.caption("Unsaved changes — press Save to apply them.")

    if st.button("Save capital settings", key="save_capital",
                 type="primary" if changed else "secondary"):
        cfg.capital.starting_capital = option
        cfg.capital.swing_capital_inr = swing
        cfg.capital.foreign_capital_inr = foreign
        save_settings()
        st.success("Saved to `data/settings.json` — gitignored, and it "
                   "survives a restart.")
        st.rerun()


def _pot_note(cfg, swing: float, p) -> None:
    """
    What the typed pot buys, and what actually limits the number of positions.

    THE COUNT DOES NOT DEPEND ON THE POT. Cash per position is
    `risk / stop%`, and risk is itself a fixed fraction of the pot, so the
    number of positions a pot funds is `stop% / risk%` - the pot cancels out.
    A Rs 30,000 account and a Rs 5,00,000 account both fund exactly 3.0
    positions at a 5% stop and 1.8 at a 3% stop.

    The first version of this note asked "is the pot big enough for `top_n`?",
    which reads as a question about the pot and is not one. What the pot sets
    is the RUPEE figures; what sets the count is the stop width - and
    counter-intuitively a TIGHTER stop funds FEWER positions, because risk per
    trade is fixed so a closer stop buys more shares.
    """
    if swing <= 0:
        banner("The Indian swing pot is ₹0, so that scan will <b>stand "
               "down</b> rather than size off the option account.",
               p.warning, "⚠")
        return

    # Off the pot you are LOOKING AT, not the one last saved. This previews a
    # number you have not committed yet, so reading `cfg` would quote risk
    # against a different balance - and, when nothing has been saved, against
    # zero, which is how this page came to divide by zero.
    #
    # Still one formula: `CapitalConfig.risk_inr(pool)` is exactly
    # `capital_inr(pool) * risk_per_trade_pct`, so this is that same
    # calculation pointed at the pot in question rather than a second one.
    pct = cfg.capital.risk_per_trade_pct
    risk = swing * pct
    if risk <= 0:
        banner(
            f"Risk per trade works out to <b>0%</b> of the pot, so no ticket "
            f"could be sized at all. That figure is "
            f"<code>session_stop_pct</code> "
            f"({cfg.capital.session_stop_pct:.0%}) divided by "
            f"<code>max_entries_per_session</code> "
            f"({cfg.capital.max_entries_per_session}) - check both in "
            f"<code>config.py</code>.", p.critical, "⛔")
        return

    n = cfg.swing.top_n
    st.caption(
        f"₹{swing:,.0f} gives **₹{risk:,.0f} of risk per trade** "
        f"({pct:.2%} of the pot). How much that *deploys* — and so how many "
        f"of the {n} tickets a scan proposes you can actually pay for — "
        f"depends on where the stop sits, not on the pot: risk is fixed, so a "
        f"**tighter stop buys more shares**."
    )

    rows = ["| Stop | Deployed per position | Tickets funded |",
            "|---|---|---|"]
    for stop_pct in (0.03, 0.05, 0.08):
        per_position = risk / stop_pct
        fits = swing / per_position
        verdict = (f"**all {n}**" if fits >= n
                   else f"{fits:.1f} of {n} — the rest refused for want of cash")
        rows.append(f"| {stop_pct:.0%} | ₹{per_position:,.0f} | {verdict} |")
    st.markdown("\n".join(rows))


# ---------------------------------------------------------------- orders

def _orders(cfg, p) -> None:
    st.subheader("Orders — Indian swing book")
    eq = cfg.equity_broker
    broker = get_equity_broker()

    ddpi = st.checkbox(
        "DDPI (or POA) is active on my Zerodha account",
        value=eq.ddpi_active, key="ddpi_active",
        help="Console → Profile. Without it, every delivery sell needs a "
             "CDSL TPIN authorisation that expires nightly.")
    if ddpi != eq.ddpi_active:
        eq.ddpi_active = ddpi
        # Persisted immediately. `settings_store.apply_to` refuses to restore
        # live orders on an account without DDPI, and that interlock reads
        # this flag - a stale value would let it wave the wrong account
        # through, or block the right one.
        save_settings()

    if not ddpi:
        banner(
            "<b>Without DDPI a resting stop cannot execute on its own.</b> "
            "Every delivery sell needs a CDSL TPIN authorisation valid for "
            "one trading day, so a stop armed on Monday is rejected on "
            "Wednesday unless you re-authorised that morning — and Kite "
            "shows the trigger as active either way. Enabling DDPI is a free "
            "one-time e-sign and removes all of this.", p.critical, "⛔")

    st.markdown("**Live order placement**")
    if eq.dry_run:
        st.caption(
            "Dry run. Every payload is written to the journal exactly as it "
            "would be sent, so you can read what an order *would* have been "
            "before any of them is real."
        )
        typed = st.text_input(
            "Type GO LIVE to enable real orders", key="go_live_confirm",
            help="Deliberately awkward. This is the only thing between the "
                 "app and your money.")
        if typed.strip().upper() == "GO LIVE":
            if st.button("Enable live orders", type="primary",
                         key="enable_live"):
                eq.dry_run = False
                save_settings()
                st.rerun()
    else:
        st.error("Live. Arming a ticket places a real order at Zerodha.")
        if st.button("Back to dry run", key="disable_live"):
            eq.dry_run = True
            save_settings()
            st.rerun()

    st.caption(
        f"Product **{eq.product}** on **{eq.exchange}**, {eq.order_type} "
        f"orders only, tagged `{eq.tag}`. The stop and target rest at Zerodha "
        f"as a two-leg GTT so they survive this app being closed — the "
        f"opposite of the option book, where the stop is recomputed every bar "
        f"and only exists while the engine runs. {broker.mode_label}."
    )
