"""
Session state: building the engine once and keeping it across reruns.

Streamlit re-executes the whole script on every interaction and every
auto-refresh. Without caching the engine in session_state, each refresh would
build a new RiskEngine - which would reset entries_taken and realised_pnl, so
the session governors would never trip and the "max 3 entries" rule would be
silently unenforced.

It would also construct a new AlertDispatcher, wiping the de-duplication
table, so every refresh would re-send every alert.

Both bugs are invisible in a screenshot and severe in use. This module exists
to prevent them.
"""
from __future__ import annotations
import base64
import io
import math
import struct
import wave
from datetime import date

import streamlit as st

from ..alerts.base import TradeAlert
from ..config import Config, DEFAULT
from ..data.factory import build_feed
from ..alerts.channels import (InAppNotifier, TelegramNotifier,
                               DesktopNotifier, EmailNotifier)
from ..alerts.dispatcher import AlertDispatcher
from ..brief import ChainView, build_chain_view
from ..engine import TradingEngine
from ..journal import Journal
from .. import settings_store
from ..strategies.registry import default_enabled_keys


def get_config() -> Config:
    """
    The one config object, shared by every page.

    Saved settings are applied ONCE, on first access. They have to be applied
    here rather than in `config.py` because `config.py` is version-controlled
    defaults and `data/settings.json` is your account - and a default that
    silently reads a local file is a default nobody can reason about.
    """
    if "cfg" not in st.session_state:
        st.session_state.cfg = DEFAULT
        try:
            st.session_state.settings_notes = settings_store.apply_to(DEFAULT)
        except Exception as e:      # never let a settings file stop the app
            st.session_state.settings_notes = [f"settings could not load: {e}"]
    return st.session_state.cfg


def settings_notes() -> list[str]:
    """Anything `settings_store.apply_to` wanted the user told about."""
    get_config()
    return st.session_state.get("settings_notes", [])


def save_settings() -> None:
    settings_store.save(get_config())


# ---------------------------------------------------------------- swing book

def get_kite_session():
    """
    The shared Kite session, or None.

    One per app run, so the chain, the feed, the option orders and the equity
    book all pass through a single rate limiter - see `broker/ratelimit.py`.
    """
    if "kite_session" not in st.session_state:
        try:
            from ..broker.kite_auth import KiteSession
            st.session_state.kite_session = KiteSession()
        except Exception as e:
            st.session_state.kite_error = str(e)
            st.session_state.kite_session = None
    return st.session_state.kite_session


def get_equity_broker():
    """
    The cash-equity order path. Always returned, authenticated or not.

    Unlike the option broker this is NOT None when Kite is unreachable: in
    dry run it journals payloads with no network at all, and its reads return
    empty rather than raising. A page that cannot render without a broker is
    a page you cannot use to find out why the broker is missing.
    """
    if "equity_broker" not in st.session_state:
        from ..broker.kite_equity import KiteEquity
        st.session_state.equity_broker = KiteEquity(
            session=get_kite_session(), cfg=get_config(),
            journal=get_journal())
    return st.session_state.equity_broker


def get_book(market_key: str = "india", rebuild: bool = False):
    """
    The position ledger, replayed from the journal.

    Cached per market for the life of the script run. `rebuild` after anything
    that writes an event, so the page renders what was just recorded rather
    than the state before it.
    """
    from ..swing.book import Book
    key = f"swing_book_{market_key}"
    if rebuild:
        st.session_state.pop(key, None)
    if key not in st.session_state:
        st.session_state[key] = Book.load(get_journal(), get_config(),
                                          market=market_key)
    return st.session_state[key]


def get_journal() -> Journal:
    if "journal" not in st.session_state:
        st.session_state.journal = Journal()
    return st.session_state.journal


def get_engine(rebuild: bool = False) -> TradingEngine | None:
    """
    The engine, built once. `rebuild` is for when the feed provider changes -
    note that rebuilding deliberately resets the session's risk state, because
    a different data source is a different session.
    """
    if rebuild:
        st.session_state.pop("engine", None)
        st.session_state.pop("engine_error", None)
        # Deliberately NOT ui_alerts. Rebuilding resets the risk session
        # because a different data source is a different session - but the
        # record of what you were told today is yours, and losing it because
        # you reconnected a feed is just losing information.

    if "engine" in st.session_state:
        return st.session_state.engine

    cfg = get_config()
    journal = get_journal()
    try:
        feed = build_feed(cfg)
    except Exception as e:
        st.session_state.engine_error = str(e)
        return None

    # The sink is the whole point of the in-app channel and it was never
    # supplied, so `InAppNotifier` pushed into a list nobody read. Settings ->
    # Send test reported success while nothing appeared anywhere.
    notifiers = [InAppNotifier(sink=push_ui_alert), TelegramNotifier(),
                 DesktopNotifier(), EmailNotifier()]
    dispatcher = AlertDispatcher(notifiers, cfg, journal_fn=journal.write)

    chain_provider, broker = _build_kite(cfg, journal)

    keys = st.session_state.get("enabled_strategies", default_enabled_keys())
    engine = TradingEngine(feed, dispatcher, cfg, keys, journal,
                           chain_provider=chain_provider, broker=broker)
    st.session_state.engine = engine
    st.session_state.engine_error = None
    return engine


def _build_kite(cfg: Config, journal: Journal):
    """
    Attach the real chain and the order path, when Kite is available.

    Returns (chain_provider, broker). Both are None-safe: without credentials
    the engine falls back to the synthetic chain and to alerts-only, which is
    exactly how it behaved before Kite existed. Nothing here can fail in a way
    that stops the engine starting - a broken broker must not cost you the
    ability to see your signals.
    """
    from ..data.chain import ChainProvider
    try:
        from ..broker.kite_chain import KiteChain
        from ..broker.kite_orders import KiteOrders

        # THE SHARED session, not a fresh one. Each `KiteSession` owns its own
        # `RateLimiter`, so building a second here would run two independent
        # limiters against one API key - which is the exact failure
        # `broker/ratelimit.py` exists to prevent.
        session = get_kite_session()
        if session is None or not session.authenticated:
            return ChainProvider(cfg), None

        chain = KiteChain(session, cfg)
        provider = ChainProvider(cfg, broker_chain=chain)
        orders = KiteOrders(session, cfg, chain=chain, journal=journal)
        return provider, orders
    except Exception as e:
        st.session_state.kite_error = str(e)
        return ChainProvider(cfg), None


def engine_error() -> str | None:
    return st.session_state.get("engine_error")


# ---------------------------------------------------------------- alert feed

MAX_FEED = 200


def _feed() -> list[TradeAlert]:
    return st.session_state.setdefault("ui_alerts", [])


def push_ui_alert(alert: TradeAlert) -> None:
    """
    The one front door to the alert feed the UI renders.

    Two things arrive here: alerts the engine produced (copied across by
    `sync_ui_alerts`) and alerts pushed straight through the in-app channel,
    which is how the Settings test button gets to be honest about having
    worked. De-duplication is on (dedupe_key, timestamp) so the same setup
    re-read from the engine on the next refresh does not stack up, while two
    genuine test sends a minute apart both show.
    """
    seen = st.session_state.setdefault("ui_alert_keys", set())
    token = (alert.dedupe_key, alert.timestamp)
    if token in seen:
        return
    seen.add(token)
    feed = _feed()
    feed.append(alert)
    del feed[:-MAX_FEED]


def sync_ui_alerts(engine) -> list[TradeAlert]:
    """Copy anything new out of `engine.state.alerts` into the session feed."""
    if engine is not None:
        for alert in engine.state.alerts:
            push_ui_alert(alert)
    return _feed()


def unannounced_alerts() -> list[TradeAlert]:
    """
    Everything added to the feed since the last time we announced.

    A high-water mark rather than a before/after length comparison, because
    alerts arrive by two routes at two different moments: the in-app channel's
    sink fires deep inside `engine.run_once()`, while `sync_ui_alerts()` runs
    after it. Measuring the length around the run therefore always saw zero
    new alerts and the toast never fired - the notification bug reappearing
    one level up from where it was fixed.
    """
    feed = _feed()
    seen = st.session_state.get("ui_alerts_announced", 0)
    st.session_state.ui_alerts_announced = len(feed)
    return feed[seen:]


# ---------------------------------------------------------------- chain cache

@st.cache_data(ttl=10, show_spinner=False)
def cached_chain_view(spot: float, option_type: str, stop_points: float,
                      day: date, entries_taken: int, open_lots: int,
                      entry_permitted: bool, halt_reason: str,
                      _cfg: Config, _provider, _risk) -> ChainView:
    """
    `build_chain_view` behind a short TTL.

    Each call is one broker round trip and the radar wants both sides, so an
    uncached panel on a 5-second refresh would be 24 chain fetches a minute on
    top of the per-position premium refetch the engine already does.

    Underscore-prefixed arguments are excluded from the cache key by
    convention. `entries_taken` and `open_lots` are in the key on purpose:
    `approve()` consults the session for free capital and the correlation cap,
    so its verdict changes when they do, and a stale "you can buy this" after
    you have already taken the trade is exactly the wrong answer to cache.
    """
    return build_chain_view(spot, option_type, stop_points, _cfg,
                            chain_provider=_provider, risk=_risk, today=day,
                            entry_permitted=entry_permitted,
                            halt_reason=halt_reason)


# ---------------------------------------------------------------- alert sound

@st.cache_data
def alert_sound_b64() -> str:
    """
    A two-tone chime generated in-process and embedded as a data URI.

    Generated rather than shipped as an asset for two reasons: no binary blob
    in the repo, and no external request - the browser can play it with the
    network down.
    """
    rate = 44100
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for freq, dur in ((880.0, 0.14), (1320.0, 0.22)):
            n = int(rate * dur)
            for i in range(n):
                # Short fade in/out so the tone does not click.
                env = min(1.0, i / 400, (n - i) / 900)
                v = int(15000 * env * math.sin(2 * math.pi * freq * i / rate))
                frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def play_alert_sound() -> None:
    """
    Ring the chime.

    Browsers block autoplaying audio until the page has seen a user gesture,
    so this only reliably fires once you have pressed **Enable sound** (the
    click is the gesture). Even then a browser may refuse, which is why the
    chime is never the only notification - `st.toast` and the desktop toast
    carry the same alert on paths that cannot be blocked.
    """
    st.html(
        f'<audio autoplay><source src="data:audio/wav;base64,{alert_sound_b64()}" '
        f'type="audio/wav"></audio>'
    )


def sound_armed() -> bool:
    return bool(st.session_state.get("sound_armed"))
