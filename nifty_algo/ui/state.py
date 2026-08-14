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

import streamlit as st

from ..config import Config, DEFAULT
from ..data.factory import build_feed
from ..alerts.channels import (InAppNotifier, TelegramNotifier,
                               DesktopNotifier, EmailNotifier)
from ..alerts.dispatcher import AlertDispatcher
from ..engine import TradingEngine
from ..journal import Journal
from ..strategies.registry import default_enabled_keys


def get_config() -> Config:
    if "cfg" not in st.session_state:
        st.session_state.cfg = DEFAULT
    return st.session_state.cfg


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

    if "engine" in st.session_state:
        return st.session_state.engine

    cfg = get_config()
    journal = get_journal()
    try:
        feed = build_feed(cfg)
    except Exception as e:
        st.session_state.engine_error = str(e)
        return None

    notifiers = [InAppNotifier(), TelegramNotifier(),
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
        from ..broker.kite_auth import KiteSession
        from ..broker.kite_chain import KiteChain
        from ..broker.kite_orders import KiteOrders

        session = KiteSession()
        if not session.authenticated:
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
    st.markdown(
        f'<audio autoplay><source src="data:audio/wav;base64,{alert_sound_b64()}" '
        f'type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )
