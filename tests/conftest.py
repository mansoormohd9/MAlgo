"""Shared fixtures: bar builders that let a test engineer an exact setup."""
from __future__ import annotations
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.strategy import Context


BASE_DAY = datetime(2026, 3, 10, 9, 15)

#: Every credential `ui.auth.bridge_secrets` bridges from `st.secrets` into
#: `os.environ`, plus the two that actually reach money.
CREDENTIAL_KEYS: tuple[str, ...] = (
    "KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD",
    "FYERS_APP_ID", "FYERS_SECRET_KEY",
    "DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN",
)


@pytest.fixture(autouse=True)
def _no_live_credentials(monkeypatch):
    """
    THE SUITE MUST NEVER REACH A BROKER. This is what makes that true.

    `tests/test_auth.py` drives `app.py` through Streamlit's `AppTest`, so
    `load_dotenv()` and `bridge_secrets()` execute and copy the real `.env`
    keys into `os.environ` FOR THE REST OF THE PROCESS. Nothing reset them, so
    `test_portfolio_connectors.py` - which asserts Kite is unconfigured - found
    a genuinely configured Kite and read the live account: real positions, a
    live FX call. No order was placed, but a repo built around keeping code
    away from money by accident should not have its test run touch the account
    at all.

    BLANK, NOT DELETE. `python-dotenv` skips any key already present in
    `os.environ`, so an empty string is what stops `load_dotenv()` re-supplying
    the real value; deleting them leaves the door open on the very next
    AppTest. `tests/test_auth._clean_env` already uses and documents this
    mechanism for the auth keys - this is the same trick applied to the broker
    keys and to the whole suite.

    Function-scoped and autouse, so a test's own `monkeypatch.setenv` still
    wins: `test_auth.py` sets `KITE_API_KEY` deliberately to prove the bridge
    never overwrites a real environment variable, and that must keep working.

    `KiteSession.configured` is `bool(api_key and api_secret)`, both read from
    here, and a cached `.kite_session.json` alone does NOT make it configured -
    so blanking these two closes the path completely rather than narrowing it.
    """
    for key in CREDENTIAL_KEYS:
        monkeypatch.setenv(key, "")


def make_bars(rows: list[dict], start: datetime = BASE_DAY,
              minutes: int = 5) -> pd.DataFrame:
    """Build a frame from explicit OHLCV dicts."""
    idx = [start + timedelta(minutes=minutes * i) for i in range(len(rows))]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def flat_bars(n: int = 40, price: float = 26_000.0, wiggle: float = 8.0,
              volume: float = 100_000, seed: int = 3,
              start: datetime = BASE_DAY) -> pd.DataFrame:
    """
    A calm baseline that establishes ATR and volume history without firing
    anything. Tests append their setup bars to this.
    """
    rng = np.random.default_rng(seed)
    rows = []
    p = price
    for _ in range(n):
        step = rng.normal(0, wiggle * 0.35)
        o, c = p, p + step
        h = max(o, c) + abs(rng.normal(0, wiggle * 0.3))
        l = min(o, c) - abs(rng.normal(0, wiggle * 0.3))
        rows.append({"open": o, "high": h, "low": l, "close": c,
                     "volume": volume})
        p = c
    return make_bars(rows, start=start)


def append_bar(df: pd.DataFrame, open_: float, high: float, low: float,
               close: float, volume: float, minutes: int = 5) -> pd.DataFrame:
    nxt = df.index[-1] + timedelta(minutes=minutes)
    row = pd.DataFrame([{"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}],
                       index=pd.DatetimeIndex([nxt]))
    return pd.concat([df, row])


def make_context(bars: pd.DataFrame, prev_high: float = 0.0,
                 prev_low: float = 0.0, prev_close: float = 0.0,
                 is_expiry: bool = False) -> Context:
    return Context(
        bars=bars,
        now=bars.index[-1].time(),
        prev_day_high=prev_high or float(bars["high"].max()) + 500,
        prev_day_low=prev_low or float(bars["low"].min()) - 500,
        prev_day_close=prev_close or float(bars["close"].iloc[0]),
        is_expiry_day=is_expiry,
    )


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def calm() -> pd.DataFrame:
    return flat_bars()


# ---------------------------------------------------------------- daily bars
#
# The swing scanner reads DAILY bars, not 5-minute ones, so it needs its own
# builders. Same idea as the intraday ones above: a series a test can reason
# about, rather than a fixture file nobody can check by eye.

DAILY_START = datetime(2026, 1, 5)


def daily_bars(closes, volume: float = 1_000_000.0,
               start: datetime = DAILY_START) -> pd.DataFrame:
    """Daily OHLCV from a close series, with plausible wicks around it."""
    idx = [start + timedelta(days=i) for i in range(len(closes))]
    rows, prev = [], closes[0]
    for c in closes:
        o = prev
        rows.append({"open": o, "high": max(o, c) * 1.004,
                     "low": min(o, c) * 0.996, "close": c, "volume": volume})
        prev = c
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def trending(n: int = 140, drift: float = 0.0035, vol: float = 0.010,
             seed: int = 7, start_price: float = 100.0) -> list[float]:
    """A geometric random walk with drift - an uptrend you can seed."""
    rng = np.random.default_rng(seed)
    return list(start_price * np.exp(np.cumsum(rng.normal(drift, vol, n))))


# ---------------------------------------------------------------- the broker
#
# NO TEST MAY EVER REACH A REAL BROKER, and until this existed several did.
#
# `state.get_equity_broker()` builds a `KiteEquity` over a real `KiteSession`,
# which reads `.kite_session.json` from the repo root. On a machine where the
# owner has logged in for the day - which is the whole point of the file - the
# UI tests quietly started making live calls to Zerodha for holdings, GTTs and
# margins. Read-only, but wrong three times over: it spends the account's API
# rate limit, it makes the tests depend on somebody's balance (they began
# failing the day that balance went negative), and a test suite has no
# business touching a live trading account at all.
#
# The fix is to pre-seed the cache `get_equity_broker()` reads, so it never
# constructs one. `offline_broker()` still exercises the REAL `KiteEquity` -
# only the session underneath it is a stub - so the reads go down the genuine
# `_transport._read` failure path and come back empty, which is exactly the
# state a machine with no token is in.

class OfflineSession:
    """A `KiteSession` that is never authenticated and never dials out."""

    authenticated = False

    def __init__(self):
        self.limiter = None

    def client(self):
        from nifty_algo.broker.kite_auth import NotAuthenticated
        raise NotAuthenticated("tests never reach a real broker")

    def login_url(self) -> str:
        return "https://kite.zerodha.com/connect/login?api_key=test"


def offline_broker(cfg=None, journal=None):
    """A real `KiteEquity` over a stub session. Holdings and GTTs come back
    empty; `free_cash()` comes back None, meaning "not checked"."""
    from nifty_algo.broker.kite_equity import KiteEquity
    from nifty_algo.config import Config
    return KiteEquity(OfflineSession(), cfg or Config(), journal)


def seed_offline_broker(at, cfg=None, journal=None):
    """Put an offline broker into an `AppTest`'s session state before it runs.

    Call this on EVERY AppTest that opens a page touching the swing book.
    """
    at.session_state["kite_session"] = OfflineSession()
    at.session_state["equity_broker"] = offline_broker(cfg, journal)
    return at


def sign_in(at):
    """Put an `AppTest` past the login gate before it runs.

    `app.py` calls `auth.require_login()` above the page imports, so without
    this an AppTest renders the sign-in form and nothing else - every page
    assertion would fail on an app that is working correctly. Setting the
    session flag is the same thing a correct password does, and it is
    preferred over `APP_AUTH_DISABLED` because that env var would leak across
    tests and switch the gate off for the one file that tests the gate.
    """
    from nifty_algo.ui import auth
    at.session_state[auth._SIGNED_IN] = True
    return at
