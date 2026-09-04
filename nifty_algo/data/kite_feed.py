"""
Zerodha Kite Connect market data.

Kite is the only one of the adapters here with deep intraday history: index
instrument tokens never expire, so NIFTY 50 minute data reaches back years.
That is what makes a real backtest possible, and it is why this feed exists.

WHAT KITE CANNOT DO, AND IT IS NOT A BUG TO BE FIXED LATER:

  Expired option contracts have no historical data. When a weekly contract
  expires its instrument_token is retired and `historical_data()` on it fails.
  Zerodha have said on their own developer forum that they have no plans to
  add this. So there is no route to a premium-level options backtest through
  Kite at any subscription tier. The backtester models premiums instead, and
  says so everywhere it prints a number.

  The same applies to expired FUTURES at intraday resolution - `continuous=True`
  reaches back across expired contracts but only for DAY candles.

VOLUME: the NIFTY 50 index has none, because nothing trades in an index. Kite
returns 0. `signals.has_traded_volume()` detects this and the participation
gates fall back to range-based proxies - see the comment block in signals.py.
`KiteFeed.has_volume` is False for indices so the UI can say so out loud.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from ..broker.kite_auth import KiteSession, NotAuthenticated
from .base import DataFeed, FeedError

# Permanent instrument tokens for NSE indices. These do not expire and are
# stable across sessions, unlike anything in the NFO segment.
NIFTY_50_TOKEN = 256265
INDIA_VIX_TOKEN = 264969

NIFTY_50_SYMBOL = "NSE:NIFTY 50"
INDIA_VIX_SYMBOL = "NSE:INDIA VIX"

INTERVALS = {
    1: "minute",
    3: "3minute",
    5: "5minute",
    10: "10minute",
    15: "15minute",
    30: "30minute",
    60: "60minute",
    # A trading day, expressed in minutes so one map serves every caller.
    # `MAX_DAYS_PER_REQUEST` has carried "day" since the beginning; only this
    # entry was missing, so the daily bars Kite has always served were
    # unreachable through this class. The factor book needs them, and adding
    # the key here is what stops it introducing a SECOND way to build a bar -
    # two bar builders that disagree is a class of bug this repo avoids on
    # purpose (see `bars.fetch_today`).
    1440: "day",
}

# Kite caps how much history one request may span, per interval. Exceeding it
# returns an error rather than a truncated result, so callers must chunk.
MAX_DAYS_PER_REQUEST = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}


class KiteFeed(DataFeed):
    name = "kite"
    is_delayed = False
    latency_note = "real-time (Kite Connect)"

    def __init__(self, session: KiteSession | None = None,
                 instrument_token: int = NIFTY_50_TOKEN,
                 interval_minutes: int = 5,
                 has_volume: bool = False):
        if interval_minutes not in INTERVALS:
            raise FeedError(
                f"Kite has no {interval_minutes}m interval; "
                f"available: {sorted(INTERVALS)}"
            )
        self.session = session or KiteSession()
        self.instrument_token = instrument_token
        self.interval = INTERVALS[interval_minutes]
        # Indices carry no traded volume. Overridable because the same class
        # serves futures tokens, which do.
        self.has_volume = has_volume

    @property
    def configured(self) -> bool:
        return self.session.authenticated

    # ---------------- the DataFeed contract ----------------

    def get_bars(self, lookback_days: int = 5) -> pd.DataFrame:
        end = datetime.now()
        # Calendar days, not trading days: a 5-day lookback over a long weekend
        # would otherwise return two sessions and starve the ATR warm-up.
        start = end - timedelta(days=max(lookback_days, 1) * 2 + 5)
        df = self.fetch(start.date(), end.date())
        if df.empty:
            raise FeedError(
                f"Kite returned no candles for token {self.instrument_token} "
                f"between {start.date()} and {end.date()}"
            )
        return self.validate(df)

    # ---------------- history ----------------

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        """
        Candles between two dates, chunked to respect Kite's per-request window.

        Returns an empty frame rather than raising when the range simply has no
        data (a holiday span), because the fetch scripts walk long ranges and a
        quiet week is not an error.
        """
        kite = self._client()
        span = MAX_DAYS_PER_REQUEST.get(self.interval, 60)

        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=span - 1), end)
            frames.append(self._fetch_chunk(kite, cursor, chunk_end))
            cursor = chunk_end + timedelta(days=1)

        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def _fetch_chunk(self, kite, start: date, end: date) -> pd.DataFrame:
        try:
            candles = kite.historical_data(
                instrument_token=self.instrument_token,
                from_date=start,
                to_date=end,
                interval=self.interval,
            )
        except Exception as e:
            raise FeedError(
                f"Kite historical_data failed for token {self.instrument_token} "
                f"{start}..{end} ({self.interval}): {e}"
            ) from e

        if not candles:
            return pd.DataFrame()
        return self._to_frame(candles)

    @staticmethod
    def _to_frame(candles: list[dict]) -> pd.DataFrame:
        """
        Kite returns tz-aware Asia/Kolkata datetimes. The DataFeed contract is
        tz-NAIVE local time, so strip the zone rather than converting - the whole
        system reasons in IST wall-clock (session windows are `time(9, 15)` etc.)
        and a UTC index would silently shift every session boundary by 5h30m.
        """
        df = pd.DataFrame(candles)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[["open", "high", "low", "close", "volume"]]

    # ---------------- live quote ----------------

    def ltp(self, symbol: str = NIFTY_50_SYMBOL) -> float:
        kite = self._client()
        try:
            data = kite.ltp([symbol])
        except Exception as e:
            raise FeedError(f"Kite ltp failed for {symbol}: {e}") from e
        if symbol not in data:
            raise FeedError(f"Kite returned no LTP for {symbol}")
        return float(data[symbol]["last_price"])

    def _client(self):
        try:
            return self.session.client()
        except NotAuthenticated as e:
            # NotAuthenticated already subclasses FeedError via NotConfigured;
            # re-raised here only to keep the traceback pointing at the feed.
            raise e


class IndiaVixFeed(KiteFeed):
    """
    India VIX daily history, used to calibrate the backtester's implied
    volatility instead of assuming a flat 14% across four years of regimes.
    """
    name = "kite-vix"

    def __init__(self, session: KiteSession | None = None):
        super().__init__(session=session, instrument_token=INDIA_VIX_TOKEN,
                         interval_minutes=5, has_volume=False)
        self.interval = "day"

    def latest(self) -> Optional[float]:
        try:
            return self.ltp(INDIA_VIX_SYMBOL)
        except FeedError:
            return None
