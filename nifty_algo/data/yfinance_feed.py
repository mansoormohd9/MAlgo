"""
yfinance feed - the zero-setup fallback.

WHAT THIS IS FOR: proving the pipeline works. Signals compute, the risk engine
approves, alerts route to Telegram, the chart renders. All of that can be
validated today with no broker account.

WHAT THIS IS NOT FOR: taking trades. Yahoo's NSE data is delayed roughly 15
minutes and the intraday bars are frequently revised after the fact. A
15-minute-old price on a strategy whose stop is one ATR wide means the level
you are alerted on may already have been breached and reversed.

`is_delayed = True` propagates to a permanent red banner in the UI and into
the `feed_latency_note` field of every alert, including the ones sent to
Telegram. That is deliberate: the warning has to travel with the alert, not
sit on a settings page you looked at once.
"""
from __future__ import annotations

import pandas as pd

from .base import DataFeed, FeedError


class YFinanceFeed(DataFeed):
    name = "yfinance"
    is_delayed = True
    latency_note = "DELAYED ~15 min (Yahoo) - validate the pipeline, do not trade it"

    #: Yahoo only serves these intraday intervals
    _SUPPORTED = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "60m"}

    def __init__(self, ticker: str = "^NSEI", interval_minutes: int = 5):
        self.ticker = ticker
        if interval_minutes not in self._SUPPORTED:
            raise FeedError(
                f"yfinance does not serve {interval_minutes}m bars; "
                f"supported: {sorted(self._SUPPORTED)}"
            )
        self.interval = self._SUPPORTED[interval_minutes]

    def get_bars(self, lookback_days: int = 5) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:                      # pragma: no cover
            raise FeedError("yfinance is not installed - pip install yfinance") from e

        # Yahoo caps intraday history: 7 days for 1m, 60 days for the rest.
        cap = 7 if self.interval == "1m" else 60
        period = f"{min(max(lookback_days, 1), cap)}d"

        try:
            raw = yf.download(
                self.ticker, period=period, interval=self.interval,
                progress=False, auto_adjust=False,
            )
        except Exception as e:
            raise FeedError(f"yfinance download failed: {e}") from e

        if raw is None or raw.empty:
            raise FeedError(
                f"yfinance returned no data for {self.ticker} at {self.interval}. "
                f"Outside market hours this is expected."
            )

        # yfinance returns a MultiIndex column frame for a single ticker in
        # recent versions; flatten to the plain OHLCV names the contract wants.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns={c: str(c).strip().lower() for c in raw.columns})

        if getattr(raw.index, "tz", None) is not None:
            # Yahoo serves NSE bars in IST already; drop tz for a naive index.
            raw.index = raw.index.tz_convert("Asia/Kolkata").tz_localize(None)

        return self.validate(raw)
