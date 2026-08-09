"""
Fyers and Dhan adapters.

STATUS: structure complete, auth untested. Neither could be exercised without
live credentials, so every method that needs one raises `NotConfigured` with
the exact environment variable it wants rather than failing obscurely later.

Both brokers are registered under the SEBI retail algo framework, which is the
compliance checklist item in the README. Note that using these for DATA only
requires nothing extra; it is order PLACEMENT that needs the Algo-ID and the
whitelisted static IP. This system never places an order, so data access is
all you need here.

To finish either adapter you supply the credentials in `.env` and verify the
response shape against the broker's current docs - the JSON key names below
follow their published formats but those change without much warning.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta

import pandas as pd

from .base import DataFeed, FeedError, NotConfigured


class FyersFeed(DataFeed):
    name = "fyers"
    is_delayed = False
    latency_note = "real-time (broker feed)"

    _RESOLUTION = {1: "1", 2: "2", 3: "3", 5: "5", 10: "10",
                   15: "15", 30: "30", 60: "60"}

    def __init__(self, symbol: str = "NSE:NIFTY50-INDEX",
                 interval_minutes: int = 5):
        self.symbol = symbol
        if interval_minutes not in self._RESOLUTION:
            raise FeedError(f"Fyers has no {interval_minutes}m resolution")
        self.resolution = self._RESOLUTION[interval_minutes]
        self.app_id = os.getenv("FYERS_APP_ID", "").strip()
        self.access_token = os.getenv("FYERS_ACCESS_TOKEN", "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.app_id and self.access_token)

    def get_bars(self, lookback_days: int = 5) -> pd.DataFrame:
        if not self.configured:
            raise NotConfigured(
                "Fyers feed needs FYERS_APP_ID and FYERS_ACCESS_TOKEN in .env. "
                "Generate the access token with the Fyers auth flow - it expires "
                "daily, so this is not a one-time setup."
            )
        import requests

        end = datetime.now()
        start = end - timedelta(days=max(lookback_days, 1))
        url = "https://api.fyers.in/data/history"
        params = {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "date_format": "1",
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": end.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }
        headers = {"Authorization": f"{self.app_id}:{self.access_token}"}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            raise FeedError(f"Fyers history request failed: {e}") from e

        candles = payload.get("candles")
        if not candles:
            raise FeedError(f"Fyers returned no candles: {payload}")

        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low",
                                            "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.set_index("ts")
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return self.validate(df)


class DhanFeed(DataFeed):
    name = "dhan"
    is_delayed = False
    latency_note = "real-time (broker feed)"

    def __init__(self, security_id: str = "13", exchange_segment: str = "IDX_I",
                 interval_minutes: int = 5):
        self.security_id = security_id          # 13 = Nifty 50 index
        self.exchange_segment = exchange_segment
        self.interval = str(interval_minutes)
        self.client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN", "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.access_token)

    def get_bars(self, lookback_days: int = 5) -> pd.DataFrame:
        if not self.configured:
            raise NotConfigured(
                "Dhan feed needs DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env."
            )
        import requests

        end = datetime.now()
        start = end - timedelta(days=max(lookback_days, 1))
        url = "https://api.dhan.co/v2/charts/intraday"
        body = {
            "securityId": self.security_id,
            "exchangeSegment": self.exchange_segment,
            "instrument": "INDEX",
            "interval": self.interval,
            "fromDate": start.strftime("%Y-%m-%d"),
            "toDate": end.strftime("%Y-%m-%d"),
        }
        headers = {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, json=body, headers=headers, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            raise FeedError(f"Dhan intraday request failed: {e}") from e

        if not payload.get("timestamp"):
            raise FeedError(f"Dhan returned no candles: {payload}")

        df = pd.DataFrame({
            "ts": pd.to_datetime(payload["timestamp"], unit="s", utc=True),
            "open": payload["open"],
            "high": payload["high"],
            "low": payload["low"],
            "close": payload["close"],
            "volume": payload.get("volume", [0] * len(payload["timestamp"])),
        }).set_index("ts")
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return self.validate(df)
