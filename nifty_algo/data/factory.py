"""
One place that turns a DataConfig into a live DataFeed object.

The UI, the engine, and run_live all go through here so a feed can never be
selectable in one and absent from another.
"""
from __future__ import annotations

from ..config import Config, DEFAULT
from .base import DataFeed, FeedError
from .csv_feed import CsvFeed
from .yfinance_feed import YFinanceFeed
from .broker_feeds import FyersFeed, DhanFeed

PROVIDERS = ("csv", "kite", "yfinance", "fyers", "dhan")


def build_feed(cfg: Config = DEFAULT, provider: str | None = None) -> DataFeed:
    provider = (provider or cfg.data.provider).lower().strip()
    d = cfg.data

    if provider == "csv":
        return CsvFeed(d.csv_path)
    if provider == "kite":
        # Imported lazily: kiteconnect is an optional dependency, and the CSV
        # and yfinance paths must keep working on a machine without it.
        from .kite_feed import KiteFeed
        return KiteFeed(interval_minutes=d.interval_minutes)
    if provider == "yfinance":
        return YFinanceFeed(d.yfinance_ticker, d.interval_minutes)
    if provider == "fyers":
        return FyersFeed(interval_minutes=d.interval_minutes)
    if provider == "dhan":
        return DhanFeed(interval_minutes=d.interval_minutes)

    raise FeedError(f"unknown data provider '{provider}'; expected one of {PROVIDERS}")
