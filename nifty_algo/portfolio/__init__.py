"""
Broker-agnostic holdings.

`aggregate.load(cfg)` is the entry point; `registry.py` is the only place a
connector is named. See `base.py` for why `ConnectorResult.available` is the
field that matters most in this package.
"""
from .base import (ASSET_CLASSES, CASH, EQUITY, ETF, MUTUAL_FUND,
                   ConnectorResult, PortfolioConnector, Position)
from .aggregate import PortfolioSnapshot, load
from .registry import UnknownConnector

__all__ = ["Position", "ConnectorResult", "PortfolioConnector",
           "PortfolioSnapshot", "load", "UnknownConnector",
           "EQUITY", "ETF", "MUTUAL_FUND", "CASH", "ASSET_CLASSES"]
