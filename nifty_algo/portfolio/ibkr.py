"""
Interactive Brokers. REGISTERED, NOT IMPLEMENTED - and that is the point.

WHY A STUB IS WORTH COMMITTING. You asked for the holdings layer to extend to
another broker, and the only honest way to demonstrate that it does is to
build the registry against two brokers rather than one. A protocol proven by a
single implementation is a protocol shaped like that implementation: this file
is what stops `Position` from quietly becoming "whatever Kite returns".

It also settles three questions Kite never raises, before they can be answered
badly by accident:

  * IBKR reports positions in the currency of the LISTING, so one account
    holds USD, GBP and EUR lines at once. `Position.currency` is native and
    `aggregate.py` converts through `swing/fx.py`, which REFUSES to guess a
    rate. A connector that converted here would need a fallback rate, and the
    fallback is how a US position ends up sized ~88x wrong while looking
    entirely ordinary.
  * IBKR fills FRACTIONAL shares. `Position.quantity` is a float throughout,
    and `markets.Market.allow_fractional` already records which venues do.
  * IBKR is one login over several accounts. `Position.account` carries the
    account id so two accounts cannot be summed into one concentration figure.

WHAT IMPLEMENTING IT ACTUALLY NEEDS. Neither route is a pip install and a key:

  1. `ib_insync` (or `ibapi`) talking to a RUNNING Trader Workstation or IB
     Gateway on localhost:7496/7497. There is no cloud endpoint - the desktop
     process IS the API - so a headless scheduled run needs the gateway up.
  2. The Client Portal Web API, which needs its own gateway process and a
     browser SSO that expires.

Both re-authenticate roughly daily, which is the same operational shape as
Kite's overnight token death, so `is_configured()` returning False is the
ordinary weekday state here too and must never be treated as an error.

UNTIL THEN, record IBKR positions in `manual.py`. That is not a workaround
kept quiet: `fetch()` says it, in the note, every time it is called.
"""
from __future__ import annotations

from ..config import Config, DEFAULT
from .base import ConnectorResult

KEY = "ibkr"
LABEL = "Interactive Brokers (not connected)"

#: The one thing a reader of a stub needs: what to install and what to run.
REQUIREMENTS = (
    "Needs `ib_insync` plus a running Trader Workstation or IB Gateway on "
    "localhost (7497 paper / 7496 live), or the Client Portal gateway. "
    "Neither works headless without that process running."
)


class IbkrConnector:
    """
    A connector that knows it cannot answer, and says why.

    `fetch()` returns `unavailable` rather than raising. A registry that
    enumerates every connector must be able to call this one without an
    unconfigured broker costing you the report on the broker that IS
    configured - that is the whole reason `PortfolioConnector.fetch` may not
    raise.
    """

    key = KEY
    label = LABEL

    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg

    def is_configured(self) -> bool:
        return False

    def fetch(self) -> ConnectorResult:
        return ConnectorResult.unavailable(
            KEY,
            "not implemented in this build. " + REQUIREMENTS + " Record IBKR "
            "positions in data/manual_positions.csv for now - the reports "
            "read them identically, they are simply not refreshed for you."
        )
