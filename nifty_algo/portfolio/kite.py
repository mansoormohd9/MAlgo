"""
Zerodha holdings, as a portfolio connector.

A READ-ONLY VIEW OVER `broker/kite_equity.py`. This module deliberately does
not import the order helpers and never calls one: `broker/` is the only
package that can spend money (invariant 7), and a research package that could
reach an order path would make every briefing a thing you have to read
carefully before running. It reads `holdings()` and nothing else.

WHY IT WRAPS RATHER THAN RE-IMPLEMENTS. `HoldingView.total_quantity` already
gets the one detail that is easy to lose: T+1 shares are yours and sellable
but Kite reports them in a separate field, so reading `quantity` alone on the
day after a fill shows zero - which reads exactly like "the buy never
happened". That logic exists, is tested, and must not be written twice.

THE FAILED-READ PROBLEM, AND HOW THIS SOLVES IT. `BrokerTransport._read`
returns `[]` when the request fails, and counts the failure in
`read_failures`. An empty list is therefore ambiguous at the call site and
must not be handed on: "you own nothing" and "we could not ask" produce
opposite risk reports. So this connector snapshots the counter either side of
the call and only reports `available=True` when the counter did not move -
the same test `swing/daily.run` applies before it trusts a reconciliation.
"""
from __future__ import annotations

from ..config import Config, DEFAULT
from ..swing import markets as markets_mod
from .base import EQUITY, ConnectorResult, Position

KEY = "kite"
LABEL = "Zerodha Kite - NSE/BSE cash equity"

#: Kite reports the exchange per holding. Both Indian exchanges map to the one
#: registered market: it is the same company, the same currency and the same
#: universe file, and keying them apart would split one position in two.
_EXCHANGE_TO_MARKET = {"NSE": markets_mod.INDIA, "BSE": markets_mod.INDIA}


class KiteConnector:
    """Holdings from the Zerodha account this app is already logged in to."""

    key = KEY
    label = LABEL

    def __init__(self, cfg: Config = DEFAULT, broker=None, journal=None):
        self.cfg = cfg
        self.journal = journal
        self._broker = broker

    # ---------------- plumbing ----------------

    def broker(self):
        """
        The equity broker, built lazily.

        Lazy because constructing one touches `KiteSession`, and the registry
        builds every connector in order to ask whether it is configured. A
        registry lookup must not be the thing that opens a session file.
        """
        if self._broker is None:
            from ..broker.kite_equity import KiteEquity
            self._broker = KiteEquity(cfg=self.cfg, journal=self.journal)
        return self._broker

    def is_configured(self) -> bool:
        """
        Whether there is a usable token. False is not an error - it is the
        ordinary state of this account outside a trading morning, because
        Kite's access token dies overnight and there is no refresh token.
        """
        try:
            return bool(self.broker().session.authenticated)
        except Exception:
            return False

    # ---------------- the read ----------------

    def fetch(self) -> ConnectorResult:
        try:
            broker = self.broker()
        except Exception as e:
            return ConnectorResult.unavailable(
                KEY, f"could not open the Kite session ({type(e).__name__}: {e})")

        before = broker.read_failures
        try:
            rows = broker.holdings()
        except Exception as e:
            # `holdings()` is not supposed to raise - `_read` catches - but a
            # connector that can take the snapshot down with it is exactly what
            # the protocol forbids, so this belt stays on.
            return ConnectorResult.unavailable(
                KEY, f"the holdings read raised ({type(e).__name__}: {e})")

        if broker.read_failures > before:
            return ConnectorResult.unavailable(
                KEY,
                "the holdings request failed - usually an expired access "
                "token, which dies overnight. Run "
                "`python -m nifty_algo.broker.kite_login` and try again. "
                "Reporting this as UNAVAILABLE rather than as an empty "
                "account, because an empty account has no risk and yours "
                "may not be empty.")

        positions = [p for p in (_to_position(r) for r in rows) if p is not None]
        skipped = len(rows) - len(positions)

        note = "read from the live account"
        if skipped:
            note += (f"; {skipped} holding(s) skipped - zero quantity or an "
                     f"exchange this build does not map")
        if not positions and not skipped:
            note = ("the account reported no holdings. This is a real answer, "
                    "not a failed read.")
        return ConnectorResult.ok(KEY, positions, note)


def _to_position(row) -> Position | None:
    """
    One `HoldingView` to one `Position`, or None if it is not a holding.

    A zero-quantity row is what Kite leaves behind after a full exit. It is not
    a position, and counted as one it would divide by zero on the way to a
    portfolio weight.
    """
    quantity = row.total_quantity
    if quantity <= 0:
        return None

    market_key = _EXCHANGE_TO_MARKET.get(row.exchange.upper())
    if market_key is None:
        return None

    symbol = row.symbol.upper()
    return Position(
        key=f"{market_key}:{symbol}",
        symbol=symbol,
        market=market_key,
        quantity=float(quantity),
        average_price=float(row.average_price),
        # Kite's `last_price` can be 0 before the first tick of the day. Fall
        # back to cost so the line still has a value: a position valued at
        # zero would silently drop out of every weight in the report.
        last_price=float(row.last_price or row.average_price),
        currency="INR",
        asset_class=EQUITY,
        source=KEY,
        account=row.exchange.upper(),
    )
