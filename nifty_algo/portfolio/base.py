"""
What a holding is, independently of which broker reported it.

WHY THIS PACKAGE EXISTS. Two of the research briefings - the macro impact
assessment and the risk framework - are about the WHOLE account rather than
about one trade, so they cannot start until something can answer "what do I
own". Until now the only answer was `KiteEquity.holdings()`, which speaks
Kite's vocabulary, lives in the package that can spend money, and knows about
exactly one broker.

Registry, not a pile of if-statements - the same reasoning as
`data/factory.py`, `strategies/registry.py` and `swing/markets.py`. A second
broker (IBKR is the one you have named) must be a new file plus one line in
`registry.py`, and must not require touching a single report.

THE ONE THING IN HERE THAT IS LOAD-BEARING: `ConnectorResult.available`.

`BrokerTransport._read` returns an empty default when a read FAILS - an
expired token, no network, a rate limit - because a reconciler that raises
mid-run is worse than one that reports nothing. That is right for the order
path and catastrophic for a risk report: an empty list read as "you hold
nothing" produces a report with no concentration, no correlation and no
single-stock risk, on a fully invested account. A clean bill of health
manufactured by an outage is exactly the failure this repo refuses to tolerate
in `protection_state()` and in `news.available`, and it is refused here the
same way.

So `available` is False by DEFAULT and every connector must set it True
deliberately, having established that it really did get an answer. Nothing
downstream may total a portfolio whose sources did not all say True.

POSITIONS CARRY NATIVE CURRENCY AND ARE NEVER CONVERTED HERE. Conversion needs
a rate, `swing/fx.py` refuses to guess one, and a module that converted at
ingest would have to choose between raising (killing the whole snapshot for one
foreign line) and guessing (an 88x error that looks entirely ordinary). The
snapshot converts, per position, and says which lines it could not - see
`aggregate.py`.

KEYS ARE `{market}:{SYMBOL}`, from `markets.Market.qualified()`. Bare tickers
collide across exchanges and a collision is not a crash: it is one company's
position sized against another's balance sheet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

#: What kind of thing a line is. Not cosmetic - the risk report treats a fund
#: as a basket to look through (see `swing/holdings.py`) and a share as a
#: single name, and cash is exposure to nothing at all.
EQUITY = "equity"
ETF = "etf"
MUTUAL_FUND = "mf"
CASH = "cash"

ASSET_CLASSES = (EQUITY, ETF, MUTUAL_FUND, CASH)


@dataclass(frozen=True)
class Position:
    """
    One line of the account, in the currency it is actually quoted in.

    Frozen because a connector reports a fact and nothing downstream is
    entitled to edit it. Derived figures (rupee value, portfolio weight) are
    computed by the snapshot, which knows the rates and can say when it
    could not get one.
    """
    key: str                        # "india:RELIANCE" - the only safe id
    symbol: str
    market: str
    quantity: float
    average_price: float
    last_price: float
    currency: str                   # ISO code, native. NOT converted.
    asset_class: str = EQUITY
    source: str = ""                # connector key that reported it
    account: str = ""               # which account, when a broker has several
    name: str = ""

    @property
    def value_native(self) -> float:
        return self.quantity * self.last_price

    @property
    def cost_native(self) -> float:
        return self.quantity * self.average_price

    @property
    def pnl_native(self) -> float:
        return self.value_native - self.cost_native

    @property
    def pnl_pct(self) -> float | None:
        """None, not zero, when there is no cost basis to divide by."""
        cost = self.cost_native
        if cost <= 0:
            return None
        return self.pnl_native / cost


@dataclass
class ConnectorResult:
    """
    What one connector had to say, and whether it was able to say anything.

    `available=False` means WE COULD NOT ASK. It never means "the account is
    empty" - that is `available=True` with an empty list, and the two must
    stay distinguishable all the way to the report. See the module docstring.
    """
    source: str
    positions: list[Position] = field(default_factory=list)
    available: bool = False
    note: str = ""
    fetched_at: datetime | None = None

    @classmethod
    def ok(cls, source: str, positions: list[Position],
           note: str = "") -> "ConnectorResult":
        return cls(source=source, positions=list(positions), available=True,
                   note=note, fetched_at=datetime.now())

    @classmethod
    def unavailable(cls, source: str, note: str) -> "ConnectorResult":
        """The only way to build a falsy result, and it demands a reason."""
        return cls(source=source, positions=[], available=False, note=note,
                   fetched_at=datetime.now())

    def summary(self) -> str:
        if not self.available:
            return f"{self.source}: could not read - {self.note}"
        n = len(self.positions)
        base = f"{self.source}: {n} position{'' if n == 1 else 's'}"
        return f"{base} - {self.note}" if self.note else base


@runtime_checkable
class PortfolioConnector(Protocol):
    """
    The contract. Four members, and `fetch` may not raise.

    A connector that raises takes the whole snapshot down, which would mean one
    unconfigured broker costing you the report on the broker that IS
    configured. Failures are returned as `ConnectorResult.unavailable`.
    """
    key: str
    label: str

    def is_configured(self) -> bool:
        """Whether there is any point calling `fetch`."""
        ...

    def fetch(self) -> ConnectorResult:
        ...
