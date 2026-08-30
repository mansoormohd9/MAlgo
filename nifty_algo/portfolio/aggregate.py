"""
Every connector's answer, merged into one account-level view.

THE THREE THINGS THIS MODULE REFUSES TO DO, each of which produces a
plausible, wrong report rather than an error:

  1. IT WILL NOT TOTAL A BOOK IT COULD NOT READ. If a connector you have
     ENABLED comes back `available=False`, the snapshot is incomplete and
     `weight()` returns None. Absolute figures still render - they are facts
     about the lines we did see - but a percentage of the portfolio computed
     against a partial denominator is wrong in a way that reads perfectly
     normal. "You are 31% in financials" against half your book is worse than
     no number, because you would act on it.

  2. IT WILL NOT CONVERT AT A GUESSED RATE. `swing/fx.py` raises
     `FxUnavailable` rather than returning a fallback, and that contract is
     kept here: an unconvertible line is recorded, named, and left out of the
     rupee total, and its presence makes the snapshot incomplete. There is no
     "last rate seen" and no hardcoded 88.

  3. IT WILL NOT MERGE TWO COMPANIES INTO ONE. Keys are `{market}:{SYMBOL}`
     from `markets.Market.qualified()`, so an NSE line and a US line that
     share a ticker stay two positions. A collision here is not a crash - it
     is one company's weight computed from another's price.

A CONNECTOR THAT IS NOT ENABLED IS NOT A FAILURE. `PortfolioConfig.connectors`
says which brokers this account actually uses. Anything outside that list is
never called and never counted - which is what keeps the registered-but-
unimplemented IBKR stub from permanently marking every snapshot incomplete.
Enable it and it will, correctly: you would then be telling the app about an
account it cannot read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..config import Config, DEFAULT
from ..swing import fx as fx_mod
from . import registry
from .base import ConnectorResult, Position


@dataclass
class PortfolioSnapshot:
    """
    What you hold, from everything that would answer, plus what would not.

    `complete` is the field every consumer must read before quoting a
    percentage. See the module docstring for why it gates `weight()` rather
    than merely annotating it.
    """
    positions: list[Position] = field(default_factory=list)
    results: list[ConnectorResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)      # not enabled
    value_inr: dict[str, float] = field(default_factory=dict)
    rate_notes: dict[str, str] = field(default_factory=dict)
    unconvertible: dict[str, str] = field(default_factory=dict)  # ccy -> why
    generated_at: datetime | None = None

    # ---------------- provenance ----------------

    @property
    def failed_sources(self) -> list[ConnectorResult]:
        return [r for r in self.results if not r.available]

    @property
    def complete(self) -> bool:
        """
        Every enabled connector answered, and every line converted.

        Both halves matter. A source that failed hides positions; a currency
        that would not convert hides value. Either one makes a denominator a
        guess.
        """
        return not self.failed_sources and not self.unconvertible

    # ---------------- money ----------------

    @property
    def total_inr(self) -> float:
        """
        Rupee value of everything that CONVERTED. Not necessarily the account.

        Deliberately not called `account_value`: when `complete` is False this
        is a subtotal, and naming it after the whole account is how a subtotal
        gets quoted as one.
        """
        return sum(self.value_inr.values())

    def weight(self, key: str) -> float | None:
        """
        This position as a fraction of the book, or **None** when the book is
        not fully known. None is the honest answer and forces the caller to
        say so; a float here would be acted on.
        """
        if not self.complete:
            return None
        total = self.total_inr
        if total <= 0:
            return None
        value = self.value_inr.get(key)
        return None if value is None else value / total

    # ---------------- slicing ----------------

    def by_market(self) -> dict[str, list[Position]]:
        out: dict[str, list[Position]] = {}
        for p in self.positions:
            out.setdefault(p.market, []).append(p)
        return out

    def by_currency(self) -> dict[str, list[Position]]:
        out: dict[str, list[Position]] = {}
        for p in self.positions:
            out.setdefault(p.currency, []).append(p)
        return out

    def by_asset_class(self) -> dict[str, list[Position]]:
        out: dict[str, list[Position]] = {}
        for p in self.positions:
            out.setdefault(p.asset_class, []).append(p)
        return out

    def get(self, key: str) -> Position | None:
        for p in self.positions:
            if p.key == key:
                return p
        return None

    # ---------------- what to print above a report ----------------

    def note(self) -> str:
        bits = [f"{len(self.positions)} position(s) from "
                f"{len([r for r in self.results if r.available])} source(s)"]
        if self.complete:
            bits.append(f"total \u20b9{self.total_inr:,.0f}")
        else:
            bits.append(f"\u20b9{self.total_inr:,.0f} of KNOWN value - the book "
                        f"is incomplete, so portfolio percentages are withheld")
        return "; ".join(bits) + "."

    def caveats(self) -> list[str]:
        """
        Everything that would make a figure below wrong. Printed above the
        report, not buried under it.
        """
        out: list[str] = []
        for r in self.failed_sources:
            out.append(f"{r.source} could not be read, so anything it holds is "
                       f"missing from every figure here: {r.note}")
        for currency, why in self.unconvertible.items():
            out.append(f"{currency} positions are excluded from the rupee "
                       f"total - no rate this build will size on: {why}")
        for key in self.skipped:
            out.append(f"{key} is registered but not enabled, so it was never "
                       f"asked. Add it to PortfolioConfig.connectors if you "
                       f"hold anything there.")
        return out


def load(cfg: Config = DEFAULT, connectors=None, **per_key) -> PortfolioSnapshot:
    """
    Read every enabled connector and value the result in rupees.

    `connectors` overrides `cfg.portfolio.connectors` (tests, and the UI's
    "just read Kite" button). `per_key` is forwarded to `registry.get`, e.g.
    `load(cfg, manual={"extra": rows})`.
    """
    enabled = list(connectors if connectors is not None
                   else cfg.portfolio.connectors)
    snapshot = PortfolioSnapshot(generated_at=datetime.now())
    snapshot.skipped = [k for k in registry.keys() if k not in enabled]

    merged: dict[str, Position] = {}
    for key in enabled:
        result = _fetch(key, cfg, per_key.get(key, {}))
        snapshot.results.append(result)
        for position in result.positions:
            merged[position.key] = _combine(merged.get(position.key), position)

    snapshot.positions = list(merged.values())
    _value(snapshot, cfg)
    return snapshot


def _fetch(key: str, cfg: Config, kwargs: dict) -> ConnectorResult:
    """
    One connector, with every failure turned into an unavailable result.

    An enabled connector that is not configured is UNAVAILABLE, not skipped:
    you told the app you hold something there, so "we could not ask" is the
    truth and it must make the snapshot incomplete. Kite is unconfigured most
    weekdays - its token dies overnight - and a book that silently shrinks to
    the manual file on those days is exactly the quiet wrong answer this
    package exists to prevent.
    """
    if key == registry.manual.KEY and "path" not in kwargs:
        kwargs = {**kwargs, "path": cfg.portfolio.manual_path}
    try:
        connector = registry.get(key, cfg, **kwargs)
    except registry.UnknownConnector:
        raise
    except Exception as e:
        return ConnectorResult.unavailable(
            key, f"could not be constructed ({type(e).__name__}: {e})")

    try:
        if not connector.is_configured():
            return ConnectorResult.unavailable(
                key, f"{connector.label} is enabled but not configured, so it "
                     f"was not asked. Nothing it holds appears below.")
        return connector.fetch()
    except Exception as e:                     # the protocol forbids this...
        return ConnectorResult.unavailable(    # ...so the belt stays on
            key, f"fetch raised, which a connector must never do "
                 f"({type(e).__name__}: {e})")


def _combine(existing: Position | None, incoming: Position) -> Position:
    """
    The same key from two sources.

    Quantities are ADDED - one company held in two accounts is one exposure,
    and that is the whole question a concentration report asks. The cost basis
    is re-derived as the blended average so it stays consistent with the
    combined quantity; taking either side's average unchanged would report a
    P&L that belongs to neither line.
    """
    if existing is None:
        return incoming

    quantity = existing.quantity + incoming.quantity
    cost = existing.cost_native + incoming.cost_native
    from dataclasses import replace
    return replace(
        existing,
        quantity=quantity,
        average_price=(cost / quantity) if quantity else 0.0,
        # The fresher price wins; a stale zero would value the whole line at
        # nothing and drop it out of every weight.
        last_price=incoming.last_price or existing.last_price,
        source=f"{existing.source}+{incoming.source}",
        account=existing.account or incoming.account,
        name=existing.name or incoming.name,
    )


def _value(snapshot: PortfolioSnapshot, cfg: Config) -> None:
    """
    Rupee-value every line, and record the ones that could not be.

    Rates are fetched ONCE per currency: `fx.rate_inr_per` caches, but a
    per-position call would still turn one failure into N identical failures
    and N identical notes.
    """
    for currency, positions in snapshot.by_currency().items():
        try:
            rate = fx_mod.rate_inr_per(currency, cfg)
        except fx_mod.FxUnavailable as e:
            # Not a fallback, not a zero, not a skip-and-carry-on: recorded,
            # named, and it makes `complete` False. See the module docstring.
            snapshot.unconvertible[currency] = str(e)
            continue
        snapshot.rate_notes[currency] = rate.note()
        for p in positions:
            snapshot.value_inr[p.key] = p.value_native * rate.inr_per_unit
