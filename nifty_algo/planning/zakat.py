"""
Zakat on the whole of your wealth, and the income you owe to purify.

WHY THIS EXISTS. For a halal-only holder, zakat is the one yearly outflow the
portfolio itself creates, and often the largest. Computing it needs exactly
what the console already assembles and nothing else did: every account, in
rupees, classified by what each line IS. Until now the snapshot totalled all of
it and then stopped one step short.

EVERY METHOD CHOICE IS THE HOLDER'S, AND UNSET UNTIL CHOSEN. The schools
differ on the nisab basis (gold or silver), on how much of a long-term share
holding is zakatable, and on retirement money you cannot yet reach. A default
shipped in code would be the code quietly picking a madhhab - the reason the
repo already treats SPUS-vs-HLAL as a toggle. So `PlanConfig` holds "" / 0
for each, and `compute` returns a result that NAMES what is missing rather
than a figure built on a guess.

NO PRICE, NO NISAB. The metal price is typed, and an unknown price is no
nisab - never a hardcoded one, never the last one seen. That is `swing/fx.py`'s
rule, for the same reason: a stale threshold produces an ordinary-looking
answer that is wrong.

AN INCOMPLETE SNAPSHOT WITHHOLDS THE AMOUNT DUE. Zakat on part of your wealth
reads exactly like zakat on all of it, and it is an amount you would pay.

Pure: no Streamlit, no network, no file reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..portfolio.base import CASH, FIXED_INCOME, GOLD, Position
from . import allocation as alloc

RATE = 0.025

#: Nisab thresholds in grams. The gold figure is 20 mithqal; the silver
#: figure is 200 dirham. The widely published modern conversions.
NISAB_GOLD_G = 87.48
NISAB_SILVER_G = 612.36

BASIS_GOLD = "gold"
BASIS_SILVER = "silver"
BASES = (BASIS_GOLD, BASIS_SILVER)

METHOD_MARKET = "market_value"
METHOD_ASSETS = "zakatable_assets"
METHODS = (METHOD_MARKET, METHOD_ASSETS)

METHOD_LABELS = {
    METHOD_MARKET: "Full market value (the trading view; also the simplest)",
    METHOD_ASSETS: "Zakatable assets only (the company's cash, receivables "
                   "and inventory - via your proxy %)",
}
BASIS_LABELS = {
    BASIS_GOLD: f"Gold - {NISAB_GOLD_G} g (a higher threshold)",
    BASIS_SILVER: f"Silver - {NISAB_SILVER_G} g (lower; the Hanafi position "
                  f"common in South Asia, and the more cautious)",
}

#: Retirement money, recognised by symbol or name. Hand-recorded lines are
#: usually called exactly this.
RETIREMENT_MARKERS = ("EPF", "PPF", "VPF", "NPS", "PROVIDENT", "PENSION")


def is_retirement(position: Position) -> bool:
    text = f"{position.symbol} {position.name}".upper()
    return any(m in text for m in RETIREMENT_MARKERS)


@dataclass(frozen=True)
class Line:
    symbol: str
    name: str
    value_inr: float
    zakatable_pct: float
    rule: str

    @property
    def zakatable_inr(self) -> float:
        return self.value_inr * self.zakatable_pct


@dataclass
class ZakatResult:
    lines: list[Line]
    complete: bool
    #: What stops an amount being stated. Empty means `due_inr` is real.
    missing: list[str] = field(default_factory=list)
    nisab_inr: float | None = None

    @property
    def base_inr(self) -> float:
        return sum(line.zakatable_inr for line in self.lines)

    @property
    def above_nisab(self) -> bool | None:
        if self.nisab_inr is None:
            return None
        return self.base_inr >= self.nisab_inr

    @property
    def due_inr(self) -> float | None:
        """The amount owed, 0.0 below nisab, or None while anything is missing."""
        if self.missing or self.nisab_inr is None:
            return None
        return self.base_inr * RATE if self.above_nisab else 0.0


def nisab_inr(plan) -> float | None:
    """The threshold in rupees, or None without a chosen basis and a price."""
    if plan.zakat_nisab_basis == BASIS_GOLD and plan.gold_price_inr_per_g > 0:
        return NISAB_GOLD_G * plan.gold_price_inr_per_g
    if (plan.zakat_nisab_basis == BASIS_SILVER
            and plan.silver_price_inr_per_g > 0):
        return NISAB_SILVER_G * plan.silver_price_inr_per_g
    return None


def rule_for(position: Position, plan) -> tuple[float, str]:
    """(share of the value that is zakatable, why). Pure."""
    symbol = position.symbol.upper()
    if position.asset_class == CASH or symbol in alloc.LIQUID_SYMBOLS:
        return 1.0, "cash - all of it"
    if (position.asset_class == GOLD or symbol in alloc.GOLD_SYMBOLS
            or symbol in alloc.SILVER_SYMBOLS or symbol.startswith("SGB")):
        return 1.0, "gold / silver - all of it"
    if position.asset_class == FIXED_INCOME:
        if is_retirement(position):
            if plan.zakat_include_retirement:
                return 1.0, "retirement account - included (your choice)"
            return 0.0, "retirement account - excluded until accessible"
        return 1.0, ("deposit / debt - the principal; the interest is "
                     "purified, not zakat'd")
    # Shares, ETFs and funds.
    if plan.zakat_equity_method == METHOD_ASSETS:
        pct = max(0.0, min(100.0, plan.zakat_equity_proxy_pct)) / 100.0
        return pct, f"investment - {pct:.0%} proxy for zakatable assets"
    if plan.zakat_equity_method == METHOD_MARKET:
        return 1.0, "investment - full market value"
    return 0.0, "investment - method not chosen yet"


def compute(snapshot, plan) -> ZakatResult:
    lines = []
    for p in snapshot.positions:
        pct, why = rule_for(p, plan)
        lines.append(Line(symbol=p.symbol, name=p.name or p.symbol,
                          value_inr=float(snapshot.value_inr.get(p.key, 0.0)),
                          zakatable_pct=pct, rule=why))
    lines.sort(key=lambda line: -line.zakatable_inr)

    missing = []
    complete = bool(snapshot.complete)
    if not complete:
        missing.append("not every connected account answered - zakat on part "
                       "of your wealth reads exactly like zakat on all of it")
    if plan.zakat_nisab_basis not in BASES:
        missing.append("choose a nisab basis (gold or silver)")
    elif nisab_inr(plan) is None:
        missing.append(f"enter today's {plan.zakat_nisab_basis} price per "
                       f"gram - no price, no nisab")
    if plan.zakat_equity_method not in METHODS:
        missing.append("choose how shares and funds are counted")
    elif (plan.zakat_equity_method == METHOD_ASSETS
          and plan.zakat_equity_proxy_pct <= 0):
        missing.append("enter the zakatable-assets proxy % for shares")
    return ZakatResult(lines=lines, complete=complete, missing=missing,
                       nisab_inr=nisab_inr(plan))


# ---------------------------------------------------------------- purification

@dataclass
class Purification:
    interest_inr: float
    #: None when dividends were entered without a ratio - unknown, not zero.
    dividend_inr: float | None
    #: Lines that generate interest, so you know which statements to read.
    sources: list[str]

    @property
    def total_inr(self) -> float | None:
        if self.dividend_inr is None:
            return None
        return self.interest_inr + self.dividend_inr


def purification(snapshot, plan) -> Purification:
    if plan.dividends_received_inr > 0 and plan.dividend_purification_pct <= 0:
        dividend = None
    else:
        dividend = (plan.dividends_received_inr
                    * max(0.0, plan.dividend_purification_pct) / 100.0)
    sources = []
    for p in snapshot.positions if snapshot is not None else ():
        symbol = p.symbol.upper()
        if p.asset_class == FIXED_INCOME:
            sources.append(f"{p.symbol} - interest")
        elif symbol in alloc.LIQUID_SYMBOLS:
            sources.append(f"{p.symbol} - overnight-rate returns")
        elif symbol.startswith("SGB"):
            sources.append(f"{p.symbol} - 2.5% coupon")
        elif p.asset_class == CASH and p.source == "ibkr":
            sources.append(f"{p.symbol} - IBKR pays interest on idle cash "
                           f"unless you opt out")
        elif p.asset_class == CASH:
            sources.append(f"{p.symbol} - savings interest, if the account "
                           f"pays it")
    return Purification(interest_inr=max(0.0, plan.interest_received_inr),
                        dividend_inr=dividend, sources=sources)
