"""
Is what you ACTUALLY OWN halal - every line, from every account?

WHY THIS EXISTS. The Shariah screen in `swing/halal.py` ran only on what the
console itself proposed: the sleeve's shortlist and the swing universes. The
investor's real wealth - Kite shares, CAS mutual funds, IBKR positions, SGBs,
EPF, cash - was never checked, and the `halal_only` switch did no more than
turn the sleeve screen on and print one note about fixed income. So a
halal-only holder could own a conventional debt fund, a bank share or
LIQUIDBEES and see a tidy allocation with nothing flagged: the "looks armed and
is not" failure this repo refuses everywhere else, on the constraint that
matters most to the person using it.

FOUR VERDICTS, NEVER TWO.

  compliant      on the cited fund list (`data/halal_funds.csv`), passed the
                 ratio screen on a cached balance sheet, or cash held idle.
  non_compliant  pays interest (debt/liquid funds, FD, PPF, EPF), a
                 conventional fund or index ETF, or a stock that FAILED the
                 screen.
  contested      scholars differ and the code will not pick: gold/silver ETFs
                 (unless you accept them - `PlanConfig.accept_metal_etfs`),
                 SGBs (a 2.5% coupon on a gold-linked bond, whatever view is
                 taken of the gold).
  unverified     a fact is missing - no cached balance sheet, a typed total
                 whose names cannot be seen, a Shariah-sounding fund not on
                 the list. NEVER COUNTED AS A PASS, the rule `HalalVerdict`
                 already follows: a missing fact is a failure to verify, not a
                 clean bill of health.

`classify` is pure: the equity screen arrives as a callable, so tests hand in a
stub and no network or file is touched. `equity_screener` and
`load_fund_list` are the only functions here that read files, and neither
fetches - `fundamentals.read_cached`, never `load_fundamentals`, which would
fire one Yahoo request per holding in the middle of drawing a table.

AN INCOMPLETE SNAPSHOT WITHHOLDS EVERY SHARE (`Summary.share` returns None),
for the reason `PortfolioSnapshot.weight()` does: "94% halal" computed against
a denominator missing an account reads exactly like one that was not.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..paths import at_root
from ..portfolio.base import (CASH, EQUITY, ETF, FIXED_INCOME, GOLD,
                              MUTUAL_FUND, Position)
from . import allocation as alloc

COMPLIANT = "compliant"
NON_COMPLIANT = "non_compliant"
CONTESTED = "contested"
UNVERIFIED = "unverified"

STATUSES = (COMPLIANT, NON_COMPLIANT, CONTESTED, UNVERIFIED)

LABELS = {
    COMPLIANT: "Halal",
    NON_COMPLIANT: "Not halal",
    CONTESTED: "Contested",
    UNVERIFIED: "Unverified",
}

ICONS = {COMPLIANT: "✅", NON_COMPLIANT: "⛔", CONTESTED: "⚖", UNVERIFIED: "❔"}

FUNDS_PATH = "data/halal_funds.csv"

#: Words that make a fund LOOK Shariah-screened. Not a pass - a fund with one
#: of these in its name that is missing from the list is `unverified`, so the
#: fix is one row in the CSV rather than a silent assumption either way.
SHARIAH_NAME_MARKERS = ("SHARIAH", "SHARIA", "ISLAMIC", "ETHICAL", "SUKUK")

#: A `Position.symbol` the US / LRS page invents for a typed total.
TYPED_TOTALS = frozenset({"DIRECT_US"})


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str
    #: What decided it: fund_list | screen | override | class | missing.
    basis: str

    @property
    def label(self) -> str:
        return f"{ICONS[self.status]} {LABELS[self.status]}"


@dataclass(frozen=True)
class FundEntry:
    symbol: str
    name_contains: str
    name: str
    status: str
    standard: str
    domicile: str
    #: Published impure share of dividends, or None - unknown is not zero.
    purification_pct: float | None
    note: str = ""


@dataclass
class FundList:
    by_symbol: dict[str, FundEntry] = field(default_factory=dict)
    by_name: list[FundEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def match(self, position: Position) -> FundEntry | None:
        entry = self.by_symbol.get(position.symbol.upper())
        if entry is not None:
            return entry
        name = (position.name or "").upper()
        if not name:
            return None
        return next((e for e in self.by_name if e.name_contains in name), None)


def load_fund_list(path: str | Path = FUNDS_PATH) -> FundList:
    """
    Read the cited fund list. A missing file is an empty list, and SAYS so.

    A bad row becomes a warning and is skipped: the file is hand-edited, and
    one typo must not stop the page - but a ruling that silently vanished is
    worse than none, so the warnings are shown (the `load_overrides` rule).
    """
    p = at_root(path)
    out = FundList()
    if not p.exists():
        out.warnings.append(f"{p} not found - no fund can be recognised as "
                            f"Shariah-screened, so every fund reads as "
                            f"unverified or not halal.")
        return out
    with p.open("r", encoding="utf-8", newline="") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    for i, row in enumerate(csv.DictReader(lines), start=2):
        symbol = (row.get("symbol") or "").strip().upper()
        contains = (row.get("name_contains") or "").strip().upper()
        status = (row.get("status") or "").strip().lower()
        if not (symbol or contains):
            continue
        if status not in (COMPLIANT, NON_COMPLIANT, CONTESTED):
            out.warnings.append(f"{p.name} row {i}: status {status!r} is not "
                                f"compliant / non_compliant / contested - "
                                f"row IGNORED.")
            continue
        raw = (row.get("purification_pct") or "").strip()
        try:
            purification = float(raw) if raw else None
        except ValueError:
            purification = None
            out.warnings.append(f"{p.name} row {i}: purification_pct {raw!r} "
                                f"is not a number - read as unknown.")
        entry = FundEntry(
            symbol=symbol, name_contains=contains,
            name=(row.get("name") or "").strip(), status=status,
            standard=(row.get("standard") or "").strip(),
            domicile=(row.get("domicile") or "").strip(),
            purification_pct=purification,
            note=(row.get("note") or "").strip())
        if symbol:
            out.by_symbol[symbol] = entry
        else:
            out.by_name.append(entry)
    return out


#: position -> (eligible, reason, source) or None when no balance sheet is
#: cached. `source` is halal.SOURCE_* ("override", "no_data", ...).
EquityScreen = Callable[[Position], "tuple[bool, str, str] | None"]


def classify(position: Position, funds: FundList, *,
             screen_equity: EquityScreen | None = None,
             accept_metal_etfs: bool = False) -> Verdict:
    """One holding's verdict. Pure - see the module docstring."""
    symbol = position.symbol.upper()

    # 1. The cited list is the strongest evidence there is: a fund's own
    #    mandate, screened by its own board. It outranks the class rules so a
    #    sukuk fund recorded as fixed income is still recognised.
    entry = funds.match(position)
    if entry is not None:
        why = f"{entry.name or symbol} - {entry.standard or 'listed'}"
        if entry.note:
            why += f" ({entry.note})"
        return Verdict(entry.status, why, "fund_list")

    # 2. Things that pay interest, whatever account holds them.
    if symbol in alloc.LIQUID_SYMBOLS:
        return Verdict(NON_COMPLIANT, "liquid / overnight ETF - earns an "
                       "overnight repo rate, i.e. interest", "class")
    if position.asset_class == FIXED_INCOME:
        return Verdict(NON_COMPLIANT, "pays interest (EPF, PPF, FD, bond or "
                       "debt fund). If you hold it by obligation, record it and "
                       "purify the interest.", "class")

    # 3. Cash held is permissible; interest credited on it is what needs
    #    purifying, and the Zakat page asks for that separately.
    if position.asset_class == CASH:
        return Verdict(COMPLIANT, "cash - permissible to hold; purify any "
                       "interest credited on it", "class")

    # 4. Metals. The coupon on an SGB is interest whatever view is taken of
    #    the gold, so SGBs never follow the metal switch.
    if symbol.startswith("SGB"):
        return Verdict(CONTESTED, "sovereign gold bond - gold-linked, but pays "
                       "a 2.5% interest coupon; many scholars advise avoiding "
                       "it or purifying the coupon", "class")
    is_metal = (position.asset_class == GOLD or symbol in alloc.GOLD_SYMBOLS
                or symbol in alloc.SILVER_SYMBOLS)
    if is_metal:
        if accept_metal_etfs:
            return Verdict(COMPLIANT, "gold / silver - you follow the view "
                           "that physically-backed metal ETFs are permitted "
                           "(AAOIFI Shariah Standard 57 conditions)", "class")
        return Verdict(CONTESTED, "gold / silver ETF - permitted by many "
                       "scholars when physically backed and redeemable, "
                       "questioned by others over T+1 settlement. Set your "
                       "view on Money & goals.", "class")

    # 5. A typed total cannot be screened - the names inside it are unseen.
    if symbol in TYPED_TOTALS:
        return Verdict(UNVERIFIED, "a typed total - the shares inside it "
                       "cannot be screened. Connect IBKR to see each name.",
                       "missing")

    # 6. Funds not on the list.
    if position.asset_class in (ETF, MUTUAL_FUND):
        name = (position.name or "").upper()
        if any(m in name for m in SHARIAH_NAME_MARKERS):
            return Verdict(UNVERIFIED, "looks Shariah-screened by name but is "
                           f"not in {FUNDS_PATH} - add it with its standard",
                           "missing")
        return Verdict(NON_COMPLIANT, "a conventional fund - its holdings are "
                       "not Shariah-screened (index and diversified funds hold "
                       "banks and lenders)", "class")

    # 7. A single share: the ratio screen on a cached balance sheet.
    if position.asset_class == EQUITY:
        result = screen_equity(position) if screen_equity else None
        if result is None:
            return Verdict(UNVERIFIED, "no cached balance sheet - open the "
                           "Monthly sleeve (or run a scan) to fetch one; "
                           "unverified is never a pass", "missing")
        eligible, reason, source = result
        if source == "no_data":
            return Verdict(UNVERIFIED, reason, "missing")
        basis = "override" if source == "override" else "screen"
        return Verdict(COMPLIANT if eligible else NON_COMPLIANT, reason, basis)

    return Verdict(UNVERIFIED, f"asset class {position.asset_class!r} has no "
                   f"rule", "missing")


# ---------------------------------------------------------------- summary

@dataclass
class Row:
    position: Position
    verdict: Verdict
    value_inr: float


@dataclass
class Summary:
    rows: list[Row]
    complete: bool

    def total(self, status: str | None = None) -> float:
        return sum(r.value_inr for r in self.rows
                   if status is None or r.verdict.status == status)

    def share(self, status: str) -> float | None:
        """A fraction of the whole, or None when the whole is unknown."""
        whole = self.total()
        if not self.complete or whole <= 0:
            return None
        return self.total(status) / whole

    def of(self, status: str) -> list[Row]:
        return sorted((r for r in self.rows if r.verdict.status == status),
                      key=lambda r: -r.value_inr)

    @property
    def needs_attention(self) -> bool:
        """Anything a halal-only holder has to act on or look at."""
        return any(r.verdict.status in (NON_COMPLIANT, UNVERIFIED)
                   for r in self.rows)


def assess(snapshot, funds: FundList, *,
           screen_equity: EquityScreen | None = None,
           accept_metal_etfs: bool = False) -> Summary:
    rows = [Row(position=p,
                verdict=classify(p, funds, screen_equity=screen_equity,
                                 accept_metal_etfs=accept_metal_etfs),
                value_inr=float(snapshot.value_inr.get(p.key, 0.0)))
            for p in snapshot.positions]
    return Summary(rows=rows, complete=bool(snapshot.complete))


# ---------------------------------------------------------------- the reads

def equity_screener(cfg) -> EquityScreen:
    """
    A screen over CACHED balance sheets. Reads files once; never fetches.

    Indian holdings are screened in the factor market's vocabulary (Yahoo /
    GICS labels copied onto the stock by `sleeve.stock_for`), because a Kite
    holding may be any of ~2,400 names and the NSE table covers the Nifty 100
    only. The factor cache is read first - it covers every name ever
    shortlisted - then the swing cache.
    """
    from ..factor.sleeve import stock_for
    from ..swing import fundamentals as fund_mod
    from ..swing import halal
    from ..swing import markets as markets_mod

    overrides, _ = halal.load_overrides(cfg.swing.halal.overrides_csv)
    factor_india = markets_mod.factor_market(cfg)
    registered = {k: markets_mod.get(cfg, k) for k in markets_mod.keys(cfg)}
    caches: dict[str, tuple[object, dict]] = {}

    def _facts(market_key: str):
        if market_key not in caches:
            if market_key == markets_mod.INDIA:
                merged = dict(fund_mod.read_cached(cfg, registered[market_key]))
                merged.update(fund_mod.read_cached(cfg, factor_india))
                caches[market_key] = (factor_india, merged)
            elif market_key in registered:
                m = registered[market_key]
                caches[market_key] = (m, fund_mod.read_cached(cfg, m))
            else:
                caches[market_key] = (None, {})
        return caches[market_key]

    # The Nifty 100 names carry NSE's hand-maintained industry labels, and the
    # swing cache holds their sheets WITHOUT Yahoo labels - screened through
    # `stock_for` alone a bank would arrive unclassified. So a name in that
    # file is screened as the swing book screens it: NSE stock, NSE vocabulary.
    nse: dict[str, object] = {}
    try:
        from ..swing.universe import load_universe
        india = registered.get(markets_mod.INDIA)
        if india is not None:
            nse = {s.symbol.upper(): s
                   for s in load_universe(at_root(india.universe_csv))}
    except Exception:
        nse = {}

    def _screen(position: Position):
        market, facts = _facts(position.market)
        if market is None:
            return None
        symbol = position.symbol.upper()
        f = facts.get(symbol)
        if f is None and symbol not in overrides:
            return None
        if position.market == markets_mod.INDIA and symbol in nse:
            stock, market = nse[symbol], registered[markets_mod.INDIA]
        else:
            stock = stock_for(symbol, market, f)
        v = halal.screen(stock, f, cfg, overrides=overrides, market=market)
        return v.eligible, v.summary, v.source

    return _screen
