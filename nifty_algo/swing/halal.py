"""
Shariah screening.

THIS IS A SCREEN, NOT A CERTIFICATION. It automates the two tests that public
data can answer - what the company does, and what its balance sheet looks like
- and it is explicit about the test it cannot answer. Nothing here is a
religious ruling, and the module is written so that a person can check its
working rather than trust it.

THE THREE LAYERS, IN ORDER:

  1. Activity      what the business earns from. Conventional finance,
                   insurance, alcohol, tobacco, gambling, pork, adult and
                   conventional entertainment.
  2. Ratios        leverage and interest-bearing assets, against TOTAL ASSETS.
  3. Overrides     your own file. It wins over both, in both directions.

WHY TOTAL ASSETS AND NOT MARKET CAP

AAOIFI and Dow Jones both use market capitalisation as the denominator. That
is defensible for an index rebalanced quarterly and wrong for a screen that
reruns every morning: with a market-cap denominator a company can be
compliant on Monday and non-compliant on Friday having filed nothing at all,
purely because the share price moved. Total assets moves when a new balance
sheet lands, which is when the answer should change. The thresholds are the
conventional 33% / 33% / 49%.

WHAT THIS CANNOT DO

The non-compliant-income test - haram revenue must be under 5% of total
revenue - needs a line-item read of the annual report, and so does the
purification amount you would owe on any dividend. No free data source
carries either. Every verdict produced here says `haram_revenue_verified =
False` and the UI prints it on the card. Treat this module as a way to narrow
a hundred names down to a handful worth checking properly, never as the check
itself.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from . import halal_taxonomy
from . import markets as markets_mod

# The two ratio methodologies this module can apply. Both are mainstream and
# they disagree; see `_METHODS` below and the note on HalalConfig.
METHOD_FTSE = "ftse"
METHOD_AAOIFI = "aaoifi"

# ---------------- layer 1: business activity ----------------
#
# Matched case-insensitively as substrings against the universe file's
# `industry` first and `sector` second. Ordered most specific first so the
# reason a stock was excluded is the useful one.

#: The Indian table, re-exported so existing callers and tests keep working.
#: The definitive copy - and the GICS table beside it - live in
#: `halal_taxonomy.py`, because one vocabulary cannot serve two exchanges.
PROHIBITED_ACTIVITIES = halal_taxonomy.NSE_ACTIVITIES

#: Substrings that mark a business as financial for the purposes of the
#: ratio screen - the balance sheet of a lender is not comparable to that of
#: a manufacturer and the thresholds were never meant to apply to it.
_FINANCIAL_HINTS = ("bank", "financ", "nbfc", "insurance", "investment")

VERDICT_COMPLIANT = "compliant"
VERDICT_NON_COMPLIANT = "non_compliant"

# How a verdict was reached. `no_data` is deliberately distinct from
# `ratio`: both exclude the stock, but only one of them is a statement
# about the company.
SOURCE_ACTIVITY = "activity"
SOURCE_RATIO = "ratio"
SOURCE_OVERRIDE = "override"
SOURCE_NO_DATA = "no_data"


@dataclass
class MethodVerdict:
    """
    One methodology's ratio result.

    Kept separate from `HalalVerdict` because two standards can reach opposite
    conclusions on the same balance sheet and both answers are worth seeing.
    """
    method: str
    label: str
    denominator_label: str
    passed: bool
    checks: dict[str, dict] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    available: bool = True          # False when the denominator was missing

    @property
    def summary(self) -> str:
        if not self.available:
            return f"{self.label}: not computed - {self.denominator_label} unavailable"
        if self.failures:
            return f"{self.label}: " + "; ".join(self.failures)
        return f"{self.label}: passes"


@dataclass
class HalalVerdict:
    """
    One stock's screening result, with every number it was decided on.

    `eligible` is the only field the scanner reads. Everything else exists so
    the page can show its working and so you can argue with it.
    """
    symbol: str
    eligible: bool
    reason: str                                   # the headline, one line
    failures: list[str] = field(default_factory=list)
    checks: dict[str, dict] = field(default_factory=dict)
    source: str = SOURCE_RATIO   # activity | ratio | override | no_data
    balance_sheet_date: str | None = None
    haram_revenue_verified: bool = False           # always False - see module docstring

    #: Every methodology that was computed, keyed by METHOD_*. `eligible` and
    #: `checks` follow the PRIMARY one; the others are here so a disagreement
    #: is visible rather than resolved silently in this module's favour.
    verdicts: dict = field(default_factory=dict)
    primary_method: str = METHOD_FTSE
    market: str = markets_mod.INDIA

    @property
    def disagreement(self) -> bool:
        """
        Whether the computed methodologies reached different conclusions.

        Worth surfacing: it is exactly the SPUS-vs-HLAL split, and it means
        the answer depends on which standard you follow rather than on the
        company being clearly one thing or the other.
        """
        outcomes = {v.passed for v in self.verdicts.values() if v.available}
        return len(outcomes) > 1

    def other_methods(self) -> list:
        return [v for k, v in self.verdicts.items() if k != self.primary_method]

    @property
    def summary(self) -> str:
        if self.source == SOURCE_OVERRIDE:
            return f"manual override: {self.reason}"
        return self.reason

    @property
    def unverifiable(self) -> bool:
        """Excluded because the screen could not run, not because it failed."""
        return self.source == SOURCE_NO_DATA


def screen(stock, fundamentals, cfg,
           overrides: dict[str, dict] | None = None, market=None) -> HalalVerdict:
    """
    Screen one stock. `fundamentals` may be None - that is a failure, not a pass.

    `market` selects the activity vocabulary (NSE labels or Yahoo/GICS ones)
    and is recorded on the verdict. It defaults to India so every existing
    caller keeps its behaviour exactly.
    """
    hcfg = cfg.swing.halal
    overrides = overrides or {}
    market = market or cfg.swing.markets[cfg.swing.default_market]
    mkey = getattr(market, "key", markets_mod.INDIA)

    # --- layer 3 runs first as a short circuit, but it is the LAST word ---
    ov = overrides.get(stock.symbol.upper())
    if ov:
        verdict = str(ov.get("verdict", "")).strip().lower()
        note = str(ov.get("note", "")).strip() or "no note recorded"
        reviewed = str(ov.get("reviewed_on", "")).strip()
        detail = f"{note}" + (f" (reviewed {reviewed})" if reviewed else "")
        if verdict == VERDICT_COMPLIANT:
            return HalalVerdict(stock.symbol, True, detail,
                                source=SOURCE_OVERRIDE, market=mkey,
                                primary_method=hcfg.primary_method)
        if verdict == VERDICT_NON_COMPLIANT:
            return HalalVerdict(stock.symbol, False, detail,
                                failures=[f"override: {detail}"],
                                source=SOURCE_OVERRIDE, market=mkey,
                                primary_method=hcfg.primary_method)
        # An unrecognised verdict is a typo in your file. Fall through to the
        # automatic screen rather than guessing which way you meant it, and
        # let the loader's warning surface the bad row.

    # --- layer 1: activity ---
    #
    # FAIL CLOSED ON AN UNCLASSIFIED STOCK. If neither `sector` nor `industry`
    # says anything, the activity table has nothing to match and the screen
    # silently reports "activity permissible" - a fail-OPEN in a module whose
    # entire contract is the opposite. It is not hypothetical: Yahoo returns
    # no classification at all for UK closed-end investment trusts, so FCIT
    # and Scottish Mortgage sailed through a screen that correctly excludes
    # III and Pershing Square, which it happens to label "Asset Management".
    # Absent data is a failure to verify here exactly as it is for a missing
    # balance sheet.
    if not _is_classified(stock):
        return HalalVerdict(
            stock.symbol, False,
            "cannot verify - no sector or industry classification, so the "
            "business activity screen could not be run",
            failures=["insufficient data: unclassified business activity"],
            source=SOURCE_NO_DATA, market=mkey,
            primary_method=hcfg.primary_method,
        )

    hit = activity_failure(stock, market=market, cfg=cfg)
    if hit:
        return HalalVerdict(
            stock.symbol, False, f"business activity: {hit}",
            failures=[f"activity: {hit}"], source=SOURCE_ACTIVITY,
            market=mkey, primary_method=hcfg.primary_method,
        )

    # --- layer 2: ratios ---
    if fundamentals is None or not fundamentals.has_balance_sheet:
        missing = _missing_fields(fundamentals)
        if hcfg.exclude_on_missing_data:
            # SOURCE_NO_DATA, not "ratio". The stock is out either way, but
            # "we could not check this" is a different statement from "this
            # failed the check", and a renamed ticker or a Yahoo outage must
            # never be presented to you as a ruling that the company is
            # non-compliant.
            return HalalVerdict(
                stock.symbol, False,
                f"cannot verify - no usable balance sheet ({missing})",
                failures=[f"insufficient data: {missing}"],
                source=SOURCE_NO_DATA, market=mkey,
                primary_method=hcfg.primary_method,
                balance_sheet_date=getattr(fundamentals, "balance_sheet_date", None),
            )
        return HalalVerdict(stock.symbol, True,
                            f"ratios not checked - {missing}",
                            source=SOURCE_RATIO, market=mkey,
                            primary_method=hcfg.primary_method)

    # --- layer 2: ratios, under every configured methodology ---
    computed = {
        method: _run_method(method, fundamentals, hcfg)
        for method in hcfg.methods()
    }
    primary = computed.get(hcfg.primary_method)
    if primary is None or not primary.available:
        # The primary standard could not be computed - most often AAOIFI with
        # no market cap. That is a failure to verify, not a pass, and it is
        # reported as such even if the other standard happened to succeed.
        detail = (primary.denominator_label if primary
                  else hcfg.primary_method)
        return HalalVerdict(
            stock.symbol, False,
            f"cannot verify - {detail} unavailable for the "
            f"{hcfg.primary_method.upper()} screen",
            failures=[f"insufficient data: {detail}"],
            source=SOURCE_NO_DATA, verdicts=computed, market=mkey,
            primary_method=hcfg.primary_method,
            balance_sheet_date=fundamentals.balance_sheet_date,
        )

    if primary.failures:
        return HalalVerdict(
            stock.symbol, False, "; ".join(primary.failures),
            failures=list(primary.failures), checks=primary.checks,
            source=SOURCE_RATIO, verdicts=computed, market=mkey,
            primary_method=hcfg.primary_method,
            balance_sheet_date=fundamentals.balance_sheet_date,
        )

    passed = ", ".join(f"{c['label'].split(' /')[0].lower()} {c['value']:.1%}"
                       for c in primary.checks.values())
    return HalalVerdict(
        stock.symbol, True, f"activity permissible; {passed}",
        checks=primary.checks, source=SOURCE_RATIO, verdicts=computed,
        market=mkey, primary_method=hcfg.primary_method,
        balance_sheet_date=fundamentals.balance_sheet_date,
    )


#: method -> (label, denominator attribute, denominator label, threshold attrs).
#: The check KEYS are deliberately identical across methods so the page can
#: render either one with the same table.
_METHODS = {
    METHOD_FTSE: (
        "FTSE / Yasaar", "total_assets", "total assets",
        ("debt_to_assets_max", "cash_and_interest_to_assets_max",
         "receivables_to_assets_max"),
    ),
    METHOD_AAOIFI: (
        "AAOIFI / S&P", "market_cap", "market capitalisation",
        ("aaoifi_debt_max", "aaoifi_cash_and_interest_max",
         "aaoifi_receivables_max"),
    ),
}


def _run_method(method: str, f, hcfg) -> MethodVerdict:
    """
    Apply one methodology's three ratios.

    The only difference between them is the denominator and the limits, which
    is why this is one function and not two screens.
    """
    label, denom_attr, denom_label, attrs = _METHODS[method]
    denominator = getattr(f, denom_attr, None)

    if denominator is None or float(denominator) <= 0:
        return MethodVerdict(method, label, denom_label, passed=False,
                             available=False)

    denominator = float(denominator)
    limits = [getattr(hcfg, a) for a in attrs]
    checks = {
        "debt_to_assets": _check(
            f"Debt / {denom_label}", f.total_debt, denominator, limits[0]),
        "cash_to_assets": _check(
            f"Cash + interest-bearing / {denom_label}",
            f.cash_and_investments, denominator, limits[1]),
        "receivables_to_assets": _check(
            f"Receivables / {denom_label}", f.receivables, denominator,
            limits[2]),
    }
    failures = [f"{c['label']} {c['value']:.1%} > {c['limit']:.0%}"
                for c in checks.values() if not c["passed"]]
    return MethodVerdict(method, label, denom_label, passed=not failures,
                         checks=checks, failures=failures)


def activity_failure(stock, market=None, cfg=None) -> str | None:
    """
    The prohibited activity this stock's classification matches, or None.

    Reads `industry` before `sector` because the industry label is the
    specific one: ITC's sector is FMCG, which is fine, and its industry names
    tobacco, which is not.

    `market` chooses the vocabulary. Defaults to the Indian table so that
    every pre-existing caller behaves identically.
    """
    taxonomy = getattr(market, "taxonomy", markets_mod.TAXONOMY_NSE)
    table = halal_taxonomy.table_for(taxonomy)
    if cfg is not None:
        table = table + halal_taxonomy.toggled_for(taxonomy, cfg.swing.halal)

    haystacks = (f"{stock.industry}".lower(), f"{stock.sector}".lower())
    for needle, label in table:
        for hay in haystacks:
            if needle in hay:
                return label
    return None


#: Placeholders that mean "we do not know", not "none of the above". Written
#: by `scripts/build_universe.py` when Yahoo returns nothing.
_UNCLASSIFIED = ("", "unclassified", "none", "n/a", "-", "unknown")


def _is_classified(stock) -> bool:
    """Whether there is anything for the activity table to match against."""
    return not all(
        str(getattr(stock, field, "") or "").strip().lower() in _UNCLASSIFIED
        for field in ("industry", "sector")
    )


def is_financial(stock) -> bool:
    """Whether the ratio thresholds were ever meant to apply to this balance sheet."""
    blob = f"{stock.industry} {stock.sector}".lower()
    return any(h in blob for h in _FINANCIAL_HINTS)


def _check(label: str, numerator: float | None, assets: float,
           limit: float) -> dict:
    if numerator is None or assets <= 0:
        return {"label": label, "value": float("nan"), "limit": limit,
                "passed": False, "note": "not reported"}
    value = float(numerator) / assets
    return {"label": label, "value": value, "limit": limit,
            "passed": value <= limit, "note": ""}


def _missing_fields(f) -> str:
    if f is None:
        return "no data fetched"
    if f.error:
        return f.error
    missing = [name for name, value in (
        ("total assets", f.total_assets),
        ("total debt", f.total_debt),
        ("cash", f.cash_and_investments),
        ("receivables", f.receivables),
    ) if value is None]
    return ("missing " + ", ".join(missing)) if missing else "unusable figures"


# ---------------- the override file ----------------

def load_overrides(path: str | Path) -> tuple[dict[str, dict], list[str]]:
    """
    Read your rulings.

    Returns (overrides, warnings). A malformed row becomes a warning and is
    skipped rather than raising: this file is hand-edited, and one bad line
    must not take the scan down. But the warnings are surfaced on the page -
    a ruling you meant to apply and which was silently dropped is worse than
    no ruling at all.
    """
    p = Path(path)
    if not p.exists():
        return {}, []

    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    except Exception as e:
        return {}, [f"could not read {p}: {e}"]

    out: dict[str, dict] = {}
    warnings: list[str] = []
    for i, row in enumerate(csv.DictReader(lines), start=2):
        symbol = (row.get("symbol") or "").strip().upper()
        verdict = (row.get("verdict") or "").strip().lower()
        if not symbol:
            continue
        if verdict not in (VERDICT_COMPLIANT, VERDICT_NON_COMPLIANT):
            warnings.append(
                f"{p.name} row {i} ({symbol}): verdict '{verdict}' is not "
                f"'{VERDICT_COMPLIANT}' or '{VERDICT_NON_COMPLIANT}' - "
                f"this row was IGNORED and the automatic screen applied."
            )
            continue
        out[symbol] = {
            "verdict": verdict,
            "note": (row.get("note") or "").strip(),
            "reviewed_on": (row.get("reviewed_on") or "").strip(),
        }
    return out, warnings


def save_overrides(path: str | Path, rows: list[dict]) -> None:
    """Rewrite the override file, preserving the explanatory header."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Your final word on the halal screen. Anything listed here wins over\n"
        "# both the activity screen and the computed ratios.\n"
        "#\n"
        "#   verdict     : compliant | non_compliant\n"
        "#   note        : why - this is the audit trail for a decision you made\n"
        "#   reviewed_on : YYYY-MM-DD, so a stale ruling is visible as stale\n"
    )
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(header)
        w = csv.writer(f)
        w.writerow(["symbol", "verdict", "note", "reviewed_on"])
        for row in rows:
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            w.writerow([symbol,
                        str(row.get("verdict", "")).strip().lower(),
                        str(row.get("note", "")).strip(),
                        str(row.get("reviewed_on", "")).strip()])


DISCLAIMER = (
    "This is an automated screen, not a certification and not a religious "
    "ruling. It checks business activity and balance-sheet ratios from public "
    "data. It CANNOT check the non-compliant-income test (haram revenue under "
    "5% of total) or compute the purification amount - both need the annual "
    "report. Verify with a scholar or a certified screening service before you "
    "act on any of this."
)
