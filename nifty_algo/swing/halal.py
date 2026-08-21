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

# ---------------- layer 1: business activity ----------------
#
# Matched case-insensitively as substrings against the universe file's
# `industry` first and `sector` second. Ordered most specific first so the
# reason a stock was excluded is the useful one.

PROHIBITED_ACTIVITIES: tuple[tuple[str, str], ...] = (
    # Interest-based finance - the largest single block in the Nifty 100.
    ("private sector bank", "conventional banking (interest-based)"),
    ("public sector bank", "conventional banking (interest-based)"),
    ("non banking financial company", "interest-based lending (NBFC)"),
    ("nbfc", "interest-based lending (NBFC)"),
    ("housing finance", "interest-based lending"),
    ("financial institution", "interest-based finance"),
    ("investment company", "conventional investment holding"),
    ("holding company", "conventional financial holding"),
    ("asset management", "conventional asset management"),
    ("stockbroking", "conventional brokerage"),
    ("depositories", "conventional financial services"),
    ("bank", "conventional banking (interest-based)"),

    # Insurance.
    ("life insurance", "conventional insurance"),
    ("general insurance", "conventional insurance"),
    ("insurance", "conventional insurance"),

    # Intoxicants and tobacco.
    ("breweries", "alcohol"),
    ("distilleries", "alcohol"),
    ("alcohol", "alcohol"),
    ("liquor", "alcohol"),
    ("tobacco", "tobacco"),
    ("cigarette", "tobacco"),

    # Gambling and conventional entertainment.
    ("gambling", "gambling"),
    ("casino", "gambling"),
    ("lottery", "gambling"),
    ("gaming", "gambling / conventional gaming"),
    ("film production", "conventional entertainment"),
    ("movies & entertainment", "conventional entertainment"),
    ("cinema", "conventional entertainment"),
    ("multiplex", "conventional entertainment"),

    # Pork and non-halal meat.
    ("pork", "pork products"),

    # Hotels: the objection is the bar and banquet revenue, which for Indian
    # listed hotel groups is material and is not separately disclosed in any
    # free feed. Excluded here, and an obvious candidate for an override if
    # you check a specific company's accounts and disagree.
    ("hotels & resorts", "hotel bar/banquet revenue not separable"),
    ("hotels", "hotel bar/banquet revenue not separable"),
)

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
           overrides: dict[str, dict] | None = None) -> HalalVerdict:
    """
    Screen one stock. `fundamentals` may be None - that is a failure, not a pass.
    """
    hcfg = cfg.swing.halal
    overrides = overrides or {}

    # --- layer 3 runs first as a short circuit, but it is the LAST word ---
    ov = overrides.get(stock.symbol.upper())
    if ov:
        verdict = str(ov.get("verdict", "")).strip().lower()
        note = str(ov.get("note", "")).strip() or "no note recorded"
        reviewed = str(ov.get("reviewed_on", "")).strip()
        detail = f"{note}" + (f" (reviewed {reviewed})" if reviewed else "")
        if verdict == VERDICT_COMPLIANT:
            return HalalVerdict(stock.symbol, True, detail, source=SOURCE_OVERRIDE)
        if verdict == VERDICT_NON_COMPLIANT:
            return HalalVerdict(stock.symbol, False, detail,
                                failures=[f"override: {detail}"],
                                source=SOURCE_OVERRIDE)
        # An unrecognised verdict is a typo in your file. Fall through to the
        # automatic screen rather than guessing which way you meant it, and
        # let the loader's warning surface the bad row.

    # --- layer 1: activity ---
    hit = activity_failure(stock)
    if hit:
        return HalalVerdict(
            stock.symbol, False, f"business activity: {hit}",
            failures=[f"activity: {hit}"], source=SOURCE_ACTIVITY,
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
                source=SOURCE_NO_DATA,
                balance_sheet_date=getattr(fundamentals, "balance_sheet_date", None),
            )
        return HalalVerdict(stock.symbol, True,
                            f"ratios not checked - {missing}", source=SOURCE_RATIO)

    assets = float(fundamentals.total_assets)
    checks = {
        "debt_to_assets": _check(
            "Debt / total assets", fundamentals.total_debt, assets,
            hcfg.debt_to_assets_max),
        "cash_to_assets": _check(
            "Cash + interest-bearing / total assets",
            fundamentals.cash_and_investments, assets,
            hcfg.cash_and_interest_to_assets_max),
        "receivables_to_assets": _check(
            "Receivables / total assets", fundamentals.receivables, assets,
            hcfg.receivables_to_assets_max),
    }

    failures = [f"{c['label']} {c['value']:.1%} > {c['limit']:.0%}"
                for c in checks.values() if not c["passed"]]

    if failures:
        return HalalVerdict(
            stock.symbol, False, "; ".join(failures), failures=failures,
            checks=checks, source=SOURCE_RATIO,
            balance_sheet_date=fundamentals.balance_sheet_date,
        )

    passed = ", ".join(f"{c['label'].split(' /')[0].lower()} {c['value']:.1%}"
                       for c in checks.values())
    return HalalVerdict(
        stock.symbol, True, f"activity permissible; {passed}",
        checks=checks, source=SOURCE_RATIO,
        balance_sheet_date=fundamentals.balance_sheet_date,
    )


def activity_failure(stock) -> str | None:
    """
    The prohibited activity this stock's classification matches, or None.

    Reads `industry` before `sector` because the industry label is the
    specific one: ITC's sector is FMCG, which is fine, and its industry names
    tobacco, which is not.
    """
    haystacks = (f"{stock.industry}".lower(), f"{stock.sector}".lower())
    for needle, label in PROHIBITED_ACTIVITIES:
        for hay in haystacks:
            if needle in hay:
                return label
    return None


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
