"""
The Nifty 100 universe.

Committed to the repo as `data/nifty100.csv` rather than fetched, for one
reason: NSE blocks unattended HTTP clients frequently and without warning. A
scanner whose first step is a request to nseindia.com is a scanner that
silently does nothing on the days that request fails, and "no picks today"
reads identically whether the market was quiet or the download 403'd.

So the committed file is the source of truth and the NSE fetch is an
explicitly-triggered refresh that either succeeds or reports why it did not.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"

# NSE serves this only to something that looks like a browser arriving from
# their own site. Without the Referer the archive host returns 403.
_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,*/*",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}


class UniverseError(RuntimeError):
    """The universe file is missing or unusable. The scan cannot start."""


@dataclass(frozen=True)
class Stock:
    symbol: str
    name: str
    sector: str
    industry: str
    yf_ticker: str

    @property
    def search_name(self) -> str:
        """
        The company name as a news query.

        Trailing corporate suffixes pull in unrelated results and cost query
        precision, so they come off: "Tata Consumer Products Ltd" -> "Tata
        Consumer Products".
        """
        name = self.name
        for suffix in (" Limited", " Ltd.", " Ltd", " Corporation", " Corp."):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name.strip()


REQUIRED_COLUMNS = ("symbol", "name", "sector", "industry", "yf_ticker")


def load_universe(path: str | Path = "data/nifty100.csv") -> list[Stock]:
    """
    Read the constituent list.

    Lines beginning with `#` are comments. A row missing a column is skipped
    rather than fatal - one malformed line should not cost you the other 99 -
    but a file with no usable rows at all raises, because that is not a
    degraded scan, it is no scan.
    """
    p = Path(path)
    if not p.exists():
        raise UniverseError(
            f"universe file not found: {p}. This file is committed to the "
            f"repo; if it is missing, restore it from git."
        )

    rows: list[Stock] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        if not all(row.get(c) for c in REQUIRED_COLUMNS):
            continue
        rows.append(Stock(
            symbol=row["symbol"].strip().upper(),
            name=row["name"].strip(),
            sector=row["sector"].strip(),
            industry=row["industry"].strip(),
            yf_ticker=row["yf_ticker"].strip(),
        ))

    if not rows:
        raise UniverseError(f"universe file {p} has no usable rows")

    # A duplicate symbol would be scanned twice and could take two of the
    # three slots with one company.
    seen: set[str] = set()
    unique: list[Stock] = []
    for s in rows:
        if s.symbol in seen:
            continue
        seen.add(s.symbol)
        unique.append(s)
    return unique


def refresh_from_nse(path: str | Path = "data/nifty100.csv",
                     timeout: int = 20) -> tuple[bool, str]:
    """
    Pull the current constituent list from NSE and rewrite the file.

    Returns (ok, detail) rather than raising - the same shape the alert
    channels use - because a failed refresh must leave you with the working
    committed list and a message, not an exception and no universe.

    NSE's file carries a macro sector but not the fine-grained industry the
    halal activity screen reads, so the existing industry is preserved per
    symbol. Genuinely new symbols come back with the sector copied into the
    industry column and are named in the return message: until you classify
    them by hand the activity screen is working from a coarse label, and that
    is something to be told about rather than to discover later.
    """
    try:
        import requests
    except ImportError:                                  # pragma: no cover
        return False, "requests is not installed - pip install requests"

    try:
        resp = requests.get(NSE_URL, headers=_NSE_HEADERS, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        return False, (f"NSE refresh failed ({e}). The committed list is "
                       f"unchanged and the scan will use it.")

    try:
        incoming = list(csv.DictReader(text.splitlines()))
    except Exception as e:
        return False, f"NSE returned something that is not a CSV: {e}"

    if not incoming or "Symbol" not in (incoming[0] or {}):
        return False, ("NSE returned no usable rows - this usually means the "
                       "request was blocked and served an HTML page.")

    existing = {s.symbol: s for s in _load_lenient(path)}
    out: list[Stock] = []
    unclassified: list[str] = []

    for row in incoming:
        symbol = (row.get("Symbol") or "").strip().upper()
        if not symbol:
            continue
        name = (row.get("Company Name") or symbol).strip()
        sector = (row.get("Industry") or "Unclassified").strip().title()
        prior = existing.get(symbol)
        if prior is not None:
            industry = prior.industry
            yf_ticker = prior.yf_ticker
        else:
            industry = sector
            yf_ticker = f"{symbol}.NS"
            unclassified.append(symbol)
        out.append(Stock(symbol, name, sector, industry, yf_ticker))

    if len(out) < 50:
        return False, (f"NSE returned only {len(out)} rows for a 100-stock "
                       f"index - refusing to overwrite the committed list.")

    _write(path, out)

    detail = f"Universe refreshed from NSE: {len(out)} symbols."
    if unclassified:
        detail += (f" NEW and not yet classified for the halal activity "
                   f"screen: {', '.join(sorted(unclassified))}. They are "
                   f"screened on their broad sector until you set a specific "
                   f"industry in data/nifty100.csv.")
    return True, detail


def _load_lenient(path: str | Path) -> list[Stock]:
    """Load for merging - a missing or broken file is simply an empty prior."""
    try:
        return load_universe(path)
    except UniverseError:
        return []


def _write(path: str | Path, stocks: list[Stock]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write("# Nifty 100 constituents (Nifty 50 + Nifty Next 50).\n")
        f.write("# Refreshed from NSE via the Daily picks page.\n")
        f.write("# `industry` is what the halal activity screen reads - "
                "keep it specific.\n")
        w = csv.writer(f)
        w.writerow(REQUIRED_COLUMNS)
        for s in stocks:
            w.writerow([s.symbol, s.name, s.sector, s.industry, s.yf_ticker])
