"""
What the Shariah ETFs already own, and therefore what you already own.

WHY A SCREENER NEEDS THIS. If you hold SPUS and HLAL and the US scan hands you
a ticket for NVDA, that ticket is not a new position - NVDA is about 13% of
SPUS and around 8% of HLAL. Buying it directly is adding to a bet you already
have, and the pick card is the only place that fact can be stated before the
order is placed. A screener that reports "diversify into NVDA" to somebody
whose funds are a quarter NVDA is not wrong about the setup; it is wrong about
the portfolio, which is the thing that actually loses money.

This is a BADGE, NEVER A REJECTION. Concentration is a decision, and it is
yours. What this module refuses to do is let the decision be made silently.

WHERE THE NUMBERS COME FROM. Both issuers publish their full book daily:

  SPUS  sp-funds.com, the Tidal administrator's holdings CSV
  HLAL  Wahed's page reads a Google Sheet the administrator writes

Same rule as `universe.py`: the committed CSV is the source of truth and the
network fetch is an explicitly-triggered refresh that either succeeds or
reports why it did not. An overlap check that silently returns "no overlap"
because a CDN was down is worse than no overlap check, because it reads as
reassurance.

WHAT THE FEEDS CONTAIN THAT YOU MUST DROP. Both files carry placeholder rows -
a zero-price line for a name in transition, an administrator's identifier
("2602335D") in the ticker column, a money-market sweep. Left in, they become
universe entries that no price feed can resolve and that show up forever as
"no data" rejections.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

#: The published books, keyed the way the CSV and the UI spell them.
SOURCES: dict[str, dict] = {
    "SPUS": {
        "label": "SP Funds S&P 500 Sharia (SPUS)",
        "url": "https://www.sp-funds.com/wp-content/uploads/data/"
               "TidalFG_Holdings_SPUS.csv",
        "domicile": "US",
        "standard": "AAOIFI / S&P",
    },
    "HLAL": {
        "label": "Wahed FTSE USA Shariah (HLAL)",
        "url": "https://docs.google.com/spreadsheets/d/"
               "1UC1Bk67bGuYsos_i8y_HQpNoHpVHAvqf71MbgrafJOQ/export?format=csv",
        "domicile": "US",
        "standard": "FTSE / Yasaar",
    },
}

#: A real US ticker is one to five letters, optionally with a class suffix
#: ("BRK.B"). Anything else in that column is an administrator's placeholder.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,*/*",
}


@dataclass(frozen=True)
class Holding:
    etf: str
    symbol: str
    name: str
    weight: float            # fraction of the fund, 0.0-1.0
    as_of: str = ""

    @property
    def weight_pct(self) -> float:
        return self.weight * 100.0


# ---------------- the committed file ----------------

def load_holdings(path: str | Path = "data/etf_holdings.csv"
                  ) -> dict[str, list[Holding]]:
    """
    Read the committed book, grouped by symbol.

    Keyed by SYMBOL rather than by ETF because every caller is asking the same
    question - "what do I already own of this one name" - and grouping the
    other way would make every lookup a scan of both funds.

    A missing file is an empty result, not an error: the overlap badge is an
    enhancement to a pick card, and losing it must not lose the scan.
    """
    p = Path(path)
    if not p.exists():
        return {}

    out: dict[str, list[Holding]] = {}
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    except OSError:
        return {}

    for row in csv.DictReader(lines):
        symbol = (row.get("symbol") or "").strip().upper()
        etf = (row.get("etf") or "").strip().upper()
        if not symbol or not etf:
            continue
        try:
            weight = float(row.get("weight") or 0.0)
        except ValueError:
            continue
        out.setdefault(symbol, []).append(Holding(
            etf=etf, symbol=symbol,
            name=(row.get("name") or "").strip(),
            weight=weight,
            as_of=(row.get("as_of") or "").strip(),
        ))
    return out


def save_holdings(path: str | Path, holdings: list[Holding]) -> None:
    """Rewrite the committed book, preserving the explanatory header."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write("# What the Shariah ETFs hold, for the overlap check on a\n"
                "# pick card. Refreshed from the issuers via the Daily picks\n"
                "# page; committed so a CDN outage cannot read as 'no overlap'.\n"
                "#\n"
                "#   weight : fraction of that fund, 0.0-1.0\n")
        w = csv.writer(f)
        w.writerow(["etf", "symbol", "name", "weight", "as_of"])
        for h in sorted(holdings, key=lambda x: (x.etf, -x.weight)):
            w.writerow([h.etf, h.symbol, h.name, f"{h.weight:.6f}", h.as_of])


# ---------------- the refresh ----------------

def fetch_holdings(etf: str, timeout: int = 30) -> tuple[list[Holding], str]:
    """
    Pull one fund's current book.

    Returns (holdings, detail) rather than raising - the same shape the alert
    channels and the NSE refresh use - so a failure leaves you with the
    committed file and a message.
    """
    source = SOURCES.get(etf.upper())
    if source is None:
        return [], f"no published source is registered for {etf}"

    try:
        import requests
    except ImportError:                                   # pragma: no cover
        return [], "requests is not installed - pip install requests"

    try:
        resp = requests.get(source["url"], headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        return [], (f"{etf} refresh failed ({e}). The committed holdings are "
                    f"unchanged.")

    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return [], (f"{etf} returned no rows - this usually means the request "
                    f"was served an HTML error page rather than the CSV.")

    holdings, dropped = _parse(etf.upper(), rows)
    if not holdings:
        return [], (f"{etf} returned {len(rows)} rows but none of them parsed "
                    f"as a holding - the file layout has probably changed.")

    detail = f"{etf}: {len(holdings)} holdings"
    if dropped:
        detail += f" ({dropped} placeholder rows dropped)"
    if holdings[0].as_of:
        detail += f", as of {holdings[0].as_of}"
    return holdings, detail


def _parse(etf: str, rows: list[dict]) -> tuple[list[Holding], int]:
    """
    Both issuers happen to publish the same administrator layout.

    Written against the columns rather than their positions, and every row
    that does not yield a usable ticker AND a usable weight is counted out
    loud rather than quietly skipped.
    """
    out: list[Holding] = []
    dropped = 0
    seen: set[str] = set()

    for row in rows:
        symbol = (row.get("StockTicker") or row.get("Ticker") or "").strip().upper()
        weight_raw = (row.get("Weightings") or row.get("Weight") or "").strip()
        name = (row.get("SecurityName") or row.get("Name") or "").strip()

        if (row.get("MoneyMarketFlag") or "").strip():
            dropped += 1
            continue
        if not _TICKER_RE.match(symbol):
            # An administrator's internal identifier, not a tradeable ticker.
            dropped += 1
            continue

        weight = _percent(weight_raw)
        if weight is None or weight <= 0:
            # A zero weight is a name in transition, priced at 0 by the
            # administrator. It is not a position.
            dropped += 1
            continue
        if symbol in seen:
            dropped += 1
            continue
        seen.add(symbol)

        out.append(Holding(etf=etf, symbol=symbol, name=name, weight=weight,
                           as_of=_iso(row.get("Date"))))
    return out, dropped


def _percent(raw: str) -> float | None:
    """'13.62%' -> 0.1362. Returns None for anything unparseable."""
    if not raw:
        return None
    try:
        value = float(raw.replace("%", "").replace(",", "").strip())
    except ValueError:
        return None
    return value / 100.0


def _iso(raw) -> str:
    """The administrator writes MM/DD/YYYY; store ISO so it sorts."""
    text = str(raw or "").strip()
    if not text:
        return ""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text


def refresh_all(path: str | Path = "data/etf_holdings.csv",
                etfs: list[str] | None = None) -> tuple[bool, str]:
    """
    Refresh every registered fund and rewrite the committed file.

    Partial success is still success - one fund's CDN being down should not
    cost you the other fund's book - but the message names what failed, so a
    half-refreshed file is never mistaken for a complete one.
    """
    etfs = etfs or list(SOURCES)
    fetched: dict[str, list[Holding]] = {}
    notes: list[str] = []
    failures: list[str] = []

    for etf in etfs:
        rows, detail = fetch_holdings(etf)
        if rows:
            fetched[etf.upper()] = rows
            notes.append(detail)
        else:
            failures.append(detail)

    if not fetched:
        return False, ("No fund could be refreshed, so the committed file is "
                       "unchanged. " + " ".join(failures))

    # A fund that failed keeps the rows it already had. Writing only what was
    # fetched would silently delete the other fund's book, and the overlap
    # check would then report "not held by any of your funds" for names you
    # very much hold - reassurance produced by an outage.
    merged: list[Holding] = list(_flatten(fetched))
    prior = load_holdings(path)
    kept = 0
    for rows in prior.values():
        for h in rows:
            if h.etf not in fetched:
                merged.append(h)
                kept += 1

    save_holdings(path, merged)
    message = "Holdings refreshed - " + "; ".join(notes) + "."
    if failures:
        message += (f" NOT refreshed: {' '.join(failures)} Those funds keep "
                    f"their previously committed rows ({kept} of them), which "
                    f"are now older than the rest of this file.")
    return True, message


def _flatten(fetched: dict[str, list[Holding]]):
    for rows in fetched.values():
        yield from rows


# ---------------- the question a pick card asks ----------------

@dataclass
class Overlap:
    """How much of one name you already hold through the funds."""
    symbol: str
    rows: list[Holding]
    value_inr: float = 0.0          # 0 when fund balances are not configured

    @property
    def total_weight(self) -> float:
        """Summed weight. Not a portfolio share - see `note`."""
        return sum(h.weight for h in self.rows)

    @property
    def any(self) -> bool:
        return bool(self.rows)

    def heavy(self, threshold: float) -> bool:
        return any(h.weight >= threshold for h in self.rows)

    def note(self) -> str:
        if not self.rows:
            return "Not held by any of your Shariah funds - this is genuinely a new position."
        parts = ", ".join(f"{h.weight:.1%} of {h.etf}" for h in
                          sorted(self.rows, key=lambda x: -x.weight))
        text = f"Already held indirectly: {parts}."
        if self.value_inr > 0:
            text += (f" At your recorded fund balances that is about "
                     f"₹{self.value_inr:,.0f} of exposure already.")
        text += " A direct position adds to it rather than diversifying."
        return text


def overlap_for(symbol: str, by_symbol: dict[str, list[Holding]],
                balances_inr: dict[str, float] | None = None) -> Overlap:
    """
    What you already own of `symbol` through the funds.

    `balances_inr` maps an ETF to what you hold of it, in rupees. Supplying it
    turns a percentage into an amount, which is the form the question is
    actually asked in.
    """
    rows = list(by_symbol.get(symbol.upper(), ()))
    balances = balances_inr or {}
    value = sum(h.weight * balances.get(h.etf, 0.0) for h in rows)
    return Overlap(symbol=symbol.upper(), rows=rows, value_inr=value)


def as_of(by_symbol: dict[str, list[Holding]]) -> str:
    """The newest date in the committed book, for the freshness strip."""
    dates = {h.as_of for rows in by_symbol.values() for h in rows if h.as_of}
    return max(dates) if dates else ""


def universe_symbols(by_symbol: dict[str, list[Holding]]) -> list[str]:
    """
    The union of every fund's book - the US scan universe.

    Pre-screened by two professional Shariah boards, which makes the local
    screen a RE-VERIFICATION rather than the only line of defence. Where the
    two disagree, that disagreement is the interesting output.
    """
    return sorted(by_symbol)
