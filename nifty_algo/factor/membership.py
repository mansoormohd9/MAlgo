"""
Where a stock lives in the Nifty universe.

The factor sleeve ranks ~2,400 NSE names, most of which are nowhere near an
index anyone quotes. "RELIANCE" and a Rs 40cr-a-day small cap are the same row
in a momentum table and are not the same decision, so every pick carries the
band it came from.

WHY THIS IS A FILE AND NOT A CALCULATION. Index membership is a committee's
published fact, not something derivable from price data. Turnover rank
correlates with it and is not it - NSE's selection uses free-float market cap,
listing history and a liquidity impact-cost test. Guessing the band from bars
would produce a label that is right most of the time, which is the worst kind.

BANDS ARE RESOLVED MOST-SPECIFIC-FIRST and a missing file is reported, never
inferred. If `nifty50.csv` is absent, a Nifty 100 name is labelled
"Nifty 100 (50/next-50 split unavailable)" rather than being guessed into one
half - the same discipline `HalalVerdict` applies to an absent fundamental.

The files are committed rather than fetched at scan time for the reason
`data/nifty100.csv` is: NSE 403s unattended clients often enough that a failed
download reads exactly like a quiet market.
"""
from __future__ import annotations

import csv
import io as _io
from dataclasses import dataclass, field
from pathlib import Path

#: band key -> (committed file, NSE's published list)
INDEX_FILES: dict[str, tuple[str, str]] = {
    "nifty50": ("data/nifty50.csv",
                "https://nsearchives.nseindia.com/content/indices/"
                "ind_nifty50list.csv"),
    "nifty100": ("data/nifty100.csv", ""),      # already maintained by the
                                                # swing book's own refresher
    "nifty500": ("data/nifty500.csv",
                 "https://nsearchives.nseindia.com/content/indices/"
                 "ind_nifty500list.csv"),
}

#: Most specific first. The label is what reaches the screen.
BANDS: tuple[tuple[str, str], ...] = (
    ("nifty50", "Nifty 50"),
    ("nifty100", "Nifty Next 50"),
    ("nifty500", "Nifty 500"),
)

OUTSIDE = "Outside the Nifty 500"
UNKNOWN = "band unavailable"

_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0 Safari/537.36"),
    "Accept": "text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class Membership:
    """Which published index each symbol belongs to, and what could not be read."""
    sets: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    @property
    def available(self) -> bool:
        """
        True only if the FULL split is readable.

        Deliberately strict. A partial answer here does not degrade gracefully
        into a smaller answer - it degrades into a confident wrong one, because
        a Nifty 50 name with `nifty50.csv` missing looks exactly like a Next 50
        name.
        """
        return not self.missing

    def band_of(self, symbol: str) -> str:
        sym = symbol.upper().strip()
        for key, label in BANDS:
            members = self.sets.get(key)
            if members is None:
                continue
            if sym in members:
                if key == "nifty100" and "nifty50" in self.missing:
                    return "Nifty 100 (50 / Next 50 split unavailable)"
                return label
        if "nifty500" in self.missing:
            return UNKNOWN
        return OUTSIDE

    def note(self) -> str:
        if self.available:
            return (f"Nifty 50 / Next 50 / 500 membership loaded "
                    f"({len(self.sets.get('nifty500', ())):,} names in the 500)")
        return ("membership incomplete - missing "
                + ", ".join(INDEX_FILES[k][0] for k in self.missing)
                + ". Refresh with: python -m nifty_algo.factor.membership --refresh")


def _symbols_from(path: Path) -> set:
    """
    Symbols out of an NSE constituent CSV or one of ours.

    NSE publishes `Company Name,Industry,Symbol,Series,ISIN Code`; the swing
    book's own file uses a lowercase `symbol`. Both are accepted so this reads
    `data/nifty100.csv` without duplicating it.
    """
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(
        _io.StringIO("\n".join(l for l in text.splitlines()
                               if l.strip() and not l.lstrip().startswith("#")))))
    out = set()
    for row in rows:
        for key in ("Symbol", "symbol", "SYMBOL"):
            value = (row.get(key) or "").strip().upper()
            if value:
                out.add(value)
                break
    return out


def load(root: str | Path = ".") -> Membership:
    root = Path(root)
    m = Membership()
    for key, (rel, _url) in INDEX_FILES.items():
        path = root / rel
        try:
            members = _symbols_from(path)
        except Exception:
            members = set()
        if members:
            m.sets[key] = members
        else:
            m.missing.append(key)
    return m


def refresh(root: str | Path = ".", timeout: int = 20) -> list[tuple[str, bool, str]]:
    """
    Re-download the lists NSE publishes. `(band, ok, detail)` per file.

    Returns rather than raises, exactly as `universe.refresh_from_nse` does: a
    failed refresh must leave the working committed list in place and give you
    a message, not an exception and no membership.
    """
    try:
        import requests
    except ImportError:                                  # pragma: no cover
        return [(k, False, "requests is not installed") for k in INDEX_FILES]

    out = []
    for key, (rel, url) in INDEX_FILES.items():
        if not url:
            out.append((key, True, f"{rel} is maintained by the swing book"))
            continue
        try:
            resp = requests.get(url, headers=_NSE_HEADERS, timeout=timeout)
            resp.raise_for_status()
            path = Path(root) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(resp.text, encoding="utf-8")
            out.append((key, True, f"{rel}: {len(_symbols_from(path)):,} symbols"))
        except Exception as e:
            out.append((key, False, f"{rel} unchanged ({e})"))
    return out


def _main() -> int:                                        # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Nifty index membership.")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    if args.refresh:
        for key, ok, detail in refresh():
            print(f"  [{'ok ' if ok else 'FAIL'}] {key}: {detail}")
    m = load()
    print(m.note())
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(_main())
