"""
Build a market's universe file from a live source.

WHY THIS IS A SCRIPT AND NOT PART OF THE SCAN. Same argument as
`universe.refresh_from_nse`: the committed CSV is the source of truth and
building it is an explicit act. A scanner whose first step is three hundred
`Ticker.info` calls is a scanner that does nothing on the days that fails, and
"no picks today" reads identically whether the market was quiet or Yahoo
rate-limited you.

WHY THE LABELS COME FROM YAHOO AND NOT FROM ME. The halal activity screen
matches substrings against `industry` and `sector`. Those fields have to be
spelled exactly the way the screen will see them at scan time, and the only
authority on that is the same `info` dict `fundamentals.py` reads. Typing
"Banks - Diversified" by hand and having Yahoo return "Banks—Diversified" is a
gap that shows up as a bank passing the activity screen.

    python scripts/build_universe.py --market us
    python scripts/build_universe.py --market uk
    python scripts/build_universe.py --market us --skip-holdings-refresh
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_algo.config import DEFAULT                       # noqa: E402
from nifty_algo.swing import holdings as holdings_mod       # noqa: E402
from nifty_algo.swing import markets as markets_mod         # noqa: E402
from nifty_algo.swing.universe import REQUIRED_COLUMNS      # noqa: E402

HOLDINGS_CSV = "data/etf_holdings.csv"

#: FTSE 100 constituents, as LSE tickers without the `.L` suffix.
#:
#: Committed rather than scraped: the index reshuffles quarterly, iShares
#: blocks unattended clients, and Wikipedia's table markup changes shape
#: without notice. Every name here is verified against Yahoo when the file is
#: built, and anything that does not resolve is REPORTED rather than dropped
#: quietly - a constituent silently missing from the universe is a candidate
#: you never see and never know you did not see.
#: Verified against Yahoo on 21 Aug 2026. Three seeds were removed then and
#: the reasons are recorded so they are not "fixed" back in later:
#:   AHT  - Ashtead moved its primary listing to the US; no longer FTSE 100.
#:   PHNX - Phoenix Group has no Yahoo entry. An insurer, so the activity
#:          screen would exclude it regardless. Do NOT substitute PGH.L, which
#:          is Personal Group Holdings, a different company entirely.
#:   BDEV - became BTRW after the Barratt/Redrow merger.
FTSE100_SEED = (
    "AAF", "AAL", "ABF", "ADM", "ANTO", "AUTO", "AV", "AZN", "BA",
    "BARC", "BATS", "BTRW", "BEZ", "BKG", "BNZL", "BP", "BRBY", "BT-A",
    "CCH", "CNA", "CPG", "CRDA", "CTEC", "DCC", "DGE", "DPLM", "EDV", "ENT",
    "EXPN", "EZJ", "FCIT", "FRAS", "FRES", "GAW", "GLEN", "GSK", "HIK",
    "HLMA", "HLN", "HSBA", "HSX", "HWDN", "IAG", "ICG", "IHG", "III", "IMB",
    "IMI", "INF", "ITRK", "JD", "KGF", "LAND", "LGEN", "LLOY", "LMP", "LSEG",
    "MKS", "MNDI", "MNG", "MRO", "NG", "NWG", "NXT", "PRU", "PSH",
    "PSN", "PSON", "REL", "RIO", "RKT", "RMV", "RR", "RTO", "SBRY", "SDR",
    "SGE", "SGRO", "SHEL", "SMIN", "SMT", "SN", "SPX", "SSE", "STAN", "STJ",
    "SVT", "TSCO", "TW", "ULVR", "UTG", "UU", "VOD", "WEIR", "WPP", "WTB",
)


def main() -> int:
    cfg = DEFAULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True,
                        choices=[markets_mod.US, markets_mod.UK])
    parser.add_argument("--skip-holdings-refresh", action="store_true",
                        help="use the committed ETF holdings instead of "
                             "re-fetching them (US only)")
    parser.add_argument("--sleep", type=float, default=0.25,
                        help="pause between Yahoo calls")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    market = markets_mod.get(cfg, args.market)
    print(f"Building {market.label} -> {market.universe_csv}\n")

    if market.key == markets_mod.US:
        tickers, names = _us_seed(args.skip_holdings_refresh)
    else:
        tickers, names = _uk_seed()

    if not tickers:
        print("No seed tickers. Nothing written.")
        return 1

    print(f"\nEnriching {len(tickers)} symbols from Yahoo "
          f"(sector, industry, currency) ...")
    rows, failures = _enrich(tickers, names, market, args.sleep)

    if not rows:
        print("Nothing resolved. The committed file is unchanged.")
        return 1

    _write(market.universe_csv, market, rows)
    print(f"\nWrote {len(rows)} rows to {market.universe_csv}")

    if failures:
        # Named, not counted. See the note on FTSE100_SEED.
        print(f"\n{len(failures)} symbol(s) did NOT resolve and are absent "
              f"from the universe:")
        for symbol, why in failures:
            print(f"   {symbol:<10} {why}")

    unclassified = [r for r in rows if r["industry"] in ("", "Unclassified")]
    if unclassified:
        print(f"\n{len(unclassified)} symbol(s) came back with no industry. "
              f"The halal activity screen has nothing specific to match on "
              f"for these, so they are screened on sector alone:")
        for r in unclassified:
            print(f"   {r['symbol']:<10} sector={r['sector']}")
    return 0


# ---------------- seeds ----------------

def _us_seed(skip_refresh: bool) -> tuple[list[str], dict[str, str]]:
    """The union of the Shariah ETF books."""
    if not skip_refresh:
        print("Refreshing ETF holdings from the issuers ...")
        ok, detail = holdings_mod.refresh_all(HOLDINGS_CSV)
        print(f"   {detail}")
        if not ok:
            print("   Falling back to the committed holdings file.")

    by_symbol = holdings_mod.load_holdings(HOLDINGS_CSV)
    if not by_symbol:
        print("   No holdings available - cannot build the US universe.")
        return [], {}

    names = {sym: rows[0].name for sym, rows in by_symbol.items() if rows}
    symbols = holdings_mod.universe_symbols(by_symbol)
    funds = sorted({h.etf for rows in by_symbol.values() for h in rows})
    print(f"   Union of {', '.join(funds)}: {len(symbols)} distinct names "
          f"(as of {holdings_mod.as_of(by_symbol) or 'unknown'})")
    return symbols, names


def _uk_seed() -> tuple[list[str], dict[str, str]]:
    print(f"FTSE 100 seed list: {len(FTSE100_SEED)} symbols")
    return list(FTSE100_SEED), {}


# ---------------- enrichment ----------------

def _enrich(tickers, names, market, sleep):
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed - pip install yfinance")
        return [], []

    rows, failures = [], []
    suffix = ".L" if market.key == markets_mod.UK else ""

    for i, symbol in enumerate(tickers, 1):
        yf_ticker = f"{symbol}{suffix}"
        print(f"   [{i}/{len(tickers)}] {yf_ticker:<12}", end="\r")
        try:
            info = yf.Ticker(yf_ticker).info or {}
        except Exception as e:
            failures.append((symbol, f"info call failed: {e}"))
            time.sleep(sleep)
            continue

        # A ticker Yahoo does not know returns a dict with no price and no
        # classification. Treated as a failure, not as an unclassified name -
        # they are different problems with different fixes.
        price = info.get("regularMarketPrice") or info.get("previousClose")
        sector = (info.get("sector") or "").strip()
        if not price and not sector:
            failures.append((symbol, "Yahoo returned no price and no sector"))
            time.sleep(sleep)
            continue

        rows.append({
            "symbol": symbol,
            "name": (info.get("longName") or info.get("shortName")
                     or names.get(symbol) or symbol).strip(),
            "sector": sector or "Unclassified",
            "industry": (info.get("industry") or "").strip() or "Unclassified",
            "yf_ticker": yf_ticker,
            "currency": (info.get("currency") or "").strip(),
        })
        time.sleep(sleep)

    print(" " * 60, end="\r")
    return rows, failures


def _write(path, market, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pence = sum(1 for r in rows if r.get("currency") == "GBp")
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {market.label}\n")
        f.write("# Built by scripts/build_universe.py. `sector` and "
                "`industry` are Yahoo's own\n")
        f.write("# labels, verbatim - the halal activity screen matches "
                "against these strings,\n")
        f.write("# so they must be spelled the way the screen will see them "
                "at scan time.\n")
        if pence:
            f.write(f"# {pence} of these are quoted in pence (GBp); "
                    f"prices.py divides them by 100.\n")
        w = csv.writer(f)
        w.writerow(REQUIRED_COLUMNS)
        for r in sorted(rows, key=lambda x: x["symbol"]):
            w.writerow([r[c] for c in REQUIRED_COLUMNS])


if __name__ == "__main__":
    raise SystemExit(main())
