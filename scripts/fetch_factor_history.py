"""
Daily bars for a WIDE NSE equity universe, for the factor book.

    python scripts/fetch_factor_history.py --years 10
    python scripts/fetch_factor_history.py --years 10 --limit 500

Writes data/cache/factor_daily_india.parquet in the same tidy long shape the
other caches use - one frame, a `symbol` column, benchmark under
`__BENCHMARK__:^NSEI`.

WHY THIS EXISTS RATHER THAN REUSING fetch_swing_history.py

That script pulls from Yahoo for the ~100 names in a hand-maintained universe
CSV. This needs roughly 2,500 names and there is no CSV for them: NSE 403s
unattended clients, which is why `build_universe.py` automates only US and UK.
The universe here is DERIVED - every NSE `EQ` tradingsymbol from Kite's own
instrument dump, minus series suffixes (-BE trade-to-trade, -SM SME, -ST) and
minus obvious ETF and index tickers, which a single-stock momentum book must
never hold.

Kite serves 2,000 days of DAILY bars per request, so a decade of history is
ONE request per symbol - about 14 minutes for the whole universe at the 3/sec
historical cap. That is cheap enough to refetch rather than reconcile.

WHAT THIS FIXES, AND WHAT IT CANNOT

It fixes the LIQUIDITY RANGE. The Nifty 100 spans Rs 69cr to Rs 2,583cr of
daily turnover - every name clears the swing book's Rs 25cr floor by at least
2.8x, so that screen is inert and the low-turnover band where Indian momentum
alpha is reported to live is entirely absent from the universe. This reaches
it.

It does NOT fix SURVIVORSHIP. Kite's dump is today's listed set, so a company
delisted in 2019 is missing here and missing from any history fetched. For a
long-only momentum study that bias runs one way and is potentially large,
because momentum's mechanism is losers continuing to lose and the worst losers
are exactly the names that stopped being listed. `factor/universe.py` bounds
it by reporting results on the subset listed before the window opened; nothing
here removes it.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_algo.broker.kite_auth import KiteSession, NotAuthenticated  # noqa: E402
from nifty_algo.config import DEFAULT                                  # noqa: E402
from nifty_algo.data import instruments as instr                       # noqa: E402
from nifty_algo.data.base import FeedError                             # noqa: E402
from nifty_algo.data.kite_feed import NIFTY_50_TOKEN, KiteFeed         # noqa: E402

#: Kite's daily archive does not reach back indefinitely.
EARLIEST = date(2005, 1, 3)

#: Flush every this many symbols. The intraday fetcher learned this the hard
#: way: a run that saves only at the end discards everything on any mid-loop
#: failure, and "resumable" then only means "across successful runs".
CHECKPOINT_EVERY = 200

CACHE_NAME = "factor_daily_india.parquet"
BENCHMARK_KEY = "__BENCHMARK__:^NSEI"

#: Series suffixes that are not ordinary rolling-settlement equity.
#: `instrument_type == "EQ"` does NOT exclude these - the dump carries
#: GATECHDVR-BE, MINDPOOL-SM, K2INFRA-ST and similar under that type.
SUFFIXED = "-"

#: Fragments that mark an ETF, index fund or fund-of-funds. A single-stock
#: momentum book holding NIFTYBEES is not running the strategy being tested,
#: and the halal screen has no opinion on a basket.
FUND_MARKERS = ("ETF", "BEES", "LIQUID", "GOLD", "SILVER", "NIFTY", "SENSEX",
                "INAV", "IETF", "MAFANG", "GILT", "SDL", "NAV", "MOM100",
                "MON100", "HNGSNGBEES")


def candidate_symbols(tokens: dict[str, int]) -> dict[str, int]:
    """Ordinary NSE equity, with funds and non-rolling series removed."""
    out = {}
    for symbol, token in tokens.items():
        upper = symbol.upper()
        if SUFFIXED in symbol:
            continue
        if any(marker in upper for marker in FUND_MARKERS):
            continue
        out[symbol] = token
    return out


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=0,
                   help="fetch only the first N symbols, for a smoke test")
    p.add_argument("--full", action="store_true",
                   help="ignore the cache and refetch everything")
    args = p.parse_args(argv)

    end = date.today()
    start = max(EARLIEST, end - timedelta(days=int(args.years * 365)))
    cache_dir = Path(cfg.intraday_equity.cache_dir)
    path = cache_dir / CACHE_NAME

    try:
        session = KiteSession()
        all_tokens, _names, fetched_on, from_cache = instr.load_tokens(
            session, cache_dir=str(cache_dir))
    except NotAuthenticated as e:
        print(f"Not logged in: {e}\nRun: python -m nifty_algo.broker.kite_login")
        return 2
    except instr.InstrumentsUnavailable as e:
        print(f"Could not resolve NSE instruments: {e}")
        return 1

    universe = candidate_symbols(all_tokens)
    print(f"{len(all_tokens):,} NSE EQ symbols "
          f"({'cache' if from_cache else 'Kite'}, {fetched_on}) "
          f"-> {len(universe):,} ordinary equity after removing series "
          f"suffixes and funds")

    wanted = sorted(universe)
    if args.limit:
        wanted = wanted[:args.limit]
    print(f"fetching {len(wanted):,} symbols, {start} -> {end}\n")

    held: dict[str, pd.DataFrame] = {}
    benchmark = None
    if path.exists() and not args.full:
        try:
            raw = pd.read_parquet(path)
            for symbol, group in raw.groupby("symbol"):
                g = group.drop(columns=["symbol"]).sort_index()
                if symbol == BENCHMARK_KEY:
                    benchmark = g
                else:
                    held[str(symbol)] = g
            print(f"resuming from {len(held):,} cached symbols\n")
        except Exception as e:
            print(f"cache unreadable ({e}) - refetching in full\n")

    def _checkpoint() -> None:
        if not held:
            return
        parts = []
        for symbol, df in held.items():
            q = df.copy()
            q["symbol"] = symbol
            parts.append(q)
        if benchmark is not None and not benchmark.empty:
            q = benchmark.copy()
            q["symbol"] = BENCHMARK_KEY
            parts.append(q)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(parts).sort_index().to_parquet(path)
        except Exception as e:                              # pragma: no cover
            print(f"\n  checkpoint failed ({e}) - continuing")

    jobs = [("__BENCHMARK__", NIFTY_50_TOKEN)] + [
        (s, universe[s]) for s in wanted if s not in held]
    print(f"{len(jobs) - 1:,} still to fetch\n")

    done = 0
    try:
        for n, (symbol, token) in enumerate(jobs, 1):
            try:
                feed = KiteFeed(session=session, instrument_token=int(token),
                                interval_minutes=1440,
                                has_volume=(symbol != "__BENCHMARK__"))
                df = feed.fetch(start, end)
                if df is not None and not df.empty:
                    if symbol == "__BENCHMARK__":
                        benchmark = df
                    else:
                        held[symbol] = df
                    done += 1
            except (FeedError, NotAuthenticated) as e:
                print(f"  {symbol}: {e}")
            time.sleep(0.35)

            if n % 50 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}] {symbol:<14} "
                      f"{len(held):,} symbols held", flush=True)
            if n % CHECKPOINT_EVERY == 0:
                _checkpoint()
    finally:
        _checkpoint()

    if not held:
        print("nothing fetched")
        return 1

    sizes = sorted(len(d) for d in held.values())
    sessions = sorted({t.date() for d in held.values() for t in d.index})
    if not sessions:
        print("every frame came back empty - check the Historical Data "
              "subscription before assuming a quiet market")
        return 1

    print(f"\ncached {len(held):,} symbols, {len(sessions):,} sessions "
          f"({sessions[0]} -> {sessions[-1]})")
    print(f"bars per symbol: min {sizes[0]:,}  "
          f"median {sizes[len(sizes) // 2]:,}  max {sizes[-1]:,}")
    print(f"benchmark: {0 if benchmark is None else len(benchmark):,} bars")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
