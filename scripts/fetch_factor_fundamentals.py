"""
Fundamentals for every name the factor sleeve ever shortlisted.

    python scripts/fetch_factor_fundamentals.py --shortlist 60

WHY A UNION RATHER THAN THE UNIVERSE. The factor universe is ~2,400 NSE names
and fundamentals are one slow request each, so screening all of them to rank 20
is the cost `swing/scanner.py`'s cheap-to-expensive gate ordering exists to
avoid. The sleeve screens DOWN the ranking, so the only names that can ever
matter are those that reached the shortlist on some rebalance - a few hundred
across twenty years rather than all of them.

WHAT THIS BUYS. Until it runs, `FactorConfig.halal_screened` can be applied
LIVE but cannot be replayed, so the recorded F1/F2 returns describe an
UNSCREENED book while a live console screens. Those are two different books,
and the gap between them is unmeasured. This makes the comparison possible.

WHAT IT STILL CANNOT FIX. The fetched balance sheets are TODAY'S. Applying them
to a 2008 rebalance is the same point-in-time distortion the swing backtest
already prints above every result, and it is not removable from free data. It
is a bound on the screened number, not a correction to it.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_algo.config import DEFAULT                        # noqa: E402
from nifty_algo.factor import momentum as mom                # noqa: E402
from nifty_algo.factor import sleeve as sl                   # noqa: E402
from nifty_algo.factor.universe import FactorUniverse, month_ends  # noqa: E402
from nifty_algo.swing import fundamentals as fund_mod        # noqa: E402
from nifty_algo.swing import markets as markets_mod          # noqa: E402


def shortlisted_symbols(cfg, universe: FactorUniverse, shortlist: int,
                        start: date | None = None) -> list[str]:
    """Every symbol that reached the top `shortlist` on any rebalance."""
    fcfg = cfg.factor
    sessions = sorted({d for sd in universe.symbols.values() for d in sd.dates})
    marks = month_ends(sessions, fcfg.hold_months)
    if start:
        marks = [m for m in marks if m >= start]
    seen: dict[str, int] = {}
    for n, day in enumerate(marks, 1):
        elig = universe.eligible_at(day, fcfg.min_price, fcfg.min_turnover_inr,
                                    fcfg.min_history_sessions, fcfg.band)
        scores = mom.score_universe(universe, elig.symbols, day, fcfg.formation)
        for symbol in mom.top_n(scores, shortlist):
            seen[symbol] = seen.get(symbol, 0) + 1
        if n % 24 == 0 or n == len(marks):
            print(f"  [{n}/{len(marks)}] {day}  {len(seen):,} distinct names",
                  flush=True)
    return sorted(seen)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shortlist", type=int, default=60)
    p.add_argument("--cache", default="", help="a different bars parquet")
    p.add_argument("--since", default="", help="YYYY-MM-DD; skip older marks")
    p.add_argument("--limit", type=int, default=0, help="smoke test")
    p.add_argument("--all-symbols", action="store_true",
                   help="fetch the WHOLE universe rather than the shortlist. "
                        "Needed for `restriction.size500` to be a clean "
                        "control: fetching only shortlisted names leaves "
                        "market caps for exactly the names that once had "
                        "strong momentum, so a size rank built on them is "
                        "pre-selected on the outcome it is supposed to be "
                        "independent of.")
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    if args.cache:
        cfg.factor.cache_name = args.cache
    market = markets_mod.factor_market(cfg)

    bars, _bench = sl.load_bars(cfg)
    print(f"{len(bars):,} symbols; walking the rebalances", flush=True)
    universe = FactorUniverse(bars, adv_window=cfg.factor.adv_window)

    start = date.fromisoformat(args.since) if args.since else None
    if args.all_symbols:
        symbols = sorted(bars)
        what = "the whole universe"
    else:
        symbols = shortlisted_symbols(cfg, universe, args.shortlist, start)
        what = f"distinct names ever in the top {args.shortlist}"
    if args.limit:
        symbols = symbols[:args.limit]
    print(f"\n{len(symbols):,} {what}\n", flush=True)

    stocks = [sl.stock_for(s, market) for s in symbols]
    started = time.time()

    def progress(done, total, label):
        if done % 25 == 0 or done == total:
            print(f"  [{done}/{total}] {label}  "
                  f"({time.time() - started:.0f}s)", flush=True)

    facts = fund_mod.load_fundamentals(stocks, cfg, market, progress=progress)

    ok = sum(1 for f in facts.values() if f and not f.error)
    classified = sum(1 for f in facts.values()
                     if f and (f.yahoo_industry or f.yahoo_sector))
    print(f"\nfetched {len(facts):,}; usable {ok:,}; classified {classified:,}")
    print(f"{len(facts) - classified:,} names have NO Yahoo classification - "
          f"the activity screen cannot run on those, and an unclassifiable "
          f"name is a REJECT rather than a pass.")
    print(f"cached in {Path(cfg.swing.cache_dir) / fund_mod.CACHE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
