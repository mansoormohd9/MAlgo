"""
F4. What does restricting the sleeve to the Nifty 500 cost, and can it be believed?

    python scripts/run_f4_universe.py
    python scripts/run_f4_universe.py --cache factor_daily_india_21y.parquet \
        --start 2005-09-12 --end 2016-08-31

The live sleeve's top 20 is 17 names outside the Nifty 500, one of them trading
Rs 2.5 cr a day. Restricting the universe is therefore a real lever - but
measuring it is where the trap is.

THE LOOK-AHEAD, WHICH IS THE WHOLE REASON THIS SCRIPT HAS THREE ARMS.
`data/nifty500.csv` is TODAY'S membership. Applying it to 2016 means the book
may only ever hold companies that GREW INTO the index: selection on the
outcome, and it flatters hard. That is worse than the plain survivorship bias
`factor/universe.py` documents, because survivorship removes losers while this
also pre-selects winners.

So `size500` is run beside it. The Nifty 500 selects on free-float market cap,
so the control ranks by size directly - today's implied share count applied to
each date's own close - and its error is share ISSUANCE, which is mechanical
and does not know whether a company later joined an index.

**`nifty500` minus `size500` is an estimate of the look-ahead.** If they agree,
the restricted number is usable. If `nifty500` is much the better, that gap is
hindsight and `size500` is what to plan against.

COMMITTED BEFORE THE RUN: F1 measured the liquid band at +15.5% against +18.8%
for all, so a 1-3pp cost is the honest prior. If `nifty500` comes back BETTER
than `all` on both windows, suspect the look-ahead before celebrating.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_algo.config import DEFAULT                        # noqa: E402
from nifty_algo.factor import backtest as fb                 # noqa: E402
from nifty_algo.factor import restriction as restr           # noqa: E402
from nifty_algo.factor import sleeve as sl                   # noqa: E402
from nifty_algo.swing.costs_equity import DEFAULT_EQUITY_COSTS  # noqa: E402
from run_f3_screened import verdict_map                      # noqa: E402

ARMS = ("all", "nifty500", "size500", "nifty100", "size100")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capital", type=float, default=500_000.0)
    p.add_argument("--slippage", type=float, default=0.0025)
    p.add_argument("--shortlist", type=int, default=60)
    p.add_argument("--cache", default="")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--no-halal", action="store_true",
                   help="measure the unscreened book instead")
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    if args.cache:
        cfg.factor.cache_name = args.cache
    fcfg = cfg.factor

    bars, _bench = sl.load_bars(cfg)
    universe = fb.FactorUniverse(bars, adv_window=fcfg.adv_window)
    print(f"{len(bars):,} symbols", flush=True)

    halal_ok = None
    if not args.no_halal:
        verdicts = verdict_map(cfg)
        print(f"halal screen ON - {len(verdicts):,} names have verdicts",
              flush=True)

        def halal_ok(symbol: str) -> bool:                   # noqa: F811
            v = verdicts.get(symbol)
            return bool(v is not None and v.eligible)

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    results, notes = {}, {}
    started = time.time()
    for key in ARMS:
        fn, note = restr.resolver(cfg, key, bars, universe)
        notes[key] = note
        results[key] = fb.run(
            bars, args.capital, top_n=fcfg.top_n, band=fcfg.band,
            formation=fcfg.formation, hold_months=fcfg.hold_months,
            min_price=fcfg.min_price, min_turnover=fcfg.min_turnover_inr,
            min_history=fcfg.min_history_sessions,
            costs=DEFAULT_EQUITY_COSTS, universe=universe,
            slippage_pct=args.slippage, start=start, end=end,
            halal_ok=halal_ok, halal_shortlist=args.shortlist,
            restrict_fn=fn)
        print(f"  [{key}] {note}", flush=True)
    print(f"\nran {len(ARMS)} arms in {time.time() - started:.0f}s\n")

    marks = [d for d, _ in results["all"].equity]
    screen = "screened" if halal_ok else "UNSCREENED"
    print(f"F4 - universe restriction, {screen}, "
          f"{marks[0]} -> {marks[-1]}, {len(marks)} marks\n")
    print(f"  {'universe':<12}{'CAGR':>9}{'maxDD':>8}{'recov':>7}{'turn':>7}"
          f"{'trades':>8}{'avg names':>11}{'eligible':>10}")
    for key in ARMS:
        r = results[key]
        held = [len(n) for _d, n in r.holdings_log if n]
        elig = [n for n in r.universe_size if n]
        print(f"  {key:<12}{r.cagr(args.capital):>+9.2%}"
              f"{r.max_drawdown():>8.1%}"
              f"{r.longest_drawdown_months():>6.0f}m{r.avg_turnover():>7.0%}"
              f"{r.trades:>8,}"
              f"{(sum(held) / len(held) if held else 0):>11.1f}"
              f"{(sum(elig) / len(elig) if elig else 0):>10.0f}")

    base = results["all"].cagr(args.capital)
    print()
    print("  EACH PUBLISHED INDEX AGAINST ITS OWN SIZE-RANKED CONTROL:")
    print(f"    {'pair':<24}{'naive':>9}{'control':>9}{'look-ahead':>13}"
          f"{'honest cost vs all':>21}")
    for index_key, control_key in (("nifty500", "size500"),
                                   ("nifty100", "size100")):
        if index_key not in results or control_key not in results:
            continue
        naive = results[index_key].cagr(args.capital)
        control = results[control_key].cagr(args.capital)
        print(f"    {index_key + ' vs ' + control_key:<24}"
              f"{naive:>+9.2%}{control:>+9.2%}"
              f"{(naive - control) * 100:>+12.2f}pp"
              f"{(control - base) * 100:>+20.2f}pp")
    print()
    print("  `nifty500` uses TODAY'S membership, so it may only hold companies "
          "that grew INTO the")
    print("  index - selection on the outcome. `size500` ranks by "
          "point-in-time size instead, so")
    print("  its error is share issuance rather than hindsight. The gap "
          "between them is how much")
    print("  of the restricted result is illusion. Plan against `size500`, "
          "not against `nifty500`.")
    print()
    print("  Both still inherit the standing distortions: the symbol list is "
          "TODAY'S listed set")
    print("  (survivorship, one way), and with the screen on the balance "
          "sheets are today's too.")
    print()
    print("  AND THE CONTROL IS NOT CLEAN EITHER, just cleaner. `size500` "
          "prices a 2008 rebalance")
    print("  with a share count measured in 2026, so a company that has "
          "diluted heavily has its past")
    print("  size overstated - and growth names dilute most. The look-ahead "
          "figure above is a LOWER")
    print("  bound on the real one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
