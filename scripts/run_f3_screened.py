"""
F3. What does the halal screen cost the factor sleeve?

    python scripts/run_f3_screened.py
    python scripts/run_f3_screened.py --cache factor_daily_india_21y.parquet \
        --start 2005-09-12 --end 2016-08-31

THE BACKTEST NEVER HAD A SCREEN. `factor/backtest.py` CAVEATS says so outright:
no point-in-time fundamentals exist, so the screen cannot be replayed and every
recorded F1/F2 number describes a LARGER universe than a live book may trade.
The console screens. Those are two different books and the gap between them was
unmeasured until this ran.

WHAT THIS IS AND IS NOT. It is not a more honest backtest - it is a DIFFERENT
question, answered under a distortion the swing backtest already prints above
every result: today's balance sheets applied to all of history. A company that
failed the debt test in 2012 and passes now is held here in 2012. That runs in
an unknown direction and it cannot be fixed with free data. What it does bound
is how much the screen moves the answer at all: if the screened and unscreened
books land on top of each other, the console's numbers are usable; if they do
not, the console must quote the screened one.

The verdicts are built ONCE from the fundamentals cache and reused for every
rebalance, which is what makes this affordable - and is also exactly why the
point-in-time distortion exists. Both facts have the same cause.
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
from nifty_algo.factor import sleeve as sl                   # noqa: E402
from nifty_algo.swing import fundamentals as fund_mod        # noqa: E402
from nifty_algo.swing import halal as halal_mod              # noqa: E402
from nifty_algo.swing import markets as markets_mod          # noqa: E402
from nifty_algo.swing.costs_equity import DEFAULT_EQUITY_COSTS  # noqa: E402


def verdict_map(cfg) -> dict:
    """
    `{symbol: HalalVerdict}` from the CACHE ONLY - no network, by construction.

    READS THE CACHE FILE RATHER THAN CALLING `load_fundamentals`. That function
    fetches whatever is missing or stale, so handing it the whole universe
    would fire ~1,500 requests as a side effect of a measurement - the same
    trap `research.holdings_prices` avoids by asking only for symbols already
    cached, and the same one `run_s1_swing_null.py` avoids by reading its
    parquet directly. A measurement script must not be able to start an
    hour-long download.

    A name with no cached fundamentals gets no verdict, and `halal_ok` treats
    that as a REJECT rather than a pass - the same rule the live screen
    follows, and the conservative direction: an unscreenable name is one you
    could not have bought with confidence.
    """
    market = markets_mod.factor_market(cfg)
    path = Path(cfg.swing.cache_dir) / fund_mod.CACHE_NAME
    cached = fund_mod._read_cache(path)
    overrides, _ = halal_mod.load_overrides(cfg.swing.halal.overrides_csv)

    prefix = f"{market.key}:"
    out = {}
    for key, f in cached.items():
        # Keys are `{market}:{SYMBOL}`; the factor market writes its own
        # namespace, so a swing-book entry for the same ticker cannot be
        # mistaken for one of these.
        if not key.startswith(prefix):
            continue
        symbol = key[len(prefix):]
        out[symbol] = halal_mod.screen(sl.stock_for(symbol, market, f), f, cfg,
                                       overrides=overrides, market=market)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capital", type=float, default=500_000.0)
    p.add_argument("--slippage", type=float, default=0.0025)
    p.add_argument("--shortlist", type=int, default=60)
    p.add_argument("--cache", default="")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    if args.cache:
        cfg.factor.cache_name = args.cache
    fcfg = cfg.factor

    bars, bench = sl.load_bars(cfg)
    universe = fb.FactorUniverse(bars, adv_window=fcfg.adv_window)
    print(f"{len(bars):,} symbols", flush=True)

    verdicts = verdict_map(cfg)
    passing = sum(1 for v in verdicts.values() if v.eligible)
    print(f"{len(verdicts):,} names have cached fundamentals; "
          f"{passing:,} pass the screen "
          f"({passing / max(len(verdicts), 1):.0%} of those screenable)",
          flush=True)
    unver = sum(1 for v in verdicts.values()
                if getattr(v, "unverifiable", False))
    print(f"{len(bars) - len(verdicts):,} names have NO fundamentals cached "
          f"and are therefore REJECTED, not passed.")
    print(f"{unver:,} of {len(verdicts):,} "
          f"({unver / max(len(verdicts), 1):.0%}) could not be SCREENED AT "
          f"ALL - no classification, or missing")
    print("  balance-sheet lines. Those are excluded for a DATA reason "
          "wearing a Shariah reason's")
    print("  clothes, and the rate rises the further back you look: 6% of "
          "the 2016-2026 shortlist,")
    print("  26% of the 2005-2016 one. Read any screened result before ~2016 "
          "with that in front of it.\n", flush=True)

    def halal_ok(symbol: str) -> bool:
        v = verdicts.get(symbol)
        return bool(v is not None and v.eligible)

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    def run(screened: bool):
        return fb.run(
            bars, args.capital, top_n=fcfg.top_n, band=fcfg.band,
            formation=fcfg.formation, hold_months=fcfg.hold_months,
            min_price=fcfg.min_price, min_turnover=fcfg.min_turnover_inr,
            min_history=fcfg.min_history_sessions,
            costs=DEFAULT_EQUITY_COSTS, universe=universe,
            slippage_pct=args.slippage, start=start, end=end,
            halal_ok=halal_ok if screened else None,
            halal_shortlist=args.shortlist)

    started = time.time()
    plain = run(False)
    screened = run(True)
    print(f"ran both in {time.time() - started:.0f}s\n")

    marks = [d for d, _ in plain.equity]
    window = f"{marks[0]} -> {marks[-1]}, {len(marks)} marks"
    print(f"F3 - the halal screen's cost, {window}\n")
    print(f"  {'arm':<22}{'CAGR':>9}{'maxDD':>8}{'recov':>7}{'turn':>7}"
          f"{'trades':>8}{'avg names':>11}")
    for label, res in (("unscreened (as tested)", plain),
                       ("screened (as traded)", screened)):
        held = [len(n) for _d, n in res.holdings_log if n]
        print(f"  {label:<22}{res.cagr(args.capital):>+9.2%}"
              f"{res.max_drawdown():>8.1%}"
              f"{res.longest_drawdown_months():>6.0f}m"
              f"{res.avg_turnover():>7.0%}{res.trades:>8,}"
              f"{(sum(held) / len(held) if held else 0):>11.1f}")

    delta = screened.cagr(args.capital) - plain.cagr(args.capital)
    print(f"\n  the screen moves CAGR by {delta * 100:+.2f}pp and max drawdown "
          f"by {(abs(plain.max_drawdown()) - abs(screened.max_drawdown())) * 100:+.1f}pp")
    print(f"  it rejected {screened.screened_out:,} name-slots and left the "
          f"book short of {fcfg.top_n} on {screened.shortlist_short} of "
          f"{screened.rebalances} rebalances")
    print("\n  POINT-IN-TIME DISTORTION, stated with the result: the balance "
          "sheets are TODAY'S,\n  applied to every rebalance. A company that "
          "failed the debt test in 2012 and passes\n  now is held here in "
          "2012. This bounds how much the screen MOVES the answer; it is\n  "
          "not a screened history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
