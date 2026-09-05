"""
S1. Does the Indian swing book's SCORING beat chance?

Runs the book and `--seeds` random-scoring nulls in every out-of-sample
window, ranks the book inside that null distribution per fold, and prints the
pre-committed gate plus the holding-period table.

Reads `data/cache/daily_prices_india.parquet` DIRECTLY rather than through
`prices.load_prices`, which re-downloads `history_days` and rewrites the
parquet on a cache miss - and `fetch_swing_history.py` deliberately puts ten
years in that same file. A measurement script must not be able to truncate the
history every other experiment depends on.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Every script in this directory does this - it is run as a file, so the repo
# root is not on the path the way it is for `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_algo import experiment_core as ec
from nifty_algo.config import DEFAULT
from nifty_algo.swing import experiment as ex
from nifty_algo.swing import markets as markets_mod
from nifty_algo.swing.universe import load_universe

BENCH_PREFIX = "__BENCHMARK__"

#: The gate, committed before the run. Expressed as a p-value and a percentile
#: rather than as "12 of 16" so it does not depend on how many folds the data
#: happens to yield - the threshold is the one that was written down
#: (p <= 0.077, the exact binomial for 12/16), not a looser one.
MAX_SIGN_P = 0.077
MIN_MEAN_PERCENTILE = 0.65


def load_bars(path: Path):
    raw = pd.read_parquet(path)
    bars, bench = {}, None
    for symbol, g in raw.groupby("symbol"):
        frame = g.drop(columns=["symbol"]).sort_index()
        if str(symbol).startswith(BENCH_PREFIX):
            bench = frame
        else:
            bars[str(symbol)] = frame
    return bars, bench


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="S1 - the swing book vs chance.")
    p.add_argument("--capital", type=float, default=200_000.0)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--years", type=float, default=6.0)
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--test-months", type=int, default=2)
    p.add_argument("--variant", default="baseline")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    cfg.capital.swing_capital_inr = args.capital
    market = markets_mod.get(cfg, "india")

    path = Path(cfg.swing.cache_dir) / f"daily_prices_{market.cache_suffix}.parquet"
    if not path.exists():
        print(f"No price cache at {path}.\n"
              f"Run: python scripts/fetch_swing_history.py --market india --years 6")
        return 2

    bars, bench = load_bars(path)
    stocks = load_universe(market.universe_csv)
    known = {s.symbol for s in stocks}
    missing = sorted(known - set(bars))
    print(f"{len(bars)} symbols, benchmark "
          f"{'present' if bench is not None else 'MISSING'}", flush=True)
    if missing:
        # A universe that quietly shrank has already cost this repo a backtest
        # of unknown vintage - these lines are findings, not noise.
        print(f"  NOT IN CACHE ({len(missing)}): {', '.join(missing[:8])}",
              flush=True)
    if bench is None:
        print("Refusing to run: the regime gate fails closed and every "
              "result would read as 'the filter killed the book'.")
        return 2

    sessions = sorted({t for f in bars.values() for t in f.index})
    window_start = (sessions[-1] - pd.Timedelta(days=int(args.years * 365))).date()
    print(f"{len(sessions)} sessions {sessions[0].date()} -> {sessions[-1].date()}; "
          f"scoring from {window_start}", flush=True)
    print(f"capital Rs {args.capital:,.0f}; "
          f"risk Rs {cfg.capital.risk_inr(market.capital_pool):,.0f}/trade; "
          f"{args.seeds} null draws per fold", flush=True)

    started = time.time()

    def progress(n, total, label):
        print(f"  [{n}/{total}] {label}  ({time.time() - started:.0f}s)",
              flush=True)

    wf = ex.walk_forward_null(
        cfg, market, bars, bench, stocks=stocks, n_seeds=args.seeds,
        train_months=args.train_months, test_months=args.test_months,
        window_start=window_start, variant=ex.variant(args.variant),
        progress=progress)

    print(f"\ndone in {time.time() - started:.0f}s\n", flush=True)
    if not wf.folds:
        print("no out-of-sample folds - not enough history")
        return 1

    # unit="R": this book scores a window in EXPECTANCY, not in a period
    # return. Defaulting to "pct" printed +0.369R as "+36.90%".
    print(ec.walk_forward_report(wf, label="book", unit="R"))
    print()
    print("HOLDING PERIOD - the number that decides whether this is a "
          "few-day book")
    print(ex.hold_stats(wf))

    # THE SIGN TEST IS TWO-SIDED, so a book that loses 8 of 33 folds scores
    # p = 0.005 - "significantly different from chance", in the WRONG
    # direction. Testing the p alone reported that as a PASS. The direction
    # has to be asserted separately, and it is the half that matters.
    scored = len(wf.scored)
    beat_half = wf.wins > scored / 2.0
    sign_ok = beat_half and wf.sign_p <= MAX_SIGN_P
    pct_ok = wf.mean_percentile >= MIN_MEAN_PERCENTILE
    print()
    print("  THE GATE, committed before the run:")
    print(f"    1. wins > half AND p <= {MAX_SIGN_P:.3f} .... "
          f"{'PASS' if sign_ok else 'FAIL'}  "
          f"({wf.wins}/{scored} folds vs {scored / 2.0:.1f} expected, "
          f"two-sided p = {wf.sign_p:.3f}"
          f"{'' if beat_half else ' - LOSING, not winning'})")
    print(f"    2. mean percentile >= {MIN_MEAN_PERCENTILE:.0%} ......... "
          f"{'PASS' if pct_ok else 'FAIL'}  "
          f"({wf.mean_percentile:.0%})")
    print()
    if sign_ok and pct_ok:
        print("  BOTH PASS - the scoring beats chance out of sample. This is "
              "the first book in the repo to clear its own gate on the "
              "requested horizon.")
    elif wf.mean_percentile < 0.5:
        print("  THE NULL BEAT THE BOOK. Same outcome as the intraday equity "
              "book. Close it; do not examine variants.")
    else:
        print("  FAILED. The book stays where it was - unproven and not "
              "tradeable. The honest response is to stop adding variants, "
              "not to add another.")

    try:
        out = Path("data/experiments")
        out.mkdir(parents=True, exist_ok=True)
        rows = [{"fold": f.index, "start": f.start, "end": f.end,
                 "book_r": f.momentum, "null_median": f.null_median,
                 "percentile": f.percentile, "won": f.won,
                 "n_nulls": len(f.nulls),
                 "trades": len(getattr(f, "trades", []))}
                for f in wf.folds]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        f = out / f"s1_swing_null_{stamp}_seeds{args.seeds}.parquet"
        pd.DataFrame(rows).to_parquet(f, index=False)
        print(f"\nledger -> {f}")
    except Exception as e:
        print(f"\ncould not write the ledger ({e}) - the table above stands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
