"""
Download India VIX daily history from Kite Connect.

    python scripts/fetch_vix.py

Writes data/india_vix_daily.csv. The backtester reads this to price each
simulated day at that day's actual implied volatility instead of a flat 14%
across four years - which spans everything from a dead-quiet 10 to a crisis 30.
A single flat IV does not merely add noise, it systematically misprices the
calm periods and the violent ones in opposite directions.
"""
from __future__ import annotations
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_algo.data.base import FeedError                       # noqa: E402
from nifty_algo.data.kite_feed import IndiaVixFeed               # noqa: E402
from nifty_algo.broker.kite_auth import KiteSession              # noqa: E402


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=float, default=6.0)
    p.add_argument("--out", default="data/india_vix_daily.csv")
    args = p.parse_args(argv)

    today = date.today()
    start = today - timedelta(days=int(args.years * 365))

    feed = IndiaVixFeed(session=KiteSession())
    print(f"  fetching India VIX daily {start} .. {today} ...")
    try:
        df = feed.fetch(start, today)
    except FeedError as e:
        print(f"\n  FAILED: {e}\n")
        return 1

    if df.empty:
        print("\n  No VIX data returned.\n")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Only the close matters for IV calibration; keep it narrow and obvious.
    series = df[["close"]].copy()
    series.index = pd.to_datetime(series.index).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.index.name = "date"
    series = series.rename(columns={"close": "vix"})
    series.to_csv(out)

    print(f"\n  wrote {len(series):,} days "
          f"({series.index[0].date()} .. {series.index[-1].date()}) -> {out}")
    print(f"  range {series['vix'].min():.2f} .. {series['vix'].max():.2f}, "
          f"median {series['vix'].median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
