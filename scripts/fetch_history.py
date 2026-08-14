"""
Download real NIFTY 50 intraday history from Kite Connect.

    python scripts/fetch_history.py                  # ~4 years of 5-minute bars
    python scripts/fetch_history.py --years 6
    python scripts/fetch_history.py --interval 1     # 1-minute (much slower)

Writes data/nifty_5m.csv in the schema CsvFeed expects. Resumable: existing
rows are kept and only the missing date range is requested, so an interrupted
run costs you nothing and a daily top-up is one cheap call.

Index tokens never expire, which is why this works at all - the same trick is
NOT available for options or for intraday futures. See the note at the top of
nifty_algo/data/kite_feed.py.
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

from nifty_algo.data.base import FeedError                       # noqa: E402
from nifty_algo.data.kite_feed import KiteFeed, NIFTY_50_TOKEN   # noqa: E402
from nifty_algo.broker.kite_auth import KiteSession              # noqa: E402

# Kite's own historical archive does not reach back indefinitely, and requests
# for dates before it return empty rather than erroring. Nothing here depends
# on the exact boundary; it just avoids years of pointless empty chunks.
EARLIEST = date(2015, 2, 2)


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()


def write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_index().copy()
    out.index.name = "timestamp"
    out.to_csv(path)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=float, default=4.0,
                   help="how far back to fetch (default 4)")
    p.add_argument("--interval", type=int, default=5,
                   help="bar interval in minutes (default 5)")
    p.add_argument("--out", default="data/nifty_5m.csv")
    p.add_argument("--token", type=int, default=NIFTY_50_TOKEN,
                   help="instrument token (default NIFTY 50)")
    p.add_argument("--full", action="store_true",
                   help="ignore existing rows and refetch the whole range")
    args = p.parse_args(argv)

    out = Path(args.out)
    today = date.today()
    want_start = max(today - timedelta(days=int(args.years * 365)), EARLIEST)

    existing = pd.DataFrame() if args.full else load_existing(out)
    if not existing.empty:
        have_from, have_to = existing.index[0].date(), existing.index[-1].date()
        print(f"  existing: {len(existing):,} bars, {have_from} .. {have_to}")
    else:
        have_from = have_to = None

    feed = KiteFeed(session=KiteSession(), instrument_token=args.token,
                    interval_minutes=args.interval)

    # Two ranges at most: older than what we have, and newer than what we have.
    ranges: list[tuple[date, date]] = []
    if have_from is None:
        ranges.append((want_start, today))
    else:
        if want_start < have_from:
            ranges.append((want_start, have_from - timedelta(days=1)))
        if have_to < today:
            ranges.append((have_to, today))       # re-fetch the last day; it may be partial

    if not ranges:
        print("  nothing to fetch - already covers the requested range.")
        return 0

    frames = [existing] if not existing.empty else []
    for start, end in ranges:
        print(f"  fetching {start} .. {end} ({args.interval}m) ...")
        try:
            chunk = feed.fetch(start, end)
        except FeedError as e:
            print(f"\n  FAILED: {e}\n")
            return 1
        print(f"    got {len(chunk):,} bars")
        if not chunk.empty:
            frames.append(chunk)
        time.sleep(0.4)          # stay well under Kite's 3 req/sec limit

    if not frames:
        print("\n  No data returned. Check that the market has traded in this "
              "range and that your Kite subscription includes historical data.\n")
        return 1

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    write(out, df)

    sessions = len({d for d in df.index.date})
    print(f"\n  wrote {len(df):,} bars across {sessions:,} sessions "
          f"({df.index[0].date()} .. {df.index[-1].date()}) -> {out}")
    if not (df["volume"] > 0).any():
        print("  note: volume is 0 throughout - this is an INDEX. The "
              "participation gates fall back to range expansion "
              "(see signals.has_traded_volume).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
