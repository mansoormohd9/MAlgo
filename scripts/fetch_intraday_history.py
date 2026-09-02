"""
Download 5-minute history for a whole equity universe from Kite Connect.

    python scripts/fetch_intraday_history.py --market india --years 3
    python scripts/fetch_intraday_history.py --years 1 --interval 15
    python scripts/fetch_intraday_history.py --symbols RELIANCE,TCS

Writes data/cache/intraday_5m_india.parquet in the shape
`intraday_equity.bars` reads: one tidy long frame with a `symbol` column, the
benchmark stored under `__BENCHMARK__:^NSEI`.

RESUMABLE, and modelled on fetch_history.py rather than on
fetch_swing_history.py. That distinction matters: the swing fetcher passes
`force_refresh=True` and re-downloads its whole window every run, which is
tolerable for 100 symbols of daily bars and is not tolerable for 100 symbols
of five-minute bars. Here each symbol computes at most two missing ranges -
older than what is held, and newer - and re-fetches only those. The last held
day is always re-fetched because it may have been captured mid-session.

WHY THIS IS POSSIBLE AT ALL: NSE equity instrument tokens are PERMANENT, the
same property that makes deep NIFTY index history available and expired
option history impossible. See the note at the top of data/kite_feed.py.

COST: about 800 requests for 100 symbols over 3 years (Kite serves at most
100 days of 5-minute bars per request), which at the 3 req/sec historical cap
is roughly five minutes and ~200 MB of parquet.
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
from nifty_algo.data.kite_feed import NIFTY_50_TOKEN                   # noqa: E402
from nifty_algo.intraday_equity import bars as bars_mod                # noqa: E402
from nifty_algo.swing import markets as markets_mod                    # noqa: E402
from nifty_algo.swing.universe import load_universe                    # noqa: E402

#: Kite's own archive does not reach back indefinitely, and requests before it
#: return empty rather than erroring.
EARLIEST = date(2015, 2, 2)

#: Flush the parquet every this many symbols. Five writes over a hundred-name
#: run - a few seconds each - against losing the whole run to one dropped
#: connection.
CHECKPOINT_EVERY = 20


def _ranges(have: pd.DataFrame, start: date, end: date):
    """
    The at-most-two date ranges still missing. The resumable core.

    Copied in spirit from `fetch_history.py`: everything already held is left
    alone, and only the head and tail gaps are requested. The last held day is
    re-fetched because a run during market hours captures a partial session.
    """
    if have is None or have.empty:
        return [(start, end)]
    held_start = have.index[0].date()
    held_end = have.index[-1].date()
    out = []
    if start < held_start:
        out.append((start, held_start - timedelta(days=1)))
    if end >= held_end:
        out.append((held_end, end))          # re-fetch the last, possibly partial
    return out


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = DEFAULT
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default=cfg.intraday_equity.market,
                   choices=markets_mod.keys(cfg))
    p.add_argument("--years", type=float, default=cfg.intraday_equity.history_years)
    p.add_argument("--interval", type=int,
                   default=cfg.intraday_equity.bar_interval_minutes)
    p.add_argument("--symbols", default="",
                   help="comma-separated subset, for a quick top-up")
    p.add_argument("--full", action="store_true",
                   help="ignore what is cached and refetch everything")
    args = p.parse_args(argv)

    market = markets_mod.get(cfg, args.market)
    if not market.is_home:
        print(f"Kite serves NSE only; {market.label} is not fetchable here.")
        return 2

    end = date.today()
    start = max(EARLIEST, end - timedelta(days=int(args.years * 365)))

    stocks = load_universe(market.universe_csv)
    wanted = [s.symbol for s in stocks]
    if args.symbols:
        subset = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
        wanted = [s for s in wanted if s in subset]
    if not wanted:
        print("no symbols selected")
        return 2

    # The universe file's `name` column, so an unresolved symbol can be
    # matched against company names rather than only against tickers. A
    # rename frequently keeps nothing of the old ticker (TATAMOTORS -> TMPV).
    names = {s.symbol: s.name for s in stocks}

    try:
        session = KiteSession()
        tokens = instr.resolve(wanted, session, names=names,
                               cache_dir=cfg.intraday_equity.cache_dir)
        if tokens.missing and not tokens.suggestions and tokens.from_cache:
            # A cache written before the `name` column was kept can resolve
            # but cannot suggest. Pay for exactly one re-download, and only
            # when there is something to diagnose.
            #
            # `from_cache` is load-bearing: without it a genuinely delisted
            # symbol - which will never produce a suggestion - re-downloads
            # the whole 10,000-row dump on every single run, forever.
            tokens = instr.resolve(wanted, session, names=names,
                                   cache_dir=cfg.intraday_equity.cache_dir,
                                   force_refresh=True)
    except NotAuthenticated as e:
        print(f"Not logged in: {e}\n"
              f"Run: python -m nifty_algo.broker.kite_login")
        return 2
    except instr.InstrumentsUnavailable as e:
        print(f"Could not resolve NSE instruments: {e}")
        return 1

    print(tokens.note())
    if tokens.missing:
        # NAMED, never a silent skip - a universe that quietly shrinks is a
        # scan that is quietly blind. And named WITH A LEAD, because the two
        # names this printed on its first real run were both renames and the
        # bare list gave nothing to act on.
        print(f"UNRESOLVED ({len(tokens.missing)}): {', '.join(tokens.missing)}")
        for line in tokens.suggestion_lines():
            print(line)
        print("  -> fix data/nifty100.csv; nothing is substituted automatically")

    existing = None if args.full else bars_mod.load_cached(
        market.cache_suffix, args.interval, wanted, market.benchmark_ticker,
        cache_dir=cfg.intraday_equity.cache_dir, run_gates=False)
    if existing is None and not args.full:
        cache_file = (Path(cfg.intraday_equity.cache_dir)
                      / bars_mod.cache_name(market.cache_suffix, args.interval))
        if cache_file.exists():
            # A cache that exists and is REJECTED refetches everything, which
            # looks exactly like resume not working. The usual cause is a
            # benchmark-key mismatch, which is the point of that key.
            print(f"{cache_file} exists but cannot satisfy this request "
                  f"(benchmark {market.benchmark_ticker}?) - refetching in full")

    out_bars: dict = dict(existing.bars) if existing else {}
    benchmark = existing.benchmark if existing else None

    jobs = [("__BENCHMARK__", NIFTY_50_TOKEN)] + [
        (s, tokens.tokens[s]) for s in wanted if s in tokens.tokens]

    #: Set when the timestamp convention check fails. Checkpointing means the
    #: old "REFUSING TO CACHE" at the very end would arrive after the data was
    #: already written, so the refusal has to be able to veto a checkpoint.
    refused = False

    def _checkpoint() -> None:
        """
        Write what has been fetched so far.

        Called every `CHECKPOINT_EVERY` symbols and again from a `finally`,
        because `save_cache` running only after the last of ~100 jobs means a
        dropped connection or an expired token at job 80 discards 80 symbols
        x 3 years of API calls. The script's docstring promises resumability;
        without this the promise only held across runs that already succeeded.
        """
        if refused or not out_bars:
            return
        bars_mod.save_cache(
            bars_mod.IntradayBarSet(
                bars=out_bars, benchmark=benchmark,
                benchmark_ticker=market.benchmark_ticker,
                interval_minutes=args.interval,
                missing=sorted(set(wanted) - set(out_bars))),
            market.cache_suffix, cache_dir=cfg.intraday_equity.cache_dir)

    fetched = 0
    checked_convention = False
    try:
        for n, (symbol, token) in enumerate(jobs, 1):
            have = benchmark if symbol == "__BENCHMARK__" else out_bars.get(symbol)
            todo = [(start, end)] if args.full else _ranges(have, start, end)
            chunks = []
            for lo, hi in todo:
                if lo > hi:
                    continue
                try:
                    df = bars_mod.fetch_symbol(
                        session, token, lo, hi, args.interval,
                        has_volume=(symbol != "__BENCHMARK__"))
                    if df is not None and not df.empty:
                        chunks.append(df)
                        fetched += 1
                except (FeedError, NotAuthenticated) as e:
                    print(f"\n  {symbol}: {e}")
                time.sleep(0.35)          # belt and braces over ThrottledKite

            if chunks:
                merged = pd.concat(([have] if have is not None else []) + chunks)
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                if symbol == "__BENCHMARK__":
                    benchmark = merged
                else:
                    out_bars[symbol] = merged
                    if not checked_convention:
                        # On the way IN, on the FIRST stock - see bars.py.
                        # Checked here rather than after the last job because
                        # a checkpoint has by then already written the file,
                        # and because a convention change should cost one
                        # symbol of fetching, not a hundred.
                        try:
                            bars_mod.assert_convention(merged, args.interval)
                        except bars_mod.ConventionError as e:
                            print(f"\nREFUSING TO CACHE: {e}")
                            refused = True
                            break
                        checked_convention = True

            # `benchmark` and `out_bars[symbol]` are DataFrames, so this must
            # never be written as `benchmark or []` - `or` calls
            # DataFrame.__bool__ and raises "truth value is ambiguous".
            held = benchmark if symbol == "__BENCHMARK__" else out_bars.get(symbol)
            print(f"  [{n}/{len(jobs)}] {symbol:<14} "
                  f"{0 if held is None else len(held):>7} bars", flush=True)

            if n % CHECKPOINT_EVERY == 0:
                _checkpoint()
    finally:
        # Ctrl-C and any unhandled exception still keep what was paid for.
        _checkpoint()

    if refused:
        return 1
    if not out_bars:
        print("nothing fetched")
        return 1

    if not checked_convention:
        # A fully up-to-date resume fetches no chunks at all, so the in-loop
        # check never fires and the gate would be skipped entirely on exactly
        # the runs that touch the most data. Check what is held instead.
        try:
            bars_mod.assert_convention(
                next(iter(out_bars.values())), args.interval)
        except bars_mod.ConventionError as e:
            print(f"CACHED BARS FAIL THE CONVENTION CHECK: {e}")
            return 1

    # The final flush already happened in the `finally` above; this is the
    # report, not the write.
    missing = sorted(set(wanted) - set(out_bars))
    sizes = sorted(len(d) for d in out_bars.values())
    sessions = sorted({t.date() for d in out_bars.values() for t in d.index})
    if not sessions:
        # Every frame came back empty. This is what a missing Kite Historical
        # Data subscription looks like, and it used to be an IndexError two
        # lines further down.
        print(f"\n{len(out_bars)} symbols returned NO BARS AT ALL. "
              f"Intraday history needs Kite's Historical Data subscription; "
              f"check that before assuming a quiet market.")
        return 1

    print(f"\ncached {len(out_bars)} symbols, {len(sessions)} sessions "
          f"({sessions[0]} -> {sessions[-1]})")
    print(f"bars per symbol: min {sizes[0]:,}  median {sizes[len(sizes)//2]:,}  "
          f"max {sizes[-1]:,}")
    print(f"benchmark {market.benchmark_ticker}: "
          f"{0 if benchmark is None else len(benchmark):,} bars")
    if missing:
        print(f"NO DATA ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
