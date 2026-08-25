"""
Pull deep daily history for a market, so the swing backtest has something to
walk.

WHY THIS IS SEPARATE FROM THE SCAN. `SwingConfig.history_days` is 400 - about
a trading year plus the 200-day warm-up - which is exactly right for a scan
that only ever reads the last bar, and nowhere near enough for a backtest. The
scan's 12-hour cache would also re-download the deep history every time it
went stale, which is minutes of waiting for data the scan does not use.

So this writes the SAME per-market parquet the scan reads, just with more of
it. A longer cache costs the scan nothing but a little memory: `setup.detect`
reads the tail, and the relative-strength and 52-week windows are fixed
lookbacks. Running this once means the backtest in the UI starts instantly
instead of blocking on a hundred downloads.

    python scripts/fetch_swing_history.py --market india --years 6

Yahoo is the source, as it is for the scan. It is delayed and revised, which
does not matter for daily closes on a multi-day horizon - and it is the only
free source that serves this much history for a hundred symbols at once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_algo.config import DEFAULT as cfg          # noqa: E402
from nifty_algo.swing import markets as markets_mod   # noqa: E402
from nifty_algo.swing import prices as prices_mod     # noqa: E402
from nifty_algo.swing.universe import load_universe   # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--market", default=cfg.swing.default_market,
                        choices=markets_mod.keys(cfg))
    parser.add_argument("--years", type=float, default=6.0,
                        help="how far back to reach (default 6)")
    args = parser.parse_args(argv)

    market = markets_mod.get(cfg, args.market)
    try:
        stocks = load_universe(market.universe_csv)
    except FileNotFoundError:
        print(f"{market.universe_csv} is missing. It is SOURCE, not data - "
              f"rebuild it with scripts/build_universe.py.")
        return 2

    tickers = {s.symbol: s.yf_ticker for s in stocks}
    # The 200-day warm-up has to come out of the window, or the first year of
    # any backtest has no usable indicators and quietly produces no trades.
    cfg.swing.history_days = int(args.years * 365) + 260

    print(f"{market.label}: {len(tickers)} symbols, ~{args.years:g} years "
          f"({cfg.swing.history_days} calendar days incl. warm-up)")

    def progress(done, total, label):
        print(f"  [{done}/{total}] {label}", end="\r")

    try:
        result = prices_mod.load_prices(tickers, cfg, market,
                                        force_refresh=True, progress=progress)
    except Exception as e:
        print(" " * 70, end="\r")
        print(f"Download failed: {e}")
        return 1

    print(" " * 70, end="\r")
    print(result.note())

    if result.missing:
        # Named, not counted. A ticker that stopped resolving is usually a
        # rename or a delisting, and both are things you want to know about
        # rather than a number to shrug at.
        print(f"\n{len(result.missing)} symbol(s) returned nothing:")
        print("  " + ", ".join(sorted(result.missing)))

    spans = [len(df) for df in result.bars.values()]
    if spans:
        print(f"\nBars per symbol: min {min(spans)}, median "
              f"{sorted(spans)[len(spans) // 2]}, max {max(spans)}")
        print("A symbol with far fewer bars than the rest listed recently - "
              "it contributes nothing to the early part of a backtest.")

    cache = Path(cfg.swing.cache_dir) / prices_mod.cache_name(market.cache_suffix)
    print(f"\nCached to {cache}")
    print(f"Now run:  python -m nifty_algo.swing.backtest "
          f"--market {market.key} --years {args.years:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
