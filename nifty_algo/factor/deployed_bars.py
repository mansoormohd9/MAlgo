"""
The slice of bars that ships with the repo, so the LIVE sleeve works deployed.

WHY THIS EXISTS. `data/cache/` is gitignored, so on Streamlit Community Cloud
the repo is cloned from GitHub without it and the Monthly sleeve reported

    No factor cache at /mount/src/malgo/data/cache/factor_daily_india.parquet

which was entirely correct: the file is 62 MB of market data and stays out of
git deliberately. The swing book survives the same clone because
`prices.load_prices` re-downloads on a miss; the factor loader refuses to
(`load_bars` is read-only by design), so it hard-failed instead.

WHAT SHIPS IS THE SMALLEST THING THAT SCANS. The live scan needs 253 sessions
to form 12-1 momentum, 253 for the volatility and 52-week figures, and 300 to
clear `min_history_sessions`. It needs `close` and `volume` and nothing else -
`FactorUniverse` reads only those two, and ATR is not allocated unless asked
for. So 400 sessions of close+volume is **6.5 MB** against 62 MB, and it was
verified to produce a byte-identical book: same 20 symbols in the same order,
same target quantities, same 472 eligible names, momentum equal to within
float32 rounding.

THE FULL CACHE ALWAYS WINS. `sleeve.load_bars` tries it first, so nothing
about running locally changes - the slice is a fallback for a machine that
does not have the real thing, never a replacement for it.

AND THE FALLBACK IS NOT IN `drawdown.load`. That function feeds F1-F5, and a
backtest that silently ran on 400 sessions would report a complete, plausible,
badly wrong CAGR - the exact failure this repo refuses everywhere else. The
backtests keep hard-failing when the full cache is absent; only the live path
degrades, and it says so on screen.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..paths import at_root

#: Committed, and NOT under `data/cache/` - that whole directory is ignored.
#: `.gitignore` carries an explicit negation for this one file, the same way it
#: does for the universe CSVs that are source rather than data.
SLICE_PATH = "data/sleeve_bars_india.parquet"

#: 400 > the 300 `min_history_sessions` wants and the 253 the signal needs,
#: with room so a name is not disqualified by the slice rather than by its own
#: listing date. Every extra 100 sessions is about 1.6 MB.
SLICE_SESSIONS = 400

#: The only columns `FactorUniverse` reads. `open`/`high`/`low` would double
#: the file to carry an ATR the live sleeve never computes.
SLICE_COLUMNS = ("close", "volume", "symbol")


@dataclass(frozen=True)
class SliceInfo:
    """What a loaded slice is, so a page can say it out loud."""
    path: Path
    sessions: int
    symbols: int
    first: date | None
    last: date | None

    def describe(self) -> str:
        return (f"deployed slice - {self.sessions} sessions of close/volume "
                f"for {self.symbols:,} symbols, ending {self.last}")


def build(frame: pd.DataFrame, sessions: int = SLICE_SESSIONS) -> pd.DataFrame:
    """
    The last `sessions` sessions, close+volume only, as float32.

    Takes the frame rather than a path so the fetch script can write the slice
    from what it just built without re-reading 62 MB off disk.
    """
    dates = sorted(frame.index.unique())
    if not dates:
        return frame.iloc[:0]
    cut = dates[-min(sessions, len(dates))]
    out = frame[frame.index >= cut][[c for c in SLICE_COLUMNS
                                     if c in frame.columns]].copy()
    for column in ("close", "volume"):
        if column in out:
            out[column] = out[column].astype("float32")
    return out.sort_index()


def write(frame: pd.DataFrame, path: str | Path | None = None,
          sessions: int = SLICE_SESSIONS) -> Path:
    """Write the slice and return where it went."""
    target = at_root(SLICE_PATH if path is None else path)
    target.parent.mkdir(parents=True, exist_ok=True)
    build(frame, sessions).to_parquet(target, compression="zstd")
    return target


def load(path: str | Path | None = None):
    """
    `(bars, benchmark, info)` from the committed slice, or `(None, None, None)`.

    Returns rather than raises when it is absent, because the caller has
    already failed to find the full cache and wants to report BOTH misses in
    one message rather than the second one only.

    `path=None` RESOLVES `SLICE_PATH` AT CALL TIME rather than binding it as a
    default argument. A default would be captured at import, so
    `monkeypatch.setattr(deployed_bars, "SLICE_PATH", ...)` would redirect
    nothing and a test aiming at a temporary file would silently read the real
    committed slice - which is exactly the trap `settings_store.DEFAULT_PATH`
    already records, and it caught two tests here before this signature did.
    """
    target = at_root(SLICE_PATH if path is None else path)
    if not target.exists():
        return None, None, None
    try:
        raw = pd.read_parquet(target)
    except Exception:                                      # pragma: no cover
        return None, None, None

    bars, bench = {}, None
    for symbol, group in raw.groupby("symbol"):
        sub = group.drop(columns=["symbol"]).sort_index()
        if str(symbol).startswith("__"):
            bench = sub
        else:
            bars[str(symbol)] = sub
    if not bars:
        return None, None, None

    dates = sorted(raw.index.unique())
    info = SliceInfo(
        path=target, sessions=len(dates), symbols=len(bars),
        first=dates[0].date() if dates else None,
        last=dates[-1].date() if dates else None)
    return bars, bench, info


def copy_fundamentals(cfg=None) -> Path | None:
    """
    Copy the live fundamentals cache to the committed one, or None if absent.

    A straight copy rather than a filtered slice: it is 1.4 MB, the names that
    reach the top 60 change every month, and a slice built from THIS month's
    shortlist would silently starve next month's. `fundamentals` reads the
    committed file only when the live cache is missing entirely.
    """
    from ..config import DEFAULT
    from ..swing import fundamentals as fund

    cfg = cfg or DEFAULT
    live = fund.cache_path(cfg)
    if not live.exists():
        return None
    target = fund.deployed_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(live.read_bytes())
    return target


def _main(argv: list[str] | None = None) -> int:            # pragma: no cover
    """
    Rebuild the slice from the full cache, without refetching anything.

    `fetch_factor_history.py` writes it automatically, so this is for the case
    where the full cache is already current and only the committed slice needs
    refreshing before a deploy.
    """
    from ..config import DEFAULT
    from . import drawdown as fd

    p = argparse.ArgumentParser(description=_main.__doc__)
    p.add_argument("--sessions", type=int, default=SLICE_SESSIONS)
    p.add_argument("--out", default=SLICE_PATH)
    args = p.parse_args(argv)

    source = at_root(Path(DEFAULT.factor.cache_dir) / DEFAULT.factor.cache_name)
    if not source.exists():
        print(f"No full cache at {source} - nothing to slice. Run "
              f"scripts/fetch_factor_history.py first.")
        return 1

    frame = pd.read_parquet(source)
    target = write(frame, args.out, args.sessions)
    bars, bench, info = load(args.out)
    print(f"wrote {target}  ({target.stat().st_size / 1e6:.1f} MB)")
    print(f"  {info.describe()}")
    print(f"  benchmark included: {'yes' if bench is not None else 'NO'}")
    print(f"  source: {source.name}, {len(frame.index.unique()):,} sessions")

    # THE SCREEN NEEDS BALANCE SHEETS OR IT REJECTS EVERYTHING. Measured on a
    # fresh clone with the halal screen on and no fundamentals cache: 60
    # shortlisted names, 60 rejected, 0 picks - "cannot verify" is a reject by
    # design, so absent data reads as a very strict screen.
    written = copy_fundamentals()
    if written is None:
        print("\n  no fundamentals cache to copy - the deployed screen will "
              "have to fetch, and rejects what it cannot verify")
    else:
        print(f"-> {written} ({written.stat().st_size / 1e6:.1f} MB)")

    print("\nCommit both - they are what make the deployed sleeve able to "
          "scan and screen.")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(_main())
