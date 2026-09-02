"""
5-minute bars for a universe of stocks: fetch, cache, and the integrity gates.

`swing/prices.py` is the model for the CACHE SHAPE and cannot be the model for
anything else - it is `interval="1d"` by explicit argument, and its own
docstring says why (Yahoo's intraday history is short, delayed and revised).
This book needs Kite, whose equity instrument tokens are permanent, which is
the same property that makes deep index history possible.

WHAT IS COPIED FROM `swing/prices.py`, DELIBERATELY

One tidy long parquet with a `symbol` column, not a file per symbol. The
benchmark stored under a key that CARRIES ITS OWN TICKER
(`__BENCHMARK__:^NSEI`), and a cache accepted only when it holds everything
asked for INCLUDING that key. That second rule is not defensive programming,
it is a bug that already happened once: with a fixed `__BENCHMARK__` key, a
US scan followed by an India scan found the key present and computed India's
relative strength against the S&P 500, with no error anywhere.

WHAT IS DELIBERATELY DIFFERENT

The filename carries the INTERVAL: `intraday_5m_india.parquet`.
`daily_prices_{market}.parquet` encodes no interval, so reusing that scheme
would let a 5-minute cache satisfy a request for daily bars and vice versa.

THE THREE INTEGRITY GATES, AND WHY THEY LIVE HERE

Each one produces a plausible wrong answer rather than an exception, so each
has to be caught at the door rather than found later in a result.

1. BAR TIMESTAMP CONVENTION. Kite stamps a 5-minute bar at its OPEN. Some
   vendors stamp the close. The difference is not cosmetic: gating an entry
   window or a 15:10 square-off on a bar stamped 15:10 under the open
   convention means acting on the bar covering 15:10-15:15, five minutes
   late, with the extra drift landing entirely in the result. `assert_convention`
   checks the first and last stamps of a session against the exchange clock,
   and every consumer in this package gates on BAR-CLOSE time.

2. CORPORATE ACTIONS. Kite's historical bars are UNADJUSTED. On an ex-split
   date the prior close is off by the split ratio, so `gap_metrics` sees a
   400% gap and a long-only book buys a "reclaim" of a move that never
   happened. There is no split feed here to consult, so the gate is
   statistical: an overnight move beyond `max_overnight_atr` ATRs is treated
   as a data artefact and the session is dropped. That will occasionally drop
   a real news gap, which is the safe direction - a dropped session costs
   opportunity, a phantom 400% gap costs a fabricated result.

3. SHORT AND HALTED SESSIONS. A 40-bar session shifts every positional index
   and truncates the opening range. Bars are addressed BY TIMESTAMP rather
   than by position everywhere downstream, and a session holding fewer than
   `min_session_bars` is dropped rather than traded.

Every drop is recorded on `IntradayBarSet.dropped` with a reason. A session
that vanished silently is indistinguishable from a session where nothing
fired.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.base import DataFeed, FeedError

#: NSE continuous session. Used only to validate the timestamp convention and
#: to size a full session - never to synthesise a bar.
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

#: 09:15 -> 15:30 is 375 minutes; 75 bars of 5 minutes.
FULL_SESSION_BARS = {1: 375, 3: 125, 5: 75, 10: 38, 15: 25, 30: 13, 60: 7}

BENCHMARK_PREFIX = "__BENCHMARK__"

#: Reasons a session is dropped. Spelled once so the ledger and the tests
#: cannot disagree about the string.
DROP_SHORT = "short_session"
DROP_SPLIT = "suspected_corporate_action"
DROP_NO_PRIOR = "no_prior_session"


def cache_name(market_key: str, interval_minutes: int) -> str:
    """Interval in the filename - see the module docstring."""
    return f"intraday_{interval_minutes}m_{market_key}.parquet"


def benchmark_key(ticker: str) -> str:
    """
    The benchmark's row-group key, carrying its ticker.

    A cache built against one benchmark must not be able to satisfy a request
    for another. See the module docstring.
    """
    return f"{BENCHMARK_PREFIX}:{ticker}"


@dataclass
class IntradayBarSet:
    """
    Sessions by symbol, plus an explicit account of everything that is not
    here. `missing` and `dropped` are reporting channels, not error channels.
    """
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark: pd.DataFrame | None = None
    benchmark_ticker: str = ""
    interval_minutes: int = 5
    missing: list[str] = field(default_factory=list)
    #: (symbol, session date, reason)
    dropped: list[tuple[str, date, str]] = field(default_factory=list)
    fetched_at: datetime | None = None
    from_cache: bool = False

    def sessions(self, symbol: str) -> list[date]:
        df = self.bars.get(symbol)
        if df is None or df.empty:
            return []
        return sorted({ts.date() for ts in df.index})

    def all_sessions(self) -> list[date]:
        """
        Every session any symbol has, sorted.

        Split out because a backtest sweep recomputes this hundreds of times
        otherwise - the same reason `swing/backtest.all_sessions` exists.
        """
        days: set[date] = set()
        for df in self.bars.values():
            days.update(ts.date() for ts in df.index)
        return sorted(days)

    def session(self, symbol: str, day: date) -> pd.DataFrame | None:
        """
        One symbol's bars for one day.

        THIS IS THE ONLY WAY THIS PACKAGE SHOULD GET A FRAME FOR A STRATEGY.
        It returns today's bars and nothing else, matching `engine.py:165`
        exactly - prior-day facts travel through `Context.prev_day_*`, never
        by prepending bars. Prepending would break `signals.opening_range`,
        which reads the first bars OF THE FRAME with no day-awareness, and it
        would silently return yesterday's opening range on every session.
        """
        df = self.bars.get(symbol)
        if df is None or df.empty:
            return None
        out = df[df.index.date == day]
        return out if not out.empty else None

    def note(self) -> str:
        src = "cache" if self.from_cache else "Kite"
        line = (f"{len(self.bars)} symbols, {self.interval_minutes}m bars "
                f"from {src}")
        if self.missing:
            shown = ", ".join(self.missing[:8])
            more = f" +{len(self.missing) - 8}" if len(self.missing) > 8 else ""
            line += f"; NO DATA: {shown}{more}"
        if self.dropped:
            line += f"; {len(self.dropped)} sessions dropped by integrity gates"
        return line

    def drop_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, _, reason in self.dropped:
            out[reason] = out.get(reason, 0) + 1
        return out


# ---------------------------------------------------------------- gates


class ConventionError(FeedError):
    """The feed's bar timestamps are not the convention this book assumes."""


def assert_convention(df: pd.DataFrame, interval_minutes: int) -> None:
    """
    Assert bars are stamped at their OPEN, which is what Kite does.

    Checked rather than assumed because the consequence is silent. If bars
    were stamped at the close, the first stamp of a session would be
    09:15+interval and the last would be 15:30; under the open convention
    they are 09:15 and 15:30-interval. Both look entirely ordinary in a
    dataframe, and the difference is a five-minute free option on every exit.

    Raises rather than adjusting: a feed that changed convention underneath
    this book is something a human needs to know about, not something to
    quietly compensate for.
    """
    if df.empty:
        return
    day = df.index[-1].date()
    session = df[df.index.date == day]
    if session.empty:
        return
    first, last = session.index[0].time(), session.index[-1].time()

    close_stamped = (
        datetime.combine(day, MARKET_CLOSE)
        - timedelta(minutes=0))
    open_stamped_last = (
        datetime.combine(day, MARKET_CLOSE)
        - timedelta(minutes=interval_minutes)).time()

    if first == MARKET_OPEN:
        return                                    # unambiguously open-stamped
    expected_close_first = (
        datetime.combine(day, MARKET_OPEN)
        + timedelta(minutes=interval_minutes)).time()
    if first == expected_close_first and last == MARKET_CLOSE:
        raise ConventionError(
            f"bars appear stamped at the CLOSE (first={first}, last={last}); "
            f"this book assumes Kite's open-stamped convention, and the "
            f"difference silently shifts every entry-window and square-off "
            f"gate by {interval_minutes} minutes")
    # A partial session (halt, or today mid-flight) is neither, and is fine.
    del close_stamped, open_stamped_last


def _session_gate(session: pd.DataFrame, prev_close: float | None,
                  interval_minutes: int, min_session_bars: int,
                  max_overnight_atr: float,
                  atr_estimate: float | None) -> str | None:
    """The reason to drop this session, or None to keep it."""
    if len(session) < min_session_bars:
        return DROP_SHORT
    if prev_close is None:
        # Gate 2 cannot be evaluated without a prior close, and the first
        # session of a history legitimately has none. Keep it - `gap_metrics`
        # consumers get `prev_day_close=0.0` and simply do not fire.
        return None
    if atr_estimate and atr_estimate > 0 and prev_close > 0:
        overnight = abs(float(session["open"].iloc[0]) - prev_close)
        if overnight > max_overnight_atr * atr_estimate:
            return DROP_SPLIT
    return None


def apply_gates(df: pd.DataFrame, symbol: str, interval_minutes: int,
                min_session_bars: int, max_overnight_atr: float,
                dropped: list) -> pd.DataFrame:
    """
    Run the integrity gates over one symbol's whole history.

    Returns the surviving bars and appends every drop to `dropped` with its
    reason. The ATR used by the corporate-action gate is a DAILY one built
    from session ranges, deliberately - a 5-minute ATR would call every
    ordinary overnight gap a split.
    """
    if df.empty:
        return df

    days = sorted({ts.date() for ts in df.index})
    daily = []
    for day in days:
        s = df[df.index.date == day]
        daily.append((day, float(s["high"].max()), float(s["low"].min()),
                      float(s["close"].iloc[-1]), len(s)))
    frame = pd.DataFrame(daily, columns=["day", "high", "low", "close", "bars"])

    # Wilder-style daily ATR over session ranges. Only ever used as a scale
    # for "is this overnight move absurd", never as a signal.
    prev_close = frame["close"].shift(1)
    tr = np.maximum(
        frame["high"] - frame["low"],
        np.maximum((frame["high"] - prev_close).abs(),
                   (frame["low"] - prev_close).abs()))
    atr = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=5).mean()

    keep: list[date] = []
    for i, row in frame.iterrows():
        day = row["day"]
        session = df[df.index.date == day]
        reason = _session_gate(
            session,
            None if i == 0 else float(prev_close.iloc[i]),
            interval_minutes, min_session_bars, max_overnight_atr,
            None if pd.isna(atr.iloc[i]) else float(atr.iloc[i]))
        if reason:
            dropped.append((symbol, day, reason))
        else:
            keep.append(day)

    if not keep:
        return df.iloc[0:0]
    mask = pd.Series([ts.date() in set(keep) for ts in df.index], index=df.index)
    return df[mask]


# ---------------------------------------------------------------- cache


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        # A torn parquet costs a re-download, never a crash.
        return None


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    except Exception:
        pass


def _unpack(raw: pd.DataFrame, wanted: set[str], bench_key: str
            ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    bars: dict[str, pd.DataFrame] = {}
    benchmark = None
    for symbol, group in raw.groupby("symbol"):
        g = group.drop(columns=["symbol"]).sort_index()
        if symbol == bench_key:
            benchmark = g
        elif symbol in wanted:
            bars[str(symbol)] = g
    return bars, benchmark


def _pack(bars: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None,
          bench_key: str) -> pd.DataFrame:
    parts = []
    for symbol, df in bars.items():
        p = df.copy()
        p["symbol"] = symbol
        parts.append(p)
    if benchmark is not None and not benchmark.empty:
        p = benchmark.copy()
        p["symbol"] = bench_key
        parts.append(p)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


def load_cached(market_key: str, interval_minutes: int, symbols,
                benchmark_ticker: str, cache_dir: str = "data/cache",
                min_session_bars: int = 60,
                max_overnight_atr: float = 4.0,
                run_gates: bool = True) -> IntradayBarSet | None:
    """
    The cached bars, or None if the cache cannot satisfy this request.

    "Cannot satisfy" includes the benchmark: the key carries its ticker, so a
    cache built for one index is structurally unable to answer for another.
    """
    path = Path(cache_dir) / cache_name(market_key, interval_minutes)
    raw = _read_cache(path)
    if raw is None or raw.empty or "symbol" not in raw.columns:
        return None

    wanted = {str(s).upper() for s in symbols}
    bench_key = benchmark_key(benchmark_ticker)
    have = set(raw["symbol"].unique())
    if bench_key not in have:
        return None

    bars, benchmark = _unpack(raw, wanted, bench_key)

    out = IntradayBarSet(
        benchmark=benchmark, benchmark_ticker=benchmark_ticker,
        interval_minutes=interval_minutes, from_cache=True,
        missing=sorted(wanted - set(bars)))

    if run_gates:
        for symbol, df in bars.items():
            bars[symbol] = apply_gates(df, symbol, interval_minutes,
                                       min_session_bars, max_overnight_atr,
                                       out.dropped)
        bars = {s: d for s, d in bars.items() if not d.empty}
    out.bars = bars
    return out


def save_cache(barset: IntradayBarSet, market_key: str, cache_dir: str = "data/cache") -> None:
    path = Path(cache_dir) / cache_name(market_key, barset.interval_minutes)
    frame = _pack(barset.bars, barset.benchmark,
                  benchmark_key(barset.benchmark_ticker))
    if not frame.empty:
        _write_cache(path, frame)


# ---------------------------------------------------------------- fetch


def fetch_symbol(session, token: int, start: date, end: date,
                 interval_minutes: int = 5,
                 has_volume: bool = True) -> pd.DataFrame:
    """
    One symbol's bars from Kite, chunked to the API's per-request window.

    Thin wrapper over `KiteFeed.fetch`, which already handles the chunking
    and returns an empty frame rather than raising on a quiet range. Kept as
    a function so the live path and the history script share one call site.
    """
    from ..data.kite_feed import KiteFeed
    feed = KiteFeed(session=session, instrument_token=int(token),
                    interval_minutes=interval_minutes, has_volume=has_volume)
    return feed.fetch(start, end)


def fetch_today(symbols_to_tokens: dict[str, int], session,
                day: date | None = None, interval_minutes: int = 5,
                pause: float = 0.35, progress=None) -> dict[str, pd.DataFrame]:
    """
    Today's bars for a handful of symbols. The LIVE path's fetch.

    Deliberately the same `KiteFeed.fetch` the history script uses, rather
    than a second bar-construction path built from ticks or quotes. Two ways
    of building a bar is two ways for them to disagree, and this book has no
    appetite for that. A websocket feed would cut latency and is the obvious
    later optimisation - but it is that second path, so it is not here.

    Sized for the RS-prefiltered watchlist (~20 names), not the full
    universe: at Kite's 3 requests/second historical cap that is about seven
    seconds, comfortably inside a 5-minute bar.
    """
    target = day or date.today()
    out: dict[str, pd.DataFrame] = {}
    for i, (symbol, token) in enumerate(sorted(symbols_to_tokens.items()), 1):
        try:
            df = fetch_symbol(session, token, target, target, interval_minutes)
            if df is not None and not df.empty:
                out[symbol] = df
        except Exception:
            # One unreachable name must not cost the whole bar. It simply has
            # no data this bar and the scanner records it as such.
            pass
        if progress:
            progress(i, len(symbols_to_tokens), symbol)
        if pause:
            _time.sleep(pause)
    return out


__all__ = [
    "IntradayBarSet", "ConventionError",
    "cache_name", "benchmark_key", "assert_convention", "apply_gates",
    "load_cached", "save_cache", "fetch_symbol", "fetch_today",
    "FULL_SESSION_BARS", "MARKET_OPEN", "MARKET_CLOSE",
    "DROP_SHORT", "DROP_SPLIT", "DROP_NO_PRIOR",
]
