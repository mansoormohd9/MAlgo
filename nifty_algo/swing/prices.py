"""
Daily OHLCV for the whole universe, in one batched download.

WHY DAILY AND NOT INTRADAY: this book holds for days. A 5-minute bar tells you
nothing about a swing that will take a week to play out, and Yahoo's intraday
NSE data is both delayed and revised after the fact. Daily closes are the one
thing Yahoo serves that is neither.

WHY BATCHED AND CACHED: a hundred sequential downloads takes minutes and gets
rate-limited. yfinance takes a list, so it is one call. The result then goes
to disk, because Streamlit re-runs the entire script on every interaction -
without a cache, scrolling the page would re-download the market.

The bars are split-adjusted (`auto_adjust=False` still back-adjusts splits,
it only leaves dividends alone). That matters: unadjusted history across a
1:5 split shows an 80% crash that no level-detection code can tell from a
real one.

TWO THINGS HERE ARE ABOUT MORE THAN ONE MARKET.

THE CACHE IS PER MARKET, AND THE BENCHMARK IS KEYED BY ITS TICKER. It used to
be one file with the benchmark under a fixed `__BENCHMARK__` key, and the cache
was accepted whenever it held every symbol asked for. Scan the US and then
India and that key is present, so India's relative strength would be computed
against the S&P 500 - a wrong answer with no error anywhere. Both halves of the
fix are needed: separate files stop the collision, and keying the benchmark by
ticker means a changed benchmark invalidates rather than silently persists.

PRICES ARE NORMALISED TO MAJOR UNITS AT INGEST. The LSE quotes most shares in
PENCE and yfinance passes that straight through, so SHEL.L arrives as 2500
meaning GBP 25.00. Dividing once, here, is the only safe place: every floor,
stop distance, turnover figure and position size downstream then reads the same
units, and nothing has to remember to divide again.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..data.base import DataFeed, FeedError

def cache_name(market_key: str) -> str:
    """One parquet per market. See the docstring - this is not tidiness."""
    return f"daily_prices_{market_key}.parquet"


def benchmark_key(ticker: str) -> str:
    """
    The benchmark's row label inside the cache.

    Carries the ticker so that a cache built against one index cannot satisfy
    a request for another.
    """
    return f"__BENCHMARK__:{ticker}"

#: yfinance accepts an arbitrary list but a very long one is a single point of
#: failure - one bad ticker can empty the whole response. Batches keep a
#: failure local to twenty-five names.
BATCH_SIZE = 25


@dataclass
class PriceSet:
    """Daily bars for the universe plus the benchmark, with provenance."""
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark: pd.DataFrame | None = None
    fetched_at: datetime | None = None
    last_bar_date: pd.Timestamp | None = None
    missing: list[str] = field(default_factory=list)
    from_cache: bool = False

    def note(self) -> str:
        bits = [f"{len(self.bars)} symbols"]
        if self.last_bar_date is not None:
            bits.append(f"bars to {self.last_bar_date:%d %b %Y} close")
        if self.fetched_at is not None:
            bits.append(f"downloaded {self.fetched_at:%d %b %H:%M}")
        bits.append("from cache" if self.from_cache else "fresh download")
        if self.missing:
            bits.append(f"{len(self.missing)} unavailable")
        return " · ".join(bits)


def load_prices(tickers: dict[str, str], cfg, market, benchmark: str | None = None,
                force_refresh: bool = False, divisors: dict[str, float] | None = None,
                progress=None) -> PriceSet:
    """
    Daily bars for `tickers` (a {symbol: yf_ticker} map), cached to disk.

    `market` is a `markets.Market`; it supplies the benchmark, the cache file
    and the default quote divisor.

    `divisors` optionally overrides `market.price_divisor` per symbol. Not
    every LSE line is quoted in pence - a handful report in GBP or USD - and
    `fundamentals.py` already knows each symbol's real currency from the `info`
    call it makes anyway, so the caller can pass the truth instead of this
    module assuming it.

    `progress` is an optional callable taking (done, total, label) so a UI can
    show a bar without this module importing Streamlit.
    """
    swing = cfg.swing
    benchmark = benchmark or market.benchmark_ticker
    bench_key = benchmark_key(benchmark)
    cache_path = Path(swing.cache_dir) / cache_name(market.cache_suffix)
    divisors = divisors or {}

    if not force_refresh:
        cached = _read_cache(cache_path, swing.price_cache_hours)
        if cached is not None:
            frame, fetched_at = cached
            have = set(frame["symbol"].unique())
            wanted = set(tickers) | {bench_key}
            # A cache built before you refreshed the universe is missing the
            # new names, and one built against a different benchmark is missing
            # this benchmark's key. Falling through to a download is cheaper
            # than scanning an incomplete universe and never saying so.
            if wanted <= have:
                return _unpack(frame, tickers, bench_key, fetched_at,
                               from_cache=True)

    period_days = max(swing.history_days, 120)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    symbols = list(tickers)
    jobs = [(bench_key, benchmark)] + [(s, tickers[s]) for s in symbols]
    batches = [jobs[i:i + BATCH_SIZE] for i in range(0, len(jobs), BATCH_SIZE)]

    done = 0
    for batch in batches:
        if progress:
            progress(done, len(jobs), "downloading daily bars")
        raw = _download([t for _, t in batch], period_days)
        for symbol, ticker in batch:
            # An index is quoted in points, never in the minor unit its
            # constituents use, so the benchmark is never divided.
            divisor = (1.0 if symbol == bench_key
                       else divisors.get(symbol, market.price_divisor))
            df = _extract(raw, ticker, divisor)
            if df is None or df.empty:
                missing.append(symbol)
            else:
                df = df.copy()
                df["symbol"] = symbol
                frames.append(df)
            done += 1
        # Yahoo throttles bursts. A short pause between batches is far
        # cheaper than the empty responses that follow a rate-limit.
        if len(batches) > 1:
            time.sleep(0.4)

    if progress:
        progress(len(jobs), len(jobs), "download complete")

    if not frames:
        raise FeedError(
            "yfinance returned no daily bars for any symbol. Check the "
            "network connection - every ticker failing at once is not a "
            "market condition."
        )

    combined = pd.concat(frames)
    fetched_at = datetime.now()
    _write_cache(cache_path, combined, fetched_at)

    result = _unpack(combined, tickers, bench_key, fetched_at, from_cache=False)
    result.missing = missing
    return result


# ---------------- download ----------------

def _download(tickers: list[str], period_days: int) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError as e:                              # pragma: no cover
        raise FeedError("yfinance is not installed - pip install yfinance") from e

    try:
        return yf.download(
            tickers, period=f"{period_days}d", interval="1d",
            group_by="ticker", auto_adjust=False, progress=False,
            threads=True,
        )
    except Exception:
        # One failed batch is not a failed scan; the symbols in it land in
        # `missing` and the page says so.
        return None


def _extract(raw: pd.DataFrame | None, ticker: str,
             divisor: float = 1.0) -> pd.DataFrame | None:
    """
    Pull one ticker out of a yfinance response and normalise it.

    `divisor` converts a minor-unit quote to major units (LSE pence -> pounds).
    Prices only: volume is a share count and turnover is recomputed from the
    normalised close, so dividing volume too would undo the correction.
    """
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.get_level_values(0):
            return None
        df = raw[ticker].copy()
    else:
        df = raw.copy()

    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    df = df.drop(columns=[c for c in ("adj close",) if c in df.columns])

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()

    if divisor and divisor != 1.0:
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = df[col] / divisor

    try:
        # Reuse the DataFeed contract rather than a second cleaning routine -
        # the strategies downstream are the same ones, so the guarantees
        # about column names, dtypes and ordering must be identical.
        return DataFeed.validate(df)
    except FeedError:
        return None


# ---------------- cache ----------------

def _read_cache(path: Path, max_age_hours: int
                ) -> tuple[pd.DataFrame, datetime] | None:
    if not path.exists():
        return None
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime)
    if datetime.now() - fetched_at > timedelta(hours=max_age_hours):
        return None
    try:
        return pd.read_parquet(path), fetched_at
    except Exception:
        # A torn parquet from an interrupted write should cost you a
        # re-download, not the scan.
        return None


def _write_cache(path: Path, frame: pd.DataFrame, fetched_at: datetime) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    except Exception:
        pass   # Caching is an optimisation; failing to cache is not an error.


def _unpack(frame: pd.DataFrame, tickers: dict[str, str], bench_key: str,
            fetched_at: datetime, from_cache: bool) -> PriceSet:
    out = PriceSet(fetched_at=fetched_at, from_cache=from_cache)
    missing: list[str] = []

    for symbol, group in frame.groupby("symbol", sort=False):
        df = group.drop(columns=["symbol"]).sort_index()
        if symbol == bench_key:
            out.benchmark = df
        elif symbol in tickers:
            out.bars[symbol] = df

    for symbol in tickers:
        if symbol not in out.bars:
            missing.append(symbol)
    out.missing = missing

    if out.bars:
        out.last_bar_date = max(df.index[-1] for df in out.bars.values())
    return out


# ---------------- derived series every stage wants ----------------

def avg_turnover(df: pd.DataFrame, divisor: float = 1e7,
                 lookback: int = 20) -> float:
    """
    Average daily traded value over `lookback` sessions, in units of `divisor`.

    Volume alone is the wrong liquidity test across a universe whose share
    prices span two orders of magnitude - ten lakh shares of a Rs 40 stock and
    ten lakh of a Rs 4,000 stock are not comparable positions.

    `divisor` is the market's unit: 1e7 renders rupees as crore, 1e6 renders
    dollars or pounds as millions. The close is already in major units by the
    time it gets here (see `_extract`), so this figure is in the market's own
    currency and must be compared against that market's floor, never a shared one.
    """
    if len(df) < lookback:
        lookback = len(df)
    if lookback == 0:
        return 0.0
    window = df.iloc[-lookback:]
    value = (window["close"] * window["volume"]).mean()
    return float(value) / float(divisor or 1.0)


def turnover_crore(df: pd.DataFrame, lookback: int = 20) -> float:
    """Rupees crore. Retained for the India-only call sites and tests."""
    return avg_turnover(df, 1e7, lookback)


def position_in_52w(df: pd.DataFrame) -> float:
    """
    Where the last close sits in the 52-week range: 0.0 at the low, 1.0 at
    the high. Returns 0.5 when the range has collapsed to nothing.
    """
    window = df.iloc[-250:] if len(df) > 250 else df
    high, low = float(window["high"].max()), float(window["low"].min())
    if high <= low:
        return 0.5
    return float((df["close"].iloc[-1] - low) / (high - low))


def relative_strength(stock: pd.DataFrame, benchmark: pd.DataFrame,
                      days: int) -> float | None:
    """
    Stock return minus benchmark return over `days` sessions.

    Aligned on shared dates first: a stock that was suspended for a week has
    fewer bars, and comparing its last-N-bars return against the index's
    last-N-bars return would silently compare two different time spans.
    """
    if benchmark is None or len(stock) < days + 1 or len(benchmark) < days + 1:
        return None
    joined = pd.concat([stock["close"], benchmark["close"]], axis=1,
                       join="inner").dropna()
    joined.columns = ["stock", "bench"]
    if len(joined) < days + 1:
        return None
    window = joined.iloc[-(days + 1):]
    s_ret = window["stock"].iloc[-1] / window["stock"].iloc[0] - 1.0
    b_ret = window["bench"].iloc[-1] / window["bench"].iloc[0] - 1.0
    return float(s_ret - b_ret)
