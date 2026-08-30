"""
Daily bars for what you actually hold, not for what the scanner screens.

WHY THIS IS NOT JUST `prices.load_prices`. That function takes a
`{symbol: yf_ticker}` map built from a universe CSV, and a portfolio is not a
universe: you can hold a mid cap that is in no index file, or a fund, or a
name that has since left the Nifty 100. So this resolves a ticker for every
position - the committed universe row where there is one, because it is
hand-maintained and handles the names that break the rule, and
`Market.yf_ticker()` otherwise - and then reuses `load_prices` unchanged, with
its cache, its batching and its pence divisor.

ONE MARKET AT A TIME, because the cache is one parquet per market and the
benchmark is keyed by ticker inside it. Scanning the US and then India against
a shared file is precisely the bug `prices.py`'s docstring describes, where
India's relative strength gets computed against the S&P 500 with no error
anywhere.

RETURNS ARE ALIGNED ON THE INTERSECTION OF DATES, and the count survives into
the output. Two names that share twelve sessions will happily produce a 0.9
correlation, and a correlation without an `n` beside it is a decoration.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.base import FeedError
from ..swing import markets as markets_mod
from ..swing import prices as prices_mod
from ..swing.universe import UniverseError, load_universe


def tickers_for(positions, market) -> dict[str, str]:
    """
    `{symbol: yf_ticker}` for every position in this market.

    Cash lines are excluded - there is nothing to price - and so is anything
    already present, so a name held in two accounts is fetched once.
    """
    try:
        rows = {s.symbol.upper(): s.yf_ticker
                for s in load_universe(market.universe_csv)}
    except UniverseError:
        rows = {}

    out: dict[str, str] = {}
    for p in positions:
        if p.market != market.key or p.asset_class == "cash":
            continue
        out.setdefault(p.symbol, rows.get(p.symbol) or market.yf_ticker(p.symbol))
    return out


def cached_symbols(cfg, market) -> set[str]:
    """
    What the market's price cache already holds.

    Needed because of a sharp edge in `prices.load_prices`: on ANY cache miss
    it re-downloads `history_days` (400 by default) for every requested symbol
    and REWRITES the parquet with just that. The committed workflow deliberately
    fetches six years into that same file (`scripts/fetch_swing_history.py`),
    and the swing backtest reads it - so a research run that asked for one
    unheld mid cap would silently truncate the history every backtest depends
    on, and nothing would report an error.

    So this package never triggers that path. It asks only for what is already
    cached, and names what it therefore could not price.
    """
    path = Path(cfg.swing.cache_dir) / prices_mod.cache_name(market.cache_suffix)
    if not path.exists():
        return set()
    try:
        import pandas as _pd
        return set(_pd.read_parquet(path, columns=["symbol"])["symbol"].unique())
    except Exception:
        return set()


def bars_for(snapshot, cfg, market_key: str = markets_mod.INDIA,
             force_refresh: bool = False):
    """
    Returns `(bars_by_symbol, benchmark_frame, note)`.

    READ-ONLY AGAINST THE PRICE CACHE unless `force_refresh` is set - see
    `cached_symbols` for why that is not timidity. A symbol you hold that has
    never been scanned is reported as unpriceable, with the command that would
    fix it, rather than quietly costing you six years of history.

    A failure returns empty bars and a note rather than raising: a briefing
    that cannot draw the correlation table is still worth reading, and it must
    say which half is missing rather than vanish.
    """
    market = markets_mod.get(cfg, market_key)
    wanted = tickers_for(snapshot.positions, market)
    if not wanted:
        return {}, None, (f"no priceable {market.label} positions - nothing to "
                          f"correlate")

    if force_refresh:
        askable, absent = wanted, []
    else:
        have = cached_symbols(cfg, market)
        askable = {s: t for s, t in wanted.items() if s in have}
        absent = sorted(set(wanted) - set(askable))

    note = ""
    bars, benchmark = {}, None
    if askable:
        try:
            priced = prices_mod.load_prices(askable, cfg, market,
                                            force_refresh=force_refresh)
            bars, benchmark = priced.bars, priced.benchmark
            note = priced.note()
            absent = sorted(set(absent) | set(priced.missing))
        except FeedError as e:
            return {}, None, f"daily bars unavailable ({e})"

    if absent:
        note += (f" No cached bars for {', '.join(absent)}, so those positions "
                 f"are absent from every correlation below. "
                 f"`python scripts/fetch_swing_history.py --market "
                 f"{market.key}` adds anything in the universe file; a holding "
                 f"outside it has to be added there first.")
    return bars, benchmark, note.strip()


def returns(bars: dict[str, pd.DataFrame], days: int) -> pd.DataFrame:
    """
    Daily percentage returns, one column per symbol, most recent `days`.

    Built from closes rather than from any adjusted series so it matches what
    the rest of this repo reads out of the same cache.
    """
    columns = {}
    for symbol, df in bars.items():
        if df is None or df.empty or "close" not in df:
            continue
        series = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(series) < 2:
            continue
        columns[symbol] = series.pct_change().dropna()
    if not columns:
        return pd.DataFrame()
    frame = pd.DataFrame(columns)
    return frame.tail(days) if days else frame


def align(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """
    Two series on their shared dates, plus the count of them.

    An index intersection rather than a join, for the same reason
    `prices.relative_strength` uses one: it is materially faster and it cannot
    introduce a NaN row that a correlation would then silently drop.
    """
    shared = left.index.intersection(right.index)
    return left.loc[shared], right.loc[shared], int(len(shared))
