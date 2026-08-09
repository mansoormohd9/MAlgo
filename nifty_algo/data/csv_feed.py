"""
CSV / Parquet feed, and the bar replayer.

THE REPLAYER IS THE MOST IMPORTANT CLASS IN THE DATA LAYER, and the reason is
`signals.find_pivots()`.

`find_pivots` uses a CENTRED rolling window: a bar is a swing high if it
exceeds `lookback` bars on BOTH sides. Live, you cannot know that until
`lookback` more bars have printed. Feed a backtester the whole frame and it
will happily identify a pivot on the bar it formed, build a level from it, and
trade a breakout of a level that did not exist yet.

That is look-ahead bias, it is invisible in the equity curve, and it makes
every backtest number meaningless in the most flattering possible direction.

`BarReplayer.next()` therefore hands out a window ending at the current bar
and nothing beyond it - exactly what a live strategy sees. `tests/` asserts
this property directly.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .base import DataFeed, FeedError, NotConfigured


class CsvFeed(DataFeed):
    name = "csv"
    is_delayed = False
    latency_note = "historical file - replay only, NOT live prices"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cache: Optional[pd.DataFrame] = None

    def get_bars(self, lookback_days: int = 5) -> pd.DataFrame:
        df = self._load()
        if lookback_days <= 0:
            return df
        days = sorted({d for d in df.index.date})[-lookback_days:]
        return df[[d in set(days) for d in df.index.date]]

    def _load(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            raise NotConfigured(
                f"data file not found: {self.path}. Generate the sample with "
                f"`python -m nifty_algo.data.sample`, or point "
                f"DataConfig.csv_path at your own file."
            )
        if self.path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(self.path)
        else:
            df = pd.read_csv(self.path)

        df = self._normalise(df)
        self._cache = self.validate(df)
        return self._cache

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Accept the column spellings brokers and exporters actually emit."""
        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
        aliases = {
            "date": "timestamp", "datetime": "timestamp", "time": "timestamp",
            "vol": "volume", "qty": "volume",
            "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
        if "timestamp" not in df.columns:
            raise FeedError(
                "no timestamp column found - expected one of "
                "timestamp/date/datetime/time"
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df


class BarReplayer:
    """
    Walk a historical frame bar by bar, exposing only what was knowable then.

    Usage:
        for window in BarReplayer(df, warmup=30):
            signal = strategy.on_bar(build_context(window))

    `window` always ENDS at the bar being decided on. There is no future data
    in it, so a strategy cannot see a pivot before it was confirmable.
    """

    def __init__(self, df: pd.DataFrame, warmup: int = 30,
                 max_window: int = 500):
        self.df = df.sort_index()
        self.warmup = max(warmup, 2)
        self.max_window = max_window
        self._i = self.warmup

    def __iter__(self) -> Iterator[pd.DataFrame]:
        while self._i < len(self.df):
            yield self.window_at(self._i)
            self._i += 1

    def window_at(self, i: int) -> pd.DataFrame:
        """
        Bars [start .. i] inclusive. The slice end is `i + 1` in Python terms;
        the bar at index i is the one being decided on and nothing after it is
        included. `max_window` caps memory on long frames without changing
        what is visible at the right edge.
        """
        start = max(0, i - self.max_window + 1)
        return self.df.iloc[start:i + 1]

    def reset(self) -> None:
        self._i = self.warmup

    def __len__(self) -> int:
        return max(0, len(self.df) - self.warmup)
