"""
The DataFeed contract.

Every feed returns the same thing: a DataFrame indexed by tz-naive local
timestamps with columns open/high/low/close/volume. Strategies never learn
which feed they are reading, which is what lets the same strategy code run
against a CSV replay, a delayed public feed, and a broker WebSocket.

`is_delayed` and `latency_note` are not decoration. A delayed feed can still
validate that signals fire and alerts route, but acting on one is trading on
stale prices. The UI renders these fields permanently so the distinction is
never left to memory.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class FeedError(RuntimeError):
    """Any failure to obtain usable bars. The engine trips the kill switch on this."""


class NotConfigured(FeedError):
    """Credentials or files are missing. Distinct from a transient failure."""


@dataclass
class PriorSession:
    """Prior-day levels that Context requires."""
    high: float
    low: float
    close: float


class DataFeed(ABC):
    name: str = "base"
    is_delayed: bool = False
    latency_note: str = "unknown latency"

    @abstractmethod
    def get_bars(self, lookback_days: int = 5) -> pd.DataFrame:
        """
        Return OHLCV bars, oldest first, at the configured interval.
        May span several sessions - callers slice out the current one.
        """

    def latest_bar(self) -> Optional[pd.Series]:
        df = self.get_bars(lookback_days=1)
        return None if df.empty else df.iloc[-1]

    # ---------------- shared helpers ----------------

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce the contract. A feed that silently returns the wrong shape is
        far more dangerous than one that raises - the strategies would happily
        compute signals off nonsense.
        """
        if df is None or df.empty:
            raise FeedError("feed returned no bars")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise FeedError(f"feed missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise FeedError("feed index is not a DatetimeIndex")

        df = df[list(REQUIRED_COLUMNS)].copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df[list(REQUIRED_COLUMNS)] = df[list(REQUIRED_COLUMNS)].apply(
            pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["volume"] = df["volume"].fillna(0.0)
        if df.empty:
            raise FeedError("feed returned no usable bars after cleaning")
        return df

    @staticmethod
    def session_slice(df: pd.DataFrame, day: Optional[date] = None) -> pd.DataFrame:
        """Bars belonging to one trading day - defaults to the most recent."""
        if df.empty:
            return df
        target = day or df.index[-1].date()
        return df[df.index.date == target]

    @staticmethod
    def prior_session(df: pd.DataFrame, day: Optional[date] = None) -> PriorSession:
        """
        Prior-day high/low/close for Context.

        Falls back to the first bar of the current session when there is no
        prior day in the frame - a same-day CSV should not crash the engine,
        but the levels it produces are meaningless, so strategies relying on
        them will simply not fire.
        """
        if df.empty:
            return PriorSession(0.0, 0.0, 0.0)
        target = day or df.index[-1].date()
        prior = df[df.index.date < target]
        if prior.empty:
            first = df.iloc[0]
            return PriorSession(float(first["high"]), float(first["low"]),
                                float(first["close"]))
        last_day = prior.index[-1].date()
        pdf = prior[prior.index.date == last_day]
        return PriorSession(
            high=float(pdf["high"].max()),
            low=float(pdf["low"].min()),
            close=float(pdf["close"].iloc[-1]),
        )

    @staticmethod
    def detect_gap(df: pd.DataFrame, interval_minutes: int,
                   max_gap_bars: int) -> Optional[str]:
        """
        Look for a hole in the bar stream. A data gap means the strategies are
        computing indicators across missing time, so the engine treats this as
        a kill-switch condition (non-negotiable #1).

        Gaps spanning a session boundary are expected and ignored.
        """
        if len(df) < 2:
            return None
        today = df.index[-1].date()
        session = df[df.index.date == today]
        if len(session) < 2:
            return None
        deltas = session.index.to_series().diff().dropna()
        threshold = pd.Timedelta(minutes=interval_minutes * (max_gap_bars + 1))
        worst = deltas.max()
        if worst > threshold:
            at = deltas.idxmax()
            return (f"data gap of {worst} before {at:%H:%M} "
                    f"(expected {interval_minutes}m bars)")
        return None


class StaticFrameFeed(DataFeed):
    """A feed wrapping an in-memory frame. Used by tests and the backtester."""
    name = "static"
    is_delayed = False
    latency_note = "in-memory frame, no latency"

    def __init__(self, df: pd.DataFrame):
        self._df = self.validate(df)

    def get_bars(self, lookback_days: int = 5) -> pd.DataFrame:
        return self._df.copy()
