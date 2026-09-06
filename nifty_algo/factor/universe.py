"""
Who is eligible on a rebalance date, and in which liquidity band.

THE ONE THING THIS MODULE EXISTS TO GET RIGHT

Everything here is evaluated from bars STRICTLY BEFORE the rebalance date. A
factor book that ranks on data including the rebalance day is the same trap as
`ranking.morning_ranks` in the intraday book - it produces a spectacular curve
and no error. The `< day` filter in `eligible_at` is the whole safety property,
and `test_factor_universe.py` asserts that deleting every bar from the
rebalance date onward changes nothing.

WHY LIQUIDITY BANDS ARE A FIRST-CLASS CONCEPT HERE

Indian momentum evidence reports that the alpha concentrates in LOW-turnover
names: a 19-year NSE study puts low-turnover momentum at 19.43% CAGR against
8.51% for high-turnover, the latter below the Nifty 50 itself. The existing
swing book cannot test that at all - measured on its own universe, the least
liquid Nifty 100 name trades Rs 69cr a day against a Rs 25cr floor, so its
liquidity screen rejects nothing and the entire low-turnover band is absent.

So the band is a pre-registered VARIANT, assigned by cross-sectional quintile
at each rebalance rather than by a fixed rupee threshold. A fixed threshold
drifts: Rs 50cr was a large stock in 2015 and is a mid-cap now, so a constant
floor silently changes which band it selects as the market grows.

SURVIVORSHIP, WHICH THIS MODULE BOUNDS AND CANNOT FIX

The symbol list comes from Kite's dump, which is TODAY'S listed set. A company
delisted in 2019 is absent from it and from any history fetched. For a
long-only momentum book that bias runs one way and may be large, because
momentum's mechanism is losers continuing to lose and the worst losers are
precisely the names that stopped being listed.

`first_bar` and `listed_before` exist so a run can be repeated on the subset
that already existed when the window opened. The gap between the two results
is a LOWER BOUND on the inflation - not a correction, and never described as
one.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

#: Cross-sectional liquidity bands, as quantile ranges of turnover among the
#: names that already passed price and history gates. Pre-registered; the
#: backtest picks between them out of sample.
BANDS: dict[str, tuple[float, float]] = {
    "all": (0.00, 1.00),
    "liquid": (0.80, 1.00),      # top quintile - the evidence says this pays least
    "midliquid": (0.40, 0.80),   # the band a small account reaches and a fund cannot
    "illiquid": (0.20, 0.40),    # highest reported alpha, worst fills
}


@dataclass
class SymbolDaily:
    """One symbol's daily series, pre-indexed for point-in-time slicing."""
    symbol: str
    dates: list
    closes: np.ndarray
    turnover: np.ndarray
    #: Wilder ATR, or None when nothing asked for it. Optional because it is a
    #: third full-length array over ~2,400 symbols and every result recorded
    #: before F5 was produced without it - a universe that silently grew 50%
    #: in memory for every caller would be a cost nobody asked for.
    atr: np.ndarray | None = None

    def count_before(self, day: date) -> int:
        return bisect_left(self.dates, day)

    def atr_before(self, day: date) -> float | None:
        """
        The last ATR reading STRICTLY BEFORE `day`.

        Same `< day` discipline as every other accessor here: a stop distance
        computed from the bar it is about to trade on is look-ahead, and it
        would produce a better curve with no error anywhere.
        """
        if self.atr is None:
            return None
        k = self.count_before(day)
        if k <= 0:
            return None
        value = float(self.atr[k - 1])
        return value if np.isfinite(value) and value > 0 else None


@dataclass
class Eligibility:
    """What was eligible on one rebalance date, and why the rest was not."""
    day: date
    symbols: list = field(default_factory=list)
    turnover: dict = field(default_factory=dict)
    rejections: dict = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def accounts_for(self, total: int) -> bool:
        """Every candidate is either eligible or counted in a rejection."""
        return len(self.symbols) + sum(self.rejections.values()) == total


class FactorUniverse:
    """
    Daily closes and turnover for a wide universe, sliced point-in-time.

    Built once over the whole history, exactly as `ranking.DailyHistory` is -
    and legitimate for the same reason: a daily row for a session before `day`
    does not depend on `day`, so precomputing it is not look-ahead. What would
    be look-ahead is READING one, which `count_before` prevents by construction.
    """

    def __init__(self, bars: dict, adv_window: int = 60,
                 atr_window: int = 0):
        self.adv_window = adv_window
        self.atr_window = atr_window
        self.symbols: dict[str, SymbolDaily] = {}
        for symbol, frame in bars.items():
            if frame is None or frame.empty or "close" not in frame:
                continue
            closes = frame["close"].to_numpy(dtype=float)
            volume = (frame["volume"].to_numpy(dtype=float)
                      if "volume" in frame else np.zeros(len(frame)))
            self.symbols[symbol] = SymbolDaily(
                symbol=symbol,
                dates=[t.date() if hasattr(t, "date") else t
                       for t in frame.index],
                closes=closes,
                turnover=closes * volume,
                atr=(_atr(frame, atr_window) if atr_window else None))

    def first_bar(self, symbol: str) -> date | None:
        sd = self.symbols.get(symbol)
        return sd.dates[0] if sd and sd.dates else None

    def listed_before(self, day: date) -> set[str]:
        """
        Symbols whose history already began before `day`.

        The survivorship bound: re-running on this set answers "what if we had
        only ever traded names that existed when the window opened", which
        removes look-ahead about WHICH companies would come to exist - though
        not about which would survive.
        """
        return {s for s in self.symbols if (self.first_bar(s) or day) < day}

    def eligible_at(self, day: date, min_price: float, min_turnover: float,
                    min_history: int, band: str = "all",
                    restrict_to: set | None = None) -> Eligibility:
        """
        Who may be ranked on `day`, using bars strictly before it.

        Gates run cheap-to-expensive, and every candidate lands in exactly one
        bucket so `accounts_for` can assert the ledger is complete - the same
        discipline as `swing.scanner`'s rejection ledger. A universe that
        quietly shrinks is a backtest that silently changes question.
        """
        out = Eligibility(day=day)
        rows = []
        candidates = (self.symbols if restrict_to is None
                      else {s: v for s, v in self.symbols.items()
                            if s in restrict_to})

        for symbol, sd in candidates.items():
            k = sd.count_before(day)            # <-- THE safety property
            if k < min_history:
                out.reject("short_history")
                continue
            price = float(sd.closes[k - 1])
            if not np.isfinite(price) or price < min_price:
                out.reject("price_floor")
                continue
            adv = float(sd.turnover[max(0, k - self.adv_window):k].mean())
            if not np.isfinite(adv) or adv < min_turnover:
                out.reject("turnover_floor")
                continue
            rows.append((symbol, adv))

        if not rows:
            return out

        lo_q, hi_q = BANDS.get(band, BANDS["all"])
        advs = np.array([a for _, a in rows], dtype=float)
        lo = float(np.quantile(advs, lo_q)) if lo_q > 0 else -np.inf
        hi = float(np.quantile(advs, hi_q)) if hi_q < 1 else np.inf

        for symbol, adv in rows:
            # Half-open on the low side so the bands tile without overlap.
            if adv < lo or adv > hi:
                out.reject(f"band_{band}")
                continue
            out.symbols.append(symbol)
            out.turnover[symbol] = adv

        out.symbols.sort()
        return out

    def closes_before(self, symbol: str, day: date) -> np.ndarray:
        sd = self.symbols.get(symbol)
        if sd is None:
            return np.array([], dtype=float)
        return sd.closes[:sd.count_before(day)]

    def price_at(self, symbol: str, day: date) -> float | None:
        """
        The last close strictly before `day` - the price a rebalance decides
        on. The FILL is a separate question and belongs to the backtest.
        """
        c = self.closes_before(symbol, day)
        return float(c[-1]) if len(c) else None

    def price_on(self, symbol: str, day: date) -> float | None:
        """The close ON `day`, or None. Used only to fill, never to rank."""
        sd = self.symbols.get(symbol)
        if sd is None:
            return None
        i = bisect_left(sd.dates, day)
        if i >= len(sd.dates) or sd.dates[i] != day:
            return None
        return float(sd.closes[i])


def _atr(frame, window: int) -> np.ndarray:
    """
    Wilder's ATR over `window` sessions, as a plain array.

    TRUE RANGE NEEDS HIGH AND LOW, and a close-to-close proxy would understate
    it on exactly the gapping small caps this book holds - which would make an
    ATR stop tighter than it looks and fire more than intended. A frame with no
    high/low falls back to the close-to-close range rather than pretending,
    and the fallback is the pessimistic direction for a stop study: it makes
    the stop wider, so it cannot manufacture a firing.
    """
    close = frame["close"].to_numpy(dtype=float)
    if "high" in frame and "low" in frame:
        high = frame["high"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)
    else:
        high = low = close
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out = np.full(len(tr), np.nan, dtype=float)
    if len(tr) < window or window <= 0:
        return out
    # Wilder smoothing, seeded on the first full window.
    out[window - 1] = float(np.mean(tr[:window]))
    for i in range(window, len(tr)):
        out[i] = (out[i - 1] * (window - 1) + tr[i]) / window
    return out


def month_ends(sessions: list, hold_months: int = 1) -> list:
    """
    Rebalance dates: the last session of every `hold_months`-th month.

    Derived from the sessions actually present rather than from a calendar, so
    a holiday cannot produce a rebalance date on which nothing trades.
    """
    if not sessions:
        return []
    last_of_month: dict = {}
    for d in sorted(sessions):
        last_of_month[(d.year, d.month)] = d
    ordered = [last_of_month[k] for k in sorted(last_of_month)]
    return ordered[::hold_months] if hold_months > 1 else ordered


__all__ = ["FactorUniverse", "SymbolDaily", "Eligibility", "BANDS",
           "month_ends"]
