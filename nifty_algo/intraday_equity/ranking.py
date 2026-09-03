"""
The cross-sectional layer: which names are worth watching today, and which of
today's signals is worth taking.

This is the one part of the book with NO analogue in either existing system.
The option book trades a single instrument; the swing book ranks daily
candidates but never intraday. Everything here is new code, and it carries
the most dangerous trap in the package.

TRAP T1: RANKING FROM THE WHOLE SESSION IS A PERFECT STOCK PICKER

Sorting 100 names by the day's move and trading the top three produces a
spectacular equity curve and is entirely fictional. It is not a subtle
look-ahead either - it is the whole result. And because it produces numbers
rather than an exception, nothing about the run looks wrong.

So the module is split in two, and the split is the safety property:

  `morning_ranks()` runs ONCE, before the session opens, from PRIOR SESSIONS
  ONLY. Nothing it touches has a timestamp inside the day being traded. This
  is what a live scanner genuinely has at 09:15.

  `bar_score()` runs per signal and may read the session up to the DECISION
  BAR, never past it. It scores a signal that already exists; it cannot
  reach forward.

`morning_ranks` doing the prefilter is what makes the book affordable both
live and in backtest - the runner fetches ~20 symbols a bar instead of 100,
and the backtest evaluates a fifth of the universe. It is a 5x saving in both
places from the same mechanism. The honesty cost is stated where results are
printed: with a prefilter you can no longer answer "how often did ANY Nifty
100 name fire", only "how often did a top-20-by-RS name fire". `norank` is
the pre-registered variant that measures what the cut costs.

RELATIVE STRENGTH IS AGAINST THE INDEX, AND THE INDEX HAS NO VOLUME

NIFTY returns volume 0 from Kite. Only `close` is ever read from the
benchmark here - never volume, never a VWAP, and the benchmark series is
never divided or normalised the way a pence-quoted LSE price is.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MorningRank:
    """
    One symbol's standing at the open, from prior sessions only.

    Every field here is knowable at 09:15. If a field is ever added that is
    not, the prefilter stops being legitimate and the whole result with it.
    """
    symbol: str
    rs_short: float          # stock vs benchmark, over rs_short_days
    rs_long: float           # stock vs benchmark, over rs_long_days
    prior_atr_pct: float     # yesterday's ATR as a fraction of yesterday's close
    prior_close: float
    adv_inr: float           # average daily turnover, prior sessions
    score: float

    def why(self) -> str:
        return (f"RS {self.rs_short:+.1%}/{self.rs_long:+.1%}, "
                f"ATR {self.prior_atr_pct:.2%}, ADV Rs {self.adv_inr/1e7:.1f}cr")


def _daily_closes(bars: pd.DataFrame) -> pd.Series:
    """Session closes from an intraday frame, indexed by date."""
    if bars is None or bars.empty:
        return pd.Series(dtype=float)
    s = bars["close"]
    return s.groupby(s.index.date).last()


def _daily_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-session OHLCV+turnover from an intraday frame."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    g = bars.groupby(bars.index.date)
    out = pd.DataFrame({
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    out["turnover"] = out["close"] * out["volume"]
    return out


def relative_strength(stock_closes: pd.Series, bench_closes: pd.Series,
                      days: int) -> float | None:
    """
    Stock return minus benchmark return over `days` sessions.

    Aligned on the SHARED index first, rather than concatenated - a suspended
    or newly listed name has fewer sessions, and joining on the union would
    compare a stock's 5-day move against the index's 5-day move measured over
    different 5 days. The swing book took the same optimisation for the same
    reason (`prices.relative_strength`).
    """
    if stock_closes is None or bench_closes is None:
        return None
    shared = stock_closes.index.intersection(bench_closes.index)
    if len(shared) < days + 1:
        return None
    s = stock_closes.reindex(shared).to_numpy(dtype=float)
    b = bench_closes.reindex(shared).to_numpy(dtype=float)
    if s[-1 - days] <= 0 or b[-1 - days] <= 0:
        return None
    return float(s[-1] / s[-1 - days] - b[-1] / b[-1 - days])


@dataclass
class _SymbolDaily:
    """One symbol's daily aggregates over its whole history."""
    dates: list
    closes: pd.Series
    atr: np.ndarray
    turnover: np.ndarray


class DailyHistory:
    """
    Per-symbol daily aggregates over the WHOLE history, computed once.

    `morning_ranks` used to rebuild these on every session: it filtered the
    intraday frame with `frame.index.date < day` - one Python date object per
    row, 55,000 per symbol - and then re-aggregated the entire prior history
    with a groupby. Measured on the India universe those two steps cost 1.09s
    and 0.96s PER SESSION, which was 40% of the whole backtest spent
    recomputing yesterday's answer 739 times.

    WHY PRECOMPUTING IS NOT LOOK-AHEAD. Everything served here is
    prefix-stable, so the value at row k is identical whether or not rows
    after k exist:

      - a daily OHLCV row for a session before `day` does not depend on `day`;
      - `ewm(adjust=False)` is a causal recursion seeded at row 0, so ATR at
        row k reads rows 0..k only;
      - `rolling(20).mean()` at row k reads rows k-19..k only.

    Reading row `k-1` after `prior_count()` therefore gives exactly what
    truncating the intraday frame first and aggregating afterwards gave -
    which `test_intraday_equity_ranking.py` asserts numerically over every
    session rather than arguing here.
    """

    def __init__(self, bars: dict, benchmark: pd.DataFrame | None,
                 atr_period: int = 14, adv_window: int = 20):
        self.symbols: dict[str, _SymbolDaily] = {}
        for symbol, frame in bars.items():
            daily = _daily_frame(frame)
            if daily.empty:
                continue
            closes = daily["close"]
            pc = closes.shift(1)
            tr = np.maximum(daily["high"] - daily["low"],
                            np.maximum((daily["high"] - pc).abs(),
                                       (daily["low"] - pc).abs()))
            atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()
            # Turnover is stored RAW and averaged at read time over the same
            # trailing slice `.tail(20).mean()` took. A precomputed
            # `rolling(20).mean()` is algebraically the same and differs in
            # the last ulp - float64 summation order - which showed up as a
            # 1e-6 discrepancy on values of order 1e10. Harmless, but exact
            # equality is a far stronger claim to verify against than a
            # tolerance, so the cheap slice is worth its cost here.
            self.symbols[symbol] = _SymbolDaily(
                dates=list(daily.index), closes=closes,
                atr=atr.to_numpy(dtype=float),
                turnover=daily["turnover"].to_numpy(dtype=float))
        self.adv_window = adv_window

        bench = _daily_closes(benchmark)
        self.bench_closes = bench
        self.bench_dates = list(bench.index)

    def prior_count(self, dates: list, day: date) -> int:
        """How many sessions fall strictly before `day`."""
        return bisect_left(dates, day)

    def bench_upto(self, day: date) -> pd.Series:
        return self.bench_closes.iloc[:bisect_left(self.bench_dates, day)]


def morning_ranks(bars: dict, benchmark: pd.DataFrame, day: date, cfg,
                  min_prior_sessions: int = 25,
                  history: "DailyHistory | None" = None) -> list[MorningRank]:
    """
    Rank the universe as at the OPEN of `day`, from prior sessions only.

    THE `< day` FILTER BELOW IS THE ENTIRE SAFETY PROPERTY OF THIS MODULE.
    Every series is truncated strictly before the session being traded, so
    nothing here can see a single bar of it. Changing `<` to `<=` would turn
    this into trap T1 and produce a book that appears to work.
    """
    ie = cfg.intraday_equity
    if history is None:
        # Built here when a caller has not supplied one, so there is exactly
        # ONE code path rather than a fast branch and a slow branch that can
        # disagree. The backtest builds it once and passes it in; the live
        # runner's `bars` is a single session, so building it costs nothing.
        history = DailyHistory(bars, benchmark)

    bench_daily = history.bench_upto(day)              # <-- see the docstring
    if len(bench_daily) < ie.rs_long_days + 1:
        return []

    out: list[MorningRank] = []
    for symbol in bars:
        sd = history.symbols.get(symbol)
        if sd is None:
            continue
        # Sessions strictly before `day` - the entire safety property, now
        # expressed as a positional bound instead of a boolean mask.
        k = history.prior_count(sd.dates, day)
        if k < min_prior_sessions:
            continue

        closes = sd.closes.iloc[:k]
        rs_s = relative_strength(closes, bench_daily, ie.rs_short_days)
        rs_l = relative_strength(closes, bench_daily, ie.rs_long_days)
        if rs_s is None or rs_l is None:
            continue

        prev_close = float(closes.iloc[-1])
        if prev_close <= 0:
            continue

        atr = float(sd.atr[k - 1])
        adv = float(sd.turnover[max(0, k - history.adv_window):k].mean())

        # Blend, tilted to the shorter window: an intraday book cares more
        # about which names are in play this week than this quarter.
        blended = 0.6 * rs_s + 0.4 * rs_l
        out.append(MorningRank(
            symbol=symbol, rs_short=rs_s, rs_long=rs_l,
            prior_atr_pct=atr / prev_close, prior_close=prev_close,
            adv_inr=adv, score=blended))

    out.sort(key=lambda r: r.score, reverse=True)
    return out


def prefilter(ranks: list[MorningRank], cfg,
              min_adv_inr: float | None = None) -> list[str]:
    """
    The watchlist: the top `rs_prefilter_n` names by morning rank.

    `rs_prefilter_n <= 0` disables the cut entirely, which is what the
    `norank` variant sets so the cost of the prefilter is measured rather
    than assumed.
    """
    ie = cfg.intraday_equity
    floor = ie.min_session_turnover_inr if min_adv_inr is None else min_adv_inr
    # NOTE the DAILY band. `prior_atr_pct` is computed from prior SESSIONS,
    # so it is a daily ATR (~2.2% on a Nifty 100 name). Testing it against
    # the 5-minute band (~0.26%) rejects the entire universe and reports it
    # as "no candidates" - see the comment on these fields in config.py.
    eligible = [r for r in ranks
                if r.adv_inr >= floor
                and ie.min_daily_atr_pct <= r.prior_atr_pct <= ie.max_daily_atr_pct]
    if ie.rs_prefilter_n and ie.rs_prefilter_n > 0:
        eligible = eligible[:ie.rs_prefilter_n]
    return [r.symbol for r in eligible]


# ------------------------------------------------------------------ scoring


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def bar_score(confidence: float, reward_risk: float, morning: MorningRank | None,
              volume_ratio: float | None, session_position: float | None,
              cfg) -> tuple[float, dict]:
    """
    Score one signal on one bar. Returns (total, attribution).

    Reads nothing past the decision bar: `volume_ratio` and
    `session_position` are computed by the caller from the truncated window.

    WHEN A COMPONENT IS UNAVAILABLE ITS WEIGHT IS REDISTRIBUTED over the
    survivors, not scored as a neutral 0.5. Scoring it 0.5 silently penalises
    a strong candidate and rewards a weak one, which is the behaviour
    `swing/scanner._score` already had to fix for its news component.
    """
    weights = cfg.intraday_equity.score_weights()
    raw: dict = {
        "setup": _clip01(confidence),
        "reward_risk": _clip01((reward_risk - 1.0) / 2.0),
        "relative_strength": (None if morning is None
                              else _clip01(0.5 + morning.score / 0.08 * 0.5)),
        "volume": None if volume_ratio is None else _clip01((volume_ratio - 0.5) / 2.0),
        "session_position": None if session_position is None else _clip01(session_position),
    }

    available = {k: w for k, w in weights.items() if raw.get(k) is not None}
    total_w = sum(available.values())
    if total_w <= 0:
        return 0.0, {}

    parts: dict = {}
    score = 0.0
    for name, w in available.items():
        share = w / total_w                       # redistribute, never 0.5
        contribution = raw[name] * share
        score += contribution
        parts[name] = {"raw": raw[name], "weight": share,
                       "contribution": contribution}
    return score, parts


def session_position(window: pd.DataFrame) -> float | None:
    """Where the last close sits in the session's range so far, 0..1."""
    if window is None or window.empty:
        return None
    hi = float(window["high"].max())
    lo = float(window["low"].min())
    if hi <= lo:
        return None
    return _clip01((float(window["close"].iloc[-1]) - lo) / (hi - lo))


def volume_ratio(window: pd.DataFrame, lookback: int = 20) -> float | None:
    """This bar's volume against its own recent average. None without volume."""
    if window is None or len(window) < lookback + 1:
        return None
    vol = window["volume"]
    if not float(vol.sum()):
        return None
    baseline = float(vol.iloc[-lookback:].mean())
    if baseline <= 0:
        return None
    return float(vol.iloc[-1]) / baseline


__all__ = ["MorningRank", "DailyHistory", "morning_ranks", "prefilter", "bar_score",
           "relative_strength", "session_position", "volume_ratio"]
