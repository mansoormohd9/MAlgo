"""
Every causal indicator for one (symbol, session), computed once.

THE ARITHMETIC THAT MAKES THIS SAFE

A function is servable from a prefix slice when its value at bar i depends
only on bars <= i. `atr` is `ewm(adjust=False)` - a causal recursion seeded at
bar 0. `vwap` is `groupby(day).cumsum()`. The rest are `rolling()` or
elementwise. For all of them, `full_series.iloc[:n]` is EXACTLY equal to
recomputing over `df.iloc[:n]`, and the tests assert that elementwise over
every prefix rather than taking it on trust.

`find_pivots` IS NOT ONE OF THEM, AND THAT IS THE WHOLE DIFFICULTY

It uses a CENTRED window (`signals.py:130`): a bar is a swing high if it
exceeds `lookback` bars on BOTH sides. Precomputing it over a full session and
slicing would tell a strategy at 10:00 about a pivot that is only confirmable
at 10:25 - and it would then "trade the breakout" of a level that did not
exist yet. That produces a flattering equity curve and no error at all, which
is why `tests/test_lookahead.py` exists and why this module gets its own.

The delay is exact and unconditional. With `min_periods == window`, a centred
window at position j in a frame of length m is non-NaN iff `j-L >= 0` and
`j+L <= m-1`. For the prefix `df.iloc[:i+1]` that means:

    is_high_window[j] = is_high_full[j]   for  L <= j <= i - L
                      = False             for  j > i - L

The `[j-L, j+L]` span is fully inside `[0, i]` exactly when `j + L <= i`, and
over that span the bars are identical to the full session's - so the rolling
max is identical too. A pivot at bar j becomes visible at bar `i = j + L`,
NEVER EARLIER, and never later.

`PivotLadder` stores `confirm_at[j] = j + L` rather than scattering `+ L`
through its readers, so the shift is one number in one place.

WHY `prices_at` IS CHEAP

The confirmed-pivot set is a STRICT GROWING PREFIX: advancing one bar can
admit exactly one new candidate, `j = i - L`. So the caller keeps two running
lists and appends, and the boolean-index inside `build_levels`
(`df.loc[is_high, "high"].tolist()`, ~1.2ms of its 2.24ms) disappears. What is
left per bar is a `sorted()` over a dozen floats.

ONE LANDMINE REPRODUCED ON PURPOSE

`signals.build_levels` computes its clustering tolerance from `atr(df)` with
the DEFAULT period of 14 - not `cfg.signal.atr_period`. That is almost
certainly unintentional, but it is what the live book does today, so the pack
must serve period 14 there. "Correcting" it would change every level in the
backtest relative to live, silently, which is precisely the divergence
invariant 1 exists to prevent. If it is ever fixed, it must be fixed in
`signals.py` for both books at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .. import indicator_cache as _cache
from .. import signals as sig

#: `build_levels` hardcodes this. See the module docstring.
BUILD_LEVELS_ATR_PERIOD = 14

#: `volume_surge` is called everywhere as `volume_surge(df, multiple=...)`,
#: so its lookback is the function default. `range_percentile` is called
#: once, in squeeze.py, with an explicit `lookback=20`. Both are spelled
#: here because a cache key that does not match the call site is a cache
#: that never hits - and a pack that never hits is a 33-hour backtest that
#: looks exactly like a working one.
VOLUME_SURGE_LOOKBACK = 20
RANGE_PERCENTILE_LOOKBACK = 20


@dataclass(frozen=True)
class PivotLadder:
    """
    Centred-window pivots over a whole session, plus the bar each becomes
    knowable. Never read the raw masks directly - go through `visible_at`.
    """
    lookback: int
    is_high: np.ndarray
    is_low: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    #: confirm_at[j] == j + lookback. Stored rather than derived so the shift
    #: lives in one place.
    confirm_at: np.ndarray

    def visible_at(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Pivot masks as known at decision bar `i`, length i+1.

        Everything after `i - lookback` is forced False because its
        confirming bars have not printed.
        """
        n = i + 1
        cut = max(0, n - self.lookback)
        h = self.is_high[:n].copy()
        l = self.is_low[:n].copy()
        h[cut:] = False
        l[cut:] = False
        return h, l

    def prices_at(self, i: int) -> tuple[list[float], list[float]]:
        """Confirmed pivot prices as at bar `i`: (resistance, support)."""
        h, l = self.visible_at(i)
        return (self.highs[:i + 1][h].tolist(),
                self.lows[:i + 1][l].tolist())


@dataclass
class SessionPack:
    """
    One (symbol, session) worth of precomputed series.

    `series` is keyed `(function name, parameter tuple)` and is what
    `indicator_cache.served()` reads. `pivots` is keyed by lookback and is
    deliberately NOT reachable through that path - see the module docstring.
    """
    symbol: str
    day: date
    index: pd.DatetimeIndex
    n: int
    series: dict = field(default_factory=dict)
    #: (fn, key) -> per-bar array; element n-1 is the answer for a prefix of
    #: length n. For functions that return a scalar rather than a Series.
    scalars: dict = field(default_factory=dict)
    pivots: dict = field(default_factory=dict)
    #: Raw numpy datetime64 view of `index`. The cache lookup compares these
    #: rather than `df.index[0]`, which builds a Timestamp on every call - on
    #: the hot path that construction was most of the lookup cost.
    index_values: object = None

    # THE PACK MUST BE FREE TO COPY, AND BY DEFAULT IT IS NOT.
    #
    # pandas propagates `.attrs` through `.iloc[]`, `.copy()` and
    # `.sort_index()` by DEEP-COPYING it. `BarReplayer.window_at` slices once
    # per decision bar, so without these two methods every bar would deep-copy
    # a dozen pandas Series and several numpy arrays - measured at roughly 5x
    # the cost of the slice itself, and far more than simply recomputing the
    # indicators. The cache would have been SLOWER than no cache, while
    # returning identical numbers: a performance bug that looks exactly like
    # correct code, and one that a results-only test can never catch.
    #
    # Declaring the pack immutable to copy is honest rather than a trick - it
    # is built once and only ever read, which is the contract `attach()`
    # documents. Nothing may mutate a pack after `build_pack` returns.
    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    def atr_at(self, i: int, period: int = 14) -> float:
        s = self.series.get(("atr", (period,)))
        if s is None or i >= len(s):
            return 0.0
        v = float(s.iloc[i])
        return 0.0 if np.isnan(v) else v

    def close_at(self, i: int) -> float:
        return float(self.series[("__close__", ())].iloc[i])



def _scalars(session: pd.DataFrame, sg) -> dict:
    """
    Per-bar answers for the two scalar-valued gates every strategy calls.

    `_preflight` runs `underlying_liquidity_ok` once per strategy per bar, and
    that function calls `has_traded_volume`, which rescans the whole volume
    column. Profiled together they were the single largest remaining cost -
    339ms of a 731ms symbol-session - because they are the only hot functions
    that return a bool and so were invisible to a series cache.

    Both are causal: the value at bar i depends only on bars <= i. That is
    asserted against a direct recompute over every prefix, same as the series.
    """
    n = len(session)
    out: dict = {}

    vol = session["volume"].fillna(0.0).to_numpy(dtype=float)
    # has_traded_volume is "any volume > 0 SO FAR", which is a running max.
    out[("has_traded_volume", ())] = np.maximum.accumulate(vol > 0)

    lookback = 20                      # the call-site default; see the note above
    for min_ratio in (0.5,):
        flags = np.zeros(n, dtype=bool)
        use_volume = out[("has_traded_volume", ())]
        rng = (session["high"] - session["low"]).abs().to_numpy(dtype=float)
        for i in range(n):
            if i + 1 < lookback + 1:
                continue
            series = vol if use_volume[i] else rng
            recent = float(series[i])
            baseline = float(series[i + 1 - lookback:i + 1].mean())
            flags[i] = baseline > 0 and recent / baseline >= min_ratio
        out[("underlying_liquidity_ok", (lookback, min_ratio))] = flags
    return out


def build_pack(session: pd.DataFrame, symbol: str, cfg,
               extra_atr_periods=()) -> SessionPack:
    """
    Compute every causal series for one session, once.

    `session` must be ONE trading day's bars, starting at the session open -
    the same frame `engine.py:165` hands a live strategy. A multi-day frame
    would make `signals.opening_range` read the wrong day, silently.
    """
    sg = cfg.signal
    st = cfg.strategy
    ie = cfg.intraday_equity

    periods = {sg.atr_period, ie.atr_period, BUILD_LEVELS_ATR_PERIOD}
    periods.update(extra_atr_periods)

    store: dict = {}

    # ATR, one series per period anyone might ask for. `build_levels`'s
    # hardcoded 14 is in `periods` for the reason in the module docstring.
    for period in sorted(periods):
        store[("atr", (period,))] = sig.atr(session, period)

    store[("body_to_range", ())] = sig.body_to_range(session)
    store[("close_position_in_range", ())] = sig.close_position_in_range(session)
    store[("typical_price", ())] = sig.typical_price(session)
    store[("vwap", ())] = sig.vwap(session)

    # The keys below mirror the EXACT call sites in the strategies, defaults
    # included. `volume_surge` is always called as
    # `volume_surge(df, multiple=cfg.signal.volume_surge_multiple)` - so the
    # lookback is the function default of 20, and the cached key has to say
    # so or every lookup misses and the pack silently buys nothing.
    store[("volume_surge", (VOLUME_SURGE_LOOKBACK, sg.volume_surge_multiple))] = (
        sig.volume_surge(session, VOLUME_SURGE_LOOKBACK, sg.volume_surge_multiple))

    # ema_fast/ema_slow and squeeze_lookback live on StrategyConfig, not
    # SignalConfig - trend_pullback and squeeze read them from there.
    store[("trend_state", (st.ema_fast, st.ema_slow))] = sig.trend_state(
        session, st.ema_fast, st.ema_slow)
    store[("narrowest_range_n", (st.squeeze_lookback,))] = sig.narrowest_range_n(
        session, st.squeeze_lookback)
    store[("range_percentile", (RANGE_PERCENTILE_LOOKBACK,))] = sig.range_percentile(
        session, RANGE_PERCENTILE_LOOKBACK)

    #: not served through the cache - a convenience for the book layer
    store[("__close__", ())] = session["close"]

    pack = SessionPack(
        symbol=symbol,
        day=session.index[-1].date(),
        index=session.index,
        index_values=session.index.values,
        n=len(session),
        series=store,
        scalars=_scalars(session, sg),
    )

    lookback = sg.pivot_lookback
    is_high, is_low = sig.find_pivots(session, lookback)
    n = len(session)
    pack.pivots[lookback] = PivotLadder(
        lookback=lookback,
        is_high=is_high.to_numpy(dtype=bool),
        is_low=is_low.to_numpy(dtype=bool),
        highs=session["high"].to_numpy(dtype=float),
        lows=session["low"].to_numpy(dtype=float),
        confirm_at=np.arange(n) + lookback,
    )
    return pack


def attach(session: pd.DataFrame, pack: SessionPack) -> pd.DataFrame:
    """Hand back the frame carrying its pack. Treat it as read-only after."""
    return _cache.attach(session, pack)


def levels_at(pack: SessionPack, i: int, cfg) -> list:
    """
    `signals.build_levels` for bar `i`, from confirmed pivots only.

    Reproduces `build_levels` exactly, including its hardcoded ATR period -
    see the module docstring - so a level built here is byte-identical to one
    built by handing the truncated frame to `signals.build_levels` directly.
    That equality is a test, not a claim.
    """
    sg = cfg.signal
    ladder = pack.pivots.get(sg.pivot_lookback)
    if ladder is None:
        return []
    highs, lows = ladder.prices_at(i)
    current_atr = pack.atr_at(i, BUILD_LEVELS_ATR_PERIOD)
    tol = current_atr * sg.level_cluster_atr_frac
    res = sig.cluster_levels(highs, tol, "resistance", sg.min_level_touches)
    sup = sig.cluster_levels(lows, tol, "support", sg.min_level_touches)
    return res + sup


__all__ = ["SessionPack", "PivotLadder", "build_pack", "attach", "levels_at",
           "BUILD_LEVELS_ATR_PERIOD", "VOLUME_SURGE_LOOKBACK",
           "RANGE_PERCENTILE_LOOKBACK"]
