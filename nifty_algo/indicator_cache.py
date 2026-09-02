"""
Serve precomputed indicator series to `signals.py` without letting a strategy
see the future.

WHY THIS EXISTS

Measured, a naive intraday-equity backtest is 33 hours per pass: 100 symbols x
750 sessions x ~45 decision bars x 10 strategies. The cause is NOT the O(n^2)
frame growth that `CLAUDE.md` documents for the swing book - an intraday
session is structurally capped at ~75 bars, so n^2 is nothing. It is fixed
pandas dispatch overhead multiplied by call count. `sig.atr` costs ~1.28ms
whether the frame is 75 rows or 7,500, and `BaseStrategy._preflight` calls it
ONCE PER STRATEGY PER BAR with identical arguments - ten identical
recomputations of one number, ~12.8ms per bar.

So the fix is to compute each series once per (symbol, session) and serve
`.iloc[:n]` views of it. Measured, that alone is 2.7x.

WHY IT IS NOT `if backtest:`

Strategy code must be byte-identical between live and backtest
(CLAUDE.md invariant 1). The branch here is not "am I in a backtest", it is
"does this frame carry a precomputed pack" - and the LIVE runner attaches one
too, because `engine`-style loops recompute the same ATR every bar for the
trail anyway. Both paths therefore run the same code and get the same
numbers; one of them just does not pay for them ten times.

WHY THE CACHE IS NOT KEYED ON `id(frame)`

Because that is a correctness bug, not an optimisation trade-off. A bar loop
creates and drops ~75 frames per symbol-session, and CPython reuses freed
addresses within milliseconds. One symbol's ATR would be served for another
with no exception anywhere - which is exactly the shape of the benchmark-cache
bug that once scored India's relative strength against the S&P 500. So
`pack_for` verifies CONTENT: the symbol, the length, and both endpoints of the
index.

THE ALLOWLIST IS THE FAILURE-SAFE DIRECTION

`CAUSAL_SERVED` names the functions that may be served from a prefix slice.
Every one is causal - a rolling, ewm or elementwise computation whose value at
bar i depends only on bars <= i - so `series.iloc[:n]` is exactly equal to
recomputing over `df.iloc[:n]`. That equality is asserted exhaustively in the
tests, not argued here.

A function NOT on the list is always recomputed. So when someone adds a
centred or backward-filled indicator to `signals.py` in future, the
consequence is that it is slow, not that it is clairvoyant. That asymmetry is
deliberate and mirrors `BOOK_ONLY_SWING_FIELDS` in the swing backtester:
forgetting to opt in costs performance, forgetting to opt out would cost
correctness.

`find_pivots` is DELIBERATELY ABSENT from the allowlist even though the pack
computes it. It uses a CENTRED window, so a prefix slice of a full-session
computation leaks the future by construction. It is served instead through
`PivotLadder.visible_at()`, which applies the exact `+lookback` confirmation
delay. See `intraday_equity/precompute.py`.

THE KILL SWITCH

Set `NIFTY_ALGO_DISABLE_PACK=1` and every lookup misses, so anything
suspicious can be re-run uncached and diffed in one command with no code
change.

THE ONE HOLE THIS DOES NOT COVER

A caller that mutates a frame in place after a pack is attached
(`df["close"] = ...`) gets stale indicators, because the checks compare the
index rather than the values. No caller in this repo does that, and frames
handed out by the bar loop are treated as read-only. It is recorded here
rather than papered over.
"""
from __future__ import annotations

import os

import pandas as pd

#: Where the pack is stashed on a DataFrame. `.attrs` propagates through both
#: `.iloc[a:b]` and `.copy()`, which is what lets a bar-loop window carry it
#: with no change to the slicing code.
PACK_KEY = "_nifty_algo_indicator_pack"

#: Environment kill switch. Read once per call rather than at import, so it
#: can be toggled inside a running process (a test, a notebook).
DISABLE_ENV = "NIFTY_ALGO_DISABLE_PACK"

#: Functions whose answer for a prefix is a SCALAR rather than a series, so
#: the pack stores one value per bar and `served_scalar` reads position n-1.
#: Causal in exactly the same sense; separated only by return type.
SCALAR_SERVED = frozenset({
    "has_traded_volume",
    "underlying_liquidity_ok",
})

#: `find_pivots` is served, but NEVER from a prefix slice - only through
#: `PivotLadder.visible_at`, which applies the +lookback confirmation delay.
#: It is named separately from `CAUSAL_SERVED` precisely so that nobody can
#: add it to the prefix path by pattern-matching the others.
PIVOT_SERVED = frozenset({"find_pivots"})

#: Functions that may be served from a prefix slice of a precomputed series.
#: EVERY ONE MUST BE CAUSAL. See the module docstring before adding to this.
CAUSAL_SERVED = frozenset({
    "atr",
    "vwap",
    "body_to_range",
    "close_position_in_range",
    "volume_surge",
    "trend_state",
    "narrowest_range_n",
    "range_percentile",
    "typical_price",
})


#: Read once per process. The lookup is on the hot path - roughly a hundred
#: thousand times per symbol-session - and an `os.environ` read there costs
#: more than some of the calls being cached. `refresh_disabled()` exists so a
#: test can still toggle it.
_DISABLED = os.environ.get(DISABLE_ENV, "") not in ("", "0", "false", "False")


def refresh_disabled() -> bool:
    """Re-read the kill switch from the environment. Returns the new state."""
    global _DISABLED
    _DISABLED = os.environ.get(DISABLE_ENV, "") not in ("", "0", "false", "False")
    return _DISABLED


def disabled() -> bool:
    return _DISABLED


def attach(df: pd.DataFrame, pack) -> pd.DataFrame:
    """
    Return `df` carrying `pack`.

    The frame is expected to be treated as read-only from here on - see the
    hole documented in the module docstring.
    """
    df.attrs[PACK_KEY] = pack
    return df


def pack_of(df: pd.DataFrame):
    """The attached pack, whatever it is, without any eligibility check."""
    try:
        return df.attrs.get(PACK_KEY)
    except Exception:
        return None


def _matches(df: pd.DataFrame, pack) -> bool:
    """
    Is `df` a non-empty prefix of `pack`'s session?

    Compares raw numpy datetime64 values rather than `df.index[0]`, which
    constructs a `pd.Timestamp` on every call. On the hot path that
    construction was most of the lookup cost.
    """
    n = len(df)
    if n == 0 or n > pack.n:
        return False
    try:
        vals = df.index.values
        want = pack.index_values
        return vals[0] == want[0] and vals[n - 1] == want[n - 1]
    except Exception:
        return False


def pack_for_pivots(df: pd.DataFrame, fn_name: str):
    """
    The pack that may serve a PIVOT function for `df`, or None.

    Deliberately a separate door from `pack_for`. Anything reached through
    here must apply the confirmation delay itself; nothing here may be served
    by slicing a full-session computation.
    """
    if _DISABLED or fn_name not in PIVOT_SERVED:
        return None
    pack = pack_of(df)
    if pack is None or not _matches(df, pack):
        return None
    return pack


def scalar_for(df: pd.DataFrame, fn_name: str):
    """The pack that may serve a scalar-valued function for `df`, or None."""
    if _DISABLED or fn_name not in SCALAR_SERVED:
        return None
    pack = pack_of(df)
    if pack is None or not _matches(df, pack):
        return None
    return pack


def served_scalar(pack, fn_name: str, key, n: int, default=None):
    """
    `pack`'s value for (fn_name, key) as at a prefix of length `n`.

    Returns `default` on a miss so a caller can distinguish "no pack" from a
    legitimately falsey cached answer - which matters, because both of the
    scalar functions here return bools.
    """
    if pack is None:
        return default
    store = getattr(pack, "scalars", None)
    if not store:
        return default
    arr = store.get((fn_name, key))
    if arr is None or n <= 0 or n > len(arr):
        return default
    return bool(arr[n - 1])


def pack_for(df: pd.DataFrame, fn_name: str):
    """
    The pack that may serve `fn_name` for `df`, or None.

    Verifies content rather than identity:

      * the function is on the causal allowlist;
      * the frame is a non-empty PREFIX of the packed session - same first
        timestamp, and its last timestamp is the packed one at that position.

    Any mismatch returns None and the caller recomputes, which is always
    correct and merely slower.
    """
    if _DISABLED or fn_name not in CAUSAL_SERVED:
        return None
    pack = pack_of(df)
    if pack is None or not _matches(df, pack):
        return None
    return pack


def served(pack, fn_name: str, key, n: int):
    """
    `pack`'s series for (fn_name, key), truncated to `n` rows - or None.

    `key` is the parameter tuple that identifies the variant (period,
    lookback, fast/slow, ...). A key the pack does not hold is a miss, not an
    error: the pack precomputes the parameters the config asks for, and a
    caller passing something else simply pays full price.
    """
    if pack is None:
        return None
    store = getattr(pack, "series", None)
    if not store:
        return None
    hit = store.get((fn_name, key))
    if hit is None:
        return None
    return hit.iloc[:n]


__all__ = ["PACK_KEY", "DISABLE_ENV", "CAUSAL_SERVED", "SCALAR_SERVED",
           "PIVOT_SERVED", "attach", "pack_of", "pack_for", "pack_for_pivots",
           "scalar_for", "served", "served_scalar", "disabled",
           "refresh_disabled"]
