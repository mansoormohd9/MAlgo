"""
Formation: how a name is scored on a rebalance date.

12-1 AND 6-1, AND WHY THE SKIP MONTH IS NOT OPTIONAL

The standard cross-sectional momentum signal is the return over the past
twelve months EXCLUDING the most recent one. The skip exists because short-
horizon returns reverse: a name up hard last month tends to give some back
next month, and including that month mixes a reversal signal into a momentum
signal with the opposite sign. Jegadeesh and Titman's original construction
skips it, and every serious replication since has.

Both 12-1 and 6-1 are pre-registered here. Neither is "the" answer; they are
the two windows the literature actually uses, and the backtest chooses between
them out of sample rather than on my prior.

WHAT THIS MODULE DELIBERATELY DOES NOT DO

No volatility scaling, no beta adjustment, no sector neutralisation, no
blending of factors. Each is defensible, each is a fork in a garden of paths,
and with 65% of published anomalies failing a t>1.96 hurdle the way to lose
here is to try many things and report the best. One signal, two windows,
chosen out of sample.

THE NULL LIVES HERE TOO. `random_scores` produces the same shape from a seeded
generator, so the whole downstream book - sizing, costs, rebalancing,
concentration limits - runs identically over a signal that cannot contain
information. A momentum book that does not beat it is not a momentum book.
"""
from __future__ import annotations

import numpy as np

#: Trading sessions per month, for converting a lookback in months to bars.
#: The NSE averages a little under 21; using a constant keeps formation
#: windows identical across symbols with different holiday histories.
SESSIONS_PER_MONTH = 21

#: Pre-registered formation windows, (lookback_months, skip_months).
FORMATIONS: dict[str, tuple[int, int]] = {
    "mom12_1": (12, 1),
    "mom6_1": (6, 1),
}


def formation_return(closes: np.ndarray, lookback_months: int = 12,
                     skip_months: int = 1,
                     sessions_per_month: int = SESSIONS_PER_MONTH
                     ) -> float | None:
    """
    Total return over the formation window, skipping the most recent month.

    `closes` must already be truncated to bars strictly before the rebalance
    date - this function has no notion of the date and cannot enforce that,
    which is why `FactorUniverse.closes_before` is the only intended source.

    Returns None rather than a number when the window cannot be filled. A
    partial window silently computes a shorter-horizon signal, which for a
    momentum book means computing a reversal signal on the newest listings -
    exactly the names where it does most damage.
    """
    lookback = lookback_months * sessions_per_month
    skip = skip_months * sessions_per_month
    need = lookback + 1
    if closes is None or len(closes) < need:
        return None

    end = len(closes) - skip                     # exclusive; skips the last month
    begin = len(closes) - lookback - 1
    if begin < 0 or end <= begin:
        return None

    start_price = float(closes[begin])
    end_price = float(closes[end - 1])
    if not np.isfinite(start_price) or start_price <= 0:
        return None
    if not np.isfinite(end_price) or end_price <= 0:
        return None
    return end_price / start_price - 1.0


def score_universe(universe, symbols, day, formation: str = "mom12_1",
                   sessions_per_month: int = SESSIONS_PER_MONTH) -> dict:
    """
    `{symbol: formation return}` for everything that can be scored.

    A symbol with too little history is OMITTED rather than scored zero.
    Scoring it zero would place it in the middle of the cross-section, which
    is a decision disguised as a default - and on a wide universe, newly
    listed names are numerous enough for that to move the ranking.
    """
    lookback, skip = FORMATIONS.get(formation, FORMATIONS["mom12_1"])
    out = {}
    for symbol in symbols:
        r = formation_return(universe.closes_before(symbol, day),
                             lookback, skip, sessions_per_month)
        if r is not None:
            out[symbol] = r
    return out


def random_scores(symbols, rng) -> dict:
    """
    THE NULL. Scores drawn from noise, same shape as a real signal.

    Everything downstream - the band, the top-N cut, sizing, costs, turnover,
    the rebalance schedule - is identical, so the difference between this and
    a momentum run is the signal and nothing else. Given that 82% of published
    anomalies fail a multiple-testing hurdle, this is the honest default
    expectation rather than a formality.
    """
    return {s: float(rng.normal()) for s in symbols}


def top_n(scores: dict, n: int) -> list:
    """
    The n highest scorers, ties broken by symbol so a run is reproducible.

    Deterministic tie-breaking matters more than it looks: on a wide universe
    with rounded prices, ties are common, and dict-order tie-breaking makes a
    result depend on the order symbols happened to be fetched in.
    """
    if n <= 0 or not scores:
        return []
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [s for s, _ in ordered[:n]]


__all__ = ["formation_return", "score_universe", "random_scores", "top_n",
           "FORMATIONS", "SESSIONS_PER_MONTH"]
