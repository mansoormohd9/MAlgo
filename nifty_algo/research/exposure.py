"""
The measurements both portfolio briefings need.

The macro assessment asks "what is this book exposed to"; the risk framework
asks "what happens when those things move together". They are two questions
about one calculation, so the calculation lives here and neither module owns
it - the same reason `scanner.evaluate_symbol` was extracted so the backtest
and the live scan could not drift apart.

EVERY FIGURE CARRIES ITS `n`, AND A SHORT SAMPLE RETURNS NOTHING. Two names
that share twelve sessions will produce a confident 0.9, and a correlation
without a sample size beside it is a decoration. Below
`PortfolioConfig.min_correlation_sessions` these functions return None with a
reason rather than a number - which forces the caller to say "not enough
history" instead of printing a figure it would have to caveat in prose.
"""
from __future__ import annotations

import pandas as pd

from ..swing.universe import UniverseError, load_universe
from . import holdings_prices

#: The factors a position is scored against, and the order they read in.
FACTORS = (
    ("nifty", "Nifty 50"),
    ("us_10y", "US 10-year yield"),
    ("usdinr", "USD/INR"),
    ("crude", "Crude"),
)


def sector_map(market) -> dict[str, str]:
    """Symbol to sector, from the committed universe. Empty if it is missing."""
    try:
        return {s.symbol.upper(): s.sector
                for s in load_universe(market.universe_csv)}
    except UniverseError:
        return {}


def sensitivities(symbol: str, stock_returns, moves, cfg,
                  factors=FACTORS) -> dict:
    """
    One position's beta and factor correlations, or nulls with a reason.

    Returns nulls rather than omitting keys, so every row of a table has the
    same shape and a missing value is visibly missing rather than absent.

    Beta is computed against the benchmark factor only. The other columns are
    CORRELATIONS, not betas: a correlation of -0.3 to the 10-year says these
    two moved opposite each other about a third of the time, and calling that
    a beta would imply a magnitude the number does not carry.
    """
    blank = {"nifty_beta": None, "sessions": 0, "sensitivity_note": ""}
    blank.update({f"corr_{key}": None for key, _ in factors})

    if symbol not in getattr(stock_returns, "columns", ()):
        blank["sensitivity_note"] = "no daily bars for this symbol"
        return blank
    if moves is None or getattr(moves, "empty", True):
        blank["sensitivity_note"] = "no macro history to correlate against"
        return blank

    series = stock_returns[symbol].dropna()
    out = dict(blank)
    for key, _label in factors:
        if key not in moves.columns:
            continue
        left, right, n = holdings_prices.align(series, moves[key].dropna())
        if n < cfg.portfolio.min_correlation_sessions:
            out["sensitivity_note"] = (
                f"only {n} overlapping sessions, below the "
                f"{cfg.portfolio.min_correlation_sessions} this build will "
                f"report a correlation on")
            continue
        out["sessions"] = max(out["sessions"], n)
        correlation = left.corr(right)
        if correlation == correlation:                     # not NaN
            out[f"corr_{key}"] = round(float(correlation), 3)
        if key == "nifty":
            variance = right.var()
            if variance and variance == variance:
                out["nifty_beta"] = round(float(left.cov(right) / variance), 3)
    return out


def pairwise(stock_returns: pd.DataFrame, cfg) -> list[dict]:
    """
    Every pair of holdings, most correlated first.

    Pairs rather than a square matrix because a matrix is half redundant and
    a briefing quotes pairs. The diagonal is meaningless and omitted.
    """
    symbols = list(getattr(stock_returns, "columns", ()))
    rows: list[dict] = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            left, right, n = holdings_prices.align(
                stock_returns[a].dropna(), stock_returns[b].dropna())
            if n < cfg.portfolio.min_correlation_sessions:
                rows.append({"a": a, "b": b, "correlation": None,
                             "sessions": n,
                             "note": f"only {n} overlapping sessions"})
                continue
            value = left.corr(right)
            rows.append({"a": a, "b": b, "sessions": n, "note": "",
                         "correlation": (round(float(value), 3)
                                         if value == value else None)})
    return sorted(rows, key=lambda r: -(r["correlation"]
                                        if r["correlation"] is not None else -2))


def average_pairwise(rows: list[dict]) -> tuple[float | None, int]:
    """
    Mean pairwise correlation, and how many pairs it averages.

    THE diversification number. Six positions at an average 0.75 is one
    position with extra brokerage; the count of names says nothing about it.
    """
    values = [r["correlation"] for r in rows if r["correlation"] is not None]
    if not values:
        return None, 0
    return round(sum(values) / len(values), 3), len(values)


def portfolio_returns(stock_returns: pd.DataFrame,
                      weights: dict[str, float]) -> pd.Series:
    """
    The book's own daily return series, at TODAY's weights.

    A backward-looking replay of the portfolio you hold now, not of the one
    you held then - which is the only version that can be computed without a
    position history, and is a real limitation rather than a detail. It
    answers "how would this book have behaved", never "how did it behave".
    """
    columns = [s for s in weights if s in getattr(stock_returns, "columns", ())]
    if not columns:
        return pd.Series(dtype=float)
    row_weights = pd.Series({s: float(weights[s]) for s in columns})
    if row_weights.sum() <= 0:
        return pd.Series(dtype=float)

    frame = stock_returns[columns]
    # RENORMALISED PER SESSION, not filled with zeroes. A holding that was
    # not listed for the whole window has no bar on the early sessions, and
    # treating that as a 0% return AT FULL WEIGHT drags every one of those
    # days toward zero. It does not fail - it quietly damps the volatility,
    # the percentiles and the worst session, in the reassuring direction,
    # by exactly the weight of whatever is missing. So each day is the
    # weighted mean of the names that actually traded that day, which is the
    # same rule the replayed drawdown uses for its coverage.
    present = frame.notna()
    covered = present.mul(row_weights, axis=1).sum(axis=1)
    weighted = frame.fillna(0.0).mul(row_weights, axis=1).sum(axis=1)
    series = (weighted / covered)[covered > 0]
    return series.dropna()
