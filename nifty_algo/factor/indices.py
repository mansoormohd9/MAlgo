"""
Published index series, adapted into the harness the sleeve already uses.

THE POINT OF THIS MODULE IS THAT IT ADDS NO STATISTICS. Every number the
Stage 0 report prints comes from `FactorResult.cagr`, `.max_drawdown`,
`.longest_drawdown_months`, `drawdown._annual_vol` and `drawdown.consistency` -
the same code that produced F1 through F5. All this file does is turn a
published index into the one shape those functions already accept: a
`FactorResult` whose `equity` is a list of `(date, value)` month-end marks.

That is the whole trick, and it is deliberate. A "compare two indices" helper
with its own CAGR and its own drawdown would be a third implementation of
arithmetic this repo has already got wrong once (`consistency`'s wealth ratio),
and the first time the two disagreed the question would be which to believe.

THE MARK GRID IS MONTH-ENDS, VIA `universe.month_ends`. The sleeve is marked at
month-ends, `consistency` calls its per-mark differences "months", and the sign
test counts them - so a daily grid here would be comparing a 4,877-observation
sign test against the sleeve's 121 and calling both "the monthly sign test".
`month_ends` derives the dates from the sessions actually present rather than
from a calendar, so a holiday cannot invent a mark on which nothing traded.

GROSS TRI, NOT NET, AND NOT BY PREFERENCE. niftyindices publishes `NTR_Value`
for the parent indices only - it is absent for BOTH Shariah indices across the
entire history. There is therefore no net-of-withholding basis on which the
comparison can be made at all, and quietly using gross for one side and net for
the other would be a tax artefact wearing a compliance result's clothes. So
both sides are gross, and `TRI_BASIS_NOTE` says so wherever it surfaces.

WHAT A PAIR COMPARISON MUST DO, AND WHY `pair` EXISTS. The Shariah indices
start on their base date of 2006-12-29; their parents start in 1995 and 1999.
Comparing each on its own full history would compare a 20-year window against a
31-year one and report the difference as a compliance effect. `pair` intersects
the two session sets first and rebases BOTH to the same capital on the same
date, so the only thing left between them is the constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_algo.factor import universe as fu
from nifty_algo.factor.backtest import FactorResult
from nifty_algo.factor.drawdown import _recovery_months

#: Where `scripts/fetch_nifty_tri.py` writes.
DEFAULT_PATH = Path("data/cache/nifty_tri.parquet")

#: The four series Stage 0 asks for, and which is whose parent. Order matters
#: only for the report; the pairing is the fact.
PARENT_OF: dict[str, str] = {
    "Nifty50 Shariah": "Nifty 50",
    "Nifty500 Shariah": "Nifty 500",
}

TRI_BASIS_NOTE = (
    "GROSS TRI on both sides. niftyindices publishes NTR (net of dividend "
    "withholding) for the parent indices only - it is absent for both Shariah "
    "indices over the whole history - so there is no net basis on which this "
    "comparison could be made. Gross TRI overstates what a taxable Indian "
    "resident receives on BOTH sides, and it overstates it more on the side "
    "with the higher payout, which is the parent."
)

#: Stated rather than measured, and it is the user's figure for this study.
#: VERIFY against what the account actually earns on idle rupees - a liquid
#: fund and a sweep FD are not the same number, and the Sharpe is linear in it.
#: `drawdown.DEFAULT_RF` is the same 0.065 and is the number to keep in step.
DEFAULT_RF = 0.065


def load(path: Path | str = DEFAULT_PATH) -> dict:
    """
    `{index_name: DataFrame(date, tri, ntr)}`, sorted, from the parquet.

    Raises if the file is missing rather than returning `{}` - an empty
    mapping here would render as "no indices lag their parent".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            str(path) + " not found. Run:  python -m scripts.fetch_nifty_tri "
            "(or python scripts/fetch_nifty_tri.py)")
    frame = pd.read_parquet(path)
    out = {}
    for name, group in frame.groupby("index_name"):
        out[str(name)] = (group.sort_values("date")
                               .drop_duplicates(subset=["date"])
                               .reset_index(drop=True))
    return out


def _sessions(frame: pd.DataFrame) -> list:
    return [d.date() if hasattr(d, "date") else d for d in frame["date"]]


def to_result(frame: pd.DataFrame, capital: float, column: str = "tri",
              marks: list | None = None) -> FactorResult:
    """
    An index series as a `FactorResult`, marked month-end and rebased.

    `marks` lets two series share one grid - which is what makes a paired
    comparison a comparison rather than two separate measurements. When it is
    omitted the grid is this series' own month-ends.
    """
    frame = frame.dropna(subset=[column])
    if frame.empty:
        return FactorResult()
    by_day = dict(zip(_sessions(frame), frame[column].astype(float)))
    if marks is None:
        marks = fu.month_ends(list(by_day))
    picked = [(d, by_day[d]) for d in marks if d in by_day]
    if len(picked) < 2:
        return FactorResult()
    base = picked[0][1]
    result = FactorResult(
        equity=[(d, capital * v / base) for d, v in picked],
        start=picked[0][0], end=picked[-1][0])
    return result


@dataclass(frozen=True)
class Pair:
    """One constrained index against its parent, on one grid."""

    name: str
    parent_name: str
    constrained: FactorResult
    parent: FactorResult
    marks: list
    capital: float

    @property
    def start(self) -> date:
        return self.marks[0]

    @property
    def end(self) -> date:
        return self.marks[-1]

    @property
    def years(self) -> float:
        return (self.end - self.start).days / 365.25


def pair(series: dict, name: str, parent_name: str, capital: float = 100.0,
         column: str = "tri", start: date | None = None,
         end: date | None = None) -> Pair:
    """
    Both indices on the INTERSECTION of their sessions, rebased together.

    The intersection is load-bearing twice over. It stops a 20-year series
    being compared against a 31-year one, and it stops a session present in
    only one of them from entering the sign test as a one-sided month.
    """
    if name not in series:
        raise KeyError(name + " is not in the TRI cache (have: "
                       + ", ".join(sorted(series)) + ")")
    if parent_name not in series:
        raise KeyError(parent_name + " is not in the TRI cache")
    a, b = series[name], series[parent_name]
    common = sorted(set(_sessions(a.dropna(subset=[column])))
                    & set(_sessions(b.dropna(subset=[column]))))
    if start is not None:
        common = [d for d in common if d >= start]
    if end is not None:
        common = [d for d in common if d <= end]
    marks = fu.month_ends(common)
    return Pair(name=name, parent_name=parent_name,
                constrained=to_result(a, capital, column, marks),
                parent=to_result(b, capital, column, marks),
                marks=marks, capital=capital)


def excess_sharpe(result: FactorResult, rf: float = DEFAULT_RF) -> float:
    """
    Sharpe with the risk-free rate actually subtracted.

    `FactorResult.sharpe()` subtracts NOTHING - it is an excess-of-zero ratio,
    which is why every Sharpe recorded in F1 is roughly rf/vol too high. That
    convention is not changed here, because changing it would silently move
    every recorded number in the sleeve's table. This is a separate function so
    the two can appear side by side and be told apart.

    The cash leg is compounded over each interval's ACTUAL day count, the same
    way `drawdown._blended` does it, rather than assumed to be a calendar
    month - the marks are month-ENDS, and February to March is not 30.44 days.
    """
    series = result.series()
    if len(series) < 3:
        return float("nan")
    dates = [d for d, _ in result.equity]
    r = np.diff(series) / series[:-1]
    months = np.array([(b - a).days for a, b in zip(dates, dates[1:])]) / 30.44
    cash = (1.0 + rf) ** (months / 12.0) - 1.0
    excess = r - cash
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(excess)) / sd * float(np.sqrt(12.0))


def daily_drawdown(frame: pd.DataFrame, column: str = "tri",
                   start: date | None = None,
                   end: date | None = None) -> tuple:
    """
    Max drawdown and recovery on the DAILY series, not on month-end marks.

    THIS IS THE ONE PLACE THIS STUDY CAN DO BETTER THAN THE SLEEVE, AND THE
    DIFFERENCE IS NOT SMALL. `FactorResult.max_drawdown` sees only the marks it
    is given, and the sleeve only ever had month-ends - which is precisely why
    `drawdown.DRAWDOWN_HAIRCUT` exists to inflate a measured figure by 15%
    before anyone sizes off it. Here the daily closes are published, so the
    trough can be read instead of estimated.

    On the Stage 0 window marking hides 3-5pp of a parent's fall, and on the
    NIFTY 50 PAIR it does something worse than flatter: it REVERSES which side
    drew down less. Nifty50 Shariah is deeper than its parent on marks and
    shallower on daily closes, so the two grids disagree about which index was
    the safer one to have held. The Nifty 500 pair does not reverse.

    THE FIGURES LIVE IN `data/v0_shariah_index.txt`, NOT HERE. This docstring
    used to carry three and two of them were wrong: it claimed -67.0% daily
    against a real -63.7%, a 6.6pp gap against a real 3.3pp, and it pinned the
    reversal on the 500 pair when the reversal belongs to the 50 pair. A number
    in a docstring has no test behind it, so it rotted quietly while the report
    this module generates said something else. `test_factor_indices.py` now
    pins both surviving claims against that committed report.

    A drawdown comparison on marks is therefore not a conservative version of
    this one. It is a different answer.
    """
    frame = frame.dropna(subset=[column])
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    if len(frame) < 2:
        return float("nan"), float("nan")
    values = frame[column].to_numpy(dtype=float)
    dates = _sessions(frame)
    peak = np.maximum.accumulate(values)
    return (float(np.min(values / peak - 1.0)),
            _recovery_months(values, dates))


def rolling_excess(pair_: Pair, window_months: int = 60) -> pd.DataFrame:
    """
    Rolling annualised excess of the constrained index over its parent.

    Annualised over the window's own day count rather than over
    `window_months/12`, because the marks are month-ends and 60 of them do not
    span exactly five years.

    Returned as a frame rather than reduced to a headline on purpose: the whole
    reason to look at a rolling window is to see whether an average is one
    stretch, and a mean of the rolling series answers the same question the
    pooled CAGR already answered.
    """
    a = pair_.constrained.series()
    b = pair_.parent.series()
    dates = pair_.marks
    n = min(len(a), len(b), len(dates))
    rows = []
    for i in range(window_months, n):
        years = (dates[i] - dates[i - window_months]).days / 365.25
        if years <= 0:
            continue
        ca = (a[i] / a[i - window_months]) ** (1 / years) - 1.0
        cb = (b[i] / b[i - window_months]) ** (1 / years) - 1.0
        rows.append({"end": dates[i], "start": dates[i - window_months],
                     "constrained_cagr": ca, "parent_cagr": cb,
                     "excess_pp": (ca - cb) * 100.0})
    return pd.DataFrame(rows)


__all__ = ["DEFAULT_PATH", "PARENT_OF", "TRI_BASIS_NOTE", "DEFAULT_RF",
           "Pair", "load", "to_result", "pair", "excess_sharpe",
           "daily_drawdown", "rolling_excess"]
