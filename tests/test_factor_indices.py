"""
The index adapter, and the two claims in it that could be wrong quietly.

Everything here is engineered rather than sampled from the real cache: the
parquet is gitignored and a test that needs a 60 MB download is a test nobody
runs. The two that matter are `test_a_pair_uses_only_shared_sessions` and
`test_month_end_marks_hide_a_trough_that_daily_data_sees` - both failures
produce a complete, plausible, wrong answer rather than an exception.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nifty_algo.factor import drawdown as fd
from nifty_algo.factor import indices as fi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _frame(start: date, values: list, name: str = "X") -> pd.DataFrame:
    """One index, one business day per value."""
    days = pd.bdate_range(start=pd.Timestamp(start), periods=len(values))
    return pd.DataFrame({"index_name": name, "date": days,
                         "tri": np.asarray(values, dtype=float),
                         "ntr": np.nan})


def _ramp(start: date, n: int, monthly: float, name: str) -> pd.DataFrame:
    step = (1.0 + monthly) ** (1 / 21.0)
    return _frame(start, [1000.0 * step ** i for i in range(n)], name)


# --------------------------------------------------------------- to_result

def test_to_result_rebases_to_capital_and_marks_month_ends():
    frame = _ramp(date(2010, 1, 4), 200, 0.01, "X")
    res = fi.to_result(frame, capital=100.0)
    assert res.equity[0][1] == pytest.approx(100.0)
    # One mark per calendar month touched, and every mark is a real session.
    sessions = {d.date() for d in frame["date"]}
    marks = [d for d, _ in res.equity]
    assert all(m in sessions for m in marks)
    assert len(marks) == len({(m.year, m.month) for m in marks})


def test_a_series_too_short_to_mark_is_an_empty_result_not_a_crash():
    assert fi.to_result(_frame(date(2010, 1, 4), [1000.0]), 100.0).equity == []


# ------------------------------------------------------------------- pair

def test_a_pair_uses_only_shared_sessions():
    """
    A session present in one index and not the other must not enter the test.

    Left alone this does not raise - it silently marks one side on a date the
    other never traded, and every statistic downstream is computed on a
    mismatched grid.
    """
    child = _ramp(date(2010, 1, 4), 120, 0.01, "Child")
    parent = _ramp(date(2010, 1, 4), 120, 0.01, "Parent")
    # Give the parent an extra session the child never had.
    extra = parent.iloc[[-1]].copy()
    extra["date"] = extra["date"] + timedelta(days=40)
    parent = pd.concat([parent, extra], ignore_index=True)

    p = fi.pair({"Child": child, "Parent": parent}, "Child", "Parent")
    shared = {d.date() for d in child["date"]} & {d.date() for d in parent["date"]}
    assert set(p.marks) <= shared
    assert len(p.constrained.equity) == len(p.parent.equity) == len(p.marks)


def test_a_pair_starts_both_sides_on_the_same_date_and_capital():
    child = _ramp(date(2010, 1, 4), 300, 0.01, "Child")
    parent = _ramp(date(2005, 1, 3), 1500, 0.01, "Parent")   # far longer
    p = fi.pair({"Child": child, "Parent": parent}, "Child", "Parent",
                capital=100.0)
    assert p.constrained.equity[0][0] == p.parent.equity[0][0]
    assert p.constrained.equity[0][1] == pytest.approx(100.0)
    assert p.parent.equity[0][1] == pytest.approx(100.0)
    assert p.start >= date(2010, 1, 4)


def test_an_unknown_index_name_raises_and_says_what_is_available():
    with pytest.raises(KeyError, match="Parent"):
        fi.pair({"Parent": _ramp(date(2010, 1, 4), 60, 0.01, "Parent")},
                "Nifty500 Shariah", "Parent")


def test_pair_respects_an_explicit_window():
    child = _ramp(date(2010, 1, 4), 800, 0.01, "Child")
    parent = _ramp(date(2010, 1, 4), 800, 0.01, "Parent")
    p = fi.pair({"Child": child, "Parent": parent}, "Child", "Parent",
                start=date(2011, 1, 1), end=date(2012, 1, 1))
    assert p.start >= date(2011, 1, 1)
    assert p.end <= date(2012, 1, 1)


# ------------------------------------------------------- sign conventions

def test_consistency_delta_is_constrained_minus_parent():
    """
    The report reads `wins` as "months the CONSTRAINED index won".

    `consistency(a, b)` differences b minus a, so the arguments must go in
    (parent, constrained). Swapped, every sign in the report inverts and the
    verdict reverses - with no error anywhere.
    """
    parent = _ramp(date(2010, 1, 4), 400, 0.005, "Parent")
    child = _ramp(date(2010, 1, 4), 400, 0.015, "Child")     # strictly better
    p = fi.pair({"Child": child, "Parent": parent}, "Child", "Parent")
    c = fd.consistency(p.parent, p.constrained)
    assert c["wins"] == c["scored"]           # the better index wins every month
    assert c["mean_pp"] > 0
    assert c["compounded_pct"] > 0


def test_excess_sharpe_subtracts_the_rate_and_the_builtin_does_not():
    frame = _ramp(date(2010, 1, 4), 900, 0.01, "X")
    res = fi.to_result(frame, 100.0)
    assert fi.excess_sharpe(res, rf=0.0) == pytest.approx(res.sharpe(), rel=1e-9)
    assert fi.excess_sharpe(res, rf=0.065) < fi.excess_sharpe(res, rf=0.0)


# ------------------------------------------------- the drawdown-grid claim

def test_month_end_marks_hide_a_trough_that_daily_data_sees():
    """
    The claim `daily_drawdown`'s docstring makes, on data built to expose it.

    A crash that begins and fully recovers inside one calendar month is
    invisible to a month-end grid. This is not a corner case - it is the
    reason `drawdown.DRAWDOWN_HAIRCUT` exists, and here it can be measured
    instead of estimated.
    """
    values = [1000.0] * 25 + [400.0] + [1000.0] * 25      # one-session crash
    frame = _frame(date(2010, 1, 4), values)
    marked = fi.to_result(frame, 100.0).max_drawdown()
    daily_dd, _ = fi.daily_drawdown(frame)
    assert daily_dd == pytest.approx(-0.60)
    assert marked == pytest.approx(0.0)
    assert daily_dd < marked


def test_daily_drawdown_honours_the_window_it_is_given():
    """The dot-com fall is outside the Shariah window and must stay outside."""
    values = ([1000.0, 300.0] + [1000.0] * 40            # an early collapse
              + [1200.0, 900.0] + [1200.0] * 40)         # a milder later one
    frame = _frame(date(2000, 1, 3), values)
    whole, _ = fi.daily_drawdown(frame)
    later, _ = fi.daily_drawdown(
        frame, start=frame["date"].iloc[42].date())
    assert whole == pytest.approx(-0.70)
    assert later == pytest.approx(-0.25)


# ------------------------------------------------------------------- load

def test_a_missing_cache_raises_rather_than_reporting_no_indices(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_nifty_tri"):
        fi.load(tmp_path / "absent.parquet")


def test_load_round_trips_a_written_cache(tmp_path):
    path = tmp_path / "tri.parquet"
    pd.concat([_ramp(date(2010, 1, 4), 40, 0.01, "A"),
               _ramp(date(2010, 1, 4), 40, 0.02, "B")]).to_parquet(path)
    got = fi.load(path)
    assert sorted(got) == ["A", "B"]
    assert len(got["A"]) == 40


# ---------------------------------------------------------- rolling excess

def test_rolling_excess_is_annualised_and_signed_the_right_way():
    parent = _ramp(date(2005, 1, 3), 2000, 0.005, "Parent")
    child = _ramp(date(2005, 1, 3), 2000, 0.010, "Child")
    p = fi.pair({"Child": child, "Parent": parent}, "Child", "Parent")
    roll = fi.rolling_excess(p, window_months=60)
    assert not roll.empty
    assert (roll["excess_pp"] > 0).all()
    assert roll["constrained_cagr"].gt(roll["parent_cagr"]).all()


def test_rolling_excess_is_empty_rather_than_wrong_on_a_short_window():
    parent = _ramp(date(2020, 1, 3), 100, 0.005, "Parent")
    child = _ramp(date(2020, 1, 3), 100, 0.010, "Child")
    p = fi.pair({"Child": child, "Parent": parent}, "Child", "Parent")
    assert fi.rolling_excess(p, window_months=60).empty


# ------------------------------------------------------ the fetcher guards

def test_the_fetcher_rejects_the_sites_html_shell():
    """
    The dead `Backpage.aspx` endpoint answers 200 with HTML.

    A status-code check passes that; only the body catches it, which is the
    same guard `swing/universe.refresh_from_nse` already applies to NSE. The
    message has to name the likely cause - "Expecting value: line 1 column 1"
    reads like a corrupt download rather than like a moved endpoint.
    """
    import fetch_nifty_tri as fnt
    with pytest.raises(ValueError, match="not JSON"):
        fnt._decode(b"<!DOCTYPE html><html><body>NIFTY</body></html>",
                    "Nifty 50")


def test_the_fetcher_rejects_json_that_is_not_a_list_of_rows():
    import fetch_nifty_tri as fnt
    with pytest.raises(ValueError, match="expected a list"):
        fnt._rows({"d": "<html>"}, "Nifty 50")


def test_the_fetcher_rejects_rows_without_the_tri_column():
    import fetch_nifty_tri as fnt
    with pytest.raises(ValueError, match="no TotalReturnsIndex"):
        fnt._rows([{"Date": "01 Jan 2024", "CLOSE": "1"}], "Nifty 50")


def test_the_fetcher_accepts_a_well_formed_response():
    import fetch_nifty_tri as fnt
    rows = [{"Date": "01 Jan 2024", "TotalReturnsIndex": "1", "NTR_Value": "1"}]
    assert fnt._rows(rows, "Nifty 50") == rows


def test_chunked_spans_cover_the_range_without_overlapping():
    """`--chunk-years` is the fallback if the server ever enforces the 1-year
    limit its own JavaScript already applies. Untested, it would silently
    drop or double-count sessions at every boundary."""
    import fetch_nifty_tri as fnt
    spans = list(fnt._spans(date(2010, 3, 5), date(2013, 7, 2), 1))
    assert spans[0][0] == date(2010, 3, 5)
    assert spans[-1][1] == date(2013, 7, 2)
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert lo == hi + timedelta(days=1)
    assert all((hi - lo).days <= 366 for lo, hi in spans)
