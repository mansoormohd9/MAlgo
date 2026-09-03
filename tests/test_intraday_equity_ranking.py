"""
The precomputed daily history, and the safety property it must not break.

`morning_ranks` used to rebuild every symbol's daily aggregates from the whole
intraday history on every single session - 40% of the backtest spent
recomputing yesterday's answer. `DailyHistory` computes them once and serves
row `k-1`.

That is only legitimate because everything it serves is PREFIX-STABLE, and
"prefix-stable" is exactly the kind of claim that is easy to argue and wrong:
an indicator that peeks one row forward still returns a plausible float. So
the first test here checks it NUMERICALLY, over every session, against the
truncate-then-aggregate path it replaced.

The second thing these guard is trap T1 (`ranking.py`): ranking the universe
from the session being traded is a perfect stock picker and produces a
spectacular curve with no error anywhere.
"""
from __future__ import annotations

from dataclasses import astuple
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.intraday_equity import backtest as bt
from nifty_algo.intraday_equity import ranking

from test_intraday_equity_backtest import _calm, _sessions


@pytest.fixture
def cfg():
    c = Config()
    c.capital.intraday_equity_capital_inr = 100_000.0
    return c


@pytest.fixture
def world():
    """Five names and a benchmark, 40 calm sessions each."""
    bars = {}
    for i in range(5):
        frames, _, _ = _sessions(40, price=500.0 + 130 * i, seed=i + 11)
        bars[f"SYM{i:02d}"] = pd.concat(frames)
    frames, _, _ = _sessions(40, price=21_000.0, seed=404)
    bench = pd.concat(frames)
    bench["volume"] = 0.0                     # the index has NO volume
    return bars, bench


def _sessions_of(frame):
    return sorted({ts.date() for ts in frame.index})


# ------------------------------------------------- prefix stability


def test_the_daily_history_serves_exactly_what_truncating_first_returns(world):
    """
    THE LOAD-BEARING TEST. Exhaustive over sessions, not sampled.

    For every session, the ATR and ADV served from the precomputed arrays
    must equal what aggregating only the prior bars produces. An indicator
    that reached one row forward would pass an eyeball check and shift every
    rank by a fraction.
    """
    bars, bench = world
    hist = ranking.DailyHistory(bars, bench)

    for symbol, frame in bars.items():
        sd = hist.symbols[symbol]
        for day in _sessions_of(frame):
            k = hist.prior_count(sd.dates, day)
            if k < 2:
                continue

            prior = frame[frame.index.date < day]        # the OLD path
            daily = ranking._daily_frame(prior)
            closes = daily["close"]
            pc = closes.shift(1)
            tr = np.maximum(daily["high"] - daily["low"],
                            np.maximum((daily["high"] - pc).abs(),
                                       (daily["low"] - pc).abs()))
            want_atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
            want_adv = float(daily["turnover"].tail(20).mean())

            assert len(daily) == k
            assert float(sd.atr[k - 1]) == want_atr
            assert float(sd.turnover[max(0, k - 20):k].mean()) == want_adv
            assert float(sd.closes.iloc[:k].iloc[-1]) == float(closes.iloc[-1])


def test_ranks_are_identical_with_and_without_a_prebuilt_history(cfg, world):
    """
    One code path, not a fast branch and a slow branch.

    `morning_ranks` builds its own `DailyHistory` when none is passed. If the
    two ever diverged, the backtest and the live runner would rank
    differently while both looked correct.
    """
    bars, bench = world
    hist = ranking.DailyHistory(bars, bench)
    day = _sessions_of(next(iter(bars.values())))[-1]

    supplied = ranking.morning_ranks(bars, bench, day, cfg, history=hist)
    built = ranking.morning_ranks(bars, bench, day, cfg)

    assert [astuple(r) for r in supplied] == [astuple(r) for r in built]
    assert supplied, "fixture produced no ranks - the test would prove nothing"


# ------------------------------------------------- trap T1


def test_morning_ranks_cannot_see_the_session_it_ranks(cfg, world):
    """
    Trap T1. Deleting the ranked session and everything after it must change
    nothing, because the rank is supposed to be what a live scanner has at
    09:15 - before a single bar of `day` exists.

    Sorting the universe by the day's own move and trading the top three is a
    perfect intraday stock picker. It produces numbers, not an exception.
    """
    bars, bench = world
    days = _sessions_of(next(iter(bars.values())))
    day = days[-1]

    full = ranking.morning_ranks(bars, bench, day, cfg)
    truncated = ranking.morning_ranks(
        {s: f[f.index.date < day] for s, f in bars.items()},
        bench[bench.index.date < day], day, cfg)

    assert full, "fixture produced no ranks"
    assert [astuple(r) for r in full] == [astuple(r) for r in truncated]


def test_a_later_session_does_not_change_an_earlier_rank(cfg, world):
    """The same property one day earlier, so it is not an end-of-data artefact."""
    bars, bench = world
    days = _sessions_of(next(iter(bars.values())))
    day = days[-5]

    full = ranking.morning_ranks(bars, bench, day, cfg)
    cut = ranking.morning_ranks(
        {s: f[f.index.date <= day] for s, f in bars.items()},
        bench[bench.index.date <= day], day, cfg)

    assert [astuple(r) for r in full] == [astuple(r) for r in cut]


# ------------------------------------------------- the session index


def test_a_session_index_slice_equals_the_boolean_mask(world):
    """
    `_SessionIndex` replaced three `frame.index.date == day` scans per symbol
    per session. Exhaustive over every session of every symbol.
    """
    bars, _ = world
    for symbol, frame in bars.items():
        idx = bt._SessionIndex(frame)
        for day in _sessions_of(frame):
            got = idx.session(frame, day)
            want = frame[frame.index.date == day]
            assert got.index.equals(want.index)
            assert got.equals(want)

            prior = frame[frame.index.date < day]
            prev = idx.previous(frame, day)
            if prior.empty:
                assert prev is None
            else:
                want_prev = prior[prior.index.date == prior.index[-1].date()]
                assert prev.index.equals(want_prev.index)


def test_the_first_session_has_no_previous(world):
    bars, _ = world
    frame = next(iter(bars.values()))
    idx = bt._SessionIndex(frame)
    assert idx.previous(frame, _sessions_of(frame)[0]) is None


def test_a_day_the_symbol_did_not_trade_returns_none(world):
    """
    A halted or newly listed name has no bars that day. The boolean mask
    returned an empty frame; the index must not return the WRONG session.
    """
    bars, _ = world
    frame = next(iter(bars.values()))
    idx = bt._SessionIndex(frame)
    days = _sessions_of(frame)

    absent = days[3] + timedelta(days=1)
    while absent in days:
        absent += timedelta(days=1)

    assert idx.session(frame, absent) is None
    # ...but `previous` must still find the last session before it.
    prev = idx.previous(frame, absent)
    assert prev is not None
    assert prev.index[-1].date() < absent


def test_a_symbol_absent_from_the_history_is_skipped_not_guessed(cfg, world):
    """
    A name with no daily rows must drop out of the ranking rather than
    resolve to another symbol's aggregates.
    """
    bars, bench = world
    hist = ranking.DailyHistory(bars, bench)
    assert hist.symbols.get("NOTREAL") is None

    day = _sessions_of(next(iter(bars.values())))[-1]
    ranks = ranking.morning_ranks(
        dict(bars, NOTREAL=pd.DataFrame()), bench, day, cfg, history=hist)
    assert all(r.symbol != "NOTREAL" for r in ranks)
