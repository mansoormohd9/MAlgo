"""
The morning the option book could not see, and the landmine that guarded it.

`Context.bars` was one session, so a 30-bar indicator warm-up was spent INSIDE
the trading day: the first decidable bar was 11:45 and `entry_start = 09:30`
was dead config. The book was structurally blind to the open, the opening-range
break and the morning trend - the most volatile hours, and the ones an option
BUYER needs, being long vega and short theta.

The obvious fix - prepend prior-day bars - was explicitly forbidden in
`config.py`, and for a real reason: `signals.opening_range` read the first bars
OF THE FRAME with no day-awareness, so a warm-up window would have made it
return a finished day's range with no error anywhere. These tests cover both
halves: that the session-scoped reads are now anchored to today, and that the
warm-up is off by default and does what it claims when switched on.
"""
from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from nifty_algo import signals as sig
from nifty_algo.backtest import Backtester
from nifty_algo.config import Config
from nifty_algo.data.base import DataFeed
from nifty_algo.data.csv_feed import CsvFeed


def _frame(days: int = 3, bars_per_day: int = 40, start_price: float = 100.0):
    """`days` sessions, each with a deliberately distinct price band."""
    rows, idx = [], []
    for d in range(days):
        base = start_price + d * 1000.0        # days cannot be confused
        stamp = pd.Timestamp(f"2026-08-{17 + d} 09:15")
        for b in range(bars_per_day):
            px = base + b
            rows.append({"open": px, "high": px + 2, "low": px - 2,
                         "close": px + 1, "volume": 1000})
            idx.append(stamp + pd.Timedelta(minutes=5 * b))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


# --------------------------------------------------------- the anchored reads

def test_last_session_returns_only_the_final_day():
    df = _frame(days=3, bars_per_day=40)
    s = sig.last_session(df)
    assert len(s) == 40
    assert len({d for d in s.index.date}) == 1
    assert s.index.date[0] == df.index.date[-1]


def test_opening_range_on_a_multi_day_frame_is_todays_range():
    """
    THE REGRESSION FOR THE DOCUMENTED LANDMINE.

    `config.py` carried an explicit warning never to prepend prior-day bars,
    because `opening_range` was positional `.iloc[:3]`. Day 3's band is
    2100-2140; day 1's is 100-140. The un-anchored version returns day 1's,
    silently, and every ORB and gap strategy then trades a range from a day
    that is already over.
    """
    df = _frame(days=3, bars_per_day=40)
    hi, lo = sig.opening_range(df, minutes=15, bar_minutes=5)
    assert 2090 <= lo <= 2150 and 2090 <= hi <= 2150, (hi, lo)
    assert lo > 1000, "day 1's range (98-104) means the read was not anchored"

    only_today = df[df.index.date == df.index.date[-1]]
    assert (hi, lo) == sig.opening_range(only_today, 15, 5), (
        "a multi-day frame must give the same answer as the day alone"
    )


def test_session_open_on_a_multi_day_frame_is_todays_open():
    df = _frame(days=3, bars_per_day=40)
    assert sig.session_open(df) == pytest.approx(2100.0)
    assert sig.session_open(df.iloc[:0]) is None


def test_the_anchored_reads_are_unchanged_on_a_single_session_frame():
    """The whole point: today's callers must see no difference at all."""
    one = _frame(days=1, bars_per_day=40)
    assert sig.last_session(one).equals(one)
    assert sig.session_open(one) == float(one["open"].iloc[0])
    assert sig.opening_range(one, 15, 5) == (
        float(one.iloc[:3]["high"].max()), float(one.iloc[:3]["low"].min()))


# ------------------------------------------------------------- the warm-up

def test_zero_warmup_is_exactly_the_old_session_slice():
    df = _frame(days=3)
    assert DataFeed.session_slice_with_warmup(df, 0).equals(
        DataFeed.session_slice(df))


def test_warmup_returns_the_requested_number_of_prior_sessions():
    df = _frame(days=4, bars_per_day=40)
    assert len({d for d in DataFeed.session_slice_with_warmup(df, 2).index.date}) == 3
    assert len({d for d in DataFeed.session_slice_with_warmup(df, 1).index.date}) == 2
    # more warm-up than history is not an error, it is all the history
    assert len({d for d in DataFeed.session_slice_with_warmup(df, 99).index.date}) == 4


def test_warmup_never_includes_a_day_after_the_target():
    df = _frame(days=4, bars_per_day=40)
    target = sorted({d for d in df.index.date})[1]
    out = DataFeed.session_slice_with_warmup(df, 2, day=target)
    assert max(out.index.date) == target, "warm-up must not leak the future"


# ------------------------------------------------- what it does to the book

@pytest.fixture(scope="module")
def sample():
    return CsvFeed("data/sample_nifty_5m.csv").get_bars(lookback_days=0)


def test_without_warmup_the_book_cannot_trade_before_1145(sample):
    """
    The defect, pinned. `entry_start` is 09:30 and yet nothing fires for two
    and a half hours, because `BarReplayer(warmup=30)` starts the day at bar
    30 - 09:15 + 150 minutes.
    """
    cfg = Config()
    cfg.session.warmup_sessions = 0
    res = Backtester(cfg).run(sample)
    assert res.trades, "the fixture must produce trades or it proves nothing"
    assert min(t.entry_time.time() for t in res.trades) >= time(11, 45)
    assert cfg.session.entry_start == time(9, 30), (
        "and entry_start is dead config while that is true"
    )


def test_with_warmup_the_morning_becomes_reachable(sample):
    cfg = Config()
    cfg.session.warmup_sessions = 2
    res = Backtester(cfg).run(sample)
    assert res.trades
    earliest = min(t.entry_time.time() for t in res.trades)
    assert earliest < time(11, 45), f"still blind to the morning: {earliest}"
    assert earliest >= cfg.session.entry_start, "entry_start must now bind"


def test_warmup_is_off_by_default():
    """
    Every result this project has produced was produced at 0. Turning it on
    changes what the book trades, so it is a pre-registered variant rather
    than a silent improvement.
    """
    assert Config().session.warmup_sessions == 0


def test_a_warmed_frame_still_ends_at_the_decision_bar(sample):
    """
    Warm-up adds history to the LEFT edge only. If it ever reached past the
    decision bar, every backtest number would become fiction - this is the
    same property `test_lookahead.py` guards for the replayer.
    """
    days = sorted({d for d in sample.index.date})
    target = days[3]
    frame = DataFeed.session_slice_with_warmup(sample, 2, day=target)
    session_start = int((frame.index.date < target).sum())
    assert session_start > 0, "the fixture needs prior sessions to be a test"
    assert frame.index.date[session_start] == target
    assert max(frame.index.date) == target


# ------------------------------------------- the fabricated prior session

def test_a_fabricated_prior_session_is_flagged():
    """
    `prior_session` invents a prior close from the current day's first bar
    when the frame has no prior day. The values are still returned - a
    same-day file must not crash the engine - but they must not read as real.
    """
    one_day = _frame(days=1, bars_per_day=40)
    p = DataFeed.prior_session(one_day)
    assert p.is_real is False
    assert p.close == float(one_day["close"].iloc[0]), "the fallback still returns"

    real = DataFeed.prior_session(_frame(days=2, bars_per_day=40))
    assert real.is_real is True


def test_the_backtester_skips_a_day_with_no_real_prior_session(sample):
    """
    Day 1 of every window had `prev_day_close` set to the close of its own
    first 5-minute bar, so `gap_metrics` computed open[0] - close[0]. That
    day could never be GAP_DAY whatever the index did overnight, and the
    gap strategy rejected it as "too small" every time.
    """
    cfg = Config()
    bt = Backtester(cfg)
    res = bt.run(sample)
    first_day = sorted({d for d in sample.index.date})[0]

    assert first_day in bt._skipped_days
    assert all(t.entry_time.date() != first_day for t in res.trades)
    assert any("fabricated prior close" in w for w in res.warnings), (
        "a shorter sample must be stated, not silently taken"
    )
