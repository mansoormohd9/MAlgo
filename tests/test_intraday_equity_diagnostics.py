"""
The excursion instrumentation, and the diagnostics built on it.

Two separate things are guarded here.

FIRST, that `mfe_r` / `mae_r` are honest. They are recorded in `_manage`
alongside the exit decision, so a wiring error would produce a number that
looks entirely reasonable and quietly misdescribes what the book did - and
these are the numbers the whole diagnosis rests on. The structural invariants
below cannot all hold by accident.

SECOND, that recording them CHANGED NOTHING. The real guard for that is the
rest of `test_intraday_equity_backtest.py`, which asserts the fill model and
every exit rule and still passes unchanged - plus the full-scale equivalence
run, which must still return 2,162 trades at -0.524R. A diagnostic that moves
a trade is not a diagnostic, and the cheapest way to know is that the tests
which describe the decisions did not have to change.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.intraday_equity import backtest as bt
from nifty_algo.intraday_equity import diagnostics as dx

from test_intraday_equity_backtest import _history_with_breakouts, _sessions


@pytest.fixture
def cfg():
    c = Config()
    c.capital.intraday_equity_capital_inr = 100_000.0
    c.intraday_equity.mis_leverage = 5.0
    return c


@pytest.fixture
def result(cfg):
    bars = {}
    for i in range(6):
        bars[f"SYM{i:02d}"] = _history_with_breakouts(
            seed=i + 1, price=800.0 + 60 * i,
            follow=+1.0 if i % 2 == 0 else -1.0)
    frames, _, _ = _sessions(36, price=22_000.0, seed=999)
    bench = pd.concat(frames)
    bench["volume"] = 0.0
    cfg.intraday_equity.rs_prefilter_n = 6
    res = bt.run(cfg, bars, bench)
    assert res.trades, "fixture produced no trades - nothing would be proven"
    return res


def _trade(**kw):
    """A minimal finished trade, for the pure-function tests."""
    base = dict(
        entry_time=pd.Timestamp("2026-01-05 11:45"),
        exit_time=pd.Timestamp("2026-01-05 15:10"),
        strategy="level_break", direction="LONG", regime="trend",
        symbol="SYM", entry_underlying=100.0, exit_underlying=101.0,
        stop_points=1.0, outcome="force_exit", bars_held=10,
        r_multiple=0.0, reason="force_exit", mfe_r=0.0, mae_r=0.0)
    base.update(kw)
    return bt.IntradayTrade(**base)


# ------------------------------------------------- the instrumentation


def test_excursions_bracket_the_entry(result):
    """
    The fill is bar i+1's OPEN, and a bar's high is never below its open and
    its low never above it. So MFE >= 0 >= MAE for every trade, always.

    Fails if the excursions were measured against the wrong reference price -
    the signal close rather than the fill, say - which is exactly the sort of
    off-by-one that produces a plausible number.
    """
    for t in result.trades:
        assert t.mfe_r >= 0.0, t
        assert t.mae_r <= 0.0, t
        assert t.mae_r <= t.mfe_r


def test_a_target_exit_must_have_reached_the_target(cfg, result):
    """
    The ladder banks at +2R only when the bar's best_r got there, so any trade
    that exited on the target must show an MFE of at least +2R. This is the
    tightest link between the recorded excursion and an actual decision.
    """
    target = cfg.capital.reward_risk_ratio
    hit = [t for t in result.trades if (t.reason or t.outcome) == "target"]
    for t in hit:
        assert t.mfe_r >= target - 1e-9, t


def test_excursions_are_populated_rather_than_left_at_the_default(result):
    """Both fields default to 0.0, so an unwired field is invisible."""
    assert any(t.mfe_r > 0 for t in result.trades)
    assert any(t.mae_r < 0 for t in result.trades)


def test_a_stopped_trade_went_against_us(result):
    stopped = [t for t in result.trades if (t.reason or t.outcome) == "stop"]
    for t in stopped:
        assert t.mae_r < 0.0, t


# ------------------------------------------------- the pure functions


def test_the_exit_census_accounts_for_every_trade(result):
    assert sum(b.n for b in dx.exit_census(result.trades)) == len(result.trades)
    assert (pytest.approx(sum(b.total_r for b in dx.exit_census(result.trades)))
            == sum(t.r_multiple for t in result.trades))


def test_mfe_attainment_is_monotonically_non_increasing():
    """A trade reaching +2R has necessarily reached +1R."""
    trades = [_trade(mfe_r=x) for x in (0.2, 0.9, 1.2, 1.7, 2.4, 3.5)]
    got = dx.mfe_attainment(trades)
    shares = [got[r] for r in dx.DEFAULT_RUNGS]
    assert shares == sorted(shares, reverse=True)
    assert got[1.0] == pytest.approx(4 / 6)
    assert got[2.0] == pytest.approx(2 / 6)


def test_capture_ratio_separates_the_two_explanations():
    """
    The whole reason this module exists.

    Both books below lose the same total R. One was never offered anything;
    the other was offered plenty and handed it back. The headline cannot tell
    them apart and the capture ratio must.
    """
    never_came = [_trade(mfe_r=0.05, r_multiple=-1.0) for _ in range(10)]
    gave_it_back = [_trade(mfe_r=1.8, r_multiple=-1.0) for _ in range(10)]

    assert dx.capture_ratio(never_came) < -10        # tiny denominator
    assert -1.0 < dx.capture_ratio(gave_it_back) < 0
    assert dx.capture_ratio(never_came) < dx.capture_ratio(gave_it_back)


def test_capture_ratio_reports_zero_when_nothing_was_offered():
    """Not a crash, and not a misleading 1.0."""
    assert dx.capture_ratio([_trade(mfe_r=0.0, r_multiple=-1.0)]) == 0.0
    assert dx.capture_ratio([]) == 0.0


def test_squareoff_pessimism_counts_only_forced_exits():
    trades = [
        _trade(reason="force_exit", mfe_r=2.5, r_multiple=0.4),
        _trade(reason="force_exit", mfe_r=0.3, r_multiple=-0.2),
        _trade(reason="target", mfe_r=2.9, r_multiple=2.0),
        _trade(reason="stop", mfe_r=0.1, r_multiple=-1.0),
    ]
    got = dx.squareoff_pessimism(trades)
    assert got["force_exits"] == 2
    assert got["reached_target_anyway"] == 1
    assert got["share"] == pytest.approx(0.5)
    assert got["mean_r_of_those"] == pytest.approx(0.4)


def test_a_banked_runner_is_not_squareoff_pessimism():
    """
    The correction that took the real figure from 79 trades to 11.

    A position that touched +2R, BANKED its half there, and had the runner
    force-exited later is the ladder working exactly as designed. Counting it
    as "the square-off cost us the target" overstated the rule's cost
    sevenfold and would have argued for reordering a rule that is fine.
    """
    banked = _trade(reason="force_exit", mfe_r=2.6, r_multiple=0.9,
                    partial_banked=True)
    genuine = _trade(reason="force_exit", mfe_r=2.6, r_multiple=0.5,
                     partial_banked=False)

    assert dx.squareoff_pessimism([banked])["reached_target_anyway"] == 0
    assert dx.squareoff_pessimism([genuine])["reached_target_anyway"] == 1
    assert dx.squareoff_pessimism([banked, genuine])["force_exits"] == 2


def test_squareoff_pessimism_on_no_forced_exits_is_not_a_division_error():
    got = dx.squareoff_pessimism([_trade(reason="stop", mfe_r=0.1)])
    assert got["force_exits"] == 0 and got["share"] == 0.0


def test_the_time_census_is_ordered_by_clock_not_by_count():
    trades = ([_trade(entry_time=pd.Timestamp("2026-01-05 14:05"))] * 3
              + [_trade(entry_time=pd.Timestamp("2026-01-05 11:50"))] * 9)
    labels = [b.label for b in dx.time_census(trades)]
    assert labels == sorted(labels)
    assert labels == ["11:30", "14:00"]


def test_strategy_by_reason_totals_match_the_strategy_census():
    trades = [_trade(strategy="a", reason="stop"),
              _trade(strategy="a", reason="force_exit"),
              _trade(strategy="b", reason="stop")]
    cross = dx.strategy_by_reason(trades)
    per = {b.label: b.n for b in dx.strategy_census(trades)}
    for strat, by_reason in cross.items():
        assert sum(b.n for b in by_reason.values()) == per[strat]


def test_the_summary_renders_without_a_backtest(result):
    text = dx.summary(result.trades)
    for expected in ("exit census", "MFE attainment", "capture ratio",
                     "square-off pessimism", "entry time"):
        assert expected in text
    assert dx.summary([]) == "no trades"
