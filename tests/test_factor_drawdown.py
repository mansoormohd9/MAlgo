"""
F2a: the two active drawdown instruments, and the statistic that judges them.

The instruments live in `factor/backtest.run` and are OFF by default, which is
the property most of these tests exist to pin. `run` is imported by F1, by the
walk-forward and by the sweep, and every factor number this repo has recorded
was produced with them absent - a new lever that quietly moves the default
result invalidates all of them at once.

The last two tests are about the REPORT rather than the book, and they are here
because the calibration window taught the lesson the hard way: the CAGR and
drawdown table said every stop arm beat the matched-drawdown cash line, and the
month-by-month statistic said the whole advantage was three months.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.factor import backtest as fb
from nifty_algo.factor import drawdown as fd


def _series(start_price, n, drift, seed, start=date(2020, 1, 1), volume=1e6):
    rng = np.random.default_rng(seed)
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    steps = rng.normal(drift, 0.01, n)
    close = start_price * np.exp(np.cumsum(steps))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, volume)},
        index=pd.to_datetime(days))


@pytest.fixture
def world():
    return {f"WIN{i:02d}": _series(100.0 + i, n=700, drift=0.0012 - i * 0.0003,
                                   seed=i + 1, volume=1e6 * (i + 1))
            for i in range(12)}


@pytest.fixture
def bench(world):
    """A benchmark on the same calendar, rising then falling."""
    idx = world["WIN00"].index
    n = len(idx)
    ramp = np.concatenate([np.linspace(100.0, 200.0, n // 2),
                           np.linspace(200.0, 120.0, n - n // 2)])
    return pd.DataFrame({"close": ramp}, index=idx)


def _run(world, **kw):
    return fb.run(world, 500_000.0, top_n=4, min_turnover=0.0,
                  min_history=300, **kw)


# ------------------------------------------------- the no-op guarantee

def test_the_new_instruments_are_byte_identical_no_ops_when_off(world):
    """
    THE REGRESSION THAT PROTECTS EVERY F1 NUMBER. `stop_pct=None` and
    `regime_ma_days=0` must leave the book exactly as it was, down to the
    equity curve, the trade count and the rupee of charges.
    """
    a = _run(world)
    b = _run(world, stop_pct=None, stop_basis="rebalance", regime_ma_days=0)
    assert [v for _, v in a.equity] == [v for _, v in b.equity]
    assert a.trades == b.trades and a.costs_paid == b.costs_paid
    assert a.contribution == b.contribution
    assert (a.stops_fired, a.regime_flat) == (0, 0)


def test_a_stop_basis_is_ignored_entirely_when_no_stop_is_set(world):
    """`stop_basis` alone must not be able to change anything."""
    a = _run(world, stop_basis="entry")
    b = _run(world, stop_basis="rebalance")
    assert [v for _, v in a.equity] == [v for _, v in b.equity]


# ------------------------------------------------- the stop

def test_the_stop_actually_fires_and_moves_the_book(world):
    tight = _run(world, stop_pct=0.02, stop_basis="rebalance")
    assert tight.stops_fired > 0
    assert tight.trades > _run(world).trades


def test_the_rebalance_basis_fires_where_the_entry_basis_does_not():
    """
    The reason both bases are in the family.

    A name that doubles and then gives back 20% is 60% ABOVE its original
    fill, so an entry-basis stop cannot see the fall at all. Re-anchoring at
    each mark is the only form of the rule that keeps working on a winner -
    and testing only the entry basis would have measured a stop that never
    triggers and called it evidence that stops do nothing.
    """
    days = pd.bdate_range("2020-01-01", periods=700)
    n = len(days)
    climb = np.concatenate([np.linspace(100.0, 400.0, n - 40),
                            np.linspace(400.0, 300.0, 40)])
    bars = {"RIDE": pd.DataFrame(
        {"open": climb, "high": climb, "low": climb, "close": climb,
         "volume": np.full(n, 1e6)}, index=days)}
    for i in range(3):                       # filler so a ranking exists
        bars[f"FLAT{i}"] = _series(100.0, n=n, drift=0.0, seed=50 + i)

    entry = fb.run(bars, 500_000.0, top_n=2, min_turnover=0.0,
                   min_history=300, stop_pct=0.15, stop_basis="entry")
    rebal = fb.run(bars, 500_000.0, top_n=2, min_turnover=0.0,
                   min_history=300, stop_pct=0.15, stop_basis="rebalance")
    assert entry.stops_fired == 0
    assert rebal.stops_fired > 0


def test_a_split_sized_move_does_not_trigger_the_stop():
    """
    Unadjusted bars make a 1:10 split look like a -90% session, and a stop is
    exactly a rule that sells on a big down move. Without the corporate-action
    guard the stop arms would liquidate on every split in the universe and the
    experiment would be measuring the feed.
    """
    days = pd.bdate_range("2020-01-01", periods=700)
    n = len(days)
    close = np.full(n, 100.0)
    close[n - 30:] = 10.0                    # a 1:10 split, not a -90% day
    bars = {"SPLIT": pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1e6)}, index=days)}
    for i in range(3):
        bars[f"FLAT{i}"] = _series(100.0, n=n, drift=0.0, seed=70 + i)

    res = fb.run(bars, 500_000.0, top_n=4, min_turnover=0.0,
                 min_history=300, stop_pct=0.15, stop_basis="rebalance")
    assert res.stops_fired == 0


def test_a_stopped_name_is_re_bought_rather_than_banished(world):
    """
    A stop in this book is a DELAY, not an exit - the sleeve re-ranks monthly
    and buys anything still in the top `top_n`. If a stop permanently removed
    a name, the instrument would be a different (and far more destructive)
    thing than the one being tested.
    """
    res = _run(world, stop_pct=0.02, stop_basis="rebalance")
    assert res.stops_fired > 0
    seen = [set(names) for _, names in res.holdings_log]
    returned = any(s in seen[i + 1] and s not in seen[i] and
                   any(s in earlier for earlier in seen[:i + 1])
                   for i in range(len(seen) - 1) for s in seen[i + 1])
    assert returned


# ------------------------------------------------- the regime gate

def test_the_regime_gate_without_a_benchmark_raises(world):
    """
    Fails LOUDLY, not closed-and-quiet. A regime arm with no benchmark that
    silently reproduced the baseline is the "variant did nothing and said
    nothing" failure this repo has already paid for once.
    """
    with pytest.raises(ValueError):
        _run(world, regime_ma_days=200)


def test_the_regime_gate_blocks_when_it_cannot_see(world):
    """
    Too short a benchmark history must BLOCK, not pass. A gate that permits
    trading whenever it lacks data is a gate that reads as armed and is not.
    """
    idx = world["WIN00"].index
    short = pd.DataFrame({"close": np.linspace(100.0, 300.0, len(idx))},
                         index=idx)
    ok = fb._regime_gate(short, 200)
    assert ok(idx[10].date()) is False        # only 10 bars behind it
    assert ok(idx[400].date()) is True        # rising, and enough history


def test_the_regime_gate_stands_the_whole_sleeve_down(world, bench):
    res = _run(world, regime_ma_days=50, benchmark=bench)
    assert res.regime_flat > 0
    flat = [names for _, names in res.holdings_log if not names]
    assert flat, "a stood-down rebalance must actually end holding nothing"


# ------------------------------------------------- the cash frontier

def test_the_full_weight_blend_reproduces_the_sleeve(world):
    """The frontier has to pass through the baseline, or it is not a frontier."""
    base = _run(world)
    full = fd._blended(base, 500_000.0, 1.0, rf=0.065)
    assert full.cagr == pytest.approx(base.cagr(500_000.0), abs=1e-12)
    assert full.max_dd == pytest.approx(base.max_drawdown(), abs=1e-12)


def test_a_lighter_blend_cuts_the_drawdown(world):
    base = _run(world)
    heavy = fd._blended(base, 500_000.0, 1.0, rf=0.065)
    light = fd._blended(base, 500_000.0, 0.5, rf=0.065)
    assert light.dd_pct < heavy.dd_pct


# ------------------------------------------------- the deciding statistic

def _curve(values):
    """A minimal stand-in for a `FactorResult`, for the statistics only."""
    days = pd.bdate_range("2020-01-31", periods=len(values), freq="ME")
    return type("R", (), {"equity": [(d.date(), v)
                                     for d, v in zip(days, values)]})()


def test_consistency_sees_through_an_advantage_that_is_three_months():
    """
    The failure the calibration window actually produced.

    The arm finishes far ahead and wins only about half the months, because
    the whole difference is a handful of them. A CAGR comparison calls that a
    result; the month count and the drop-the-best-three column both refuse to.
    """
    rng = np.random.default_rng(4)
    # The arm differs from the book by noise with a slight negative drift, so
    # it wins about half the months and loses on average - plus three months
    # that carry the whole decade. This is the shape the real stop arms had.
    delta = rng.normal(-0.001, 0.012, 60)
    delta[[7, 23, 41]] = 0.35
    base = [100.0]
    arm = [100.0]
    for i in range(60):
        base.append(base[-1] * 1.002)
        arm.append(arm[-1] * (1.002 + delta[i]))

    c = fd.consistency(_curve(base), _curve(arm))
    assert 0.4 < c["win_rate"] < 0.6           # a coin flip, month by month
    assert c["compounded_pct"] > 50.0          # it "won" by a mile
    assert c["p"] > 0.05                       # and not in a single month
    assert c["top_k_drop_mean_pp"] < 0.0       # it WAS those three months


def test_a_small_sign_p_is_not_a_pass_the_direction_has_to_be_read():
    """
    THE TWO-SIDED TRAP, pinned so it cannot be walked into again.

    An arm that loses in almost every month while finishing ahead on three of
    them scores a TINY sign p - "significantly different from chance", in the
    wrong direction. A gate written as `p <= threshold` reports that as a
    pass; S1's gate did exactly that at 8/33 folds and printed PASS. Any
    reader of `consistency` has to check `win_rate` as well as `p`.
    """
    base = [100.0]
    arm = [100.0]
    for i in range(60):
        base.append(base[-1] * 1.002)
        arm.append(arm[-1] * (1.35 if i in (7, 23, 41) else 1.0015))

    c = fd.consistency(_curve(base), _curve(arm))
    assert c["compounded_pct"] > 0.0           # ahead overall
    assert c["p"] < 0.01                       # and "significant"
    assert c["win_rate"] < 0.5                 # while LOSING nearly every month


def test_consistency_reports_a_genuinely_consistent_arm():
    base = [100.0]
    arm = [100.0]
    for _ in range(60):
        base.append(base[-1] * 1.002)
        arm.append(arm[-1] * 1.006)
    c = fd.consistency(_curve(base), _curve(arm))
    assert c["p"] < 0.05
    assert c["top_k_drop_mean_pp"] > 0.0       # survives losing its best


def test_every_arm_is_reachable_and_uniquely_keyed():
    keys = [a.key for a in fd.ARMS]
    assert len(keys) == len(set(keys)) == 8
    for k in keys:
        assert fd.arm(k).key == k
    with pytest.raises(KeyError):
        fd.arm("nope")


def test_a_factor_config_field_actually_reaches_the_book(world, bench):
    """
    THE `partial_exit_at_r` TRAP, pinned for this package.

    In the swing sweep a variant set a field on the wrong dataclass, so the
    override silently did nothing while its NAME said the rule had changed -
    caught only by printing the resolved config. `run_arms` therefore takes
    its defaults FROM `FactorConfig`, and this asserts they arrive: a config
    with a stop set must produce stops on the arm that declares none.
    """
    from dataclasses import replace
    from nifty_algo.config import DEFAULT

    cfg = replace(DEFAULT.factor, top_n=4, min_turnover_inr=0.0,
                  min_history_sessions=300, stop_pct=0.02,
                  stop_basis="rebalance")
    rows, results = fd.run_arms(world, 500_000.0, cfg, benchmark=bench,
                                arms=(fd.arm("none"),))
    assert results["none"].stops_fired > 0

    off = replace(cfg, stop_pct=None)
    _, plain = fd.run_arms(world, 500_000.0, off, benchmark=bench,
                           arms=(fd.arm("none"),))
    assert plain["none"].stops_fired == 0


def test_an_arm_overrides_the_config_rather_than_the_reverse(world, bench):
    """An arm names the instrument it tests; the config must not outrank it."""
    from dataclasses import replace
    from nifty_algo.config import DEFAULT

    cfg = replace(DEFAULT.factor, top_n=4, min_turnover_inr=0.0,
                  min_history_sessions=300, stop_pct=0.02,
                  stop_basis="rebalance")
    _, results = fd.run_arms(world, 500_000.0, cfg, benchmark=bench,
                             arms=(fd.arm("none"), fd.arm("regime50")))
    assert results["regime50"].regime_flat > 0
