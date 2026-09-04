"""
The option book's sweep: the pack, the folds, the date range, the variants.

The load-bearing test here is `test_the_indicator_pack_changes_nothing`. The
pack makes a sweep affordable (measured 4.3x on real bars), and it does that by
serving precomputed prefix slices instead of recomputing. If a single served
series were ever off by one bar, every number the harness produced would be
wrong in a way no other test would catch - the trades would still look
entirely plausible.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from datetime import date

import pandas as pd
import pytest

from nifty_algo import indicator_cache as ic
from nifty_algo.backtest import Backtester
from nifty_algo.config import Config
from nifty_algo.data.csv_feed import CsvFeed
from nifty_algo.experiment_intraday import VARIANTS, sweep, variant
from nifty_algo.swing.backtest import fold_windows


@pytest.fixture(scope="module")
def sample():
    return CsvFeed("data/sample_nifty_5m.csv").get_bars(lookback_days=0)


def _trades_frame(res):
    return pd.DataFrame([asdict(t) for t in res.trades])


# --------------------------------------------------------------- the pack

def test_the_indicator_pack_changes_nothing(sample):
    """
    Identical on every field, or the speed-up is buying wrong answers.

    `_DISABLED` is read once at import (`indicator_cache.py`), so flipping the
    env var mid-process does nothing without `refresh_disabled()` - a detail
    that would otherwise make this test silently compare the pack against
    itself and pass for the wrong reason.
    """
    before = os.environ.get(ic.DISABLE_ENV)
    try:
        os.environ[ic.DISABLE_ENV] = "1"
        ic.refresh_disabled()
        assert ic.disabled(), "the kill switch did not take - test is vacuous"
        off = _trades_frame(Backtester(Config()).run(sample))

        os.environ[ic.DISABLE_ENV] = "0"
        ic.refresh_disabled()
        assert not ic.disabled()
        on = _trades_frame(Backtester(Config()).run(sample))
    finally:
        if before is None:
            os.environ.pop(ic.DISABLE_ENV, None)
        else:
            os.environ[ic.DISABLE_ENV] = before
        ic.refresh_disabled()

    assert not off.empty, "the fixture must produce trades or this proves nothing"
    assert off.equals(on)


# --------------------------------------------------------------- the folds

def test_folds_tile_the_period_without_scoring_a_session_twice():
    """
    THE REGRESSION FOR THE OLD SPLITTER.

    `_make_folds` used to set fold i's `test_end` equal to fold i+1's
    `test_start`, and `run()` filters inclusively at both ends - so every
    boundary session was scored in two folds and appeared twice in the pooled
    trade list. Nothing errored; the sample was simply inflated.
    """
    windows = fold_windows(date(2022, 1, 1), date(2026, 1, 1), 6, 2)
    assert len(windows) > 5
    for a, b in zip(windows, windows[1:]):
        assert a.test_end < b.test_start, (
            f"fold {a.index} ends {a.test_end} and fold {b.index} starts "
            f"{b.test_start} - the boundary session is scored twice"
        )
    for w in windows:
        assert w.train_end < w.test_start
        assert w.test_start <= w.test_end


def test_the_backtester_uses_the_shared_splitter():
    """One definition of a fold across three books, or 'fold' means nothing."""
    cfg = Config()
    bt = Backtester(cfg)
    idx = pd.date_range("2022-01-03 09:15", "2026-01-01 15:25", freq="5min")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 0}, index=idx)
    folds = bt._make_folds(bars)
    assert folds and hasattr(folds[0], "test_start"), "expected Window objects"
    assert folds[0].index == 0


# ---------------------------------------------------------- the date range

def test_a_date_range_scores_exactly_that_range(sample):
    days = sorted({d for d in sample.index.date})
    lo, hi = days[2], days[5]
    res = Backtester(Config()).run(sample, start=lo, end=hi)
    assert not res.folds, "a bounded run is one pass, not a walk-forward"
    for t in res.trades:
        assert lo <= t.entry_time.date() <= hi


def test_a_bounded_run_is_a_subset_of_the_unbounded_one(sample):
    """
    The range must select sessions, not change how they are scored. Same
    sessions in, same trades out.
    """
    days = sorted({d for d in sample.index.date})
    full = Backtester(Config()).run(sample)
    part = Backtester(Config()).run(sample, start=days[3], end=days[-1])

    got = {(t.entry_time, t.strategy) for t in part.trades}
    have = {(t.entry_time, t.strategy) for t in full.trades
            if t.entry_time.date() >= days[3]}
    assert got == have


# ------------------------------------------------------------ MFE and MAE

def test_trades_record_their_excursions(sample):
    """
    Without these `diagnostics.mfe_attainment` and `capture_ratio` read
    `getattr(t, "mfe_r", 0.0)` and report a confident zero - "how much of the
    move did the ladder capture" answered as "none of it", for every trade.
    """
    res = Backtester(Config()).run(sample)
    assert res.trades
    assert any(t.mfe_r > 0 for t in res.trades), "no trade ever went favourable"
    for t in res.trades:
        assert t.mfe_r >= 0.0 >= t.mae_r, (t.mfe_r, t.mae_r)
        assert t.mfe_r >= t.r_multiple - 1e-9, (
            "a trade cannot realise more R than it ever saw"
        )


# ------------------------------------------------------------- the variants

def test_every_variant_is_registered_with_a_hypothesis():
    """
    A variant without a stated `why` is a knob someone turned, and the sweep
    becomes a search for the best of ten rather than a test of five claims.
    """
    keys = [v.key for v in VARIANTS]
    assert keys[0] == "baseline", "the control must exist and come first"
    assert len(keys) == len(set(keys))
    for v in VARIANTS:
        assert len(v.why) > 80, f"{v.key} has no real hypothesis"
        assert v.label


def test_the_novwap_variant_actually_drops_the_strategy():
    cfg = variant("novwap").configure(Config())
    assert "vwap_reclaim" not in cfg.backtest.strategy_keys
    assert cfg.backtest.strategy_keys, "it must name the survivors, not empty"


def test_the_baseline_variant_changes_nothing():
    base, configured = Config(), variant("baseline").configure(Config())
    assert configured.regime.gap_atr_multiple == base.regime.gap_atr_multiple
    assert configured.session.warmup_sessions == base.session.warmup_sessions
    assert configured.trade.enable_runner == base.trade.enable_runner


def test_a_sweep_produces_train_and_test_cells_for_every_variant(sample):
    """
    Both phases must be run. `Backtester.run`'s own fold machinery executes
    TEST windows only and carries the train dates as labels, so a sweep that
    relied on it would have no train column and could never select.
    """
    cfg = Config()
    chosen = (variant("baseline"), variant("ladder1r"))
    sw = sweep(cfg, sample, chosen, train_months=1, test_months=1)
    if not sw.cells:
        pytest.skip("sample file too short for even one fold")
    frame = sw.frame()
    assert set(frame["phase"]) == {"train", "test"}
    assert set(frame["variant"]) == {"baseline", "ladder1r"}
