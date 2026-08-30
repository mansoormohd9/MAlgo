"""
The variant sweep.

What is being protected here is not arithmetic, it is honesty: that the
out-of-sample cell is scored on data the selection never saw, that a thin cell
cannot cast a vote, and that two variants which change the signals cannot end
up sharing one cached scan. A sweep that got any of those wrong would print a
perfectly formatted table describing a book nobody ran.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.swing import backtest as bt
from nifty_algo.swing import experiment as ex
from nifty_algo.swing import markets as markets_mod


def _wander(n=300, drift=0.0008, vol=0.013, seed=3, start=500.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes, p = [], start
    for _ in range(n):
        p *= (1 + drift + rng.normal(0, vol))
        closes.append(p)
    c = np.array(closes)
    return pd.DataFrame({
        "open": c * (1 + rng.normal(0, 0.002, n)),
        "high": c * (1 + np.abs(rng.normal(0, 0.008, n))),
        "low": c * (1 - np.abs(rng.normal(0, 0.008, n))),
        "close": c,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    }, index=pd.bdate_range(datetime(2024, 1, 2), periods=n))


@pytest.fixture(scope="module")
def cfg() -> Config:
    c = Config()
    c.capital.swing_capital_inr = 100_000.0
    m = c.swing.markets["india"]
    m.min_price, m.min_avg_turnover = 1.0, 0.0
    return c


@pytest.fixture(scope="module")
def market(cfg):
    return markets_mod.get(cfg, "india")


@pytest.fixture(scope="module")
def world():
    bars = {f"SYM{i}": _wander(seed=i, drift=0.0004 + i * 0.0004)
            for i in range(2)}
    return bars, _wander(seed=99, drift=0.0002, vol=0.007, start=20_000.0)


@pytest.fixture(scope="module")
def swept(cfg, market, world):
    """
    One sweep for the whole module. Two symbols, two variants, four folds -
    every path exercised, and the fixture is shared because `sweep` walks the
    sessions once per (variant, window, phase) and a per-test rebuild made
    this file a minute long on its own.
    """
    bars, bench = world
    return ex.sweep(cfg, market, bars, bench,
                    variants=(ex.variant("baseline"), ex.variant("be15")),
                    train_months=6, test_months=2)


# ---------------------------------------------------------------- the shape

def test_every_variant_is_scored_on_both_sides_of_every_fold(swept):
    assert swept.windows
    assert len(swept.cells) == 2 * len(swept.windows) * 2
    for cell in swept.cells:
        assert cell.phase in ("train", "test")
        if cell.phase == "test":
            window = swept.windows[cell.fold]
            # The test window begins after the train window ends. If this ever
            # reversed, the "out-of-sample" column would be in-sample and
            # would say nothing at all.
            assert cell.start > window.train_end


def test_a_thin_cell_is_reported_but_cannot_vote(swept):
    """
    `judgeable` gates selection, not reporting.

    Hiding thin cells would make the table look better than the evidence;
    letting them choose would let three trades pick the variant that trades
    live money.
    """
    for cell in swept.cells:
        assert cell.judgeable == (cell.trades >= ex.MIN_TRADES_TO_JUDGE)

    thin = ex.Cell(variant="x", fold=0, phase="train", start=None, end=None,
                   trades=ex.MIN_TRADES_TO_JUDGE - 1, win_rate=1.0,
                   breakeven_win_rate=0.0, expectancy_r=99.0, total_r=99.0,
                   max_drawdown_r=0.0, capital_blocks=0, regime_blocks=0)
    assert not thin.judgeable


def test_selection_reads_train_and_reports_the_test_window(swept):
    """
    One row per fold, and the reported number is always the test one.

    This is the only line in the sweep a live book could have acted on, so it
    is the only one allowed to claim anything.
    """
    picked = swept.selected_by_train()
    assert len(picked) == len(swept.windows)
    for window, key, cell in picked:
        if cell is None:
            continue
        assert cell.phase == "test"
        assert cell.variant == key
        assert cell.fold == window.index


# ---------------------------------------------------------------- the cache

def test_ladder_variants_share_a_scan_and_signal_variants_do_not(cfg, market):
    """
    The sweep is affordable only because the breakeven and regime levers
    reuse one pass - and correct only because the stop-band and setup-mix
    levers do not.
    """
    base = bt.scan_signature(cfg, market)

    for key in ("be15", "beoff", "regime200", "regime50"):
        assert bt.scan_signature(ex.variant(key).configure(cfg), market) == base

    for key in ("stopwide", "breakoutonly", "pullbackonly"):
        assert bt.scan_signature(ex.variant(key).configure(cfg), market) != base


def test_a_variant_cannot_leak_into_the_next_one(cfg, market):
    """
    `configure` deep-copies. A shallow copy would share one SwingConfig, so
    the second variant would silently inherit the first's changes - and every
    row after the first would describe a book that was never specified.
    """
    first = ex.variant("regime200").configure(cfg)
    second = ex.variant("be15").configure(cfg)

    assert first.swing.regime_ma_days == 200
    assert second.swing.regime_ma_days == 0        # not 200
    assert cfg.swing.regime_ma_days == 0           # the base is untouched
    assert second.swing.breakeven_at_r == pytest.approx(1.5)
    assert first.swing.breakeven_at_r is None


def test_the_sweep_says_windows_start_flat(swept):
    """
    A caveat that is not printed is a caveat nobody has.
    """
    assert any("starts flat" in w for w in swept.warnings)


# ---------------------------------------------------------------- the ledger

def _row(variant, fold, phase, trades, expectancy):
    return {
        "variant": variant, "fold": fold, "phase": phase,
        "start": None, "end": None, "trades": trades, "win_rate": 0.3,
        "breakeven_win_rate": 0.25, "expectancy_r": expectancy,
        "total_r": expectancy * trades, "max_drawdown_r": 1.0,
        "capital_blocks": 0, "regime_blocks": 0,
        "judgeable": trades >= ex.MIN_TRADES_TO_JUDGE,
    }


def test_the_ledger_reads_every_signature_group_back_as_one(tmp_path):
    """
    Signature groups run as separate parallel invocations, so one experiment
    lands in several files. Reading only the newest would silently report a
    fraction of it as the whole.
    """
    pd.DataFrame([_row("baseline", 0, "test", 30, -0.08)]).to_parquet(
        tmp_path / "sweep_india_20260101-100000_baseline.parquet")
    pd.DataFrame([_row("stopwide", 0, "test", 30, +0.05)]).to_parquet(
        tmp_path / "sweep_india_20260101-100500_stopwide.parquet")

    frame = ex.read_ledger(str(tmp_path))
    assert set(frame["variant"]) == {"baseline", "stopwide"}
    assert len(frame) == 2


def test_a_rerun_cell_replaces_the_older_one(tmp_path):
    """
    The reason to re-run a cell is that the earlier one was wrong, so the
    newest file wins. Keeping both would double-count it in every pooled
    average.
    """
    pd.DataFrame([_row("baseline", 0, "test", 30, -0.50)]).to_parquet(
        tmp_path / "sweep_india_20260101-090000_baseline.parquet")
    pd.DataFrame([_row("baseline", 0, "test", 30, +0.10)]).to_parquet(
        tmp_path / "sweep_india_20260102-090000_baseline.parquet")

    frame = ex.read_ledger(str(tmp_path))
    assert len(frame) == 1
    assert frame.iloc[0]["expectancy_r"] == pytest.approx(0.10)


def test_an_empty_ledger_says_so_rather_than_failing(tmp_path, capsys):
    ex.report(ex.read_ledger(str(tmp_path)))
    assert "No sweep results" in capsys.readouterr().out


def test_the_live_table_and_the_ledger_table_are_one_implementation(swept, cfg,
                                                                    capsys):
    """
    A live sweep and a re-read of its own parquet must agree, and the only way
    to be sure is for both to run the same code. `_print` adds the caveats and
    then calls `report`.
    """
    ex._print(swept, cfg)
    live = capsys.readouterr().out
    ex.report(swept.frame())
    from_frame = capsys.readouterr().out

    assert "Walk-forward expectancy" in live or "no variant had enough" in live
    # Every line the ledger reporter produced appears in the live output too.
    for line in from_frame.splitlines():
        if line.strip():
            assert line in live


def test_the_window_start_bounds_the_folds_without_dropping_warmup(cfg, market,
                                                                    world):
    """
    `--years` names the TEST WINDOW, and the bars before it stay loaded.

    For a long time it named only the fetch, so a run labelled "3 years"
    scored every session the parquet happened to hold - 5.4 of them - and the
    label on the result was simply wrong. Trimming the BARS instead would fix
    the label and break the indicators, because the first scored session needs
    history behind it.
    """
    from datetime import timedelta

    bars, bench = world
    sessions = bt.all_sessions(bars)
    late = (sessions[-1] - timedelta(days=200)).date()

    full = ex.sweep(cfg, market, bars, bench,
                    variants=(ex.variant("baseline"),),
                    train_months=3, test_months=1)
    bounded = ex.sweep(cfg, market, bars, bench,
                       variants=(ex.variant("baseline"),),
                       train_months=3, test_months=1, window_start=late)

    assert bounded.windows
    assert len(bounded.windows) < len(full.windows)
    assert bounded.windows[0].train_start >= late
    # The bars themselves were never trimmed, so warm-up is intact.
    assert len(bt.all_sessions(bars)) == len(sessions)
