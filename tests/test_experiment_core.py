"""
The shared sweep harness: the statistics, and the ways they lie.

Every test here guards a number that would otherwise be confidently wrong
rather than absent - a pooled average carried by one thin fold, an interval
narrowed by resampling correlated trades, a variant that silently tests
nothing because its field name was misspelled.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from nifty_algo import experiment_core as core
from nifty_algo.config import Config
from nifty_algo.experiment_core import Cell, Sweep, Variant, set_on


def _cell(variant, fold, phase, trades, exp, total=None):
    return Cell(variant=variant, fold=fold, phase=phase,
                start=date(2024, 1, 1), end=date(2024, 3, 1),
                trades=trades, expectancy_r=exp,
                total_r=total if total is not None else exp * trades)


# ------------------------------------------------------------------ variants

def test_configure_deep_copies_so_variants_cannot_leak_into_each_other():
    """
    `dataclasses.replace` is shallow. A variant mutating `cfg.regime` through
    a shallow copy would edit the caller's config, and every later variant in
    the sweep would inherit it - the sweep would measure a cumulative stack of
    changes while reporting each as if it stood alone.
    """
    base = Config()
    original = base.regime.gap_atr_multiple

    v = Variant("v", "l", "w", set_on("regime", gap_atr_multiple=99.0))
    changed = v.configure(base)

    assert changed.regime.gap_atr_multiple == 99.0
    assert base.regime.gap_atr_multiple == original, "the base config moved"

    other = Variant("o", "l", "w", lambda cfg: None).configure(base)
    assert other.regime.gap_atr_multiple == original, "leaked into the next variant"


def test_a_variant_that_sets_a_nonexistent_field_raises():
    """A typo must be an error at build time, not a variant that tests nothing."""
    v = Variant("typo", "l", "w", set_on("regime", gap_atr_multiplier=3.0))
    with pytest.raises(AttributeError, match="no field"):
        v.configure(Config())


# ------------------------------------------------------------ the sign test

def test_the_sign_test_matches_the_known_warmup_result():
    """
    The measured result this harness exists to institutionalise: a warm-up
    variant won 10 of 21 walk-forward windows. It must read as a coin flip.
    """
    assert core.two_sided_sign_p(10, 21) > 0.5
    assert core.two_sided_sign_p(21, 21) < 0.001
    assert core.two_sided_sign_p(0, 21) < 0.001
    assert core.two_sided_sign_p(10, 21) == core.two_sided_sign_p(11, 21), (
        "the test is symmetric"
    )
    # no folds is not a p-value of 1.0, it is no answer
    assert core.two_sided_sign_p(0, 0) != core.two_sided_sign_p(0, 0)


def test_sign_test_counts_folds_and_excludes_ties():
    sw = Sweep()
    # base loses 2, wins 1, ties 1
    for fold, (b, v) in enumerate([(0.0, 0.1), (0.0, 0.2), (0.5, 0.1), (0.3, 0.3)]):
        sw.cells.append(_cell("baseline", fold, "test", 50, b))
        sw.cells.append(_cell("cand", fold, "test", 50, v))

    st = sw.sign_test("cand", against="baseline")
    assert st["wins"] == 2
    assert st["losses"] == 1
    assert st["ties"] == 1
    assert st["n"] == 3, "a tie is not evidence either way and leaves n"
    assert st["coin_flip"] == pytest.approx(1.5)


def test_sign_test_is_blank_when_the_baseline_is_missing():
    sw = Sweep()
    sw.cells.append(_cell("cand", 0, "test", 50, 0.1))
    st = sw.sign_test("cand", against="baseline")
    assert st["n"] == 0 and st["wins"] == 0


# ------------------------------------------------------------ pooling and CI

def test_pooled_expectancy_is_trade_weighted_not_a_mean_of_folds():
    """
    A fold with 3 trades and one with 60 are not equally informative. A plain
    mean lets the thin fold swing the answer - here it would report +0.45R for
    a book that actually made +0.014R per trade.
    """
    sw = Sweep()
    sw.cells.append(_cell("v", 0, "test", 3, 0.90))
    sw.cells.append(_cell("v", 1, "test", 60, 0.00))

    pooled = sw.pooled()
    got = float(pooled[pooled["variant"] == "v"]["expectancy_r"].iloc[0])
    assert got == pytest.approx((3 * 0.90 + 60 * 0.0) / 63)
    assert got < 0.05
    assert got != pytest.approx((0.90 + 0.0) / 2)


def test_bootstrap_refuses_an_interval_from_too_few_folds():
    sw = Sweep()
    for fold in range(2):
        sw.cells.append(_cell("v", fold, "test", 40, 0.1))
    lo, hi = sw.bootstrap_ci("v")
    assert lo != lo and hi != hi, "fewer than 3 folds cannot support an interval"


def test_bootstrap_interval_brackets_the_pooled_estimate():
    sw = Sweep()
    for fold, e in enumerate([0.10, -0.05, 0.20, 0.00, 0.15, -0.10]):
        sw.cells.append(_cell("v", fold, "test", 40, e))
    lo, hi = sw.bootstrap_ci("v")
    pooled = float(sw.pooled().query("variant == 'v'")["expectancy_r"].iloc[0])
    assert lo < pooled < hi
    assert lo == lo and hi == hi


def test_the_interval_is_deterministic_for_a_given_seed():
    sw = Sweep()
    for fold, e in enumerate([0.1, -0.2, 0.3, 0.0, 0.05]):
        sw.cells.append(_cell("v", fold, "test", 30, e))
    assert sw.bootstrap_ci("v") == sw.bootstrap_ci("v")


# -------------------------------------------------------------- selection

def test_selection_picks_on_train_and_scores_on_the_untouched_test_window():
    sw = Sweep()
    # fold 0: "good" looks best on train but is worse on test
    sw.cells.append(_cell("good", 0, "train", 50, 0.50))
    sw.cells.append(_cell("dull", 0, "train", 50, 0.10))
    sw.cells.append(_cell("good", 0, "test", 50, -0.20))
    sw.cells.append(_cell("dull", 0, "test", 50, +0.10))

    sel = sw.selected_by_train()
    assert len(sel) == 1
    assert sel.iloc[0]["picked"] == "good"
    assert sel.iloc[0]["test_r"] == pytest.approx(-0.20), (
        "selection must be scored on what it actually paid, not on train"
    )


def test_a_thin_train_cell_cannot_be_selected():
    sw = Sweep()
    sw.cells.append(_cell("thin", 0, "train", core.MIN_TRADES_TO_JUDGE - 1, 9.9))
    sw.cells.append(_cell("solid", 0, "train", 50, 0.10))
    sw.cells.append(_cell("thin", 0, "test", 50, 0.0))
    sw.cells.append(_cell("solid", 0, "test", 50, 0.0))
    sel = sw.selected_by_train()
    assert sel.iloc[0]["picked"] == "solid", "6 trades is a sample, not a measurement"


# ----------------------------------------------------------------- ledger

def test_the_ledger_round_trips_and_the_newest_row_wins(tmp_path):
    sw = Sweep()
    sw.cells.append(_cell("v", 0, "test", 10, 0.1))
    first = core.write_ledger(sw, tmp_path, "sweep_x")
    assert first is not None and first.exists()

    # the same cell, re-run with a corrected number
    again = Sweep()
    again.cells.append(_cell("v", 0, "test", 10, 0.9))
    import time
    time.sleep(1.05)              # the stamp is second-resolution
    core.write_ledger(again, tmp_path, "sweep_x")

    frame = core.read_ledger(tmp_path, "sweep_x")
    assert len(frame) == 1, "a re-run supersedes rather than double-counts"
    assert float(frame["expectancy_r"].iloc[0]) == pytest.approx(0.9)


def test_an_empty_sweep_writes_no_ledger(tmp_path):
    assert core.write_ledger(Sweep(), tmp_path, "sweep_x") is None
    assert core.read_ledger(tmp_path, "sweep_x").empty


def test_book_specific_columns_survive_into_the_ledger(tmp_path):
    sw = Sweep()
    c = _cell("v", 0, "test", 10, 0.1)
    c.extra = {"friction_r": 0.06, "skipped_days": 2}
    sw.cells.append(c)
    core.write_ledger(sw, tmp_path, "sweep_x")
    frame = core.read_ledger(tmp_path, "sweep_x")
    assert float(frame["friction_r"].iloc[0]) == pytest.approx(0.06)
    assert int(frame["skipped_days"].iloc[0]) == 2


# ------------------------------------------------------------------ report

def test_the_report_shows_the_test_column_and_the_sign_test():
    sw = Sweep()
    for fold in range(5):
        sw.cells.append(_cell("baseline", fold, "train", 40, 0.00))
        sw.cells.append(_cell("baseline", fold, "test", 40, 0.00))
        sw.cells.append(_cell("cand", fold, "train", 40, 0.50))
        sw.cells.append(_cell("cand", fold, "test", 40, 0.02))

    out = core.report(sw)
    assert "TEST R" in out and "95% CI (test)" in out
    assert "vs base" in out and "sign test" in out
    assert "cand" in out and "baseline" in out
    # the fitting gap must be visible: train 0.50 against test 0.02
    assert "+0.480" in out


def test_the_report_says_so_when_nothing_was_scored_out_of_sample():
    sw = Sweep()
    sw.cells.append(_cell("v", 0, "train", 40, 0.1))
    assert "out of sample" in core.report(sw)
    assert "no cells" in core.report(Sweep())
