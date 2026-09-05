"""
F1's gate logic, tested on constructed verdicts.

The arithmetic here decides whether a book gets traded, so it is tested
independently of any backtest: a gate that reads the wrong way round, or that
divides by a negative excess and returns a plausible ratio, would pass a dead
sleeve with no error anywhere.
"""
from __future__ import annotations

import math

import pytest

from nifty_algo.factor import verdict as fv


def _arm(label, cagr):
    return fv.Arm(label=label, cagr=cagr, sharpe=1.0, max_dd=-0.3,
                  recovery_months=12.0, turnover=0.6, costs_inr=1000.0,
                  trades=100)


def _verdict(drift, reweighted, nulls):
    return fv.Verdict(slippage_pct=0.0025, drift=_arm("drift", drift),
                      reweighted=_arm("even", reweighted),
                      nulls=[_arm(f"n{i}", c) for i, c in enumerate(nulls)],
                      capital=500_000.0)


def test_empirical_p_is_the_permutation_form_and_never_zero():
    """
    A p of exactly zero claims more certainty than 40 draws can support, and
    it is what `#{null >= obs} / n` returns when the signal wins outright.
    """
    v = _verdict(0.30, 0.28, [0.10] * 40)
    assert v.empirical_p(0.30) == pytest.approx(1.0 / 41.0)
    assert v.empirical_p(0.30) > 0.0


def test_gate1_needs_to_beat_the_stated_share_of_seeds():
    beats_all = _verdict(0.30, 0.28, [0.10] * 40)
    assert beats_all.gate1_pass

    # Beating 36 of 40 gives p = (1+4)/41 = 0.122 - not enough.
    mixed = _verdict(0.30, 0.28, [0.10] * 36 + [0.40] * 4)
    assert mixed.empirical_p(0.30) == pytest.approx(5.0 / 41.0)
    assert not mixed.gate1_pass


def test_gate2_fails_when_reweighting_gives_up_the_excess():
    """The concentration kill: excess must survive equal weighting."""
    # median null 0.10; drift excess +0.20; reweighted excess +0.05 -> 25%.
    concentrated = _verdict(0.30, 0.15, [0.10] * 40)
    assert concentrated.retained == pytest.approx(0.25)
    assert not concentrated.gate2_pass

    # reweighted keeps 90% of the excess.
    broad = _verdict(0.30, 0.28, [0.10] * 40)
    assert broad.retained == pytest.approx(0.90)
    assert broad.gate2_pass


def test_gate2_cannot_pass_on_a_negative_excess():
    """
    If momentum does not beat the median null at all, the ratio is undefined
    and must NOT come back as a pass. Dividing two negatives would otherwise
    produce a healthy-looking +1.0.
    """
    dead = _verdict(0.05, 0.02, [0.10] * 40)
    assert dead.excess_drift < 0
    assert math.isnan(dead.retained)
    assert not dead.gate2_pass


def test_a_verdict_with_no_nulls_does_not_pass_by_default():
    v = fv.Verdict(slippage_pct=0.0025, drift=_arm("d", 0.30),
                   reweighted=_arm("e", 0.28), nulls=[], capital=500_000.0)
    assert math.isnan(v.empirical_p(0.30))
    assert not v.gate1_pass


def test_report_states_both_gates_and_the_survivorship_note():
    text = fv.report(_verdict(0.30, 0.15, [0.10] * 40))
    assert "GATE 2 FAILED" in text
    assert "listedonly" in text
    assert "ann(arith)" not in text          # this driver never prints one


# ------------------------------------------------------- walk-forward stats

def _fold(i, momentum, nulls):
    return fv.Fold(index=i, start=None, end=None, momentum=momentum,
                   nulls=list(nulls))


def test_percentile_is_the_share_of_nulls_beaten():
    f = _fold(0, 0.10, [0.0] * 9 + [0.20])
    assert f.percentile == pytest.approx(0.9)
    assert f.won


def test_a_fold_with_no_nulls_is_nan_not_a_win():
    """A missing null must not read as a victory."""
    f = _fold(0, 0.10, [])
    assert math.isnan(f.percentile)
    assert math.isnan(f.null_median)


def test_sign_test_matches_the_exact_binomial():
    # 12 of 16 -> the same 0.077 the single-seed sweep reported.
    wf = fv.WalkForward(
        folds=[_fold(i, 1.0 if i < 12 else -1.0, [0.0]) for i in range(16)],
        slippage_pct=0.0025)
    assert wf.wins == 12
    assert wf.sign_p == pytest.approx(0.0768, abs=5e-4)


def test_combined_p_sees_magnitude_the_sign_test_discards():
    """
    Two books that win the same number of folds but by different margins must
    NOT get the same answer. This is the whole reason the combined statistic
    is reported beside the sign test.
    """
    strong = fv.WalkForward(
        folds=[_fold(i, 0.5, [0.0] * 39 + [1.0]) for i in range(16)],
        slippage_pct=0.0025)          # 97th percentile every window
    weak = fv.WalkForward(
        folds=[_fold(i, 0.01, [0.0] * 20 + [0.5] * 19) for i in range(16)],
        slippage_pct=0.0025)          # just past the median every window
    assert strong.wins == weak.wins == 16
    assert strong.sign_p == weak.sign_p
    assert strong.combined_p < weak.combined_p


def test_combined_p_is_centred_under_the_null():
    """Momentum at the median of every fold must not look significant."""
    wf = fv.WalkForward(
        folds=[_fold(i, 0.0, [-1.0] * 20 + [1.0] * 20) for i in range(16)],
        slippage_pct=0.0025)
    assert wf.mean_percentile == pytest.approx(0.5)
    assert wf.combined_p == pytest.approx(0.5, abs=0.02)
