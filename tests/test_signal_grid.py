"""
E5's instruments, tested against brute force and against planted truth.

Every check here exists because the failure it guards produces a PLAUSIBLE
number rather than an exception. A conjugate on the wrong operand in
`_xcorr_all_shifts` yields a perfectly well-formed null that is simply not the
null anyone intended; a `_forward` that reaches across a session boundary
returns tomorrow morning's price with no error; and a null that is too narrow
manufactures a discovery rather than raising.

So: brute force where a closed form is used, a PLANTED effect that must be
recovered, and pure noise that must NOT be.
"""
from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.intraday_equity import signal_grid as sg


# ------------------------------------------------------------ small helpers

def _panel(S=25, D=80, C=75, seed=0) -> sg.Panel:
    """
    A synthetic panel with the REAL shape and no structure at all.

    The shape matters more than it looks. An earlier version of this helper
    used 12 clocks, and every signal that needs a 20-bar trailing window
    silently fired zero times - the grid returned 234 empty cells and the
    calibration test "passed" a NaN. A test panel narrower than the trailing
    windows under test is not a small version of the problem, it is a
    different one.

    Ranges and volumes are randomised per bar rather than held proportional to
    price, because `range_exp` and `vol_surge` are defined against their own
    trailing means and a constant series can never trip them.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.002, size=(S, D, C))
    close = (100.0 * np.exp(np.cumsum(steps, axis=-1))).astype(np.float32)
    span = close * rng.lognormal(-6.0, 0.6, size=(S, D, C)).astype(np.float32)
    bench = (100.0 * np.exp(np.cumsum(
        rng.normal(0, 0.001, size=(D, C)), axis=-1))).astype(np.float32)
    return sg.Panel(
        symbols=[f"S{i}" for i in range(S)],
        dates=np.array([date(2024, 1, 1) + timedelta(days=i)
                        for i in range(D)]),
        clocks=[time(9 + (15 + 5 * i) // 60, (15 + 5 * i) % 60)
                for i in range(C)],
        close=close,
        high=(close + span).astype(np.float32),
        low=(close - span).astype(np.float32),
        volume=rng.lognormal(10.0, 0.8, size=(S, D, C)).astype(np.float32),
        bench_close=bench)


def test_every_pre_registered_signal_actually_fires():
    """
    A signal that never fires contributes nothing and costs nothing, so the
    grid reports it as a clean zero rather than as a broken cell. That is the
    exact shape of the bug this file was written after: four of thirteen
    signals fired zero times on the first test panel and the suite went green.
    """
    p = _panel()
    silent = [s.key for s in sg.SIGNALS
              if not np.asarray(s.build(p), dtype=bool).any()]
    assert not silent, f"signals that never fire: {silent}"


# --------------------------------------------------------- the closed forms

def test_xcorr_matches_brute_force():
    """
    THE test in this file. `_xcorr_all_shifts` replaces a 10^12-operation loop
    with an FFT, and an off-by-one in the shift direction would still produce
    a complete, symmetric, entirely wrong null.
    """
    rng = np.random.default_rng(7)
    mask = rng.random((3, 17, 5)) < 0.3
    y = rng.normal(size=(3, 17, 5))

    fast = sg._xcorr_all_shifts(mask, y)
    D = mask.shape[1]
    slow = np.array([float((mask * np.roll(y, -k, axis=1)).sum())
                     for k in range(D)])
    assert np.allclose(fast, slow, atol=1e-9)


def test_xcorr_shift_zero_is_the_unshifted_sum():
    """k=0 must be the observed statistic, not a shifted one."""
    rng = np.random.default_rng(3)
    mask = rng.random((2, 9, 4)) < 0.5
    y = rng.normal(size=(2, 9, 4))
    assert sg._xcorr_all_shifts(mask, y)[0] == pytest.approx(
        float((mask * y).sum()))


def test_roll_mean_is_trailing_and_exclusive():
    """A trailing window that includes the current bar is a look-ahead."""
    a = np.arange(10, dtype=float).reshape(1, 1, 10)
    got = sg._roll_mean(a, 3)
    # out[i] = mean(a[i-3:i]); at i=3 that is mean(0,1,2) = 1.0
    assert np.isnan(got[0, 0, :3]).all()
    assert got[0, 0, 3] == pytest.approx(1.0)
    assert got[0, 0, 9] == pytest.approx(7.0)


def test_forward_never_crosses_a_session():
    """
    The whole reason the panel is (session x clock): a positional shift on a
    flat frame returns tomorrow morning's price for a bar near the close.
    """
    p = _panel(S=2, D=5, C=6)
    f = sg._forward(p.close, 2)
    assert np.isnan(f[..., -2:]).all()
    assert np.isfinite(f[..., :-2]).all()


def test_executable_mask_excludes_the_signal_bar():
    m = np.zeros((1, 1, 5), dtype=bool)
    m[0, 0, 2] = True
    out = sg._executable(m)
    assert not out[0, 0, 2]
    assert out[0, 0, 3]
    assert out.sum() == 1


def test_excess_columns_sum_to_zero():
    """
    The leave-one-out control makes every (symbol, clock) column sum to
    exactly zero. That is what makes the circular-shift null exactly centred -
    if this drifts, every cell acquires a baseline that the null cannot see.
    """
    p = _panel(S=3, D=30, C=8, seed=11)
    e = sg.excess_forward(p, 2)
    col = np.nansum(e, axis=1)
    assert np.allclose(col[np.isfinite(col)], 0.0, atol=1e-9)


# ------------------------------------------------------- planted vs nothing

def _grid_on(panel, signals, horizons=(1,)):
    windows = (("all", panel.clocks[0], time(23, 59)),)
    cells = sg.run_grid(panel, horizons=horizons, signals=signals,
                        windows=windows)
    return sg.family_wise(cells)


def test_a_planted_effect_is_recovered(monkeypatch):
    """
    Build a panel where one signal genuinely predicts, and require the grid to
    find it. A test suite that only proves the null is never rejected cannot
    tell a working instrument from a broken one.
    """
    monkeypatch.setattr(sg, "MIN_FIRINGS", 50)
    rng = np.random.default_rng(5)
    S, D, C = 4, 60, 12
    fire = rng.random((S, D, C)) < 0.25
    steps = rng.normal(0, 0.002, size=(S, D, C))
    # Every bar AFTER a firing gets a large positive drift.
    steps += np.roll(fire, 2, axis=-1) * 0.02
    close = (100.0 * np.exp(np.cumsum(steps, axis=-1))).astype(np.float32)
    p = _panel(S=S, D=D, C=C, seed=5)
    p.close[:] = close
    p.bench_close[:] = 100.0

    sig = (sg.Signal("planted", "the truth", lambda _p: fire),)
    res = _grid_on(p, sig, horizons=(2,))
    assert res.best is not None
    assert res.best.mean_excess > 0
    assert res.family_p <= 0.05, f"planted effect missed, p={res.family_p}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pure_noise_does_not_manufacture_a_discovery(seed, monkeypatch):
    """
    The calibration check. A null that is too narrow - which is what shuffling
    per symbol instead of circular-shifting all symbols together would give -
    shows up here and nowhere else.
    """
    monkeypatch.setattr(sg, "MIN_FIRINGS", 200)
    p = _panel(seed=seed)
    res = _grid_on(p, sg.SIGNALS, horizons=(1, 4))
    assert res.family_p > 0.01, (
        f"noise cleared a 0.01 family-wise gate at seed {seed} "
        f"(p={res.family_p}) - the null is too narrow")


def test_family_wise_p_uses_the_max_across_cells_not_the_best_cell():
    """
    The multiple-testing guard, asserted structurally: adding a pile of pure
    noise cells must never make the reported p SMALLER.
    """
    p = _panel(seed=2)
    one = _grid_on(p, sg.SIGNALS[:1], horizons=(1,))
    many = _grid_on(p, sg.SIGNALS, horizons=(1, 2, 4))
    assert many.family_p >= one.family_p or many.best.t > one.best.t


def test_a_thin_cell_cannot_win():
    """A cell that fires 3 times has been sampled, not measured."""
    p = _panel(S=4, D=40, seed=4)
    rare = np.zeros(p.close.shape, dtype=bool)
    rare[0, :3, 4] = True
    cells = sg.run_grid(p, horizons=(1,),
                        signals=(sg.Signal("rare", "thin", lambda _p: rare),),
                        windows=(("all", p.clocks[0], time(23, 59)),))
    assert all(not c.judgeable for c in cells)
    assert sg.family_wise(cells).best is None


def test_report_states_the_gate_even_with_no_judgeable_cell():
    p = _panel(S=4, D=20, seed=6)
    rare = np.zeros(p.close.shape, dtype=bool)
    rare[0, 0, 1] = True
    cells = sg.run_grid(p, horizons=(1,),
                        signals=(sg.Signal("rare", "thin", lambda _p: rare),),
                        windows=(("all", p.clocks[0], time(23, 59)),))
    text = sg.report(sg.family_wise(cells))
    assert "no judgeable cell" in text


def test_the_null_shifts_sessions_only_and_never_clocks():
    """
    The circular shift must move a firing to a DIFFERENT SESSION at the SAME
    clock. Shifting along the clock axis instead would scramble the very
    time-of-day structure the leave-one-out control exists to hold fixed, and
    the resulting null would be wrong in the flattering direction - wider at
    some clocks, narrower at others, and centred nowhere in particular.

    Asserted by construction: a mask confined to one clock column must produce
    a statistic that only ever reads that same column, whatever the shift.
    """
    rng = np.random.default_rng(13)
    S, D, C = 3, 24, 6
    y = np.zeros((S, D, C))
    y[:, :, 2] = rng.normal(size=(S, D))      # signal ONLY in column 2
    mask = np.zeros((S, D, C), dtype=bool)
    mask[:, :, 2] = rng.random((S, D)) < 0.5

    got = sg._xcorr_all_shifts(mask, y)
    # Every shift's value must be reproducible from column 2 alone.
    want = np.array([float((mask[:, :, 2] * np.roll(y[:, :, 2], -k, axis=1)).sum())
                     for k in range(D)])
    assert np.allclose(got, want, atol=1e-9)

    # And a mask in a column with no data can never borrow from a neighbour.
    other = np.zeros_like(mask)
    other[:, :, 4] = True
    assert np.allclose(sg._xcorr_all_shifts(other, y), 0.0, atol=1e-12)


def test_shifting_preserves_the_firing_count():
    """
    A circular shift relocates firings, it does not create or destroy them.
    An implementation that zero-padded instead of wrapping would quietly
    shrink `n` at large shifts and inflate every null t that used it.
    """
    rng = np.random.default_rng(21)
    mask = rng.random((2, 20, 4)) < 0.4
    valid = np.ones((2, 20, 4))
    n = sg._xcorr_all_shifts(mask, valid)
    assert np.allclose(n, float(mask.sum()))
