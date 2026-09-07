"""
The swing ticket: entry, stop, target.

Most of these are INVARIANTS rather than assertions about one hand-built
chart. A setup detector tested only on a chart engineered to trigger it tells
you the detector fires; what actually matters here is that whatever it fires
on, the resulting ticket is geometrically sane - the stop is below the entry
and inside the ATR band, the target is far enough to be worth taking and near
enough to be reachable, and none of it was computed from a bar that had not
happened yet.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from conftest import daily_bars, trending

from nifty_algo.config import Config
from nifty_algo.signals import atr
from nifty_algo.swing import setup as S


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def uptrend() -> pd.DataFrame:
    return daily_bars(trending())


# ---------------------------------------------------------------- guards

def test_too_few_bars_is_refused_with_a_reason(cfg):
    df = daily_bars(trending(n=30))
    found, note = S.detect("T", df, cfg)
    assert found is None
    assert "daily bars" in note


def test_flat_tape_produces_no_setup(cfg):
    """Zero range means zero ATR, and every distance would be a divide by it."""
    df = daily_bars([100.0] * 140)
    df[["open", "high", "low", "close"]] = 100.0
    found, note = S.detect("T", df, cfg)
    assert found is None
    assert "ATR" in note


# ---------------------------------------------------------------- invariants

#: How many series the geometry invariant is asserted on.
GEOMETRY_SEED_COUNT = 10


def _seeds_with_setups(want: int = GEOMETRY_SEED_COUNT,
                       limit: int = 200) -> list[int]:
    """
    Seeds that actually produce a setup - FOUND, not guessed.

    THIS INVARIANT USED TO RUN ON 6 SERIES WHILE CLAIMING 10. It was
    parametrized over a hand-picked `[1, 2, 3, 5, 7, 11, 13, 17, 19, 23]` and
    skipped when `detect()` found nothing, and on the current detector seeds 1,
    13, 19 and 23 find nothing - so 40% of the ticket-geometry claim silently
    did not execute, and the skip message said as much ("the geometry claim is
    vacuous") to anyone who read the four skips in a green run. They were 4 of
    the suite's 5.

    A fresh hand-picked list would rot the same way, because a seed stops
    qualifying whenever the detector moves and a skip is invisible. Discovering
    them costs about 12 `detect()` calls on 140 bars - 88 of the first 120
    seeds qualify - and it keeps the invariant at 10/10 by construction.

    Returns fewer than `want` only if the detector has degraded far enough that
    `limit` series cannot produce them, which
    `test_the_geometry_invariant_runs_on_every_seed_it_claims` turns into a
    failure rather than a quieter suite.
    """
    cfg = Config()
    found: list[int] = []
    for seed in range(1, limit + 1):
        setup, _ = S.detect("T", daily_bars(trending(seed=seed)), cfg)
        if setup is not None:
            found.append(seed)
            if len(found) == want:
                break
    return found


GEOMETRY_SEEDS = _seeds_with_setups()


def test_the_geometry_invariant_runs_on_every_seed_it_claims(cfg):
    """
    A parametrized invariant that quietly shrinks is worse than a missing one.

    The count is asserted separately from the invariant itself so that "the
    detector stopped finding setups" cannot present as a smaller passing run.
    """
    assert len(GEOMETRY_SEEDS) == GEOMETRY_SEED_COUNT, (
        f"only {len(GEOMETRY_SEEDS)} of {GEOMETRY_SEED_COUNT} seeds produce a "
        f"setup - the detector has degraded, or `trending()` has")
    for seed in GEOMETRY_SEEDS:
        setup, note = S.detect("T", daily_bars(trending(seed=seed)), cfg)
        assert setup is not None, f"seed {seed} stopped qualifying: {note}"


@pytest.mark.parametrize("seed", GEOMETRY_SEEDS)
def test_ticket_geometry_holds_for_any_setup_found(cfg, seed):
    df = daily_bars(trending(seed=seed))
    found, note = S.detect("T", df, cfg)
    # ASSERT, DO NOT SKIP. The seeds were chosen for producing a setup, so one
    # that produces none is a regression in the detector - not a reason to
    # assert the geometry on fewer series than the test name promises.
    assert found is not None, f"seed {seed} produced no setup: {note}"

    a = float(atr(df, cfg.swing.atr_period).iloc[-1])
    close = float(df["close"].iloc[-1])
    swing = cfg.swing

    # A long entry is a STOP-BUY: it must sit above where price is now, or
    # you are not waiting for confirmation, you are just buying.
    assert found.entry > close

    # ...but not so far above that it is somebody else's trade next week.
    assert found.entry - close <= swing.max_entry_distance_atr * a + 1e-9

    # The stop is below the entry and inside the band. Outside it, the stop is
    # either inside daily noise or too wide for the risk budget to size.
    assert found.stop < found.entry
    distance = (found.entry - found.stop) / a
    assert swing.swing_atr_stop_min_multiple - 1e-9 <= distance
    assert distance <= swing.swing_atr_stop_multiple + 1e-9

    # The target is worth taking and possible to reach.
    assert found.target >= found.entry + swing.target_min_atr * a - 1e-9
    assert found.target <= found.entry + swing.target_max_atr * a + 1e-9

    assert found.reward_risk == pytest.approx(
        (found.target - found.entry) / (found.entry - found.stop))
    assert 0.0 < found.quality <= 1.0
    assert found.reasons, "a setup with no stated reason is not explainable"


def test_reward_risk_arithmetic():
    s = S.SwingSetup("T", "breakout", "Level breakout", entry=100.0,
                     stop=95.0, target=115.0, trigger_note="", quality=0.5)
    assert s.risk_points == pytest.approx(5.0)
    assert s.reward_points == pytest.approx(15.0)
    assert s.reward_risk == pytest.approx(3.0)
    assert s.stop_pct == pytest.approx(0.05)
    assert s.target_pct == pytest.approx(0.15)


def test_reward_risk_is_zero_rather_than_infinite_when_the_stop_is_the_entry():
    s = S.SwingSetup("T", "x", "x", entry=100.0, stop=100.0, target=115.0,
                     trigger_note="", quality=0.5)
    assert s.reward_risk == 0.0


# ---------------------------------------------------------------- look-ahead

def test_the_ticket_does_not_move_when_later_bars_change(cfg):
    """
    The signal bar's ticket must be a function of the bars up to it and
    nothing else. `find_pivots` uses a CENTRED window, so a level formed in
    the last few bars cannot be confirmed yet - and a detector that reached
    past the end of its frame would produce a different answer here.
    """
    full = daily_bars(trending(n=200))
    cut = 150

    before, _ = S.detect("T", full.iloc[:cut], cfg)

    crash = full.iloc[:cut].copy()
    tail = daily_bars(trending(n=50, drift=-0.02, seed=99,
                               start_price=float(crash["close"].iloc[-1])),
                      start=datetime(2027, 1, 1))
    alternate = pd.concat([crash, tail])

    after, _ = S.detect("T", alternate.iloc[:cut], cfg)

    assert (before is None) == (after is None)
    if before is not None:
        assert before.entry == pytest.approx(after.entry)
        assert before.stop == pytest.approx(after.stop)
        assert before.target == pytest.approx(after.target)
        assert before.key == after.key


def test_a_level_from_the_last_few_bars_cannot_be_used_yet(cfg):
    """
    The centred pivot window is what buys the no-look-ahead guarantee, so if
    it were ever widened to a trailing window this test should fail.
    """
    df = daily_bars(trending(n=160))
    ctx = S._context(df, cfg)
    newest_confirmable = df.index[-(cfg.swing.pivot_lookback + 1)]
    recent_highs = df.loc[df.index > newest_confirmable, "high"]
    for level in ctx["resistances"]:
        # No resistance may sit exactly on an unconfirmable recent high.
        assert not any(abs(level.price - h) < 1e-9 for h in recent_highs)


# ---------------------------------------------------------------- the cap

def test_an_unreachable_target_is_capped_and_says_so(cfg):
    """
    A resistance eight ATR overhead is a real level and a fictional target.
    The ticket must show the reachable number, and must not quietly pretend
    the cap was the level all along.
    """
    df = daily_bars(trending(n=160, drift=0.006, vol=0.004, seed=4))
    found, _ = S.detect("T", df, cfg)
    if found is None:
        pytest.skip("no setup on this series")

    a = float(atr(df, cfg.swing.atr_period).iloc[-1])
    assert found.target <= found.entry + cfg.swing.target_max_atr * a + 1e-9

    capped = [r for r in found.reasons if "capped at" in r]
    if found.target == pytest.approx(found.entry + cfg.swing.target_max_atr * a):
        assert capped, "the target was capped but the ticket does not say so"


def test_raising_the_cap_lets_a_further_target_through(cfg):
    df = daily_bars(trending(n=160, drift=0.006, vol=0.004, seed=4))
    tight, _ = S.detect("T", df, cfg)
    if tight is None:
        pytest.skip("no setup on this series")

    cfg.swing.target_max_atr = 20.0
    loose, _ = S.detect("T", df, cfg)
    assert loose is not None
    assert loose.target >= tight.target


# ---------------------------------------------------------------- archetypes

def test_a_clean_uptrend_produces_a_long_setup(cfg, uptrend):
    found, note = S.detect("T", uptrend, cfg)
    assert found is not None, note
    assert found.key in {"breakout", "pullback", "squeeze", "reclaim"}


def test_a_downtrend_does_not_produce_a_breakout(cfg):
    """Long only, and a breakout setup requires the fast EMA above the slow."""
    df = daily_bars(trending(n=160, drift=-0.004, seed=21))
    found, _ = S.detect("T", df, cfg)
    if found is not None:
        assert found.key != "breakout"


def test_confluence_is_recorded_when_more_than_one_archetype_agrees(cfg):
    for seed in range(1, 40):
        df = daily_bars(trending(seed=seed))
        found, _ = S.detect("T", df, cfg)
        if found and found.detail.get("confluence"):
            assert any("Also reads as" in r for r in found.reasons)
            return
    pytest.skip("no confluent series in the sample")
