"""
The factor sleeve: point-in-time eligibility, formation, and the rebalance.

THE TEST THAT MATTERS IS THE FIRST ONE. A factor book that ranks using the
rebalance day's own bar is the same trap as ranking the intraday universe from
the session being traded - it produces a spectacular curve and raises nothing.
Everything else here is arithmetic that can be checked by inspection; the
look-ahead tests cannot be, which is why they are exhaustive over dates rather
than sampled.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.factor import backtest as fb
from nifty_algo.factor import momentum as mom
from nifty_algo.factor import universe as fu


def _series(start_price, n, drift, seed, start=date(2020, 1, 1), volume=1e6):
    """A daily series with a controlled drift, on weekdays only."""
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
    """
    Twelve names with deliberately ordered drifts, so the momentum ranking has
    a knowable right answer: WIN00 is the strongest, WIN11 the weakest.
    """
    bars = {}
    for i in range(12):
        bars[f"WIN{i:02d}"] = _series(
            100.0 + i, n=700, drift=0.0012 - i * 0.0003, seed=i + 1,
            volume=1e6 * (i + 1))          # also spans a liquidity range
    return bars


# ------------------------------------------------- point-in-time


def test_eligibility_cannot_see_the_rebalance_day_or_later(world):
    """
    THE LOAD-BEARING TEST, exhaustive over rebalance dates.

    Deleting every bar from the rebalance date onward must leave eligibility
    and turnover identical, because both are defined from bars strictly
    before it. An off-by-one here is a look-ahead of one session on every
    holding, and it would show up only as a better curve.
    """
    full = fu.FactorUniverse(world)
    sessions = sorted({d for sd in full.symbols.values() for d in sd.dates})

    for day in fu.month_ends(sessions)[6:]:
        truncated = fu.FactorUniverse(
            {s: f[f.index < pd.Timestamp(day)] for s, f in world.items()})
        a = full.eligible_at(day, 1.0, 0.0, 300)
        b = truncated.eligible_at(day, 1.0, 0.0, 300)
        assert a.symbols == b.symbols, day
        assert a.turnover == pytest.approx(b.turnover), day


def test_scores_cannot_see_the_rebalance_day_or_later(world):
    """The same property one layer up, where the ranking is actually formed."""
    full = fu.FactorUniverse(world)
    sessions = sorted({d for sd in full.symbols.values() for d in sd.dates})
    day = fu.month_ends(sessions)[-1]

    truncated = fu.FactorUniverse(
        {s: f[f.index < pd.Timestamp(day)] for s, f in world.items()})
    syms = full.eligible_at(day, 1.0, 0.0, 300).symbols

    assert (mom.score_universe(full, syms, day)
            == pytest.approx(mom.score_universe(truncated, syms, day)))


def test_a_later_bar_cannot_change_a_finished_rebalance(world):
    """
    Truncating the future leaves every already-finished equity point
    byte-identical - the factor twin of the swing book's truncation test.
    """
    sessions = sorted({t.date() for f in world.values() for t in f.index})
    cut = fu.month_ends(sessions)[-4]

    full = fb.run(world, 1_000_000.0, top_n=4, min_history=300,
                  min_turnover=0.0, min_price=1.0)
    short = fb.run({s: f[f.index <= pd.Timestamp(cut)] for s, f in world.items()},
                   1_000_000.0, top_n=4, min_history=300,
                   min_turnover=0.0, min_price=1.0)

    assert short.equity, "fixture produced no rebalances"
    for (d_a, v_a), (d_b, v_b) in zip(full.equity, short.equity):
        assert d_a == d_b
        assert v_a == pytest.approx(v_b)


# ------------------------------------------------- formation


def test_the_skip_month_is_actually_skipped():
    """
    12-1 must ignore the most recent month. Engineered directly: a flat year
    followed by a sharp final month must score ~0, not the spike.

    Including the skip month mixes short-term REVERSAL into a momentum signal
    with the opposite sign, which is why every serious replication skips it.
    """
    flat = np.full(12 * mom.SESSIONS_PER_MONTH + 1, 100.0)
    spiked = flat.copy()
    spiked[-mom.SESSIONS_PER_MONTH:] = 200.0

    assert mom.formation_return(spiked, 12, 1) == pytest.approx(0.0)
    assert mom.formation_return(spiked, 12, 0) > 0.5      # without the skip


def test_a_short_history_scores_none_rather_than_a_partial_window():
    """
    A partial window computes a SHORTER-horizon signal, which for momentum is
    a reversal signal - and it would be applied to exactly the newest listings.
    """
    assert mom.formation_return(np.full(50, 100.0), 12, 1) is None
    assert mom.formation_return(np.array([]), 12, 1) is None


def test_a_symbol_that_cannot_be_scored_is_omitted_not_zeroed(world):
    """Zero would place it mid-cross-section, which is a decision in disguise."""
    bars = dict(world)
    bars["NEWLY"] = _series(100.0, n=30, drift=0.0, seed=99)
    u = fu.FactorUniverse(bars)
    day = fu.month_ends(sorted({d for sd in u.symbols.values()
                                for d in sd.dates}))[-1]

    scores = mom.score_universe(u, list(bars), day)
    assert "NEWLY" not in scores
    assert len(scores) >= 10


def test_top_n_breaks_ties_deterministically():
    """Dict-order tie-breaking makes a result depend on fetch order."""
    tied = {"ZZZ": 1.0, "AAA": 1.0, "MMM": 1.0}
    assert mom.top_n(tied, 2) == ["AAA", "MMM"]
    assert mom.top_n(tied, 0) == []
    assert mom.top_n({}, 5) == []


# ------------------------------------------------- bands and the ledger


def test_the_rejection_ledger_accounts_for_every_candidate(world):
    u = fu.FactorUniverse(world)
    day = fu.month_ends(sorted({d for sd in u.symbols.values()
                                for d in sd.dates}))[-1]
    e = u.eligible_at(day, min_price=1.0, min_turnover=0.0,
                      min_history=300, band="liquid")
    assert e.accounts_for(len(u.symbols))


def test_the_bands_partition_the_universe(world):
    """
    liquid + midliquid + illiquid must not overlap, or a name is counted in
    two bands and the comparison between them means nothing.
    """
    u = fu.FactorUniverse(world)
    day = fu.month_ends(sorted({d for sd in u.symbols.values()
                                for d in sd.dates}))[-1]
    got = {b: set(u.eligible_at(day, 1.0, 0.0, 300, band=b).symbols)
           for b in ("liquid", "midliquid", "illiquid")}

    assert got["liquid"] & got["midliquid"] == set()
    assert got["midliquid"] & got["illiquid"] == set()
    assert got["liquid"] & got["illiquid"] == set()
    assert got["liquid"], "the liquid band is empty - the test proves nothing"


def test_listed_before_is_the_survivorship_bound(world):
    bars = dict(world)
    bars["LATE"] = _series(100.0, n=400, drift=0.001, seed=7,
                           start=date(2021, 6, 1))
    u = fu.FactorUniverse(bars)

    assert "LATE" not in u.listed_before(date(2021, 1, 1))
    assert "LATE" in u.listed_before(date(2023, 1, 1))


# ------------------------------------------------- the rebalance


def test_the_null_runs_the_identical_book_over_noise(world):
    """
    `seed` must change the SIGNAL and nothing else - same rebalance dates,
    same eligibility, same machinery. If the null changed the schedule too,
    a comparison against it would not isolate the signal.
    """
    real = fb.run(world, 1_000_000.0, top_n=4, min_history=300,
                  min_turnover=0.0, min_price=1.0)
    null = fb.run(world, 1_000_000.0, top_n=4, min_history=300,
                  min_turnover=0.0, min_price=1.0, seed=11)

    assert [d for d, _ in real.equity] == [d for d, _ in null.equity]
    assert real.universe_size == null.universe_size
    assert null.rebalances == real.rebalances


def test_momentum_beats_the_null_on_a_fixture_built_to_have_momentum(world):
    """
    A sanity check on the machinery, NOT evidence about markets. The fixture
    has monotonic drifts by construction, so if ranking on past return cannot
    beat noise here, the ranking is wired wrong.
    """
    real = fb.run(world, 1_000_000.0, top_n=3, min_history=300,
                  min_turnover=0.0, min_price=1.0)
    null = fb.run(world, 1_000_000.0, top_n=3, min_history=300,
                  min_turnover=0.0, min_price=1.0, seed=5)
    assert real.final_value > null.final_value


def test_costs_are_charged_and_reduce_the_book(world):
    priced = fb.run(world, 1_000_000.0, top_n=4, min_history=300,
                    min_turnover=0.0, min_price=1.0)
    assert priced.costs_paid > 0
    assert priced.trades > 0


def test_a_corporate_action_sized_move_is_not_traded(world):
    """
    An unadjusted 1:10 split reads as -90%, and a long-only book must not buy
    the "crash". Asserted on the guard itself rather than through a full run:
    routing a run through it depends on the split landing on a date where that
    symbol happens to be traded, which makes the test pass or fail on fixture
    luck rather than on the behaviour.
    """
    bars = dict(world)
    frame = bars["WIN00"].copy()
    close_col = frame.columns.get_loc("close")
    split_day = frame.index[-1].date()
    frame.iloc[-1, close_col] *= 0.1                     # the 1:10 split
    bars["WIN00"] = frame

    u = fu.FactorUniverse(bars)
    res = fb.FactorResult()
    assert fb._tradeable_price(u, "WIN00", split_day, res) is None
    assert res.skipped_corp_action == 1

    # An ordinary day on the same symbol still prices normally.
    ordinary = frame.index[-2].date()
    assert fb._tradeable_price(u, "WIN00", ordinary, res) is not None


def test_a_symbol_with_no_bar_that_day_is_not_filled_at_a_stale_price(world):
    """A missing bar must leave the slot unused, not fill from yesterday."""
    u = fu.FactorUniverse(world)
    res = fb.FactorResult()
    absent = world["WIN00"].index[-1].date() + timedelta(days=400)
    assert fb._tradeable_price(u, "WIN00", absent, res) is None
    assert res.skipped_no_bar == 1


def test_metrics_are_defined_on_a_degenerate_run():
    """No rebalances must not be a division error."""
    empty = fb.run({}, 1_000_000.0)
    assert empty.rebalances == 0
    assert empty.cagr(1_000_000.0) == 0.0
    assert empty.max_drawdown() == 0.0
    assert empty.sharpe() == 0.0
    assert empty.avg_turnover() == 0.0
    assert empty.longest_drawdown_months() == 0.0


# ------------------------------------------------- the sweep, on the shared harness


def test_the_sweep_reuses_the_shared_statistics_not_its_own(world):
    """
    The factor book must produce `experiment_core.Cell`s, so pooling, the
    fold-resampled interval and the sign test are the SAME code the option
    book uses. Three copies of the statistics is how two books end up unable
    to compare a result.
    """
    from nifty_algo import experiment_core as ec
    from nifty_algo.config import Config
    from nifty_algo.factor import experiment as fx

    cfg = Config()
    cfg.factor.min_history_sessions = 300
    cfg.factor.min_turnover_inr = 0.0
    cfg.factor.min_price = 1.0
    cfg.factor.top_n = 3

    sw = fx.sweep(cfg, world, 1_000_000.0,
                  variants=(fx.VARIANTS[0], fx.VARIANTS[-1]),
                  train_months=6, test_months=3)

    assert sw.cells, "fixture produced no folds"
    assert all(isinstance(c, ec.Cell) for c in sw.cells)
    assert {c.phase for c in sw.cells} == {"train", "test"}
    # Both phases are EXECUTED, not carried as labels - that is what makes
    # selecting on train and scoring on test possible at all.
    assert any(c.phase == "train" and c.trades > 0 for c in sw.cells)


def test_the_null_is_a_first_class_variant():
    """Not a footnote. It is run and reported on the same footing."""
    from nifty_algo.factor import experiment as fx
    assert fx.NULL_KEY in [v.key for v in fx.VARIANTS]


def test_the_cell_unit_is_a_monthly_return_not_an_r(world):
    """
    `expectancy_r` carries a RETURN here so the shared statistics apply. It
    must be return-shaped - a few percent - not R-shaped. A factor month
    reading like an R multiple would invite exactly the cross-book comparison
    the module docstring forbids.
    """
    from nifty_algo.config import Config
    from nifty_algo.factor import experiment as fx

    cfg = Config()
    cfg.factor.min_history_sessions = 300
    cfg.factor.min_turnover_inr = 0.0
    cfg.factor.min_price = 1.0
    cfg.factor.top_n = 3

    sw = fx.sweep(cfg, world, 1_000_000.0, variants=(fx.VARIANTS[0],),
                  train_months=6, test_months=3)
    scored = [c for c in sw.cells if c.trades > 0]
    assert scored
    for c in scored:
        assert -0.5 < c.expectancy_r < 0.5, c
        assert "cagr" in c.extra and "turnover" in c.extra


def test_variants_do_not_leak_into_each_other():
    """
    `Variant.configure` deep-copies. If it did not, the sweep would measure a
    cumulative stack of changes rather than each variant alone.
    """
    from nifty_algo.config import Config
    from nifty_algo.factor import experiment as fx

    base = Config()
    for v in fx.VARIANTS:
        v.configure(base)
    assert base.factor.band == "all"
    assert base.factor.formation == "mom12_1"
    assert base.factor.hold_months == 1
    assert base.factor.random_seed is None
    assert base.factor.listed_only is False
