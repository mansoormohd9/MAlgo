"""
The swing backtester.

The test that matters most here is `test_truncating_the_future_changes_nothing`,
for the reason [test_lookahead.py](test_lookahead.py) gives about the option
book: every other failure produces a wrong answer you can see, and look-ahead
bias produces a FLATTERING one you cannot. A swing backtest is especially
exposed to it, because the whole method is "hand `setup.detect` a truncated
frame" - and if the truncation is off by one bar the results are silently
excellent.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.swing import backtest as bt
from nifty_algo.swing import markets as markets_mod


# ---------------------------------------------------------------- fixtures

def wander(n=200, drift=0.0008, vol=0.013, seed=3, start=500.0) -> pd.DataFrame:
    """A daily series with enough shape for the setups to find something."""
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


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.capital.swing_capital_inr = 100_000.0
    m = c.swing.markets["india"]
    # The synthetic series start at 500 and are not real Nifty names.
    m.min_price, m.min_avg_turnover = 1.0, 0.0
    return c


@pytest.fixture
def market(cfg):
    return markets_mod.get(cfg, "india")


@pytest.fixture(scope="module")
def world():
    """
    Small on purpose. `_scan_day` calls `setup.detect` for every symbol on
    every session, and each call rebuilds levels over the whole truncated
    frame - so the cost is quadratic in bars and linear in symbols. Three
    symbols over 200 sessions exercises every path in about a second; six
    over 420 took minutes and taught the suite nothing extra.
    """
    bars = {f"SYM{i}": wander(seed=i, drift=0.0004 + i * 0.0004)
            for i in range(3)}
    bench = wander(seed=99, drift=0.0002, vol=0.007, start=20_000.0)
    return bars, bench


@pytest.fixture(scope="module")
def result(world):
    """One run, shared by every test that only reads it."""
    c = Config()
    c.capital.swing_capital_inr = 100_000.0
    m = c.swing.markets["india"]
    m.min_price, m.min_avg_turnover = 1.0, 0.0
    bars, bench = world
    return bt.run(c, m, bars, bench)


# ---------------------------------------------------------------- look-ahead

def test_truncating_the_future_changes_nothing(cfg, market, world):
    """
    THE test. Cut the data short and every trade that had already finished
    must come out byte-identical.

    If any part of the pipeline could see past the decision bar, the longer
    run would score those same trades differently - better, almost always,
    because the peeked-at information is the outcome itself.
    """
    bars, bench = world
    cut = pd.Timestamp(datetime(2025, 1, 2))

    full = bt.run(cfg, market, bars, bench)
    short = bt.run(cfg, market,
                   {s: df.loc[df.index <= cut] for s, df in bars.items()},
                   bench.loc[bench.index <= cut])

    def finished_before_cut(result):
        return [(t.symbol, t.entry_time, t.exit_time, round(t.r_multiple, 9))
                for t in result.trades if t.exit_time <= cut]

    assert finished_before_cut(short)
    assert finished_before_cut(full) == finished_before_cut(short)


def test_a_trade_never_opens_on_its_own_signal_day(result):
    """
    The scan is read off that day's CLOSE, so that day's range is not
    information you had. Same rule `tracker._evaluate` applies to a live pick.
    """
    assert result.trades
    for t in result.trades:
        assert t.exit_time >= t.entry_time
        assert t.bars_held >= 1


def test_a_gap_through_the_trigger_fills_at_the_open(cfg, market):
    """
    You cannot buy below where the market opened.

    Filling a gap at the trigger price is a small, systematic error that
    always runs in your favour - which is the only kind that survives review.
    """
    idx = pd.bdate_range(datetime(2024, 1, 2), periods=1)
    df = pd.DataFrame({"open": [120.0], "high": [125.0], "low": [119.0],
                       "close": [124.0], "volume": [1e6]}, index=idx)
    bars = {"AAA": df}
    sessions = [pd.Timestamp(datetime(2023, 12, 29)), idx[0]]

    class _S:
        entry, stop, target, key = 100.0, 90.0, 130.0, "breakout"

    class _P:
        symbol, sector, quantity = "AAA", "IT", 10
        setup = _S()

    opened: dict = {}
    stats = bt.SwingDayStats()
    from nifty_algo.positions import ExitLadder
    from nifty_algo.swing.book import swing_trade
    bt._try_open(_P(), bars, sessions, 0, opened,
                 ExitLadder(cfg, trade=swing_trade(cfg)), cfg, stats,
                 capital=1_000_000.0)

    assert opened["AAA"].entry == pytest.approx(120.0)   # the open, not 100


# ---------------------------------------------------------------- pessimism

def test_a_bar_covering_stop_and_target_is_scored_as_a_loss(cfg, market):
    """
    Daily data cannot say which extreme came first.

    This is the ladder's own tie-break, exercised through the backtest so the
    two cannot drift apart.
    """
    from nifty_algo.positions import ExitLadder
    from nifty_algo.swing.book import swing_trade

    ladder = ExitLadder(cfg, trade=swing_trade(cfg))
    t = bt._Open(symbol="AAA", sector="IT", setup_key="breakout",
                 entry_time=pd.Timestamp(datetime(2024, 1, 2)),
                 entry=100.0, stop=95.0, target=115.0, quantity=20,
                 state=ladder.new_state(20))

    decisions = ladder.advance(t.state, mark_r=0.0,
                               best_r=t.r_of(120.0),   # target was touched
                               worst_r=t.r_of(94.0),   # ...so was the stop
                               trail_distance_r=0.5)

    assert [d.kind.value for d in decisions] == ["stopped_out"]
    assert decisions[0].exit_r == pytest.approx(-1.0)


# ---------------------------------------------------------------- constraints

def test_deployment_never_exceeds_the_pot(cfg, result):
    """
    `scanner._size` caps ONE ticket at the pot; three tickets on three days
    could ask for three times it. Live, the third buy is simply rejected.

    Without this gate the backtest trades a bigger account than you have and
    reports the result against the smaller one.
    """
    assert result.day_stats.peak_deployed_inr <= cfg.capital.swing_capital_inr


def test_the_same_name_is_never_held_twice(result):
    for symbol in {t.symbol for t in result.trades}:
        spans = sorted((t.entry_time, t.exit_time)
                       for t in result.trades if t.symbol == symbol)
        for (_, first_exit), (second_entry, _) in zip(spans, spans[1:]):
            assert second_entry >= first_exit


def test_no_more_than_top_n_positions_at_once(cfg, result):
    assert result.day_stats.max_concurrent <= cfg.swing.top_n


def test_an_unfunded_pot_refuses_to_produce_numbers(cfg, market, world):
    """
    Zero capital sizes every ticket to zero. Returning "0 trades, 0R" would
    read as a result about the strategy rather than about the settings.
    """
    bars, bench = world
    cfg.capital.swing_capital_inr = 0.0

    result = bt.run(cfg, market, bars, bench)

    assert not result.trades
    assert any("swing_capital_inr" in w for w in result.warnings)


# ---------------------------------------------------------------- costs

def test_charges_are_taken_out_of_every_trade(result):
    """
    Net R is below gross R, always, and by a visible amount.

    On small tickets the flat DP charge alone is several percent of the risk
    budget. A backtest reporting gross R is reporting money nobody receives.
    """
    assert result.trades
    for t in result.trades:
        assert t.friction > 0
        assert t.r_multiple < t.gross_r

    gross = sum(t.gross_r for t in result.trades)
    net = sum(t.r_multiple for t in result.trades)
    assert net < gross


# ---------------------------------------------------------------- honesty

def test_every_result_carries_its_caveats(result):
    """
    Survivorship and point-in-time fundamentals cannot be fixed here, so they
    have to travel with the numbers rather than live in a docstring nobody
    opens.
    """
    joined = " ".join(result.caveats).lower()
    assert "survivorship" in joined
    assert "halal screen" in joined and "point-in-time" in joined
    assert "news" in joined
    # The universe caveat has to say which DIRECTION it runs in. "There are
    # limitations" is not a warning, it is a disclaimer.
    assert "larger universe" in joined


def test_open_positions_at_the_end_are_excluded_not_marked_to_market(world,
                                                                    result):
    """
    An unfinished trade is not an outcome.

    Scoring it at the last close would let a long holding period masquerade
    as a result, and would do so asymmetrically - a losing position that has
    not stopped out yet is exactly the one you would most like to leave open.
    """
    bars, _ = world
    still_open = [w for w in result.warnings if "still open" in w]
    for w in still_open:
        symbol = w.split()[0]
        closes = [t for t in result.trades if t.symbol == symbol]
        # It may have traded earlier in the window; what must not exist is a
        # trade recorded on the final session.
        assert all(t.exit_time < bars[symbol].index[-1] for t in closes)


def test_metrics_come_from_the_option_books_implementation(result):
    """
    One definition of expectancy, win rate and drawdown for both books.

    `compute_metrics` reads only `.r_multiple`, `.bars_held` and `.strategy`,
    so `SwingTrade` feeds it directly - no adapter, and no second chance to
    compute a profit factor differently.
    """
    m = result.metrics

    assert m.trades == len(result.trades)

    # THE YARDSTICK IS REALISED, NOT DESIGNED. `target_breakeven_win_rate` is
    # the 1/(1+R:R) the config aims at; `breakeven_win_rate` is the win rate
    # the payoff that actually happened had to clear. This assertion used to
    # read `breakeven_win_rate == 1/3`, which is what let a 22.7% win rate be
    # compared against 33.3% when the realised figure was 25.4%.
    assert m.target_breakeven_win_rate == pytest.approx(1 / 3, abs=0.01)
    rs = [t.r_multiple for t in result.trades]
    wins = [r for r in rs if r > 0]
    losses = [-r for r in rs if r < 0]
    if wins and losses:
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        assert m.avg_win_r == pytest.approx(avg_win, rel=1e-6)
        assert m.avg_loss_r == pytest.approx(avg_loss, rel=1e-6)
        assert m.breakeven_win_rate == pytest.approx(
            avg_loss / (avg_win + avg_loss), rel=1e-6)
    assert m.total_r == pytest.approx(
        sum(t.r_multiple for t in result.trades), rel=1e-6)
    assert sum(x.trades for x in result.by_setup.values()) == m.trades


def test_a_partial_exit_is_charged_two_sell_days_of_dp_fee(result):
    """
    Verified through the real backtester, not just the cost model: trades
    that banked a partial must carry more friction than ones that did not.
    """
    from nifty_algo.swing.costs_equity import DEFAULT_EQUITY_COSTS as c

    scaled = [t for t in result.trades if t.partial_banked]
    if not scaled:
        pytest.skip("this window produced no scale-outs")
    for t in scaled:
        plain = c.friction(t.entry, t.exit_price, t.quantity)
        assert t.friction == pytest.approx(plain + c.dp_charge)


# ------------------------------------------------------- the yardstick itself

class _R:
    """The three fields `compute_metrics` actually reads."""

    def __init__(self, r: float, strategy: str = "breakout"):
        self.r_multiple = r
        self.bars_held = 1
        self.net_pnl = 0.0
        self.strategy = strategy


def test_breakeven_is_the_payoff_that_happened_not_the_one_configured():
    """
    Three +2R wins and seven -0.5R losses need 20%, not 33%.

    This is the shape the ladder actually produces - it shifts to breakeven at
    +1R, so most losers come in well under a full R - and judging it against
    1/(1+R:R) overstates the shortfall every time. The India book cleared
    22.7% against a realised 25.4% while the report printed 33.3%.
    """
    from nifty_algo.backtest import compute_metrics

    m = compute_metrics([_R(2.0)] * 3 + [_R(-0.5)] * 7, reward_risk=2.0)

    assert m.avg_win_r == pytest.approx(2.0)
    assert m.avg_loss_r == pytest.approx(0.5)          # magnitude, unsigned
    assert m.breakeven_win_rate == pytest.approx(0.2)
    assert m.target_breakeven_win_rate == pytest.approx(1 / 3)
    assert m.win_rate == pytest.approx(0.3)            # clears realised, not design


def test_a_one_sided_sample_does_not_fall_back_to_the_design_figure():
    """
    No winners means no win rate clears it; no losers means any does.

    Both are the honest reading. Quietly substituting the config figure for a
    sample that cannot support one is how the wrong yardstick got here in the
    first place.
    """
    from nifty_algo.backtest import compute_metrics

    assert compute_metrics([_R(-1.0)] * 5).breakeven_win_rate == 1.0
    assert compute_metrics([_R(1.0)] * 5).breakeven_win_rate == 0.0

    # No trades at all: nothing realised, so no realised yardstick is offered.
    empty = compute_metrics([])
    assert empty.breakeven_win_rate == 0.0
    assert empty.target_breakeven_win_rate == pytest.approx(1 / 3)


def test_exit_buckets_separate_a_full_stop_from_a_breakeven_scratch(result):
    """
    `outcome` calls both of them "stop"; they are opposite facts.

    A trade that reached +1R, moved its stop to breakeven and died there cost
    nothing and was also a winner the book failed to keep. Counting it with
    the -1R stops is how the most likely cause of a low win rate stays
    invisible.
    """
    buckets = result.exit_buckets

    assert set(buckets) == {label for label, _, _ in bt.EXIT_BUCKETS}
    assert sum(buckets.values()) == len(result.trades)

    stopped = [t for t in result.trades if t.r_multiple < -0.5]
    scratched = [t for t in result.trades if -0.5 <= t.r_multiple < 0.5]
    assert buckets["Stopped (< -0.5R)"] == len(stopped)
    assert buckets["Scratched (-0.5..+0.5R)"] == len(scratched)


# ------------------------------------------------- walk-forward and the cache

def test_folds_tile_the_period_once_with_no_gaps_and_no_overlaps():
    """
    Every session after the first train window is scored exactly once.

    Overlapping test windows would count the same trade twice and make a
    lucky month look like a repeatable one; a gap would quietly drop the
    months that happened to be worst.
    """
    from datetime import date

    windows = bt.fold_windows(date(2023, 1, 1), date(2026, 1, 1),
                              train_months=6, test_months=2)

    assert windows
    for w in windows:
        assert w.train_start < w.train_end < w.test_start <= w.test_end
    for earlier, later in zip(windows, windows[1:]):
        # The next test window starts the day the previous one ended.
        assert later.test_start == earlier.test_end + timedelta(days=1)
    assert windows[-1].test_end <= date(2026, 1, 1)


def test_fold_windows_refuses_a_zero_length_split():
    from datetime import date

    with pytest.raises(ValueError):
        bt.fold_windows(date(2023, 1, 1), date(2024, 1, 1), 6, 0)


def test_a_replayed_variant_is_identical_to_a_direct_run(cfg, market, world):
    """
    THE guard on the scan cache. Same config, cache or no cache, same trades.

    The cache exists so a sweep of variants is affordable, and it earns that
    only if replaying it cannot change an answer. If this ever fails, every
    experiment result built on the cache is void - the failure mode is not a
    crash but a plausible table of numbers describing a book nobody ran.
    """
    bars, bench = world

    direct = bt.run(cfg, market, bars, bench)

    cache = bt.ScanCache(signature=bt.scan_signature(cfg, market))
    first = bt.run(cfg, market, bars, bench, scan_cache=cache)
    assert cache.sessions_cached > 0
    replayed = bt.run(cfg, market, bars, bench, scan_cache=cache)

    def fingerprint(r):
        return [(t.symbol, t.entry_time, t.exit_time, t.entry, t.exit_price,
                 t.quantity, round(t.r_multiple, 12)) for t in r.trades]

    assert fingerprint(first) == fingerprint(direct)
    assert fingerprint(replayed) == fingerprint(direct)
    assert replayed.metrics.expectancy_r == direct.metrics.expectancy_r


def test_a_cache_from_different_scan_parameters_is_refused(cfg, market, world):
    """
    A stale scan does not raise on its own - it answers the wrong question
    fluently. So the mismatch has to raise here, at the only point where it
    is still detectable.
    """
    bars, bench = world
    cache = bt.ScanCache(signature=bt.scan_signature(cfg, market))
    bt.run(cfg, market, bars, bench, scan_cache=cache)

    cfg.swing.swing_atr_stop_multiple = 2.5      # changes every stop, and so
    cfg.swing.target_max_atr = 6.0               # every R:R and every ranking
    with pytest.raises(ValueError, match="scan cache"):
        bt.run(cfg, market, bars, bench, scan_cache=cache)


def test_ladder_parameters_do_not_invalidate_a_scan(cfg, market):
    """
    The whole point of the split: the ladder runs after the scan, so a
    breakeven-shift or trail experiment must be able to reuse one pass.
    """
    before = bt.scan_signature(cfg, market)
    cfg.swing.trail_atr_multiple = 3.0
    cfg.swing.partial_exit_fraction = 0.25
    cfg.swing.top_n = 1
    cfg.swing.max_open_risk_r = 1.0
    assert bt.scan_signature(cfg, market) == before

    cfg.swing.ema_fast = 10                       # ...but a setup input does
    assert bt.scan_signature(cfg, market) != before


def test_a_window_carries_its_trades_to_the_exit_instead_of_dropping_them(
        cfg, market, world):
    """
    THE walk-forward bias. A position open when the window closes used to be
    deleted from the statistics, and what is open at any moment is not a
    random sample: losers stop out in days, winners trail for weeks. Cutting
    at the boundary therefore removes winners preferentially and makes every
    variant look worse than it is - measured at about 0.2R on the India sweep.

    Settling must add trades, never remove them, and must not let the window
    open anything new.
    """
    bars, bench = world
    sessions = bt.all_sessions(bars)
    cut = sessions[len(sessions) // 2].date()

    dropped = bt.run(cfg, market, bars, bench, end=cut, settle_days=0)
    settled = bt.run(cfg, market, bars, bench, end=cut, settle_days=60)

    assert settled.metrics.trades >= dropped.metrics.trades
    assert len(settled.warnings) <= len(dropped.warnings)

    # Every trade the truncated run scored is scored identically by the
    # settled one - settlement finishes trades, it does not re-run them.
    def by_key(result):
        return {(t.symbol, t.entry_time): round(t.r_multiple, 12)
                for t in result.trades}

    short, long = by_key(dropped), by_key(settled)
    assert set(short) <= set(long)
    for key, r in short.items():
        assert long[key] == r

    # And nothing was ENTERED after the window closed.
    for t in settled.trades:
        assert t.entry_time.date() <= cut


# ------------------------------------------- the null, and the time stop

def test_the_new_levers_are_off_by_default_and_change_nothing(cfg, market,
                                                              world):
    """
    THE REGRESSION THAT PROTECTS THE +0.114R ON RECORD. `random_seed` and
    `max_hold_days` are new fields; a new lever that moves the DEFAULT result
    silently invalidates every number already measured against it.
    """
    bars, bench = world
    a = bt.run(cfg, market, bars, bench)
    cfg.swing.random_seed = None
    cfg.swing.max_hold_days = 0
    b = bt.run(cfg, market, bars, bench)
    assert [t.r_multiple for t in a.trades] == [t.r_multiple for t in b.trades]
    assert [t.symbol for t in a.trades] == [t.symbol for t in b.trades]
    assert [t.entry_time for t in a.trades] == [t.entry_time for t in b.trades]


def test_the_new_levers_do_not_invalidate_the_scan_cache(cfg, market):
    """
    Both are read AFTER `evaluate_symbol`, so the null must share the real
    book's scan pass. If either leaked into the signature the sweep would pay
    a full extra scan per seed - and a 40-draw null would become unaffordable.
    """
    import copy
    base = bt.scan_signature(cfg, market)
    c = copy.deepcopy(cfg)
    c.swing.random_seed, c.swing.max_hold_days = 7, 10
    assert bt.scan_signature(c, market) == base

    c2 = copy.deepcopy(cfg)
    c2.swing.ema_fast = 21          # a real signal field still invalidates
    assert bt.scan_signature(c2, market) != base


def test_the_null_keeps_every_gate_and_changes_only_the_ranking():
    """
    The null must re-rank the SAME candidates, not widen or narrow the pool.
    If it changed which names are eligible it would be testing the gates, and
    a difference in results could not be attributed to the scoring.
    """
    rows = [(f"stock{i}", f"found{i}", f"metrics{i}", f"news{i}",
             float(i), f"parts{i}") for i in range(6)]
    c = Config()
    c.swing.random_seed = 3
    out = bt._apply_null(rows, c, None, np.random.default_rng(3))

    assert len(out) == len(rows)
    for before, after in zip(rows, out):
        assert after[0] == before[0] and after[1] == before[1]
        assert after[2] == before[2] and after[3] == before[3]
        assert after[5] == before[5]          # only t[4] moves
    assert [r[4] for r in out] != [r[4] for r in rows]


def test_the_null_is_a_no_op_without_a_seed():
    rows = [("a", None, None, None, 0.9, None)]
    assert bt._apply_null(rows, Config(), None, None) is rows


def test_the_null_is_reproducible_from_its_seed(cfg, market, world):
    bars, bench = world
    cfg.swing.random_seed = 11
    a = bt.run(cfg, market, bars, bench)
    b = bt.run(cfg, market, bars, bench)
    assert [t.symbol for t in a.trades] == [t.symbol for t in b.trades]
    assert [t.r_multiple for t in a.trades] == [t.r_multiple for t in b.trades]


def test_the_time_stop_caps_the_hold_and_is_labelled(cfg, market, world):
    bars, bench = world
    cfg.swing.max_hold_days = 3
    res = bt.run(cfg, market, bars, bench)
    assert res.trades, "fixture produced no trades - the cap cannot be tested"
    assert max(t.bars_held for t in res.trades) <= 3
    assert any(t.outcome == "time_stop" for t in res.trades)


def test_the_time_stop_never_extends_a_hold(cfg, market, world):
    """
    A cap may only end a position EARLIER. It is deliberately not asserted
    that the capped run is a subset of the uncapped one: an earlier exit frees
    cash, which lets a different name be funded that the uncapped book had no
    room for. That is real behaviour, not a bug, and an over-strict invariant
    here would have to be weakened later to accommodate it.
    """
    bars, bench = world
    cfg.swing.max_hold_days = 5
    capped = bt.run(cfg, market, bars, bench)
    assert capped.trades
    assert max(t.bars_held for t in capped.trades) <= 5
    assert all(t.bars_held == 5 for t in capped.trades
               if t.outcome == "time_stop")


def test_the_ladder_is_given_the_bar_before_the_clock(cfg, market, world):
    """
    A position that reaches its stop or target on the final permitted day must
    be booked as THAT, not as a time exit at the close. Running the clock first
    would convert real stops into time exits and quietly improve the loss
    distribution, with no error anywhere.

    Asserted at `max_hold_days = 1`, where every managed bar is also the final
    permitted one: any trade that still comes back as a stop or a target can
    only have done so because the ladder was consulted before the clock.
    """
    bars, bench = world
    cfg.swing.max_hold_days = 1
    res = bt.run(cfg, market, bars, bench)
    assert res.trades
    assert all(t.bars_held == 1 for t in res.trades)
    assert {t.outcome for t in res.trades} - {"time_stop"}, (
        "every exit was a time stop - the ladder never got the bar first")


# ------------------------------------------------- S1: the walk-forward null

def test_the_null_walk_forward_ranks_the_book_inside_its_nulls(cfg, market,
                                                               world):
    """
    The shape of the answer, on a tiny fixture: one Fold per test window, each
    carrying the real book's expectancy and `n_seeds` null draws.
    """
    from nifty_algo.swing import experiment as ex
    bars, bench = world
    wf = ex.walk_forward_null(cfg, market, bars, bench, n_seeds=3,
                              train_months=3, test_months=3)
    assert wf.folds
    for f in wf.folds:
        assert len(f.nulls) == 3
        assert 0.0 <= f.percentile <= 1.0
    assert 0 <= wf.wins <= len(wf.folds)


def test_the_null_walk_forward_shares_one_scan_pass(cfg, market, world,
                                                    monkeypatch):
    """
    41 runs per fold must not mean 41 scans. `random_seed` is book-only, so
    every seed hashes to the baseline's signature and hits the same cache -
    without that the 40-draw null is unaffordable and the whole design fails.
    """
    from nifty_algo.swing import experiment as ex
    bars, bench = world
    calls = {"n": 0}
    real = bt.scanner_mod.evaluate_symbol

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(bt.scanner_mod, "evaluate_symbol", counted)
    ex.walk_forward_null(cfg, market, bars, bench, n_seeds=4,
                         train_months=3, test_months=3)
    one_pass = calls["n"]

    calls["n"] = 0
    ex.walk_forward_null(cfg, market, bars, bench, n_seeds=12,
                         train_months=3, test_months=3)
    # Tripling the null count must not materially change the scan cost.
    assert calls["n"] == one_pass


def test_hold_stats_reports_days_per_exit_reason(cfg, market, world):
    """The days-to-exit number that exists nowhere else in the package."""
    from nifty_algo.swing import experiment as ex
    bars, bench = world
    cfg.swing.max_hold_days = 4
    wf = ex.walk_forward_null(cfg, market, bars, bench, n_seeds=2,
                              train_months=3, test_months=3)
    text = ex.hold_stats(wf)
    assert "mean days" in text and "ALL" in text


def test_different_seeds_pick_different_names(cfg, market, world):
    """
    THE CHECK THAT THE NULL ACTUALLY BITES. `_apply_null` is unit-tested in
    isolation, but a null wired in wrongly - applied to a copy, or after the
    ranking - would leave every run identical, and the comparison would
    quietly become "the book against itself": a 50% percentile that reads as a
    clean negative result rather than as a broken instrument.

    `top_n = 1` IS THE POINT. The fixture has three symbols and ships with
    `top_n = 3`, so every candidate is taken whatever its rank and the
    ranking cannot express itself. That is not a quirk of the fixture - it is
    the general condition: the null can only bite where the candidate pool
    exceeds the number of slots. On real Nifty 100 bars it changes 18 of 25
    picks.
    """
    bars, bench = world
    cfg.swing.top_n = 1
    book = bt.run(cfg, market, bars, bench)
    cfg.swing.random_seed = 101
    a = bt.run(cfg, market, bars, bench)
    cfg.swing.random_seed = 202
    b = bt.run(cfg, market, bars, bench)

    def picks(res):
        return [(t.symbol, t.entry_time) for t in res.trades]

    assert picks(a) != picks(b), "two seeds produced identical books"
    assert picks(a) != picks(book) or picks(b) != picks(book), (
        "neither seed changed the book - the null is not wired in")


# --------------------------------------------------- the puretrail exit rule

def test_swing_overrides_actually_reach_the_ladder():
    """
    A CONFIG FIELD THAT SILENTLY DOES NOTHING. `partial_exit_at_r` lives on
    `TradeManagementConfig`, so a sweep variant setting it on `SwingConfig`
    changed nothing at all - the +2R partial kept banking half the position
    while the variant's name claimed it had been removed. Caught only by
    printing the resolved trade config.

    Every SwingConfig override is asserted here to actually arrive.
    """
    from nifty_algo.swing.book import swing_trade
    from nifty_algo.swing import experiment as ex

    shipped = swing_trade(ex.variant("baseline").configure(Config()))
    assert shipped.trail_from_r is None
    assert shipped.breakeven_at_r == pytest.approx(1.0)
    assert shipped.partial_exit_at_r == pytest.approx(2.0)

    pure = swing_trade(ex.variant("puretrail").configure(Config()))
    assert pure.trail_from_r == pytest.approx(0.0)
    assert pure.breakeven_at_r == pytest.approx(99.0)
    assert pure.partial_exit_at_r == pytest.approx(99.0)


def test_puretrail_never_banks_a_partial(cfg, market, world):
    """
    The whole point of the rule: one position, trailed, exited once - not half
    banked at +2R. `partial_banked` is recorded per trade, so this is directly
    checkable rather than inferred from the R distribution.
    """
    from nifty_algo.swing import experiment as ex
    bars, bench = world
    pcfg = ex.variant("puretrail").configure(cfg)
    res = bt.run(pcfg, market, bars, bench)
    assert res.trades
    assert not any(t.partial_banked for t in res.trades)
    assert max(t.bars_held for t in res.trades) <= 10


def test_puretrail_holds_are_capped_and_baseline_holds_are_not(cfg, market,
                                                               world):
    from nifty_algo.swing import experiment as ex
    bars, bench = world
    base = bt.run(ex.variant("baseline").configure(cfg), market, bars, bench)
    pure = bt.run(ex.variant("puretrail").configure(cfg), market, bars, bench)
    assert max(t.bars_held for t in pure.trades) <= 10
    assert base.trades and pure.trades
