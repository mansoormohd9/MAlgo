"""
The intraday-equity backtester: the fill model and the pessimism rules.

Every test here guards a failure that would produce a FLATTERING result
rather than an error. None of them would raise on their own - a leaking fill
model just reports a better number.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.intraday_equity import backtest as bt
from nifty_algo.intraday_equity import scanner, sizing


@pytest.fixture
def cfg():
    c = Config()
    c.capital.intraday_equity_capital_inr = 100_000.0
    c.intraday_equity.mis_leverage = 5.0      # so sizing is risk-bound
    return c


def _calm(day, n=75, price=1000.0, seed=5, vol=0.0030, volume=200_000.0):
    """
    A calm session that establishes ATR and volume history without firing
    anything - the intraday analogue of `conftest.flat_bars`.
    """
    rng = np.random.default_rng(seed)
    close = price + rng.normal(0.0, price * vol, n).cumsum()
    spread = np.abs(rng.normal(0, price * 0.0007, n)) + price * 0.0003
    o = np.r_[price, close[:-1]]
    return pd.DataFrame(
        {"open": o, "high": np.maximum(o, close) + spread,
         "low": np.minimum(o, close) - spread, "close": close,
         "volume": np.full(n, volume)},
        index=pd.date_range(f"{day} 09:15", periods=n, freq="5min"))


def _breakout_session(day, prev_high, price, at_bar=34, follow=+1.0,
                      n=75, seed=7, volume=200_000.0):
    """
    Engineer ONE level-break setup at `at_bar`, then follow through or fail.

    `strategy._all_levels` injects the PRIOR DAY HIGH as a 2-touch level, so
    a session that coils just under it and then breaks it decisively - big
    body, closing at the high, on a volume surge - is the cleanest setup to
    build deliberately. `at_bar` is placed after the 30-bar warm-up and
    inside the entry window.

    `follow` is +1 to run on after the break and -1 to reverse through the
    stop, which is how a test chooses a winner or a loser.
    """
    rng = np.random.default_rng(seed)
    coil = prev_high * 0.997
    close = np.full(n, coil, dtype=float)
    close[:at_bar] = coil + rng.normal(0, price * 0.0004, at_bar)
    step = price * 0.004
    close[at_bar] = prev_high + step                      # the break
    for i in range(at_bar + 1, n):
        close[i] = close[i - 1] + follow * step * 0.45

    o = np.r_[price, close[:-1]]
    hi = np.maximum(o, close) + price * 0.0004
    lo = np.minimum(o, close) - price * 0.0004
    # a decisive candle: opens at the coil, closes at its own high
    o[at_bar] = coil
    hi[at_bar] = close[at_bar]
    lo[at_bar] = coil - price * 0.0002

    vol = np.full(n, volume)
    vol[at_bar] = volume * 4.0                            # the surge
    return pd.DataFrame(
        {"open": o, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.date_range(f"{day} 09:15", periods=n, freq="5min"))


def _sessions(n_days, price=1000.0, seed=1):
    """`n_days` calm sessions, returned with the running price."""
    frames, day, p = [], date(2026, 1, 5), price
    made = 0
    while made < n_days:
        if day.weekday() < 5:
            s = _calm(day, price=p, seed=seed * 1000 + made)
            frames.append(s)
            p = float(s["close"].iloc[-1])
            made += 1
        day += timedelta(days=1)
    return frames, p, day


def _history_with_breakouts(seed, warm_days=30, setups=6, price=1000.0,
                            follow=+1.0):
    """Calm warm-up, then alternating engineered breakout sessions."""
    frames, p, day = _sessions(warm_days, price=price, seed=seed)
    made = 0
    while made < setups:
        if day.weekday() < 5:
            prev_high = float(frames[-1]["high"].max())
            s = _breakout_session(day, prev_high, p, seed=seed * 77 + made,
                                  follow=follow)
            frames.append(s)
            p = float(s["close"].iloc[-1])
            made += 1
        day += timedelta(days=1)
    return pd.concat(frames)


@pytest.fixture
def world(cfg):
    """
    Six names that each present engineered level-break setups, so the fill
    and exit assertions have real trades to inspect. Half run on after the
    break, half reverse - so the fixture produces both winners and losers.
    """
    bars = {}
    for i in range(6):
        bars[f"SYM{i:02d}"] = _history_with_breakouts(
            seed=i + 1, price=800.0 + 60 * i,
            follow=+1.0 if i % 2 == 0 else -1.0)
    frames, _, _ = _sessions(36, price=22_000.0, seed=999)
    bench = pd.concat(frames)
    bench["volume"] = 0.0                     # the index has NO volume
    cfg.intraday_equity.rs_prefilter_n = 6
    return bars, bench


# ------------------------------------------------------- the fill model


def test_a_signal_never_fills_on_its_own_bar(cfg, world):
    """
    You cannot buy the close of the bar you are still deciding on.

    Asserted structurally: every trade's entry timestamp must be strictly
    later than the bar its signal was read on.
    """
    bars, bench = world
    res = bt.run(cfg, bars, benchmark=bench)
    assert res.trades, "fixture produced no trades - the test is vacuous"

    interval = pd.Timedelta(minutes=cfg.intraday_equity.bar_interval_minutes)
    for t in res.trades:
        session = bars[t.symbol]
        day_bars = session[session.index.date == t.entry_time.date()]
        pos = day_bars.index.get_loc(t.entry_time)
        assert pos >= 1, "a trade filled on the session's first bar"
        # the fill must be at THIS bar's open, not the previous bar's close
        assert t.entry_underlying >= float(day_bars["open"].iloc[pos]) - 1e-6


def test_a_fill_is_never_better_than_the_next_bars_open(cfg, world):
    """
    TRAP T6. Filling at the signal bar's close on a gap-through is the
    flattering version, and it is always favourable by construction.
    """
    bars, bench = world
    res = bt.run(cfg, bars, benchmark=bench)
    assert res.trades

    for t in res.trades:
        session = bars[t.symbol]
        day_bars = session[session.index.date == t.entry_time.date()]
        bar_open = float(day_bars.loc[t.entry_time, "open"])
        assert t.entry_underlying >= bar_open - 1e-6, (
            f"{t.symbol} filled at {t.entry_underlying:.2f}, better than the "
            f"bar open {bar_open:.2f} - slippage went the wrong way")


def test_a_gap_beyond_the_chase_limit_is_rejected(cfg):
    """
    TRAP T6, the other half. A setup that gapped away is gone.

    The tempting alternative - "fill at the signal bar's close instead" - is
    fiction, and it is fiction that always pays, because it only ever
    triggers when the market moved your way.
    """
    frames, p, day = _sessions(30, price=1000.0, seed=3)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    prev_high = float(frames[-1]["high"].max())
    s = _breakout_session(day, prev_high, p, seed=11)

    # every bar now opens 20% above the previous close: unchaseable
    s = s.copy()
    s["open"] = (s["close"].shift(1) * 1.20).fillna(s["open"])
    s["high"] = s[["open", "high"]].max(axis=1)

    cfg.intraday_equity.max_chase_pct = 0.004
    bars = {"AAA": pd.concat(frames + [s])}
    bench_frames, _, _ = _sessions(31, price=22_000.0, seed=999)
    bench = pd.concat(bench_frames)

    res = bt.run(cfg, bars, benchmark=bench, start=day, end=day)
    assert not res.trades, "a signal was filled through an unchaseable gap"


def test_a_stop_on_the_entry_bar_is_a_real_stop(cfg):
    """
    TRAP T4, and the one place this book deliberately does the OPPOSITE of
    the swing backtester.

    `swing/backtest.py` skips the entry bar because a daily entry fills at a
    trigger somewhere inside it, so that bar's low may precede the fill. Here
    the fill is the OPEN of the next bar - the first tick - so the rest of
    that bar, including its low, is reachable. Skipping it would delete
    same-bar stop-outs, and the deletion would fall entirely on LOSERS.
    """
    ladder_cfg = bt.intraday_trade_cfg(cfg)
    from nifty_algo.positions import ExitLadder
    ladder = ExitLadder(cfg, trade=ladder_cfg)

    sized = sizing.Sized(quantity=100, entry=1000.0, stop=994.0, target=1012.0,
                         risk_inr=600.0, deployed_inr=100_000.0, stop_pct=0.006)
    sig = scanner.BarSignal(
        symbol="AAA", day=date(2026, 3, 10), bar_index=30,
        bar_time=pd.Timestamp("2026-03-10 11:45"), strategy="level_break",
        direction="long", confidence=0.7, reason="", ref_close=1000.0,
        atr=6.0, stop=994.0, stop_pct=0.006, regime="expansion")

    pos = bt._Open(signal=sig, sized=sized, state=ladder.new_state(100),
                   ladder=ladder, entry_time=pd.Timestamp("2026-03-10 11:50"),
                   entry_index=31, quantity_open=100)

    # The ENTRY bar itself trades straight down through the stop.
    bar = pd.Series({"open": 1000.0, "high": 1000.5, "low": 990.0,
                     "close": 991.0, "volume": 100_000.0})
    res = bt.Result()
    closed = bt._manage(pos, bar, pd.Timestamp("2026-03-10 11:50"), 31,
                        ladder, cfg.intraday_equity.trail_atr_multiple,
                        cfg, bt.DEFAULT_INTRADAY_EQUITY_COSTS, res)

    assert closed, "a stop-out on the entry bar was not taken"
    assert len(res.trades) == 1
    assert res.trades[0].r_multiple < 0, "the entry-bar stop was not a loss"


def test_a_bar_covering_stop_and_target_is_a_loss(cfg):
    """
    Ties resolve pessimistically. One 5-minute bar spanning both levels is
    ambiguous and is counted a LOSS, exactly as the swing book does.
    """
    from nifty_algo.positions import ExitLadder
    ladder = ExitLadder(cfg, trade=bt.intraday_trade_cfg(cfg))
    sized = sizing.Sized(quantity=100, entry=1000.0, stop=994.0, target=1012.0,
                         risk_inr=600.0, deployed_inr=100_000.0, stop_pct=0.006)
    sig = scanner.BarSignal(
        symbol="AAA", day=date(2026, 3, 10), bar_index=30,
        bar_time=pd.Timestamp("2026-03-10 11:45"), strategy="level_break",
        direction="long", confidence=0.7, reason="", ref_close=1000.0,
        atr=6.0, stop=994.0, stop_pct=0.006, regime="expansion")
    pos = bt._Open(signal=sig, sized=sized, state=ladder.new_state(100),
                   ladder=ladder, entry_time=pd.Timestamp("2026-03-10 11:50"),
                   entry_index=31, quantity_open=100)

    # spans BOTH the stop (994) and +2R (1012)
    bar = pd.Series({"open": 1000.0, "high": 1015.0, "low": 990.0,
                     "close": 1005.0, "volume": 100_000.0})
    res = bt.Result()
    bt._manage(pos, bar, pd.Timestamp("2026-03-10 11:50"), 31, ladder,
               cfg.intraday_equity.trail_atr_multiple, cfg,
               bt.DEFAULT_INTRADAY_EQUITY_COSTS, res)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.ambiguous, "the bar was not flagged ambiguous"
    assert t.r_multiple <= -1.0, (
        "an ambiguous bar must be counted a full loss, not a win")


def test_nothing_is_open_after_the_square_off_time(cfg, world):
    """
    Flat by 15:10, always. This is what makes the whole walk-forward
    boundary-bias problem structurally absent from this book.
    """
    bars, bench = world
    res = bt.run(cfg, bars, benchmark=bench)
    assert res.trades

    cutoff = cfg.intraday_equity.force_exit
    for t in res.trades:
        assert t.exit_time.date() == t.entry_time.date(), (
            f"{t.symbol} held overnight - an MIS book cannot")
        # exit bar CLOSE must not be past the square-off
        closing = scanner._bar_close_time(
            t.exit_time, cfg.intraday_equity.bar_interval_minutes)
        assert closing <= cutoff or t.outcome == "force_exit"


# ------------------------------------------------------- determinism


def test_entry_order_is_by_rank_not_by_dict_order(cfg, world):
    """
    TRAP T11. Iterating the universe in file order under a concurrency cap
    gives alphabetically early names systematic priority and makes the result
    depend on how the CSV happens to be sorted.
    """
    bars, bench = world
    first = bt.run(cfg, bars, benchmark=bench)

    reshuffled = {k: bars[k] for k in reversed(list(bars))}
    second = bt.run(cfg, reshuffled, benchmark=bench)

    a = [(t.symbol, t.entry_time, round(t.r_multiple, 9)) for t in first.trades]
    b = [(t.symbol, t.entry_time, round(t.r_multiple, 9)) for t in second.trades]
    assert a == b, "reversing the universe order changed the trade list"


def test_truncating_the_future_changes_nothing(cfg, world):
    """
    The generic look-ahead guard, mirroring the swing backtester's
    `test_truncating_the_future_changes_nothing`.
    """
    bars, bench = world
    sessions = sorted({ts.date() for df in bars.values() for ts in df.index})
    cut = sessions[len(sessions) * 3 // 4]

    full = bt.run(cfg, bars, benchmark=bench, end=cut)
    truncated_bars = {s: df[df.index.date <= cut] for s, df in bars.items()}
    truncated = bt.run(cfg, truncated_bars,
                       benchmark=bench[bench.index.date <= cut], end=cut)

    a = [(t.symbol, t.entry_time, t.exit_time, round(t.r_multiple, 9))
         for t in full.trades]
    b = [(t.symbol, t.entry_time, t.exit_time, round(t.r_multiple, 9))
         for t in truncated.trades]
    assert a == b, "bars after the cut changed a decision before it"


# ------------------------------------------------------- the cache


def test_a_replayed_variant_is_identical_to_a_direct_run(cfg, world):
    """
    TRAP T14. The cache is what makes a sweep affordable; if a replay differs
    from a direct run, every swept result is answering a different question.
    """
    bars, bench = world
    direct = bt.run(cfg, bars, benchmark=bench)

    cache = bt.SignalCache(bt.signal_signature(cfg))
    warmed = bt.run(cfg, bars, benchmark=bench, cache=cache)
    replayed = bt.run(cfg, bars, benchmark=bench, cache=cache)

    assert cache.sessions_cached > 0, "the cache was never populated"
    for other in (warmed, replayed):
        a = [(t.symbol, t.entry_time, round(t.r_multiple, 9)) for t in direct.trades]
        b = [(t.symbol, t.entry_time, round(t.r_multiple, 9)) for t in other.trades]
        assert a == b


def test_a_changed_signal_field_raises_rather_than_reusing(cfg, world):
    """
    A stale scan answers a different question FLUENTLY, so a mismatch must
    raise rather than fall back - the same rule as `swing/backtest.py`.
    """
    bars, bench = world
    cache = bt.SignalCache(bt.signal_signature(cfg))
    bt.run(cfg, bars, benchmark=bench, cache=cache)

    # NOTE 3.0, not 2.0 - 2.0 is now the DEFAULT, so setting it would
    # change nothing and the test would pass by doing nothing at all.
    cfg.intraday_equity.atr_stop_multiple = 3.0     # changes every signal
    with pytest.raises(ValueError, match="signature"):
        bt.run(cfg, bars, benchmark=bench, cache=cache)


def test_book_only_fields_do_not_invalidate_the_cache(cfg):
    """
    The exemption list is what makes a ladder sweep cost ONE scan pass. If a
    book-only field invalidated, every variant would rescan and the sweep
    would be unaffordable rather than wrong.
    """
    before = bt.signal_signature(cfg)
    cfg.intraday_equity.trail_atr_multiple = 2.5
    cfg.intraday_equity.partial_exit_fraction = 0.25
    cfg.intraday_equity.min_confidence = 0.4
    cfg.intraday_equity.top_n = 1
    assert bt.signal_signature(cfg) == before


def test_a_scan_affecting_field_does_invalidate(cfg):
    """
    And the default direction is 'invalidates' - forgetting to exempt a
    book-only field costs a rescan, which is merely slow.
    """
    before = bt.signal_signature(cfg)
    cfg.intraday_equity.entry_cutoff = cfg.intraday_equity.entry_cutoff.replace(hour=13)
    assert bt.signal_signature(cfg) != before


def test_the_universe_is_in_the_signature(cfg):
    """
    Here the cache IS the universe. A symbol added between runs would
    otherwise produce a cache that is correct for every day it holds and
    silently short one name - which reads as "that name never had a setup".
    """
    assert bt.signal_signature(cfg, "a") != bt.signal_signature(cfg, "b")


# ------------------------------------------------------- reporting


def test_the_result_reports_friction_and_the_pot_constraint(cfg, world):
    """
    Both numbers must be computed and surfaced, not left to be inferred from
    disappointing results.
    """
    bars, bench = world
    res = bt.run(cfg, bars, benchmark=bench)
    assert res.friction_r > 0
    assert "stop" in res.pot_note
    assert ("RISK-bound" in res.pot_note) or ("CAPITAL-bound" in res.pot_note)


def test_the_rejection_ledger_is_populated(cfg, world):
    """
    "Nothing fired" and "the data was not there" are different facts, and on
    a new book the ledger is usually the finding.
    """
    bars, bench = world
    res = bt.run(cfg, bars, benchmark=bench)
    assert res.rejections, "no rejections recorded at all"
    assert res.stats.days > 0


def test_a_halted_bar_does_not_misalign_the_universe(cfg):
    """
    Positional bar indices only line up while every session has exactly the
    same bars, and one halted bar breaks that.

    A session with 74 bars instead of 75 still clears the 60-bar integrity
    gate, so it reaches the book - and if the cross-symbol loop keys on
    POSITION, every later index for that name is shifted by one. Ranking then
    compares an 11:45 signal against an 11:50 signal and management marks a
    position against a bar that never coincided with it. Nothing raises; the
    trades are simply against prices that did not happen together.

    So the loop keys on the TIMESTAMP, and this pins it.
    """
    bars = {}
    for i in range(4):
        h = _history_with_breakouts(seed=i + 1, price=800.0 + 60 * i,
                                    follow=+1.0 if i % 2 == 0 else -1.0)
        if i == 1:
            # drop one mid-session bar from every session of ONE symbol
            keep = []
            for day, g in h.groupby(h.index.date):
                keep.append(g.drop(g.index[40]))
            h = pd.concat(keep)
        bars[f"SYM{i:02d}"] = h

    frames, _, _ = _sessions(36, price=22_000.0, seed=999)
    bench = pd.concat(frames)
    cfg.intraday_equity.rs_prefilter_n = 4

    res = bt.run(cfg, bars, benchmark=bench)

    # every managed bar must belong to the symbol's own session, at a
    # timestamp that symbol actually printed
    for t in res.trades:
        session = bars[t.symbol]
        assert t.entry_time in session.index, (
            f"{t.symbol} entered at {t.entry_time}, which it never printed")
        assert t.exit_time in session.index, (
            f"{t.symbol} exited at {t.exit_time}, which it never printed")
        assert t.exit_time >= t.entry_time
