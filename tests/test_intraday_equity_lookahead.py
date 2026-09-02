"""
Look-ahead tests for the precomputed indicator pack.

The companion to [test_lookahead.py](test_lookahead.py), and it matters for
the same reason: every other failure produces a wrong answer you can see,
look-ahead bias produces a FLATTERING one you cannot, and it invalidates
every backtest number silently.

`test_lookahead.py` asserts that `BarReplayer` hides the future. These assert
that the PRECOMPUTED PATH hides exactly the same future - because the whole
point of the pack is to skip the recomputation those tests police, and a pack
that leaks is a backtest that reports a good number instead of an error.

Two hazards, two groups of tests:

  CAUSAL SERIES. `atr`, `vwap`, `volume_surge` and friends are served as
  `full_series.iloc[:n]`. That is only legitimate if slicing equals
  recomputing, so it is asserted EXHAUSTIVELY - every served function, every
  prefix, exact float equality - rather than argued.

  CENTRED PIVOTS. `find_pivots` is not causal, so it is never served through
  the cache; it goes through `PivotLadder.visible_at`, which applies a
  `+lookback` confirmation delay. The delay is asserted from BOTH sides,
  because an off-by-one that leaks a single bar early shows up as a slightly
  better expectancy and nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_algo import indicator_cache as cache
from nifty_algo import signals as sig
from nifty_algo.config import Config
from nifty_algo.data.csv_feed import BarReplayer
from nifty_algo.intraday_equity import precompute as pre


def _session(n=75, seed=5, start="2026-03-10 09:15", price=1000.0):
    """A realistic single 5-minute session with real volume."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, price * 0.0012, n).cumsum()
    close = price + steps
    spread = np.abs(rng.normal(0, price * 0.0006, n)) + price * 0.0002
    open_ = np.r_[price, close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.integers(4_000, 40_000, n).astype(float),
        },
        index=pd.date_range(start, periods=n, freq="5min"),
    )


@pytest.fixture
def cfg():
    return Config()


# --------------------------------------------------------------- causal


def test_pack_serves_exactly_what_recompute_returns(cfg):
    """
    Every served function, every prefix, exact equality.

    Exhaustive rather than sampled on purpose: this is the assertion that
    licenses the whole cache, and `signals.py` is consumed by both other
    books, so a wrong slice here is wrong everywhere.
    """
    session = _session()
    pack = pre.build_pack(session, "TEST", cfg)

    checks = {
        "atr": lambda df: sig.atr(df, cfg.signal.atr_period),
        "body_to_range": sig.body_to_range,
        "close_position_in_range": sig.close_position_in_range,
        "typical_price": sig.typical_price,
        "vwap": sig.vwap,
        "volume_surge": lambda df: sig.volume_surge(
            df, multiple=cfg.signal.volume_surge_multiple),
        "trend_state": lambda df: sig.trend_state(
            df, cfg.strategy.ema_fast, cfg.strategy.ema_slow),
        "narrowest_range_n": lambda df: sig.narrowest_range_n(
            df, cfg.strategy.squeeze_lookback),
        "range_percentile": lambda df: sig.range_percentile(df, lookback=20),
    }

    for name, fn in checks.items():
        for n in range(2, len(session) + 1):
            plain = fn(session.iloc[:n])                       # no pack
            windowed = pre.attach(session.iloc[:n].copy(), pack)
            served = fn(windowed)                              # through the pack
            pd.testing.assert_series_equal(
                served, plain, check_exact=True, check_names=False,
                obj=f"{name} at prefix {n}")


def test_the_pack_is_actually_being_used(cfg):
    """
    A pack that never hits is a 33-hour backtest that looks like a working
    one. The equality test above passes trivially if every lookup misses, so
    this asserts the cache is genuinely serving.
    """
    session = _session()
    pack = pre.build_pack(session, "TEST", cfg)
    windowed = pre.attach(session.iloc[:40].copy(), pack)

    assert cache.pack_for(windowed, "atr") is not None
    assert cache.served(pack, "atr", (cfg.signal.atr_period,), 40) is not None
    # and the key the STRATEGIES actually use, defaults included
    assert cache.served(pack, "volume_surge",
                        (pre.VOLUME_SURGE_LOOKBACK,
                         cfg.signal.volume_surge_multiple), 40) is not None


def test_find_pivots_is_never_served_from_the_cache():
    """
    The allowlist is the failure-safe direction, and this is the one entry
    whose absence is load-bearing: `find_pivots` is centred, so serving a
    prefix of a full-session computation would leak the future.
    """
    assert "find_pivots" not in cache.CAUSAL_SERVED
    assert "build_levels" not in cache.CAUSAL_SERVED
    assert "fit_trendline" not in cache.CAUSAL_SERVED


def test_a_frame_from_another_symbol_is_not_served(cfg):
    """
    `pack_for` verifies CONTENT, not identity.

    Keying on `id(frame)` would pass this by accident and fail in production,
    because a bar loop drops ~75 frames per session and CPython reuses the
    addresses within milliseconds - serving one symbol's ATR for another with
    no exception anywhere.
    """
    a = _session(seed=1)
    b = _session(seed=2, start="2026-03-11 09:15")
    pack_a = pre.build_pack(a, "AAA", cfg)

    wrong = pre.attach(b.iloc[:40].copy(), pack_a)
    assert cache.pack_for(wrong, "atr") is None, (
        "a frame whose timestamps do not match the pack was served anyway")

    # and a non-prefix slice of the RIGHT symbol is also refused
    tail = pre.attach(a.iloc[10:40].copy(), pack_a)
    assert cache.pack_for(tail, "atr") is None


def test_the_kill_switch_disables_every_lookup(cfg, monkeypatch):
    """
    The escape hatch: re-run anything suspicious uncached, no code change.

    The flag is read ONCE at import rather than per call, because an
    `os.environ` lookup on a path taken ~100k times per symbol-session costs
    more than some of the calls being cached. `refresh_disabled()` is how a
    process picks up a change, and is what this asserts.
    """
    session = _session()
    pack = pre.build_pack(session, "TEST", cfg)
    windowed = pre.attach(session.iloc[:40].copy(), pack)

    assert cache.pack_for(windowed, "atr") is not None
    monkeypatch.setenv(cache.DISABLE_ENV, "1")
    cache.refresh_disabled()
    try:
        assert cache.pack_for(windowed, "atr") is None
        assert cache.scalar_for(windowed, "has_traded_volume") is None
        assert cache.pack_for_pivots(windowed, "find_pivots") is None
    finally:
        monkeypatch.delenv(cache.DISABLE_ENV, raising=False)
        cache.refresh_disabled()


def test_scalar_gates_match_a_direct_recompute_at_every_prefix(cfg):
    """
    `has_traded_volume` and `underlying_liquidity_ok` return bools, so a
    series cache could not see them - and profiled, they were the single
    largest remaining cost (339ms of a 731ms symbol-session) because
    `_preflight` runs the pair once per strategy per bar.

    They are cached as per-bar arrays, which is only legitimate if the value
    at bar i depends solely on bars <= i. Asserted, not argued.
    """
    session = _session()
    pack = pre.build_pack(session, "TEST", cfg)

    for n in range(2, len(session) + 1):
        plain_prefix = session.iloc[:n]
        windowed = pre.attach(session.iloc[:n].copy(), pack)

        assert sig.has_traded_volume(windowed) == sig.has_traded_volume(plain_prefix), (
            f"has_traded_volume diverged at prefix {n}")
        assert sig.underlying_liquidity_ok(windowed) ==             sig.underlying_liquidity_ok(plain_prefix), (
                f"underlying_liquidity_ok diverged at prefix {n}")


def test_find_pivots_through_the_pack_equals_the_replayer(cfg):
    """
    `find_pivots` IS served - but only through the ladder, never by slicing a
    full-session computation. This asserts the hooked function returns
    exactly what an uncached call on the same window returns, which is what
    makes `build_levels` and `fit_trendline` cheap for free.
    """
    session = _session(n=75, seed=17)
    pack = pre.build_pack(session, "TEST", cfg)
    lookback = cfg.signal.pivot_lookback

    for n in range(2, len(session) + 1):
        plain = session.iloc[:n]
        windowed = pre.attach(session.iloc[:n].copy(), pack)

        want_h, want_l = sig.find_pivots(plain, lookback)
        got_h, got_l = sig.find_pivots(windowed, lookback)
        assert np.array_equal(got_h.to_numpy(bool), want_h.to_numpy(bool)), (
            f"swing highs diverged at prefix {n}")
        assert np.array_equal(got_l.to_numpy(bool), want_l.to_numpy(bool)), (
            f"swing lows diverged at prefix {n}")


# --------------------------------------------------------------- pivots


def test_pack_pivots_equal_the_replayer_at_every_bar(cfg):
    """
    THE load-bearing test.

    For every decision bar, the shifted pivot masks must equal what
    `find_pivots` returns when handed only the bars that had printed.
    """
    session = _session(n=75, seed=9)
    pack = pre.build_pack(session, "TEST", cfg)
    lookback = cfg.signal.pivot_lookback
    ladder = pack.pivots[lookback]
    replayer = BarReplayer(session, warmup=2, max_window=10_000)

    for i in range(2, len(session)):
        window = replayer.window_at(i)
        want_high, want_low = sig.find_pivots(window, lookback)
        got_high, got_low = ladder.visible_at(i)

        assert np.array_equal(got_high, want_high.to_numpy(dtype=bool)), (
            f"swing-high mask diverged at bar {i}")
        assert np.array_equal(got_low, want_low.to_numpy(dtype=bool)), (
            f"swing-low mask diverged at bar {i}")


def test_a_pivot_is_invisible_for_exactly_lookback_bars(cfg):
    """
    The boundary, asserted from both sides.

    Leaking one bar early is a real look-ahead bug worth a few hundredths of
    an R; hiding one bar late merely costs a signal. Both are wrong, and only
    checking one side would let the first through.
    """
    lookback = cfg.signal.pivot_lookback
    n, peak = 41, 20
    base = 1000.0
    rows = []
    for i in range(n):
        p = base + (50.0 if i == peak else 0.0)
        rows.append({"open": p, "high": p + 2, "low": p - 2,
                     "close": p, "volume": 10_000.0})
    session = pd.DataFrame(
        rows, index=pd.date_range("2026-03-10 09:15", periods=n, freq="5min"))

    pack = pre.build_pack(session, "TEST", cfg)
    ladder = pack.pivots[lookback]

    for i in range(peak, peak + lookback):
        got_high, _ = ladder.visible_at(i)
        assert not bool(got_high[peak]), (
            f"pivot at bar {peak} was visible at bar {i} - "
            f"{peak + lookback - i} confirming bars had not printed. "
            f"This is look-ahead bias.")

    got_high, _ = ladder.visible_at(peak + lookback)
    assert bool(got_high[peak]), (
        "pivot should be confirmable once lookback bars have printed - "
        "hiding it costs a real signal")


def test_build_levels_through_the_pack_is_identical_to_uncached(cfg):
    """
    Catches the pivot shift AND the hardcoded-period-14 landmine at once.

    `signals.build_levels` computes its clustering tolerance from `atr(df)`
    with the DEFAULT period, not `cfg.signal.atr_period`. `levels_at` has to
    reproduce that, bug and all - a "corrected" tolerance would silently
    produce different levels in the backtest than the live book sees.
    """
    session = _session(n=75, seed=13)
    pack = pre.build_pack(session, "TEST", cfg)
    s = cfg.signal
    replayer = BarReplayer(session, warmup=2, max_window=10_000)

    for i in range(20, len(session)):
        want = sig.build_levels(
            replayer.window_at(i), lookback=s.pivot_lookback,
            cluster_atr_frac=s.level_cluster_atr_frac,
            min_touches=s.min_level_touches)
        got = pre.levels_at(pack, i, cfg)

        key = lambda lv: (lv.kind, round(lv.price, 9), lv.touches)
        assert sorted(map(key, got)) == sorted(map(key, want)), (
            f"level set diverged at bar {i}")


def test_a_future_bar_cannot_change_a_past_decision(cfg):
    """
    The generic truncation test, which catches leaks the pivot-specific
    tests do not. Mirrors the swing backtester's
    `test_truncating_the_future_changes_nothing`.
    """
    full = _session(n=75, seed=21)
    cut = 50
    truncated = full.iloc[:cut]

    pack_full = pre.build_pack(full, "TEST", cfg)
    pack_cut = pre.build_pack(truncated, "TEST", cfg)

    key = lambda lv: (lv.kind, round(lv.price, 9), lv.touches)
    for i in range(20, cut):
        assert (sorted(map(key, pre.levels_at(pack_full, i, cfg)))
                == sorted(map(key, pre.levels_at(pack_cut, i, cfg)))), (
            f"a bar after {cut} changed the level set at bar {i}")
        assert pack_full.atr_at(i) == pytest.approx(pack_cut.atr_at(i)), (
            f"a bar after {cut} changed ATR at bar {i}")


def test_the_pack_survives_the_bar_loops_slicing(cfg):
    """
    The mechanism the whole design rests on.

    `BarReplayer.window_at` does `df.iloc[start:i+1]`, and the pack rides on
    `.attrs`. If pandas ever stopped propagating those through a slice, every
    lookup would miss - results would stay CORRECT but the backtest would
    quietly go back to taking 33 hours, which is the kind of regression that
    gets diagnosed as "it was always slow".
    """
    session = _session()
    pack = pre.build_pack(session, "TEST", cfg)
    pre.attach(session, pack)

    replayer = BarReplayer(session, warmup=30, max_window=10_000)
    window = replayer.window_at(40)
    assert cache.pack_of(window) is pack, (
        "the pack did not survive BarReplayer's slice")
    assert cache.pack_for(window, "atr") is pack
