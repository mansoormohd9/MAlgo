"""
Which universe the sleeve ranks inside, and the two ways that can lie.

The lie this module exists to prevent is not a crash. `nifty500` applied to a
backtest uses TODAY'S membership, so it holds only companies that grew INTO the
index - a spectacular curve with no error anywhere, which is exactly what F4
measured at +32% CAGR against a +20% control. These tests cannot detect that
bias (no test can; it is a property of the data), so they guard the two things
that CAN be asserted: that the default changes nothing, and that the
point-in-time control really is point-in-time.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import DEFAULT
from nifty_algo.factor import backtest as fb
from nifty_algo.factor import restriction as restr
from nifty_algo.factor import sleeve as sl
from nifty_algo.factor.universe import FactorUniverse, month_ends


def _series(start_price, n, drift, seed, start=date(2020, 1, 1), volume=1e6):
    rng = np.random.default_rng(seed)
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    close = start_price * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, volume)},
        index=pd.to_datetime(days))


@pytest.fixture
def world():
    return {f"WIN{i:02d}": _series(100.0 + i, 700, 0.0012 - i * 0.0003, i + 1)
            for i in range(12)}


@pytest.fixture
def universe(world):
    return FactorUniverse(world)


def _last_mark(world) -> date:
    sessions = sorted({d.date() for f in world.values() for d in f.index})
    return month_ends(sessions, 1)[-1]


def _run(world, **kw):
    return fb.run(world, 500_000.0, top_n=4, min_turnover=0.0,
                  min_history=300, min_price=0.0, **kw)


# ------------------------------------------------------- the default no-op

def test_no_restriction_is_byte_identical_to_the_old_book(world):
    """
    THE REGRESSION THAT PROTECTS EVERY RECORDED NUMBER. `restrict_fn=None` and
    `universe="all"` must leave `run()` exactly as it was, or F1, F2 and F3 all
    describe a book that no longer exists.
    """
    a = _run(world)
    b = _run(world, restrict_fn=None)
    c = _run(world, restrict_fn=restr.resolver(
        DEFAULT, "all", world, FactorUniverse(world))[0])
    assert [v for _, v in a.equity] == [v for _, v in b.equity]
    assert [v for _, v in a.equity] == [v for _, v in c.equity]
    assert (a.trades, a.costs_paid) == (c.trades, c.costs_paid)


def test_the_all_resolver_returns_no_opinion_rather_than_everything(world,
                                                                    universe):
    """
    `all` must return None, not the full symbol set.

    A resolver that returned every name would compose with `listed_only` by
    intersection and produce the same answer today - and would silently become
    a filter the moment the two sets disagreed about a name the bars know and
    the universe does not.
    """
    fn, note = restr.resolver(DEFAULT, "all", world, universe)
    assert fn(_last_mark(world)) is None
    assert "as tested" in note


def test_an_unknown_universe_raises_rather_than_defaulting(world, universe):
    with pytest.raises(restr.UnknownUniverse):
        restr.resolver(DEFAULT, "nifty1", world, universe)


# --------------------------------------------------- the point-in-time rule

def test_the_size_rank_cannot_see_the_rebalance_day_or_later(world, universe):
    """
    THE SAFETY PROPERTY, asserted the way `test_factor_universe.py` asserts it
    for eligibility: deleting every bar from the rebalance date onward must not
    change the set for that date.

    A size rank that peeks is the same trap as a ranking that peeks, and it
    fails the same way - a better curve and no exception.
    """
    day = _last_mark(world)
    shares = {s: 1000.0 + i for i, s in enumerate(sorted(world))}

    full = restr.SizeRank(universe, shares, n=5).at(day)

    truncated = {s: f[f.index < pd.Timestamp(day)] for s, f in world.items()}
    trimmed = restr.SizeRank(FactorUniverse(truncated), shares, n=5).at(day)

    assert full == trimmed
    assert len(full) == 5


def test_the_size_rank_moves_with_price_not_with_hindsight(universe, world):
    """
    Two dates, two answers. A size restriction that returned the same set on
    every rebalance would be a static list wearing a point-in-time label - the
    precise thing `nifty500` is and `size500` is not.
    """
    sessions = sorted({d.date() for f in world.values() for d in f.index})
    marks = month_ends(sessions, 1)
    shares = {s: 1.0 for s in sorted(world)}     # size is then pure price
    rank = restr.SizeRank(universe, shares, n=3)
    early, late = rank.at(marks[2]), rank.at(marks[-1])
    assert early and late
    assert early != late, "a drifting cross-section must reorder"


def test_a_name_without_a_market_cap_is_excluded_not_ranked_at_zero(universe,
                                                                    world):
    """
    Ranking a missing cap at zero puts it last in every list and guarantees it
    is never held - indistinguishable from "it was small". A missing fact is
    not a measurement, so it is excluded and counted.
    """
    known = sorted(world)[:4]
    rank = restr.SizeRank(universe, {s: 1000.0 for s in known}, n=100)
    got = rank.at(_last_mark(world))
    assert got == set(known)
    assert set(rank.skipped) == set(world) - set(known)


def test_implied_shares_divide_todays_cap_by_todays_close(world):
    caps = {"WIN00": 1e9}
    shares = restr._implied_shares(world, caps)
    last = float(world["WIN00"]["close"].iloc[-1])
    assert shares["WIN00"] == pytest.approx(1e9 / last)
    assert "WIN01" not in shares          # no cap given, so no share count


def test_a_zero_or_missing_price_cannot_produce_an_infinite_company(world):
    flat = {"ZERO": world["WIN00"].assign(close=0.0)}
    assert restr._implied_shares(flat, {"ZERO": 1e9}) == {}


# ------------------------------------------------------------- composition

def test_restrictions_narrow_and_never_replace(world, universe):
    """
    `listed_only` builds a restriction of its own. A universe key that replaced
    it would answer a different question under the same label, so they compose
    by intersection.
    """
    a = restr.static({"WIN00", "WIN01", "WIN02"})
    b = restr.static({"WIN01", "WIN02", "WIN03"})
    day = _last_mark(world)
    assert restr.compose(a, b)(day) == {"WIN01", "WIN02"}
    assert restr.compose(a, None)(day) == {"WIN00", "WIN01", "WIN02"}
    assert restr.compose(None, None) is None


def test_listed_only_still_bounds_survivorship_under_a_universe_key(world):
    """The survivorship bound must survive being combined with a universe."""
    keep = restr.static({"WIN00", "WIN01", "WIN02", "WIN03"})
    res = _run(world, listed_only=True, restrict_fn=keep)
    for _day, names in res.holdings_log:
        assert set(names) <= {"WIN00", "WIN01", "WIN02", "WIN03"}


def test_a_restriction_actually_restricts_the_book(world):
    allowed = {"WIN05", "WIN06", "WIN07"}
    res = _run(world, restrict_fn=restr.static(allowed))
    held = {s for _d, names in res.holdings_log for s in names}
    assert held and held <= allowed


# ------------------------------------------------------- live == backtested

def test_the_restricted_live_scan_is_the_restricted_backtest(world,
                                                             monkeypatch):
    """
    The invariant that keeps the console honest, now under a universe key: the
    names the live scan recommends must be the names the backtest's RANKING
    asked for on the same date, with the same restriction.
    """
    cfg = DEFAULT
    cfg.factor.top_n = 4
    cfg.factor.min_turnover_inr = 0.0
    cfg.factor.min_price = 0.0
    cfg.factor.min_history_sessions = 300
    cfg.capital.factor_capital_inr = 500_000.0
    allowed = {"WIN02", "WIN03", "WIN04", "WIN05", "WIN06"}
    monkeypatch.setattr(
        restr, "resolver",
        lambda c, key, bars, uni, root=".": (restr.static(allowed), "test"))
    try:
        day = _last_mark(world)
        res = _run(world, restrict_fn=restr.static(allowed), end=day)
        scan = sl.scan(cfg, world, today=day)
        assert sorted(p.symbol for p in scan.picks) == sorted(
            res.wanted_log[-1][1])
        assert scan.picks
    finally:
        cfg.factor.top_n = 20
        cfg.factor.min_turnover_inr = 1.0e7
        cfg.factor.min_price = 20.0
        cfg.capital.factor_capital_inr = 0.0


def test_the_scan_says_which_universe_produced_the_book(world, monkeypatch):
    cfg = DEFAULT
    cfg.factor.top_n = 4
    cfg.factor.min_turnover_inr = 0.0
    cfg.factor.min_price = 0.0
    try:
        scan = sl.scan(cfg, world, today=_last_mark(world))
        assert scan.universe_key == "all"
        assert scan.universe_note
        assert scan.universe_note in sl.report(scan, sl.decide(scan))
    finally:
        cfg.factor.top_n = 20
        cfg.factor.min_turnover_inr = 1.0e7
        cfg.factor.min_price = 20.0


def test_the_market_cap_loader_never_reaches_the_network(tmp_path):
    """
    Reads the cache FILE. `load_fundamentals` fetches whatever is missing, so
    asking it about a universe fires a thousand requests as a side effect of a
    measurement - the trap `run_s1_swing_null` avoids by reading its parquet
    directly.
    """
    from nifty_algo.config import Config
    cfg = Config()
    cfg.swing.cache_dir = str(tmp_path)
    assert restr.load_market_caps(cfg) == {}          # no file, no crash

    (tmp_path / "fundamentals.json").write_text(
        '{"factor_india:ACME": {"market_cap": 1000.0},'
        ' "india:ACME": {"market_cap": 9.0},'
        ' "factor_india:NOCAP": {"market_cap": null}}', encoding="utf-8")
    caps = restr.load_market_caps(cfg)
    assert caps == {"ACME": 1000.0}      # the swing namespace cannot leak in


def test_the_nifty500_resolver_returns_the_committed_list(world, universe):
    """
    Against the real file, not a fixture: the resolver must hand back exactly
    what `data/nifty500.csv` holds, and must say in its own note that using it
    in a backtest is look-ahead. A restriction that quietly disagreed with the
    committed list would put names in the book that the mandate excludes.
    """
    from nifty_algo.factor import membership as mb
    members = mb.load()
    if not members.sets.get("nifty500"):
        pytest.skip("data/nifty500.csv is absent - run membership --refresh")

    fn, note = restr.resolver(DEFAULT, "nifty500", world, universe)
    got = fn(_last_mark(world))
    assert got == set(members.sets["nifty500"])
    assert "LOOK-AHEAD" in note


def test_a_missing_membership_file_refuses_rather_than_passing_everything(
        world, universe, tmp_path):
    """
    No list must mean no answer, never "everyone qualifies". A resolver that
    fell back to the full universe would silently run the unrestricted book
    under a restricted label.
    """
    with pytest.raises(restr.UnknownUniverse):
        restr.resolver(DEFAULT, "nifty500", world, universe, root=tmp_path)
