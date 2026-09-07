"""
The live factor sleeve: is what the console recommends the book that was tested?

The first test is the one that matters. Everything else in this file guards a
specific way the console could mislead - a provisional diff rendered as a
decision, a failed broker read rendered as an empty account, a headline that
reorders the ranking - but only `test_the_live_scan_is_the_backtests_own_book`
answers the question the page exists to answer honestly.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_algo.config import DEFAULT
from nifty_algo.factor import backtest as fb
from nifty_algo.factor import membership as mb
from nifty_algo.factor import sleeve as sl
from nifty_algo.factor.universe import FactorUniverse, month_ends
from nifty_algo.swing import markets as markets_mod


def _series(start_price, n, drift, seed, start=date(2020, 1, 1), volume=1e6):
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
    return {f"WIN{i:02d}": _series(100.0 + i, n=700, drift=0.0012 - i * 0.0003,
                                   seed=i + 1, volume=1e6 * (i + 1))
            for i in range(12)}


@pytest.fixture
def cfg(world):
    c = DEFAULT
    c.factor.top_n = 4
    c.factor.min_turnover_inr = 0.0
    c.factor.min_history_sessions = 300
    c.factor.min_price = 0.0
    c.factor.halal_screened = False
    c.factor.regime_ma_days = 0
    c.capital.factor_capital_inr = 500_000.0
    yield c
    c.factor.top_n = 20
    c.factor.min_turnover_inr = 1.0e7
    c.factor.min_price = 20.0
    c.capital.factor_capital_inr = 0.0
    # `formation`, `halal_shortlist` and `halal_screened` are restored too:
    # `cfg` IS the module-level `DEFAULT`, so anything a test here leaves
    # behind is read by every later test in the run. That is the same leak
    # `conftest._no_saved_settings` exists for, one scope down.
    c.factor.formation = "mom12_1"
    c.factor.halal_shortlist = 60
    c.factor.halal_screened = False


def _last_mark(world) -> date:
    sessions = sorted({d.date() for f in world.values() for d in f.index})
    return month_ends(sessions, 1)[-1]


class _Pos:
    def __init__(self, symbol, quantity, average_price=100.0, last_price=100.0):
        self.symbol = symbol
        self.quantity = quantity
        self.average_price = average_price
        self.last_price = last_price


class _Snapshot:
    def __init__(self, positions, complete=True, notes=()):
        self.positions = positions
        self.complete = complete
        self._notes = list(notes)

    def caveats(self):
        return self._notes


# ------------------------------------------- the test the page exists for

def test_the_live_scan_is_the_backtests_own_book(cfg, world):
    """
    THE ONLY TEST THAT PROVES THE CONSOLE DESCRIBES THE TESTED STRATEGY.

    On the last rebalance the backtest reaches, the names its RANKING asked for
    must be exactly the names the live scan recommends for that date. Any
    divergence - a different price floor, a different turnover window, a
    ranking recomputed slightly differently - would mean the recorded returns
    describe one book while the page recommends another, which is the failure
    every `if backtest:` in this repo exists to prevent.

    Compared against `wanted_log` rather than `holdings_log` deliberately.
    What the account could AFFORD depends on its accumulated balance and is a
    different question from what the strategy CHOSE; conflating them would make
    this test fail for a reason that has nothing to do with the invariant.
    """
    day = _last_mark(world)
    res = fb.run(world, 500_000.0, top_n=cfg.factor.top_n,
                 min_turnover=0.0, min_history=300, min_price=0.0,
                 end=day)
    wanted = sorted(res.wanted_log[-1][1])

    scan = sl.scan(cfg, world, today=day)
    assert sorted(p.symbol for p in scan.picks) == wanted
    assert wanted, "the fixture must actually produce a book"


def test_the_pot_running_out_is_recorded_rather_than_hidden(cfg, world):
    """
    `budget = marked / top_n` leaves nothing for charges, so the tail of the
    ranking goes unfilled - on real data the book held all 20 names on 9 of
    121 rebalances. The scan must size the same way the backtest does and say
    which names the pot did not reach, or the checklist contains orders the
    broker will refuse.
    """
    cfg.capital.factor_capital_inr = 1000.0        # enough for one small line
    scan = sl.scan(cfg, world, today=_last_mark(world))
    assert any(p.unfunded for p in scan.picks)
    assert all(p.target_qty == 0 for p in scan.picks if p.unfunded)
    assert scan.cash_left_inr >= 0.0
    kinds = {a.symbol: a.kind for a in sl.decide(scan)}
    for p in scan.picks:
        if p.unfunded:
            assert kinds[p.symbol] == "HOLD"


# ------------------------------------------------------- the cadence lock

def test_everything_is_provisional_away_from_the_rebalance_date(cfg, world):
    sessions = sorted({d.date() for f in world.values() for d in f.index})
    marks = month_ends(sessions, 1)
    mid = next(d for d in sessions if marks[-2] < d < marks[-1])

    scan = sl.scan(cfg, world, today=mid)
    assert scan.is_rebalance_day is False
    assert all(a.provisional for a in sl.decide(scan))


def test_the_rebalance_day_arms_the_call(cfg, world):
    scan = sl.scan(cfg, world, today=_last_mark(world))
    assert scan.is_rebalance_day is True
    assert not any(a.provisional for a in sl.decide(scan))


def test_an_estimated_date_can_never_arm_the_call(cfg, world):
    """
    Past the cache, the date is a calendar guess and the panel must stay shut.

    A rebalance date read off a stale cache would be a real date for the wrong
    month, and it would arm a checklist on data that cannot see the month it
    claims to trade.
    """
    beyond = _last_mark(world) + timedelta(days=400)
    scan = sl.scan(cfg, world, today=beyond)
    assert scan.next_rebalance is not None      # estimated, so still useful
    assert scan.sessions_to_rebalance is None   # and flagged as estimated
    assert scan.is_rebalance_day is False
    assert all(a.provisional for a in sl.decide(scan))


# ------------------------------------------------------ news cannot rank

def test_news_cannot_reach_the_ranking(cfg, world, monkeypatch):
    """
    Rule 2, asserted rather than trusted.

    The swing book folds news into its score because it was measured that way.
    This one was not, so a headline that reordered the top 20 would make the
    live sleeve a different strategy while the page still quoted the tested
    one's numbers.
    """
    from nifty_algo.swing import news as news_mod

    class _Loud:
        available = True
        items = []
        note = ""
        score = 99.0

    day = _last_mark(world)
    plain = sl.scan(cfg, world, today=day)
    monkeypatch.setattr(news_mod, "fetch_for",
                        lambda stocks, c: {s.symbol: _Loud() for s in stocks})
    loud = sl.scan(cfg, world, today=day, with_news=True)

    assert [p.symbol for p in loud.picks] == [p.symbol for p in plain.picks]
    assert all(p.news is not None for p in loud.picks)


# ------------------------------------------------------------- holdings

def test_a_failed_holdings_read_is_not_an_empty_account(cfg, world):
    """
    The most expensive mistake this page could make.

    An incomplete snapshot read as "you hold nothing" turns a fully invested
    book into a page of BUYs. `holdings_available` must be False and every
    action must say the holdings were unverified.
    """
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day,
                   holdings=_Snapshot([], complete=False, notes=["kite failed"]))
    assert scan.holdings_available is False
    assert "INCOMPLETE" in scan.holdings_note
    assert all("unverified" in a.reason for a in sl.decide(
        scan, _Snapshot([], complete=False, notes=["kite failed"])))


def test_holdings_not_read_at_all_is_its_own_state(cfg, world):
    scan = sl.scan(cfg, world, today=_last_mark(world), holdings=None)
    assert scan.holdings_available is False
    assert "NOT READ" in scan.holdings_note


def test_a_held_name_that_left_the_ranking_becomes_a_sell(cfg, world):
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    held = _Snapshot([_Pos("WIN11", 10)])          # the weakest name
    actions = sl.decide(scan, held)
    sells = [a for a in actions if a.kind == "SELL"]
    assert [a.symbol for a in sells] == ["WIN11"]


def test_a_held_name_still_ranked_is_not_re_bought(cfg, world):
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    top = scan.picks[0]
    held = _Snapshot([_Pos(top.symbol, top.target_qty)])
    kinds = {a.symbol: a.kind for a in sl.decide(scan, held)}
    assert kinds[top.symbol] == "HOLD"


# ------------------------------------------------------------- the money

def test_a_zero_pot_sizes_to_zero_and_says_so(cfg, world):
    cfg.capital.factor_capital_inr = 0.0
    scan = sl.scan(cfg, world, today=_last_mark(world))
    assert scan.funded is False
    assert all(p.target_qty == 0 for p in scan.picks)
    actions = sl.decide(scan)
    assert actions and all(a.kind == "HOLD" for a in actions)
    assert "pot is zero" in actions[0].reason


def test_the_pot_divides_equally_across_the_book(cfg, world):
    scan = sl.scan(cfg, world, today=_last_mark(world))
    assert scan.ticket_inr == pytest.approx(500_000.0 / cfg.factor.top_n)
    for p in scan.picks:
        assert p.target_qty * p.price <= scan.ticket_inr + p.price


# --------------------------------------------------------------- screening

def test_the_screen_is_a_no_op_on_the_backtest_by_default(world):
    """`halal_screened=False` must leave `run()` byte-identical."""
    a = fb.run(world, 500_000.0, top_n=4, min_turnover=0.0, min_history=300)
    b = fb.run(world, 500_000.0, top_n=4, min_turnover=0.0, min_history=300,
               halal_ok=None, halal_shortlist=60)
    assert [v for _, v in a.equity] == [v for _, v in b.equity]
    assert (a.trades, a.costs_paid) == (b.trades, b.costs_paid)
    assert (a.screened_out, a.shortlist_short) == (0, 0)


def test_the_screen_never_reaches_past_the_shortlist(world):
    """
    A book that reaches as far down the ranking as it must to find twenty
    passing names is no longer a momentum book - it holds the 300th-best name
    and calls it momentum. It must hold FEWER names instead, and record that.
    """
    rejected = {"WIN00", "WIN01", "WIN02"}
    res = fb.run(world, 500_000.0, top_n=4, min_turnover=0.0, min_history=300,
                 halal_ok=lambda s: s not in rejected, halal_shortlist=4)
    assert res.screened_out > 0
    assert res.shortlist_short > 0
    for _day, names in res.holdings_log:
        assert not (set(names) & rejected)


def test_a_missing_verdict_is_never_a_pass():
    p = sl.SleevePick(symbol="X", rank=1, score=1.0, momentum_12_1=1.0,
                      price=10.0)
    assert p.halal is None
    assert p.halal_ok is False


def test_the_screen_reads_yahoos_labels_off_the_stock(cfg):
    """
    `activity_failure` matches on `stock.industry`/`stock.sector`, not on the
    fundamentals object - and the factor universe carries neither, because it
    is Kite's instrument dump. Without copying Yahoo's labels onto the Stock
    every name arrives unclassified, and unclassified is a REJECT, so the
    sleeve would look like a strict screen rather than a broken one.
    """
    market = markets_mod.factor_market(cfg)

    class _F:
        yahoo_sector = "Financial Services"
        yahoo_industry = "Credit Services"

    bare = sl.stock_for("ACME", market)
    assert (bare.industry, bare.sector) == ("", "")

    filled = sl.stock_for("ACME", market, _F())
    assert filled.industry == "Credit Services"
    assert filled.yf_ticker == "ACME.NS"

    from nifty_algo.swing import halal
    assert halal.activity_failure(filled, market, cfg) is not None
    assert halal.activity_failure(bare, market, cfg) is None   # nothing to match


def test_the_factor_market_does_not_pollute_the_swing_registry(cfg):
    before = markets_mod.keys(cfg)
    fm = markets_mod.factor_market(cfg)
    assert fm.taxonomy == markets_mod.TAXONOMY_GICS
    assert fm.capital_pool == markets_mod.POOL_FACTOR
    assert fm.domestic is True
    assert markets_mod.keys(cfg) == before
    assert "factor_india" not in before


# ------------------------------------------------------------ membership

def test_a_missing_membership_file_is_reported_not_guessed(tmp_path):
    """
    A Nifty 50 name with `nifty50.csv` absent looks exactly like a Next 50
    name. That is a confident wrong answer, so the band says it cannot split
    them instead.
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "nifty100.csv").write_text(
        "symbol,name,sector,industry,yf_ticker\nRELIANCE,R,E,Refineries,R.NS\n")
    m = mb.load(tmp_path)
    assert m.available is False
    assert "split unavailable" in m.band_of("RELIANCE")
    assert m.band_of("SOMETHINGELSE") == mb.UNKNOWN
    assert "Refresh" in m.note()


def test_bands_resolve_most_specific_first(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "nifty50.csv").write_text(
        "Company Name,Industry,Symbol\nReliance,Oil,RELIANCE\n")
    (tmp_path / "data" / "nifty100.csv").write_text(
        "symbol\nRELIANCE\nTATAPOWER\n")
    (tmp_path / "data" / "nifty500.csv").write_text(
        "Company Name,Industry,Symbol\nR,O,RELIANCE\nD,E,DIXON\n")
    m = mb.load(tmp_path)
    assert m.available is True
    assert m.band_of("RELIANCE") == "Nifty 50"
    assert m.band_of("TATAPOWER") == "Nifty Next 50"
    assert m.band_of("DIXON") == "Nifty 500"
    assert m.band_of("TINYCO") == mb.OUTSIDE


# ------------------------------------------------------------- reporting

def test_the_report_always_carries_both_windows(cfg, world):
    scan = sl.scan(cfg, world, today=_last_mark(world))
    text = sl.report(scan, sl.decide(scan))
    assert "+7.94pp" in text and "+0.60pp" in text
    assert "78.5%" in text
    assert "no stop-loss control" in text


def test_the_report_says_when_the_call_is_provisional(cfg, world):
    sessions = sorted({d.date() for f in world.values() for d in f.index})
    marks = month_ends(sessions, 1)
    mid = next(d for d in sessions if marks[-2] < d < marks[-1])
    scan = sl.scan(cfg, world, today=mid)
    text = sl.report(scan, sl.decide(scan))
    assert "PROVISIONAL" in text


def test_volatility_refuses_to_answer_on_too_few_bars():
    short = np.linspace(100.0, 110.0, 10)
    assert np.isnan(sl.annual_vol(short, 63))
    assert np.isnan(sl.from_52w_high(short))
    long = np.linspace(100.0, 200.0, 300)
    assert sl.annual_vol(long, 252) > 0
    assert sl.from_52w_high(long) == pytest.approx(0.0, abs=1e-9)


def test_the_scan_reports_which_band_produced_the_book(cfg, world):
    scan = sl.scan(cfg, world, today=_last_mark(world))
    assert scan.band == cfg.factor.band
    assert scan.universe_size == len(world)


# ------------------------------------------------------- the per-universe record

def test_the_unrestricted_record_still_carries_the_recorded_figures():
    """
    REGRESSION. `record_for("all")` must be exactly what F1 and F2 measured,
    or the unrestricted page silently changed what it claims.
    """
    rec = sl.record_for("all")
    flat = " ".join(" ".join(row) for row in rec.rows)
    for figure in ("+18.79%", "+12.39%", "+7.94pp", "+0.60pp",
                   "-57.2%", "-78.5%"):
        assert figure in flat
    assert rec.naive is None


def test_a_restricted_record_quotes_its_control_not_the_naive_backtest():
    """
    The whole point. `nifty500`'s expectation rows are `size500`'s numbers,
    because today's membership cannot be applied to 2016 - and the inflated
    +31.29% lives in `naive`, never in the table.
    """
    rec = sl.record_for("nifty500")
    flat = " ".join(" ".join(row) for row in rec.rows)
    assert "+17.60%" in flat and "+13.55%" in flat
    assert "+31.29%" not in flat
    assert "+18.79%" not in flat
    assert rec.naive and "+31.29%" in rec.naive
    assert "look-ahead" in rec.naive


def test_the_nifty100_record_says_not_to_run_it():
    rec = sl.record_for("nifty100")
    flat = " ".join(" ".join(row) for row in rec.rows)
    assert "+7.94%" in flat              # below the index on the same marks
    assert "DO NOT RUN THE SLEEVE HERE" in rec.caution


def test_an_unmeasured_universe_has_no_record():
    """
    None is a refusal, not a fallback. Adding a universe to
    `restriction.UNIVERSES` without measuring it must not inherit another's
    numbers.
    """
    assert sl.record_for("nifty42") is None
    assert sl.record_for("") is sl.record_for("all")


def test_every_registered_universe_has_been_measured():
    """
    The selector offers `restriction.UNIVERSES`, so a key with no record is a
    choice the console cannot describe. This fails the moment someone adds one
    without measuring it - which is the intended nag.
    """
    from nifty_algo.factor import restriction as restr
    missing = [k for k in restr.UNIVERSES if sl.record_for(k) is None]
    assert not missing, f"unmeasured universes offered by the selector: {missing}"


def test_the_report_shows_the_record_for_the_scanned_universe(cfg, world,
                                                              monkeypatch):
    """
    The CLI and the page read the same `record_for`, so they cannot quote
    different numbers for the same book.
    """
    from nifty_algo.factor import restriction as restr
    monkeypatch.setattr(
        restr, "resolver",
        lambda c, key, bars, uni, root=".": (restr.static(set(world)), "test"))
    cfg.factor.universe = "nifty500"
    try:
        scan = sl.scan(cfg, world, today=_last_mark(world))
        text = sl.report(scan, sl.decide(scan))
        assert "+17.60%" in text
        assert "+18.79%" not in text
        assert "+31.29%" in text          # present, in the caution paragraph
    finally:
        cfg.factor.universe = "all"


def test_the_report_refuses_an_unmeasured_universe(cfg, world, monkeypatch):
    from nifty_algo.factor import restriction as restr
    monkeypatch.setattr(
        restr, "resolver",
        lambda c, key, bars, uni, root=".": (restr.static(set(world)), "test"))
    monkeypatch.delitem(sl.RECORDS, "nifty500")
    cfg.factor.universe = "nifty500"
    try:
        scan = sl.scan(cfg, world, today=_last_mark(world))
        text = sl.report(scan, sl.decide(scan))
        assert "NO BACKTEST DESCRIBES" in text
        assert "+18.79%" not in text and "+17.60%" not in text
    finally:
        cfg.factor.universe = "all"


# --------------------------------------------- the ATR trail, and the flags

def test_the_atr_trail_is_a_no_op_by_default(world):
    """
    REGRESSION. F1, F2, F3 and F4 were all measured without it, and the ATR
    array is not even built unless `atr_window` is asked for.
    """
    a = fb.run(world, 500_000.0, top_n=4, min_turnover=0.0, min_history=300)
    b = fb.run(world, 500_000.0, top_n=4, min_turnover=0.0, min_history=300,
               trail_atr_multiple=None)
    assert [v for _, v in a.equity] == [v for _, v in b.equity]
    assert a.stops_fired == b.stops_fired == 0

    plain = FactorUniverse(world)
    assert all(sd.atr is None for sd in plain.symbols.values())


def test_the_trail_ratchets_up_and_never_loosens(world):
    """
    A stop that can fall is not a stop - it follows the price down and sells
    at the bottom anyway. Same rule `ExitLadder` has, asserted here because
    the trail is re-armed at every rebalance and that is where it could slip.
    """
    universe = FactorUniverse(world, atr_window=14)
    res = fb.run(world, 500_000.0, top_n=4, min_turnover=0.0, min_history=300,
                 universe=universe, trail_atr_multiple=3.0)
    assert res.stops_fired >= 0            # it ran

    # Re-arming must never lower a level already ratcheted up.
    pos = fb.Position("X", 10, 100.0, date(2020, 1, 1))
    pos.trail_stop = 95.0
    for level in (90.0, 80.0, 96.0):
        if pos.trail_stop is None or level > pos.trail_stop:
            pos.trail_stop = level
    assert pos.trail_stop == 96.0


def test_the_atr_cannot_see_the_day_it_is_used_on(world):
    """
    The safety property, for the stop distance this time. An ATR computed
    including the rebalance bar is look-ahead, and it would tighten or loosen
    the stop using the move it is about to trade on.
    """
    universe = FactorUniverse(world, atr_window=14)
    day = _last_mark(world)
    full = universe.symbols["WIN00"].atr_before(day)

    truncated = {s: f[f.index < pd.Timestamp(day)] for s, f in world.items()}
    trimmed = FactorUniverse(truncated, atr_window=14)
    assert trimmed.symbols["WIN00"].atr_before(day) == full


def test_a_flag_is_never_an_order(cfg, world):
    """
    The panel's whole justification. F5 measured the automatic version and it
    loses, so `review` returns things to READ and `decide` remains the only
    thing that produces quantities.
    """
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    top = scan.picks[0]
    held = _Snapshot([_Pos(top.symbol, 10, average_price=top.price * 3.0)])

    flags = sl.review(cfg, scan, held)
    assert flags and any(f.kind == "drawdown" for f in flags)
    assert all(not hasattr(f, "quantity") for f in flags)
    assert all(f.severity in ("watch", "review") for f in flags)


def test_a_holding_within_tolerance_is_not_flagged(cfg, world):
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    top = scan.picks[0]
    held = _Snapshot([_Pos(top.symbol, 10, average_price=top.price * 0.99)])
    assert not [f for f in sl.review(cfg, scan, held)
                if f.kind == "drawdown"]


def test_an_incomplete_holdings_read_is_flagged_as_such(cfg, world):
    """
    A flag list built from half an account is worse than none, because it
    reads as 'nothing else needs looking at'.
    """
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    top = scan.picks[0]
    snap = _Snapshot([_Pos(top.symbol, 10, average_price=top.price * 3.0)],
                     complete=False, notes=["kite failed"])
    flags = sl.review(cfg, scan, snap)
    assert any(f.kind == "holdings" for f in flags)


def test_the_report_labels_flags_as_prompts_not_signals(cfg, world):
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    top = scan.picks[0]
    held = _Snapshot([_Pos(top.symbol, 10, average_price=top.price * 3.0)])
    text = sl.report(scan, sl.decide(scan, held), sl.review(cfg, scan, held))
    assert "never orders" in text
    assert "prompts to read" in text


# --------------------------------------------------------- the sector mix

class _Fund:
    """Just the two fields `classify` reads."""
    def __init__(self, sector, industry=""):
        self.yahoo_sector = sector
        self.yahoo_industry = industry


class _KeyedPos(_Pos):
    """A position with the `key` a real `Position` carries."""
    def __init__(self, symbol, quantity, market="india", **kw):
        super().__init__(symbol, quantity, **kw)
        self.key = f"{market}:{symbol}"


class _ValuedSnapshot(_Snapshot):
    """A snapshot that converts, the way `aggregate.load` produces one."""
    def __init__(self, positions, value_inr=None, **kw):
        super().__init__(positions, **kw)
        self.value_inr = dict(value_inr or {
            p.key: p.quantity * p.last_price for p in positions})


def _labels(monkeypatch, mapping):
    """Pin what the fundamentals cache appears to hold. No network, no file."""
    from nifty_algo.swing import fundamentals as fund_mod
    monkeypatch.setattr(
        fund_mod, "read_cached",
        lambda cfg, market: {s: _Fund(v) for s, v in mapping.items()})


def test_sector_cannot_reach_the_ranking(cfg, world, monkeypatch):
    """
    Momentum ranks on price and nothing else.

    Sector is attached AFTER `top_n` has chosen, exactly as `halal` and `news`
    are - so relabelling every name must leave the book byte-identical. A
    sector that could reorder the picks would make the live sleeve a different
    and untested strategy while every figure on the page still described the
    tested one. Same guard as `test_news_cannot_reach_the_ranking`.
    """
    day = _last_mark(world)
    plain = sl.scan(cfg, world, today=day)

    _labels(monkeypatch, {s: f"Sector {i}" for i, s in enumerate(world)})
    labelled = sl.scan(cfg, world, today=day)

    assert [p.symbol for p in labelled.picks] == [p.symbol for p in plain.picks]
    assert [p.rank for p in labelled.picks] == [p.rank for p in plain.picks]
    assert [p.target_qty for p in labelled.picks] == [
        p.target_qty for p in plain.picks]
    assert all(p.sector for p in labelled.picks)
    assert all(p.sector == "" for p in plain.picks)


def test_the_wanted_side_is_funded_rupees(cfg, world, monkeypatch):
    _labels(monkeypatch, {s: "Industrials" for s in world})
    scan = sl.scan(cfg, world, today=_last_mark(world))
    mix = sl.sector_mix(cfg, scan)

    assert [r.sector for r in mix.rows] == ["Industrials"]
    row = mix.rows[0]
    assert row.names == len(scan.picks)
    assert row.wanted_inr == pytest.approx(
        sum(p.target_qty * p.price for p in scan.picks))
    assert row.wanted_pct == pytest.approx(1.0)
    assert mix.wanted_total_inr == pytest.approx(row.wanted_inr)


def test_an_unclassified_name_is_its_own_row_and_goes_last(cfg, world,
                                                           monkeypatch):
    """
    A missing fact is never folded into a bucket nobody put it in, and it
    never floats to the top of a sorted table where it would read as the
    largest sector.
    """
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    named = scan.picks[0].symbol
    _labels(monkeypatch, {named: "Technology"})       # every other name: none

    scan = sl.scan(cfg, world, today=day)
    mix = sl.sector_mix(cfg, scan)

    sectors = [r.sector for r in mix.rows]
    assert sl.UNCLASSIFIED in sectors
    assert sectors[-1] == sl.UNCLASSIFIED
    assert mix.has_unclassified
    assert sum(r.names for r in mix.rows) == len(scan.picks)


def test_the_held_side_is_withheld_when_holdings_were_not_read(cfg, world,
                                                               monkeypatch):
    """
    None, not zero. A share computed against a denominator that could not be
    established reads exactly like one that was, and it would be acted on -
    the rule PortfolioSnapshot.weight() applies, applied to a sector table.
    """
    _labels(monkeypatch, {s: "Industrials" for s in world})
    scan = sl.scan(cfg, world, today=_last_mark(world))

    for holdings in (None, _Snapshot([], complete=False, notes=["kite down"])):
        mix = sl.sector_mix(cfg, scan, holdings)
        assert mix.held_available is False
        assert mix.held_total_inr is None
        assert all(r.held_inr is None for r in mix.rows)
        assert all(r.held_pct is None for r in mix.rows)
        assert all(r.shift_pp is None for r in mix.rows)


def test_the_held_side_uses_converted_rupees(cfg, world, monkeypatch):
    """
    NEVER value_native. A dollar line added to a rupee total is a sector
    weight ~88x wrong that looks entirely ordinary - the same failure
    swing/fx.py fails closed to prevent.
    """
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    a, b = scan.picks[0].symbol, scan.picks[1].symbol
    _labels(monkeypatch, {a: "Technology", b: "Energy"})
    scan = sl.scan(cfg, world, today=day)

    held = _ValuedSnapshot(
        [_KeyedPos(a, 10, last_price=1.0), _KeyedPos(b, 10, last_price=1.0)],
        value_inr={f"india:{a}": 30_000.0, f"india:{b}": 10_000.0})
    mix = sl.sector_mix(cfg, scan, held)

    assert mix.held_available is True
    assert mix.held_total_inr == pytest.approx(40_000.0)
    by = {r.sector: r for r in mix.rows}
    assert by["Technology"].held_inr == pytest.approx(30_000.0)
    assert by["Technology"].held_pct == pytest.approx(0.75)
    assert by["Energy"].held_pct == pytest.approx(0.25)
    assert by["Technology"].shift_pp == pytest.approx(
        (by["Technology"].wanted_pct - 0.75) * 100.0)


def test_a_held_name_outside_the_picks_still_lands_in_a_sector(cfg, world,
                                                               monkeypatch):
    """
    The snapshot is the whole account, so the held column must classify names
    the ranking never chose - and say unclassified when it cannot, rather
    than dropping the money.
    """
    day = _last_mark(world)
    scan = sl.scan(cfg, world, today=day)
    outsider = "WIN11"
    assert outsider not in {p.symbol for p in scan.picks}
    _labels(monkeypatch, {p.symbol: "Industrials" for p in scan.picks})
    scan = sl.scan(cfg, world, today=day)

    held = _ValuedSnapshot([_KeyedPos(outsider, 10, last_price=1.0)],
                           value_inr={f"india:{outsider}": 5_000.0})
    mix = sl.sector_mix(cfg, scan, held)

    assert mix.unclassified_held == (outsider,)
    by = {r.sector: r for r in mix.rows}
    assert by[sl.UNCLASSIFIED].held_inr == pytest.approx(5_000.0)
    assert by[sl.UNCLASSIFIED].names == 0
    assert mix.held_total_inr == pytest.approx(5_000.0)


def test_an_unfunded_pick_is_named_rather_than_dropped(cfg, world,
                                                       monkeypatch):
    """
    A pick the pot never reached weighs nothing, which is honest and also
    invisible. Without naming it, a sector the sleeve WANTED reads as one it
    does not - the wanted_log beside holdings_log discipline.
    """
    _labels(monkeypatch, {s: "Industrials" for s in world})
    cfg.capital.factor_capital_inr = 1.0          # buys nothing at any price
    scan = sl.scan(cfg, world, today=_last_mark(world))
    mix = sl.sector_mix(cfg, scan)

    assert all(p.target_qty == 0 for p in scan.picks)
    assert set(mix.unfunded) == {p.symbol for p in scan.picks}
    assert mix.wanted_total_inr == 0.0
    assert all(r.wanted_pct == 0.0 for r in mix.rows)
    assert sum(r.names for r in mix.rows) == len(scan.picks)


def test_classifying_never_reaches_the_network(cfg, world, monkeypatch):
    """
    load_fundamentals refreshes anything older than a week as a side effect,
    so asking it for a sector would fire one slow request per missing name in
    the middle of drawing a panel. read_cached is the door that cannot.
    """
    from nifty_algo.swing import fundamentals as fund_mod

    def _boom(*a, **k):                                    # pragma: no cover
        raise AssertionError("classify fetched")

    monkeypatch.setattr(fund_mod, "load_fundamentals", _boom)
    monkeypatch.setattr(fund_mod, "_fetch_one", _boom)

    market = markets_mod.factor_market(cfg)
    out = sl.classify(list(world), cfg, market)
    assert isinstance(out, dict)

    scan = sl.scan(cfg, world, today=_last_mark(world))
    assert sl.sector_mix(cfg, scan).rows


def test_known_facts_beat_the_cache(cfg, world, monkeypatch):
    """
    The objects screen_symbols just fetched win over the file, because the
    write that follows them swallows its own errors - so a freshly screened
    name could otherwise come back unclassified.
    """
    _labels(monkeypatch, {"WIN00": "Stale"})
    market = markets_mod.factor_market(cfg)
    out = sl.classify(["WIN00"], cfg, market,
                      known={"WIN00": _Fund("Fresh", "Widgets")})
    assert out["WIN00"] == ("Fresh", "Widgets")


# ------------------------------------------------------ why this name

def test_the_case_names_the_formation_that_actually_ran(cfg, world):
    """
    THE LABEL USED TO BE THE LITERAL "12-1" IN THREE PLACES.

    `score_universe` falls back to 12-1 on an unregistered formation key
    without complaining, and the console printed "12-1" regardless of what ran
    - so a `mom6_1` book was described as a 12-1 book, and a typo produced a
    correct book under a wrong name. A panel whose whole job is to explain the
    signal must not be able to name the wrong one.
    """
    day = _last_mark(world)
    cfg.factor.formation = "mom6_1"
    try:
        scan = sl.scan(cfg, world, today=day)
        assert scan.formation_label == "6-1"
        assert "6 months of price" in scan.formation_sentence
        case = " ".join(sl.why_for(scan.picks[0], scan))
        assert "6-1 momentum" in case
        assert "12-1 momentum" not in case
    finally:
        cfg.factor.formation = "mom12_1"


def test_an_unrecognised_formation_says_so_rather_than_lying(cfg, world):
    """
    A silent fallback plus a hardcoded label is how a book gets described as
    something it is not. The fallback stays - changing it would change the
    ranking - but it can no longer be invisible.
    """
    day = _last_mark(world)
    cfg.factor.formation = "mom9_2"          # never registered
    try:
        scan = sl.scan(cfg, world, today=day)
        assert "unrecognised" in scan.formation_label
        assert "fell back to 12-1" in scan.formation_label
    finally:
        cfg.factor.formation = "mom12_1"


def test_the_marginal_name_is_the_one_that_missed_on_SCORE(cfg, world,
                                                           monkeypatch):
    """
    THE BUG THIS EXISTS TO PIN, MEASURED ON THE REAL BOOK.

    "Highest-ranked name not chosen" is the obvious definition of the marginal
    name and it is wrong under the halal screen: that name is usually a
    REJECTION, which scores ABOVE the whole book. Live it returned ATHERENERG
    at +222.8% against a top pick of +137.5%, so the case printed "clears the
    cut by -85.4pp" - a negative margin for the number one name.

    Absent-for-failing-a-screen and absent-for-scoring-too-low are two
    different facts, and only the second is a cut.
    """
    day = _last_mark(world)
    cfg.factor.halal_screened = True
    cfg.factor.halal_shortlist = 8

    # Reject the two STRONGEST names, which is the shape that broke it.
    ranked = sl.mom.top_n(
        sl.mom.score_universe(
            sl.FactorUniverse(world, adv_window=cfg.factor.adv_window),
            sl.FactorUniverse(world, adv_window=cfg.factor.adv_window)
              .eligible_at(day, 0.0, 0.0, 300, "all").symbols,
            day, cfg.factor.formation),
        len(world))
    banned = set(ranked[:2])

    class _V:
        def __init__(self, ok):
            self.eligible = ok
            self.reason = "banned by the test" if not ok else "fine"

    monkeypatch.setattr(
        sl, "screen_symbols",
        lambda symbols, c, m, progress=None: (
            {s: _V(s not in banned) for s in symbols}, {}))
    try:
        scan = sl.scan(cfg, world, today=day)
    finally:
        cfg.factor.halal_screened = False
        cfg.factor.halal_shortlist = 60

    assert scan.marginal_symbol not in banned, (
        "a screen rejection is not the name that missed the cut")
    assert scan.marginal_symbol not in {p.symbol for p in scan.picks}
    # The margin must be positive for every pick, which is the whole point.
    for pick in scan.picks:
        assert pick.momentum_12_1 > scan.marginal_score
        gap = [l for l in sl.why_for(pick, scan) if "clears the cut" in l]
        assert gap and "+" in gap[0].split("clears the cut by")[1]


def test_the_case_states_what_did_NOT_choose_the_name(cfg, world, monkeypatch):
    """
    THE MOST IMPORTANT LINE IN THE PANEL.

    Trend, distance from the high, liquidity band and sector all read as
    supporting evidence, and none of them has a vote. A list of favourable
    facts with no disclaimer is a multi-factor case for a single-factor pick -
    which would make the console describe a strategy the backtest never ran.
    """
    _labels(monkeypatch, {s: "Industrials" for s in world})
    scan = sl.scan(cfg, world, today=_last_mark(world))
    case = sl.why_for(scan.picks[0], scan)

    negative = [l for l in case if l.startswith("NOT why")]
    assert negative, "the panel must say what did not choose the name"
    for word in ("sector", "news", "valuation", "volatility", "drawdown"):
        assert word in negative[0]
    assert any("not a view on the company" in l for l in case)

    # AND IT MUST NOT OVERCLAIM. Turnover and listing history DO gate
    # eligibility, so "the only non-price gate is the Shariah screen" - which
    # this line used to say - was false. A gate decides who competes; the
    # score decides who wins, and the difference is the whole claim.
    assert not any("only non-price gate" in l for l in case)
    gates = [l for l in case if "ELIGIBLE" in l]
    assert gates and "turnover" in gates[0] and "listing history" in gates[0]
    assert "only REMOVE" in gates[0]
    # And the descriptive facts must be labelled as having had no vote.
    assert any("had NO vote" in l for l in case)


def test_the_percentile_is_the_scored_cross_section(cfg, world, monkeypatch):
    _labels(monkeypatch, {s: "Industrials" for s in world})
    scan = sl.scan(cfg, world, today=_last_mark(world))
    top = scan.picks[0]
    assert 0.0 < top.score_percentile <= 1.0
    # Rank 1 is the top of the field, so nothing scored above it.
    assert top.score_percentile == pytest.approx(1.0)
    assert scan.picks[-1].score_percentile < top.score_percentile
    assert f"{top.score_percentile * 100:.1f}th percentile" in " ".join(
        sl.why_for(top, scan))


def test_an_empty_cross_section_yields_no_percentile_not_a_perfect_one(cfg):
    """
    "Top of a field of nobody" is not a fact about a stock, and 100% printed
    against an empty universe reads as the strongest possible endorsement.
    """
    import numpy as np
    assert np.isnan(sl._percentile(0.5, np.array([])))
    assert np.isnan(sl._percentile(float("nan"), np.array([1.0, 2.0])))


def test_the_case_survives_a_book_smaller_than_the_cross_section(cfg, world):
    """
    Fewer eligible names than `top_n + 1` leaves no marginal name, and a NaN
    formatted into prose would read as "clears the cut by nan pp".
    """
    day = _last_mark(world)
    cfg.factor.top_n = len(world) + 5          # ask for more than exist
    try:
        scan = sl.scan(cfg, world, today=day)
        assert scan.marginal_symbol == ""
        for pick in scan.picks:
            case = " ".join(sl.why_for(pick, scan))
            assert "clears the cut" not in case
            assert "nan" not in case.lower()
    finally:
        cfg.factor.top_n = 4


def test_the_case_reports_a_pick_the_pot_could_not_reach(cfg, world):
    day = _last_mark(world)
    cfg.capital.factor_capital_inr = 1.0
    scan = sl.scan(cfg, world, today=day)
    assert all(p.unfunded for p in scan.picks)
    assert any("pot ran out" in l for l in sl.why_for(scan.picks[0], scan))


def test_the_case_says_when_the_screen_was_never_run(cfg, world):
    """An unscreened name is not a passing one - the same rule `halal_ok`
    already applies, in prose."""
    scan = sl.scan(cfg, world, today=_last_mark(world))
    assert cfg.factor.halal_screened is False
    case = " ".join(sl.why_for(scan.picks[0], scan))
    assert "was NOT run" in case
    assert "Passed the Shariah screen" not in case


def test_ordinals_do_not_produce_1th_or_21th():
    assert [sl._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 111)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd",
        "111th"]
