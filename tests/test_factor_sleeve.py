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
