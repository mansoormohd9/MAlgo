"""
The pipeline: a hundred stocks in, at most three tickets out.

Every network call is stubbed here. What is under test is the ORDER and the
BOOKKEEPING of the gates - that the halal screen runs before anything is
downloaded, that each gate rejects for the stated reason, that the sector cap
and the top-N cap are applied after ranking and not before, that sizing comes
off the risk budget rather than off the capital, and above all that every
single symbol ends up accounted for. A scanner that quietly loses a stock
between two stages is a scanner whose "no picks today" cannot be trusted.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest
from conftest import daily_bars, trending

from nifty_algo.config import Config
from nifty_algo.swing import news as news_mod
from nifty_algo.swing import scanner
from nifty_algo.swing.fundamentals import Fundamentals
from nifty_algo.swing.prices import PriceSet
from nifty_algo.swing.universe import Stock

TODAY = date(2026, 8, 20)


# ---------------------------------------------------------------- the world

def make_stock(symbol, sector="Information Technology",
               industry="Computers - Software & Consulting") -> Stock:
    return Stock(symbol, f"{symbol} Ltd", sector, industry, f"{symbol}.NS")


def clean_fundamentals(symbol, **kw) -> Fundamentals:
    base = dict(total_assets=1000.0, total_debt=100.0,
                cash_and_investments=100.0, receivables=100.0,
                market_cap=5e11, balance_sheet_date="2026-03-31",
                fetched_at="2026-08-19T09:00:00")
    base.update(kw)
    return Fundamentals(symbol=symbol, **base)


@pytest.fixture
def cfg() -> Config:
    c = Config()
    # The Indian swing pot ships at Rs 0 so an unfunded account stands the
    # market down rather than sizing off the option book. Fund it here or
    # every scan in this file returns `stood_down` and asserts nothing.
    c.capital.swing_capital_inr = 100_000.0
    # The synthetic series start at 100 and drift up; the shipped turnover and
    # price floors are calibrated for real Nifty 100 names, not for them.
    # They live on the market now - they are denominated in its currency.
    c.swing.markets["india"].min_price = 1.0
    c.swing.markets["india"].min_avg_turnover = 0.0
    return c


@pytest.fixture
def world(monkeypatch):
    """
    Stub the three network boundaries and record what they were asked for.

    Returns a mutable dict the tests configure: `stocks`, `bars`, `funds`,
    `news`, plus `asked_for` which captures the tickers the price fetch was
    handed - that is how the "halal runs before the download" claim is tested.
    """
    state = {"stocks": [], "bars": {}, "funds": {}, "news": {},
             "asked_for": None, "news_asked_for": None,
             "priced_market": None, "funds_market": None, "divisors": {}}

    def fake_prices(tickers, cfg, market, benchmark=None, force_refresh=False,
                    divisors=None, progress=None):
        state["asked_for"] = set(tickers)
        state["priced_market"] = market.key
        state["divisors"] = dict(divisors or {})
        bars = {s: state["bars"][s] for s in tickers if s in state["bars"]}
        return PriceSet(
            bars=bars,
            benchmark=state["bars"].get("__BENCHMARK__",
                                        daily_bars(trending(seed=99))),
            missing=[s for s in tickers if s not in state["bars"]],
        )

    def fake_fundamentals(stocks, cfg, market, force_refresh=False,
                          progress=None):
        state["funds_market"] = market.key
        return {s.symbol: state["funds"].get(s.symbol,
                                             clean_fundamentals(s.symbol))
                for s in stocks}

    def fake_news(stocks, cfg, progress=None):
        state["news_asked_for"] = [s.symbol for s in stocks]
        return {s.symbol: state["news"].get(
            s.symbol, news_mod.NewsResult(s.symbol, available=True,
                                          note="stubbed"))
            for s in stocks}

    monkeypatch.setattr(scanner.prices_mod, "load_prices", fake_prices)
    monkeypatch.setattr(scanner.fundamentals_mod, "load_fundamentals",
                        fake_fundamentals)
    monkeypatch.setattr(scanner.fundamentals_mod, "cache_age_days",
                        lambda cfg: 1)
    monkeypatch.setattr(scanner.news_mod, "fetch_for", fake_news)
    return state


#: Seeds whose random walk actually produces a setup clearing the 2:1 floor.
#: Picked rather than counted from 1 upwards because roughly a third of seeds
#: produce no setup at all - and a gate test fed a stock that was never a
#: candidate passes without testing the gate.
QUALIFYING_SEEDS = (2, 3, 5, 6, 7, 8, 10, 11, 12, 15, 17, 18, 20, 24, 25,
                    26, 27, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)


def populate(world, symbols, sector="Information Technology",
             industry="Computers - Software & Consulting", seed_base=0):
    """Give each symbol a trending series that produces a qualifying setup."""
    for i, symbol in enumerate(symbols):
        seed = QUALIFYING_SEEDS[(seed_base + i) % len(QUALIFYING_SEEDS)]
        world["stocks"].append(make_stock(symbol, sector, industry))
        world["bars"][symbol] = daily_bars(trending(seed=seed))
    return world["stocks"]


def run(cfg, world, **kw):
    return scanner.scan(cfg, universe=world["stocks"], today=TODAY, **kw)


# ---------------------------------------------------------------- accounting

def test_every_symbol_is_accounted_for(cfg, world):
    populate(world, [f"SYM{i}" for i in range(12)])
    result = run(cfg, world)
    assert result.universe_size == 12
    assert result.accounts_for_everything(), (
        f"{len(result.picks)} picked + {len(result.excluded_halal)} excluded "
        f"+ {len(result.rejections)} rejected != 12")


def test_a_symbol_is_never_both_picked_and_rejected(cfg, world):
    populate(world, [f"SYM{i}" for i in range(12)])
    result = run(cfg, world)
    picked = {p.symbol for p in result.picks}
    rejected = {r["symbol"] for r in result.rejections}
    excluded = {x.symbol for x in result.excluded_halal}
    assert not picked & rejected
    assert not picked & excluded


def test_missing_price_data_is_recorded_not_dropped(cfg, world):
    populate(world, ["GOOD", "GONE"])
    del world["bars"]["GONE"]
    result = run(cfg, world)
    assert [r["symbol"] for r in result.rejections_for(scanner.STAGE_NO_DATA)] \
        == ["GONE"]


# ---------------------------------------------------------------- gate order

def test_the_halal_screen_runs_before_anything_is_downloaded(cfg, world):
    """
    A bank must never cost an HTTP request. This is the cheap-gates-first
    ordering the module docstring claims, asserted rather than assumed.
    """
    populate(world, ["CLEAN"])
    world["stocks"].append(
        make_stock("SOMEBANK", "Financial Services", "Private Sector Bank"))
    world["bars"]["SOMEBANK"] = daily_bars(trending(seed=42))

    result = run(cfg, world)
    assert world["asked_for"] == {"CLEAN"}
    assert [x.symbol for x in result.excluded_halal] == ["SOMEBANK"]


def test_news_is_fetched_only_for_the_finalists(cfg, world):
    cfg.swing.news_finalists = 3
    populate(world, [f"SYM{i}" for i in range(12)])
    run(cfg, world)
    assert world["news_asked_for"] is not None
    assert len(world["news_asked_for"]) <= 3


# ---------------------------------------------------------------- the gates

def test_reward_risk_below_the_floor_is_rejected(cfg, world):
    """
    Raise the floor beyond anything a real chart offers and every candidate
    must fall out at that gate, naming it.
    """
    cfg.capital.session_target_pct = 5.0        # -> reward:risk floor of 100:1
    populate(world, [f"SYM{i}" for i in range(8)])

    result = run(cfg, world)
    assert result.picks == []
    rejects = result.rejections_for(scanner.STAGE_REWARD_RISK)
    assert rejects, "nothing was rejected for reward:risk"
    assert "below the" in rejects[0]["reason"]


def test_a_setup_that_clears_the_floor_survives_it(cfg, world):
    populate(world, [f"SYM{i}" for i in range(8)])
    result = run(cfg, world)
    for pick in result.picks:
        assert pick.reward_risk >= cfg.capital.reward_risk_ratio


def test_earnings_inside_the_blackout_are_stood_aside(cfg, world):
    # Different sectors, so the sector cap is not what separates them.
    populate(world, ["SOON"], seed_base=0)
    populate(world, ["LATER"], sector="Healthcare",
             industry="Pharmaceuticals", seed_base=1)
    world["funds"]["LATER"] = clean_fundamentals(
        "LATER", next_earnings_date=(TODAY + timedelta(days=45)).isoformat())

    # Without the blackout both are picks, so the gate is doing the work.
    baseline = run(cfg, world)
    assert {p.symbol for p in baseline.picks} == {"SOON", "LATER"}

    world["funds"]["SOON"] = clean_fundamentals(
        "SOON", next_earnings_date=(TODAY + timedelta(days=2)).isoformat())
    result = run(cfg, world)

    blacked = result.rejections_for(scanner.STAGE_EARNINGS)
    assert [r["symbol"] for r in blacked] == ["SOON"]
    assert "coin flip" in blacked[0]["reason"]
    assert {p.symbol for p in result.picks} == {"LATER"}


def test_an_unknown_earnings_date_does_not_block_the_trade(cfg, world):
    """Yahoo has no date for plenty of names; that is not a reason to refuse."""
    populate(world, ["NODATE"])
    world["funds"]["NODATE"] = clean_fundamentals("NODATE",
                                                  next_earnings_date=None)
    result = run(cfg, world)
    assert not result.rejections_for(scanner.STAGE_EARNINGS)


def test_a_news_veto_kills_a_technically_perfect_candidate(cfg, world):
    populate(world, ["VETOED"])
    assert run(cfg, world).picks, "the stock must be a pick before the veto"

    world["news"]["VETOED"] = news_mod.NewsResult(
        "VETOED", available=True, veto="SEBI investigation",
        veto_headline="SEBI probe into VETOED promoter dealings")
    result = run(cfg, world)
    assert result.picks == []
    vetoes = result.rejections_for(scanner.STAGE_NEWS)
    assert len(vetoes) == 1
    assert "SEBI investigation" in vetoes[0]["reason"]
    assert "promoter dealings" in vetoes[0]["reason"], \
        "the headline that did it must be shown, so a false veto is arguable"


def test_illiquid_names_are_rejected_for_tradeability(cfg, world):
    cfg.swing.markets["india"].min_avg_turnover = 1e9
    populate(world, ["THIN"])
    result = run(cfg, world)
    assert result.rejections_for(scanner.STAGE_TRADEABILITY)


# ---------------------------------------------------------------- ranking

def test_at_most_top_n_picks_are_returned(cfg, world):
    populate(world, [f"SYM{i}" for i in range(20)],
             sector="Healthcare", industry="Pharmaceuticals")
    cfg.swing.max_per_sector = 99          # isolate the top-N cap
    result = run(cfg, world)
    assert len(result.picks) <= cfg.swing.top_n


def test_picks_are_ordered_by_score(cfg, world):
    populate(world, [f"SYM{i}" for i in range(20)])
    cfg.swing.max_per_sector = 99
    result = run(cfg, world)
    scores = [p.score for p in result.picks]
    assert scores == sorted(scores, reverse=True)


def test_one_sector_cannot_take_every_slot(cfg, world):
    """Three tickets in one sector is one bet, not three."""
    populate(world, [f"PHARMA{i}" for i in range(10)],
             sector="Healthcare", industry="Pharmaceuticals")
    result = run(cfg, world)
    sectors = [p.sector for p in result.picks]
    assert len(sectors) == len(set(sectors))
    assert any("already has" in r["reason"]
               for r in result.rejections_for(scanner.STAGE_SECTOR))


def test_a_second_sector_can_fill_the_remaining_slots(cfg, world):
    populate(world, [f"PHARMA{i}" for i in range(6)],
             sector="Healthcare", industry="Pharmaceuticals", seed_base=0)
    populate(world, [f"TECH{i}" for i in range(6)],
             sector="Information Technology",
             industry="Computers - Software & Consulting", seed_base=10)
    result = run(cfg, world)
    if len(result.picks) >= 2:
        assert len({p.sector for p in result.picks}) == len(result.picks)


# ---------------------------------------------------------------- scoring

def test_unavailable_news_redistributes_its_weight_rather_than_scoring_neutral(cfg):
    """
    Scoring an unreachable feed as 0.5 would award half the news weight to
    every stock at once - a ranking artefact wearing the costume of a signal.
    """
    from nifty_algo.swing.setup import SwingSetup

    found = SwingSetup("T", "breakout", "Level breakout", entry=100.0,
                       stop=95.0, target=115.0, trigger_note="", quality=0.6)
    metrics = {"rs_short": 0.05, "rs_long": 0.05, "pos_52w": 0.9,
               "volume_ratio": 1.5}

    with_news = news_mod.NewsResult("T", available=True, score=0.0)
    without = news_mod.NewsResult("T", available=False)

    _, parts_with = scanner._score(found, metrics, with_news, cfg)
    total_without, parts_without = scanner._score(found, metrics, without, cfg)

    assert parts_with["news"]["weight"] > 0
    assert parts_without["news"]["weight"] == 0
    assert parts_without["news"]["raw"] is None

    live = sum(p["weight"] for k, p in parts_without.items() if k != "news")
    assert live == pytest.approx(1.0), "redistributed weights must still sum to 1"
    assert 0.0 <= total_without <= 1.0


def test_every_component_is_reported_so_the_ranking_is_auditable(cfg, world):
    populate(world, [f"SYM{i}" for i in range(8)])
    result = run(cfg, world)
    for pick in result.picks:
        assert set(pick.score_parts) >= {"setup", "relative_strength",
                                         "reward_risk", "volume",
                                         "position_52w", "news"}
        assert pick.score == pytest.approx(
            sum(p["contribution"] for p in pick.score_parts.values()), abs=1e-3)


# ---------------------------------------------------------------- sizing

def test_quantity_comes_off_the_risk_budget_not_the_capital(cfg, world):
    populate(world, [f"SYM{i}" for i in range(8)])
    result = run(cfg, world)
    budget = cfg.capital.risk_per_trade_rupees

    for pick in result.picks:
        expected = int(budget // pick.setup.risk_points)
        if not pick.capital_note:
            assert pick.quantity == expected
            assert pick.risk_amount <= budget + 1e-6
            # ...and it is the LARGEST whole size that fits, not a timid one.
            assert (pick.quantity + 1) * pick.setup.risk_points > budget


def test_reward_follows_the_same_quantity(cfg, world):
    populate(world, [f"SYM{i}" for i in range(8)])
    for pick in run(cfg, world).picks:
        assert pick.reward_amount == pytest.approx(
            pick.quantity * pick.setup.reward_points)
        assert pick.deployed == pytest.approx(pick.quantity * pick.setup.entry)


def test_capital_caps_the_size_and_says_so(cfg, world):
    """A tight stop on an expensive share can ask for more than you have."""
    cfg.capital.swing_capital_inr = 4_000.0
    populate(world, [f"SYM{i}" for i in range(8)])
    result = run(cfg, world)
    for pick in result.picks:
        assert pick.deployed <= cfg.capital.swing_capital_inr + 1e-6
        if pick.capital_note:
            assert "Capped at" in pick.capital_note


def test_a_stop_wider_than_the_whole_budget_is_refused(cfg, world):
    cfg.capital.swing_capital_inr = 100.0       # risk/trade = Rs 1.67
    populate(world, [f"SYM{i}" for i in range(6)])
    result = run(cfg, world)
    assert result.picks == []
    assert result.rejections_for(scanner.STAGE_SIZING)


def test_the_combined_capital_note_flags_over_deployment(cfg, world):
    populate(world, [f"SYM{i}" for i in range(6)], seed_base=0)
    populate(world, [f"MED{i}" for i in range(6)], sector="Healthcare",
             industry="Pharmaceuticals", seed_base=10)
    populate(world, [f"MAT{i}" for i in range(6)], sector="Metals & Mining",
             industry="Iron & Steel", seed_base=20)
    result = run(cfg, world)
    if len(result.picks) == 3:
        assert "All 3 together" in result.capital_note
        total = sum(p.deployed for p in result.picks)
        if total > cfg.capital.starting_capital:
            # "your capital" became ambiguous once there are two pools, so the
            # note names the pool and its size instead.
            assert "more than the" in result.capital_note
            assert "without margin" in result.capital_note


# ---------------------------------------------------------------- records

def test_picks_serialise_to_something_the_journal_can_hold(cfg, world):
    import json

    populate(world, [f"SYM{i}" for i in range(8)])
    for pick in run(cfg, world).picks:
        record = pick.to_record()
        json.dumps(record)                      # must not raise
        assert record["symbol"] == pick.symbol
        assert record["entry"] > record["stop"]
        assert record["target"] > record["entry"]
        assert record["valid_until"] > record["scanned_on"]


def test_the_case_for_a_pick_is_always_stated(cfg, world):
    populate(world, [f"SYM{i}" for i in range(8)])
    for pick in run(cfg, world).picks:
        reasons = pick.why()
        assert len(reasons) >= 3
        assert any("Halal screen" in r for r in reasons)


def test_an_all_excluded_universe_warns_rather_than_returning_silence(cfg, world):
    world["stocks"].append(
        make_stock("BANKA", "Financial Services", "Private Sector Bank"))
    result = run(cfg, world)
    assert result.picks == []
    assert any("halal screen" in w for w in result.warnings)


def test_skipping_news_marks_it_unavailable_rather_than_neutral(cfg, world):
    populate(world, [f"SYM{i}" for i in range(8)])
    result = run(cfg, world, skip_news=True)
    for pick in result.picks:
        assert pick.news.available is False
        assert pick.score_parts["news"]["weight"] == 0
