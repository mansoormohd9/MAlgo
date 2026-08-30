"""
The research fact packs, and the rule that makes them safe to write prose from.

THE RULE: a fact this build could not establish is carried as
`available=False` WITH A REASON, and never as a blank or a zero. Roughly a
third of what these briefings ask for does not exist for Indian equities on
any free source - India publishes no short interest at all, insider dealing is
a SEBI filing rather than an API, and the CPI/GDP prints have no free machine
endpoint. Rendered as 0.0 or as an empty cell, every one of those reads as
good news, and a briefing would be written on it.

Nothing here touches the network: the macro series and the price cache are
both injected.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.portfolio.aggregate import PortfolioSnapshot
from nifty_algo.portfolio.base import ConnectorResult, Position
from nifty_algo.research import exposure, macro, risk_report
from nifty_algo.research.base import Fact, FactPack, Section
from nifty_algo.research.providers import macro_series as ms


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.swing.cache_dir = str(tmp_path)
    cfg.portfolio.manual_path = str(tmp_path / "manual.csv")
    cfg.portfolio.connectors = ("manual",)
    cfg.portfolio.min_correlation_sessions = 10
    return cfg


def _position(symbol, value, market="india", ccy="INR", asset_class="equity"):
    return Position(key=f"{market}:{symbol}", symbol=symbol, market=market,
                    quantity=1.0, average_price=value, last_price=value,
                    currency=ccy, asset_class=asset_class, source="manual")


def _snapshot(positions, complete=True):
    s = PortfolioSnapshot(
        positions=positions,
        results=[ConnectorResult.ok("manual", positions)],
        value_inr={p.key: p.value_native for p in positions},
        generated_at=datetime.now())
    if not complete:
        s.results.append(ConnectorResult.unavailable("kite", "token expired"))
    return s


def _bars(days=300, seed=1, start=100.0):
    """A deterministic random walk with OHLCV, indexed by business day."""
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=days)
    close = start * (1 + rng.normal(0, 0.01, days)).cumprod()
    return pd.DataFrame({"open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1_000_000.0}, index=idx)


# ---------------------------------------------------------------- the contract

def test_an_unavailable_fact_carries_no_value_and_demands_a_reason():
    """
    The whole package rests on this. A `Fact` we do not have must be
    impossible to accidentally do arithmetic with, and must never be
    constructible without saying why it is missing.
    """
    with pytest.raises(TypeError):
        Fact.unknown("India short interest")          # no reason given

    fact = Fact.unknown("India short interest",
                        "India publishes no short-interest disclosure")
    assert fact.available is False
    assert fact.value is None
    assert "no short-interest" in fact.display()


def test_a_fact_defaults_to_unavailable():
    """`available` is False unless a constructor deliberately sets it, so a
    hand-built Fact cannot slip through as verified."""
    assert Fact(label="x").available is False


def test_the_pack_can_list_everything_it_could_not_establish():
    pack = FactPack(report="t")
    s = pack.section("a")
    s.add(Fact.known("here", 1.0, "src"))
    s.add(Fact.unknown("missing", "no free endpoint"))
    s.judgment = ["what the Fed does next"]

    missing = pack.unavailable()
    assert [m["label"] for m in missing] == ["missing"]
    assert missing[0]["section"] == "a"
    assert pack.judgment_required() == [
        {"section": "a", "question": "what the Fed does next"}]


def test_the_pack_serialises_with_both_lists_at_the_top_level():
    """The skill checks its two hard rules off these lists rather than by
    walking the tree, so they have to be there."""
    import json
    pack = FactPack(report="t")
    pack.section("a").add(Fact.unknown("m", "why"))
    payload = json.loads(pack.to_json())
    assert "unavailable" in payload and "judgment_required" in payload
    assert payload["sections"][0]["facts"][0]["available"] is False


# ---------------------------------------------------------------- units

def test_a_yield_moves_in_basis_points_and_an_index_in_percent():
    """
    4.0 -> 4.4 is 40 basis points, not 10%; 24000 -> 26400 is 10%, not 2400
    points. Either quoted in the other's unit is dimensionally plausible and
    completely wrong.
    """
    rate = ms.Series("r", "r", "^TNX", ms.RATE, "%", "")
    level = ms.Series("l", "l", "^NSEI", ms.LEVEL, "index", "")
    assert ms._change(rate, 4.0, 4.4) == pytest.approx(40.0)
    assert ms._change(level, 24000.0, 26400.0) == pytest.approx(10.0)


def test_factor_moves_differences_a_rate_and_percents_a_level():
    frame = pd.DataFrame({"us_10y": [4.0, 4.1, 4.2],
                          "nifty": [100.0, 110.0, 121.0]})
    moves = ms.factor_moves(frame)
    assert moves["us_10y"].iloc[-1] == pytest.approx(0.1)      # absolute
    assert moves["nifty"].iloc[-1] == pytest.approx(0.1)       # fractional


def test_a_manual_series_is_unavailable_with_the_reason_not_zero(tmp_path):
    """India's CPI has no free endpoint. Reported as 0.0 it would read as
    'no inflation', which a briefing would build a rotation on."""
    readings = ms.load_manual(_cfg(tmp_path))
    cpi = readings["india_cpi_yoy"]
    assert cpi.available is False
    assert cpi.last is None
    assert "free machine endpoint" in cpi.note


# ---------------------------------------------------------------- exposure

def test_a_short_overlap_returns_no_correlation_rather_than_a_confident_one():
    """Two names that share twelve sessions will happily produce a 0.9."""
    cfg = Config()
    cfg.portfolio.min_correlation_sessions = 60
    returns = pd.DataFrame({"A": _bars(20)["close"].pct_change().dropna()})
    moves = pd.DataFrame({"nifty": _bars(20, seed=2)["close"].pct_change()
                          .dropna()})
    out = exposure.sensitivities("A", returns, moves, cfg)
    assert out["corr_nifty"] is None
    assert out["nifty_beta"] is None
    assert "overlapping sessions" in out["sensitivity_note"]


def test_a_symbol_with_no_bars_reports_why_rather_than_vanishing():
    out = exposure.sensitivities("MISSING", pd.DataFrame(), pd.DataFrame(),
                                 Config())
    assert out["nifty_beta"] is None
    assert out["sensitivity_note"] == "no daily bars for this symbol"


def test_pairwise_correlation_carries_its_sample_size():
    cfg = Config()
    cfg.portfolio.min_correlation_sessions = 10
    a = _bars(120, seed=1)["close"].pct_change().dropna()
    returns = pd.DataFrame({"A": a, "B": _bars(120, seed=2)["close"]
                            .pct_change().dropna()})
    rows = exposure.pairwise(returns, cfg)
    assert len(rows) == 1
    assert rows[0]["sessions"] > 100
    assert -1.0 <= rows[0]["correlation"] <= 1.0


def test_a_perfectly_correlated_pair_is_found():
    cfg = Config()
    cfg.portfolio.min_correlation_sessions = 10
    a = _bars(120)["close"].pct_change().dropna()
    rows = exposure.pairwise(pd.DataFrame({"A": a, "B": a * 2.0}), cfg)
    assert rows[0]["correlation"] == pytest.approx(1.0)


def test_portfolio_returns_normalise_the_weights_they_can_price():
    """A position with no bars must not silently shrink the whole book's
    return towards zero - the covered names are renormalised instead."""
    a = pd.Series([0.10, 0.10], index=pd.bdate_range("2024-01-01", periods=2))
    returns = pd.DataFrame({"A": a})
    series = exposure.portfolio_returns(returns, {"A": 50.0, "NOBARS": 50.0})
    assert series.iloc[0] == pytest.approx(0.10)


# ---------------------------------------------------------------- drawdowns

def test_a_drawdown_episode_is_found_rather_than_hardcoded():
    """A list of dates in the source is wrong the moment the cache covers a
    different span, and would replay nothing without saying so."""
    idx = pd.bdate_range("2024-01-01", periods=200)
    close = pd.Series([100.0] * 50 + list(range(100, 60, -1))
                      + [60.0 + i for i in range(110)], index=idx)[:200]
    found = risk_report.episodes(pd.DataFrame({"close": close}),
                                 min_drawdown=0.10, limit=3)
    assert found
    assert found[0]["benchmark_drawdown_pct"] < -30
    assert found[0]["trough"] > found[0]["from"]


def test_a_flat_history_produces_no_episodes_rather_than_a_fake_one():
    idx = pd.bdate_range("2024-01-01", periods=200)
    flat = pd.DataFrame({"close": pd.Series([100.0] * 200, index=idx)})
    assert risk_report.episodes(flat) == []


def test_a_short_history_cannot_manufacture_an_episode():
    idx = pd.bdate_range("2024-01-01", periods=30)
    frame = pd.DataFrame({"close": pd.Series(range(30, 0, -1), index=idx,
                                             dtype=float)})
    assert risk_report.episodes(frame) == []


# ------------------------------------------------- incomplete books, end to end

def _no_network(monkeypatch, cfg):
    """Every external read stubbed. These packs must build with no network."""
    monkeypatch.setattr(ms, "load", lambda c, force_refresh=False, series=None:
                        {s.key: ms.Reading(s, available=False,
                                           note="stubbed in tests")
                         for s in ms.SERIES})
    monkeypatch.setattr(ms, "history", lambda c, force_refresh=False: None)
    from nifty_algo.research import holdings_prices
    monkeypatch.setattr(holdings_prices, "bars_for",
                        lambda *a, **kw: ({}, None, "stubbed"))


def test_the_risk_report_withholds_percentages_on_a_partial_book(tmp_path,
                                                                 monkeypatch):
    """
    The single most dangerous number in this package. A concentration figure
    computed against a denominator we could not establish reads exactly like
    one that was, and it would be acted on.
    """
    cfg = _cfg(tmp_path)
    _no_network(monkeypatch, cfg)
    snapshot = _snapshot([_position("INFY", 1000.0),
                          _position("TCS", 3000.0)], complete=False)

    pack = risk_report.build(cfg, snapshot=snapshot)
    concentration = _section(pack, "Concentration")
    assert all(r["weight_pct"] is None for r in concentration.rows)
    assert any(f.label == "Largest single position" and not f.available
               for f in concentration.facts)
    # ...and the absolute values are still there, because they are facts.
    assert {r["value_inr"] for r in concentration.rows} == {1000.0, 3000.0}


def test_the_risk_report_quotes_percentages_once_the_book_is_whole(tmp_path,
                                                                  monkeypatch):
    cfg = _cfg(tmp_path)
    _no_network(monkeypatch, cfg)
    snapshot = _snapshot([_position("INFY", 1000.0),
                          _position("TCS", 3000.0)])

    concentration = _section(risk_report.build(cfg, snapshot=snapshot),
                             "Concentration")
    weights = {r["symbol"]: r["weight_pct"] for r in concentration.rows}
    assert weights == {"TCS": 75.0, "INFY": 25.0}


def test_an_empty_portfolio_stands_the_risk_report_down(tmp_path, monkeypatch):
    """
    "No positions" is not a low-risk result, it is an absent one - and after
    a failed broker read it is also a wrong one.
    """
    cfg = _cfg(tmp_path)
    _no_network(monkeypatch, cfg)
    pack = risk_report.build(cfg, snapshot=_snapshot([]))
    assert pack.stood_down
    assert "not a low-risk result" in pack.stood_down


def test_the_macro_pack_builds_with_every_series_unavailable(tmp_path,
                                                             monkeypatch):
    """An outage must produce a pack full of stated absences, not a crash and
    not a pack of zeroes."""
    cfg = _cfg(tmp_path)
    _no_network(monkeypatch, cfg)
    pack = macro.build(cfg, snapshot=_snapshot([_position("INFY", 1000.0)]))

    assert pack.sections
    assert len(pack.unavailable()) >= len(ms.SERIES)
    assert all(f["value"] is None for f in pack.unavailable())
    assert pack.judgment_required()


def test_the_macro_pack_names_the_judgment_it_is_not_making(tmp_path,
                                                            monkeypatch):
    """Fed outlook, geopolitics and sector rotation have no series. They are
    declared as judgment IN THE DATA so the writer is told, not trusted."""
    cfg = _cfg(tmp_path)
    _no_network(monkeypatch, cfg)
    pack = macro.build(cfg, snapshot=_snapshot([_position("INFY", 1000.0)]))
    questions = " ".join(q["question"].lower()
                         for q in pack.judgment_required())
    for topic in ("policy", "geopolitics", "rotation"):
        assert topic in questions


def test_a_foreign_currency_share_is_withheld_on_a_partial_book(tmp_path,
                                                                monkeypatch):
    cfg = _cfg(tmp_path)
    _no_network(monkeypatch, cfg)
    snapshot = _snapshot([_position("INFY", 1000.0)], complete=False)
    section = _section(macro.build(cfg, snapshot=snapshot),
                       "Currency exposure")
    share = [f for f in section.facts
             if f.label == "Share of the book in foreign currency"][0]
    assert share.available is False


def _section(pack, name) -> Section:
    for s in pack.sections:
        if s.name == name:
            return s
    raise AssertionError(f"{name} not in {[s.name for s in pack.sections]}")


def test_a_symbol_with_no_bar_that_day_does_not_damp_the_books_return():
    """
    THE bug this function had. Filling a missing session with a 0% return at
    full weight drags that day toward zero by exactly the missing weight - so
    a holding listed partway through the window quietly understates the
    volatility, the percentiles and the worst session, in the reassuring
    direction, with no error anywhere.
    """
    idx = pd.bdate_range("2024-01-01", periods=3)
    frame = pd.DataFrame({"OLD": [0.10, 0.10, 0.10],
                          "NEW": [float("nan"), float("nan"), 0.10]},
                         index=idx)
    series = exposure.portfolio_returns(frame, {"OLD": 50.0, "NEW": 50.0})

    # Day 1 is OLD alone at +10%, not half of it.
    assert series.iloc[0] == pytest.approx(0.10)
    assert series.iloc[-1] == pytest.approx(0.10)


def test_a_session_with_no_bars_at_all_is_dropped_not_scored_zero():
    idx = pd.bdate_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"A": [float("nan"), 0.05]}, index=idx)
    series = exposure.portfolio_returns(frame, {"A": 100.0})
    assert len(series) == 1
    assert series.iloc[0] == pytest.approx(0.05)


def test_holding_nothing_in_the_requested_market_is_stated_not_a_crash(
        tmp_path, monkeypatch):
    """
    `--market us` on a purely Indian book. Every market-scoped section would
    otherwise run over an empty list, and the concentration one raised on
    `rows[0]`. An empty MARKET is not an empty account, and only one of those
    is a low-risk result.
    """
    cfg = _cfg(tmp_path)
    _no_network(monkeypatch, cfg)
    snapshot = _snapshot([_position("INFY", 1000.0, market="india")])

    pack = risk_report.build(cfg, market_key="us", snapshot=snapshot)
    fact = _section(pack, "Concentration").facts[0]
    assert fact.available is False
    assert "india" in fact.note
    assert "empty MARKET, not an empty account" in fact.note
