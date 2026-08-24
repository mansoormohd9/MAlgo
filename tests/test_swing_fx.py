"""
FX, and the sizing bug it exists to prevent.

The bug: `quantity = risk_budget / stop_distance`. The budget is in rupees
because that is where the governors live; a US stop distance is in dollars.
Divide one by the other and you get a share count roughly 88x too large, on a
ticket that looks entirely ordinary.

The contract: this module FAILS CLOSED. It raises rather than guessing, and
the scan stands the whole market down rather than sizing off a rate it does
not trust. Standing down for a day costs nothing measurable. Sizing every
ticket off a stale rate costs money and announces nothing.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from nifty_algo.config import Config
from nifty_algo.swing import fx as fx_mod
from nifty_algo.swing import markets as markets_mod


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config()
    c.swing.cache_dir = str(tmp_path)
    return c


# ---------------------------------------------------------------- the home rate

def test_the_home_currency_is_exactly_one_and_never_fetched(cfg, monkeypatch):
    def explode(pair):
        raise AssertionError("INR must never cost a network call")

    monkeypatch.setattr(fx_mod, "_download", explode)
    rate = fx_mod.rate_inr_per("INR", cfg)
    assert rate.inr_per_unit == 1.0


def test_an_inr_market_never_touches_fx(cfg, monkeypatch):
    monkeypatch.setattr(fx_mod, "_download",
                        lambda pair: pytest.fail("no fetch for a home market"))
    india = markets_mod.get(cfg, markets_mod.INDIA)
    assert fx_mod.rate_for_market(india, cfg).inr_per_unit == 1.0


# ---------------------------------------------------------------- failing closed

def test_an_unfetchable_rate_raises_rather_than_guessing(cfg, monkeypatch):
    monkeypatch.setattr(fx_mod, "_download", lambda pair: None)
    with pytest.raises(fx_mod.FxUnavailable):
        fx_mod.rate_inr_per("USD", cfg)


def test_there_is_no_hardcoded_fallback_rate(cfg, monkeypatch):
    """
    A default of 83 would be invisible and wrong every day after the one it
    was written on.
    """
    monkeypatch.setattr(fx_mod, "_download", lambda pair: None)
    try:
        fx_mod.rate_inr_per("USD", cfg)
    except fx_mod.FxUnavailable as e:
        assert "stands down" in str(e)
    else:
        pytest.fail("a missing rate must not produce a rate")


@pytest.mark.parametrize("bad", [0.0, -80.0])
def test_a_non_positive_quote_is_treated_as_absent(cfg, monkeypatch, bad):
    monkeypatch.setattr(fx_mod, "_download", lambda pair: bad)
    with pytest.raises(fx_mod.FxUnavailable):
        fx_mod.rate_inr_per("USD", cfg)


@pytest.mark.parametrize("absurd", [3.5, 9_000.0])
def test_an_implausible_quote_is_refused(cfg, monkeypatch, absurd):
    """
    An inverted or partial quote is a bad number, not a market move, and it
    would size a position off by orders of magnitude.
    """
    monkeypatch.setattr(fx_mod, "_download", lambda pair: absurd)
    with pytest.raises(fx_mod.FxUnavailable) as e:
        fx_mod.rate_inr_per("USD", cfg)
    assert "plausible band" in str(e.value)


def test_an_unregistered_currency_raises(cfg):
    with pytest.raises(fx_mod.FxUnavailable):
        fx_mod.rate_inr_per("JPY", cfg)


# ---------------------------------------------------------------- caching

def test_a_fresh_cached_rate_is_reused(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(fx_mod, "_download",
                        lambda pair: calls.append(pair) or 88.0)

    first = fx_mod.rate_inr_per("USD", cfg)
    second = fx_mod.rate_inr_per("USD", cfg)

    assert len(calls) == 1
    assert second.inr_per_unit == first.inr_per_unit
    assert second.from_cache


def test_a_stale_cached_rate_is_not_reused(cfg, monkeypatch, tmp_path):
    stale = datetime.now() - timedelta(hours=cfg.swing.price_cache_hours + 5)
    (tmp_path / fx_mod.CACHE_NAME).write_text(json.dumps({
        "USD": {"inr_per_unit": 70.0, "fetched_at": stale.isoformat()}
    }), encoding="utf-8")

    monkeypatch.setattr(fx_mod, "_download", lambda pair: 88.0)
    assert fx_mod.rate_inr_per("USD", cfg).inr_per_unit == pytest.approx(88.0)


def test_a_stale_cache_with_no_network_is_refused_not_used(cfg, tmp_path,
                                                          monkeypatch):
    """
    The tempting fallback - "use the old rate, it is close enough" - is
    exactly the behaviour this module refuses. A stale rate is not an
    approximation, it is a position-size error.
    """
    stale = datetime.now() - timedelta(days=9)
    (tmp_path / fx_mod.CACHE_NAME).write_text(json.dumps({
        "USD": {"inr_per_unit": 70.0, "fetched_at": stale.isoformat()}
    }), encoding="utf-8")
    monkeypatch.setattr(fx_mod, "_download", lambda pair: None)

    with pytest.raises(fx_mod.FxUnavailable):
        fx_mod.rate_inr_per("USD", cfg)


def test_a_corrupt_cache_costs_a_refetch_not_the_scan(cfg, tmp_path, monkeypatch):
    (tmp_path / fx_mod.CACHE_NAME).write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(fx_mod, "_download", lambda pair: 88.0)
    assert fx_mod.rate_inr_per("USD", cfg).inr_per_unit == pytest.approx(88.0)


# ---------------------------------------------------------------- the pools

def test_the_foreign_pool_uses_the_same_formula_as_the_home_one():
    """
    Two pools, one governor. If these ever diverge, the two books have started
    sizing off different rules.
    """
    c = Config()
    c.capital.foreign_capital_inr = 800_000.0
    assert c.capital.foreign_risk_per_trade_inr == pytest.approx(
        800_000.0 * c.capital.risk_per_trade_pct)
    assert c.capital.risk_inr("home") == pytest.approx(
        c.capital.risk_per_trade_rupees)
    assert c.capital.risk_inr("foreign") == pytest.approx(
        c.capital.foreign_risk_per_trade_inr)


def test_an_unfunded_foreign_pool_is_zero():
    c = Config()
    assert c.capital.risk_inr("foreign") == 0.0
