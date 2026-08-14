"""
The Kite adapters, and the moment the liquidity gates start biting.

No live API calls. Everything here runs against recorded response shapes, so
the suite passes with no credentials, no subscription and no network.
"""
from datetime import date

import pandas as pd
import pytest

from nifty_algo.config import Config
from nifty_algo.data.chain import ChainProvider, next_weekly_expiry
from nifty_algo.data.kite_feed import KiteFeed
from nifty_algo.pricing import bs_price, implied_vol, synthetic_chain
from nifty_algo.risk import ApprovedOrder, OptionQuote, RiskEngine


# ---------------------------------------------------------------- feed parsing

def _candles():
    """The shape kiteconnect's historical_data() actually returns."""
    idx = pd.date_range("2026-08-07 09:15", periods=4, freq="5min",
                        tz="Asia/Kolkata")
    return [{"date": t, "open": 25800.0 + i, "high": 25815.0 + i,
             "low": 25795.0 + i, "close": 25810.0 + i, "volume": 0}
            for i, t in enumerate(idx)]


def test_kite_candles_become_tz_naive_ist():
    """
    The whole system reasons in IST wall-clock - session windows are
    time(9, 15) and time(15, 10). A UTC index would silently shift every
    session boundary by five and a half hours.
    """
    df = KiteFeed._to_frame(_candles())
    assert df.index.tz is None
    assert df.index[0].hour == 9 and df.index[0].minute == 15
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_kite_frame_passes_the_datafeed_contract():
    df = KiteFeed._to_frame(_candles())
    cleaned = KiteFeed.validate(df)
    assert len(cleaned) == 4
    assert cleaned["close"].iloc[-1] == pytest.approx(25813.0)


def test_index_candles_have_no_volume():
    """
    Not a defect - an index has nothing trading in it. It matters because the
    participation gates have to notice and substitute a proxy rather than
    silently passing on every bar.
    """
    from nifty_algo.signals import has_traded_volume
    assert not has_traded_volume(KiteFeed._to_frame(_candles()))


def test_unknown_interval_is_refused_up_front():
    from nifty_algo.data.base import FeedError
    with pytest.raises(FeedError):
        KiteFeed(interval_minutes=7)


# ---------------------------------------------------------------- implied vol

def test_implied_vol_round_trips_a_black_scholes_price():
    spot, strike, t, r = 25_800.0, 25_850.0, 5 / 365, 0.065
    price = bs_price(spot, strike, t, 0.147, r, "CE")
    assert implied_vol(price, spot, strike, t, r, "CE") == pytest.approx(0.147, rel=1e-3)


def test_implied_vol_refuses_a_premium_below_intrinsic():
    """
    No volatility prices an option under its own intrinsic value. Such a quote
    is stale or crossed, and callers must drop the strike rather than
    substitute a guessed IV - a guessed IV produces a wrong delta, and a wrong
    delta buys the wrong strike.
    """
    assert implied_vol(50.0, 26_000.0, 25_000.0, 5 / 365, 0.065, "CE") is None


def test_implied_vol_refuses_nonsense_inputs():
    assert implied_vol(0.0, 25_800.0, 25_800.0, 5 / 365, 0.065, "CE") is None
    assert implied_vol(100.0, 25_800.0, 25_800.0, 0.0, 0.065, "CE") is None


# ---------------------------------------------------------------- the gates

def _quote(**kw):
    base = dict(strike=25_850, option_type="CE", premium=100.0, delta=0.40,
                bid=99.8, ask=100.2, open_interest=500_000)
    base.update(kw)
    return OptionQuote(**base)


def test_a_wide_spread_is_rejected():
    """
    THE TEST THAT PROVES THE GATES NOW BITE.

    On the synthetic chain the spread is fabricated at 0.4% and the OI at
    500,000 - numbers invented to pass - so select_strike() has never once
    rejected a contract for illiquidity. Against a real chain it must.
    """
    risk = RiskEngine(Config())
    tight = _quote(bid=99.8, ask=100.2)          # 0.4%
    wide = _quote(bid=97.0, ask=103.0)           # 6%

    assert risk.select_strike([tight], 30.0, "CE", lots=2) is tight
    assert risk.select_strike([wide], 30.0, "CE", lots=2) is None
    assert any("spread" in g for g in risk.gate_failures(wide, 30.0, lots=2))


def test_thin_open_interest_is_rejected():
    risk = RiskEngine(Config())
    thin = _quote(open_interest=4_000)
    assert risk.select_strike([thin], 30.0, "CE", lots=2) is None
    assert any("OI" in g for g in risk.gate_failures(thin, 30.0, lots=2))


def test_doubling_the_size_halves_the_delta_ceiling():
    """
    Risk per trade is fixed, so size does not scale risk - it scales the delta
    you may buy. This is the constraint that decides whether the runner is
    even possible on a given day.
    """
    risk = RiskEngine(Config())
    one = risk.required_max_delta(30.0, lots=1)
    two = risk.required_max_delta(30.0, lots=2)
    assert two == pytest.approx(one / 2, rel=1e-9)


@pytest.mark.parametrize("stop_points", [10, 25, 30, 50, 63, 64, 65, 80, 150])
def test_the_chain_free_size_is_an_upper_bound_never_an_underestimate(stop_points):
    """
    `lots_for()` is what the backtester and the daily brief use when they have
    no chain to consult; `approve()` is what the live engine uses when it does.

    They may differ, in ONE direction only. `lots_for()` checks that the delta
    ceiling clears min_delta; it cannot know whether a strike exists in that
    band, and strikes are 50 points apart. Near a 63-64 point stop the two-lot
    band is about 0.003 wide and can be empty, so approve() sizes down.

    The direction is what matters: approve() must never size UP past the shared
    prediction, or the backtest is understating the position it would have held.
    """
    cfg = Config()
    risk = RiskEngine(cfg)
    chain = synthetic_chain(25_800, 5 / 365, "CE", cfg)

    predicted = risk.lots_for(float(stop_points))
    decision = risk.approve(chain, "CE", float(stop_points), risk.capital)

    if isinstance(decision, ApprovedOrder):
        assert decision.lots <= predicted


def test_the_two_sizings_agree_away_from_the_boundary():
    """The gap must stay confined to the narrow band, not be everywhere."""
    cfg = Config()
    chain = synthetic_chain(25_800, 5 / 365, "CE", cfg)
    for stop_points in (20, 25, 30, 35, 40, 45, 50, 90, 120):
        risk = RiskEngine(cfg)
        decision = risk.approve(chain, "CE", float(stop_points), risk.capital)
        if isinstance(decision, ApprovedOrder):
            assert decision.lots == risk.lots_for(float(stop_points)), stop_points


def test_wide_stops_force_the_one_lot_fallback():
    """
    At a 70-point stop the two-lot delta ceiling drops under min_delta and no
    strike survives, so approve() must fall back to one lot - and say so.
    """
    cfg = Config()
    risk = RiskEngine(cfg)
    chain = synthetic_chain(25_800, 5 / 365, "CE", cfg)

    narrow = risk.approve(chain, "CE", 25.0, risk.capital)
    assert narrow.lots == 2 and narrow.runner_enabled

    risk2 = RiskEngine(cfg)
    wide = risk2.approve(chain, "CE", 80.0, risk2.capital)
    assert wide.lots == 1
    assert not wide.runner_enabled
    assert "RUNNER DISABLED" in wide.sizing_note


# ---------------------------------------------------------------- provider

class _FakeChain:
    """Stands in for KiteChain."""

    def __init__(self, quotes=None, expiry=date(2026, 8, 11), fail=False):
        self._quotes = quotes if quotes is not None else [_quote()]
        self._expiry = expiry
        self._fail = fail

    def nearest_expiry(self, today=None):
        return self._expiry

    def get_chain(self, spot, option_type, expiry=None):
        if self._fail:
            raise RuntimeError("kite is down")
        return self._quotes, self._expiry


def test_broker_chain_is_preferred_and_labelled_real():
    provider = ChainProvider(Config(), broker_chain=_FakeChain())
    result = provider.get_chain(25_800, "CE", date(2026, 8, 7))
    assert result.source == "broker"
    assert not result.is_synthetic
    assert result.note == ""


def test_expiry_comes_from_the_dump_not_the_weekday_guess():
    """
    next_weekly_expiry() hardcodes Tuesday and its own docstring flags that the
    day has moved twice. A wrong expiry means a wrong time-to-expiry means a
    wrong delta - silently. The dump is the authority when it is there.
    """
    real = date(2026, 8, 13)                      # deliberately not a Tuesday
    provider = ChainProvider(Config(), broker_chain=_FakeChain(expiry=real))
    assert provider.resolve_expiry(date(2026, 8, 7)) == real

    bare = ChainProvider(Config())
    assert bare.resolve_expiry(date(2026, 8, 7)) == next_weekly_expiry(date(2026, 8, 7))


def test_a_broker_failure_falls_back_but_says_why():
    """
    Downgrading from real quotes to fabricated ones without telling you is
    worse than stopping. The reason has to travel with the chain.
    """
    provider = ChainProvider(Config(), broker_chain=_FakeChain(fail=True))
    result = provider.get_chain(25_800, "CE", date(2026, 8, 7))
    assert result.is_synthetic
    assert "kite is down" in result.note
    assert "SYNTHETIC" in result.note


def test_empty_broker_quotes_fall_back_rather_than_alerting_on_nothing():
    provider = ChainProvider(Config(), broker_chain=_FakeChain(quotes=[]))
    result = provider.get_chain(25_800, "CE", date(2026, 8, 7))
    assert result.is_synthetic
    assert "no usable quotes" in result.note
