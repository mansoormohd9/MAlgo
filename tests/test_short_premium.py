"""
The short-premium book: the sign, the stops, the gates, the pot.

Every test here corresponds to a way this book fails PLAUSIBLY rather than
loudly - a winning trade reported as a loss, a gap loss capped at 1R, a
fabricated chain passing a liquidity gate. None of them raise an exception in
the broken version; all of them produce a number you would act on.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from nifty_algo.config import CapitalConfig, Config
from nifty_algo.positions import ExitKind
from nifty_algo.pricing import synthetic_chain
from nifty_algo.risk import OptionQuote
from nifty_algo.short_premium import costs_sell, strikes
from nifty_algo.short_premium.position import (
    STOP_SHORT_STRIKE, STOP_UNDERLYING, ShortPosition, underlying_stop_for,
)


@pytest.fixture
def cfg():
    c = Config()
    c.capital.short_premium_capital_inr = 500_000.0
    return c


def _quote(strike=26300, option_type="CE", premium=30.0, delta=0.15,
           oi=400_000, volume=50_000, iv=0.14, spread=0.01):
    half = premium * spread / 2.0
    return OptionQuote(strike=strike, option_type=option_type, premium=premium,
                       delta=delta, bid=round(premium - half, 2),
                       ask=round(premium + half, 2), open_interest=oi,
                       iv=iv, volume=volume)


def _position(cfg, credit=30.0, lots=2, spot=26000.0, strike=26300,
              option_type="CE", atr=100.0):
    q = _quote(strike=strike, option_type=option_type, premium=credit)
    return ShortPosition(
        quote=q, strategy_key="t", lots=lots, lot_size=65,
        entry_credit=credit, entry_spot=spot,
        underlying_stop=underlying_stop_for(option_type, strike, atr, cfg),
        cfg=cfg,
    )


# ------------------------------------------------------------------ the sign

def test_a_winning_short_reports_positive_r(cfg):
    """
    THE REGRESSION FOR THE SIGN LOCK.

    `positions.ManagedPosition.r_of` is `(premium - entry) / per_r`. Reuse it
    for a short and a premium falling from 30 to 20 - the seller's whole
    objective - reports -0.33R and stops out immediately.
    """
    pos = _position(cfg, credit=30.0)
    assert pos.r_of(20.0) > 0, "premium falling must be profit for a seller"
    assert pos.r_of(20.0) == pytest.approx(1.0 / 3.0)
    assert pos.r_of(30.0) == pytest.approx(0.0)
    assert pos.r_of(60.0) == pytest.approx(-1.0), "2x credit is exactly -1R"
    assert pos.pnl_for(1, 20.0) > 0
    assert pos.pnl_for(1, 60.0) < 0


def test_max_achievable_r_is_bounded_by_the_credit(cfg):
    """
    A seller's profit is capped, and the cap is what forces the ladder to be
    re-parameterised. At the default 2x stop the best possible outcome is
    +1R, so the inherited +2R partial rung could never fire.
    """
    pos = _position(cfg, credit=30.0)
    assert pos.max_r == pytest.approx(1.0)
    assert pos.r_of(0.0) == pytest.approx(pos.max_r)

    ladder = cfg.short_premium.ladder_settings()
    assert ladder.partial_exit_at_r < pos.max_r, (
        "the partial rung must be reachable, or the runner and trail are "
        "dead code"
    )
    assert Config().trade.partial_exit_at_r > pos.max_r, (
        "the option book's 2R rung is unreachable here - this is the bug "
        "this configuration exists to prevent"
    )


# ------------------------------------------------------------------ the stops

def test_a_gap_through_the_stop_fills_at_the_open_and_exceeds_one_r(cfg):
    """
    The tail that ends short-premium accounts.

    `positions._to_action` fills a stop at the LEVEL. Here the bar OPENS at
    150 - five times the credit, straight through the 60 the stop sat at -
    and the fill must be 150. Capping it at 60 reports -1R for a -4R day.
    """
    pos = _position(cfg, credit=30.0, lots=2)
    actions = pos.advance(mark_premium=155.0, spot=26010.0,
                          bar_open_premium=150.0,
                          bar_high_premium=160.0, bar_low_premium=148.0)
    assert len(actions) == 1
    a = actions[0]
    assert a.kind is ExitKind.STOPPED_OUT
    assert a.exit_premium == pytest.approx(150.0), "fills at the gap open"
    assert a.realised_r == pytest.approx(-4.0)
    assert a.realised_r < -1.0, "a seller's loss is not capped at 1R"
    assert a.gross_pnl == pytest.approx((30.0 - 150.0) * 65 * 2)


def test_a_stop_merely_touched_fills_at_the_level_not_the_bars_worst_tick(cfg):
    """
    THE OVER-PESSIMISM REGRESSION.

    A bar that wicks through the stop and recovers does not fill you at its
    worst print - you had a stop order, and it triggered at the stop. Pricing
    these at the bar high overstates every trailing exit in the book, which
    is the same class of error as understating the gaps, pointing the other
    way.
    """
    pos = _position(cfg, credit=30.0, lots=2)
    actions = pos.advance(mark_premium=45.0, spot=26010.0,
                          bar_open_premium=40.0,
                          bar_high_premium=95.0, bar_low_premium=38.0)
    a = actions[0]
    assert a.kind is ExitKind.STOPPED_OUT
    assert a.exit_premium == pytest.approx(60.0), "the stop level, not the 95 high"
    assert a.realised_r == pytest.approx(-1.0)


def test_omitting_the_bar_open_prices_every_stop_as_a_gap(cfg):
    """The conservative default: no open supplied means no claim it did not gap."""
    pos = _position(cfg, credit=30.0, lots=2)
    a = pos.advance(mark_premium=95.0, spot=26010.0, bar_high_premium=95.0)[0]
    assert a.exit_premium == pytest.approx(95.0)


def test_the_underlying_stop_fires_before_the_premium_stop(cfg):
    """
    A premium stop alone is a vega stop. Spot has run to the short strike
    while the premium has barely moved - the position is in real trouble and
    the ladder, which sees only R, has no idea.
    """
    pos = _position(cfg, credit=30.0, spot=26000.0, strike=26300, atr=100.0)
    actions = pos.advance(mark_premium=34.0, spot=26300.0)
    assert len(actions) == 1
    assert actions[0].kind is ExitKind.STOPPED_OUT
    assert STOP_SHORT_STRIKE in actions[0].detail
    assert pos.state.closed


def test_the_underlying_stop_is_anchored_to_the_strike_not_to_entry_spot(cfg):
    """
    THE DEAD-BRANCH REGRESSION.

    Anchored to entry spot at 1.5 ATR, the stop for a 300-point-OTM short
    call sat 150 points from spot - it fired on noise while the position was
    healthy, and because it sat INSIDE the strike the strike was never
    reached and the strike-touch branch could not execute. Anchored to the
    strike, the stop scales with the thing the risk scales with.
    """
    pos = _position(cfg, credit=30.0, spot=26000.0, strike=26300, atr=100.0)
    assert pos.underlying_stop == pytest.approx(26250.0)
    assert pos.underlying_stop < pos.short_strike, "the buffer is inside the strike"

    # healthy: 200 points of room left, nothing fires
    assert pos.underlying_breached(26100.0) is None
    assert pos.advance(mark_premium=28.0, spot=26100.0) == []
    assert not pos.state.closed

    # inside the buffer but not yet at the strike
    assert pos.underlying_breached(26260.0) == STOP_UNDERLYING
    # and at the strike, the other label for the same level
    assert pos.underlying_breached(26300.0) == STOP_SHORT_STRIKE


def test_the_underlying_stop_uses_the_bars_adverse_extreme(cfg):
    """
    Measured on closes only, the underlying stop understates how often it
    fires. The bar traded through the level and came back.
    """
    pos = _position(cfg, credit=30.0, spot=26000.0, strike=26300, atr=100.0)
    quiet = pos.advance(mark_premium=31.0, spot=26050.0)
    assert quiet == [] or not pos.state.closed

    pos2 = _position(cfg, credit=30.0, spot=26000.0, strike=26300, atr=100.0)
    fired = pos2.advance(mark_premium=31.0, spot=26050.0,
                         adverse_spot=26310.0)
    assert fired and fired[0].kind is ExitKind.STOPPED_OUT


def test_a_short_put_stops_on_the_other_side(cfg):
    """The direction that hurts is the opposite one. Easy to get backwards."""
    pos = _position(cfg, credit=30.0, spot=26000.0, strike=25700,
                    option_type="PE", atr=100.0)
    assert pos.underlying_stop == pytest.approx(25750.0)
    assert pos.underlying_stop > pos.short_strike, "the buffer is above a short put"
    assert pos.underlying_breached(26400.0) is None, "a rally helps a short put"
    assert pos.underlying_breached(25900.0) is None
    assert pos.underlying_breached(25740.0) == STOP_UNDERLYING
    assert pos.underlying_breached(25700.0) == STOP_SHORT_STRIKE


def test_the_ladder_advances_and_the_trail_never_loosens(cfg):
    """
    `ExitLadder` is reused unchanged - fed correctly-signed R it walks the
    seller's rungs. The trail ratchets and cannot give back.
    """
    pos = _position(cfg, credit=40.0, lots=2, spot=26000.0, strike=26600,
                    atr=100.0)
    # premium 40 -> 30 is +0.25R, the breakeven rung
    first = pos.advance(mark_premium=30.0, spot=26000.0)
    assert any(a.kind is ExitKind.STOP_TO_BREAKEVEN for a in first)
    assert pos.state.stop_r == pytest.approx(0.0)

    # -> 20 is +0.5R, the partial
    second = pos.advance(mark_premium=20.0, spot=26000.0)
    assert any(a.kind is ExitKind.PARTIAL_EXIT for a in second)
    assert pos.state.lots_remaining == 1

    before = pos.state.stop_r
    pos.advance(mark_premium=12.0, spot=26000.0)
    assert pos.state.stop_r >= before, "the trail must never loosen"

    pos.advance(mark_premium=25.0, spot=26000.0)
    assert pos.state.stop_r >= before


def test_force_exit_closes_at_the_mark(cfg):
    pos = _position(cfg, credit=30.0, lots=1)
    actions = pos.force_exit(mark_premium=18.0)
    assert len(actions) == 1
    assert actions[0].kind is ExitKind.FORCE_EXIT
    assert actions[0].exit_premium == pytest.approx(18.0)
    assert actions[0].gross_pnl > 0
    assert pos.state.closed


# ----------------------------------------------------------- strike selection

def test_the_buying_engines_band_and_this_one_do_not_overlap(cfg):
    """
    The clearest statement that these are two engines: no delta is acceptable
    to both. If this ever overlaps, one of the two bands has drifted.
    """
    assert cfg.short_premium.max_delta <= cfg.signal.min_delta


def test_a_high_delta_strike_is_refused(cfg):
    v = strikes.evaluate(_quote(delta=0.45, premium=90.0), cfg)
    assert not v.eligible
    assert any(strikes.REJECT_DELTA_HIGH in f for f in v.failures)


def test_the_ledger_names_every_refused_strike_and_its_gate(cfg):
    """
    For a seller the strike IS the strategy, so a selection that reports only
    its winner is unauditable. Every strike considered must appear.
    """
    chain = [
        _quote(strike=26100, delta=0.40, premium=70.0),   # delta too high
        _quote(strike=26300, delta=0.15, premium=30.0),   # fine
        _quote(strike=26600, delta=0.04, premium=4.0),    # delta+credit
        _quote(strike=26400, delta=0.12, premium=20.0, oi=1_000),   # OI
        _quote(strike=26450, delta=0.11, premium=18.0, spread=0.30),  # spread
    ]
    sel = strikes.select(chain, "CE", cfg)
    assert len(sel.ledger) == len(chain), "every strike considered is recorded"
    assert sel.picked is not None and sel.picked.quote.strike == 26300

    rejected = {r["strike"]: r["failures"] for r in sel.rejection_rows()}
    assert set(rejected) == {26100, 26600, 26400, 26450}
    assert any(strikes.REJECT_OI in f for f in rejected[26400])
    assert any(strikes.REJECT_SPREAD in f for f in rejected[26450])
    # a strike that fails several gates says so, rather than short-circuiting
    assert len(rejected[26600]) >= 2


def test_a_credit_that_does_not_clear_the_round_trip_is_refused(cfg):
    """
    The mistake a delta filter alone will not catch. A 2-rupee credit on a
    65-lot is inside the delta band and is still a donation.
    """
    v = strikes.evaluate(_quote(delta=0.08, premium=2.0), cfg)
    assert not v.eligible
    assert any(strikes.REJECT_CREDIT_VS_COST in f or
               strikes.REJECT_CREDIT_LOW in f for f in v.failures)
    assert v.scratch_cost > 0


def test_a_synthetic_chain_cannot_pass_the_liquidity_gates(cfg):
    """
    `SyntheticChainSpec` fabricates OI of 500,000 and a 0.4% spread chosen to
    sit just inside the buying engine's gates, which is why that engine has
    never once rejected a contract for illiquidity. A seller must not be able
    to trade fiction, and the volume it does NOT fabricate is what stops it.
    """
    chain = synthetic_chain(26000.0, 3 / 365, "CE", cfg)
    sel = strikes.select(chain, "CE", cfg)
    assert sel.picked is None
    assert all(not v.eligible for v in sel.ledger)


def test_ranking_without_a_broker_margin_is_labelled_provisional(cfg):
    """
    Credit per rupee of margin is the seller's real efficiency metric and it
    can only come from the broker. The fallback is allowed; pretending it is
    the same number is not.
    """
    chain = [replace(q, volume=50_000)
             for q in synthetic_chain(26000.0, 3 / 365, "CE", cfg)]
    sel = strikes.select(chain, "CE", cfg)
    assert sel.picked is not None
    assert "PROVISIONAL" in sel.picked.score_basis
    assert "PROVISIONAL" in sel.note

    priced = strikes.select(chain, "CE", cfg,
                            margin_fn=lambda q: 95_000.0 + 400.0 * q.premium)
    assert priced.picked.score_basis == "credit_per_rupee_of_margin"
    assert "PROVISIONAL" not in priced.note


def test_partial_margin_coverage_does_not_mix_two_metrics(cfg):
    """
    Ranking half the candidates on margin and half on delta produces an
    ordering that means nothing. Partial coverage must fall back wholesale.
    """
    chain = [replace(q, volume=50_000)
             for q in synthetic_chain(26000.0, 3 / 365, "CE", cfg)]
    sel = strikes.select(chain, "CE", cfg,
                         margin_fn=lambda q: 95_000.0 if q.strike < 26400 else None)
    assert all("PROVISIONAL" in v.score_basis for v in sel.eligible)
    assert "broker priced" in sel.note


def test_a_margin_call_that_raises_is_not_a_margin(cfg):
    chain = [replace(q, volume=50_000)
             for q in synthetic_chain(26000.0, 3 / 365, "CE", cfg)]

    def boom(q):
        raise RuntimeError("kite down")

    sel = strikes.select(chain, "CE", cfg, margin_fn=boom)
    assert sel.picked is not None
    assert "PROVISIONAL" in sel.picked.score_basis


# ------------------------------------------------------------------ the costs

def test_a_seller_pays_stt_on_the_entry_leg(cfg):
    """
    `costs.CostModel` charges STT on the exit because a buyer buys first.
    Reversed, and it understates a seller's cost by most of its largest
    statutory component - and understates it MORE the better the trade went.
    """
    qty = 65
    entry = costs_sell.entry_friction(30.0, qty)
    exit_near_zero = costs_sell.exit_friction(1.0, qty)
    assert entry > exit_near_zero, "STT lands on the credit, at entry"

    from nifty_algo.costs import DEFAULT_COSTS
    stt = 30.0 * qty * DEFAULT_COSTS.stt_sell_pct
    assert stt > 0
    assert entry == pytest.approx(
        DEFAULT_COSTS.leg(30.0 * qty, is_sell=True)
        + DEFAULT_COSTS.slippage(qty, legs=1))


def test_the_public_leg_helper_is_pure_delegation():
    """`leg()` was added to CostModel for this book; it must change nothing."""
    from nifty_algo.costs import DEFAULT_COSTS as c
    for turnover in (0.0, 1_000.0, 250_000.0):
        for sell in (True, False):
            assert c.leg(turnover, sell) == c._leg(turnover, sell)


# ------------------------------------------------------------------- the pool

def test_an_unfunded_short_premium_pot_is_not_the_option_account():
    """
    The failure `_pool()` was made to raise on: sizing off the wrong pot
    produces an entirely plausible ticket.
    """
    c = CapitalConfig(starting_capital=100_000.0)
    assert c.capital_inr("short_premium") == 0.0
    assert c.risk_inr("short_premium") == 0.0
    assert c.capital_inr("home") == 100_000.0
    with pytest.raises(KeyError):
        c.capital_inr("short-premium")


def test_the_fifth_pool_did_not_change_the_other_four():
    """
    `risk_per_trade_pct` is now `risk_pct_for("home")` and `risk_inr` reads a
    per-pool entry count. Neither may move a number for the books that were
    already using them.
    """
    c = CapitalConfig(starting_capital=100_000.0, swing_capital_inr=100_000.0,
                      foreign_capital_inr=100_000.0,
                      intraday_equity_capital_inr=100_000.0)
    assert c.risk_per_trade_pct == pytest.approx(0.05 / 3)
    assert c.reward_per_trade_pct == pytest.approx(0.10 / 3)
    assert c.reward_risk_ratio == pytest.approx(2.0)
    for pool in ("home", "swing_india", "foreign", "intraday_equity"):
        assert c.entries_for(pool) == 3
        assert c.risk_inr(pool) == pytest.approx(100_000.0 * 0.05 / 3)
        assert c.reward_inr(pool) == pytest.approx(100_000.0 * 0.10 / 3)


def test_the_short_premium_pool_has_its_own_entry_count_and_one_formula():
    c = CapitalConfig(short_premium_capital_inr=600_000.0)
    assert c.entries_for("short_premium") == 6
    # still session_stop_pct / entries - not a second formula
    assert c.risk_pct_for("short_premium") == pytest.approx(0.05 / 6)
    assert c.risk_inr("short_premium") == pytest.approx(600_000.0 * 0.05 / 6)
