"""
The exit ladder.

`ExitLadder` is unit-free - it takes R and returns a decision - so it can be
tested without a chain, a premium or a bar. That is the point of the design:
the live engine and the backtester feed the same state machine, so proving it
here proves it for both.
"""
import pytest

from nifty_algo.config import Config
from nifty_algo.positions import (ExitKind, ExitLadder, LadderMode,
                                  PositionManager)
from nifty_algo.risk import ApprovedOrder, OptionQuote


@pytest.fixture
def ladder():
    return ExitLadder(Config())


def kinds(decisions):
    return [d.kind for d in decisions]


def only(decisions):
    """The single decision a bar produced - asserts there was exactly one."""
    assert len(decisions) == 1, kinds(decisions)
    return decisions[0]


def quote(strike=26_000, premium=120.0, delta=0.42):
    return OptionQuote(strike=strike, option_type="CE", premium=premium,
                       delta=delta, bid=premium - 0.2, ask=premium + 0.2,
                       open_interest=500_000)


# ---------------------------------------------------------------- breakeven

def test_stop_moves_to_breakeven_at_exactly_one_r(ladder):
    st = ladder.new_state(lots=2)
    assert st.stop_r == pytest.approx(-1.0)

    assert ladder.advance(st, mark_r=0.99) == []
    assert st.stop_r == pytest.approx(-1.0)

    d = only(ladder.advance(st, mark_r=1.0))
    assert d.kind is ExitKind.STOP_TO_BREAKEVEN
    assert st.stop_r == pytest.approx(0.0)
    assert st.mode is LadderMode.BREAKEVEN


def test_after_breakeven_the_trade_cannot_lose(ladder):
    st = ladder.new_state(lots=2)
    ladder.advance(st, mark_r=1.2)
    d = only(ladder.advance(st, mark_r=-0.5, worst_r=-0.5))
    assert d.kind is ExitKind.STOPPED_OUT
    assert d.exit_r == pytest.approx(0.0)      # scratch, not a loss


# ---------------------------------------------------------------- the runner

def test_partial_exit_at_two_r_leaves_a_runner(ladder):
    st = ladder.new_state(lots=2)
    ladder.advance(st, mark_r=1.0)             # breakeven first

    d = only(ladder.advance(st, mark_r=2.0))
    assert d.kind is ExitKind.PARTIAL_EXIT
    assert d.exit_lots == 1
    assert st.lots_remaining == 1
    assert st.mode is LadderMode.TRAIL
    assert not st.closed


def test_one_fast_bar_crosses_every_rung_it_passed(ladder):
    """
    A gap or a violent 5-minute candle can carry a trade from flat through
    breakeven and past +2R before another price is seen. Emitting only the
    first transition would leave the partial un-banked until the next bar.
    """
    st = ladder.new_state(lots=2)
    out = ladder.advance(st, mark_r=3.0, trail_distance_r=1.0)

    assert kinds(out) == [ExitKind.STOP_TO_BREAKEVEN,
                          ExitKind.PARTIAL_EXIT,
                          ExitKind.TRAIL_UPDATE]
    assert st.lots_remaining == 1
    assert st.stop_r == pytest.approx(2.0)     # peak 3R, 1R trail


def test_single_lot_exits_fully_at_two_r(ladder):
    """
    NSE fills whole lots only - 65 quantity cannot be halved - so a one-lot
    position has no runner and takes the plain 2:1 target instead.
    """
    st = ladder.new_state(lots=1)
    assert not ladder.runner_enabled(1)

    ladder.advance(st, mark_r=1.0)
    d = only(ladder.advance(st, mark_r=2.0))
    assert d.kind is ExitKind.TARGET_EXIT
    assert d.exit_lots == 1
    assert st.closed


def test_runner_can_reach_far_beyond_two_r(ladder):
    """This is what makes the 10%-in-the-first-order rule reachable at all."""
    st = ladder.new_state(lots=2)
    ladder.advance(st, mark_r=1.0)
    ladder.advance(st, mark_r=2.0)

    for r in (3.0, 4.0, 5.0, 6.0):
        ladder.advance(st, mark_r=r, trail_distance_r=1.0)
    assert st.lots_remaining == 1
    assert st.stop_r == pytest.approx(5.0)     # peak 6R minus a 1R trail


# ---------------------------------------------------------------- the trail

def test_trailing_stop_only_ever_ratchets(ladder):
    """
    The single most important assertion in this file. A trail that can loosen
    is not a trail, it is a stop that moves in whichever direction flatters
    the equity curve.
    """
    st = ladder.new_state(lots=2)
    ladder.advance(st, mark_r=1.0)
    ladder.advance(st, mark_r=2.0)

    ladder.advance(st, mark_r=5.0, trail_distance_r=1.0)
    assert st.stop_r == pytest.approx(4.0)

    # Price falls back. The stop must not follow it down.
    ladder.advance(st, mark_r=4.2, trail_distance_r=1.0)
    assert st.stop_r == pytest.approx(4.0)

    # A wider ATR must not loosen it either.
    ladder.advance(st, mark_r=4.5, trail_distance_r=3.0)
    assert st.stop_r == pytest.approx(4.0)


def test_trail_does_not_engage_before_the_partial(ladder):
    st = ladder.new_state(lots=2)
    ladder.advance(st, mark_r=1.5, trail_distance_r=0.5)
    assert st.stop_r == pytest.approx(0.0)     # breakeven only, no trail yet


# ---------------------------------------------------------------- pessimism

def test_a_bar_touching_both_the_stop_and_the_next_promotion_is_a_stop(ladder):
    """
    Intrabar path is unknowable. Scoring such a bar as the favourable outcome
    is the most common way an equity curve inflates itself, so the stop wins.
    """
    st = ladder.new_state(lots=2)
    d = only(ladder.advance(st, mark_r=0.5, best_r=2.5, worst_r=-1.0))
    assert d.kind is ExitKind.STOPPED_OUT
    assert d.exit_r == pytest.approx(-1.0)
    assert st.closed


def test_the_stop_is_tested_at_its_pre_bar_level(ladder):
    """
    A bar cannot both raise the stop and then be measured against the raised
    stop - that would let a promotion rescue a bar that had already stopped out.
    """
    st = ladder.new_state(lots=2)
    ladder.advance(st, mark_r=1.0)             # stop now 0R
    d = only(ladder.advance(st, mark_r=1.0, best_r=3.0, worst_r=-0.2))
    assert d.kind is ExitKind.STOPPED_OUT
    assert d.exit_r == pytest.approx(0.0)


# ---------------------------------------------------------------- force exit

def test_force_exit_closes_whatever_remains(ladder):
    st = ladder.new_state(lots=2)
    ladder.advance(st, mark_r=1.0)
    ladder.advance(st, mark_r=2.0)             # one lot banked

    d = ladder.force_exit(st, mark_r=3.4)
    assert d.kind is ExitKind.FORCE_EXIT
    assert d.exit_lots == 1
    assert st.closed


def test_close_all_flattens_for_a_governor(ladder):
    st = ladder.new_state(lots=2)
    d = ladder.close_all(st, mark_r=1.1, reason="day target hit")
    assert d.kind is ExitKind.CLOSE_ALL
    assert d.exit_lots == 2
    assert st.closed


def test_a_closed_position_produces_nothing_further(ladder):
    st = ladder.new_state(lots=1)
    ladder.advance(st, mark_r=-1.0, worst_r=-1.0)
    assert st.closed
    assert ladder.advance(st, mark_r=5.0) == []
    assert ladder.force_exit(st, mark_r=5.0).kind is None


# ---------------------------------------------------------------- prices

def _order(lots=2, premium=120.0):
    cfg = Config()
    qty = lots * cfg.instrument.lot_size
    risk = cfg.capital.risk_per_trade_rupees
    return ApprovedOrder(
        quote=quote(premium=premium), lots=lots, quantity=qty,
        entry_premium=premium,
        premium_stop=premium - risk / qty,
        premium_target=premium + 2 * risk / qty,
        underlying_stop_points=30.0,
        rupee_risk=risk, rupee_reward=cfg.capital.reward_per_trade_rupees,
    )


def test_one_r_in_premium_matches_the_risk_engines_stop():
    """
    The ladder's -1R and RiskEngine's premium_stop must be the SAME price,
    not two numbers that happen to land near each other.
    """
    mgr = PositionManager(Config())
    order = _order()
    pos = mgr.open(order, "long", "level_break", entry_underlying=26_000)

    assert pos.premium_of(-1.0) == pytest.approx(order.premium_stop)
    assert pos.premium_of(2.0) == pytest.approx(order.premium_target)
    assert pos.r_of(order.entry_premium) == pytest.approx(0.0)


def test_partial_exit_banks_close_to_one_lots_worth_of_reward():
    cfg = Config()
    mgr = PositionManager(cfg)
    order = _order(lots=2)
    pos = mgr.open(order, "long", "level_break", entry_underlying=26_000)

    two_r = pos.premium_of(2.0)
    actions = mgr.update({"26000CE": two_r}, atr=30.0)

    # With ATR 30 against a 30-point stop the trail is 1R wide, so the runner's
    # stop lands at +1R the moment the partial is banked.
    assert kinds(actions) == [ExitKind.STOP_TO_BREAKEVEN,
                              ExitKind.PARTIAL_EXIT,
                              ExitKind.TRAIL_UPDATE]
    a = actions[1]
    assert a.quantity == cfg.instrument.lot_size
    # One lot of a two-lot position at +2R is 1R of rupees.
    assert a.gross_pnl == pytest.approx(cfg.capital.risk_per_trade_rupees, rel=0.01)
    assert pos.state.stop_r == pytest.approx(1.0)


def test_a_position_with_no_quote_is_skipped_not_marked_stale():
    """
    Acting on a price you did not receive is how a trailing stop fires on a
    gap that never happened.
    """
    mgr = PositionManager(Config())
    mgr.open(_order(), "long", "level_break", entry_underlying=26_000)
    assert mgr.update({}, atr=30.0) == []
    assert mgr.open_lots == 2


def test_force_exit_time_overrides_everything():
    from datetime import time
    cfg = Config()
    mgr = PositionManager(cfg)
    pos = mgr.open(_order(), "long", "level_break", entry_underlying=26_000)

    actions = mgr.update({"26000CE": pos.premium_of(1.9)}, atr=30.0,
                         now=time(15, 10))
    assert [a.kind for a in actions] == [ExitKind.FORCE_EXIT]
    assert mgr.open_lots == 0
