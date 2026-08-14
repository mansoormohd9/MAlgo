"""
The day rules.

These are pure arithmetic on realised P&L, which is exactly why they get their
own file: no chain, no bars, no strategy. If the ratchet is wrong, it is wrong
here and nowhere else.
"""
import pytest

from nifty_algo.config import Config
from nifty_algo.governor import (SessionGovernor, GovernorAction,
                                 GovernorReason)


@pytest.fixture
def gov():
    g = SessionGovernor(Config(), capital=100_000.0)
    g.start_day()
    return g


# ---------------------------------------------------------------- the ladder

@pytest.mark.parametrize("peak, expected_floor", [
    (0,      -5_000),      # opening budget: 5% of capital
    (2_000,  -3_000),      # the stated rule - day stop now reads 3%
    (4_000,  -1_000),
    (6_000,   1_000),      # from here the day cannot finish red
    (8_000,   3_000),
])
def test_floor_ratchets_with_the_peak(gov, peak, expected_floor):
    gov.register_exit(peak)
    assert gov.floor == pytest.approx(expected_floor)


def test_two_percent_peak_reads_as_a_three_percent_day_stop(gov):
    """The rule as it was actually stated: 5% becomes 3% once 2% is banked."""
    assert gov.floor_pct_of_capital == pytest.approx(-0.05)
    gov.register_exit(2_000)
    assert gov.floor_pct_of_capital == pytest.approx(-0.03)


def test_floor_never_loosens(gov):
    """
    A floor that could fall back would let a good morning fund a worse
    afternoon - the exact failure the rule exists to prevent.
    """
    gov.register_exit(6_000)
    assert gov.floor == pytest.approx(1_000)

    gov.register_exit(-3_000)          # realised back to 3,000
    assert gov.peak_realised_pnl == pytest.approx(6_000)
    assert gov.floor == pytest.approx(1_000)

    gov.register_exit(-1_500)          # realised 1,500, still above the floor
    assert gov.floor == pytest.approx(1_000)


def test_floor_is_never_looser_than_the_opening_budget(gov):
    """A losing day must not widen its own stop below -5%."""
    gov.register_exit(-2_000)
    assert gov.floor == pytest.approx(-5_000)
    assert gov.peak_realised_pnl == pytest.approx(0.0)


def test_risk_remaining_is_always_three_trades_below_the_peak(gov):
    """
    Give-back equals the opening budget, so there is always exactly three
    trades' worth of room under the peak. That is what keeps the ratchet
    consistent with max_entries_per_session = 3.
    """
    risk_per_trade = Config().capital.risk_per_trade_rupees
    for banked in (0, 1_000, 4_000, 7_000):
        g = SessionGovernor(Config(), capital=100_000.0)
        g.start_day()
        g.register_exit(banked)
        assert g.risk_remaining == pytest.approx(3 * risk_per_trade, abs=1.0)


# ---------------------------------------------------------------- verdicts

def test_day_target_closes_everything(gov):
    v = gov.register_exit(10_000)
    assert v.action is GovernorAction.CLOSE_ALL
    assert v.reason is GovernorReason.SESSION_TARGET_HIT


def test_first_trade_alone_can_end_the_day(gov):
    """
    The runner exists so trade 1 can reach the day target by itself. When it
    does, the rule has to CLOSE, not merely stop new entries - there is still
    an open position at that instant.
    """
    gov.register_entry()
    v = gov.register_exit(10_500)
    assert v.action is GovernorAction.CLOSE_ALL
    assert gov.entries_taken == 1


def test_partial_exits_move_the_peak_and_can_trip_the_target(gov):
    """A partial fill is a realised P&L like any other."""
    gov.register_entry()
    assert gov.register_exit(3_400).action is GovernorAction.CONTINUE
    assert gov.floor == pytest.approx(-1_600)
    v = gov.register_exit(6_800)
    assert v.action is GovernorAction.CLOSE_ALL
    assert v.reason is GovernorReason.SESSION_TARGET_HIT


def test_hitting_the_ratcheted_floor_closes_the_day(gov):
    gov.register_exit(4_000)           # floor now -1,000
    v = gov.register_exit(-5_000)      # realised -1,000
    assert v.action is GovernorAction.CLOSE_ALL
    assert v.reason is GovernorReason.GIVE_BACK_FLOOR_HIT


def test_three_losses_land_exactly_on_the_opening_floor(gov):
    risk = Config().capital.risk_per_trade_rupees
    for _ in range(2):
        assert gov.register_exit(-risk).action is not GovernorAction.CLOSE_ALL
    v = gov.register_exit(-risk)
    assert v.action is GovernorAction.CLOSE_ALL
    assert gov.realised_pnl == pytest.approx(-5_000)


def test_three_entries_blocks_new_ones_but_does_not_flatten(gov):
    """
    Running out of entries is not the same event as losing the day. Open
    positions keep being managed.
    """
    for _ in range(3):
        gov.register_entry()
    v = gov.evaluate()
    assert v.action is GovernorAction.BLOCK_NEW_ENTRIES
    assert not v.day_over


def test_delayed_arming_variant():
    """
    The alternative reading of the rule - do not trail until +2% is banked -
    is a config change, not a code change.
    """
    cfg = Config()
    cfg.capital.ratchet_arm_at_pct = 0.02
    g = SessionGovernor(cfg, capital=100_000.0)
    g.start_day()

    g.register_exit(1_000)
    assert g.floor == pytest.approx(-5_000)     # not yet armed
    g.register_exit(1_000)                      # peak now 2,000
    assert g.floor == pytest.approx(-3_000)     # armed
