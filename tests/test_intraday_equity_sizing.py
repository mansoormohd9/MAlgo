"""
Sizing, the two stop bounds, and the leverage invariant.

Every test here guards a failure that would produce a FLATTERING result
rather than an error - a bigger position, a better R multiple, or a trade
that should not have existed. None of them would raise.
"""
from __future__ import annotations

import pytest

from nifty_algo.config import Config
from nifty_algo.intraday_equity import sizing


@pytest.fixture
def cfg():
    c = Config()
    c.capital.intraday_equity_capital_inr = 100_000.0
    return c


# ------------------------------------------------------------ the stop


def test_a_stop_wider_than_the_cap_is_rejected_not_clamped(cfg):
    """
    TRAP T2, and the second most dangerous error available.

    Clamping caps the DENOMINATOR of R while leaving the numerator free, so
    the most volatile names get their results inflated by the largest factor
    - and it looks like risk control while doing the opposite.
    """
    entry = 1000.0
    atr = 90.0                      # a 9% stop at multiple 1.0, well over the 5% cap
    stop, reason = sizing.resolve_stop(entry, atr, cfg)

    assert reason == sizing.REJECT_STOP_TOO_WIDE
    assert stop == 0.0, "a rejected signal must not come back with a stop at all"

    # and specifically: it must NOT have been clamped to the 5% boundary
    clamped = entry * (1 - cfg.intraday_equity.max_stop_pct)
    assert stop != clamped, "the stop was clamped to the cap instead of rejected"


def test_a_stop_tighter_than_the_floor_is_rejected(cfg):
    """A stop inside the spread is not a stop, it is a guaranteed scratch."""
    stop, reason = sizing.resolve_stop(1000.0, 0.5, cfg)   # 0.05%, under the floor
    assert reason == sizing.REJECT_STOP_TOO_TIGHT
    assert stop == 0.0


def test_a_stop_inside_both_bounds_is_accepted(cfg):
    """The ordinary case, so the two rejections above are not vacuous."""
    stop, reason = sizing.resolve_stop(1000.0, 6.0, cfg)   # 0.6%
    assert reason is None
    assert stop == pytest.approx(994.0)


def test_the_five_percent_cap_is_a_ceiling_not_the_working_stop(cfg):
    """
    The measured fact this whole design turns on: a 5% intraday stop is
    unreachable on ~96% of Nifty 100 sessions, so it can only be a ceiling.
    A typical ~0.6% ATR stop must pass it untouched.
    """
    typical_atr_pct = 0.006
    stop, reason = sizing.resolve_stop(1000.0, 1000.0 * typical_atr_pct, cfg)
    assert reason is None
    assert stop == pytest.approx(994.0)
    assert (1000.0 - stop) / 1000.0 < cfg.intraday_equity.max_stop_pct


# ------------------------------------------------------------ leverage


def test_leverage_never_increases_the_risk_budget(cfg):
    """
    TRAP T5. Leverage multiplies BUYING POWER, not risk appetite.

    Asserted as the global invariant rather than a spot check, because the
    failure mode is that every R in the result is quietly multiplied.

    Note the leverages used: at a 0.6% stop the book is CAPITAL-bound below
    ~2.8x (see `test_the_book_is_capital_bound_at_intraday_stops`), so the
    quantities only become comparable once leverage clears that. The
    invariant below holds at every leverage regardless.
    """
    entry, stop = 1000.0, 994.0
    budget = cfg.capital.risk_inr("intraday_equity")

    sizes = {}
    for lev in (1.0, 2.0, 5.0, 10.0, 50.0):
        cfg.intraday_equity.mis_leverage = lev
        s, reason = sizing.size(entry, stop, cfg)
        assert reason is None
        sizes[lev] = s.quantity
        assert s.risk_inr <= budget + 1e-9, (
            f"at {lev}x leverage the ticket risks Rs {s.risk_inr:.2f} against "
            f"a budget of Rs {budget:.2f} - leverage has reached the risk "
            f"formula, and every R in the result is inflated by {lev}x")

    # Once capital stops binding, more leverage must change NOTHING.
    assert sizes[5.0] == sizes[10.0] == sizes[50.0], (
        "leverage kept increasing the quantity after capital stopped being "
        "the binding constraint; it may only ever RELAX the affordability cap")


def test_the_book_is_capital_bound_at_intraday_stops(cfg):
    """
    A structural property worth pinning, because it is invisible in a result.

    Cash per ticket is `risk / stop_pct`. A swing stop of ~4% makes that ~40%
    of the pot. An intraday ATR stop of ~0.6% makes it 2.8 TIMES the pot, and
    `top_n` of those needs 8.3x - more than Zerodha's ~5x MIS allowance.

    What cancels is the pot size: the ratio depends only on the risk fraction
    and the stop, so a bigger account does not fix it. Below that leverage
    the book still trades, just smaller than the governors intend, and every
    realised R is a fraction of the budgeted one. That reads as weak
    performance rather than as a funding problem, which is why `pot_note()`
    prints it above every result.
    """
    stop_pct = 0.006
    need = sizing.leverage_to_be_risk_bound(stop_pct, cfg)
    assert need == pytest.approx(cfg.capital.risk_per_trade_pct / stop_pct)
    assert need > 2.5, "a 0.6% stop should need meaningful leverage to be risk-bound"

    # independent of the pot
    cfg.capital.intraday_equity_capital_inr = 10_000_000.0
    assert sizing.leverage_to_be_risk_bound(stop_pct, cfg) == pytest.approx(need)

    # and the note says which way round it is
    cfg.intraday_equity.mis_leverage = 1.0
    assert "CAPITAL-bound" in sizing.pot_note(stop_pct, cfg)
    cfg.intraday_equity.mis_leverage = need + 0.5
    assert "RISK-bound" in sizing.pot_note(stop_pct, cfg)


def test_an_unfunded_pool_note_says_so(cfg):
    cfg.capital.intraday_equity_capital_inr = 0.0
    note = sizing.pot_note(0.006, cfg)
    assert "unfunded" in note and "stands down" in note
    assert "capital.intraday_equity_capital_inr" in note


def test_leverage_only_relaxes_a_binding_capital_cap(cfg):
    """
    The other half: leverage must genuinely do something when capital IS the
    binding constraint, or it is not implemented at all.
    """
    cfg.capital.intraday_equity_capital_inr = 20_000.0   # small pot, big share price
    entry, stop = 5000.0, 4970.0

    cfg.intraday_equity.mis_leverage = 1.0
    unlevered, _ = sizing.size(entry, stop, cfg)
    cfg.intraday_equity.mis_leverage = 5.0
    levered, _ = sizing.size(entry, stop, cfg)

    assert levered.quantity > unlevered.quantity
    assert unlevered.bound_by == sizing.REJECT_NOT_AFFORDABLE
    # ...and even levered, the risk budget still binds it
    assert levered.risk_inr <= cfg.capital.risk_inr("intraday_equity") + 1e-9


# ------------------------------------------------------------ the caps


def test_an_unfunded_pool_stands_the_book_down(cfg):
    """
    Never size off another pot. The swing book's `_pool()` already raises on
    an unknown key; this is the funded-but-zero case.
    """
    cfg.capital.intraday_equity_capital_inr = 0.0
    s, reason = sizing.size(1000.0, 994.0, cfg)
    assert s is None
    assert reason == sizing.REJECT_NO_RISK_BUDGET


def test_quantity_never_exceeds_participation_cap_of_the_fill_bar(cfg):
    """
    TRAP T10. Risk-based sizing will happily buy a third of a 5-minute bar
    on a mid-cap, and a backtest reports that fill as free.
    """
    entry, stop = 1000.0, 994.0
    thin_bar = 1_000.0            # only 1000 shares traded in the fill bar
    s, reason = sizing.size(entry, stop, cfg, bar_volume=thin_bar)

    assert reason is None
    cap = thin_bar * cfg.intraday_equity.participation_cap_pct
    assert s.quantity <= cap
    assert s.bound_by == sizing.REJECT_PARTICIPATION
    assert "volume" in s.note


def test_the_caps_only_ever_reduce(cfg):
    """
    Risk sets the quantity; capital, leverage and participation are ceilings.
    None of them may raise it.
    """
    entry, stop = 1000.0, 994.0
    unconstrained, _ = sizing.size(entry, stop, cfg)

    for kwargs in ({"free_capital_inr": 5_000.0},
                   {"bar_volume": 500.0},
                   {"free_capital_inr": 5_000.0, "bar_volume": 500.0}):
        s, reason = sizing.size(entry, stop, cfg, **kwargs)
        if s is not None:
            assert s.quantity <= unconstrained.quantity, (
                f"a cap increased the size with {kwargs}")


def test_reward_risk_comes_from_the_governors(cfg):
    """
    Three books, one ratio. `reward_risk_ratio` is derived from the session
    governors in `CapitalConfig`, never set independently in this book.
    """
    s, _ = sizing.size(1000.0, 994.0, cfg)
    assert s.reward_risk == pytest.approx(cfg.capital.reward_risk_ratio)
    assert s.target == pytest.approx(
        1000.0 + 6.0 * cfg.capital.reward_risk_ratio)


def test_risk_inr_is_what_is_actually_at_risk(cfg):
    """
    Not the budget - the budget rounded down to whole shares. Reporting the
    budget would overstate risk on every ticket and understate every R.
    """
    s, _ = sizing.size(1000.0, 994.0, cfg)
    assert s.risk_inr == pytest.approx(s.quantity * 6.0)
    assert s.risk_inr <= cfg.capital.risk_inr("intraday_equity")
