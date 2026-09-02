"""
From a signal to a quantity: the stop bounds, the risk budget, and the three
caps that may only ever REDUCE it.

THE STOP IS BOUNDED ON BOTH SIDES, AND BOTH BOUNDS REJECT RATHER THAN CLAMP

The original brief for this book was a 5% stop. Measured on the cached Nifty
100 history, the median stock's median daily high-low range is 2.04% and the
whole day's range exceeds 5% on only 3.7% of sessions - so a 5% intraday stop
is unreachable on ~96% of days. It is not a tight risk control, it is no
control, and it would make 1R a 5% move so the +1R/+2R ladder could never
fire once.

So it became a CEILING. And a ceiling has to reject, not clamp:

    clamping caps the DENOMINATOR of R while leaving the numerator free.
    A stock whose ATR wants a 9% stop, clamped to 5%, reports every 5% move
    as +1R instead of +0.55R. The most volatile names in the universe get
    their results inflated by the largest factor, and the equity curve looks
    like risk control working.

Rejecting is a fact the ledger records. Clamping is a lie the ledger cannot
see. There is also a FLOOR, because a stop tighter than the spread plus
slippage is not a stop, it is a guaranteed scratch.

LEVERAGE MULTIPLIES BUYING POWER, NOT RISK APPETITE

`mis_leverage` appears in exactly ONE expression in this module - the
affordability cap - and never in the numerator of the quantity. Getting this
wrong is trap T5, and it is attractive because it reads as "efficient capital
use" while multiplying every R in the result by the leverage factor. The
invariant that makes it impossible is asserted directly:

    quantity * (entry - stop)  <=  risk_inr,   for every ticket, always.

THE CAPS MAY ONLY REDUCE

Risk sets the quantity. Capital, leverage and participation are CEILINGS
applied afterwards, and none of them may ever raise it. `swing/scanner._size`
takes the same shape for the same reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Why a signal produced no ticket. Spelled once so the ledger, the tests and
#: the page cannot disagree about the string.
REJECT_STOP_TOO_WIDE = "stop_wider_than_cap"
REJECT_STOP_TOO_TIGHT = "stop_inside_slippage"
REJECT_NO_RISK_BUDGET = "pool_unfunded"
REJECT_NOT_AFFORDABLE = "capital"
REJECT_ZERO_QUANTITY = "quantity_rounds_to_zero"
REJECT_PARTICIPATION = "participation_cap"


@dataclass(frozen=True)
class Sized:
    """A quantity, and an honest account of what decided it."""
    quantity: int
    entry: float
    stop: float
    target: float
    risk_inr: float              # what is actually at risk: qty x (entry-stop)
    deployed_inr: float
    stop_pct: float
    note: str = ""
    #: Which cap bound the size, if any. "risk" means nothing else bit.
    bound_by: str = "risk"

    @property
    def risk_points(self) -> float:
        return self.entry - self.stop

    @property
    def reward_risk(self) -> float:
        rp = self.risk_points
        return (self.target - self.entry) / rp if rp > 0 else 0.0


def resolve_stop(entry: float, atr: float, cfg) -> tuple[float, str | None]:
    """
    The ATR stop, and the reason to reject it if it fails either bound.

    Returns (stop, None) or (0.0, reason). NEVER returns a clamped stop - see
    the module docstring.
    """
    ie = cfg.intraday_equity
    if entry <= 0 or atr <= 0:
        return 0.0, REJECT_STOP_TOO_TIGHT

    risk_points = atr * ie.atr_stop_multiple
    stop_pct = risk_points / entry

    if stop_pct > ie.max_stop_pct:
        return 0.0, REJECT_STOP_TOO_WIDE
    if stop_pct < ie.min_stop_pct:
        return 0.0, REJECT_STOP_TOO_TIGHT
    return entry - risk_points, None


def target_for(entry: float, stop: float, cfg) -> float:
    """
    Reward:risk comes from `CapitalConfig`, derived from the session
    governors - never set independently here. Three books, one ratio.
    """
    return entry + (entry - stop) * cfg.capital.reward_risk_ratio


def size(entry: float, stop: float, cfg, free_capital_inr: float | None = None,
         bar_volume: float | None = None,
         pool: str = "intraday_equity") -> tuple[Sized | None, str | None]:
    """
    Quantity from the risk budget, reduced by three caps.

    Returns (Sized, None) or (None, reason).

    `free_capital_inr` defaults to the whole pool. The live path passes what
    the broker actually reports, so a position the account cannot fund is
    refused before it is armed rather than after it is rejected.
    """
    ie = cfg.intraday_equity
    cap = cfg.capital

    risk_points = entry - stop
    if risk_points <= 0:
        return None, REJECT_STOP_TOO_TIGHT

    budget = cap.risk_inr(pool)
    if budget <= 0:
        # An unfunded pool STANDS THE BOOK DOWN rather than sizing off
        # another pot. `_pool` already raises on an unknown key.
        return None, REJECT_NO_RISK_BUDGET

    # ---- the quantity, from RISK and nothing else --------------------
    quantity = math.floor(budget / risk_points)
    bound_by = "risk"
    if quantity <= 0:
        return None, REJECT_ZERO_QUANTITY

    # ---- cap 1: what the account can actually fund --------------------
    # THE ONLY PLACE `mis_leverage` IS ALLOWED TO APPEAR.
    capital = cap.capital_inr(pool) if free_capital_inr is None else free_capital_inr
    buying_power = capital * max(ie.mis_leverage, 0.0)
    if entry > 0 and buying_power > 0:
        affordable = math.floor(buying_power / entry)
        if affordable < quantity:
            quantity, bound_by = affordable, REJECT_NOT_AFFORDABLE
    if quantity <= 0:
        return None, REJECT_NOT_AFFORDABLE

    # ---- cap 2: do not be the whole bar -------------------------------
    if bar_volume is not None and bar_volume > 0:
        allowed = math.floor(bar_volume * ie.participation_cap_pct)
        if allowed < quantity:
            quantity, bound_by = allowed, REJECT_PARTICIPATION
    if quantity <= 0:
        return None, REJECT_PARTICIPATION

    actual_risk = quantity * risk_points
    note = ""
    if bound_by == REJECT_NOT_AFFORDABLE:
        note = (f"size cut to what the pot funds "
                f"(Rs {capital:,.0f} x {ie.mis_leverage:g})")
    elif bound_by == REJECT_PARTICIPATION:
        note = (f"size cut to {ie.participation_cap_pct:.1%} of the fill "
                f"bar's volume")

    return Sized(
        quantity=int(quantity), entry=float(entry), stop=float(stop),
        target=float(target_for(entry, stop, cfg)),
        risk_inr=float(actual_risk),
        deployed_inr=float(quantity * entry),
        stop_pct=float(risk_points / entry),
        note=note, bound_by=bound_by,
    ), None



def leverage_to_be_risk_bound(stop_pct: float, cfg) -> float:
    """
    The leverage at which RISK rather than CAPITAL decides the quantity.

    Cash per ticket is `risk / stop_pct`, so the pot must cover
    `risk_pct / stop_pct` times its own value for one full-size position.
    Note what cancels: the answer does NOT depend on the size of the pot. A
    Rs 1 lakh account and a Rs 1 crore account are equally capital-bound at
    the same stop distance.

    This is the swing book's "the pot must be big enough for top_n
    positions" arithmetic, and it bites far harder here. A swing stop is ~4%,
    so one ticket is ~40% of the pot. An intraday ATR stop is ~0.6%, so one
    ticket is 2.8 TIMES the pot, and three are 8.3 times - more than
    Zerodha's ~5x MIS allowance on Nifty 100 names.

    The consequence is not an error. It is that positions come out smaller
    than the governors intend, every realised R is a fraction of the
    budgeted R, and the book quietly stops being the book that was designed.
    So `pot_note()` prints it above every result rather than leaving it to be
    inferred from disappointing numbers.
    """
    if stop_pct <= 0:
        return float("inf")
    return cfg.capital.risk_per_trade_pct / stop_pct


def pot_note(typical_stop_pct: float, cfg, pool: str = "intraday_equity") -> str:
    """
    One line on whether risk or capital is actually setting the size.

    Belongs above every backtest result and on the settings page, for the
    same reason `page_settings._pot_note` exists in the swing book: the
    arithmetic is not obvious, and getting it wrong looks like tight gates
    rather than a funding problem.
    """
    ie = cfg.intraday_equity
    cap = cfg.capital
    capital = cap.capital_inr(pool)
    if capital <= 0:
        return (f"{cap.pool_label(pool)} is unfunded - the book stands down. "
                f"Set {cap.pool_field(pool)}.")

    need_one = leverage_to_be_risk_bound(typical_stop_pct, cfg)
    need_all = need_one * ie.top_n
    have = ie.mis_leverage
    cash = cap.risk_inr(pool) / typical_stop_pct if typical_stop_pct > 0 else 0.0

    verdict = "RISK-bound" if have >= need_one else "CAPITAL-bound"
    line = (f"At a {typical_stop_pct:.2%} stop one full-size ticket is "
            f"Rs {cash:,.0f} ({cash / capital:.1f}x the pot). "
            f"Risk decides the size from {need_one:.1f}x leverage; "
            f"{ie.top_n} concurrent need {need_all:.1f}x. "
            f"You have {have:g}x, so sizing is {verdict}.")
    if have < need_one:
        line += (" Positions will be SMALLER than the governors intend and "
                 "every realised R is a fraction of the budgeted R.")
    return line


__all__ = ["Sized", "resolve_stop", "target_for", "size",
           "leverage_to_be_risk_bound", "pot_note",
           "REJECT_STOP_TOO_WIDE", "REJECT_STOP_TOO_TIGHT",
           "REJECT_NO_RISK_BUDGET", "REJECT_NOT_AFFORDABLE",
           "REJECT_ZERO_QUANTITY", "REJECT_PARTICIPATION"]
