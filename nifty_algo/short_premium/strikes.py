"""
Which strike to sell, and - just as importantly - why every other one was not.

THE INVERSION. `risk.select_strike()` returns `max(candidates, key=abs(delta))`
because a buyer wants the most directional capture per rupee of theta paid.
Run that logic on a seller and it picks the strike with the highest chance of
finishing in the money. This module returns the strike that pays the best
credit for the risk it takes on, from a band the buying engine refuses
entirely.

THE LEDGER IS THE DELIVERABLE. `risk.gate_failures()` exists because "a chain
table that only shows the winner teaches you nothing about why the other
twenty-four lost". That applies harder here: for a directional buyer the
strategy is the signal and the strike is an implementation detail, whereas for
a seller THE STRIKE IS THE STRATEGY. `Selection.ledger` therefore carries every
strike that was looked at, eligible or not, with the gate it failed and by how
much - and the caller journals all of it, not just the fill.

RANKING, AND WHY IT IS PROVISIONAL WITHOUT MARGIN. A seller's real efficiency
is credit per rupee of margin blocked, because margin is what the position
actually consumes. That number can only come from the broker
(`basket_order_margins`), never from a local SPAN model. When it is absent this
module ranks on credit per unit of delta instead and SAYS SO in
`Selection.note` - it does not silently substitute one metric for the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import Config, DEFAULT
from ..costs import CostModel, DEFAULT_COSTS
from ..risk import OptionQuote
from . import costs_sell

# Rejection reasons. Strings rather than an Enum because they are journalled
# and read by a human looking for the phrase they saw in the UI.
REJECT_DELTA_HIGH = "delta_above_band"
REJECT_DELTA_LOW = "delta_below_band"
REJECT_CREDIT_LOW = "credit_below_floor"
REJECT_CREDIT_VS_COST = "credit_not_worth_the_round_trip"
REJECT_SPREAD = "spread_too_wide"
REJECT_OI = "open_interest_too_thin"
REJECT_VOLUME = "volume_too_thin"
REJECT_NO_IV = "premium_not_invertible"


@dataclass
class StrikeVerdict:
    """One strike, judged. Carries the arithmetic, not just the answer."""
    quote: OptionQuote
    credit: float                 # per unit, the mid
    scratch_cost: float           # per unit, to open and close flat
    credit_to_cost: float
    failures: list[str] = field(default_factory=list)
    margin_per_lot: Optional[float] = None
    score: Optional[float] = None
    score_basis: str = ""

    @property
    def eligible(self) -> bool:
        return not self.failures

    @property
    def label(self) -> str:
        return f"{self.quote.strike}{self.quote.option_type}"

    def as_row(self) -> dict:
        """Flat, for the journal and the UI table."""
        return {
            "strike": self.quote.strike,
            "option_type": self.quote.option_type,
            "credit": round(self.credit, 2),
            "delta": round(self.quote.delta, 4),
            "iv": self.quote.iv,
            "spread_pct": round(self.quote.spread_pct, 5),
            "open_interest": self.quote.open_interest,
            "volume": self.quote.volume,
            "scratch_cost": round(self.scratch_cost, 3),
            "credit_to_cost": round(self.credit_to_cost, 2),
            "margin_per_lot": self.margin_per_lot,
            "score": self.score,
            "score_basis": self.score_basis,
            "eligible": self.eligible,
            "failures": list(self.failures),
        }


@dataclass
class Selection:
    picked: Optional[StrikeVerdict]
    ledger: list[StrikeVerdict]
    note: str = ""

    @property
    def eligible(self) -> list[StrikeVerdict]:
        return [v for v in self.ledger if v.eligible]

    def rejection_rows(self) -> list[dict]:
        return [v.as_row() for v in self.ledger if not v.eligible]


def evaluate(quote: OptionQuote, cfg: Config = DEFAULT,
             lot_size: Optional[int] = None,
             costs: CostModel = DEFAULT_COSTS) -> StrikeVerdict:
    """
    Judge one strike against every gate, recording each failure and its
    margin. Never short-circuits: a strike that fails three gates says so,
    because "it failed" and "it failed everything" are different findings
    when you are tuning a band.
    """
    sp = cfg.short_premium
    qty = int(lot_size if lot_size is not None else cfg.instrument.lot_size)
    credit = float(quote.premium)

    scratch = costs_sell.scratch_cost_per_unit(credit, qty, costs)
    ratio = (credit / scratch) if scratch > 0 else 0.0

    v = StrikeVerdict(quote=quote, credit=credit, scratch_cost=scratch,
                      credit_to_cost=ratio)

    # An un-invertible premium reached us with no IV. `kite_chain.get_chain`
    # drops those already; a synthetic or replayed chain may not.
    if quote.iv is None:
        v.failures.append(f"{REJECT_NO_IV}: no IV could be solved")

    d = abs(quote.delta)
    if d > sp.max_delta:
        v.failures.append(
            f"{REJECT_DELTA_HIGH}: delta {d:.3f} > {sp.max_delta:.2f}")
    elif d < sp.min_delta:
        v.failures.append(
            f"{REJECT_DELTA_LOW}: delta {d:.3f} < {sp.min_delta:.2f}")

    if credit < sp.min_credit:
        v.failures.append(
            f"{REJECT_CREDIT_LOW}: credit {credit:.2f} < {sp.min_credit:.2f}")

    if ratio < sp.min_credit_to_cost_multiple:
        v.failures.append(
            f"{REJECT_CREDIT_VS_COST}: credit {credit:.2f} is "
            f"{ratio:.1f}x the {scratch:.2f} round trip, "
            f"want {sp.min_credit_to_cost_multiple:.1f}x")

    if quote.spread_pct > sp.max_spread_pct_of_premium:
        v.failures.append(
            f"{REJECT_SPREAD}: spread {quote.spread_pct:.2%} > "
            f"{sp.max_spread_pct_of_premium:.2%}")

    if quote.open_interest < sp.min_open_interest:
        v.failures.append(
            f"{REJECT_OI}: OI {quote.open_interest:,} < "
            f"{sp.min_open_interest:,}")

    if quote.volume < sp.min_volume:
        v.failures.append(
            f"{REJECT_VOLUME}: volume {quote.volume:,} < {sp.min_volume:,}")

    return v


def select(chain: list[OptionQuote], option_type: str,
           cfg: Config = DEFAULT,
           lot_size: Optional[int] = None,
           margin_fn: Optional[Callable[[OptionQuote], Optional[float]]] = None,
           costs: CostModel = DEFAULT_COSTS) -> Selection:
    """
    The strike to sell, plus the full ledger of what was refused.

    `margin_fn` maps a quote to the margin one lot of it blocks, and is the
    broker's answer or nothing - see the module docstring. When it is absent,
    or returns None for every candidate, ranking falls back to credit per unit
    of delta and `Selection.note` records that the ranking is provisional.
    """
    sp = cfg.short_premium
    considered = [q for q in chain if q.option_type == option_type]
    ledger = [evaluate(q, cfg, lot_size, costs) for q in considered]

    if not considered:
        return Selection(None, ledger,
                         note=f"no {option_type} quotes in the chain")

    eligible = [v for v in ledger if v.eligible]
    if not eligible:
        return Selection(
            None, ledger,
            note=(f"{len(considered)} {option_type} strikes considered, "
                  f"none passed - see the ledger for the gate each one failed"))

    priced = 0
    if margin_fn is not None:
        for v in eligible:
            try:
                m = margin_fn(v.quote)
            except Exception:
                m = None            # a margin call that fails is not a margin
            if m is not None and m > 0:
                v.margin_per_lot = float(m)
                priced += 1

    if priced == len(eligible):
        for v in eligible:
            v.score = v.credit / v.margin_per_lot
            v.score_basis = "credit_per_rupee_of_margin"
        note = (f"{len(eligible)} of {len(considered)} {option_type} strikes "
                f"eligible; ranked on credit per rupee of margin")
    else:
        # PROVISIONAL, and named as such. Partial margin coverage is not used
        # at all: ranking half the candidates on one metric and half on
        # another produces an ordering that means nothing.
        for v in eligible:
            d = max(abs(v.quote.delta), 1e-6)
            v.score = v.credit / d
            v.score_basis = "credit_per_unit_delta_PROVISIONAL"
        why = ("no margin function supplied" if margin_fn is None
               else f"broker priced {priced} of {len(eligible)} candidates")
        note = (f"{len(eligible)} of {len(considered)} {option_type} strikes "
                f"eligible; ranked PROVISIONALLY on credit per unit delta "
                f"({why}) - this is not the seller's real efficiency metric")

    picked = max(eligible, key=lambda v: (v.score, -abs(v.quote.delta)))
    return Selection(picked, ledger, note=note)


def band_note(cfg: Config = DEFAULT) -> str:
    """One line describing the band, for a brief or a stand-down message."""
    sp = cfg.short_premium
    return (f"delta {sp.min_delta:.2f}-{sp.max_delta:.2f}, "
            f"credit >= {sp.min_credit:.0f} and >= "
            f"{sp.min_credit_to_cost_multiple:.0f}x the round trip, "
            f"spread <= {sp.max_spread_pct_of_premium:.1%}, "
            f"OI >= {sp.min_open_interest:,}")
