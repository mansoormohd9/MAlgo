"""
The tax you can legally not pay: the yearly LTCG exemption on Indian equity.

WHY THIS EXISTS. Long-term gains on equity-oriented holdings (listed shares,
equity ETFs and equity mutual funds) are exempt up to ₹1.25 lakh each
financial year (section 112A). Booking gains up to that line and buying back
resets the cost basis at no tax - worth up to ₹15,625 a year at 12.5% - and
an unused exemption does not carry forward. Nothing surfaced it.

WHAT THIS CANNOT KNOW, AND SAYS: the purchase date of each lot. Kite reports an
average price, not lots, so the unrealised gain here includes lots held under
12 months, which are short-term (20%) and do NOT qualify. The figure is an
upper bound on what can be harvested, labelled as one. Gold ETFs, debt funds
and fund-of-funds investing abroad are not equity-oriented and are excluded -
the exemption does not apply to them.

Pure: no Streamlit, no network, no file reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..portfolio.base import EQUITY, ETF, MUTUAL_FUND
from . import allocation as alloc

LTCG_EXEMPTION_INR = 125_000.0
LTCG_RATE = 0.125
STCG_RATE = 0.20
VERIFIED_ON = "2026-09-27"      # Finance Act 2024 rates, unchanged since


@dataclass(frozen=True)
class HarvestView:
    #: Unrealised gain on equity-oriented Indian lines with a cost basis.
    #: An UPPER BOUND - see the module docstring.
    gain_inr: float
    lines_with_cost: int
    #: Lines with no cost basis (typed values, CAS without cost) - their gain
    #: is unknown, not zero.
    lines_without_cost: int

    @property
    def harvestable_inr(self) -> float:
        return min(self.gain_inr, LTCG_EXEMPTION_INR)

    @property
    def tax_saved_inr(self) -> float:
        return self.harvestable_inr * LTCG_RATE


def harvest_view(snapshot) -> HarvestView:
    gain, with_cost, without = 0.0, 0, 0
    for p in snapshot.positions:
        if p.market != "india" or p.asset_class not in (EQUITY, ETF,
                                                          MUTUAL_FUND):
            continue
        if alloc.bucket_of(p) != alloc.INDIA_EQUITY:
            continue
        if p.cost_native <= 0:
            without += 1
            continue
        with_cost += 1
        gain += max(0.0, p.value_native - p.cost_native)
    return HarvestView(gain_inr=gain, lines_with_cost=with_cost,
                       lines_without_cost=without)


def fy_end(today: date) -> date:
    """31 March of the financial year `today` falls in."""
    return date(today.year + (1 if today.month > 3 else 0), 3, 31)


def harvest_season(today: date) -> bool:
    """January to March - the last quarter, when an unused exemption expires."""
    return today.month in (1, 2, 3)
