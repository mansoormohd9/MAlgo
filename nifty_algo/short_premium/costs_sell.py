"""
Option-SELLING friction. The same rate table as `costs.py`, the legs reversed.

`costs.CostModel` is correct arithmetic wired for a buyer:
`entry_friction()` charges stamp duty and `exit_friction()` charges STT,
because a buyer buys first. A seller sells first, so:

  STT (0.1% of premium turnover) is paid at ENTRY, on the credit received.
  Stamp duty (0.003%) is paid at EXIT, on the buy-back.

That is not a rounding difference. On a Rs 12 credit over one 65-lot the STT
alone is ~Rs 0.78 at entry; charged at exit on a buy-back near zero it would
be ~Rs 0.00. Getting the leg wrong understates a seller's cost by most of its
largest statutory component, and it understates it MORE the better the trade
went - which is the direction that flatters a backtest.

WHY THIS IS NOT A SECOND RATE TABLE. Every rate lives in `CostModel` and is
read from it here. This module contributes leg ORDER and nothing else; if
Zerodha change a rate there is still exactly one place to change it.

NOT MODELLED, AND IT MATTERS: STT on EXERCISED ITM options at expiry is
0.125% of the SETTLEMENT value, not of the premium - roughly two orders of
magnitude larger, and charged on the short who let a strike go in the money.
This book is flat by 15:10 and never carries to expiry, which is the only
reason the omission is safe. If that ever changes, this is the first thing
that must change with it.
"""
from __future__ import annotations

from ..costs import CostModel, DEFAULT_COSTS


def entry_friction(credit: float, quantity: int,
                   costs: CostModel = DEFAULT_COSTS) -> float:
    """The SELL leg: brokerage, txn, SEBI, IPFT, GST and STT on the credit."""
    return (costs.leg(credit * quantity, is_sell=True)
            + costs.slippage(quantity, legs=1))


def exit_friction(debit: float, quantity: int,
                  costs: CostModel = DEFAULT_COSTS) -> float:
    """The BUY-BACK leg: the same, with stamp duty instead of STT."""
    return (costs.leg(debit * quantity, is_sell=False)
            + costs.slippage(quantity, legs=1))


def round_trip(credit: float, buyback: float, quantity: int,
               costs: CostModel = DEFAULT_COSTS) -> float:
    """Sell at `credit`, buy back at `buyback`. Total rupees of friction."""
    return (entry_friction(credit, quantity, costs)
            + exit_friction(buyback, quantity, costs))


def scratch_cost_per_unit(credit: float, quantity: int,
                          costs: CostModel = DEFAULT_COSTS) -> float:
    """
    What one unit costs to open and close with no move at all, in premium
    points. This is the number `ShortPremiumConfig.min_credit_to_cost_multiple`
    is measured against.

    Buying back at the same price you sold is the honest reference: it is the
    cost of being wrong about nothing. Referencing a buy-back at zero would
    flatter every strike by pretending the exit is free, and it is precisely
    the far-OTM strikes - the ones a seller is tempted by - where the fixed
    Rs 20 brokerage and two ticks of slippage dwarf the credit.
    """
    if quantity <= 0:
        return float("inf")
    return round_trip(credit, credit, quantity, costs) / quantity
