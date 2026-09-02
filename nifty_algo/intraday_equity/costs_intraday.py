"""
What an INTRADAY (MIS) equity round trip actually costs, in rupees.

The third rate card, and it is a third one because neither sibling is close.
[costs.py](../costs.py) is options - NFO transaction rates and a flat Rs 20
per order. [costs_equity.py](../swing/costs_equity.py) is cash DELIVERY. Using
the delivery model here would be wrong in two directions at once, and both
are large enough to change whether a trade was worth taking:

  STT IS SELL-SIDE ONLY, AND A QUARTER THE RATE. Delivery pays 0.1% on BOTH
  legs; intraday pays 0.025% on the sell alone. On a Rs 10,000 round trip
  that is Rs 2.50 against Rs 20 - the delivery model overcharges 8x here.

  THERE IS NO DP CHARGE AT ALL. The Rs 15.34-per-scrip-per-sell that
  dominates a small delivery ticket does not exist intraday, because nothing
  leaves the demat account. That single absence is worth ~3% of a Rs 500
  risk budget on a Rs 10,000 position.

  BUT BROKERAGE IS NO LONGER FREE. Delivery is zero-brokerage; intraday is
  0.03% or Rs 20 per executed order, WHICHEVER IS LOWER, on each leg. So the
  cost delivery avoids is the one that now scales with position size, and
  the cost delivery cannot escape is the one that vanishes.

Net effect: friction is roughly FLAT IN PERCENTAGE terms rather than
punishing small tickets the way delivery's fixed DP charge does. That sounds
like good news and it is not - an intraday book pays it several times a WEEK
on the same capital, where a swing book amortises a larger cost over a
multi-week hold. `friction_r` is the number that makes the two comparable.

Substituting `EquityCostModel` "to be safe" is not safe. It errs pessimistic,
which is the better direction, but it is a wrong number reported as a
measured one - roughly 0.07R of pure fiction on every trade at a Rs 500 risk
budget - and a book judged against fictional costs is not judged at all.

EVERY RATE IS DATE-STAMPED, following [crossborder.py](../swing/crossborder.py)
and the delivery sibling. These change; a rate with no date is a rate nobody
will ever re-check.

WHAT THIS DOES NOT MODEL: slippage. Deliberately - it is not a published
rate, it is a property of your fills. `slippage_pct` is a PRE-REGISTERED
ASSUMPTION until `book.record_fill` has measured twenty real ones, and it
must be printed above every backtest result rather than left buried here. It
matters more intraday than on a swing book: a 5-minute bar's open is a
thinner print than a daily trigger, and at a ~0.5% stop a 0.05% slip is
already 0.1R.
"""
from __future__ import annotations

from dataclasses import dataclass

#: When each rate below was last checked against zerodha.com/charges.
VERIFIED_ON = "2026-09-01"

#: Source for all of it, so a future reader re-checks rather than trusts.
SOURCE = "https://zerodha.com/charges/ (equity intraday column)"


@dataclass
class IntradayEquityCostModel:
    """
    Indian cash-equity INTRADAY (MIS) charges. Rupees in, rupees out.

    Mutable because these are tunables a sweep may want to move - notably
    `slippage_pct`, the only figure here that is not a published rate.
    """

    # Brokerage: 0.03% or Rs 20 per executed order, whichever is LOWER, per
    # leg. Delivery's is zero, which is why this reads as a new cost rather
    # than a changed one.
    brokerage_pct: float = 0.0003
    brokerage_cap: float = 20.0

    # Securities Transaction Tax: SELL SIDE ONLY, at 0.025% rather than
    # delivery's 0.1% on both legs. The single largest difference between
    # this file and its sibling.
    stt_sell_pct: float = 0.00025

    # NSE equity transaction charge, both legs - a function of the exchange
    # and segment, not the product, so it matches delivery exactly.
    transaction_pct: float = 0.0000307

    # Stamp duty, buy side only. 0.003% against delivery's 0.015%: a fifth,
    # because no shares are being transferred.
    stamp_duty_pct: float = 0.00003

    # SEBI turnover fee: Rs 10 per crore.
    sebi_pct: float = 0.000001

    # GST on (brokerage + transaction + SEBI). Not on STT, not on stamp duty.
    gst_pct: float = 0.18

    # THERE IS NO DP CHARGE FIELD, and its absence is load-bearing rather
    # than an oversight: nothing leaves the demat account intraday. If a
    # future edit adds one here, this has silently become a delivery model.

    #: PLACEHOLDER, not a measurement. See the module docstring.
    slippage_pct: float = 0.0005

    # ---------------- legs ----------------

    def _variable(self, turnover: float) -> float:
        """Charges that scale with turnover and attract GST."""
        brokerage = min(turnover * self.brokerage_pct, self.brokerage_cap)
        txn = turnover * self.transaction_pct
        sebi = turnover * self.sebi_pct
        return brokerage + txn + sebi + (brokerage + txn + sebi) * self.gst_pct

    def buy_cost(self, price: float, quantity: float) -> float:
        turnover = price * quantity
        return self._variable(turnover) + turnover * self.stamp_duty_pct

    def sell_cost(self, price: float, quantity: float) -> float:
        """
        STT lands here and only here.

        No DP charge, so unlike the delivery model this is not dearer than
        the buy leg by a fixed amount - which is why banking a partial
        intraday is cheap, where splitting a delivery exit across two
        sessions is not.
        """
        turnover = price * quantity
        return self._variable(turnover) + turnover * self.stt_sell_pct

    def round_trip(self, entry: float, exit_price: float,
                   quantity: float) -> float:
        return (self.buy_cost(entry, quantity)
                + self.sell_cost(exit_price, quantity))

    def slippage(self, price: float, quantity: float) -> float:
        return price * quantity * self.slippage_pct

    # ---------------- what a reader actually wants to know ----------------

    def friction(self, entry: float, exit_price: float, quantity: float,
                 include_slippage: bool = True) -> float:
        """Everything the round trip costs, charges plus modelled slippage."""
        total = self.round_trip(entry, exit_price, quantity)
        if include_slippage:
            total += self.slippage(entry, quantity)
            total += self.slippage(exit_price, quantity)
        return total

    def friction_r(self, entry: float, stop: float, quantity: float,
                   include_slippage: bool = True) -> float:
        """
        Round-trip friction expressed in R. THE number for this book.

        A cost of "0.12% of turnover" sounds like nothing. The same cost as
        "0.13R off every trade" is immediately comparable to the 2R the
        ladder is trying to earn - and to the fact that this book pays it
        two or three times a week rather than twice a month.
        """
        risk = (entry - stop) * quantity
        if risk <= 0:
            return 0.0
        return self.friction(entry, entry, quantity, include_slippage) / risk

    def breakeven_move_pct(self, price: float, quantity: float) -> float:
        """How far the share must rise before you are square."""
        turnover = price * quantity
        if turnover <= 0:
            return 0.0
        return self.friction(price, price, quantity) / turnover

    def note(self, entry: float, stop: float, quantity: float) -> str:
        """One line for a ticket card."""
        cost = self.friction(entry, entry, quantity)
        return (f"Round trip ~₹{cost:,.0f} "
                f"({self.friction_r(entry, stop, quantity):.2f}R, "
                f"{self.breakeven_move_pct(entry, quantity):.2%} of the "
                f"position) — MIS charges verified {VERIFIED_ON}")


DEFAULT_INTRADAY_EQUITY_COSTS = IntradayEquityCostModel()
