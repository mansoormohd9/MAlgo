"""
What a delivery round trip actually costs, in rupees.

[costs.py](../costs.py) is the OPTION cost model - STT on the sell side only,
NFO transaction rates, brokerage per executed order. Every one of those numbers
is wrong for cash equity, and two of the differences are large enough to change
whether a trade was worth taking:

  STT IS CHARGED ON BOTH LEGS, at 0.1% each. The option book pays it once.
  On a Rs 10,000 position that is Rs 20 rather than Rs 10.

  THERE IS A FLAT DP CHARGE ON EVERY SELL. Rs 15.34 per scrip per day,
  regardless of quantity. It is the only fixed cost here, and on a small
  ticket it dominates: on Rs 10,000 with Rs 500 of risk it is 3% of the risk
  budget on its own, and it does not shrink when the position does. This is
  precisely the charge a percentage-based model cannot see.

Together the round trip on a Rs 10,000 delivery ticket is roughly Rs 45 -
about 9% of a Rs 500 risk budget, or put another way, the trade starts about
0.45% underwater. `SwingSetup.reward_risk` of 2.0 is really about 1.9 once
this is paid, and a scan that ranks on R:R without it is ranking on a number
nobody receives.

EVERY RATE IS DATE-STAMPED, following [crossborder.py](crossborder.py). These
change - stamp duty was harmonised in 2020, exchange transaction charges moved
in 2024, and STT on delivery has been revised twice. A rate with no date is a
rate nobody will ever re-check.

WHAT THIS DOES NOT MODEL: slippage. That is deliberate - it is not a published
rate, it is a property of your fills, and `book.record_fill` measures it on
every trade so that after twenty of them you have a number instead of an
assumption. `SLIPPAGE_PCT` below is only the placeholder the backtest uses
until that measurement exists, and it says so where it surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass

#: When each rate below was last checked against zerodha.com/charges.
VERIFIED_ON = "2026-08-24"

#: Source for all of it, so a future reader re-checks rather than trusts.
SOURCE = "https://zerodha.com/charges/ (equity delivery column)"


@dataclass
class EquityCostModel:
    """
    Indian cash-equity delivery charges. Rupees in, rupees out.

    Mutable dataclass rather than module constants because these are tunables
    and invariant #3 says tunables are overridable - a different broker or a
    female account holder's lower DP charge is a field, not a fork.
    """

    # --- brokerage ---
    #: Zerodha charges nothing for delivery. Kept as a field because a second
    #: broker would, and because zero is a value worth stating rather than a
    #: line worth omitting.
    brokerage_pct: float = 0.0
    brokerage_cap: float = 20.0

    # --- statutory, both legs ---
    stt_pct: float = 0.001                 # 0.1% on BUY and on SELL

    # --- statutory, one leg ---
    stamp_duty_pct: float = 0.00015        # 0.015%, buy side only

    # --- exchange and regulator ---
    transaction_pct: float = 0.0000307     # NSE 0.00307%
    sebi_pct: float = 0.000001             # Rs 10 per crore
    gst_pct: float = 0.18                  # on brokerage + txn + SEBI

    # --- the flat one ---
    #: Rs 15.34 = Rs 3.5 CDSL + Rs 9.5 Zerodha + Rs 2.34 GST. Per scrip, per
    #: day, on the sell side, irrespective of quantity. GST already included,
    #: so it is NOT run through `gst_pct` again.
    dp_charge: float = 15.34

    #: Placeholder only, until `book.record_fill`'s measured figure replaces
    #: it. A GTT crosses the spread by design (see `kite_equity`), so zero
    #: would be the one clearly wrong answer.
    slippage_pct: float = 0.0005           # 0.05% per leg

    # ---------------- legs ----------------

    def _variable(self, turnover: float) -> float:
        """Charges that scale with turnover and attract GST."""
        brokerage = min(turnover * self.brokerage_pct, self.brokerage_cap)
        txn = turnover * self.transaction_pct
        sebi = turnover * self.sebi_pct
        return brokerage + txn + sebi + (brokerage + txn + sebi) * self.gst_pct

    def buy_cost(self, price: float, quantity: float) -> float:
        turnover = price * quantity
        return (self._variable(turnover)
                + turnover * self.stt_pct
                + turnover * self.stamp_duty_pct)

    def sell_cost(self, price: float, quantity: float) -> float:
        """
        Includes the flat DP charge.

        Charged per scrip per day, so a scale-out that sells the same stock on
        two different days pays it twice - which is a real, if small, argument
        against splitting an exit across sessions on a small ticket.
        """
        turnover = price * quantity
        return (self._variable(turnover)
                + turnover * self.stt_pct
                + self.dp_charge)

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
        Round-trip friction expressed in R.

        THE number for this book. A cost of "0.45% of turnover" sounds like
        nothing; the same cost as "0.09R off every trade" is immediately
        comparable to the 2R the ladder is trying to earn, and to the 33%
        breakeven win rate a 2:1 payoff implies.
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
                f"position) — charges verified {VERIFIED_ON}")


DEFAULT_EQUITY_COSTS = EquityCostModel()
