"""
Indian option-buying cost model.

A backtest that ignores these is fiction. On a ~Rs 8,000 option position,
round-trip friction is typically Rs 70-110, i.e. ~1% - which is more than
half your per-trade risk budget.

RATES CHANGE. Verify against your broker's charges page each quarter.
Defaults below reflect NSE index options for a discount broker.
"""
from dataclasses import dataclass


@dataclass
class CostModel:
    brokerage_per_order: float = 20.0       # flat, per executed order
    stt_sell_pct: float = 0.001             # 0.1% on sell-side premium
    exchange_txn_pct: float = 0.0003503     # NSE options, both sides
    sebi_charges_pct: float = 0.000001      # Rs 10 per crore
    ipft_pct: float = 0.000005              # NSE investor protection fund
    stamp_duty_buy_pct: float = 0.00003     # 0.003% on buy side
    gst_pct: float = 0.18                   # on brokerage + txn + sebi

    # Execution friction - usually larger than all charges combined.
    # Expressed in ticks of adverse fill vs your decision price.
    slippage_ticks: float = 2.0
    tick_size: float = 0.05

    def _leg(self, turnover: float, is_sell: bool) -> float:
        brokerage = self.brokerage_per_order
        txn = turnover * self.exchange_txn_pct
        sebi = turnover * self.sebi_charges_pct
        ipft = turnover * self.ipft_pct
        gst = (brokerage + txn + sebi) * self.gst_pct

        stt = turnover * self.stt_sell_pct if is_sell else 0.0
        stamp = 0.0 if is_sell else turnover * self.stamp_duty_buy_pct

        return brokerage + txn + sebi + ipft + gst + stt + stamp

    def leg(self, turnover: float, is_sell: bool) -> float:
        """
        One leg's statutory + brokerage cost. Public because the SHORT
        PREMIUM book pays these same rates in the opposite leg order - it
        sells first, so STT lands on entry - and the alternative was either
        a second copy of the rate table or a reach into `_leg` from another
        package. Pure delegation: no behaviour here, and nothing in this
        module calls it.
        """
        return self._leg(turnover, is_sell)

    def round_trip(self, entry_premium: float, exit_premium: float,
                   quantity: int) -> float:
        """Total statutory + brokerage cost for a full buy->sell cycle."""
        buy_turnover = entry_premium * quantity
        sell_turnover = exit_premium * quantity
        return self._leg(buy_turnover, is_sell=False) + \
               self._leg(sell_turnover, is_sell=True)

    def slippage(self, quantity: int, legs: int = 2) -> float:
        """Adverse fill cost, in rupees, across entry and exit."""
        return self.slippage_ticks * self.tick_size * quantity * legs

    def total_friction(self, entry_premium: float, exit_premium: float,
                       quantity: int) -> float:
        return (self.round_trip(entry_premium, exit_premium, quantity)
                + self.slippage(quantity))

    # ---- per-leg, for positions that exit in more than one piece ----
    #
    # A runner that banks half at +2R and trails the rest is ONE buy and TWO
    # sells, so it pays the flat brokerage three times, not twice, plus an
    # extra slippage leg. That is a real cost of scaling out and the backtest
    # would overstate the runner's benefit if it charged a single round trip.

    def entry_friction(self, entry_premium: float, quantity: int) -> float:
        return (self._leg(entry_premium * quantity, is_sell=False)
                + self.slippage(quantity, legs=1))

    def exit_friction(self, exit_premium: float, quantity: int) -> float:
        return (self._leg(exit_premium * quantity, is_sell=True)
                + self.slippage(quantity, legs=1))

    def breakeven_move(self, entry_premium: float, quantity: int) -> float:
        """
        How far the premium must move in your favour just to break even.
        Print this before every strategy goes live - it is sobering.
        """
        friction = self.total_friction(entry_premium, entry_premium, quantity)
        return friction / quantity


DEFAULT_COSTS = CostModel()
