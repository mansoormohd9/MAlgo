"""
Cash-equity orders and GTT triggers for the swing book.

THE SECOND MODULE IN THIS PROJECT THAT CAN SPEND MONEY. Same three guards as
`kite_orders.py`, and none of them are optional:

  1. `EquityBrokerConfig.dry_run` defaults True. Every write logs the exact
     payload it WOULD have sent, journals it, and returns a synthetic id so
     the position book's state machine still runs. Nothing reaches Zerodha.

  2. LIMIT orders only. A GTT places a limit order when it triggers, so the
     limit is set PAST the trigger by `gtt_limit_buffer_pct` - above it on a
     buy, below on a sell. Zerodha's own wording is that this "acts like a
     market order with the protection of your limit set". An exact-price limit
     on a fast move does not fill, and a GTT whose order does not fill that day
     is CANCELLED rather than retried - so the stop simply ceases to exist.

  3. CNC only. This book holds for days. MIS would be squared off by the
     broker's own risk system at 15:20, turning every swing trade into an
     intraday one without telling you.

WHY THE STOP RESTS AT THE BROKER HERE AND DELIBERATELY DOES NOT IN
`kite_orders.modify_stop()`. That module trails an option stop on the
underlying's ATR, recomputed every 5-minute bar; a resting order would need
modifying on every one of them, each modify a request that can fail, be
rejected, or race the fill. A swing stop moves at most once a day. So here the
resting GTT is cheap, and it is the only version of the stop that survives the
laptop being shut. The two books reach opposite conclusions from the same
reasoning, which is why they are two files rather than one with a flag.

THE CONSTRAINT YOU MUST KNOW ABOUT: CDSL TPIN.

Unless DDPI (or the older POA) is active on the account, EVERY delivery sell
needs a CDSL TPIN authorisation - not just GTT, any CNC sell - and that
authorisation is valid for ONE TRADING DAY. A stop GTT placed on Monday is
rejected on Wednesday unless the account was re-authorised that morning after
07:00. It still displays as active in Kite either way.

Kite Connect exposes no endpoint for this, so it cannot be automated from
here. This module therefore refuses to describe a stop as protected on a day
the authorisation has not been recorded - see `protection_state()`. A stop
that looks armed and is not is worse than no stop, because you stop watching.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from ..config import Config, DEFAULT
from ._transport import BrokerTransport
from .kite_auth import KiteSession

log = logging.getLogger(__name__)

#: Kite's GTT vocabulary. Spelled out rather than read off the SDK so this
#: module imports with `kiteconnect` absent - the dry-run path, the tests and
#: the whole position book work with no broker library installed.
GTT_SINGLE = "single"
GTT_OCO = "two-leg"

#: Journal event recording that holdings were authorised for a given day. The
#: app cannot verify this with Kite; it records what you told it, and the
#: record expires with the trading day exactly as the authorisation does.
EVENT_AUTHORISED = "holdings_authorised"

#: What `protection_state()` can say. Three states, not two - "we cannot know"
#: is a real and common answer here and must not collapse into either of the
#: other two.
PROTECTED = "protected"
UNPROTECTED = "unprotected"
UNVERIFIED = "unverified"


@dataclass
class GttView:
    """One trigger as Zerodha reports it, flattened to what we reconcile on."""
    trigger_id: str
    symbol: str
    exchange: str
    status: str
    trigger_type: str
    trigger_values: list[float]
    quantity: float
    transaction_type: str

    @property
    def is_active(self) -> bool:
        return str(self.status).lower() == "active"

    @property
    def is_exit(self) -> bool:
        return self.transaction_type.upper() == "SELL"

    @property
    def stop(self) -> Optional[float]:
        """Lower leg of an OCO, or the only trigger on a stop-only GTT."""
        return min(self.trigger_values) if self.trigger_values else None

    @property
    def target(self) -> Optional[float]:
        if len(self.trigger_values) < 2:
            return None
        return max(self.trigger_values)


@dataclass
class HoldingView:
    """One demat holding, flattened."""
    symbol: str
    exchange: str
    quantity: float
    t1_quantity: float
    average_price: float
    last_price: float
    pnl: float

    @property
    def total_quantity(self) -> float:
        """
        What you own, settled or not.

        T+1 shares are yours and are sellable, but Kite reports them in a
        separate field. Reading only `quantity` on the day after a fill shows
        zero, which reads exactly like "the buy never happened" - and would
        make the reconciler delete a live position's stop.
        """
        return self.quantity + self.t1_quantity


class KiteEquity(BrokerTransport):
    def __init__(self, session: KiteSession | None = None,
                 cfg: Config = DEFAULT, journal=None):
        self.session = session or KiteSession()
        self.cfg = cfg
        self.journal = journal
        self.placed: list[dict] = []

    # ---------------- mode ----------------

    @property
    def eq(self):
        return self.cfg.equity_broker

    @property
    def dry_run(self) -> bool:
        return self.eq.dry_run

    @property
    def mode_label(self) -> str:
        return "DRY RUN - no orders sent" if self.dry_run else "LIVE - real money"

    # ---------------- prices ----------------

    def _tick(self, price: float) -> float:
        t = self.eq.tick_size
        return round(round(price / t) * t, 2)

    def buy_limit_for(self, trigger: float) -> float:
        """Limit ABOVE the trigger, so a fast breakout still fills."""
        return self._tick(trigger * (1.0 + self.eq.gtt_limit_buffer_pct))

    def sell_limit_for(self, trigger: float) -> float:
        """Limit BELOW the trigger. An unfilled exit is the worse failure."""
        return self._tick(max(trigger * (1.0 - self.eq.gtt_limit_buffer_pct),
                              self.eq.tick_size))

    # ---------------- reads ----------------
    # None of these are gated by dry_run. See `_transport._read` - suppressing
    # facts in the safest mode would leave it the least informed.

    def holdings(self) -> list[HoldingView]:
        raw = self._read("holdings", lambda k: k.holdings(), [])
        return [_holding(r) for r in raw or []]

    def positions(self) -> dict:
        empty = {"net": [], "day": []}
        return self._read("positions", lambda k: k.positions(), empty) or empty

    def orders(self) -> list[dict]:
        return self._read("orders", lambda k: k.orders(), []) or []

    def order_history(self, order_id: str) -> list[dict]:
        return self._read("order_history",
                          lambda k: k.order_history(order_id), []) or []

    def trades(self) -> list[dict]:
        return self._read("trades", lambda k: k.trades(), []) or []

    def gtts(self) -> list[GttView]:
        raw = self._read("gtts", lambda k: k.get_gtts(), [])
        return [_gtt(r) for r in raw or []]

    def free_cash(self) -> Optional[float]:
        """
        Withdrawable equity balance, or None when it could not be read.

        None is not zero, and the distinction is load-bearing. A GTT blocks no
        margin at placement - Zerodha checks funds only when it TRIGGERS - so
        arming three buys against an empty account produces three rejections
        days later, at the exact moment the setups were working. This is the
        number that prevents that, and defaulting it to 0 or to a guess would
        defeat the purpose in opposite directions.
        """
        m = self._read("margins", lambda k: k.margins("equity"), None)
        if not isinstance(m, dict):
            return None
        available = m.get("available") or {}
        for key in ("live_balance", "cash", "opening_balance"):
            if key in available:
                try:
                    return float(available[key])
                except (TypeError, ValueError):
                    continue
        try:
            net = m.get("net")
            return float(net) if net is not None else None
        except (TypeError, ValueError):
            return None

    # ---------------- TPIN / DDPI ----------------

    def record_authorisation(self, day: Optional[date] = None) -> None:
        """
        Note that you authorised holdings today.

        Recorded, never verified - Kite Connect has no endpoint for it. That
        makes this a claim by you rather than a fact, which is exactly why
        `protection_state()` returns UNVERIFIED and not PROTECTED on it.
        """
        day = day or date.today()
        if self.journal is not None:
            self.journal.write(EVENT_AUTHORISED, {"day": day.isoformat()},
                               day=day)

    def authorised_on(self, day: Optional[date] = None) -> bool:
        day = day or date.today()
        if self.journal is None:
            return False
        return any(r.get("event") == EVENT_AUTHORISED
                   for r in self.journal.read_day(day))

    def protection_state(self, day: Optional[date] = None) -> tuple[str, str]:
        """
        Whether an exit GTT would actually execute today. Returns `(state, why)`.

        The single most important thing this module reports, and it cannot
        return PROTECTED without DDPI. With DDPI, Zerodha sells on your behalf
        and a resting stop is a real stop. Without it the sell needs a TPIN
        authorisation that dies every evening, so the most this can honestly
        say is UNVERIFIED - and it says UNPROTECTED outright the moment today's
        authorisation is missing.
        """
        if self.eq.ddpi_active:
            return PROTECTED, "DDPI is active - a resting exit sells without you."
        if self.authorised_on(day):
            return UNVERIFIED, (
                "Holdings were marked authorised today, but Kite Connect "
                "cannot confirm it, and the authorisation expires tonight."
            )
        return UNPROTECTED, (
            "No DDPI, and today's CDSL TPIN authorisation has not been "
            "recorded. A sell GTT will show as active in Kite and be REJECTED "
            "when it triggers. Authorise holdings in Kite, or enable DDPI once "
            "and stop doing this every morning."
        )

    # ---------------- writes: GTT ----------------

    def place_buy_gtt(self, symbol: str, trigger: float, quantity: float,
                      last_price: float) -> Optional[str]:
        """
        Arm an entry. Single-leg BUY trigger, good at Zerodha for a year.

        This is what makes "one click, whenever convenient" work. Every setup
        in this book puts its entry ABOVE the last close, because each one
        demands the stock prove itself before you pay for it - so the order
        must WAIT rather than fill now. Zerodha does the waiting. The app does
        not need to be running, or even installed, between arming and the fill.
        """
        qty = int(quantity)
        if qty <= 0:
            log.error("refusing a buy GTT for %s: %s shares", symbol, quantity)
            return None
        payload = {
            "trigger_type": GTT_SINGLE,
            "tradingsymbol": symbol,
            "exchange": self.eq.exchange,
            "trigger_values": [self._tick(trigger)],
            "last_price": self._tick(last_price),
            "orders": [self._leg("BUY", qty, self.buy_limit_for(trigger))],
        }
        return self._gtt_call("gtt_buy:" + symbol, payload,
                              lambda k: k.place_gtt(**payload))

    def place_exit_gtt(self, symbol: str, quantity: float, stop: float,
                       target: float, last_price: float) -> Optional[str]:
        """
        Arm the exit. Two-leg OCO: stop below, target above, either cancels
        the other.

        Placed AFTER the buy fills, never before. A buy GTT cannot chain into
        a sell GTT, and a sell trigger on shares you do not yet own would fire
        into a short you never asked for.
        """
        qty = int(quantity)
        if qty <= 0:
            return None
        if not stop < target:
            log.error("refusing an OCO for %s: stop %.2f is not below target "
                      "%.2f", symbol, stop, target)
            return None
        payload = {
            "trigger_type": GTT_OCO,
            "tradingsymbol": symbol,
            "exchange": self.eq.exchange,
            # Ascending, and the legs below must be in the same order.
            "trigger_values": [self._tick(stop), self._tick(target)],
            "last_price": self._tick(last_price),
            "orders": [
                self._leg("SELL", qty, self.sell_limit_for(stop)),
                self._leg("SELL", qty, self.sell_limit_for(target)),
            ],
        }
        return self._gtt_call("gtt_exit:" + symbol, payload,
                              lambda k: k.place_gtt(**payload))

    def modify_exit_gtt(self, trigger_id: str, symbol: str, quantity: float,
                        stop: float, target: float,
                        last_price: float) -> Optional[str]:
        """
        Move a resting exit - this is how the trailing stop reaches Zerodha.

        Modify, never delete-then-place: between a delete and a place there is
        a window with no stop at all, and that window is precisely when you
        would least like one.
        """
        qty = int(quantity)
        if qty <= 0 or not stop < target:
            return None
        payload = {
            "trigger_id": trigger_id,
            "trigger_type": GTT_OCO,
            "tradingsymbol": symbol,
            "exchange": self.eq.exchange,
            "trigger_values": [self._tick(stop), self._tick(target)],
            "last_price": self._tick(last_price),
            "orders": [
                self._leg("SELL", qty, self.sell_limit_for(stop)),
                self._leg("SELL", qty, self.sell_limit_for(target)),
            ],
        }
        return self._gtt_call("gtt_modify:" + symbol, payload,
                              lambda k: k.modify_gtt(**payload))

    def modify_buy_gtt(self, trigger_id: str, symbol: str, trigger: float,
                       quantity: float, last_price: float) -> Optional[str]:
        """Re-price an unfilled entry onto today's fresh numbers."""
        qty = int(quantity)
        if qty <= 0:
            return None
        payload = {
            "trigger_id": trigger_id,
            "trigger_type": GTT_SINGLE,
            "tradingsymbol": symbol,
            "exchange": self.eq.exchange,
            "trigger_values": [self._tick(trigger)],
            "last_price": self._tick(last_price),
            "orders": [self._leg("BUY", qty, self.buy_limit_for(trigger))],
        }
        return self._gtt_call("gtt_rearm:" + symbol, payload,
                              lambda k: k.modify_gtt(**payload))

    def delete_gtt(self, trigger_id: str) -> Optional[str]:
        payload = {"trigger_id": trigger_id}
        return self._gtt_call("gtt_delete:" + str(trigger_id), payload,
                              lambda k: k.delete_gtt(trigger_id))

    # ---------------- writes: plain orders ----------------

    def place_market_exit(self, symbol: str, quantity: float,
                          last_price: float) -> Optional[str]:
        """
        Close a position now, at a protective limit rather than at market.

        For the case the ladder cannot express: you have decided to be out.
        Still LIMIT, still CNC - "get me out" is not a reason to go and find
        out what the far side of a thin book looks like.
        """
        qty = int(quantity)
        if qty <= 0:
            return None
        payload = {
            "variety": self.eq.variety,
            "exchange": self.eq.exchange,
            "tradingsymbol": symbol,
            "transaction_type": "SELL",
            "quantity": qty,
            "product": self.eq.product,
            "order_type": self.eq.order_type,
            "price": self.sell_limit_for(last_price),
            "validity": "DAY",
            "tag": self.eq.tag,
        }
        return self._call("exit_now:" + symbol, payload,
                          lambda k: k.place_order(**payload))

    # ---------------- helpers ----------------

    def _gtt_call(self, what: str, payload: dict, invoke) -> Optional[str]:
        """
        Place/modify/delete a GTT and return its trigger id as a string.

        THE SHAPES DIFFER AND IT MATTERS. pykiteconnect's `place_order` ends
        `...["order_id"]` and hands back a bare id; `place_gtt`, `modify_gtt`
        and `delete_gtt` return the RAW response, which is `{"trigger_id": N}`.
        Storing `str(result)` would put the literal text `{'trigger_id': 123}`
        in the ledger, so no id would ever match one from `get_gtts()` - and
        the run would re-place every exit OCO on every pass while believing
        it had none, and never actually delete an expired buy trigger.
        """
        result = self._call(what, payload, invoke)
        if result is None:
            return None
        if isinstance(result, dict):
            trigger_id = result.get("trigger_id")
            return str(trigger_id) if trigger_id is not None else None
        return str(result)          # dry run hands back a synthetic id

    def _leg(self, side: str, quantity: int, price: float) -> dict:
        return {
            "transaction_type": side,
            "quantity": quantity,
            "order_type": self.eq.order_type,
            "product": self.eq.product,
            "price": price,
        }


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _holding(raw: dict) -> HoldingView:
    return HoldingView(
        symbol=str(raw.get("tradingsymbol", "")).upper(),
        exchange=str(raw.get("exchange", "")),
        quantity=_num(raw.get("quantity")),
        t1_quantity=_num(raw.get("t1_quantity")),
        average_price=_num(raw.get("average_price")),
        last_price=_num(raw.get("last_price")),
        pnl=_num(raw.get("pnl")),
    )


def _gtt(raw: dict) -> GttView:
    """
    Flatten Kite's nested GTT record.

    Symbol and quantity sit in different places depending on how much of the
    trigger Zerodha has filled in, so both `condition` and the first order leg
    are consulted rather than trusting either one alone.
    """
    cond = raw.get("condition") or {}
    orders = raw.get("orders") or []
    first = orders[0] if orders else {}
    values = [_num(v) for v in (cond.get("trigger_values") or [])]
    return GttView(
        trigger_id=str(raw.get("id", "")),
        symbol=str(cond.get("tradingsymbol")
                   or first.get("tradingsymbol") or "").upper(),
        exchange=str(cond.get("exchange") or first.get("exchange") or ""),
        status=str(raw.get("status", "")),
        trigger_type=str(raw.get("type", "")),
        trigger_values=values,
        quantity=_num(first.get("quantity")),
        transaction_type=str(first.get("transaction_type", "")).upper(),
    )
