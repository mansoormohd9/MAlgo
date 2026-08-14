"""
Order placement through Kite Connect.

THE ONLY MODULE IN THIS PROJECT THAT CAN SPEND MONEY.

Three guards, and none of them are optional:

  1. `BrokerConfig.dry_run` defaults to True. In dry-run every method logs the
     exact payload it WOULD have sent and returns a synthetic order id. Nothing
     reaches Zerodha. Flipping it is a deliberate edit to config, never a side
     effect of anything else.

  2. LIMIT orders only. A MARKET order on an option book is how you discover
     what the far side of a thin spread looks like. The limit crosses the
     spread by `limit_buffer_ticks` so it still fills, but the worst case is
     bounded and known before the order goes out.

  3. MIS product only. This system is intraday and flat by 15:10 - see the
     force-exit rule. NRML would let a position survive the close, which is
     the one thing the strategy is not designed to survive.

COMPLIANCE. SEBI's retail algo framework requires strategies placing automated
orders to be registered with the broker and routed under an Algo ID, with a
static IP whitelist. `confirm_entry()` in engine.py is human-initiated, which
is why this is built as one-click confirmation and not autonomous execution.
Check your own obligations with Zerodha before turning dry_run off - the README
carries the same warning.
"""
from __future__ import annotations
import logging
from datetime import date
from typing import Optional

from ..config import Config, DEFAULT
from ..positions import ExitAction, ManagedPosition
from ..risk import ApprovedOrder
from .kite_auth import KiteSession, NotAuthenticated

log = logging.getLogger(__name__)


class KiteOrders:
    def __init__(self, session: KiteSession | None = None, cfg: Config = DEFAULT,
                 chain=None, journal=None):
        self.session = session or KiteSession()
        self.cfg = cfg
        self.chain = chain            # KiteChain, for resolving tradingsymbols
        self.journal = journal
        self.placed: list[dict] = []

    @property
    def dry_run(self) -> bool:
        return self.cfg.broker.dry_run

    @property
    def mode_label(self) -> str:
        return "DRY RUN — no orders sent" if self.dry_run else "LIVE — real money"

    # ---------------- symbol resolution ----------------

    def tradingsymbol_for(self, strike: int, option_type: str,
                          expiry: Optional[date] = None) -> Optional[str]:
        """
        The exchange symbol for a strike, from Kite's instrument dump.

        Constructing it by string formatting is a trap: NSE's weekly and
        monthly option symbol formats differ from each other and have both been
        changed. The dump is the authority.
        """
        if self.chain is None:
            return None
        try:
            df = self.chain.instruments()
            expiry = expiry or self.chain.nearest_expiry()
            match = df[(df["strike"] == strike)
                       & (df["instrument_type"] == option_type)
                       & (df["expiry"] == expiry)]
            if match.empty:
                return None
            return str(match.iloc[0]["tradingsymbol"])
        except Exception as e:
            log.warning("tradingsymbol lookup failed: %s", e)
            return None

    # ---------------- entry ----------------

    def place_entry(self, order: ApprovedOrder, alert=None) -> Optional[str]:
        """
        Buy the approved contract. Returns the tradingsymbol on success.

        The limit price crosses the spread upward by `limit_buffer_ticks`,
        because an unfilled entry on a breakout is worse than a tick of
        slippage - but the exposure is capped and visible in the payload.
        """
        b = self.cfg.broker
        symbol = self.tradingsymbol_for(order.quote.strike,
                                        order.quote.option_type)
        if symbol is None:
            log.error("no tradingsymbol for %s%s - order NOT placed",
                      order.quote.strike, order.quote.option_type)
            return None

        tick = self.cfg.instrument.tick_size
        limit = order.entry_premium + b.limit_buffer_ticks * tick

        payload = {
            "variety": b.variety,
            "exchange": b.exchange,
            "tradingsymbol": symbol,
            "transaction_type": "BUY",
            "quantity": order.quantity,
            "product": b.product,
            "order_type": b.order_type,
            "price": round(limit / tick) * tick,
            "tag": "niftyalgo",
        }
        return symbol if self._send(payload, "entry") else None

    # ---------------- exit ----------------

    def exit_position(self, pos: ManagedPosition, action: ExitAction) -> bool:
        """
        Sell `action.lots` lots. Called by the engine for every exit rung -
        the partial at +2R as well as the final close.
        """
        b = self.cfg.broker
        symbol = pos.tradingsymbol or self.tradingsymbol_for(
            pos.quote.strike, pos.quote.option_type)
        if symbol is None:
            log.error("no tradingsymbol for %s%s - EXIT NOT PLACED. "
                      "Close this position manually.",
                      pos.quote.strike, pos.quote.option_type)
            return False

        tick = self.cfg.instrument.tick_size
        # Exits cross DOWNWARD - the same reasoning, opposite direction. An
        # unfilled exit is strictly worse than an unfilled entry: it leaves
        # you in a trade the system believes it has already left.
        limit = max(action.premium - b.limit_buffer_ticks * tick, tick)

        payload = {
            "variety": b.variety,
            "exchange": b.exchange,
            "tradingsymbol": symbol,
            "transaction_type": "SELL",
            "quantity": action.quantity,
            "product": b.product,
            "order_type": b.order_type,
            "price": round(limit / tick) * tick,
            "tag": "niftyalgo",
        }
        return self._send(payload, f"exit:{action.kind.value}")

    def modify_stop(self, pos: ManagedPosition, new_stop_premium: float) -> bool:
        """
        Record a stop move.

        NOT sent to the broker as a resting SL order, deliberately. The stop
        here trails on the UNDERLYING's ATR, so its premium level is recomputed
        every bar; a resting stop-loss order would have to be modified on every
        one of those bars, and each modify is a request that can fail, be
        rejected, or race the fill. The engine holds the stop and issues a
        market-side SELL when it triggers - which means the engine must be
        RUNNING for the stop to exist. That is a real operational dependency
        and you should know it rather than discover it.
        """
        self._log("stop_moved", {
            "strike": pos.quote.strike,
            "option_type": pos.quote.option_type,
            "new_stop": round(new_stop_premium, 2),
        })
        return True

    # ---------------- transport ----------------

    def _send(self, payload: dict, what: str) -> bool:
        self.placed.append({"what": what, **payload})

        if self.dry_run:
            log.info("DRY RUN %s: %s", what, payload)
            self._log("order_dry_run", {"what": what, "payload": payload})
            return True

        try:
            kite = self.session.client()
            order_id = kite.place_order(**payload)
        except NotAuthenticated as e:
            log.error("%s NOT placed - not authenticated: %s", what, e)
            self._log("order_failed", {"what": what, "payload": payload,
                                       "error": str(e)})
            return False
        except Exception as e:
            log.error("%s NOT placed: %s", what, e)
            self._log("order_failed", {"what": what, "payload": payload,
                                       "error": f"{type(e).__name__}: {e}"})
            return False

        log.info("%s placed, order_id=%s", what, order_id)
        self._log("order_placed", {"what": what, "payload": payload,
                                   "order_id": order_id})
        return True

    def _log(self, event: str, payload: dict) -> None:
        if self.journal is not None:
            try:
                self.journal.write(event, payload)
            except Exception:
                pass          # journalling must never break an order path
