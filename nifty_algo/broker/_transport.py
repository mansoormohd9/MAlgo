"""
The shared plumbing under every order this project can send.

Two modules place orders - `kite_orders.py` (NFO options, intraday) and
`kite_equity.py` (NSE cash, held for days). They disagree about almost
everything: product, exchange, whether the stop rests at the broker, whether
quantity is a lot multiple. They must NOT disagree about any of this:

  * `dry_run` short-circuits before the network, and returns a synthetic id so
    the caller's state machine still runs end to end. A dry run that returns
    None exercises only the failure path, which is the one path you least need
    rehearsed.
  * A broker error is caught, journalled and turned into a falsy return.
    Never an exception that escapes into a Streamlit rerun.
  * `NotAuthenticated` is distinguished from everything else, because an
    expired token is a morning chore and a rejected order is a decision.
  * Every attempt lands in the journal - dry-run, success and failure alike.
    Non-negotiable #3 says rejections are recorded as carefully as fills.
  * Journalling can never break an order path. If the disk is full the order
    still goes.

Written as a mixin rather than a base class so each broker keeps its own
`__init__` and its own vocabulary in its own file.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from .kite_auth import NotAuthenticated

log = logging.getLogger(__name__)

#: Journal events. Spelled once so a reader grepping the journal finds every
#: producer, and so the Journal page's help table cannot drift from reality.
EVENT_DRY_RUN = "order_dry_run"
EVENT_PLACED = "order_placed"
EVENT_FAILED = "order_failed"


class BrokerTransport:
    """
    Mixin providing `_call` and `_log`.

    Expects the host to define `session`, `journal`, `placed` and a `dry_run`
    property.
    """

    #: Reads that failed since construction. A caller comparing this before
    #: and after a batch of reads learns whether what it got back was fact or
    #: an empty default - which is the difference between "nothing is resting
    #: at the broker" and "we could not ask".
    read_failures: int = 0

    def _dry_id(self, what: str) -> str:
        """A traceable stand-in id, so dry-run state transitions are real."""
        return f"DRY-{what.replace(':', '-')}-{len(self.placed):04d}"

    def _call(self, what: str, payload: dict,
              invoke: Callable[[Any], Any]) -> Any:
        """
        Perform one broker interaction.

        `invoke` receives the authenticated client and returns whatever the
        SDK returns - an order id, a trigger id, a list. Returns that value on
        success, a synthetic id in dry-run, and **None** on any failure.

        The return value is the whole point and is why this is not
        `kite_orders._send`, which returns a bare bool and drops the id. An id
        you did not keep is an order you cannot cancel, poll or reconcile -
        which is exactly the state the option path is in today.
        """
        self.placed.append({"what": what, **payload})

        if self.dry_run:
            fake = self._dry_id(what)
            log.info("DRY RUN %s: %s", what, payload)
            self._log(EVENT_DRY_RUN, {"what": what, "payload": payload,
                                      "result": fake})
            return fake

        try:
            client = self.session.client()
            result = invoke(client)
        except NotAuthenticated as e:
            log.error("%s NOT sent - not authenticated: %s", what, e)
            self._log(EVENT_FAILED, {"what": what, "payload": payload,
                                     "error": str(e), "kind": "auth"})
            return None
        except Exception as e:
            log.error("%s NOT sent: %s", what, e)
            self._log(EVENT_FAILED, {"what": what, "payload": payload,
                                     "error": f"{type(e).__name__}: {e}",
                                     "kind": "broker"})
            return None

        log.info("%s ok, result=%s", what, result)
        self._log(EVENT_PLACED, {"what": what, "payload": payload,
                                 "result": result})
        return result

    def _read(self, what: str, invoke: Callable[[Any], Any], default: Any):
        """
        A READ against the broker. Never gated by `dry_run`.

        Holdings, positions, orders and margins are the facts this system
        reconciles against; suppressing them in dry-run would leave the safest
        mode with the least information about what is actually true. Returns
        `default` on failure rather than raising, and says so in the journal.

        FAILURES ARE COUNTED, and callers must check. Returning an empty list
        for "the request failed" is indistinguishable from "you have no open
        triggers", and a reconciler that cannot tell those apart FAILS OPEN -
        it would place a second stop on a position that already has one. See
        `read_failures` and `daily.run`'s use of it.
        """
        try:
            return invoke(self.session.client())
        except NotAuthenticated as e:
            log.warning("%s unavailable - not authenticated: %s", what, e)
            self.read_failures += 1
            self._log(EVENT_FAILED, {"what": f"read:{what}", "error": str(e),
                                     "kind": "auth"})
            return default
        except Exception as e:
            log.warning("%s unavailable: %s", what, e)
            self.read_failures += 1
            self._log(EVENT_FAILED, {"what": f"read:{what}",
                                     "error": f"{type(e).__name__}: {e}",
                                     "kind": "broker"})
            return default

    def _log(self, event: str, payload: dict) -> None:
        if getattr(self, "journal", None) is None:
            return
        try:
            self.journal.write(event, payload)
        except Exception:      # pragma: no cover - journalling is best-effort
            pass               # ...but it must NEVER break an order path
