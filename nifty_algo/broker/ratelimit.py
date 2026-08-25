"""
One throttle, in front of every Kite call in the project.

WHY THIS IS A PROXY AND NOT A DECORATOR ON EACH CALL SITE. Kite's limits are
per-endpoint-family and per-second, and this repo reaches the API from five
places: the option chain, the intraday feed, the two history scripts, the
option order path and the swing equity book. Sprinkling `time.sleep(0.4)` at
each one - which is what `scripts/fetch_history.py` and `swing/prices.py`
already do - gives you five limiters that each believe they are alone. The
moment two of them run in the same process the ceiling is breached, and the
symptom is a 429 from a call that is nowhere near the throttle you were
looking at.

`KiteSession.client()` is the single seam every one of those paths goes
through, so wrapping its return value is the only change needed to cover all
of them at once, including anything added later.

THE LIMITS (Kite Connect v3, published):

    quote                1 req/second
    historical candle    3 req/second
    order placement     10 req/second
    everything else     10 req/second

A SLIDING WINDOW, NOT FIXED SPACING. Fixed spacing (sleep 1/N between calls)
would cap a 10/sec bucket at exactly one call every 100 ms even when the
bucket has been idle for a minute. The window here allows the full burst the
limit actually permits and only blocks once N calls have genuinely happened
inside one second, which is what the published figure means.

FAIL-SLOW, NEVER FAIL-CLOSED. When the limit is reached this BLOCKS until the
window reopens. It never drops a call and never raises. A dropped order is
indistinguishable from a rejected one at the call site, and this is the one
package that can spend money.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable

log = logging.getLogger(__name__)

#: bucket name -> calls allowed per second
LIMITS: dict[str, float] = {
    "quote": 1.0,
    "historical": 3.0,
    "order": 10.0,
    "default": 10.0,
}

WINDOW_SECONDS = 1.0

#: Kite method name -> bucket. Anything absent lands in "default", which is
#: the correct answer for it under Kite's own "all other endpoints" line - so
#: a method added by a future SDK release is throttled rather than unthrottled.
BUCKET_FOR: dict[str, str] = {
    # 1/sec
    "quote": "quote",
    "ltp": "quote",
    "ohlc": "quote",
    # 3/sec
    "historical_data": "historical",
    # 10/sec, and separately metered by Kite from the rest
    "place_order": "order",
    "modify_order": "order",
    "cancel_order": "order",
    "exit_order": "order",
    "place_gtt": "order",
    "modify_gtt": "order",
    "delete_gtt": "order",
    "place_mf_order": "order",
    "cancel_mf_order": "order",
}


def bucket_for(method_name: str) -> str:
    return BUCKET_FOR.get(method_name, "default")


class RateLimiter:
    """
    Sliding-window limiters, one per bucket, shared across threads.

    Streamlit re-runs the script on every interaction and can service more
    than one session at a time, so the lock is not decorative.
    """

    def __init__(self, limits: dict[str, float] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.limits = dict(limits or LIMITS)
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._calls: dict[str, deque[float]] = {k: deque() for k in self.limits}
        #: Cumulative seconds spent waiting, per bucket. Surfaced in the UI so
        #: "the app feels slow" can be answered with a number instead of a
        #: guess about the network.
        self.waited: dict[str, float] = {k: 0.0 for k in self.limits}

    def acquire(self, bucket: str = "default") -> float:
        """Block until a call on `bucket` is allowed. Returns seconds waited."""
        if bucket not in self.limits:
            bucket = "default"
        per_second = self.limits[bucket]
        if per_second <= 0:
            return 0.0
        allowed = max(1, int(per_second))

        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                q = self._calls[bucket]
                cutoff = now - WINDOW_SECONDS
                while q and q[0] <= cutoff:
                    q.popleft()
                if len(q) < allowed:
                    q.append(now)
                    self.waited[bucket] += waited
                    return waited
                # The oldest call in the window decides when a slot frees up.
                delay = q[0] + WINDOW_SECONDS - now
            # Sleep OUTSIDE the lock, or one waiter blocks every other bucket.
            delay = max(delay, 0.001)
            self._sleep(delay)
            waited += delay

    def note(self) -> str:
        busy = {k: v for k, v in self.waited.items() if v > 0.05}
        if not busy:
            return "no rate-limit waiting"
        parts = ", ".join(f"{k} {v:.1f}s" for k, v in sorted(busy.items()))
        return f"waited on rate limits: {parts}"


class ThrottledKite:
    """
    A transparent stand-in for `KiteConnect`.

    Attribute access is forwarded unchanged; callables are wrapped so the
    matching bucket is acquired first. Non-callable attributes (the SDK's
    VARIETY_*, PRODUCT_*, GTT_TYPE_* constants) pass straight through, which
    is what lets callers keep writing `kite.GTT_TYPE_OCO`.
    """

    def __init__(self, client: Any, limiter: RateLimiter | None = None):
        # Bypass our own __setattr__ - these two are ours, not the client's.
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_limiter", limiter or RateLimiter())

    @property
    def limiter(self) -> RateLimiter:
        return object.__getattribute__(self, "_limiter")

    @property
    def raw(self) -> Any:
        """The unthrottled client. For tests and introspection only."""
        return object.__getattribute__(self, "_client")

    def __getattr__(self, name: str) -> Any:
        client = object.__getattribute__(self, "_client")
        attr = getattr(client, name)
        if not callable(attr):
            return attr
        limiter = object.__getattribute__(self, "_limiter")
        bucket = bucket_for(name)

        def throttled(*args, **kwargs):
            waited = limiter.acquire(bucket)
            if waited > 0.5:
                log.debug("rate limit: waited %.2fs for %s (%s)",
                          waited, name, bucket)
            return attr(*args, **kwargs)

        throttled.__name__ = name
        throttled.__doc__ = getattr(attr, "__doc__", None)
        return throttled

    def __setattr__(self, name: str, value: Any) -> None:
        # `set_access_token` is a method, but KiteConnect also carries plain
        # attributes some callers assign. Forward them to the real client so
        # the proxy cannot silently hold a different value from the thing it
        # is standing in for.
        setattr(object.__getattribute__(self, "_client"), name, value)

    def __repr__(self) -> str:
        return f"ThrottledKite({object.__getattribute__(self, '_client')!r})"
