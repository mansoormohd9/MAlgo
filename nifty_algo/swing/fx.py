"""
Rupees per unit of a foreign currency, for position sizing only.

WHY THIS EXISTS. `_size()` in the scanner divides a risk budget by a stop
distance. The budget is in rupees because that is where the governors in
`CapitalConfig` live; the stop distance is in whatever the exchange quotes.
For the Nifty those were the same unit and the division was correct by
accident. For a US stock it is rupees divided by dollars, which returns a
share count roughly 88x too large - a number that looks perfectly ordinary on
a ticket and is catastrophic in an order box.

THIS MODULE FAILS CLOSED, AND THAT IS THE WHOLE POINT.

`rate_inr_per()` raises when it cannot produce a rate it trusts. It does not
fall back to a hardcoded 83, to the last rate it ever saw, or to 1.0. Every
other "missing data" path in this book already works this way - the halal
screen treats an absent balance sheet as a failure to verify, not a pass -
and the argument is the same and stronger here, because a wrong FX rate does
not exclude a candidate, it mis-sizes an accepted one. Standing a foreign scan
down for a day costs you nothing you can measure. Sizing every ticket in it
off a stale rate costs real money and announces nothing.

A cached rate is allowed to be a few hours old: the daily bars this book runs
on are themselves a day old, so a rate from this morning is not the weak link.
A rate older than the cache window is treated as absent, not as approximate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

CACHE_NAME = "fx.json"

#: The home currency. Never fetched, never cached, always exactly 1.
HOME = "INR"

#: yfinance pairs, quoted as INR per one unit of the key.
PAIRS = {
    "USD": "USDINR=X",
    "GBP": "GBPINR=X",
    "EUR": "EURINR=X",
}

#: Sanity bounds per currency, INR per unit. A pair that returns a number
#: outside these has not moved - something has gone wrong upstream (a ticker
#: change, an inverted quote, a partial response) and the right response is to
#: refuse rather than to size off it.
_PLAUSIBLE = {
    "USD": (50.0, 200.0),
    "GBP": (60.0, 260.0),
    "EUR": (55.0, 220.0),
}


class FxUnavailable(RuntimeError):
    """
    No rate this module is willing to size a position with.

    Deliberately not a subclass of anything the scanner catches broadly: the
    caller must decide to stand the market down, in writing, on the page.
    """


@dataclass
class Rate:
    currency: str
    inr_per_unit: float
    fetched_at: datetime
    from_cache: bool = False

    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.fetched_at).total_seconds() / 3600.0

    def note(self) -> str:
        when = self.fetched_at.strftime("%d %b %H:%M")
        how = "cached" if self.from_cache else "fresh"
        return (f"1 {self.currency} = ₹{self.inr_per_unit:,.2f} "
                f"({how}, {when})")


def rate_inr_per(currency: str, cfg, force_refresh: bool = False) -> Rate:
    """
    INR per one unit of `currency`.

    Raises `FxUnavailable` rather than returning a guess. See the module
    docstring; this is the contract the caller depends on.
    """
    currency = (currency or "").upper()
    if currency == HOME:
        # Not a fetch and not a cache entry - the identity rate is the one
        # thing here that cannot be stale or wrong.
        return Rate(HOME, 1.0, datetime.now())

    pair = PAIRS.get(currency)
    if pair is None:
        raise FxUnavailable(
            f"no INR pair is registered for {currency}. Add one to fx.PAIRS "
            f"with a plausibility band before sizing anything in it."
        )

    cache_path = Path(cfg.swing.cache_dir) / CACHE_NAME
    max_age = timedelta(hours=cfg.swing.price_cache_hours)

    if not force_refresh:
        cached = _read_cache(cache_path).get(currency)
        if cached is not None and datetime.now() - cached.fetched_at <= max_age:
            return cached

    fetched = _download(pair)
    if fetched is None:
        raise FxUnavailable(
            f"could not fetch {pair}, and no cached rate is within "
            f"{cfg.swing.price_cache_hours}h. Sizing a {currency} trade needs "
            f"a rate this module trusts, so this market stands down for now. "
            f"The Indian book is unaffected."
        )

    low, high = _PLAUSIBLE.get(currency, (0.0, float("inf")))
    if not (low <= fetched <= high):
        raise FxUnavailable(
            f"{pair} returned {fetched:,.4f}, outside the plausible band "
            f"{low:,.0f}-{high:,.0f} INR per {currency}. That is a bad quote, "
            f"not a market move - refusing to size on it."
        )

    rate = Rate(currency, float(fetched), datetime.now())
    _write_cache(cache_path, {**_read_cache(cache_path), currency: rate})
    return rate


def rate_for_market(market, cfg, force_refresh: bool = False) -> Rate:
    """The rate a `markets.Market` needs. Home markets never touch the network."""
    return rate_inr_per(market.currency, cfg, force_refresh=force_refresh)


# ---------------- fetch ----------------

def _download(pair: str) -> float | None:
    try:
        import yfinance as yf
    except ImportError:                                   # pragma: no cover
        return None
    try:
        hist = yf.Ticker(pair).history(period="5d", interval="1d")
    except Exception:
        return None
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    series = hist["Close"].dropna()
    if series.empty:
        return None
    value = float(series.iloc[-1])
    # A zero or negative rate would sail through as "present" and produce a
    # division by zero or a negative share count.
    return value if value > 0 else None


# ---------------- cache ----------------

def _read_cache(path: Path) -> dict[str, Rate]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, Rate] = {}
    for currency, payload in (raw or {}).items():
        if not isinstance(payload, dict):
            continue
        try:
            out[currency] = Rate(
                currency=currency,
                inr_per_unit=float(payload["inr_per_unit"]),
                fetched_at=datetime.fromisoformat(payload["fetched_at"]),
                from_cache=True,
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _write_cache(path: Path, rates: dict[str, Rate]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            c: {"inr_per_unit": r.inr_per_unit,
                "fetched_at": r.fetched_at.isoformat()}
            for c, r in rates.items()
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except Exception:
        pass    # Caching is an optimisation; failing to cache is not an error.
