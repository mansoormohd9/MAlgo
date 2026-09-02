"""
NSE equity instrument tokens, resolved once a day and cached.

Kite's historical API is addressed by INSTRUMENT TOKEN, not by trading
symbol. Nothing in this project resolved an equity token before: `kite_chain`
calls `instruments("NFO")` for the option chain, and `kite_equity` places GTTs
by tradingsymbol and never needs one. The intraday equity book does, for every
name in the universe, so this module exists.

WHY THIS WORKS AT ALL, AND WHERE IT DOES NOT

NSE equity tokens are PERMANENT. That is the same property that makes deep
NIFTY index history possible (`kite_feed.py:1-24`) and expired-option history
impossible - an option's token is retired at expiry, an equity's is not. So a
token cached today is still correct next year, and the daily refresh here is
about newly listed names and renames, not expiry.

The dump is large - tens of thousands of rows across all segments - and Kite
rate-limits it. Fetching it once per session and caching to disk is the
difference between a 3-second startup and a 3-minute one.

A SYMBOL THAT IS NOT IN THE DUMP IS A NAMED FAILURE, NEVER A SKIP.

That rule is the whole reason `resolve()` returns a `TokenSet` with a
`missing` list rather than just a dict. A universe of 100 names that silently
resolves to 97 produces a scan that is quietly 3% blind - no error, no empty
result, just three companies that never appear in any rejection ledger and
therefore look like they never had a setup. The swing book learned the same
lesson with `PriceSet.missing`, which `fetch_swing_history` reports BY NAME
rather than by count.

DELISTINGS ARE THE OTHER HALF OF THAT, and this module cannot fix them. The
dump is today's listed universe, so a company that left the index two years
ago is absent here AND absent from any history you fetch. See the survivorship
caveat in `backtest.py` - it is a limit of the data, not a bug to be found.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from ..broker.kite_auth import KiteSession, NotAuthenticated

#: Where the resolved map lives between sessions.
CACHE_NAME = "nse_instruments.json"

#: Kite's exchange segment for cash equity.
EXCHANGE = "NSE"

#: Only ordinary equity. The NSE segment also carries ETFs, government
#: securities, REITs and rights entitlements; "EQ" is the series that means
#: "a share you can take an intraday position in".
EQUITY_SERIES = "EQ"


class InstrumentsUnavailable(RuntimeError):
    """The dump could not be fetched and no usable cache existed."""


@dataclass
class TokenSet:
    """
    Resolved tokens, plus an explicit account of what did not resolve.

    `missing` is not an error channel - a universe legitimately contains
    names that have been delisted or renamed. It is a REPORTING channel, and
    the caller is expected to say the names out loud.
    """
    tokens: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    fetched_on: date | None = None
    from_cache: bool = False

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, symbol: str) -> bool:
        return symbol.upper() in self.tokens

    def get(self, symbol: str) -> int | None:
        return self.tokens.get(symbol.upper())

    def note(self) -> str:
        src = "cache" if self.from_cache else "Kite"
        when = self.fetched_on.isoformat() if self.fetched_on else "unknown"
        line = f"{len(self.tokens)} NSE equity tokens from {src} ({when})"
        if self.missing:
            shown = ", ".join(self.missing[:8])
            more = f" +{len(self.missing) - 8} more" if len(self.missing) > 8 else ""
            line += f"; UNRESOLVED: {shown}{more}"
        return line


def cache_path(cache_dir: str = "data/cache") -> Path:
    return Path(cache_dir) / CACHE_NAME


def _read_cache(path: Path, max_age_days: int) -> tuple[dict, date] | None:
    """The cached map, or None if absent, corrupt or stale."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        fetched = date.fromisoformat(raw["fetched_on"])
        tokens = {str(k): int(v) for k, v in raw["tokens"].items()}
    except Exception:
        # A torn cache costs one re-download, never a crash.
        return None
    if not tokens:
        return None
    if (date.today() - fetched).days > max_age_days:
        return None
    return tokens, fetched


def _write_cache(path: Path, tokens: dict[str, int], fetched: date) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_on": fetched.isoformat(), "tokens": tokens}),
            encoding="utf-8")
    except Exception:
        # Caching is an optimisation; failing to cache is not failing.
        pass


def _download(session: KiteSession) -> dict[str, int]:
    """
    Every NSE `EQ` tradingsymbol mapped to its instrument token.

    Filtered to `EQUITY_SERIES` on purpose: the unfiltered NSE dump includes
    ETFs and government securities whose tradingsymbols can collide with the
    shapes a universe file uses, and an ETF is not something this book should
    ever be able to take an MIS position in by accident.
    """
    kite = session.client()
    try:
        rows = kite.instruments(EXCHANGE)
    except Exception as e:
        raise InstrumentsUnavailable(
            f"Kite instruments({EXCHANGE}) failed: {e}") from e

    tokens: dict[str, int] = {}
    for row in rows or []:
        if str(row.get("segment") or "") != EXCHANGE:
            continue
        if str(row.get("instrument_type") or "") != EQUITY_SERIES:
            continue
        symbol = str(row.get("tradingsymbol") or "").upper()
        token = row.get("instrument_token")
        if symbol and token:
            tokens[symbol] = int(token)

    if not tokens:
        raise InstrumentsUnavailable(
            f"Kite returned no {EXCHANGE} {EQUITY_SERIES} instruments")
    return tokens


def load_tokens(session: KiteSession | None = None,
                cache_dir: str = "data/cache",
                max_age_days: int = 1,
                force_refresh: bool = False) -> tuple[dict[str, int], date, bool]:
    """
    The full NSE equity token map: (tokens, fetched_on, from_cache).

    Falls back to a STALE cache if the download fails, because permanent
    tokens do not rot - a month-old map is still correct for every name it
    holds. It will simply be missing anything listed since, which `resolve()`
    then reports as unresolved rather than hiding.
    """
    path = cache_path(cache_dir)
    if not force_refresh:
        hit = _read_cache(path, max_age_days)
        if hit is not None:
            return hit[0], hit[1], True

    if session is None:
        session = KiteSession()

    try:
        tokens = _download(session)
    except (InstrumentsUnavailable, NotAuthenticated):
        stale = _read_cache(path, max_age_days=36500)
        if stale is not None:
            return stale[0], stale[1], True
        raise

    today = date.today()
    _write_cache(path, tokens, today)
    return tokens, today, False


def resolve(symbols, session: KiteSession | None = None,
            cache_dir: str = "data/cache",
            max_age_days: int = 1,
            force_refresh: bool = False) -> TokenSet:
    """
    Resolve an iterable of NSE tradingsymbols to instrument tokens.

    Every symbol lands in exactly one of `tokens` or `missing`, so the caller
    can assert the two account for the whole universe. That assertion is the
    point of the class.
    """
    all_tokens, fetched, from_cache = load_tokens(
        session, cache_dir=cache_dir, max_age_days=max_age_days,
        force_refresh=force_refresh)

    out = TokenSet(fetched_on=fetched, from_cache=from_cache)
    for sym in symbols:
        key = str(sym).upper()
        token = all_tokens.get(key)
        if token is None:
            out.missing.append(key)
        else:
            out.tokens[key] = token
    return out


def accounts_for(tokens: TokenSet, symbols) -> bool:
    """
    Every requested symbol is either resolved or explicitly named missing.

    The guard against a silently shrinking universe. Mirrors
    `ScanResult.accounts_for_everything()` in the swing scanner.
    """
    wanted = {str(s).upper() for s in symbols}
    seen = set(tokens.tokens) | set(tokens.missing)
    return wanted == seen


__all__ = ["TokenSet", "InstrumentsUnavailable", "resolve", "load_tokens",
           "accounts_for", "cache_path", "EXCHANGE", "EQUITY_SERIES"]
