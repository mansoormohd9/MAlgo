"""
Interactive Brokers, READ-ONLY, through the Flex Web Service.

WHY FLEX AND NOT THE TRADING API. The two trading routes - `ib_insync` against
a running TWS / IB Gateway, or the Client Portal gateway - both need a desktop
process up and a login that dies roughly daily. For a book you look at once a
month that is a daily chore bought for nothing: this console never places an
order at IBKR. A Flex Query is a statement IBKR generates server-side from a
long-lived token (up to a year) and a query id, fetched with two plain HTTPS
requests. It cannot trade, which is the property wanted here.

Set up once in Client Portal -> Performance & Reports -> Flex Queries:
  1. Create an Activity Flex Query with the **Open Positions** section (and
     **Cash Report** if you want cash counted), format XML.
  2. Settings -> Flex Web Service -> enable, generate a token.
  3. Put `IBKR_FLEX_TOKEN` and `IBKR_FLEX_QUERY_ID` in `.env`.

THREE THINGS KITE NEVER RAISES, settled here as the stub promised:

  * Positions arrive in the currency of the LISTING. `Position.currency` stays
    native and `aggregate.py` converts through `swing/fx.py`, which refuses to
    guess a rate.
  * Fractional shares: `Position.quantity` is a float throughout.
  * One login, several accounts: `Position.account` carries the account id so
    two accounts are never summed into one concentration figure.

THE STATEMENT IS CACHED for `CACHE_HOURS`. IBKR throttles Flex requests and a
statement is end-of-day data anyway; re-requesting it on every page load would
get the token rate-limited for no new information. A failed request is
`unavailable` - never an empty account - and a stale cache is used only while
it is younger than the cache window, never as a silent fallback beyond it.
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from ..config import Config, DEFAULT
from ..paths import at_root
from .base import CASH, ETF, EQUITY, MUTUAL_FUND, ConnectorResult, Position

KEY = "ibkr"
LABEL = "Interactive Brokers (Flex, read-only)"

TOKEN_ENV = "IBKR_FLEX_TOKEN"
QUERY_ENV = "IBKR_FLEX_QUERY_ID"

SEND_URL = ("https://ndcdyn.interactivebrokers.com/AccountManagement/"
            "FlexWebService/SendRequest")
CACHE_PATH = "data/cache/ibkr_flex.xml"
CACHE_HOURS = 12.0
#: IBKR answers "statement generation in progress" (code 1019) for a few
#: seconds after SendRequest. Poll, briefly, rather than failing the read.
POLL_ATTEMPTS = 6
POLL_SECONDS = 2.0

REQUIREMENTS = (
    f"Set {TOKEN_ENV} and {QUERY_ENV} in .env (Client Portal -> Performance "
    f"& Reports -> Flex Queries; enable the Flex Web Service for the token). "
    f"The query needs the Open Positions section, XML format. Until then, "
    f"record IBKR positions in data/manual_positions.csv."
)

#: Exchanges whose listings are UK - everything else IBKR reports for this
#: account is treated as US. Deliberately short: extend it when you hold
#: something that proves it wrong, not before.
_UK_EXCHANGES = frozenset({"LSE", "LSEETF", "LSEIOB1"})


class IbkrConnector:
    key = KEY
    label = LABEL

    def __init__(self, cfg: Config = DEFAULT, cache_path: str | Path = CACHE_PATH,
                 fetcher=None):
        self.cfg = cfg
        self.cache_path = Path(cache_path)
        # Injected in tests so the suite never reaches IBKR.
        self._fetcher = fetcher

    def is_configured(self) -> bool:
        return bool(os.getenv(TOKEN_ENV, "").strip()
                    and os.getenv(QUERY_ENV, "").strip())

    def fetch(self) -> ConnectorResult:
        if not self.is_configured():
            return ConnectorResult.unavailable(KEY, "not configured. " + REQUIREMENTS)
        try:
            xml_text, age_note = self._statement()
        except Exception as e:
            return ConnectorResult.unavailable(
                KEY, f"the Flex statement could not be fetched "
                     f"({type(e).__name__}: {e}). Reported as unavailable, "
                     f"not as an empty account.")
        try:
            positions, notes = parse_statement(xml_text)
        except Exception as e:
            return ConnectorResult.unavailable(
                KEY, f"the Flex statement did not parse ({e}).")
        note = "; ".join([age_note] + notes)
        return ConnectorResult.ok(KEY, positions, note)

    # ---------------- the two requests ----------------

    def _statement(self) -> tuple[str, str]:
        path = at_root(self.cache_path)
        if path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            if age_h < CACHE_HOURS:
                return (path.read_text(encoding="utf-8"),
                        f"Flex statement cached {age_h:.1f}h ago")
        fetch = self._fetcher or _http_get
        token = os.getenv(TOKEN_ENV, "").strip()
        query = os.getenv(QUERY_ENV, "").strip()
        xml_text = request_statement(fetch, token, query)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml_text, encoding="utf-8")
        return xml_text, f"Flex statement fetched {datetime.now():%d %b %H:%M}"


def request_statement(fetch, token: str, query: str,
                      sleep=time.sleep) -> str:
    """SendRequest, then GetStatement, polling while IBKR is generating it."""
    first = ET.fromstring(fetch(SEND_URL, {"t": token, "q": query, "v": "3"}))
    if (first.findtext("Status") or "").strip() != "Success":
        raise RuntimeError(first.findtext("ErrorMessage")
                           or "SendRequest was refused")
    ref = (first.findtext("ReferenceCode") or "").strip()
    url = (first.findtext("Url") or "").strip()
    if not ref or not url:
        raise RuntimeError("SendRequest returned no reference code")
    for _ in range(POLL_ATTEMPTS):
        body = fetch(url, {"t": token, "q": ref, "v": "3"})
        root = ET.fromstring(body)
        if root.tag == "FlexQueryResponse":
            return body
        code = (root.findtext("ErrorCode") or "").strip()
        if code != "1019":            # anything but "still generating"
            raise RuntimeError(root.findtext("ErrorMessage")
                               or f"GetStatement failed (code {code})")
        sleep(POLL_SECONDS)
    raise RuntimeError("the statement was still generating after "
                       f"{POLL_ATTEMPTS} attempts - try again in a minute")


def _http_get(url: str, params: dict) -> str:
    import requests
    r = requests.get(url, params=params, timeout=30,
                     headers={"User-Agent": "nifty-algo/portfolio"})
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------- parsing

def parse_statement(xml_text: str) -> tuple[list[Position], list[str]]:
    """
    Positions from a Flex XML statement. Pure, so it is tested from a fixture.

    Cash comes from the Cash Report, one line per currency, excluding the
    BASE_SUMMARY row - which is the other rows restated in the base currency,
    and counting it would double the cash.
    """
    root = ET.fromstring(xml_text)
    positions: list[Position] = []
    notes: list[str] = []
    skipped = 0

    for el in root.iter("OpenPosition"):
        # Summary rows repeat lot rows when the query includes both.
        if el.get("levelOfDetail", "SUMMARY").upper() not in ("SUMMARY", ""):
            continue
        category = (el.get("assetCategory") or "").upper()
        if category not in ("STK", "FUND"):
            skipped += 1
            continue
        symbol = (el.get("symbol") or "").strip().upper()
        qty = _f(el.get("position"))
        price = _f(el.get("markPrice"))
        currency = (el.get("currency") or "").strip().upper()
        if not symbol or not qty or price is None or not currency:
            skipped += 1
            continue
        exchange = (el.get("listingExchange") or "").upper()
        market = "uk" if exchange in _UK_EXCHANGES else "us"
        description = el.get("description") or ""
        if category == "FUND":
            asset_class = MUTUAL_FUND
        elif any(w in description.upper() for w in ("ETF", "UCITS", " FUND")):
            asset_class = ETF
        else:
            asset_class = EQUITY
        positions.append(Position(
            key=f"{market}:{symbol}", symbol=symbol, market=market,
            quantity=float(qty),
            average_price=_f(el.get("costBasisPrice")) or 0.0,
            last_price=float(price), currency=currency,
            asset_class=asset_class, source=KEY,
            account=el.get("accountId") or "", name=description))

    for el in root.iter("CashReportCurrency"):
        currency = (el.get("currency") or "").strip().upper()
        if not currency or currency == "BASE_SUMMARY":
            continue
        cash = _f(el.get("endingCash"))
        if not cash:
            continue
        positions.append(Position(
            key=f"us:IBKR-{currency}", symbol=f"IBKR-{currency}",
            market="us", quantity=1.0, average_price=0.0, last_price=cash,
            currency=currency, asset_class=CASH, source=KEY,
            account=el.get("accountId") or "", name=f"IBKR cash ({currency})"))

    if skipped:
        notes.append(f"{skipped} line(s) skipped (options, FX or incomplete)")
    return positions, notes


def _f(raw) -> float | None:
    try:
        return None if raw in (None, "") else float(raw)
    except (TypeError, ValueError):
        return None
