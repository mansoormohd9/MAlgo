"""
Mutual funds, from a CAMS / KFintech Consolidated Account Statement (CAS).

WHY A STATEMENT AND NOT AN API. There is no free API that returns an Indian
investor's mutual-fund holdings across AMCs. The CAS is the one document that
does: every folio at every AMC, emailed on request from CAMS or KFintech,
password-protected with your PAN. `casparser` reads it.

SO THIS CONNECTOR IS AN IMPORT, NOT A LIVE READ. You upload the PDF on the
Connect page; `import_pdf` parses it once and writes the normalised holdings
to `data/cas_holdings.json` (gitignored - these are your balances). The
connector then reads that file. The statement date travels with every line,
because a CAS is as old as the day you requested it, and a six-month-old
statement read as today's balance is a stale fact wearing a fresh one's
clothes.

ENABLED AND NEVER IMPORTED IS `unavailable`, not empty. Enabling this
connector is a claim that you hold mutual funds; a snapshot that silently
counted them as zero would understate the equity share and push every
new-money suggestion toward equity. Same rule as every other connector.

The PDF and its password are never written to disk.
"""
from __future__ import annotations

import io
import json
from datetime import date, datetime
from pathlib import Path

from ..config import Config, DEFAULT
from ..paths import at_root
from .base import FIXED_INCOME, GOLD, MUTUAL_FUND, ConnectorResult, Position

KEY = "cas"
LABEL = "Mutual funds (CAS import)"
DEFAULT_PATH = "data/cas_holdings.json"

#: casparser's scheme `type`, mapped to an asset class. Anything unlisted
#: (EQUITY, HYBRID, None, a new label) is counted as an equity fund - the
#: allocation page names the mapping, and hybrid funds are mostly equity.
_DEBT_TYPES = frozenset({"DEBT", "LIQUID", "MONEY MARKET", "GILT",
                         "OVERNIGHT", "FIXED MATURITY"})


class CasConnector:
    key = KEY
    label = LABEL

    def __init__(self, cfg: Config = DEFAULT, path: str | Path = DEFAULT_PATH):
        self.cfg = cfg
        self.path = Path(path)

    def is_configured(self) -> bool:
        return at_root(self.path).exists()

    def fetch(self) -> ConnectorResult:
        path = at_root(self.path)
        if not path.exists():
            return ConnectorResult.unavailable(
                KEY, "no CAS imported yet - upload one on Connect. Reported "
                     "as unavailable rather than as holding no funds.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return ConnectorResult.unavailable(KEY, f"{path} is unreadable ({e})")
        positions = [to_position(h, data.get("statement_date", ""))
                     for h in data.get("holdings", [])]
        positions = [p for p in positions if p is not None]
        return ConnectorResult.ok(KEY, positions, _age_note(data))


def _age_note(data: dict) -> str:
    stamp = data.get("statement_date") or "unknown date"
    try:
        days = (date.today() - date.fromisoformat(stamp)).days
        note = f"CAS dated {stamp} ({days} days old)"
    except ValueError:
        note = f"CAS dated {stamp}"
    warnings = data.get("parse_warnings") or []
    if warnings:
        note += (f"; {len(warnings)} parse warning(s) - check these schemes "
                 f"against the PDF: " + "; ".join(warnings[:2]))
    return note


def to_position(h: dict, statement_date: str) -> Position | None:
    units = float(h.get("units") or 0.0)
    value = h.get("value")
    if units <= 0 or value in (None, ""):
        return None
    value = float(value)
    isin = (h.get("isin") or h.get("amfi") or (h.get("name") or "")[:20]).upper()
    if not isin:
        return None
    name = h.get("name") or isin
    kind = (h.get("type") or "").upper()
    if kind in _DEBT_TYPES:
        asset_class = FIXED_INCOME
    elif "GOLD" in name.upper():
        asset_class = GOLD
    else:
        asset_class = MUTUAL_FUND
    cost = h.get("cost")
    return Position(
        key=f"india:{isin}", symbol=isin, market="india", quantity=units,
        average_price=(float(cost) / units) if cost else 0.0,
        last_price=value / units, currency="INR", asset_class=asset_class,
        source=KEY, account=h.get("folio") or "", name=name)


def normalise(parsed: dict) -> dict:
    """
    casparser's output -> the small JSON this connector reads. Pure.

    Only closing balances are kept: transactions are not needed to know what
    you hold, and a file of every purchase you ever made is more personal data
    than an allocation needs.
    """
    holdings = []
    latest = ""
    for folio in parsed.get("folios", []) or []:
        for s in folio.get("schemes", []) or []:
            val = s.get("valuation") or {}
            # casparser returns Decimals; serialised they are STRINGS, and the
            # string "0.000" is truthy - a closed-out scheme would survive a
            # bare `if not units`.
            units = _num(s.get("close"))
            if units <= 0:
                continue
            holdings.append({
                "isin": s.get("isin") or "",
                "amfi": s.get("amfi") or "",
                "name": s.get("scheme") or "",
                "type": s.get("type") or "",
                "folio": folio.get("folio") or "",
                "units": units,
                "nav": _num(val.get("nav")) or None,
                "value": _num(val.get("value")),
                "cost": _num(val.get("cost")) or None,
            })
            d = str(val.get("date") or "")
            latest = max(latest, d)
    period = (parsed.get("statement_period") or {}).get("to") or ""
    return {
        "statement_date": latest or _iso(period),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
        "holdings": holdings,
        # casparser's own reconciliation against the statement's running unit
        # balance. Non-empty means a row was probably mis-read - kept and
        # shown, never dropped.
        "parse_warnings": list(parsed.get("parse_warnings") or []),
    }


def _num(raw) -> float:
    try:
        return float(str(raw).replace(",", "")) if raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _iso(text: str) -> str:
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text


def import_pdf(pdf_bytes: bytes, password: str,
               path: str | Path = DEFAULT_PATH) -> dict:
    """
    Parse a CAS PDF and write the normalised holdings. Returns what was written.

    The PDF is handed to casparser as an in-memory stream, so neither it nor
    the password ever touches the disk.
    """
    try:
        import casparser
    except ImportError as e:
        raise RuntimeError("casparser is not installed - "
                           "`pip install casparser`") from e
    parsed = casparser.read_cas_pdf(io.BytesIO(pdf_bytes), password,
                                    output="dict")
    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    out = normalise(json.loads(json.dumps(parsed, default=str)))
    target = at_root(Path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
