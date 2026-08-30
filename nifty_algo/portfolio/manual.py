"""
Holdings you typed in, for everything no broker API will tell us.

THREE THINGS ONLY THIS CONNECTOR CAN COVER, and they are not edge cases:

  1. The Shariah ETFs. SPUS, HLAL and the Ireland-domiciled UCITS are held in
     a foreign broker this build cannot reach, and they are the largest single
     input to the risk report - `swing/holdings.py` can look THROUGH them to
     the underlying names, so a fund balance turns into real single-stock
     concentration. Without this connector that look-through has nothing to
     look through.

  2. The account you have not connected yet. IBKR is registered and not
     implemented; until it is, its positions arrive here.

  3. Running the briefings at all with no broker session. Kite's token dies
     overnight, so on most days the only thing that answers is this file.

THE FILE IS GITIGNORED, deliberately and by an existing rule - `data/*.csv`
with explicit negations for the universe files (see `.gitignore`). Balances
are numbers about YOUR account, and this repo does not commit those, the same
way `data/settings.json` and `journal/` are excluded.

A MISSING FILE IS A REAL ANSWER; AN UNREADABLE ONE IS NOT. No file means you
have recorded nothing, which is true and reportable (`available=True`, empty).
A file that exists and will not parse means we could not find out what you
hold, which is `available=False`. Collapsing those two would let a typo in a
CSV read as an empty portfolio.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..config import Config, DEFAULT
from ..swing import markets as markets_mod
from .base import ASSET_CLASSES, EQUITY, ConnectorResult, Position

KEY = "manual"
LABEL = "Recorded by hand"
DEFAULT_PATH = "data/manual_positions.csv"

TEMPLATE = (
    "# Positions no broker API reports - your ETFs, another broker account,\n"
    "# anything held off-platform. Gitignored: these are numbers about your\n"
    "# account, not about the strategy.\n"
    "#\n"
    "# Give EITHER quantity + last_price, OR a single `value` (the line's\n"
    "# current worth in its own currency). `cost` is optional; without it the\n"
    "# line has no P&L and the reports say so rather than showing zero.\n"
    "#\n"
    "# market : india | us | uk   asset_class : equity | etf | mf | cash\n"
    "market,symbol,name,quantity,average_price,last_price,value,cost,"
    "currency,asset_class,account\n"
)


class ManualConnector:
    """
    Positions from `data/manual_positions.csv`, plus anything handed in.

    `extra` exists so the Streamlit page can pass the fund balances you type
    into it without writing them to disk - the same rule `page_portfolio` has
    always followed. The file and the typed rows are merged, and a typed row
    wins on a key collision, because it is the more recent statement.
    """

    key = KEY
    label = LABEL

    def __init__(self, cfg: Config = DEFAULT, path: str | Path = DEFAULT_PATH,
                 extra: list[Position] | None = None):
        self.cfg = cfg
        self.path = Path(path)
        self.extra = list(extra or ())

    def is_configured(self) -> bool:
        """Always. There is nothing to configure - it is a file and a list."""
        return True

    def fetch(self) -> ConnectorResult:
        rows: list[Position] = []
        notes: list[str] = []

        if self.path.exists():
            try:
                parsed, skipped = _read_file(self.path)
            except OSError as e:
                return ConnectorResult.unavailable(
                    KEY, f"{self.path} exists but could not be read ({e}). "
                         f"Refusing to report this as an empty portfolio.")
            rows.extend(parsed)
            notes.append(f"{len(parsed)} from {self.path}")
            if skipped:
                notes.append(f"{len(skipped)} row(s) skipped: "
                             + "; ".join(skipped[:3])
                             + ("..." if len(skipped) > 3 else ""))
        else:
            notes.append(f"no {self.path} - nothing recorded by hand")

        if self.extra:
            notes.append(f"{len(self.extra)} entered in the app")

        merged: dict[str, Position] = {p.key: p for p in rows}
        for p in self.extra:                      # typed rows win, see docstring
            merged[p.key] = p

        return ConnectorResult.ok(KEY, list(merged.values()), "; ".join(notes))

    def write_template(self) -> Path:
        """Create the file with its explanatory header. Never overwrites."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(TEMPLATE, encoding="utf-8")
        return self.path


# ---------------- parsing ----------------

def _read_file(path: Path) -> tuple[list[Position], list[str]]:
    """
    Returns (positions, reasons rows were skipped).

    A bad row is counted out loud rather than dropped. One malformed line
    should not cost you the other twenty, but a line that vanishes silently is
    a position missing from every weight in the report.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]

    out: list[Position] = []
    skipped: list[str] = []
    for i, row in enumerate(csv.DictReader(lines), start=2):
        position, why = _to_position(row)
        if position is None:
            skipped.append(f"line {i}: {why}")
            continue
        out.append(position)
    return out, skipped


def _to_position(row: dict) -> tuple[Position | None, str]:
    symbol = (row.get("symbol") or "").strip().upper()
    if not symbol:
        return None, "no symbol"

    market = (row.get("market") or markets_mod.INDIA).strip().lower()
    currency = (row.get("currency") or "").strip().upper()
    if not currency:
        return None, f"{symbol}: no currency, and there is no safe default"

    asset_class = (row.get("asset_class") or EQUITY).strip().lower()
    if asset_class not in ASSET_CLASSES:
        return None, (f"{symbol}: asset_class {asset_class!r} is not one of "
                      f"{', '.join(ASSET_CLASSES)}")

    quantity = _num(row.get("quantity"))
    last_price = _num(row.get("last_price"))
    value = _num(row.get("value"))
    cost = _num(row.get("cost"))
    average_price = _num(row.get("average_price"))

    if quantity is not None and last_price is not None:
        if average_price is None:
            # A cost total is the other way people record the same fact.
            average_price = (cost / quantity) if (cost and quantity) else 0.0
    elif value is not None:
        # The value form: one notional unit priced at the whole line. This is
        # how a fund balance arrives, and it keeps `value_native` honest
        # without inventing a share count nobody has.
        quantity = 1.0
        last_price = value
        average_price = cost if cost is not None else 0.0
    else:
        return None, f"{symbol}: needs either quantity + last_price, or a value"

    if quantity <= 0:
        return None, f"{symbol}: quantity {quantity} is not a position"

    return Position(
        key=f"{market}:{symbol}",
        symbol=symbol,
        market=market,
        quantity=float(quantity),
        average_price=float(average_price or 0.0),
        last_price=float(last_price),
        currency=currency,
        asset_class=asset_class,
        source=KEY,
        account=(row.get("account") or "").strip(),
        name=(row.get("name") or "").strip(),
    ), ""


def _num(raw) -> float | None:
    """None, not zero. A blank cell is a fact we do not have."""
    text = str(raw or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
