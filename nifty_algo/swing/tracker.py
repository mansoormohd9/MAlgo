"""
What happened to the picks after the scan said them.

Without this the scanner is unfalsifiable. A ranking that is never checked
against outcomes is a ranking you will keep believing regardless of whether it
works, and the whole reason the intraday book journals its rejections is that
the same trap exists there.

Nothing new is persisted. The picks were already written to the append-only
journal on the day they were made; this module replays those records against
the daily bars that have arrived since. That means the follow-up cannot drift
out of sync with what you were actually told, because it is derived from what
you were actually told.

ONE HONEST AMBIGUITY: when a single daily bar's range covers both the stop and
the target, the daily data cannot say which was touched first. That outcome is
recorded as `ambiguous` and counted as a LOSS. Assuming the good half of an
unknowable coin flip is how a backtest flatters itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

EVENT = "swing_scan"

OUTCOME_OPEN = "open"
OUTCOME_TARGET = "target"
OUTCOME_STOPPED = "stopped"
OUTCOME_AMBIGUOUS = "ambiguous"
OUTCOME_EXPIRED = "never_triggered"
OUTCOME_UNKNOWN = "no_data"


@dataclass
class PickOutcome:
    symbol: str
    scanned_on: date
    setup: str
    entry: float
    stop: float
    target: float
    # Float, not int: a US ticket can be 17.0034 shares, and truncating it
    # silently understates the position it is meant to be following.
    quantity: float
    outcome: str
    triggered_on: date | None = None
    closed_on: date | None = None
    last_price: float | None = None
    r_multiple: float | None = None
    # In the MARKET's currency, not necessarily rupees - hence the name. A
    # field called `rupees` holding dollars is the bug this rename prevents.
    amount: float | None = None
    amount_inr: float | None = None
    market: str = "india"
    currency: str = "INR"
    currency_symbol: str = "₹"
    note: str = ""

    @property
    def rupees(self) -> float | None:
        """
        Deprecated alias. Returns the rupee figure, which for India is the
        same number `amount` holds and for a foreign pick is the converted
        one - never the raw foreign amount under a rupee name.
        """
        return self.amount_inr

    def money(self, value: float | None) -> str:
        return "—" if value is None else f"{self.currency_symbol}{value:+,.0f}"

    @property
    def is_closed(self) -> bool:
        return self.outcome in (OUTCOME_TARGET, OUTCOME_STOPPED,
                                OUTCOME_AMBIGUOUS)

    @property
    def risk_points(self) -> float:
        return self.entry - self.stop


@dataclass
class TrackerSummary:
    outcomes: list[PickOutcome] = field(default_factory=list)
    note: str = ""

    @property
    def open(self) -> list[PickOutcome]:
        return [o for o in self.outcomes if o.outcome == OUTCOME_OPEN]

    @property
    def closed(self) -> list[PickOutcome]:
        return [o for o in self.outcomes if o.is_closed]

    @property
    def wins(self) -> int:
        return sum(1 for o in self.closed if o.outcome == OUTCOME_TARGET)

    @property
    def net_r(self) -> float:
        return sum(o.r_multiple or 0.0 for o in self.closed)

    def headline(self) -> str:
        closed = self.closed
        if not closed:
            return (f"{len(self.open)} open, nothing closed yet - too early to "
                    f"say whether this is working.")
        rate = self.wins / len(closed)
        return (f"{len(closed)} closed · {self.wins} hit target "
                f"({rate:.0%}) · net {self.net_r:+.2f}R · "
                f"{len(self.open)} still open")


def record_scan(journal, result) -> None:
    """Write today's picks to the journal. Rejections stay out - there are
    dozens a day and the ledger is already on the page; what matters here is
    what you were actually told to trade."""
    journal.write(EVENT, {
        "scanned_on": result.scanned_on.isoformat() if result.scanned_on else None,
        "universe_size": result.universe_size,
        "eligible_size": result.eligible_size,
        "prices_note": result.prices_note,
        "picks": [p.to_record() for p in result.picks],
    }, day=result.scanned_on)


def open_picks(journal, bars: dict[str, pd.DataFrame],
               lookback_days: int = 30,
               today: date | None = None,
               market: str | None = None) -> TrackerSummary:
    """
    Replay past scans against the bars that have arrived since.

    `bars` is the {symbol: daily DataFrame} map the scan already downloaded,
    so this costs no network calls.

    `market` filters to one exchange. The journal holds every market's picks
    in one file and `bars` only ever covers one of them, so without this a US
    pick would be replayed against India's bar map, find nothing, and be
    reported as "no bars available" forever. Records written before markets
    existed carry no key and are treated as India, which is what they were.
    """
    today = today or date.today()
    cutoff = today - timedelta(days=lookback_days)

    summary = TrackerSummary()
    seen: set[tuple[str, str]] = set()

    for day in journal.available_days():
        if day < cutoff:
            continue
        for record in journal.read_day(day):
            if record.get("event") != EVENT:
                continue
            scanned_on = _parse_date(record.get("scanned_on")) or day
            for pick in record.get("picks") or []:
                symbol = pick.get("symbol")
                if not symbol:
                    continue
                pick_market = str(pick.get("market") or "india")
                if market is not None and pick_market != market:
                    continue
                # The same scan re-run twice in a day would otherwise be
                # counted as two positions in the same name.
                token = (f"{pick_market}:{symbol}", scanned_on.isoformat())
                if token in seen:
                    continue
                seen.add(token)
                summary.outcomes.append(
                    _evaluate(pick, scanned_on, bars.get(symbol), today))

    summary.outcomes.sort(key=lambda o: o.scanned_on, reverse=True)
    if not summary.outcomes:
        summary.note = ("No scans recorded in the last "
                        f"{lookback_days} days.")
    return summary


def _evaluate(pick: dict, scanned_on: date, df: pd.DataFrame | None,
              today: date) -> PickOutcome:
    entry = float(pick.get("entry", 0.0))
    stop = float(pick.get("stop", 0.0))
    target = float(pick.get("target", 0.0))
    quantity = float(pick.get("quantity", 0) or 0)
    valid_until = _parse_date(pick.get("valid_until")) or (
        scanned_on + timedelta(days=5))

    out = PickOutcome(
        symbol=str(pick.get("symbol")), scanned_on=scanned_on,
        setup=str(pick.get("setup_label") or pick.get("setup") or ""),
        entry=entry, stop=stop, target=target, quantity=quantity,
        outcome=OUTCOME_UNKNOWN,
        market=str(pick.get("market") or "india"),
        currency=str(pick.get("currency") or "INR"),
        currency_symbol=_SYMBOLS.get(str(pick.get("currency") or "INR"), "₹"),
    )

    if df is None or df.empty or entry <= stop:
        out.note = "no bars available to evaluate this pick"
        return out

    # Only bars AFTER the scan can trigger it. The scan was made on that
    # day's close, so that day's own range is not tradeable information.
    forward = df[df.index.date > scanned_on]
    if forward.empty:
        out.outcome = OUTCOME_OPEN
        out.note = "no sessions since the scan"
        return out

    window = forward[forward.index.date <= valid_until]
    triggered_at = None
    for ts, bar in window.iterrows():
        if float(bar["high"]) >= entry:
            triggered_at = ts
            break

    if triggered_at is None:
        if today > valid_until:
            out.outcome = OUTCOME_EXPIRED
            out.note = (f"entry {entry:,.2f} was never reached before "
                        f"{valid_until:%d %b}")
        else:
            out.outcome = OUTCOME_OPEN
            out.note = f"waiting for {entry:,.2f}, valid to {valid_until:%d %b}"
            out.last_price = float(forward["close"].iloc[-1])
        return out

    out.triggered_on = triggered_at.date()
    risk = entry - stop
    after = forward[forward.index >= triggered_at]

    for ts, bar in after.iterrows():
        hit_stop = float(bar["low"]) <= stop
        hit_target = float(bar["high"]) >= target
        if hit_stop and hit_target:
            out.outcome = OUTCOME_AMBIGUOUS
            out.closed_on = ts.date()
            out.r_multiple = -1.0
            out.note = ("one bar covered both the stop and the target - daily "
                        "data cannot say which came first, so this is counted "
                        "as a loss")
            break
        if hit_stop:
            out.outcome = OUTCOME_STOPPED
            out.closed_on = ts.date()
            out.r_multiple = -1.0
            out.note = f"stopped at {stop:,.2f}"
            break
        if hit_target:
            out.outcome = OUTCOME_TARGET
            out.closed_on = ts.date()
            out.r_multiple = (target - entry) / risk if risk > 0 else 0.0
            out.note = f"target {target:,.2f} hit"
            break
    else:
        last = float(after["close"].iloc[-1])
        out.outcome = OUTCOME_OPEN
        out.last_price = last
        out.r_multiple = (last - entry) / risk if risk > 0 else 0.0
        out.note = f"open since {out.triggered_on:%d %b}"

    if out.r_multiple is not None and quantity:
        out.amount = out.r_multiple * risk * quantity
        # The rate the ticket was sized at, not today's. This is what that
        # trade actually risked in rupees; re-converting at a later rate would
        # mix an FX move into a strategy result.
        rate = float(pick.get("fx_inr_per_unit") or 1.0) or 1.0
        out.amount_inr = out.amount * rate
    return out


#: Currency symbols for rendering a past pick. Kept here rather than imported
#: from `markets.py` so the tracker can read a journal record for a market
#: that has since been de-registered.
_SYMBOLS = {"INR": "₹", "USD": "$", "GBP": "£", "EUR": "€"}


def _parse_date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
