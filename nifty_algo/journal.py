"""
Append-only journal. Non-negotiable #3: every order, fill, and rejection
written down.

Append-only and never rewritten, for one reason: the journal's value is that
it records what you knew AT THE TIME. A log you can edit is a log that will
eventually be edited to agree with your memory of a trade, and then it is
worth nothing as evidence about whether the system works.

One JSONL file per trading day. Rejections are recorded as carefully as
approvals - "the risk engine refused 14 setups this week because no strike
fit the budget" is exactly the kind of finding that changes a system, and it
is invisible if you only log the trades you took.
"""
from __future__ import annotations
import json
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DIR = Path("journal")


def _encode(o: Any) -> Any:
    """Make dataclasses, enums, datetimes, and numpy scalars JSON-safe."""
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if hasattr(o, "item"):              # numpy scalar
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


class Journal:
    def __init__(self, directory: str | Path = DEFAULT_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, day: date | None = None) -> Path:
        day = day or date.today()
        return self.dir / f"{day:%Y-%m-%d}.jsonl"

    def write(self, event: str, payload: dict | None = None,
              day: date | None = None) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **(payload or {}),
        }
        line = json.dumps(record, default=_encode, ensure_ascii=False)
        # "a" is the whole point - never "w", never seek, never truncate.
        with self.path_for(day).open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---------------- reading ----------------

    def read_day(self, day: date | None = None) -> list[dict]:
        path = self.path_for(day)
        if not path.exists():
            return []
        out: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    # A torn final line from an interrupted write should not
                    # make the rest of the day unreadable.
                    out.append({"event": "unparseable_line", "raw": ln[:500]})
        return out

    def available_days(self) -> list[date]:
        days: list[date] = []
        for p in sorted(self.dir.glob("*.jsonl")):
            try:
                days.append(datetime.strptime(p.stem, "%Y-%m-%d").date())
            except ValueError:
                continue
        return sorted(days, reverse=True)

    def read_all(self) -> Iterator[dict]:
        for d in sorted(self.available_days()):
            yield from self.read_day(d)

    # ---------------- convenience wrappers ----------------

    def signal(self, strategy_key: str, signal_obj: Any, regime: str) -> None:
        self.write("signal", {"strategy": strategy_key, "regime": regime,
                              "signal": signal_obj})

    def rejection(self, strategy_key: str, reason: str, detail: str = "") -> None:
        self.write("rejected", {"strategy": strategy_key,
                                "reason": reason, "detail": detail})

    def halt(self, reason: str) -> None:
        self.write("halt", {"reason": reason})

    def paper_fill(self, alert_dict: dict, fill_premium: float,
                   note: str = "") -> None:
        self.write("paper_fill", {"alert": alert_dict,
                                  "fill_premium": fill_premium, "note": note})

    def paper_exit(self, alert_dict: dict, exit_premium: float,
                   net_pnl: float, note: str = "") -> None:
        self.write("paper_exit", {"alert": alert_dict, "exit_premium": exit_premium,
                                  "net_pnl": net_pnl, "note": note})
