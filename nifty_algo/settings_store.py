"""
The handful of settings that must survive a restart.

WHY THIS EXISTS NOW AND DID NOT BEFORE. `page_portfolio.py` says, correctly,
that "a file of account balances is not something this repo should be creating
without being asked". While the app was a screener that was the right call. It
stopped being the right call the moment a capital pot decides whether an order
can be placed: `foreign_capital_inr` was set on one page, mutated the process-
global config, and was gone on the next launch - at which point the US and UK
scans silently went back to standing down and nothing said why.

WHAT GOES IN HERE, AND WHAT DOES NOT.

  IN: the numbers YOU typed that the code cannot derive - the three capital
      pots, and the two broker flags that describe your account rather than
      your strategy.

  OUT: anything derivable (every per-trade figure is a property off the
      governors), anything secret (the API key lives in `.env`, the access
      token in `.kite_session.json`), and every strategy tunable. Those belong
      in `config.py` under version control, where a change to them is a diff
      you can read rather than a JSON file you forgot you edited.

`dry_run` IS PERSISTED, AND ONLY IN ONE DIRECTION. Turning it off is written
down, because otherwise going live would mean editing source on every launch
and you would eventually leave it edited. But `load()` refuses to turn it off
for an account with no DDPI and no recorded authorisation - see `apply_to`.
The file is gitignored for the obvious reason.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/settings.json")

#: (config path, json key). Spelled once so a field cannot be saved under one
#: name and loaded under another.
FIELDS: tuple[tuple[str, str, str], ...] = (
    ("capital", "starting_capital", "option_capital_inr"),
    ("capital", "swing_capital_inr", "swing_capital_inr"),
    ("capital", "foreign_capital_inr", "foreign_capital_inr"),
    ("equity_broker", "dry_run", "equity_dry_run"),
    ("equity_broker", "ddpi_active", "ddpi_active"),
)


def load(path: Path | str = DEFAULT_PATH) -> dict:
    """Whatever is on disk, or `{}`. A corrupt file is treated as absent."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception as e:
        log.warning("settings file unreadable, ignoring it: %s", e)
        return {}


def save(cfg, path: Path | str = DEFAULT_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for section, attr, key in FIELDS:
        payload[key] = getattr(getattr(cfg, section), attr)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_to(cfg, path: Path | str = DEFAULT_PATH) -> list[str]:
    """
    Push saved values onto `cfg`. Returns anything worth telling the user.

    Called once at startup. Type errors are skipped rather than raised: a
    hand-edited file with a string where a float belongs should not stop the
    app from opening, it should be ignored and mentioned.
    """
    raw = load(path)
    notes: list[str] = []
    if not raw:
        return notes

    for section, attr, key in FIELDS:
        if key not in raw:
            continue
        current = getattr(getattr(cfg, section), attr)
        value = raw[key]
        try:
            value = bool(value) if isinstance(current, bool) else type(current)(value)
        except (TypeError, ValueError):
            notes.append(f"{key} in {path} is not a {type(current).__name__} "
                         f"- ignored.")
            continue
        setattr(getattr(cfg, section), attr, value)

    # The one refusal. Live orders on an account that cannot execute a sell
    # unattended is the combination this whole design is built to prevent, so
    # a persisted `dry_run = False` does not survive a restart into an
    # unauthorised account. You can still turn it off again in the UI, having
    # been told - the point is that it is never on by inheritance.
    eq = cfg.equity_broker
    if not eq.dry_run and not eq.ddpi_active:
        eq.dry_run = True
        notes.append(
            "Live order placement was saved as ON, but DDPI is not active on "
            "this account - so every sell needs a CDSL TPIN authorisation "
            "that expires nightly. Dry run has been restored. Turn it off "
            "again deliberately if that is what you want."
        )
    return notes
