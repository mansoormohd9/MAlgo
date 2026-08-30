"""
The one place a portfolio connector is named.

Same rule as `data/factory.py`, `strategies/registry.py` and
`swing/markets.py`, and for the same reason: a connector that the reports can
read but the UI cannot list, or that the UI offers and the CLI has never heard
of, is a bug this shape makes impossible. Adding IBKR for real should be one
new file and one line in `CONNECTORS`.

`get()` RAISES on an unknown key rather than falling back to Kite. A typo that
silently returned the Indian broker would produce a portfolio report that is
internally consistent, plausible, and about the wrong account - the same class
of failure `markets.get` and `CapitalConfig._pool` were both changed to
prevent.
"""
from __future__ import annotations

from ..config import Config, DEFAULT
from . import ibkr, kite, manual

#: Construction order is display order. Manual first: it is the one that
#: always answers, so a page that shows connectors in this order never opens
#: on a wall of "not connected".
CONNECTORS: dict[str, type] = {
    manual.KEY: manual.ManualConnector,
    kite.KEY: kite.KiteConnector,
    ibkr.KEY: ibkr.IbkrConnector,
}


class UnknownConnector(KeyError):
    """Asked for a connector that is not registered. Never guess a default."""


def keys() -> list[str]:
    """Registered keys, in display order."""
    return list(CONNECTORS)


def get(key: str, cfg: Config = DEFAULT, **kwargs):
    """Build one connector by key."""
    try:
        factory = CONNECTORS[key]
    except KeyError:
        raise UnknownConnector(
            f"unknown portfolio connector {key!r} - registered connectors are "
            f"{', '.join(sorted(CONNECTORS))}"
        ) from None
    return factory(cfg=cfg, **kwargs)


def build_all(cfg: Config = DEFAULT, **per_key) -> list:
    """
    Every registered connector, constructed.

    `per_key` passes keyword arguments to one connector by name, e.g.
    `build_all(cfg, manual={"extra": rows})`. Construction must be cheap and
    must not touch the network - `KiteConnector` builds its broker lazily for
    exactly this reason, because asking "which of these are configured" should
    not be the thing that opens a session file.
    """
    return [get(key, cfg, **per_key.get(key, {})) for key in CONNECTORS]
