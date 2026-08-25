"""
The option order path — the module that had no tests and can spend money.

Written to pin the behaviour BEFORE `_send` was lifted onto the shared
`_transport`, so the refactor could be shown to change nothing. That is the
whole reason they exist in this shape: they assert the payload and the return
value, which are the two things a caller depends on and the two things a
"tidy-up" is most likely to alter.

`modify_stop` gets its own test because its behaviour is genuinely surprising
and deliberately so: it sends NOTHING to the broker. The stop trails on the
underlying's ATR and is recomputed every bar, so a resting order would need
modifying on every one of them. The engine holds it instead - which means the
stop only exists while the engine is running. That is a real operational
dependency, and a test is the right place to make sure nobody "fixes" it by
accident.
"""
from __future__ import annotations

import pytest

from nifty_algo.broker.kite_auth import NotAuthenticated
from nifty_algo.broker.kite_orders import KiteOrders
from nifty_algo.config import Config
from nifty_algo.journal import Journal
from nifty_algo.positions import ExitKind


# ---------------------------------------------------------------- fakes

class _Client:
    def __init__(self, fail: Exception | None = None):
        self.orders: list[dict] = []
        self.fail = fail

    def place_order(self, **kw):
        if self.fail:
            raise self.fail
        self.orders.append(kw)
        return "250824000123456"


class _Session:
    def __init__(self, client=None, error: Exception | None = None):
        self._client, self._error = client, error

    def client(self):
        if self._error:
            raise self._error
        return self._client


class _Quote:
    def __init__(self, strike=26_000, option_type="CE"):
        self.strike, self.option_type = strike, option_type


class _Order:
    """Quacks like an ApprovedOrder for the two attributes place_entry reads."""

    def __init__(self, premium=120.0, quantity=130):
        self.quote = _Quote()
        self.entry_premium = premium
        self.quantity = quantity


class _Position:
    def __init__(self, tradingsymbol="NIFTY26AUG26000CE"):
        self.quote = _Quote()
        self.tradingsymbol = tradingsymbol


class _Action:
    def __init__(self, premium=240.0, quantity=65, kind=ExitKind.PARTIAL_EXIT):
        self.premium, self.quantity, self.kind = premium, quantity, kind


class _Chain:
    """The instrument dump is the authority on symbol names, not a format string."""

    def __init__(self, symbol="NIFTY26AUG26000CE"):
        self.symbol = symbol

    def instruments(self):
        import pandas as pd
        from datetime import date
        return pd.DataFrame([{
            "strike": 26_000, "instrument_type": "CE",
            "expiry": date(2026, 8, 27), "tradingsymbol": self.symbol,
        }])

    def nearest_expiry(self):
        from datetime import date
        return date(2026, 8, 27)


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal")


@pytest.fixture
def live(cfg, journal):
    cfg.broker.dry_run = False
    client = _Client()
    return KiteOrders(_Session(client), cfg, chain=_Chain(),
                      journal=journal), client


# ---------------------------------------------------------------- the guards

def test_dry_run_is_the_shipped_default_and_sends_nothing(cfg, journal):
    client = _Client()
    broker = KiteOrders(_Session(client), cfg, chain=_Chain(), journal=journal)

    assert broker.dry_run is True
    assert broker.place_entry(_Order()) == "NIFTY26AUG26000CE"
    assert client.orders == []


def test_orders_are_mis_and_limit(live):
    """
    NRML would let an intraday option survive the close, which is the one
    thing this strategy is not designed to survive. MARKET on a thin option
    book is how you find out what the far side looks like.
    """
    broker, client = live
    broker.place_entry(_Order())
    broker.exit_position(_Position(), _Action())

    assert client.orders
    for payload in client.orders:
        assert payload["product"] == "MIS"
        assert payload["order_type"] == "LIMIT"
        assert payload["exchange"] == "NFO"


# ---------------------------------------------------------------- payloads

def test_the_entry_limit_crosses_the_spread_upward(live):
    """An unfilled entry on a breakout is worse than a tick of slippage."""
    broker, client = live
    broker.place_entry(_Order(premium=120.0, quantity=130))

    payload = client.orders[0]
    assert payload["transaction_type"] == "BUY"
    assert payload["quantity"] == 130
    assert payload["price"] > 120.0
    assert payload["tradingsymbol"] == "NIFTY26AUG26000CE"


def test_the_exit_limit_crosses_downward(live):
    """
    Same reasoning, opposite direction, and the stakes are higher: an
    unfilled exit leaves you in a trade the system believes it has left.
    """
    broker, client = live
    broker.exit_position(_Position(), _Action(premium=240.0))

    payload = client.orders[0]
    assert payload["transaction_type"] == "SELL"
    assert payload["price"] < 240.0


def test_prices_are_rounded_to_the_nse_tick(live):
    broker, client = live
    broker.place_entry(_Order(premium=123.456))

    price = client.orders[0]["price"]
    assert abs(round(price / 0.05) * 0.05 - price) < 1e-9


# ---------------------------------------------------------------- failures

def test_an_unresolvable_symbol_blocks_the_order(cfg, journal):
    """
    Constructing a tradingsymbol by string formatting is a trap - NSE's
    weekly and monthly formats differ and have both been changed. With no
    chain there is no authority, so nothing is sent.
    """
    cfg.broker.dry_run = False
    client = _Client()
    broker = KiteOrders(_Session(client), cfg, chain=None, journal=journal)

    assert broker.place_entry(_Order()) is None
    assert broker.exit_position(_Position(tradingsymbol=""), _Action()) is False
    assert client.orders == []


def test_a_broker_rejection_returns_falsy_and_is_journalled(cfg, journal):
    cfg.broker.dry_run = False
    broker = KiteOrders(_Session(_Client(fail=RuntimeError("margin"))), cfg,
                        chain=_Chain(), journal=journal)

    assert broker.place_entry(_Order()) is None
    assert any(r["event"] == "order_failed" for r in journal.read_day())


def test_an_expired_token_is_journalled_as_an_auth_problem(cfg, journal):
    cfg.broker.dry_run = False
    broker = KiteOrders(_Session(None, error=NotAuthenticated("stale")), cfg,
                        chain=_Chain(), journal=journal)

    assert broker.place_entry(_Order()) is None
    failed = [r for r in journal.read_day() if r["event"] == "order_failed"]
    assert failed and failed[0].get("kind") == "auth"


def test_journalling_failure_never_breaks_an_order(cfg):
    """An order must go even if the disk is full."""
    class _BadJournal:
        def write(self, *a, **kw):
            raise OSError("disk full")

    cfg.broker.dry_run = False
    client = _Client()
    broker = KiteOrders(_Session(client), cfg, chain=_Chain(),
                        journal=_BadJournal())

    assert broker.place_entry(_Order()) == "NIFTY26AUG26000CE"
    assert client.orders


# ---------------------------------------------------------------- the stop

def test_modify_stop_deliberately_sends_nothing_to_the_broker(live):
    """
    NOT a missing feature. The stop trails on the underlying's ATR and is
    recomputed every bar; a resting SL would need modifying on every one of
    them, each modify a request that can fail, be rejected, or race the fill.

    The consequence is real and worth knowing: the stop exists only while the
    engine is running. The swing book reaches the opposite conclusion from
    the same reasoning, because its stop moves once a day - see
    `broker/kite_equity.py`.
    """
    broker, client = live
    assert broker.modify_stop(_Position(), 95.0) is True
    assert client.orders == []


def test_every_attempt_lands_in_the_audit_list(cfg, journal):
    """`placed` is the in-memory trail, dry run and live alike."""
    broker = KiteOrders(_Session(_Client()), cfg, chain=_Chain(),
                        journal=journal)
    broker.place_entry(_Order())
    broker.exit_position(_Position(), _Action())

    assert [p["what"] for p in broker.placed] == ["entry", "exit:partial_exit"]
