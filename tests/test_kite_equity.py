"""
The cash-equity order path and the rate limiter.

WHY THIS FILE EXISTS. Before it, `kite_orders.py` and `kite_auth.py` had zero
test coverage - the two modules that can spend money and hold a bearer
credential. Adding a third such module without tests would have been the same
mistake a second time.

Nothing here touches the network. The broker is handed a fake client, so every
assertion is about the PAYLOAD - which is the part you cannot inspect once it
has been sent.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from nifty_algo.broker import kite_equity as eq_mod
from nifty_algo.broker.kite_auth import NotAuthenticated
from nifty_algo.broker.kite_equity import KiteEquity
from nifty_algo.broker.ratelimit import (RateLimiter, ThrottledKite,
                                         bucket_for)
from nifty_algo.config import Config
from nifty_algo.journal import Journal

TODAY = date(2026, 8, 24)


# ---------------------------------------------------------------- fakes

class _Client:
    """Records what it was asked to do. Returns plausible broker ids."""

    def __init__(self, fail: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail
        self.GTT_TYPE_OCO = "two-leg"

    def _record(self, name, **kwargs):
        if self.fail:
            raise self.fail
        self.calls.append((name, kwargs))
        return f"{name}-1"

    # pykiteconnect's GTT methods return the RAW response - a dict - while
    # `place_order` ends `...["order_id"]` and returns a bare string. Getting
    # that wrong put the literal text "{'trigger_id': 123}" in the ledger.
    def place_gtt(self, **kw):
        self._record("place_gtt", **kw)
        return {"trigger_id": 4242}

    def modify_gtt(self, **kw):
        self._record("modify_gtt", **kw)
        return {"trigger_id": 4242}

    def delete_gtt(self, trigger_id):
        self._record("delete_gtt", trigger_id=trigger_id)
        return {"trigger_id": int(trigger_id)}

    def place_order(self, **kw):
        return self._record("place_order", **kw)

    def holdings(self):
        return [{"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 6,
                 "t1_quantity": 0, "average_price": 1500.0,
                 "last_price": 1550.0, "pnl": 300.0}]

    def get_gtts(self):
        return [{
            "id": 77, "status": "active", "type": "two-leg",
            "condition": {"tradingsymbol": "INFY", "exchange": "NSE",
                          "trigger_values": [1420.0, 1660.0]},
            "orders": [{"transaction_type": "SELL", "quantity": 6,
                        "tradingsymbol": "INFY"}],
        }]

    def margins(self, segment=None):
        return {"available": {"live_balance": 42_000.0}, "net": 42_000.0}


class _Session:
    def __init__(self, client=None, error: Exception | None = None):
        self._client = client
        self._error = error
        self.authenticated = client is not None

    def client(self):
        if self._error:
            raise self._error
        return self._client


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal")


@pytest.fixture
def live(cfg, journal):
    """A broker wired to a fake client, with dry run OFF."""
    cfg.equity_broker.dry_run = False
    client = _Client()
    return KiteEquity(_Session(client), cfg, journal), client


# ---------------------------------------------------------------- the guards

def test_dry_run_sends_nothing_and_still_returns_an_id(cfg, journal):
    """
    A dry run that returned None would rehearse only the failure path.

    The synthetic id lets the position book's whole state machine run - armed,
    filled, laddered, closed - with no network and no money, which is what
    makes the dry run worth having.
    """
    client = _Client()
    broker = KiteEquity(_Session(client), cfg, journal)

    assert broker.dry_run is True                 # the shipped default
    gtt_id = broker.place_buy_gtt("INFY", 1500.0, 6, 1480.0)

    assert gtt_id and gtt_id.startswith("DRY-")
    assert client.calls == []


def test_every_write_is_journalled_even_in_dry_run(cfg, journal):
    broker = KiteEquity(_Session(_Client()), cfg, journal)
    broker.place_buy_gtt("INFY", 1500.0, 6, 1480.0)

    events = [r["event"] for r in journal.read_day()]
    assert "order_dry_run" in events
    record = next(r for r in journal.read_day()
                  if r["event"] == "order_dry_run")
    assert record["payload"]["tradingsymbol"] == "INFY"


def test_orders_are_cnc_and_limit_and_never_market(live):
    """
    MIS would be squared off by the broker at 15:20, turning every swing
    trade into an intraday one without saying so.
    """
    broker, client = live
    broker.place_exit_gtt("INFY", 6, 1420.0, 1660.0, 1500.0)
    broker.place_market_exit("INFY", 6, 1500.0)

    for _, kw in client.calls:
        legs = kw.get("orders") or [kw]
        for leg in legs:
            assert leg["product"] == "CNC"
            assert leg["order_type"] == "LIMIT"


# ---------------------------------------------------------------- payloads

def test_a_buy_gtt_is_single_leg_with_the_limit_above_the_trigger(live):
    """
    A GTT places a LIMIT order when it fires, and if that limit does not fill
    the same day the GTT is CANCELLED rather than retried. An exact-price
    limit on a fast breakout therefore loses the trade AND the trigger.
    """
    broker, client = live
    broker.place_buy_gtt("INFY", 1500.0, 6, 1480.0)

    name, kw = client.calls[0]
    assert name == "place_gtt"
    assert kw["trigger_type"] == eq_mod.GTT_SINGLE
    assert kw["trigger_values"] == [1500.0]
    leg = kw["orders"][0]
    assert leg["transaction_type"] == "BUY"
    assert leg["price"] > 1500.0


def test_an_exit_gtt_is_two_leg_with_stop_below_and_target_above(live):
    broker, client = live
    broker.place_exit_gtt("INFY", 6, 1420.0, 1660.0, 1500.0)

    _, kw = client.calls[0]
    assert kw["trigger_type"] == eq_mod.GTT_OCO
    assert kw["trigger_values"] == [1420.0, 1660.0]      # ascending
    stop_leg, target_leg = kw["orders"]
    assert stop_leg["price"] < 1420.0                    # sells cross DOWN
    assert target_leg["price"] < 1660.0
    assert stop_leg["transaction_type"] == target_leg["transaction_type"] == "SELL"


def test_an_inverted_oco_is_refused_before_it_reaches_the_broker(live):
    """A stop above the target would sell at the wrong end of the move."""
    broker, client = live
    assert broker.place_exit_gtt("INFY", 6, 1700.0, 1660.0, 1500.0) is None
    assert client.calls == []


def test_zero_quantity_is_refused(live):
    broker, client = live
    assert broker.place_buy_gtt("INFY", 1500.0, 0, 1480.0) is None
    assert broker.place_market_exit("INFY", 0, 1500.0) is None
    assert client.calls == []


def test_prices_are_rounded_to_the_tick(live):
    broker, client = live
    broker.place_buy_gtt("INFY", 1500.037, 6, 1480.011)

    _, kw = client.calls[0]
    for value in (*kw["trigger_values"], kw["last_price"],
                  kw["orders"][0]["price"]):
        assert abs(round(value / 0.05) * 0.05 - value) < 1e-6


def test_the_trigger_id_is_returned_and_retained(live):
    """
    An id you did not keep is an order you cannot cancel, poll or reconcile.

    `kite_orders._send` discards it, which is exactly why nothing on the
    option side can be reconciled against the broker.
    """
    broker, _ = live
    # A bare "4242", not the stringified dict Kite actually hands back. An id
    # that does not match one from `get_gtts()` means every exit OCO is
    # re-placed on every run and no expired trigger is ever really deleted.
    assert broker.place_buy_gtt("INFY", 1500.0, 6, 1480.0) == "4242"
    assert broker.modify_exit_gtt("77", "INFY", 6, 1450.0, 1660.0,
                                  1500.0) == "4242"
    assert broker.delete_gtt("77") == "77"


def test_a_gtt_id_round_trips_against_what_get_gtts_reports(live):
    """
    THE point of unwrapping: the id has to match, or reconciliation cannot
    recognise its own trigger and the run re-arms the same stop for ever.
    """
    broker, _ = live
    placed = broker.place_buy_gtt("INFY", 1500.0, 6, 1480.0)
    reported = {g.trigger_id for g in broker.gtts()}
    assert placed not in reported          # the fake reports id 77, not 4242
    assert isinstance(placed, str) and placed.isdigit()

    from nifty_algo.broker.kite_equity import _gtt
    view = _gtt({"id": 4242, "status": "active", "type": "single",
                 "condition": {"tradingsymbol": "INFY",
                               "trigger_values": [1500.0]},
                 "orders": [{"transaction_type": "BUY", "quantity": 6}]})
    assert view.trigger_id == placed


def test_a_plain_order_still_returns_its_bare_order_id(live):
    """`place_order` is the other shape and must not be unwrapped."""
    broker, _ = live
    assert broker.place_market_exit("INFY", 6, 1500.0) == "place_order-1"


# ---------------------------------------------------------------- failures

def test_a_broker_error_returns_none_rather_than_raising(cfg, journal):
    """
    An exception escaping here would land in a Streamlit rerun and lose the
    page. A falsy return keeps the caller in charge - and `arm_pick` then
    refuses to record a ticket for an order that was never placed.
    """
    cfg.equity_broker.dry_run = False
    broker = KiteEquity(_Session(_Client(fail=RuntimeError("rejected"))),
                        cfg, journal)

    assert broker.place_buy_gtt("INFY", 1500.0, 6, 1480.0) is None
    failed = [r for r in journal.read_day() if r["event"] == "order_failed"]
    assert failed and failed[0]["kind"] == "broker"


def test_an_expired_token_is_recorded_as_an_auth_problem_not_a_rejection(
        cfg, journal):
    """
    A morning chore and a refused order are different facts.

    The option book already makes this distinction via `NotConfigured`; a
    journal that conflated them would make "the token expired" look like "the
    exchange said no".
    """
    cfg.equity_broker.dry_run = False
    broker = KiteEquity(_Session(None, error=NotAuthenticated("stale")),
                        cfg, journal)

    assert broker.place_buy_gtt("INFY", 1500.0, 6, 1480.0) is None
    failed = [r for r in journal.read_day() if r["event"] == "order_failed"]
    assert failed and failed[0]["kind"] == "auth"


def test_reads_are_not_gated_by_dry_run(cfg, journal):
    """
    Suppressing holdings in dry run would leave the safest mode the least
    informed about what is actually true.
    """
    broker = KiteEquity(_Session(_Client()), cfg, journal)
    assert broker.dry_run

    holdings = broker.holdings()
    assert holdings and holdings[0].symbol == "INFY"
    assert broker.free_cash() == pytest.approx(42_000.0)
    assert broker.gtts()[0].trigger_id == "77"


def test_an_unreadable_balance_is_none_and_not_zero(cfg, journal):
    """
    None means "not checked". Zero means "you have no money".

    Collapsing them would either block every arm or wave every one through,
    and both are wrong for the same reason.
    """
    broker = KiteEquity(_Session(None, error=NotAuthenticated("no token")),
                        cfg, journal)
    assert broker.free_cash() is None


# ---------------------------------------------------------------- GTT parsing

def test_a_gtt_record_is_flattened_from_both_places_kite_puts_things(cfg):
    raw = {
        "id": 12345, "status": "active", "type": "two-leg",
        "condition": {"tradingsymbol": "tcs", "exchange": "NSE",
                      "trigger_values": [3100.0, 3600.0]},
        "orders": [{"transaction_type": "sell", "quantity": 4}],
    }
    view = eq_mod._gtt(raw)

    assert view.trigger_id == "12345"
    assert view.symbol == "TCS"              # upper-cased for matching
    assert view.is_active and view.is_exit
    assert view.stop == 3100.0 and view.target == 3600.0


def test_t1_shares_count_towards_what_you_own(cfg):
    """
    The day after a buy, Kite reports the shares under `t1_quantity`.

    Reading only `quantity` shows zero, which reads exactly like "the buy
    never happened".
    """
    view = eq_mod._holding({"tradingsymbol": "INFY", "quantity": 0,
                            "t1_quantity": 6, "average_price": 1500.0})
    assert view.total_quantity == 6


# ---------------------------------------------------------------- TPIN / DDPI

def test_without_ddpi_and_without_todays_authorisation_it_says_unprotected(
        cfg, journal):
    broker = KiteEquity(_Session(_Client()), cfg, journal)
    state, why = broker.protection_state(TODAY)

    assert state == eq_mod.UNPROTECTED
    assert "REJECTED" in why


def test_recording_an_authorisation_only_reaches_unverified(cfg, journal):
    """
    Kite Connect has no TPIN endpoint, so this is a claim by you, not a fact.

    Reporting PROTECTED on the strength of a checkbox is precisely the
    "looks armed and is not" failure this whole design exists to prevent.
    """
    broker = KiteEquity(_Session(_Client()), cfg, journal)
    broker.record_authorisation(TODAY)

    assert broker.protection_state(TODAY)[0] == eq_mod.UNVERIFIED


def test_an_authorisation_does_not_survive_the_night(cfg, journal):
    """CDSL authorisations are valid for one trading day. So is this record."""
    broker = KiteEquity(_Session(_Client()), cfg, journal)
    broker.record_authorisation(TODAY)

    assert broker.protection_state(TODAY + timedelta(days=1))[0] \
        == eq_mod.UNPROTECTED


def test_ddpi_is_the_only_thing_that_reaches_protected(cfg, journal):
    cfg.equity_broker.ddpi_active = True
    broker = KiteEquity(_Session(_Client()), cfg, journal)

    assert broker.protection_state(TODAY)[0] == eq_mod.PROTECTED


# ---------------------------------------------------------------- rate limit

def test_each_endpoint_family_lands_in_its_documented_bucket():
    assert bucket_for("quote") == "quote"                  # 1/sec
    assert bucket_for("historical_data") == "historical"   # 3/sec
    assert bucket_for("place_gtt") == "order"              # 10/sec
    # Anything unrecognised falls into Kite's own "all other endpoints" line,
    # so a method added by a future SDK release is throttled, not exempt.
    assert bucket_for("something_new_in_v4") == "default"


def test_the_quote_bucket_blocks_after_one_call_a_second():
    clock = [0.0]
    waits: list[float] = []

    def sleep(d):
        waits.append(d)
        clock[0] += d

    limiter = RateLimiter(sleep=sleep, clock=lambda: clock[0])
    for _ in range(3):
        limiter.acquire("quote")

    assert len(waits) == 2
    assert all(w == pytest.approx(1.0) for w in waits)


def test_a_ten_a_second_bucket_allows_the_full_burst():
    """
    A sliding window, not fixed spacing.

    Spacing at 1/N would cap a 10/sec bucket at one call every 100ms even
    after a minute of idleness, which is stricter than the published limit
    and slower than the app needs to be.
    """
    clock = [0.0]
    limiter = RateLimiter(sleep=lambda d: clock.__setitem__(0, clock[0] + d),
                          clock=lambda: clock[0])
    for _ in range(10):
        assert limiter.acquire("order") == 0.0
    assert limiter.acquire("order") > 0.0


def test_the_proxy_forwards_arguments_and_results_unchanged():
    client = _Client()
    clock = [0.0]
    kite = ThrottledKite(client, RateLimiter(
        sleep=lambda d: clock.__setitem__(0, clock[0] + d),
        clock=lambda: clock[0]))

    # Unchanged means unchanged: the proxy hands back whatever the SDK
    # returned, dict or string. Unwrapping is `KiteEquity._gtt_call`'s job,
    # one layer up - a transport that reshapes results is not transparent.
    assert kite.place_gtt(tradingsymbol="INFY") == {"trigger_id": 4242}
    assert kite.place_order(tradingsymbol="INFY") == "place_order-1"
    assert client.calls[0][1] == {"tradingsymbol": "INFY"}


def test_the_proxy_passes_constants_straight_through():
    """
    `kite.GTT_TYPE_OCO` has to keep working, or every caller has to know it
    is talking to a wrapper - which defeats the point of a transparent proxy.
    """
    kite = ThrottledKite(_Client(), RateLimiter())
    assert kite.GTT_TYPE_OCO == "two-leg"
