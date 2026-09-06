"""
The suite must never reach a broker, and this is what proves it.

`conftest._no_live_credentials` is invisible when it works, which is exactly
the kind of guard that rots. These assert the two properties it has to have:
the credentials are unusable during a run, and a test that deliberately sets
one still wins.

The bug it exists to stop was real. `test_auth.py` drives `app.py` through
Streamlit's `AppTest`, so `load_dotenv()` copied the real `.env` into
`os.environ` for the rest of the process, and `test_portfolio_connectors.py`
then read the live brokerage account while asserting Kite was unconfigured.
"""
from __future__ import annotations

import os

import pytest
from conftest import CREDENTIAL_KEYS

from nifty_algo.broker.kite_auth import KiteSession


def test_no_credential_is_usable_during_a_test_run():
    """Blank, not absent - see the fixture for why that distinction matters."""
    for key in CREDENTIAL_KEYS:
        assert not os.getenv(key), f"{key} is set during a test run"


def test_a_kite_session_reports_itself_unconfigured():
    """
    The property that actually matters. `configured` is
    `bool(api_key and api_secret)`, both from the environment, so this is the
    single check standing between the suite and a live account read.
    """
    assert KiteSession().configured is False


def test_a_test_that_sets_a_key_deliberately_still_wins(monkeypatch):
    """
    The fixture is function-scoped and autouse, so it runs BEFORE the test
    body - which means a test may still set a credential for its own purposes.
    `test_auth.py::test_the_bridge_never_overwrites_a_real_environment_variable`
    depends on exactly this precedence.
    """
    monkeypatch.setenv("KITE_API_KEY", "set-by-the-test")
    assert os.environ["KITE_API_KEY"] == "set-by-the-test"


def test_the_guard_is_restored_after_a_test_that_overrode_it():
    """Runs after the test above; the override must not have escaped it."""
    assert not os.getenv("KITE_API_KEY")


@pytest.mark.parametrize("key", ["KITE_API_KEY", "KITE_API_SECRET"])
def test_the_two_keys_that_reach_money_are_covered(key):
    """
    Named individually rather than trusted to the tuple, so removing one from
    `CREDENTIAL_KEYS` fails here instead of silently reopening the path.
    """
    assert key in CREDENTIAL_KEYS
