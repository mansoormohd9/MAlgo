"""
The login gate, and the two ways it could be wrong without raising.

WHY THIS FILE EXISTS. `auth.py` is the only thing between a public
`*.streamlit.app` URL and an app that can flip `dry_run` off, arm GTTs and
sell positions. Both of its failure modes are silent:

  1. It opens when it should not. An unconfigured gate that waves everyone
     through looks EXACTLY like a working app to the person who deployed it,
     because that person knows the password and never sees the difference.
     `test_an_unconfigured_gate_refuses_rather_than_opening` is the one that
     matters most in this file.

  2. It leaks which half was wrong. `check()` combines its two comparisons
     with `&` rather than `and` precisely so a wrong username costs the same
     as a wrong password; `and` would short-circuit and make usernames
     enumerable by timing. That is not observable from a return value, so
     `test_a_wrong_username_is_rejected_and_still_costs_an_attempt` asserts
     the consequence that IS observable - it must consume an attempt.

The pure rules take plain arguments and a plain dict, so most of this file
needs no Streamlit runtime at all. The last group drives the real `app.py`
through `AppTest`, because "nothing renders behind the form" is a claim about
`app.py`'s import order and cannot be tested against `auth.py` alone.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import seed_offline_broker
from streamlit.testing.v1 import AppTest

from nifty_algo.config import Config
from nifty_algo.ui import auth

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class _NoSecrets:
    """Stands in for `st.secrets` on a machine that has no secrets.toml."""

    def items(self):
        raise FileNotFoundError("no secrets.toml")

    def __getitem__(self, key):
        raise FileNotFoundError("no secrets.toml")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Neither source of credentials may leak in from the machine.

    Set to EMPTY rather than deleted, deliberately. `app.py` calls
    `load_dotenv()` on every AppTest run, and python-dotenv skips any key
    already present in `os.environ` - so an empty string blocks a real `.env`
    from re-supplying the password, while `_lookup` still reads it as unset.
    Deleting the keys instead would make the two end-to-end tests below pass
    on this machine today and fail the moment somebody puts APP_PASSWORD in
    their own `.env`, which is precisely what they are being told to do.

    `st.secrets` is stubbed for the same reason: a local secrets.toml would
    otherwise decide the result.

    monkeypatch rather than touching os.environ directly, or the order the
    tests happen to run in decides the result.
    """
    for key in (auth.USER_KEY, auth.PASSWORD_KEY, auth.DISABLE_KEY):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(auth.st, "secrets", _NoSecrets())
    # `_attempts()` is a @st.cache_resource dict, so it is shared by every
    # AppTest in this process - and it is SUPPOSED to be, that is the whole
    # point of a throttle that survives reconnecting. Left uncleared, one
    # test's failed sign-in backs off the next one and the second test sees
    # "too many attempts" instead of "wrong password", intermittently,
    # depending on how long the first AppTest took.
    auth._attempts().clear()


def _configure(monkeypatch, user="orion", password="a-long-passphrase"):
    monkeypatch.setenv(auth.USER_KEY, user)
    monkeypatch.setenv(auth.PASSWORD_KEY, password)
    return user, password


# --------------------------------------------------------------------------
# configuration - the fail-closed rules
# --------------------------------------------------------------------------
def test_an_unconfigured_gate_refuses_rather_than_opening():
    """No password set is NOT "no password needed"."""
    assert auth.credentials() is None
    assert auth.auth_disabled() is False


def test_half_a_credential_pair_is_not_configured(monkeypatch):
    """A blank username with a password set would be a decorative field."""
    monkeypatch.setenv(auth.PASSWORD_KEY, "a-long-passphrase")
    assert auth.credentials() is None

    monkeypatch.delenv(auth.PASSWORD_KEY)
    monkeypatch.setenv(auth.USER_KEY, "orion")
    assert auth.credentials() is None


def test_a_configured_pair_is_read_from_the_environment(monkeypatch):
    user, password = _configure(monkeypatch)
    assert auth.credentials() == (user, password)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_the_escape_hatch_opens_the_gate_only_when_set_deliberately(
        monkeypatch, value):
    monkeypatch.setenv(auth.DISABLE_KEY, value)
    assert auth.auth_disabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "maybe"])
def test_anything_that_is_not_an_affirmative_leaves_the_gate_shut(
        monkeypatch, value):
    monkeypatch.setenv(auth.DISABLE_KEY, value)
    assert auth.auth_disabled() is False


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------
def test_the_right_pair_signs_in():
    assert auth.check("orion", "pw", "orion", "pw") is True


@pytest.mark.parametrize("user,password", [
    ("orion", "wrong"),          # right user, wrong password
    ("someone", "pw"),           # wrong user, right password
    ("someone", "wrong"),        # neither
    ("", ""),                    # an empty submit
    ("orion", "pw "),            # a trailing space is a different password
    ("Orion", "pw"),             # case matters
])
def test_every_other_pair_is_rejected(user, password):
    assert auth.check(user, password, "orion", "pw") is False


def test_the_comparison_does_not_short_circuit_on_the_username():
    """Both halves must be evaluated, so `and` cannot creep back in.

    A short-circuiting `and` would return before touching the password when
    the username is wrong, which is measurable and makes usernames
    enumerable. The observable proxy is that a wrong username with a
    *matching* password still returns False rather than raising or passing.
    """
    assert auth.check("someone", "pw", "orion", "pw") is False
    assert auth.check("orion", "wrong", "orion", "pw") is False


# --------------------------------------------------------------------------
# the throttle
# --------------------------------------------------------------------------
def test_a_clean_bucket_waits_for_nothing():
    assert auth.lockout_remaining({}, now=1000.0) == 0.0


def test_failures_back_off_before_they_lock_out():
    state: dict = {}
    auth.register_failure(state, now=1000.0)
    first = auth.lockout_remaining(state, now=1000.0)
    auth.register_failure(state, now=1000.0)
    second = auth.lockout_remaining(state, now=1000.0)

    assert 0 < first < second <= auth.MAX_BACKOFF_SECONDS


def test_the_backoff_is_capped_below_the_lockout():
    """Four failures must still be a nuisance, not a five-minute wall."""
    state = {"fails": auth.MAX_FAILS - 1, "last": 1000.0}
    assert auth.lockout_remaining(state, now=1000.0) <= auth.MAX_BACKOFF_SECONDS


def test_the_lockout_engages_at_max_fails_and_expires():
    state: dict = {}
    for _ in range(auth.MAX_FAILS):
        auth.register_failure(state, now=1000.0)

    assert auth.lockout_remaining(state, now=1000.0) == auth.LOCKOUT_SECONDS
    assert auth.lockout_remaining(
        state, now=1000.0 + auth.LOCKOUT_SECONDS - 1) == pytest.approx(1.0)
    assert auth.lockout_remaining(state, now=1000.0 + auth.LOCKOUT_SECONDS) == 0.0


def test_a_failure_after_the_lockout_expires_re_arms_it():
    """Waiting out the lockout must not buy an unlimited retry budget."""
    state = {"fails": auth.MAX_FAILS, "last": 1000.0}
    later = 1000.0 + auth.LOCKOUT_SECONDS + 1
    assert auth.lockout_remaining(state, now=later) == 0.0

    auth.register_failure(state, now=later)
    assert auth.lockout_remaining(state, now=later) == auth.LOCKOUT_SECONDS


def test_a_successful_sign_in_clears_the_bucket():
    state = {"fails": 3, "last": 1000.0}
    auth.clear(state)
    assert auth.lockout_remaining(state, now=1000.0) == 0.0


def test_the_unknown_client_bucket_can_never_lock_anyone_out():
    """A global lockout is a self-DoS any stranger could trigger.

    Every client whose IP could not be established shares one bucket, so if
    that bucket could reach LOCKOUT_SECONDS, one attacker would lock out the
    owner. It is capped at the backoff instead.
    """
    state = {"fails": auth.MAX_FAILS * 10, "last": 1000.0}
    assert auth.lockout_remaining(state, now=1000.0) == auth.LOCKOUT_SECONDS
    assert auth.wait_for(
        auth.UNKNOWN_CLIENT, state, now=1000.0) == auth.MAX_BACKOFF_SECONDS
    assert auth.wait_for("203.0.113.7", state, now=1000.0) == auth.LOCKOUT_SECONDS


def test_reading_the_throttle_never_allocates_a_bucket():
    """A page load must not cost memory, or the bot traffic IS the attack.

    `X-Forwarded-For` is client-supplied, so if merely rendering the form
    created a bucket, one forged header per request would grow the table
    forever - a memory leak driven by exactly the traffic this resists.
    """
    auth._attempts().clear()
    for _ in range(50):
        auth.wait_for("203.0.113.9", auth._attempts().get("203.0.113.9", {}),
                      now=1000.0)
    assert auth._attempts() == {}


def test_the_bucket_table_is_capped_against_forged_client_addresses():
    auth._attempts().clear()
    now = 1000.0
    for i in range(auth.MAX_TRACKED_CLIENTS + 200):
        auth.note_failure(f"198.51.100.{i}", now)

    assert len(auth._attempts()) <= auth.MAX_TRACKED_CLIENTS
    auth._attempts().clear()


def test_expired_buckets_are_pruned_rather_than_accumulated():
    """An expired lockout is no longer evidence, so it should not be kept."""
    auth._attempts().clear()
    auth.note_failure("203.0.113.1", now=1000.0)
    assert "203.0.113.1" in auth._attempts()

    auth.note_failure("203.0.113.2", now=1000.0 + auth.LOCKOUT_SECONDS + 1)
    assert "203.0.113.1" not in auth._attempts(), "an expired bucket was kept"
    assert "203.0.113.2" in auth._attempts()
    auth._attempts().clear()


def test_the_client_key_is_always_a_string(monkeypatch):
    """A truthy non-string would be wrong in the DANGEROUS direction.

    `st.context` yields mocks under some runtimes. A mock is truthy, so a
    plain `or ""` kept it, and the resulting key can never equal
    UNKNOWN_CLIENT - which would promote an unidentified client out of the
    capped shared bucket and into the full five-minute lockout, silently.
    """
    class Mocky:
        def __bool__(self):
            return True

        def get(self, _key):
            return Mocky()

        def split(self, _sep):
            return [Mocky()]

        def strip(self):
            return Mocky()

    class FakeContext:
        headers = Mocky()
        ip_address = Mocky()

    monkeypatch.setattr(auth.st, "context", FakeContext())
    key = auth.client_key()
    assert isinstance(key, str)
    assert key == auth.UNKNOWN_CLIENT


def test_the_client_key_prefers_the_forwarded_first_hop(monkeypatch):
    """On Cloud the socket peer is the proxy - one bucket for every visitor."""
    class FakeContext:
        headers = {"X-Forwarded-For": "203.0.113.5, 70.41.3.18"}
        ip_address = "10.0.0.1"

    monkeypatch.setattr(auth.st, "context", FakeContext())
    assert auth.client_key() == "203.0.113.5"


def test_the_client_key_falls_back_to_the_socket_peer(monkeypatch):
    class FakeContext:
        headers: dict = {}
        ip_address = "10.0.0.1"

    monkeypatch.setattr(auth.st, "context", FakeContext())
    assert auth.client_key() == "10.0.0.1"


def test_an_unidentifiable_client_lands_in_the_shared_bucket(monkeypatch):
    class FakeContext:
        headers: dict = {}
        ip_address = None

    monkeypatch.setattr(auth.st, "context", FakeContext())
    assert auth.client_key() == auth.UNKNOWN_CLIENT


def test_a_successful_sign_in_forgets_that_client_entirely():
    auth._attempts().clear()
    auth.note_failure("203.0.113.3", now=1000.0)
    auth.forget("203.0.113.3")
    assert "203.0.113.3" not in auth._attempts()


# --------------------------------------------------------------------------
# the secrets bridge
# --------------------------------------------------------------------------
def test_the_bridge_never_overwrites_a_real_environment_variable(monkeypatch):
    """`os.environ` is authoritative, or env and secrets disagree silently."""
    monkeypatch.setenv("KITE_API_KEY", "from-the-environment")

    class FakeSecrets:
        def items(self):
            return [("KITE_API_KEY", "from-secrets"),
                    ("TELEGRAM_CHAT_ID", "12345"),
                    ("a_section", {"nested": "ignored"})]

    monkeypatch.setattr(auth.st, "secrets", FakeSecrets())
    auth.bridge_secrets()

    import os
    assert os.environ["KITE_API_KEY"] == "from-the-environment"
    assert os.environ["TELEGRAM_CHAT_ID"] == "12345"
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


def test_the_bridge_is_a_no_op_without_a_secrets_file(monkeypatch):
    """A local run has no secrets.toml, and `st.secrets` raises. Not a crash."""
    class Exploding:
        def items(self):
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setattr(auth.st, "secrets", Exploding())
    auth.bridge_secrets()          # must not raise


# --------------------------------------------------------------------------
# end to end, against the real app.py
# --------------------------------------------------------------------------
def test_an_unauthenticated_session_gets_the_form_and_no_pages(monkeypatch):
    """The gate is above the page imports, so nothing renders behind it.

    This is a claim about `app.py`'s import order rather than about `auth.py`,
    and it is the whole reason the gate is at module level: `st.stop()` here
    means no engine, no feed, no journal, no broker and no refresh fragment
    is ever constructed for a visitor who never signs in.
    """
    _configure(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=180).run()

    assert not at.exception, _why(at)
    assert at.text_input, "the sign-in form did not render"
    assert not at.sidebar.radio, "the page picker rendered behind the gate"


def test_a_wrong_password_typed_into_the_form_is_rejected(monkeypatch):
    """The form wiring, not just `check()`.

    Reading the wrong widget, or comparing the username field against the
    password, would pass every test above and still be a gate that opens on
    anything - so this drives the actual form.
    """
    _configure(monkeypatch, "orion", "pw123")
    at = AppTest.from_file(APP, default_timeout=180).run()

    at.text_input[0].set_value("orion")
    at.text_input[1].set_value("not-the-password")
    at.button[0].click().run()

    assert not at.exception, _why(at)
    assert any("wrong username or password" in e.value.lower() for e in at.error)
    assert not at.sidebar.radio, "a rejected sign-in still rendered the pages"


def test_a_wrong_username_is_rejected_and_still_costs_an_attempt(monkeypatch):
    """The right password under the wrong username must not be a near miss.

    The attempt count is the observable half of the constant-time comparison:
    if `check()` ever short-circuits on the username, the natural next step is
    to stop charging for it, and enumeration becomes free.
    """
    _configure(monkeypatch, "orion", "pw123")
    at = AppTest.from_file(APP, default_timeout=180).run()

    at.text_input[0].set_value("someone-else")
    at.text_input[1].set_value("pw123")
    at.button[0].click().run()

    assert not at.exception, _why(at)
    assert any("wrong username or password" in e.value.lower() for e in at.error)
    assert not at.sidebar.radio
    charged = sum(int(b.get("fails", 0)) for b in auth._attempts().values())
    assert charged == 1, "a wrong username was not charged an attempt"


def test_the_right_pair_typed_into_the_form_opens_the_app(monkeypatch):
    """And the console behind it is the one that was always there."""
    _configure(monkeypatch, "orion", "pw123")
    at = AppTest.from_file(APP, default_timeout=180)
    seed_offline_broker(at, Config())      # never a real broker - see conftest
    at.run()

    at.text_input[0].set_value("orion")
    at.text_input[1].set_value("pw123")
    at.button[0].click().run()

    assert not at.exception, _why(at)
    assert at.sidebar.radio, "the correct password did not open the app"
    assert "Live alerts" in at.sidebar.radio[0].options
    assert any(b.label == "Sign out" for b in at.sidebar.button), \
        "no way back out once signed in"


def test_the_gate_refuses_to_open_when_it_is_not_configured():
    """Deployed with no APP_PASSWORD, the app must be shut, not wide open."""
    at = AppTest.from_file(APP, default_timeout=180).run()

    assert not at.exception, _why(at)
    assert not at.sidebar.radio, "an unconfigured app rendered its pages"
    assert any("not configured" in e.value.lower() for e in at.error), \
        "the app did not say why it refused"


def _why(at) -> str:
    return "; ".join(f"{type(e.value).__name__}: {e.value}"
                     for e in (at.exception or []))
