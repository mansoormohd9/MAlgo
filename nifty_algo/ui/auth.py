"""
The front door: a username/password gate in front of the whole console.

WHY THIS EXISTS. `app.py` is deployed to Streamlit Community Cloud, which
means a public URL, and this app is not a demo over sample data - it is a
viewer onto one real account. Unauthenticated, a stranger who found that URL
would be operating it: `page_live` and `page_brief` hit the feed and the Kite
chain ON RENDER, and with sidebar auto-refresh on `st.fragment(run_every=...)`
keeps doing so every few seconds without anybody clicking anything; Settings
has a switch that flips `dry_run` off; Trade book has a bare `Sell {qty}` per
position and arms GTTs; Journal discloses the whole trade log.

WHERE THE GATE SITS, AND WHY THAT IS THE POINT. `require_login()` is called
from `app.py` between `st.set_page_config()` and the `nifty_algo.ui.page_*`
imports, and it calls `st.stop()`. That raises out of the script run, so an
unauthenticated session never imports a page module and therefore never
constructs the engine, the feed, the journal, the broker or the refresh
fragment. The bot protection is not a CAPTCHA - it is that a crawler's visit
costs one text form and nothing else. `nifty_algo/ui/__init__.py` is a bare
docstring, which is what keeps importing THIS module free.

IT FAILS CLOSED. With `APP_PASSWORD` unset the gate refuses to open rather
than waving everyone through; the single escape hatch is an explicit
`APP_AUTH_DISABLED=1`, so running without a gate is a decision somebody typed.
Sniffing the hostname to decide "am I deployed?" was rejected deliberately: it
makes the safe path depend on a guess about the environment.

THE THROTTLE IS PER-CLIENT AND NEVER GLOBAL. A global lockout is a self-DoS
that any stranger can trigger, so `_attempts()` is keyed by client IP, and the
bucket for clients whose IP could not be established is capped at a delay and
can never reach the lockout. Note also that a rejected attempt is rejected
WITHOUT sleeping: holding the server thread open is what an attacker wants,
and the wait is enforced on the next attempt instead.

AND THE THROTTLE IS BOUNDED, because `X-Forwarded-For` is client-supplied.
Anyone who can forge that header can both evade the throttle and mint an
unlimited number of buckets, so buckets are created only by a FAILED attempt
(never by merely loading the page), expired ones are pruned, and the table is
capped at `MAX_TRACKED_CLIENTS`. A memory leak driven by exactly the traffic
this module exists to resist would be a poor trade. This is why the throttle
is a cost-reducer and the GATE is the security control: the throttle is
best-effort, `st.stop()` is not.

WHAT THIS DELIBERATELY DOES NOT DO: try to keep itself out of search results.
A `<meta name="robots">` emitted through `st.markdown` survives the server and
is then stripped by the frontend's HTML sanitiser, so it would be a control
that reads as armed and does nothing - the exact failure this repo refuses
elsewhere. Community Cloud serves its own `index.html` and there is no hook
for one. Use the Cloud private-app viewer allowlist instead, which rejects
unauthenticated traffic before this script ever runs; see README.

The credential pair is plaintext, compared in constant time, and read from
`os.environ` first and `st.secrets` second - one source of truth for both a
local `.env` run and a Cloud deployment, and rotation is editing one value.
The pure rules (`check`, `register_failure`, `lockout_remaining`) take plain
arguments and a plain dict so they are testable without a Streamlit runtime,
the same split `governor.py` uses.
"""
from __future__ import annotations

import hmac
import os
import time

import streamlit as st

USER_KEY = "APP_USERNAME"
PASSWORD_KEY = "APP_PASSWORD"
DISABLE_KEY = "APP_AUTH_DISABLED"

#: Failures allowed before a client is locked out rather than merely slowed.
MAX_FAILS = 5
#: How long a locked-out client waits. Reached at MAX_FAILS and re-armed by
#: every further failure, so a patient attacker gains nothing by waiting.
LOCKOUT_SECONDS = 300.0
#: Ceiling on the pre-lockout backoff, and the ceiling on the whole wait for
#: the shared "IP unknown" bucket.
MAX_BACKOFF_SECONDS = 8.0

#: The bucket used when no client IP could be established. Shared by every
#: such client, which is exactly why it is never allowed to lock out.
UNKNOWN_CLIENT = "unknown"

#: Ceiling on retained client buckets. X-Forwarded-For is client-supplied, so
#: without a cap one forged header per request grows this table forever.
MAX_TRACKED_CLIENTS = 2048

_TRUTHY = {"1", "true", "yes", "on"}

_SIGNED_IN = "_auth_ok"


# --------------------------------------------------------------------------
# configuration - env wins over secrets, because bridge_secrets() never
# overwrites an env var and the two must not disagree about which is which.
# --------------------------------------------------------------------------
def _lookup(key: str) -> str | None:
    value = os.environ.get(key)
    if value:
        return value
    try:
        value = st.secrets[key]           # raises when there is no secrets file
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def auth_disabled() -> bool:
    """True only when somebody deliberately turned the gate off."""
    return (_lookup(DISABLE_KEY) or "").strip().lower() in _TRUTHY


def credentials() -> tuple[str, str] | None:
    """The configured pair, or None when either half is missing.

    Both halves are required. A blank username with a set password would be a
    gate whose username field is decoration, and it would look configured.
    """
    user = (_lookup(USER_KEY) or "").strip()
    password = _lookup(PASSWORD_KEY) or ""
    if not user or not password:
        return None
    return user, password


def bridge_secrets() -> None:
    """Copy flat `st.secrets` keys into `os.environ`, never overwriting.

    There is no `.env` on Streamlit Cloud, and fourteen existing readers -
    KITE_*, TELEGRAM_*, SMTP_*, FYERS_*, DHAN_* - all use `os.getenv`.
    Bridging once at startup means not one of those call sites changes, and
    "never overwrite" keeps a real environment variable authoritative.
    """
    try:
        items = list(st.secrets.items())
    except Exception:
        return                            # no secrets file: a local run
    for key, value in items:
        if isinstance(value, str) and value:
            os.environ.setdefault(key, value)


# --------------------------------------------------------------------------
# pure rules
# --------------------------------------------------------------------------
def check(user: str, password: str, expect_user: str, expect_password: str) -> bool:
    """Constant-time credential comparison.

    Both halves are ALWAYS evaluated and combined with `&` rather than `and`,
    which short-circuits: a wrong username must not return faster than a wrong
    password, or the username becomes enumerable.
    """
    ok_user = hmac.compare_digest(user.encode("utf-8"), expect_user.encode("utf-8"))
    ok_password = hmac.compare_digest(
        password.encode("utf-8"), expect_password.encode("utf-8")
    )
    return bool(ok_user & ok_password)


def register_failure(state: dict, now: float) -> None:
    """Record one rejected attempt against a client bucket."""
    state["fails"] = int(state.get("fails", 0)) + 1
    state["last"] = float(now)


def clear(state: dict) -> None:
    """Reset one bucket in place. `forget()` is the Streamlit-side wrapper."""
    state.pop("fails", None)
    state.pop("last", None)


def lockout_remaining(state: dict, now: float) -> float:
    """Seconds before this bucket's next attempt is even looked at."""
    fails = int(state.get("fails", 0))
    if fails <= 0:
        return 0.0
    wait = (LOCKOUT_SECONDS if fails >= MAX_FAILS
            else min(2.0**fails, MAX_BACKOFF_SECONDS))
    return max(0.0, float(state.get("last", 0.0)) + wait - now)


def wait_for(client: str, state: dict, now: float) -> float:
    """`lockout_remaining`, capped for the shared unknown-client bucket."""
    remaining = lockout_remaining(state, now)
    if client == UNKNOWN_CLIENT:
        return min(remaining, MAX_BACKOFF_SECONDS)
    return remaining


# --------------------------------------------------------------------------
# Streamlit shell
# --------------------------------------------------------------------------
@st.cache_resource
def _attempts() -> dict[str, dict]:
    """Failure buckets shared by every session in this container.

    `st.session_state` is per browser session, so a per-session counter alone
    is bypassed by reconnecting. This survives that.
    """
    return {}


def note_failure(client: str, now: float) -> None:
    """Charge one failed attempt to a client, keeping the table bounded.

    A bucket is created HERE and nowhere else - reading the throttle must not
    allocate, or every page load from a forged IP would cost memory.
    """
    buckets = _attempts()
    for key in [k for k, s in buckets.items() if lockout_remaining(s, now) <= 0]:
        del buckets[key]                  # expired: it is no longer evidence
    if client not in buckets and len(buckets) >= MAX_TRACKED_CLIENTS:
        # Still full of live lockouts. Drop the least recent, so a flood of
        # forged IPs cannot grow the table without bound.
        del buckets[min(buckets, key=lambda k: buckets[k].get("last", 0.0))]
    register_failure(buckets.setdefault(client, {}), now)


def forget(client: str) -> None:
    """Drop a client's failures - called on a successful sign-in."""
    _attempts().pop(client, None)


def client_key() -> str:
    """A coarse client identity for throttling. Never trusted for identity.

    The first hop of `X-Forwarded-For` is preferred over `st.context.ip_address`
    because on Cloud the socket peer is the proxy, which would put every
    visitor in one bucket - and one bucket is the global lockout this refuses
    to have.

    Every branch is `isinstance`-checked rather than merely truth-tested. A
    non-string that is truthy - which is what `st.context` yields under some
    runtimes - would otherwise become the bucket key, and since it can never
    equal `UNKNOWN_CLIENT` it would be treated as a KNOWN client and made
    eligible for the full lockout instead of the capped shared bucket. That is
    a throttle that is wrong in the dangerous direction and raises nothing.
    """
    try:
        headers = st.context.headers or {}
        forwarded = headers.get("X-Forwarded-For")
        candidate = forwarded.split(",")[0] if isinstance(forwarded, str) else ""
        if not candidate.strip():
            address = st.context.ip_address
            candidate = address if isinstance(address, str) else ""
        return candidate.strip() or UNKNOWN_CLIENT
    except Exception:
        return UNKNOWN_CLIENT


def is_authenticated() -> bool:
    return auth_disabled() or bool(st.session_state.get(_SIGNED_IN))


def require_login() -> None:
    """Render the gate and STOP the script unless this session is signed in."""
    if is_authenticated():
        return
    creds = credentials()
    if creds is None:
        _unconfigured()
    else:
        _form(creds)
    st.stop()


def logout_control() -> None:
    """A Sign out button. No-op when there is no gate to sign out of."""
    if auth_disabled() or not st.session_state.get(_SIGNED_IN):
        return
    if st.button("Sign out", width="stretch"):
        st.session_state.pop(_SIGNED_IN, None)
        st.rerun()


def _unconfigured() -> None:
    st.error(
        f"**Login is not configured, so this app will not open.**\n\n"
        f"Set `{USER_KEY}` and `{PASSWORD_KEY}` - in `.env` locally, or in "
        f"**Settings -> Secrets** on Streamlit Cloud. To run with no gate at "
        f"all, set `{DISABLE_KEY}=1` deliberately."
    )


def _form(creds: tuple[str, str]) -> None:
    expect_user, expect_password = creds
    client = client_key()
    # .get, not .setdefault: reading the throttle must never allocate.
    waiting = wait_for(client, _attempts().get(client, {}), time.time())

    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.markdown("### 📈 Nifty Algo")
        st.caption("Alert console — sign in to continue.")
        with st.form("sign_in"):
            user = st.text_input("Username", autocomplete="username")
            password = st.text_input(
                "Password", type="password", autocomplete="current-password"
            )
            submitted = st.form_submit_button("Sign in", width="stretch")

        if submitted and waiting > 0:
            st.error(f"Too many failed attempts. Try again in {waiting:.0f}s.")
        elif submitted and check(user, password, expect_user, expect_password):
            forget(client)
            st.session_state[_SIGNED_IN] = True
            st.rerun()
        elif submitted:
            note_failure(client, time.time())
            st.error("Wrong username or password.")
        elif waiting > 0:
            st.warning(f"Locked for another {waiting:.0f}s.")

        st.caption(
            "Not investment advice. SEBI data shows over 90% of retail F&O "
            "traders lose money."
        )
