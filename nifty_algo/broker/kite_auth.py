"""
Kite Connect authentication.

Kite's access token is NOT a long-lived API key. It is issued per login and
dies overnight, which means this is a daily manual step, every trading day,
forever. There is no supported way around it - Zerodha requires the interactive
login for regulatory reasons. Any library promising otherwise is scraping the
login form and will break, silently, at the worst possible moment.

The flow:

    1. Open login_url() in a browser, log in.
    2. Kite redirects to your registered redirect URL with ?request_token=...
    3. Paste that token into exchange(). It is single-use and expires in minutes.
    4. The resulting access_token is cached to disk and reused all day.

We cache to `.kite_session.json` (gitignored) with the date it was issued, and
refuse to use it once stale. A stale token does not fail loudly at the broker -
it returns a 403 on the first data call, which the engine would convert into a
kill switch. Checking the date here turns that into a clear message instead.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

from ..data.base import NotConfigured

# Kite invalidates tokens in the early morning. Treat anything issued before
# this hour on a previous calendar day as dead.
TOKEN_EXPIRY_HOUR = 6

SESSION_FILE = Path(".kite_session.json")


class NotAuthenticated(NotConfigured):
    """No usable access token. Inherits NotConfigured so the engine's existing
    feed-error handling treats it as a configuration problem, not a transient
    outage worth retrying."""


@dataclass
class CachedSession:
    access_token: str
    issued_on: date
    user_id: str = ""

    def is_stale(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now()
        today = now.date()
        if self.issued_on == today:
            return False
        # A token issued yesterday still works until the early-morning cutoff,
        # which matters if you leave the engine running past midnight.
        if self.issued_on == today - timedelta(days=1):
            return now.time() >= time(TOKEN_EXPIRY_HOUR, 0)
        return True


class KiteSession:
    """
    Owns the credentials and the cached token. Hand this to KiteFeed,
    kite_chain and kite_orders so they all share one authenticated client
    rather than each building their own.
    """

    def __init__(self, api_key: str | None = None, api_secret: str | None = None,
                 session_file: Path | str = SESSION_FILE):
        self.api_key = (api_key or os.getenv("KITE_API_KEY", "")).strip()
        self.api_secret = (api_secret or os.getenv("KITE_API_SECRET", "")).strip()
        self.session_file = Path(session_file)
        self._kite = None
        self._cached: Optional[CachedSession] = None

    # ---------------- credentials ----------------

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _require_credentials(self) -> None:
        if not self.configured:
            raise NotAuthenticated(
                "Kite needs KITE_API_KEY and KITE_API_SECRET in .env. "
                "Create an app at https://developer.kite.trade/apps (Rs 500/month "
                "per key) and copy both values from the app page."
            )

    # ---------------- the login dance ----------------

    def login_url(self) -> str:
        self._require_credentials()
        kc = self._new_client()
        return kc.login_url()

    def exchange(self, request_token: str) -> CachedSession:
        """
        Turn a single-use request_token into an access_token and cache it.

        The request_token is valid for a few minutes and can only be redeemed
        once. If this raises TokenException you almost certainly re-used one.
        """
        self._require_credentials()
        request_token = request_token.strip()
        if not request_token:
            raise NotAuthenticated("empty request_token")

        kc = self._new_client()
        try:
            data = kc.generate_session(request_token, api_secret=self.api_secret)
        except Exception as e:
            raise NotAuthenticated(
                f"Kite rejected the request_token: {e}. These are single-use and "
                f"expire within minutes - fetch a fresh one from login_url()."
            ) from e

        cached = CachedSession(
            access_token=data["access_token"],
            issued_on=date.today(),
            user_id=data.get("user_id", ""),
        )
        self._save(cached)
        self._cached = cached
        self._kite = None                    # rebuild with the new token
        return cached

    # ---------------- the client ----------------

    def client(self):
        """
        An authenticated KiteConnect instance. Raises NotAuthenticated with
        instructions rather than returning something that will 403 later.
        """
        if self._kite is not None:
            return self._kite

        self._require_credentials()
        cached = self._load()
        if cached is None:
            raise NotAuthenticated(
                "No cached Kite session. Run `python -m nifty_algo.broker.kite_login` "
                "and follow the printed steps - this is required once per trading day."
            )
        if cached.is_stale():
            raise NotAuthenticated(
                f"Kite access token was issued on {cached.issued_on} and has expired. "
                f"Re-run `python -m nifty_algo.broker.kite_login`. Kite tokens die "
                f"overnight; there is no refresh token."
            )

        kc = self._new_client()
        kc.set_access_token(cached.access_token)
        self._kite = kc
        self._cached = cached
        return kc

    @property
    def authenticated(self) -> bool:
        """Non-raising check, for the UI to render a status badge."""
        try:
            self.client()
            return True
        except NotConfigured:
            return False

    def _new_client(self):
        try:
            from kiteconnect import KiteConnect
        except ImportError as e:
            raise NotAuthenticated(
                "kiteconnect is not installed. `pip install kiteconnect`."
            ) from e
        return KiteConnect(api_key=self.api_key)

    # ---------------- disk cache ----------------

    def _load(self) -> Optional[CachedSession]:
        if self._cached is not None:
            return self._cached
        if not self.session_file.exists():
            return None
        try:
            raw = json.loads(self.session_file.read_text())
            return CachedSession(
                access_token=raw["access_token"],
                issued_on=date.fromisoformat(raw["issued_on"]),
                user_id=raw.get("user_id", ""),
            )
        except Exception:
            # A corrupt cache is not worth crashing over; treat it as absent.
            return None

    def _save(self, cached: CachedSession) -> None:
        self.session_file.write_text(json.dumps({
            "access_token": cached.access_token,
            "issued_on": cached.issued_on.isoformat(),
            "user_id": cached.user_id,
        }, indent=2))
        # The token is a bearer credential for a live trading account.
        try:
            os.chmod(self.session_file, 0o600)
        except OSError:
            pass        # Windows ACLs; best-effort only
