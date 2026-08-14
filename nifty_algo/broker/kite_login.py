"""
Daily Kite login. Run this once each trading morning, before the engine.

    python -m nifty_algo.broker.kite_login

It prints a login URL, waits for you to paste back the request_token from the
redirect, and caches the resulting access token for the rest of the day.
"""
from __future__ import annotations
import sys
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv

from .kite_auth import KiteSession, NotAuthenticated


def extract_request_token(pasted: str) -> str:
    """
    Accept either the bare token or the whole redirect URL.

    The redirect URL is what is actually in the address bar, so pasting it
    directly is what everyone does first. Parsing it here saves a round of
    "it says invalid token".
    """
    pasted = pasted.strip()
    if not pasted:
        return ""
    if "request_token=" in pasted:
        qs = parse_qs(urlparse(pasted).query)
        values = qs.get("request_token", [])
        return values[0].strip() if values else ""
    return pasted


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    argv = argv if argv is not None else sys.argv[1:]
    session = KiteSession()

    try:
        url = session.login_url()
    except NotAuthenticated as e:
        print(f"\n  {e}\n")
        return 2

    if argv:
        token = extract_request_token(argv[0])
    else:
        print("\n  1. Open this URL and log in to Kite:\n")
        print(f"     {url}\n")
        print("  2. You will be redirected to your app's redirect URL. Copy the")
        print("     whole address from the browser bar and paste it below.\n")
        try:
            token = extract_request_token(input("  Redirect URL or request_token: "))
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 1

    if not token:
        print("\n  No request_token found in that input.\n")
        return 2

    try:
        cached = session.exchange(token)
    except NotAuthenticated as e:
        print(f"\n  {e}\n")
        return 2

    print(f"\n  Authenticated as {cached.user_id or 'unknown user'}.")
    print(f"  Token cached in {session.session_file} and valid for today "
          f"({cached.issued_on}).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
