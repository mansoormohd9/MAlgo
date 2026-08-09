"""
The four delivery channels.

A hard rule runs through all of them: `send()` never raises. A Telegram outage,
a wrong SMTP password, or a missing desktop-toast backend must degrade to a
logged failure, never take down the decision loop. Losing an alert is bad;
losing the engine mid-session is worse.

Each channel reports `configured` so the Settings page can show what is
actually wired up rather than what was merely ticked.
"""
from __future__ import annotations
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Callable

from .base import Notifier, TradeAlert, AlertKind


# ---------------------------------------------------------------- in-app

class InAppNotifier(Notifier):
    """
    Pushes alerts into a list the Streamlit UI renders.

    Always "configured" - it needs nothing. It is also the only channel that
    works with zero setup, and the only one that does NOT work when the
    browser tab is closed. That asymmetry is why `run_live.py` exists.
    """
    name = "inapp"

    def __init__(self, sink: Callable[[TradeAlert], None] | None = None,
                 max_kept: int = 100):
        self.alerts: list[TradeAlert] = []
        self.sink = sink
        self.max_kept = max_kept

    @property
    def configured(self) -> bool:
        return True

    def send(self, alert: TradeAlert) -> tuple[bool, str]:
        try:
            self.alerts.append(alert)
            del self.alerts[:-self.max_kept]
            if self.sink:
                self.sink(alert)
            return True, "queued for UI"
        except Exception as e:                      # pragma: no cover
            return False, f"in-app queue failed: {e}"


# ---------------------------------------------------------------- telegram

class TelegramNotifier(Notifier):
    """
    Telegram Bot API. The only channel that reliably reaches you away from
    the desk, which makes it the one worth configuring first.

    Setup: create a bot with @BotFather, send it any message, then read your
    chat id from https://api.telegram.org/bot<TOKEN>/getUpdates
    """
    name = "telegram"

    def __init__(self, token: str | None = None, chat_id: str | None = None,
                 timeout: int = 10):
        self.token = (token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, alert: TradeAlert) -> tuple[bool, str]:
        if not self.configured:
            return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env"
        try:
            import requests
            text = f"*{self._escape(alert.title())}*\n```\n{alert.as_text()}\n```"
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text,
                      "parse_mode": "Markdown"},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return False, f"telegram HTTP {resp.status_code}: {resp.text[:200]}"
            return True, "sent"
        except Exception as e:
            return False, f"telegram failed: {e}"

    @staticmethod
    def _escape(s: str) -> str:
        for ch in ("_", "*", "[", "]", "`"):
            s = s.replace(ch, f"\\{ch}")
        return s


# ---------------------------------------------------------------- desktop

class DesktopNotifier(Notifier):
    """
    Windows toast via plyer. Works when the browser is minimised, but only on
    the machine running the engine - so it complements Telegram rather than
    replacing it.

    Toast bodies are truncated by the OS, so this sends a compressed line and
    leaves the full detail to the other channels.
    """
    name = "desktop"

    def __init__(self, app_name: str = "Nifty Algo", timeout_seconds: int = 20):
        self.app_name = app_name
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        try:
            import plyer  # noqa: F401
            return True
        except Exception:
            return False

    def send(self, alert: TradeAlert) -> tuple[bool, str]:
        if not self.configured:
            return False, "plyer not installed - pip install plyer"
        try:
            from plyer import notification

            if alert.kind is AlertKind.ENTRY:
                body = (f"Entry {alert.entry_premium:.2f} | "
                        f"Tgt {alert.target_premium:.2f} | "
                        f"SL {alert.stop_premium:.2f} | Qty {alert.quantity}")
            else:
                body = alert.message[:200]

            notification.notify(
                title=alert.title()[:64],
                message=body[:240],
                app_name=self.app_name,
                timeout=self.timeout_seconds,
            )
            return True, "toast shown"
        except Exception as e:
            return False, f"desktop toast failed: {e}"


# ---------------------------------------------------------------- email

class EmailNotifier(Notifier):
    """
    SMTP. The slowest channel - provider-side delays of a minute or more are
    normal, which is a long time against a 5-minute bar. Useful as a durable
    written record of the day's alerts; poor as your primary trigger.

    Gmail needs an App Password, not the account password.
    """
    name = "email"

    def __init__(self, host: str | None = None, port: int | None = None,
                 user: str | None = None, password: str | None = None,
                 to_addr: str | None = None, timeout: int = 15):
        self.host = (host or os.getenv("SMTP_HOST", "")).strip()
        self.port = int(port or os.getenv("SMTP_PORT", "587") or 587)
        self.user = (user or os.getenv("SMTP_USER", "")).strip()
        self.password = (password or os.getenv("SMTP_PASSWORD", "")).strip()
        self.to_addr = (to_addr or os.getenv("SMTP_TO", "")).strip() or self.user
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.host and self.user and self.password and self.to_addr)

    def send(self, alert: TradeAlert) -> tuple[bool, str]:
        if not self.configured:
            return False, "SMTP_HOST / SMTP_USER / SMTP_PASSWORD / SMTP_TO not set in .env"
        try:
            msg = EmailMessage()
            msg["Subject"] = f"[Nifty Algo] {alert.title()}"
            msg["From"] = self.user
            msg["To"] = self.to_addr
            msg.set_content(alert.as_text())

            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as s:
                s.starttls()
                s.login(self.user, self.password)
                s.send_message(msg)
            return True, "sent"
        except Exception as e:
            return False, f"email failed: {e}"


def build_test_alert() -> TradeAlert:
    """Payload for the Settings page's per-channel test button."""
    return TradeAlert(
        kind=AlertKind.TEST,
        timestamp=datetime.now(),
        message=("Channel test from the Nifty algo UI. If you can read this, "
                 "this channel will deliver your trade alerts."),
    )
