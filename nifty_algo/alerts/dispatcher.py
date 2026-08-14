"""
Fan-out with de-duplication.

The de-duplication is not a nicety, it is what makes the whole alerting
system usable. Consider the arithmetic: the UI refreshes every 15 seconds, a
bar lasts 5 minutes, and a setup that is true on that bar stays true for the
whole bar. That is 20 identical alerts per setup, per channel. Multiply by
four channels and one breakout produces 80 messages.

Two independent guards:

  dedupe_window   the same alert (same strategy, direction, strike, bar) is
                  sent once and suppressed for this many minutes.

  cooldown        a strategy that just alerted stays quiet for this long even
                  if it produces a genuinely DIFFERENT setup. This exists
                  because you only get three entries a day - being pinged
                  four times in six minutes by one strategy is not useful
                  information, it is a strategy misbehaving.

Every attempt, delivered or failed, is handed to the journal callback. A
silent delivery failure would be indistinguishable from "no setup today",
which is the worst possible failure mode for this system.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Callable, Iterable

from ..config import Config, DEFAULT
from .base import Notifier, TradeAlert, AlertKind, POSITION_KINDS


class AlertDispatcher:
    def __init__(self, notifiers: Iterable[Notifier],
                 cfg: Config = DEFAULT,
                 journal_fn: Callable[[str, dict], None] | None = None):
        self.notifiers = list(notifiers)
        self.cfg = cfg
        self.journal_fn = journal_fn
        self._sent: dict[str, datetime] = {}          # dedupe key -> when
        self._last_by_strategy: dict[str, datetime] = {}

    # ---------------- suppression ----------------

    def _suppressed(self, alert: TradeAlert, now: datetime) -> str | None:
        # Operational alerts are never suppressed by cooldown. A kill switch
        # that gets rate-limited is a kill switch that did not fire.
        if alert.kind in (AlertKind.KILL_SWITCH, AlertKind.TEST):
            return None

        # Nor are alerts about a position you already hold. Suppressing a
        # duplicate ENTRY costs an opportunity; suppressing a stop move or an
        # exit costs the trade. These describe something that has ALREADY
        # happened to your money and must always get through.
        if alert.kind in POSITION_KINDS or alert.kind is AlertKind.FORCE_EXIT:
            return None

        a = self.cfg.alerts

        seen_at = self._sent.get(alert.dedupe_key)
        if seen_at and now - seen_at < timedelta(minutes=a.dedupe_window_minutes):
            return f"duplicate within {a.dedupe_window_minutes}m window"

        if alert.kind is AlertKind.ENTRY and alert.strategy_key:
            last = self._last_by_strategy.get(alert.strategy_key)
            if last and now - last < timedelta(minutes=a.per_strategy_cooldown_minutes):
                mins = (now - last).total_seconds() / 60
                return (f"{alert.strategy_key} cooling down "
                        f"({mins:.1f}m of {a.per_strategy_cooldown_minutes}m)")
        return None

    # ---------------- dispatch ----------------

    def dispatch(self, alert: TradeAlert,
                 force: bool = False) -> dict[str, tuple[bool, str]] | None:
        """
        Send to every enabled channel. `force` bypasses suppression - used by
        the Settings test button.

        Returns None when the alert was SUPPRESSED, and a dict of
        {channel: (delivered, detail)} when it was not - a dict that is empty
        when no channel happens to be enabled.

        That distinction matters. Both cases used to return {}, so a caller
        could not tell "this is a duplicate, drop it" from "this is new, but
        you have every channel switched off". The engine got that wrong: with
        all channels disabled it dropped genuine alerts from the UI feed and
        labelled them 'duplicate or within cooldown', which was simply untrue.
        """
        now = alert.timestamp or datetime.now()

        if not force:
            reason = self._suppressed(alert, now)
            if reason:
                self._journal("alert_suppressed",
                              {"key": alert.dedupe_key, "reason": reason})
                return None

        results: dict[str, tuple[bool, str]] = {}
        for n in self._enabled():
            try:
                results[n.name] = n.send(alert)
            except Exception as e:
                # Defence in depth: channels promise not to raise, but a
                # broken one must never propagate into the decision loop.
                results[n.name] = (False, f"unhandled channel error: {e}")

        self._sent[alert.dedupe_key] = now
        if alert.kind is AlertKind.ENTRY and alert.strategy_key:
            self._last_by_strategy[alert.strategy_key] = now

        self._journal("alert_sent", {
            "alert": alert.to_dict(),
            "delivery": {k: {"ok": v[0], "detail": v[1]} for k, v in results.items()},
        })
        return results

    def _enabled(self) -> list[Notifier]:
        a = self.cfg.alerts
        flags = {
            "inapp": a.enable_inapp,
            "telegram": a.enable_telegram,
            "desktop": a.enable_desktop,
            "email": a.enable_email,
        }
        return [n for n in self.notifiers if flags.get(n.name, True)]

    def enabled_names(self) -> list[str]:
        """Which channels an alert would actually go to right now."""
        return [n.name for n in self._enabled()]

    def _journal(self, event: str, payload: dict) -> None:
        if self.journal_fn:
            try:
                self.journal_fn(event, payload)
            except Exception:
                pass          # journalling must never break dispatch

    # ---------------- introspection for the UI ----------------

    def channel_status(self) -> list[dict]:
        a = self.cfg.alerts
        flags = {
            "inapp": a.enable_inapp,
            "telegram": a.enable_telegram,
            "desktop": a.enable_desktop,
            "email": a.enable_email,
        }
        return [
            {"name": n.name,
             "enabled": flags.get(n.name, True),
             "configured": n.configured}
            for n in self.notifiers
        ]

    def reset_suppression(self) -> None:
        self._sent.clear()
        self._last_by_strategy.clear()
