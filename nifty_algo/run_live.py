"""
Headless runner.

    python -m nifty_algo.run_live --provider yfinance --telegram

This exists because Streamlit cannot notify you when the tab is closed. If
Telegram, desktop, and email alerts are to be worth anything, the decision
loop has to survive without a browser - so it lives here and in engine.py,
and the UI is only ever a viewer over it.

ALERT ONLY. This process deliberately attaches no broker, so `confirm_entry()`
is never called and nothing is ever ordered. It will still attach a REAL option
chain when Kite is authenticated, because a real chain makes the alert's strike
and premium real - reading a quote and placing an order are different
permissions, and only the first one is granted here.

Position management therefore does not run in this process either: there are
no positions to manage. Confirm entries from the Streamlit UI if you want the
exit ladder.
"""
from __future__ import annotations
import argparse
import sys
import time as time_mod
from datetime import datetime

from dotenv import load_dotenv

from .config import DEFAULT, Config
from .data.factory import build_feed, PROVIDERS
from .alerts.channels import (InAppNotifier, TelegramNotifier,
                              DesktopNotifier, EmailNotifier)
from .alerts.dispatcher import AlertDispatcher
from .data.chain import ChainProvider
from .engine import TradingEngine
from .journal import Journal
from .strategies.registry import all_strategies, default_enabled_keys


def build_chain_provider(cfg: Config) -> ChainProvider:
    """
    A real chain when Kite is authenticated, synthetic otherwise.

    Note `broker=None` at the TradingEngine call site: reading quotes and
    placing orders are separate permissions and this runner only takes the
    first.
    """
    try:
        from .broker.kite_auth import KiteSession
        from .broker.kite_chain import KiteChain
        session = KiteSession()
        if session.authenticated:
            return ChainProvider(cfg, broker_chain=KiteChain(session, cfg))
    except Exception:
        pass
    return ChainProvider(cfg)


def build_engine(cfg: Config, strategy_keys: list[str]) -> TradingEngine:
    journal = Journal()
    feed = build_feed(cfg)
    notifiers = [InAppNotifier(), TelegramNotifier(),
                 DesktopNotifier(), EmailNotifier()]
    dispatcher = AlertDispatcher(notifiers, cfg, journal_fn=journal.write)
    return TradingEngine(feed, dispatcher, cfg, strategy_keys, journal,
                         chain_provider=build_chain_provider(cfg),
                         broker=None)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    keys = [s.key for s in all_strategies()]

    p = argparse.ArgumentParser(
        prog="nifty_algo.run_live",
        description="Headless alert runner. Never places an order.")
    p.add_argument("--provider", choices=PROVIDERS, default=DEFAULT.data.provider)
    p.add_argument("--csv-path", default=DEFAULT.data.csv_path)
    p.add_argument("--interval", type=int, default=DEFAULT.data.poll_seconds,
                   help="seconds between evaluations")
    p.add_argument("--strategies", nargs="*", choices=keys, default=None,
                   help=f"default: {' '.join(default_enabled_keys())}")
    p.add_argument("--telegram", action="store_true")
    p.add_argument("--desktop", action="store_true")
    p.add_argument("--email", action="store_true")
    p.add_argument("--once", action="store_true", help="single pass then exit")
    args = p.parse_args(argv)

    cfg = DEFAULT
    cfg.data.provider = args.provider
    cfg.data.csv_path = args.csv_path
    cfg.alerts.enable_telegram = args.telegram
    cfg.alerts.enable_desktop = args.desktop
    cfg.alerts.enable_email = args.email

    engine = build_engine(cfg, args.strategies)

    print(f"Feed      : {engine.feed.name} — {engine.feed.latency_note}")
    chain_kind = ("broker (real quotes)"
                  if engine.chain_provider._broker_chain is not None
                  else "synthetic (Black-Scholes, fabricated OI and spread)")
    print(f"Chain     : {chain_kind}")
    print(f"Strategies: {', '.join(engine.strategies) or 'none'}")
    print(f"Channels  : {[c['name'] for c in engine.dispatcher.channel_status() if c['enabled']]}")
    for c in engine.dispatcher.channel_status():
        if c["enabled"] and not c["configured"]:
            print(f"  WARNING: {c['name']} is enabled but not configured — see .env")
    print("Alert only. This process never places an order.\n")

    try:
        while True:
            state = engine.run_once()
            stamp = datetime.now().strftime("%H:%M:%S")
            if state.last_error:
                print(f"[{stamp}] ERROR: {state.last_error}")
            else:
                regime = state.regime.regime.value if state.regime else "?"
                fired = [e for e in state.evaluations if e["outcome"] == "ALERT"]
                print(f"[{stamp}] {state.underlying_price:,.1f} "
                      f"regime={regime} halt={state.halt_reason} "
                      f"alerts={len(fired)}")
                for e in fired:
                    print(f"          -> {e['strategy']}: {e['detail']}")

            if args.once:
                return 0
            time_mod.sleep(max(args.interval, 5))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
