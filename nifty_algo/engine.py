"""
The decision loop.

This is the file that wires everything together, and it is deliberately
headless - it imports nothing from Streamlit and knows nothing about a
browser. Both `app.py` and `run_live.py` drive the same TradingEngine, which
means the alerts you get from the UI and the alerts you get from the headless
runner are produced by identical code.

The pipeline, in order, and the order matters:

    feed.get_bars()
        -> gap check                 (data gap = kill switch)
        -> build Context
        -> regime.classify()         (which strategies may speak today)
        -> strategy.on_bar()         (proposes)
        -> RiskEngine.check_halt()   (session governors)
        -> RiskEngine.approve()      (disposes: strike, size, stop, target)
        -> dispatcher.dispatch()     (fan out, de-duplicated)
        -> journal.write()

`RiskEngine.approve()` sits between the strategy and the alert with no way
around it. A strategy cannot produce an alert on its own; it can only produce
a Signal, and a Signal without an approved order is journalled and discarded.

THIS ENGINE NEVER PLACES AN ORDER. There is no broker order path in this file
or anywhere else in the package. It computes what to do and tells you.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Optional

import pandas as pd

from .config import Config, DEFAULT
from .data.base import DataFeed, FeedError
from .data.chain import ChainProvider
from .alerts.base import TradeAlert, AlertKind
from .alerts.dispatcher import AlertDispatcher
from .journal import Journal
from .regime import classify, is_allowed, RegimeReading, Regime
from .risk import RiskEngine, ApprovedOrder, RejectedOrder, HaltReason
from .strategy import Context, Signal
from .strategies.registry import build_enabled, get_info


@dataclass
class EngineState:
    """A snapshot the UI renders. Never mutated by the UI."""
    last_run: Optional[datetime] = None
    last_error: Optional[str] = None
    bars: Optional[pd.DataFrame] = None
    regime: Optional[RegimeReading] = None
    underlying_price: float = 0.0
    halt_reason: str = HaltReason.NONE.value
    alerts: list[TradeAlert] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)   # per-strategy, this bar
    feed_name: str = ""
    feed_latency_note: str = ""
    kill_switch: bool = False


class TradingEngine:
    def __init__(self, feed: DataFeed, dispatcher: AlertDispatcher,
                 cfg: Config = DEFAULT,
                 enabled_strategies: list[str] | None = None,
                 journal: Journal | None = None,
                 risk: RiskEngine | None = None):
        self.cfg = cfg
        self.feed = feed
        self.dispatcher = dispatcher
        self.journal = journal or Journal()
        self.risk = risk or RiskEngine(cfg)
        self.chain_provider = ChainProvider(cfg)
        self.strategies = build_enabled(enabled_strategies, cfg)
        self.state = EngineState(
            feed_name=feed.name,
            feed_latency_note=feed.latency_note,
        )
        self._day: Optional[date] = None
        self._force_exit_sent_for: Optional[date] = None

    # ---------------- public API ----------------

    def run_once(self, now: Optional[datetime] = None) -> EngineState:
        """
        One full evaluation pass. Safe to call on a timer.

        Every exception is caught and converted into a kill switch. An engine
        that dies on an unexpected input leaves you with no alerts and no
        notification that alerts have stopped - which reads exactly like a
        quiet day. Non-negotiable #1 exists to prevent that.
        """
        now = now or datetime.now()
        try:
            return self._run_once_inner(now)
        except FeedError as e:
            return self._trip_kill_switch(f"data feed failure: {e}")
        except Exception as e:
            return self._trip_kill_switch(f"unhandled engine error: {type(e).__name__}: {e}")

    def reset_kill_switch(self) -> None:
        """Manual re-arm. Deliberately not automatic - a kill switch that
        clears itself is a warning you will never read."""
        self.risk._kill_switch = False
        self.risk.session.halted = False
        self.risk.session.halt_reason = HaltReason.NONE
        self.state.kill_switch = False
        self.state.last_error = None
        self.journal.write("kill_switch_reset", {})

    def set_strategies(self, keys: list[str]) -> None:
        self.strategies = build_enabled(keys, self.cfg)

    # ---------------- the pass ----------------

    def _run_once_inner(self, now: datetime) -> EngineState:
        st = self.state
        st.last_run = now

        df = self.feed.get_bars(self.cfg.data.lookback_days)
        st.bars = df

        # --- data integrity before anything else ---
        gap = self.feed.detect_gap(df, self.cfg.data.interval_minutes,
                                   self.cfg.data.max_data_gap_bars)
        if gap:
            return self._trip_kill_switch(gap)

        session_df = self.feed.session_slice(df)
        if session_df.empty:
            st.last_error = "no bars for the current session"
            return st

        trading_day = session_df.index[-1].date()
        if self._day != trading_day:
            self.risk.start_day(trading_day)
            self.dispatcher.reset_suppression()
            self._day = trading_day
            self.journal.write("session_start", {"day": trading_day.isoformat(),
                                                 "feed": self.feed.name})

        st.underlying_price = float(session_df["close"].iloc[-1])
        bar_time = session_df.index[-1].to_pydatetime()

        prior = self.feed.prior_session(df, trading_day)
        ctx = Context(
            bars=session_df,
            now=bar_time.time(),
            prev_day_high=prior.high,
            prev_day_low=prior.low,
            prev_day_close=prior.close,
            is_expiry_day=self._is_expiry_day(trading_day),
        )

        # --- regime ---
        reading = classify(session_df, prior.close, self.cfg)
        st.regime = reading

        # --- 15:10 force-exit reminder (non-negotiable #2) ---
        self._maybe_force_exit(bar_time, trading_day)

        # --- session governors ---
        halt = self.risk.check_halt(
            now=bar_time.time(),
            is_expiry_day=ctx.is_expiry_day,
            in_event_blackout=False,
        )
        st.halt_reason = halt.value
        if halt is not HaltReason.NONE:
            st.evaluations = [{"strategy": "-", "outcome": "halted",
                               "detail": halt.value}]
            self.journal.halt(halt.value)
            return st

        # --- strategies ---
        st.evaluations = []
        for key, strategy in self.strategies.items():
            info = get_info(key)
            label = info.label if info else key

            if info and not is_allowed(reading, info.allowed_regimes, self.cfg):
                st.evaluations.append({
                    "strategy": label, "outcome": "gated",
                    "detail": f"regime is {reading.regime.value}, "
                              f"needs {info.regimes_text}",
                })
                continue

            try:
                signal = strategy.on_bar(ctx)
            except Exception as e:
                st.evaluations.append({"strategy": label, "outcome": "error",
                                       "detail": f"{type(e).__name__}: {e}"})
                self.journal.write("strategy_error",
                                   {"strategy": key, "error": str(e)})
                continue

            if not signal.direction:
                st.evaluations.append({"strategy": label, "outcome": "no signal",
                                       "detail": signal.reason})
                continue

            if signal.confidence < self.cfg.strategy.min_confidence:
                st.evaluations.append({
                    "strategy": label, "outcome": "low confidence",
                    "detail": f"{signal.confidence:.2f} < "
                              f"{self.cfg.strategy.min_confidence:.2f}",
                })
                continue

            self.journal.signal(key, signal.__dict__, reading.regime.value)
            self._evaluate_signal(key, label, signal, ctx, reading, bar_time, st)

        return st

    # ---------------- signal -> approved order -> alert ----------------

    def _evaluate_signal(self, key: str, label: str, signal: Signal,
                         ctx: Context, reading: RegimeReading,
                         bar_time: datetime, st: EngineState) -> None:
        spot = st.underlying_price
        chain = self.chain_provider.get_chain(spot, signal.option_type,
                                              bar_time.date())

        decision = self.risk.approve(
            chain=chain.quotes,
            option_type=signal.option_type,
            underlying_stop_points=signal.stop_points,
            free_capital=self.risk.capital,
        )

        if isinstance(decision, RejectedOrder):
            st.evaluations.append({
                "strategy": label, "outcome": "risk rejected",
                "detail": f"{decision.reason.value}: {decision.detail}",
            })
            self.journal.rejection(key, decision.reason.value, decision.detail)
            return

        alert = self._build_alert(key, label, signal, decision, reading,
                                  bar_time, chain, spot)
        results = self.dispatcher.dispatch(alert)

        if results:
            st.alerts.append(alert)
            del st.alerts[:-50]
            delivered = ", ".join(f"{k}{'' if v[0] else ' (FAILED)'}"
                                  for k, v in results.items())
            st.evaluations.append({
                "strategy": label, "outcome": "ALERT",
                "detail": f"{signal.direction} {decision.quote.strike}"
                          f"{signal.option_type} -> {delivered}",
            })
        else:
            st.evaluations.append({
                "strategy": label, "outcome": "suppressed",
                "detail": "duplicate or within cooldown",
            })

    def _build_alert(self, key: str, label: str, signal: Signal,
                     order: ApprovedOrder, reading: RegimeReading,
                     bar_time: datetime, chain, spot: float) -> TradeAlert:
        return TradeAlert(
            kind=AlertKind.ENTRY,
            timestamp=bar_time,
            strategy_key=key,
            strategy_label=label,
            direction=signal.direction,
            option_type=signal.option_type,
            strike=order.quote.strike,
            expiry=chain.expiry.isoformat(),
            entry_premium=order.entry_premium,
            target_premium=order.premium_target,
            stop_premium=order.premium_stop,
            quantity=order.quantity,
            lots=order.lots,
            rupee_risk=order.rupee_risk,
            rupee_reward=order.rupee_reward,
            delta=abs(order.quote.delta),
            underlying_stop_points=order.underlying_stop_points,
            underlying_price=spot,
            reason=signal.reason,
            confidence=signal.confidence,
            regime=reading.regime.value,
            feed_name=self.feed.name,
            feed_latency_note=self.feed.latency_note if self.feed.is_delayed else "",
            chain_source=chain.source,
            chain_note=chain.note if chain.is_synthetic else "",
        )

    # ---------------- operational alerts ----------------

    def _maybe_force_exit(self, bar_time: datetime, trading_day: date) -> None:
        if not self.cfg.alerts.force_exit_reminder:
            return
        if self._force_exit_sent_for == trading_day:
            return
        if bar_time.time() < self.cfg.session.force_exit:
            return

        self._force_exit_sent_for = trading_day
        open_count = len(self.risk.session.open_positions)
        self.dispatcher.dispatch(TradeAlert(
            kind=AlertKind.FORCE_EXIT,
            timestamp=bar_time,
            message=(f"{self.cfg.session.force_exit:%H:%M} force-exit time. "
                     f"{open_count} position(s) tracked as open. Flat before close - "
                     f"never carry an intraday option overnight."),
        ))

    def _trip_kill_switch(self, detail: str) -> EngineState:
        self.risk.trip_kill_switch()
        st = self.state
        st.kill_switch = True
        st.last_error = detail
        st.halt_reason = HaltReason.KILL_SWITCH.value
        self.journal.write("kill_switch", {"detail": detail})
        self.dispatcher.dispatch(TradeAlert(
            kind=AlertKind.KILL_SWITCH,
            timestamp=datetime.now(),
            message=f"KILL SWITCH TRIPPED — no further alerts until re-armed. {detail}",
        ))
        return st

    def _is_expiry_day(self, day: date) -> bool:
        from .data.chain import next_weekly_expiry
        return next_weekly_expiry(day) == day
