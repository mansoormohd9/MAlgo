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
        -> PositionManager.update()  (MANAGE WHAT IS OPEN, FIRST)
        -> governor verdict          (a realised exit may end the day)
        -> strategy.on_bar()         (proposes)
        -> RiskEngine.check_halt()   (session governors)
        -> RiskEngine.approve()      (disposes: strike, size, stop, target)
        -> dispatcher.dispatch()     (fan out, de-duplicated)
        -> journal.write()

`RiskEngine.approve()` sits between the strategy and the alert with no way
around it. A strategy cannot produce an alert on its own; it can only produce
a Signal, and a Signal without an approved order is journalled and discarded.

POSITION MANAGEMENT RUNS BEFORE SIGNAL GENERATION, always. An open position is
money already at risk; a new signal is only an opportunity. Evaluating entries
first would let the engine propose a fourth trade on a bar where the day had
already ended.

THIS ENGINE STILL PLACES NO ORDER BY ITSELF. Entries require an explicit
`confirm_entry()` call - that is the one-click confirmation in the UI - and
even then `BrokerConfig.dry_run` must be off. Exits, once you are in a
position, ARE managed automatically, because a stop that needs a human to
press a button is not a stop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Optional

import pandas as pd

from . import signals as sig
from .config import Config, DEFAULT
from .costs import CostModel, DEFAULT_COSTS
from .data.base import DataFeed, FeedError, NotConfigured
from .data.chain import ChainProvider
from .alerts.base import TradeAlert, AlertKind
from .alerts.dispatcher import AlertDispatcher
from .governor import GovernorAction
from .journal import Journal
from .positions import ExitAction, ExitKind, ManagedPosition, PositionManager
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
    open_positions: list[ManagedPosition] = field(default_factory=list)
    pending_orders: list[dict] = field(default_factory=list)  # awaiting confirm
    day_summary: dict = field(default_factory=dict)
    has_volume: bool = True


class TradingEngine:
    def __init__(self, feed: DataFeed, dispatcher: AlertDispatcher,
                 cfg: Config = DEFAULT,
                 enabled_strategies: list[str] | None = None,
                 journal: Journal | None = None,
                 risk: RiskEngine | None = None,
                 chain_provider: ChainProvider | None = None,
                 broker=None, costs: CostModel = DEFAULT_COSTS):
        self.cfg = cfg
        self.feed = feed
        self.costs = costs
        self.dispatcher = dispatcher
        self.journal = journal or Journal()
        self.risk = risk or RiskEngine(cfg)
        self.chain_provider = chain_provider or ChainProvider(cfg)
        self.positions = PositionManager(cfg)
        # Optional. None means "alerts only" - confirm_entry still tracks the
        # position so exits are managed, it simply does not send an order.
        self.broker = broker
        self.strategies = build_enabled(enabled_strategies, cfg)
        self.state = EngineState(
            feed_name=feed.name,
            feed_latency_note=feed.latency_note,
        )
        self._day: Optional[date] = None
        self._force_exit_sent_for: Optional[date] = None
        self._pending: dict[str, tuple[ApprovedOrder, Signal, str]] = {}
        self._last_config_error: Optional[str] = None

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
        except NotConfigured as e:
            # A missing data file or an expired Kite token is a SETUP problem,
            # not a data gap. It must be loud - but latching the kill switch
            # would mean a manual re-arm every single morning before login,
            # which trains you to click through the warning that exists to stop
            # you. So: loud, blocking, and self-healing once you fix it.
            return self._not_configured(str(e))
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
            self.positions.clear()
            self._pending.clear()
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
        st.has_volume = sig.has_traded_volume(session_df)

        # --- manage what is already open, BEFORE looking for anything new ---
        # An open position is money at risk; a new signal is only an
        # opportunity. This also means a realised exit can end the day before
        # the strategy loop ever runs, which is the correct order of events.
        self._manage_open_positions(session_df, bar_time, st)
        st.open_positions = list(self.positions.positions)
        st.day_summary = self.risk.summary()

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

        # Park the approved order so confirm_entry() places exactly what was
        # alerted, rather than re-deriving it from a chain that has since moved.
        self._pending[alert.dedupe_key] = (decision, signal, key)
        st.pending_orders = [{
            "key": k, "strike": o.quote.strike,
            "option_type": o.quote.option_type, "lots": o.lots,
            "quantity": o.quantity, "entry": o.entry_premium,
            "stop": o.premium_stop, "target": o.premium_target,
            "runner": o.runner_enabled, "sizing": o.sizing_note,
        } for k, (o, _, _) in self._pending.items()]

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
            extra={
                "sizing_note": order.sizing_note,
                "runner_enabled": order.runner_enabled,
                "day_floor": round(self.risk.governor.floor, 2),
                "day_target": round(self.risk.governor.target, 2),
                "entries_used": self.risk.governor.entries_taken,
            },
        )

    # ---------------- position management ----------------

    def _manage_open_positions(self, session_df: pd.DataFrame,
                               bar_time: datetime, st: EngineState) -> None:
        """
        Advance the exit ladder for every open position, then act on whatever
        the day rules say about the P&L that just became real.
        """
        if not self.positions.positions:
            return

        atr_series = sig.atr(session_df, self.cfg.signal.atr_period)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        if pd.isna(atr):
            atr = 0.0

        premiums = self._live_premiums(st.underlying_price, bar_time)
        actions = self.positions.update(premiums, atr=atr, now=bar_time.time())

        for action in actions:
            self._handle_exit_action(action, bar_time, st)

        # A realised exit may have ended the day while other positions are
        # still open. check_halt() only gates ENTRIES, so it cannot help here.
        verdict = self.risk.governor.evaluate()
        if verdict.action is GovernorAction.CLOSE_ALL and self.positions.positions:
            for action in self.positions.close_all(premiums, verdict.detail):
                self._handle_exit_action(action, bar_time, st)
            self.dispatcher.dispatch(TradeAlert(
                kind=AlertKind.CLOSE_ALL, timestamp=bar_time,
                message=(f"DAY OVER — {verdict.reason.value}. {verdict.detail}. "
                         f"All positions flattened."),
            ))

    def _handle_exit_action(self, action: ExitAction, bar_time: datetime,
                            st: EngineState) -> None:
        pos = action.position
        net = 0.0

        if action.lots:
            # Exit costs are charged per leg. A runner banking a partial pays
            # the flat brokerage on that sell and again on the final one.
            net = action.gross_pnl - self.costs.exit_friction(
                action.premium, action.quantity)
            if self.broker is not None:
                self.broker.exit_position(pos, action)
            # Booking the P&L re-runs the day rules; the caller acts on the
            # verdict once every action for this bar has been processed.
            self.risk.register_exit(None, net)
        elif self.broker is not None and action.new_stop_premium is not None:
            self.broker.modify_stop(pos, action.new_stop_premium)

        kind = {
            ExitKind.STOP_TO_BREAKEVEN: AlertKind.MANAGE,
            ExitKind.TRAIL_UPDATE: AlertKind.MANAGE,
            ExitKind.PARTIAL_EXIT: AlertKind.PARTIAL_EXIT,
            ExitKind.CLOSE_ALL: AlertKind.CLOSE_ALL,
            ExitKind.FORCE_EXIT: AlertKind.FORCE_EXIT,
        }.get(action.kind, AlertKind.EXIT)

        parts = [f"{pos.quote.strike}{pos.quote.option_type} "
                 f"({pos.strategy_key}): {action.detail}."]
        if action.lots:
            parts.append(f"{action.quantity} qty at {action.premium:.2f}, "
                         f"net Rs {net:,.0f}.")
        if action.new_stop_premium is not None:
            parts.append(f"New stop {action.new_stop_premium:.2f}.")
        message = " ".join(parts)

        alert = TradeAlert(
            kind=kind, timestamp=bar_time,
            strategy_key=pos.strategy_key,
            option_type=pos.quote.option_type,
            strike=pos.quote.strike,
            entry_premium=pos.entry_premium,
            stop_premium=(action.new_stop_premium
                          if action.new_stop_premium is not None
                          else pos.stop_premium),
            quantity=action.quantity,
            lots=action.lots,
            underlying_price=st.underlying_price,
            message=message,
            feed_name=self.feed.name,
            feed_latency_note=(self.feed.latency_note
                               if self.feed.is_delayed else ""),
        )
        self.dispatcher.dispatch(alert)
        st.alerts.append(alert)
        del st.alerts[:-50]
        self.journal.write("position_action", {
            "kind": action.kind.value,
            "strike": pos.quote.strike,
            "option_type": pos.quote.option_type,
            "lots": action.lots,
            "premium": action.premium,
            "gross": round(action.gross_pnl, 2),
            "net": round(net, 2) if action.lots else 0.0,
            "detail": action.detail,
        })

    def _live_premiums(self, spot: float, bar_time: datetime) -> dict[str, float]:
        """
        Current premium for every open position.

        Prefers a real broker quote. Falls back to re-pricing the same strike
        off the chain provider, which on a synthetic chain means a MODELLED
        premium - fine for paper, and labelled as such in the alert, but it is
        not a price anyone would fill you at.
        """
        out: dict[str, float] = {}
        for pos in self.positions.positions:
            key = pos.tradingsymbol or f"{pos.quote.strike}{pos.quote.option_type}"
            try:
                chain = self.chain_provider.get_chain(
                    spot, pos.quote.option_type, bar_time.date())
                match = next((q for q in chain.quotes
                              if q.strike == pos.quote.strike), None)
                if match is not None:
                    out[key] = match.premium
            except Exception:
                continue          # no quote: PositionManager skips this position
        return out

    # ---------------- confirmed entry ----------------

    def confirm_entry(self, alert: TradeAlert) -> Optional[ManagedPosition]:
        """
        The one-click confirmation. Places the order (unless dry_run) and
        starts managing the position.

        Returns None when the alert is stale, already confirmed, or the day
        rules have since closed the session - all of which are ordinary, and
        all of which would otherwise put on a trade you no longer want.
        """
        pending = self._pending.pop(alert.dedupe_key, None)
        if pending is None:
            return None
        order, signal, strategy_key = pending

        if self.risk._kill_switch:
            self.journal.write("entry_refused",
                               {"reason": "kill switch", "key": alert.dedupe_key})
            return None
        verdict = self.risk.governor.evaluate()
        if verdict.day_over or verdict.action is GovernorAction.BLOCK_NEW_ENTRIES:
            self.journal.write("entry_refused", {"reason": verdict.reason.value,
                                                 "key": alert.dedupe_key})
            return None

        tradingsymbol = ""
        if self.broker is not None:
            tradingsymbol = self.broker.place_entry(order, alert) or ""

        self.risk.register_entry(order, signal.direction)
        pos = self.positions.open(
            order, direction=signal.direction, strategy_key=strategy_key,
            entry_underlying=alert.underlying_price,
            opened_at=alert.timestamp, tradingsymbol=tradingsymbol,
        )
        self.journal.write("entry_confirmed", {
            "strike": order.quote.strike,
            "option_type": order.quote.option_type,
            "lots": order.lots, "quantity": order.quantity,
            "entry": order.entry_premium,
            "stop": order.premium_stop, "target": order.premium_target,
            "runner": order.runner_enabled,
            "dry_run": self.broker is None or self.cfg.broker.dry_run,
        })
        return pos

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

    def _not_configured(self, detail: str) -> EngineState:
        """
        Credentials or files are missing. Blocks, does not latch.

        Announced once per distinct message so an unattended runner does not
        send the same "log in to Kite" notification every poll for an hour.
        """
        st = self.state
        st.last_error = detail
        st.evaluations = [{"strategy": "-", "outcome": "not configured",
                           "detail": detail.splitlines()[0]}]
        if self._last_config_error != detail:
            self._last_config_error = detail
            self.journal.write("not_configured", {"detail": detail})
            self.dispatcher.dispatch(TradeAlert(
                kind=AlertKind.HALT, timestamp=datetime.now(),
                message=f"NOT CONFIGURED — no evaluations until this is fixed. "
                        f"{detail.splitlines()[0]}",
            ))
        return st

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
