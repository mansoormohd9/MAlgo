"""
Walk-forward backtester.

======================================================================
READ THIS BEFORE READING ANY NUMBER THIS MODULE PRODUCES
======================================================================

There are no historical option chains in this project. That is not an
oversight to be patched later - it is the central limitation, and pretending
otherwise is how backtests come to justify losing systems.

So this backtester has two modes, and they are honest about different things:

  UNDERLYING (default, trustworthy)
      Measures the only question the signal layer can actually answer: when
      this setup fires, does the UNDERLYING reach +2R before it reaches -1R?
      R is defined by the strategy's own ATR stop. Friction from costs.py is
      converted into R-units and subtracted.

      What it validates: whether the setup has directional edge at 2:1.
      What it cannot tell you: anything about theta, vega, IV crush, or
      whether a strike existed at a tradeable spread.

  SYNTHETIC_PREMIUM (optional, flagged everywhere it appears)
      Prices entry and exit with Black-Scholes off spot at a single flat
      assumed IV. Per the README's own Phase 3 note, this is OPTIMISTIC BY
      15-25%, because synthesised premiums understate slippage, ignore skew,
      and never fail to fill.

      Its real use is not the P&L number. It is showing you how much of a
      winning underlying signal survives being expressed through an option -
      which is often far less than people expect.

Neither mode is evidence to trade live capital. Roadmap Phase 4 stands: forty
paper sessions against real quotes, comparing realised slippage to the model.

The walk-forward split is train 6 / test 2 months, rolling. Only test-window
results are reported. There is nothing to "fit" in these strategies yet - the
parameters are hand-set - so the split's job today is to expose regime
dependence: a strategy whose edge lives entirely in one fold did not have edge,
it had a good quarter.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from . import signals as sig
from .config import Config, DEFAULT
from .costs import CostModel, DEFAULT_COSTS
from .data.csv_feed import BarReplayer
from .data.base import DataFeed
from .positions import ExitKind, ExitLadder
from .pricing import bs_price, bs_delta, time_to_expiry_years
from .data.chain import next_weekly_expiry
from .regime import classify, is_allowed
from .risk import RiskEngine
from .strategy import Context
from .strategies.registry import build_enabled, get_info


class Mode(str, Enum):
    UNDERLYING = "underlying"
    SYNTHETIC_PREMIUM = "synthetic_premium"


class Outcome(str, Enum):
    TARGET = "target"
    STOP = "stop"
    TIMEOUT = "timeout"          # neither hit within max_bars_in_trade
    SESSION_END = "session_end"  # forced flat at 15:10


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    strategy: str
    direction: str
    option_type: str
    regime: str
    entry_underlying: float
    exit_underlying: float
    stop_points: float
    outcome: Outcome
    bars_held: int
    r_multiple: float                 # net of friction, in R units
    reason: str = ""
    lots: int = 1
    exit_legs: int = 1                # >1 means the runner banked a partial
    partial_banked: bool = False
    # synthetic-premium mode only
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    quantity: int = 0
    net_pnl: float = 0.0


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_r: float = 0.0
    longest_losing_streak: int = 0
    avg_bars_held: float = 0.0
    net_pnl: float = 0.0              # synthetic mode only
    breakeven_win_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "Trades": self.trades,
            "Wins": self.wins,
            "Win rate": f"{self.win_rate:.1%}",
            "Breakeven win rate": f"{self.breakeven_win_rate:.1%}",
            "Expectancy (R)": f"{self.expectancy_r:+.3f}",
            "Total (R)": f"{self.total_r:+.2f}",
            "Profit factor": f"{self.profit_factor:.2f}",
            "Max drawdown (R)": f"{self.max_drawdown_r:.2f}",
            "Longest losing streak": self.longest_losing_streak,
            "Avg bars held": f"{self.avg_bars_held:.1f}",
        }


@dataclass
class Fold:
    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    metrics: Metrics
    trades: list[BacktestTrade] = field(default_factory=list)


@dataclass
class DayStats:
    """
    How the SESSION RULES behaved, as opposed to how the signals behaved.

    Trade-level R tells you whether the setups have edge. This tells you
    whether your day rules help or hurt: how often the +10% target is actually
    reached, how often the ratcheting floor ends a day that had been green,
    and how much of the three-entry budget gets used.
    """
    days: int = 0
    days_hit_target: int = 0
    days_hit_floor: int = 0
    days_ratcheted: int = 0
    days_max_entries: int = 0
    avg_entries: float = 0.0
    avg_realised: float = 0.0
    green_days: int = 0
    trades_with_partial: int = 0

    def as_dict(self) -> dict:
        return {
            "Trading days": self.days,
            "Green days": f"{self.green_days} "
                          f"({self.green_days / self.days:.0%})" if self.days else "0",
            "Hit +10% target": self.days_hit_target,
            "Hit give-back floor": self.days_hit_floor,
            "Floor ratcheted": self.days_ratcheted,
            "Used all 3 entries": self.days_max_entries,
            "Avg entries/day": f"{self.avg_entries:.2f}",
            "Avg realised/day": f"Rs {self.avg_realised:,.0f}",
            "Trades that banked a partial": self.trades_with_partial,
        }


@dataclass
class BacktestResult:
    mode: Mode
    folds: list[Fold]
    trades: list[BacktestTrade]
    metrics: Metrics
    by_strategy: dict[str, Metrics]
    warnings: list[str]
    day_stats: DayStats = field(default_factory=DayStats)

    @property
    def equity_curve_r(self) -> pd.Series:
        if not self.trades:
            return pd.Series(dtype=float)
        s = pd.Series([t.r_multiple for t in self.trades],
                      index=[t.exit_time for t in self.trades])
        return s.cumsum()


# ---------------------------------------------------------------- metrics

def compute_metrics(trades: list[BacktestTrade], reward_risk: float = 2.0) -> Metrics:
    m = Metrics(breakeven_win_rate=1.0 / (1.0 + reward_risk))
    if not trades:
        return m

    rs = np.array([t.r_multiple for t in trades], dtype=float)
    m.trades = len(trades)
    m.wins = int((rs > 0).sum())
    m.win_rate = m.wins / m.trades
    m.expectancy_r = float(rs.mean())
    m.total_r = float(rs.sum())
    m.avg_bars_held = float(np.mean([t.bars_held for t in trades]))
    m.net_pnl = float(sum(t.net_pnl for t in trades))

    gains = rs[rs > 0].sum()
    losses = abs(rs[rs < 0].sum())
    m.profit_factor = float(gains / losses) if losses > 0 else float("inf")

    equity = rs.cumsum()
    peak = np.maximum.accumulate(equity)
    m.max_drawdown_r = float((peak - equity).max()) if len(equity) else 0.0

    streak = worst = 0
    for r in rs:
        streak = streak + 1 if r < 0 else 0
        worst = max(worst, streak)
    m.longest_losing_streak = worst
    return m


# ---------------------------------------------------------------- engine

class Backtester:
    def __init__(self, cfg: Config = DEFAULT, costs: CostModel = DEFAULT_COSTS):
        self.cfg = cfg
        self.costs = costs
        self.warnings: list[str] = []
        self.ladder = ExitLadder(cfg)
        self._vix: Optional[pd.Series] = None
        self._days: list[dict] = []
        self._day_outcomes: list[str] = []
        self.load_vix()

    def run(self, bars: pd.DataFrame, strategy_keys: list[str] | None = None,
            mode: Mode = Mode.UNDERLYING) -> BacktestResult:
        bars = bars.sort_index()
        self.warnings = []
        self._days = []
        self._day_outcomes = []

        if not sig.has_traded_volume(bars):
            self.warnings.append(
                "This frame has NO TRADED VOLUME - it is an index series. The "
                "participation gates fall back to range expansion and VWAP "
                "degrades to a session TWAP (see signals.has_traded_volume). "
                "Both test the same claim as the volume versions but they are "
                "not the same measurement, so a strategy tuned on volume data "
                "is not being reproduced here."
            )

        if mode is Mode.SYNTHETIC_PREMIUM:
            b = self.cfg.backtest
            iv_note = (f"per-day IV from India VIX ({len(self._vix):,} days loaded)"
                       if self._vix is not None and b.use_vix_iv
                       else f"a single flat IV of {b.assumed_iv:.0%}")
            self.warnings.append(
                f"SYNTHETIC_PREMIUM mode: premiums are Black-Scholes at {iv_note}, "
                "with NO SKEW, no spread and guaranteed fills. Per the README's "
                "Phase 3 note these results are OPTIMISTIC BY 15-25%. Do not treat "
                "this P&L as achievable."
            )
            if self._vix is None and b.use_vix_iv:
                self.warnings.append(
                    f"India VIX not found at {self.cfg.data.vix_csv_path} - fell "
                    f"back to a flat {b.assumed_iv:.0%}. Run scripts/fetch_vix.py. "
                    f"A flat IV overprices calm periods and underprices violent "
                    f"ones, in opposite directions."
                )

        folds = self._make_folds(bars)
        if not folds:
            self.warnings.append(
                f"Not enough history for a "
                f"{self.cfg.backtest.train_months}/{self.cfg.backtest.test_months}-month "
                f"walk-forward split — ran a single in-sample pass instead. "
                f"An in-sample result is a description of the past, not a forecast."
            )

        all_trades: list[BacktestTrade] = []
        fold_objs: list[Fold] = []

        if folds:
            for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
                window = bars[(bars.index.date >= te_s) & (bars.index.date <= te_e)]
                trades = self._run_window(window, strategy_keys, mode)
                fold_objs.append(Fold(
                    index=i, train_start=tr_s, train_end=tr_e,
                    test_start=te_s, test_end=te_e,
                    metrics=compute_metrics(trades, self.cfg.capital.reward_risk_ratio),
                    trades=trades,
                ))
                all_trades.extend(trades)
        else:
            all_trades = self._run_window(bars, strategy_keys, mode)

        rr = self.cfg.capital.reward_risk_ratio
        by_strategy: dict[str, Metrics] = {}
        for key in {t.strategy for t in all_trades}:
            by_strategy[key] = compute_metrics(
                [t for t in all_trades if t.strategy == key], rr)

        return BacktestResult(
            mode=mode,
            folds=fold_objs,
            trades=all_trades,
            metrics=compute_metrics(all_trades, rr),
            by_strategy=by_strategy,
            warnings=self.warnings,
            day_stats=self._compute_day_stats(all_trades),
        )

    def _compute_day_stats(self, trades: list[BacktestTrade]) -> DayStats:
        d = DayStats(days=len(self._days))
        d.trades_with_partial = sum(1 for t in trades if t.partial_banked)
        if not self._days:
            return d
        d.days_hit_target = sum(1 for x in self._days
                                if x["outcome"] == "session_target_hit")
        d.days_hit_floor = sum(1 for x in self._days
                               if x["outcome"] == "give_back_floor_hit")
        d.days_ratcheted = sum(1 for x in self._days if x["ratcheted"])
        d.days_max_entries = sum(
            1 for x in self._days
            if x["entries"] >= self.cfg.capital.max_entries_per_session)
        d.green_days = sum(1 for x in self._days if x["realised"] > 0)
        d.avg_entries = float(np.mean([x["entries"] for x in self._days]))
        d.avg_realised = float(np.mean([x["realised"] for x in self._days]))
        return d

    # ---------------- walk-forward split ----------------

    def _make_folds(self, bars: pd.DataFrame):
        b = self.cfg.backtest
        days = sorted({d for d in bars.index.date})
        if not days:
            return []
        span_days = (days[-1] - days[0]).days
        need = (b.train_months + b.test_months) * 30
        if span_days < need:
            return []

        folds = []
        train_len = pd.Timedelta(days=b.train_months * 30)
        test_len = pd.Timedelta(days=b.test_months * 30)
        cursor = pd.Timestamp(days[0])
        end = pd.Timestamp(days[-1])

        while cursor + train_len + test_len <= end:
            tr_s = cursor.date()
            tr_e = (cursor + train_len).date()
            te_s = tr_e
            te_e = (cursor + train_len + test_len).date()
            folds.append((tr_s, tr_e, te_s, te_e))
            cursor = cursor + test_len
        return folds

    # ---------------- one window ----------------

    def _run_window(self, bars: pd.DataFrame, strategy_keys: list[str] | None,
                    mode: Mode) -> list[BacktestTrade]:
        trades: list[BacktestTrade] = []
        for day in sorted({d for d in bars.index.date}):
            session = bars[bars.index.date == day]
            prior = DataFeed.prior_session(bars, day)
            trades.extend(self._run_session(session, prior, day,
                                            strategy_keys, mode))
        return trades

    def _run_session(self, session: pd.DataFrame, prior, day: date,
                     strategy_keys: list[str] | None, mode: Mode
                     ) -> list[BacktestTrade]:
        """
        One simulated trading day, INCLUDING the money-management rules.

        The original version built a RiskEngine and never called register_exit,
        so realised P&L stayed at zero, the day target and the give-back floor
        could never fire, and the only governor with any effect was the entry
        count. It measured the signals. It did not measure the system.

        Now every simulated exit is booked, so a day can end because it reached
        +10%, or because it gave back to the ratcheting floor - and the metrics
        below report how often each actually happens.
        """
        s = self.cfg.session
        b = self.cfg.backtest
        strategies = build_enabled(strategy_keys, self.cfg)
        risk = RiskEngine(self.cfg)
        risk.start_day(day)
        gov = risk.governor

        trades: list[BacktestTrade] = []
        cooldown_until = -1
        atr_series = sig.atr(session, self.cfg.signal.atr_period)

        # BarReplayer is what keeps this honest - see its docstring. The
        # window it yields ends at the decision bar, so find_pivots() cannot
        # see the future bars it would need to confirm a pivot early.
        replayer = BarReplayer(session, warmup=max(self.cfg.signal.atr_period, 30))

        for i in range(replayer.warmup, len(session)):
            if gov.entries_taken >= self.cfg.capital.max_entries_per_session:
                break
            if b.apply_session_governors and gov.evaluate().day_over:
                self._day_outcomes.append(gov.evaluate().reason.value)
                break
            if i <= cooldown_until:
                continue

            window = replayer.window_at(i)
            bar_time = window.index[-1]
            now = bar_time.time()
            if not (s.entry_start <= now <= s.entry_cutoff):
                continue

            reading = classify(window, prior.close, self.cfg)
            ctx = Context(
                bars=window, now=now,
                prev_day_high=prior.high, prev_day_low=prior.low,
                prev_day_close=prior.close,
                is_expiry_day=next_weekly_expiry(day) == day,
            )

            for key, strat in strategies.items():
                info = get_info(key)
                if info and not is_allowed(reading, info.allowed_regimes, self.cfg):
                    continue
                try:
                    sig_out = strat.on_bar(ctx)
                except Exception:
                    continue
                if not sig_out.direction:
                    continue
                if sig_out.confidence < self.cfg.strategy.min_confidence:
                    continue

                # Shared with approve() so the backtest cannot size a trade the
                # live engine would refuse.
                lots = risk.lots_for(sig_out.stop_points)
                trade = self._simulate(session, i, key, sig_out,
                                       reading.regime.value, day, mode,
                                       atr_series=atr_series, lots=lots)
                if trade:
                    trades.append(trade)
                    gov.register_entry()
                    if b.apply_session_governors:
                        # The rupee P&L of one R is the risk budget by
                        # definition, so an R-multiple converts directly.
                        gov.register_exit(
                            trade.r_multiple *
                            self.cfg.capital.risk_per_trade_rupees)
                    cooldown_until = i + trade.bars_held
                break                       # one entry per bar, first match wins

        self._record_day(gov, trades)
        return trades

    def _record_day(self, gov, trades: list[BacktestTrade]) -> None:
        if not trades:
            return
        verdict = gov.evaluate()
        self._days.append({
            "realised": gov.realised_pnl,
            "peak": gov.peak_realised_pnl,
            "entries": gov.entries_taken,
            "ratcheted": gov.floor > gov.base_floor,
            "outcome": verdict.reason.value,
        })

    # ---------------- trade simulation ----------------

    def _simulate(self, session: pd.DataFrame, entry_idx: int, key: str,
                  sig_out, regime: str, day: date, mode: Mode,
                  atr_series: Optional[pd.Series] = None,
                  lots: int = 1) -> Optional[BacktestTrade]:
        """
        Walk forward bar by bar from the entry, driving the SAME `ExitLadder`
        the live engine uses.

        That shared state machine is the point. The ladder takes R and returns
        decisions; here R comes from the underlying, live it comes from the
        option premium, and `RiskEngine.approve()` sizes the position so the
        two agree. Reimplementing the breakeven shift, the partial and the
        trail here would mean backtesting a system you do not run.

        Intrabar ambiguity is resolved PESSIMISTICALLY: the stop is tested
        against the bar's adverse extreme, at the level it held when the bar
        opened, BEFORE any promotion from the favourable extreme. A bar whose
        range contains both the stop and the next rung is scored as a stop.
        Without that a backtest quietly awards itself the best of both on every
        volatile bar, which is the most common way an equity curve is inflated.
        """
        b = self.cfg.backtest
        entry_price = float(session.iloc[entry_idx]["close"])
        entry_time = session.index[entry_idx]
        stop_pts = float(sig_out.stop_points)
        if stop_pts <= 0:
            return None

        long_side = sig_out.direction == "long"
        st = self.ladder.new_state(lots)

        # Realised R accumulates per leg. Exiting k of n lots at `exit_r`
        # realises exit_r * k / n, because 1R is defined for the FULL position.
        realised_r = 0.0
        legs: list[tuple[float, int]] = []      # (exit_r, lots)
        outcome = Outcome.TIMEOUT
        exit_price = entry_price
        exit_time = entry_time
        bars_held = 0

        last_idx = min(entry_idx + b.max_bars_in_trade, len(session) - 1)
        for j in range(entry_idx + 1, last_idx + 1):
            bar = session.iloc[j]
            hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
            bars_held = j - entry_idx
            exit_time = session.index[j]
            exit_price = close

            if long_side:
                best_r = (hi - entry_price) / stop_pts
                worst_r = (lo - entry_price) / stop_pts
            else:
                best_r = (entry_price - lo) / stop_pts
                worst_r = (entry_price - hi) / stop_pts
            mark_r = ((close - entry_price) if long_side
                      else (entry_price - close)) / stop_pts

            trail_r = 0.0
            if atr_series is not None and b.apply_trade_management:
                a = float(atr_series.iloc[j]) if not pd.isna(atr_series.iloc[j]) else 0.0
                trail_r = a * self.cfg.trade.trail_atr_multiple / stop_pts

            forced = session.index[j].time() >= self.cfg.session.force_exit
            if forced:
                d = self.ladder.force_exit(st, mark_r)
                if d.exit_lots:
                    legs.append((d.exit_r, d.exit_lots))
                    realised_r += d.exit_r * d.exit_lots / lots
                outcome = Outcome.SESSION_END
                exit_price = close
                break

            for d in self.ladder.advance(st, mark_r=mark_r, best_r=best_r,
                                         worst_r=worst_r,
                                         trail_distance_r=trail_r):
                if not d.exit_lots:
                    continue
                legs.append((d.exit_r, d.exit_lots))
                realised_r += d.exit_r * d.exit_lots / lots
                if d.kind is ExitKind.STOPPED_OUT:
                    outcome = Outcome.STOP
                elif d.kind in (ExitKind.TARGET_EXIT, ExitKind.PARTIAL_EXIT):
                    outcome = Outcome.TARGET
                exit_price = (entry_price + d.exit_r * stop_pts if long_side
                              else entry_price - d.exit_r * stop_pts)

            if st.closed:
                break

        if bars_held == 0:
            return None

        if not st.closed:
            # Ran out of bars (or hit the bar cap) with lots still open.
            mark_r = ((exit_price - entry_price) if long_side
                      else (entry_price - exit_price)) / stop_pts
            d = self.ladder.force_exit(st, mark_r, "bar cap / end of data")
            if d.exit_lots:
                legs.append((d.exit_r, d.exit_lots))
                realised_r += d.exit_r * d.exit_lots / lots
            if outcome is Outcome.TIMEOUT and len(legs) > 1:
                outcome = Outcome.TARGET      # partial banked, runner timed out

        if mode is Mode.UNDERLYING:
            r_multiple, entry_prem, exit_prem, qty, pnl = self._r_underlying(
                realised_r, legs, lots)
        else:
            r_multiple, entry_prem, exit_prem, qty, pnl = self._r_synthetic(
                entry_price, legs, stop_pts, sig_out.option_type, day,
                long_side, lots)

        return BacktestTrade(
            entry_time=entry_time, exit_time=exit_time, strategy=key,
            direction=sig_out.direction, option_type=sig_out.option_type,
            regime=regime, entry_underlying=entry_price, exit_underlying=exit_price,
            stop_points=stop_pts, outcome=outcome, bars_held=bars_held,
            r_multiple=r_multiple, reason=sig_out.reason,
            entry_premium=entry_prem, exit_premium=exit_prem,
            quantity=qty, net_pnl=pnl, lots=lots, exit_legs=len(legs),
            partial_banked=len(legs) > 1,
        )

    def _r_underlying(self, realised_r: float, legs: list[tuple[float, int]],
                      lots: int):
        """
        R in underlying points, with friction converted into R.

        Friction is a rupee amount; R is a rupee amount too (risk_per_trade),
        so the conversion is direct and does not need an option price. That is
        what makes this mode trustworthy - no premium is assumed anywhere.

        Friction is charged PER LEG. A runner that banks half and trails the
        rest is one buy and two sells, so it pays the flat brokerage three
        times. Charging a single round trip would quietly subsidise the runner
        and make scaling out look better than it is.
        """
        risk_rupees = self.cfg.capital.risk_per_trade_rupees
        lot = self.cfg.instrument.lot_size
        qty = lot * lots

        # Friction is evaluated at a representative premium in the sweet spot
        # the README identifies (Rs 90-150); it is nearly flat across that range.
        ref = 120.0
        friction = self.costs.entry_friction(ref, qty)
        for _, leg_lots in legs:
            friction += self.costs.exit_friction(ref, lot * leg_lots)

        return realised_r - friction / risk_rupees, 0.0, 0.0, qty, 0.0

    def _r_synthetic(self, entry: float, legs: list[tuple[float, int]],
                     stop_pts: float, option_type: str, day: date,
                     long_side: bool, lots: int):
        """Express the same underlying move through a Black-Scholes option."""
        b = self.cfg.backtest
        expiry = next_weekly_expiry(day)
        t_entry = max(time_to_expiry_years(day, expiry), 0.5 / 365.0)
        # Time decays while the trade is open; a single session is ~1/365 year.
        t_exit = max(t_entry - (1.0 / 365.0) * 0.4, 0.25 / 365.0)
        iv = self.iv_for(day)

        risk = RiskEngine(self.cfg)
        ceiling = min(risk.required_max_delta(stop_pts, lots),
                      self.cfg.signal.max_delta)
        strike = self._strike_for_delta(entry, ceiling, option_type, t_entry, iv)
        if strike is None:
            return 0.0, 0.0, 0.0, 0, 0.0

        lot = self.cfg.instrument.lot_size
        qty = lot * lots
        entry_prem = bs_price(entry, strike, t_entry, iv,
                              b.risk_free_rate, option_type)

        net = -self.costs.entry_friction(entry_prem, qty)
        last_exit_prem = entry_prem
        for exit_r, leg_lots in legs:
            underlying_exit = (entry + exit_r * stop_pts if long_side
                               else entry - exit_r * stop_pts)
            exit_prem = bs_price(underlying_exit, strike, t_exit, iv,
                                 b.risk_free_rate, option_type)
            leg_qty = lot * leg_lots
            net += (exit_prem - entry_prem) * leg_qty
            net -= self.costs.exit_friction(max(exit_prem, 0.05), leg_qty)
            last_exit_prem = exit_prem

        return net / self.cfg.capital.risk_per_trade_rupees, \
            entry_prem, last_exit_prem, qty, net

    def _strike_for_delta(self, spot: float, target_delta: float,
                          option_type: str, t_years: float,
                          iv: float) -> Optional[int]:
        """Highest-delta strike at or under the risk-engine's ceiling."""
        b = self.cfg.backtest
        step = self.cfg.instrument.strike_step
        atm = int(round(spot / step) * step)
        best, best_delta = None, -1.0
        for i in range(-15, 16):
            strike = atm + i * step
            if strike <= 0:
                continue
            d = bs_delta(spot, strike, t_years, iv, b.risk_free_rate, option_type)
            if self.cfg.signal.min_delta <= d <= target_delta and d > best_delta:
                best, best_delta = strike, d
        return best

    # ---------------- implied volatility ----------------

    def iv_for(self, day: date) -> float:
        """
        That day's implied volatility, from India VIX when it is loaded.

        A flat 14% across four years spans everything from a dead-quiet 10 to
        a crisis 30. That does not merely add noise - it systematically
        overprices calm periods and underprices violent ones, in opposite
        directions, which is exactly the bias an options backtest cannot
        absorb. Falls back to `assumed_iv` for days with no reading.
        """
        b = self.cfg.backtest
        if not b.use_vix_iv or self._vix is None:
            return b.assumed_iv
        try:
            ts = pd.Timestamp(day)
            window = self._vix.loc[:ts]
            if window.empty:
                return b.assumed_iv
            return float(window.iloc[-1]) / 100.0 * b.vix_to_iv_scale
        except Exception:
            return b.assumed_iv

    def load_vix(self, path: str | None = None) -> bool:
        """Load the India VIX daily series written by scripts/fetch_vix.py."""
        from pathlib import Path
        p = Path(path or self.cfg.data.vix_csv_path)
        if not p.exists():
            self._vix = None
            return False
        try:
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            self._vix = df["vix"].sort_index()
            return True
        except Exception:
            self._vix = None
            return False
