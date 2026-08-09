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

from .config import Config, DEFAULT
from .costs import CostModel, DEFAULT_COSTS
from .data.csv_feed import BarReplayer
from .data.base import DataFeed
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
class BacktestResult:
    mode: Mode
    folds: list[Fold]
    trades: list[BacktestTrade]
    metrics: Metrics
    by_strategy: dict[str, Metrics]
    warnings: list[str]

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

    def run(self, bars: pd.DataFrame, strategy_keys: list[str] | None = None,
            mode: Mode = Mode.UNDERLYING) -> BacktestResult:
        bars = bars.sort_index()
        self.warnings = []

        if mode is Mode.SYNTHETIC_PREMIUM:
            self.warnings.append(
                "SYNTHETIC_PREMIUM mode: premiums are Black-Scholes at a single flat "
                f"IV of {self.cfg.backtest.assumed_iv:.0%}, with no skew, no spread, "
                "and guaranteed fills. Per the README's Phase 3 note these results are "
                "OPTIMISTIC BY 15-25%. Do not treat this P&L as achievable."
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
        )

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
        s = self.cfg.session
        strategies = build_enabled(strategy_keys, self.cfg)
        risk = RiskEngine(self.cfg)
        risk.start_day(day)

        trades: list[BacktestTrade] = []
        entries = 0
        cooldown_until = -1

        # BarReplayer is what keeps this honest - see its docstring. The
        # window it yields ends at the decision bar, so find_pivots() cannot
        # see the future bars it would need to confirm a pivot early.
        replayer = BarReplayer(session, warmup=max(self.cfg.signal.atr_period, 30))

        for i in range(replayer.warmup, len(session)):
            if entries >= self.cfg.capital.max_entries_per_session:
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

                trade = self._simulate(session, i, key, sig_out,
                                       reading.regime.value, day, mode)
                if trade:
                    trades.append(trade)
                    entries += 1
                    cooldown_until = i + trade.bars_held
                break                       # one entry per bar, first match wins

        return trades

    # ---------------- trade simulation ----------------

    def _simulate(self, session: pd.DataFrame, entry_idx: int, key: str,
                  sig_out, regime: str, day: date, mode: Mode
                  ) -> Optional[BacktestTrade]:
        """
        Walk forward bar by bar from the entry until the stop, the target, the
        bar cap, or the force-exit time.

        Intrabar ambiguity is resolved PESSIMISTICALLY: if a single bar's range
        contains both the stop and the target, the stop is assumed to have hit
        first. Without that assumption a backtest quietly awards itself the
        best of both on every volatile bar, which is the single most common way
        an equity curve is inflated.
        """
        b = self.cfg.backtest
        entry_bar = session.iloc[entry_idx]
        entry_price = float(entry_bar["close"])
        entry_time = session.index[entry_idx]
        stop_pts = float(sig_out.stop_points)
        if stop_pts <= 0:
            return None

        long_side = sig_out.direction == "long"
        rr = self.cfg.capital.reward_risk_ratio
        target_pts = stop_pts * rr

        stop_price = entry_price - stop_pts if long_side else entry_price + stop_pts
        target_price = entry_price + target_pts if long_side else entry_price - target_pts

        outcome = Outcome.TIMEOUT
        exit_price = entry_price
        exit_time = entry_time
        bars_held = 0

        last_idx = min(entry_idx + b.max_bars_in_trade, len(session) - 1)
        for j in range(entry_idx + 1, last_idx + 1):
            bar = session.iloc[j]
            hi, lo = float(bar["high"]), float(bar["low"])
            bars_held = j - entry_idx
            exit_time = session.index[j]

            hit_stop = lo <= stop_price if long_side else hi >= stop_price
            hit_target = hi >= target_price if long_side else lo <= target_price

            if hit_stop:                       # pessimistic: stop wins ties
                outcome, exit_price = Outcome.STOP, stop_price
                break
            if hit_target:
                outcome, exit_price = Outcome.TARGET, target_price
                break
            if session.index[j].time() >= self.cfg.session.force_exit:
                outcome, exit_price = Outcome.SESSION_END, float(bar["close"])
                break
            exit_price = float(bar["close"])

        if bars_held == 0:
            return None

        if mode is Mode.UNDERLYING:
            r_multiple, entry_prem, exit_prem, qty, pnl = self._r_underlying(
                entry_price, exit_price, stop_pts, long_side)
        else:
            r_multiple, entry_prem, exit_prem, qty, pnl = self._r_synthetic(
                entry_price, exit_price, stop_pts, sig_out.option_type, day)

        return BacktestTrade(
            entry_time=entry_time, exit_time=exit_time, strategy=key,
            direction=sig_out.direction, option_type=sig_out.option_type,
            regime=regime, entry_underlying=entry_price, exit_underlying=exit_price,
            stop_points=stop_pts, outcome=outcome, bars_held=bars_held,
            r_multiple=r_multiple, reason=sig_out.reason,
            entry_premium=entry_prem, exit_premium=exit_prem,
            quantity=qty, net_pnl=pnl,
        )

    def _r_underlying(self, entry: float, exit_: float, stop_pts: float,
                      long_side: bool):
        """
        R in underlying points, with friction converted into R.

        Friction is a rupee amount; R is a rupee amount too (risk_per_trade),
        so the conversion is direct and does not need an option price. That is
        what makes this mode trustworthy - no premium is assumed anywhere.
        """
        move = (exit_ - entry) if long_side else (entry - exit_)
        gross_r = move / stop_pts

        risk_rupees = self.cfg.capital.risk_per_trade_rupees
        qty = self.cfg.instrument.lot_size
        # Friction is evaluated at a representative premium in the sweet spot
        # the README identifies (Rs 90-150); it is nearly flat across that range.
        friction = self.costs.total_friction(120.0, 120.0, qty)
        return gross_r - friction / risk_rupees, 0.0, 0.0, qty, 0.0

    def _r_synthetic(self, entry: float, exit_: float, stop_pts: float,
                     option_type: str, day: date):
        """Express the same underlying move through a Black-Scholes option."""
        b = self.cfg.backtest
        expiry = next_weekly_expiry(day)
        t_entry = max(time_to_expiry_years(day, expiry), 0.5 / 365.0)
        # Time decays while the trade is open; a single session is ~1/365 year.
        t_exit = max(t_entry - (1.0 / 365.0) * 0.4, 0.25 / 365.0)

        risk = RiskEngine(self.cfg)
        ceiling = min(risk.required_max_delta(stop_pts), self.cfg.signal.max_delta)
        strike = self._strike_for_delta(entry, ceiling, option_type, t_entry, b)
        if strike is None:
            return 0.0, 0.0, 0.0, 0, 0.0

        entry_prem = bs_price(entry, strike, t_entry, b.assumed_iv,
                              b.risk_free_rate, option_type)
        exit_prem = bs_price(exit_, strike, t_exit, b.assumed_iv,
                             b.risk_free_rate, option_type)

        qty = self.cfg.instrument.lot_size
        gross = (exit_prem - entry_prem) * qty
        friction = self.costs.total_friction(entry_prem, max(exit_prem, 0.05), qty)
        net = gross - friction
        return net / self.cfg.capital.risk_per_trade_rupees, \
            entry_prem, exit_prem, qty, net

    def _strike_for_delta(self, spot: float, target_delta: float,
                          option_type: str, t_years: float, b) -> Optional[int]:
        """Highest-delta strike at or under the risk-engine's ceiling."""
        step = self.cfg.instrument.strike_step
        atm = int(round(spot / step) * step)
        best, best_delta = None, -1.0
        for i in range(-15, 16):
            strike = atm + i * step
            if strike <= 0:
                continue
            d = bs_delta(spot, strike, t_years, b.assumed_iv,
                         b.risk_free_rate, option_type)
            if self.cfg.signal.min_delta <= d <= target_delta and d > best_delta:
                best, best_delta = strike, d
        return best
