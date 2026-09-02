"""
Does the intraday equity book work?

It calls `scanner.evaluate_session()` - the function the live runner calls -
and drives `positions.ExitLadder`, which the other two books drive. There is
no `if backtest:` anywhere in the decision path, so this measures the system
that would be traded rather than a reimplementation of it.

THE PESSIMISM RULES, EACH OF WHICH EXISTS TO STOP A FLATTERING NUMBER

  THE SIGNAL BAR NEVER FILLS. A signal on bar i fills at bar i+1's OPEN. You
  cannot buy the close of the bar you are still deciding on. If there is no
  bar i+1 - the signal came on the session's last bar - the signal is
  DISCARDED, never carried.

  A GAP THROUGH THE TRIGGER IS REJECTED, NOT BACKFILLED. If bar i+1 opens
  more than `max_chase_pct` above the signal close, the setup is gone. The
  tempting alternative - "fill at the signal close instead" - is fiction, and
  it is fiction that pays. A gap DOWN is taken at the open: it is a genuinely
  better fill, and refusing it would be a bias in the other direction.

  THE ENTRY BAR IS MANAGED, WHICH IS THE OPPOSITE OF THE SWING BOOK.
  `swing/backtest.py` skips the entry bar, correctly: a daily entry fills at
  a trigger somewhere inside the bar, so that bar's low may have happened
  BEFORE the fill. Here the fill is at the OPEN of bar i+1 - the first tick -
  so every subsequent tick in that bar, including its low, is reachable.
  Skipping it would delete same-bar stop-outs, which on 5-minute bars are a
  large share of all stops, and the deletion would fall entirely on LOSERS.

  TIES GO TO THE STOP. `ExitLadder.advance` tests `worst_r` against the stop
  the bar OPENED with, before any promotion from `best_r`. A bar whose range
  covers both the stop and the +2R rung is a stop, at the pre-bar level.

  THE STOP IS RE-CHECKED AGAINST THE FILL. `BarSignal.stop` is provisional,
  resolved off the signal bar's close; a gap can move a signal across the 5%
  cap between deciding and filling. Re-resolved, and rejected if it no longer
  fits - never clamped.

WHAT THE SWING BACKTESTER DOES THAT IS UNNECESSARY HERE

`settle_days` and the entire settlement tail are ABSENT, and their absence is
deliberate rather than an oversight. That machinery corrects a measured
0.176R bias where a walk-forward window deleted positions still open at its
boundary - losers stop out in days, winners trail for weeks, so it removed
winners preferentially. Here EVERY position closes by 15:10 the same day, so
a fold boundary is a session boundary and the bias is structurally zero. Do
not add settlement back; there is nothing for it to settle.

Also absent: the multi-day `valid_for_days` forward scan (the fill window is
one bar), news (there is no news layer, and a scoring component that never
takes its live branch is a divergence waiting to happen), and the earnings
blackout (a book flat by 15:10 never holds through a results gap - the one
swing caveat that genuinely does not transfer).

WHAT IT CANNOT MEASURE - printed above every result, never only here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd

from ..backtest import Metrics, compute_metrics
from ..config import DEFAULT, Config
from ..positions import ExitKind, ExitLadder
from . import ranking, scanner, sizing
from .costs_intraday import (DEFAULT_INTRADAY_EQUITY_COSTS, VERIFIED_ON,
                             IntradayEquityCostModel)

CAVEATS = """\
WHAT THIS BACKTEST CANNOT SEE
  1. SURVIVORSHIP, TWICE OVER. The universe file is TODAY'S index, and a
     long-only book on today's constituents is long precisely the names that
     survived. Worse than the swing book's version: 5-minute history for a
     delisted name does not exist in the cache at all, so those names are
     absent SILENTLY rather than loudly.
  2. NO HALAL SCREEN. There is no point-in-time fundamental data, so the
     screen cannot be replayed. This tests a LARGER universe than the live
     book will trade.
  3. SLIPPAGE IS AN ASSUMPTION, NOT A MEASUREMENT. See the figure printed
     below; it is a pre-registered guess until real fills have measured it,
     and at these stop distances it is a large share of total friction.
  4. THE PREFILTER CHANGES THE QUESTION. With an RS prefilter this cannot
     answer "how often did ANY Nifty 100 name fire", only "how often did a
     top-N-by-RS name fire". The `norank` variant measures what that costs.
  5. VOLUME GATES BEHAVE DIFFERENTLY HERE than in the option book. Stocks
     have real volume, so volume_surge/vwap/liquidity take their real branch
     rather than the index's range/TWAP proxy. The strategies were calibrated
     against the proxy. This is a behaviour change, not a port.
"""


# --------------------------------------------------------------- results


@dataclass
class IntradayTrade:
    """
    One completed round trip.

    Field names deliberately mirror `backtest.BacktestTrade` so
    `compute_metrics` consumes these with no adapter - the same trick
    `swing/backtest.SwingTrade` uses.
    """
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    strategy: str
    direction: str
    regime: str
    symbol: str
    entry_underlying: float
    exit_underlying: float
    stop_points: float
    outcome: str
    bars_held: int
    r_multiple: float
    reason: str = ""
    quantity: int = 0
    net_pnl: float = 0.0
    exit_legs: int = 1
    partial_banked: bool = False
    ambiguous: bool = False
    option_type: str = ""
    lots: int = 1


@dataclass
class DayStats:
    days: int = 0
    days_flat: int = 0
    max_concurrent: int = 0
    signals_seen: int = 0
    capital_blocks: int = 0
    heat_blocks: int = 0
    slot_blocks: int = 0
    gap_blocks: int = 0
    stop_cap_blocks: int = 0
    late_signal_blocks: int = 0


@dataclass
class Result:
    trades: list = field(default_factory=list)
    metrics: Metrics | None = None
    stats: DayStats = field(default_factory=DayStats)
    rejections: dict = field(default_factory=dict)
    start: date | None = None
    end: date | None = None
    symbols: int = 0
    missing_symbols: list = field(default_factory=list)
    costs: IntradayEquityCostModel = DEFAULT_INTRADAY_EQUITY_COSTS
    pot_note: str = ""
    typical_stop_pct: float = 0.0
    friction_r: float = 0.0

    @property
    def ambiguous_count(self) -> int:
        return sum(1 for t in self.trades if t.ambiguous)

    def headline(self) -> str:
        m = self.metrics
        if m is None or not m.trades:
            return "no trades"
        return (f"{m.trades} trades  win {m.win_rate:.1%} "
                f"(realised breakeven {m.breakeven_win_rate:.1%}, "
                f"design {m.target_breakeven_win_rate:.1%})  "
                f"expectancy {m.expectancy_r:+.3f}R  total {m.total_r:+.1f}R")

    def by_strategy(self) -> dict:
        out: dict = {}
        for t in self.trades:
            b = out.setdefault(t.strategy, {"n": 0, "r": 0.0})
            b["n"] += 1
            b["r"] += t.r_multiple
        for b in out.values():
            b["expectancy"] = b["r"] / b["n"] if b["n"] else 0.0
        return out


# --------------------------------------------------------------- cache


#: Fields the SCAN cannot see, so changing them must NOT invalidate a cached
#: scan. The default is "invalidates", exactly as in
#: `swing/backtest.BOOK_ONLY_SWING_FIELDS`, and the asymmetry is the point:
#: forgetting to exempt a book-only field costs a re-scan, while failing to
#: notice a scan-affecting field would serve a stale answer fluently.
#:
#: `min_confidence` and `enforce_regime_gate` are exempt DELIBERATELY - they
#: are stored on each BarSignal and applied by the consumer, so they are the
#: two most useful knobs and also free to sweep. Putting them inside the
#: cached unit would have made them the most expensive.
BOOK_ONLY_FIELDS = frozenset({
    "top_n", "max_open_risk_r", "max_per_sector",
    "trail_atr_multiple", "partial_exit_fraction", "breakeven_at_r",
    "mis_leverage", "participation_cap_pct", "max_chase_pct",
    "min_confidence", "enforce_regime_gate",
    "rs_prefilter_n", "rs_short_days", "rs_long_days",
    "w_setup", "w_relative_strength", "w_reward_risk", "w_volume",
    "w_session_position",
    "cache_dir", "instruments_cache_days", "history_years", "halal_screen",
    "min_session_turnover_inr",
})


def signal_signature(cfg: Config, universe_key: str = "") -> str:
    """
    Identity of everything `evaluate_session` reads.

    Hashes all of `IntradayEquityConfig` except `BOOK_ONLY_FIELDS`, plus the
    whole of SignalConfig, StrategyConfig and RegimeConfig - every field of
    which is read inside `on_bar` - plus the universe.

    The universe is in the hash and the swing version has no equivalent,
    because there the cache is keyed per day and iterated fresh. HERE THE
    CACHE IS THE UNIVERSE: a symbol added between two runs would otherwise
    produce a cache that is correct for every day it holds and silently short
    one name, which reads as "that name never had a setup".
    """
    ie = cfg.intraday_equity
    parts = []
    for name, value in sorted(vars(ie).items()):
        if name in BOOK_ONLY_FIELDS:
            continue
        parts.append(f"ie.{name}={value!r}")
    for section in ("signal", "strategy", "regime"):
        obj = getattr(cfg, section)
        for name, value in sorted(vars(obj).items()):
            parts.append(f"{section}.{name}={value!r}")
    parts.append(f"universe={universe_key}")
    blob = "|".join(parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class SignalCache:
    """
    (symbol, day) -> the signals that session produced.

    Legitimate because `evaluate_session` is pure in the tape and the signal
    config and does not read the book. Everything downstream - ranking,
    sizing, the ladder, concurrency, capital, costs - runs outside it, so one
    scan pass serves every ladder variant.

    A signature mismatch RAISES. It never falls back to rescanning, because a
    stale scan answers a different question fluently.
    """

    def __init__(self, signature: str):
        self.signature = signature
        self.days: dict = {}

    def get(self, symbol: str, day: date):
        return self.days.get((symbol, day))

    def put(self, symbol: str, day: date, scan) -> None:
        self.days[(symbol, day)] = scan

    @property
    def sessions_cached(self) -> int:
        return len(self.days)


# --------------------------------------------------------------- the run


@dataclass
class _Open:
    signal: scanner.BarSignal
    sized: sizing.Sized
    state: object
    ladder: ExitLadder
    entry_time: pd.Timestamp
    entry_index: int
    quantity_open: int
    realised_r: float = 0.0
    legs: int = 1
    partial_banked: bool = False


def intraday_trade_cfg(cfg: Config):
    """
    The ladder settings for this book.

    `dataclasses.replace` off `cfg.trade`, exactly as `swing.book.swing_trade`
    does, so a RUNG added to the ladder cannot exist for one book and not
    another. Only the trail distance and the partial fraction differ.
    """
    ie = cfg.intraday_equity
    fields = dict(trail_atr_multiple=ie.trail_atr_multiple,
                  partial_exit_fraction=ie.partial_exit_fraction)
    if ie.breakeven_at_r is not None:
        fields["breakeven_at_r"] = ie.breakeven_at_r
    return replace(cfg.trade, **fields)


def _fill_price(next_bar, cfg, costs) -> float:
    """The next bar's open, plus adverse slippage. Never the signal close."""
    slip = 1.0 + costs.slippage_pct
    tick = cfg.intraday_equity_broker.tick_size
    raw = float(next_bar["open"]) * slip
    return round(round(raw / tick) * tick, 2) if tick else raw


def run(cfg: Config, bars: dict, benchmark: pd.DataFrame | None = None,
        start: date | None = None, end: date | None = None,
        costs: IntradayEquityCostModel = DEFAULT_INTRADAY_EQUITY_COSTS,
        sector_of: dict | None = None,
        cache: SignalCache | None = None,
        universe_key: str = "",
        progress=None) -> Result:
    """
    Replay the book over `bars` - {symbol: intraday DataFrame}.

    One pass per session: rank the universe from PRIOR sessions, scan the
    watchlist, then walk the day's bars managing open positions BEFORE
    considering new ones (invariant 5 - open money outranks a new
    opportunity).
    """
    ie = cfg.intraday_equity
    result = Result(costs=costs, symbols=len(bars))

    if cache is not None:
        want = signal_signature(cfg, universe_key)
        if cache.signature != want:
            raise ValueError(
                f"signal cache signature {cache.signature} does not match the "
                f"current config ({want}). A stale scan answers a different "
                f"question fluently, so this refuses rather than falling back.")

    sessions = sorted({ts.date() for df in bars.values() for ts in df.index})
    if start:
        sessions = [d for d in sessions if d >= start]
    if end:
        sessions = [d for d in sessions if d <= end]
    if not sessions:
        return result
    result.start, result.end = sessions[0], sessions[-1]

    ladder = ExitLadder(cfg, trade=intraday_trade_cfg(cfg))
    trail_r = ie.trail_atr_multiple
    stop_pcts: list[float] = []

    for n, day in enumerate(sessions, 1):
        if progress:
            progress(n, len(sessions), day)
        result.stats.days += 1

        # ---- the morning cut, from PRIOR sessions only (trap T1) --------
        ranks = ranking.morning_ranks(bars, benchmark, day, cfg)
        watch = ranking.prefilter(ranks, cfg)
        morning = {r.symbol: r for r in ranks}
        if not watch:
            result.stats.days_flat += 1
            continue

        # ---- scan the watchlist ----------------------------------------
        per_symbol = {}
        for symbol in watch:
            frame = bars.get(symbol)
            if frame is None:
                continue
            session = frame[frame.index.date == day]
            if session.empty:
                continue
            prior = frame[frame.index.date < day]
            if prior.empty:
                continue
            last = prior[prior.index.date == prior.index[-1].date()]

            hit = cache.get(symbol, day) if cache is not None else None
            if hit is None:
                hit = scanner.evaluate_session(
                    symbol, session, cfg,
                    prev_high=float(last["high"].max()),
                    prev_low=float(last["low"].min()),
                    prev_close=float(last["close"].iloc[-1]))
                if cache is not None:
                    cache.put(symbol, day, hit)
            per_symbol[symbol] = (session, hit)
            for (stage, detail), count in hit.rejections.items():
                result.rejections[stage] = result.rejections.get(stage, 0) + count
            result.stats.signals_seen += len(hit.signals)

        if not per_symbol:
            result.stats.days_flat += 1
            continue

        # ---- walk the day ----------------------------------------------
        by_bar: dict = {}
        for symbol, (_, scan) in per_symbol.items():
            for s in scan.signals:
                by_bar.setdefault(s.bar_index, []).append(s)

        open_positions: dict = {}
        deployed = 0.0
        traded_today = 0
        bar_count = max(len(s) for s, _ in per_symbol.values())

        for i in range(bar_count):
            # 1. MANAGE FIRST. Open money outranks a new opportunity.
            for symbol in list(open_positions):
                pos = open_positions[symbol]
                session = per_symbol[symbol][0]
                if i >= len(session) or i <= pos.entry_index - 1:
                    continue
                bar = session.iloc[i]
                closed = _manage(pos, bar, session.index[i], i, ladder,
                                 trail_r, cfg, costs, result)
                if closed:
                    deployed -= pos.sized.deployed_inr
                    del open_positions[symbol]

            result.stats.max_concurrent = max(result.stats.max_concurrent,
                                              len(open_positions))

            # 2. then look for entries, on signals from bar i
            candidates = by_bar.get(i, [])
            if not candidates:
                continue
            room = ie.top_n - len(open_positions)
            if room <= 0:
                result.stats.slot_blocks += len(candidates)
                continue
            if traded_today >= cfg.capital.max_entries_per_session:
                continue

            open_risk = sum(1.0 for _ in open_positions)
            if open_risk + 1 > ie.max_open_risk_r:
                result.stats.heat_blocks += len(candidates)
                continue

            picks = scanner.rank_signals(
                candidates, cfg, morning=morning, top_n=room,
                held=set(open_positions), sector_of=sector_of)

            for _score, sig, _parts in picks:
                session = per_symbol[sig.symbol][0]
                # THE SIGNAL BAR NEVER FILLS.
                if i + 1 >= len(session):
                    result.stats.late_signal_blocks += 1
                    continue
                nxt = session.iloc[i + 1]

                fill = _fill_price(nxt, cfg, costs)
                if fill > sig.ref_close * (1.0 + ie.max_chase_pct):
                    result.stats.gap_blocks += 1
                    continue

                # Re-resolve the stop against the ACTUAL fill.
                stop, why = sizing.resolve_stop(fill, sig.atr, cfg)
                if why:
                    result.stats.stop_cap_blocks += 1
                    continue

                free = cfg.capital.capital_inr("intraday_equity") - deployed
                sized, why = sizing.size(fill, stop, cfg,
                                         free_capital_inr=max(free, 0.0),
                                         bar_volume=float(nxt.get("volume", 0)))
                if sized is None or sized.quantity <= 0:
                    result.stats.capital_blocks += 1
                    continue

                st = ladder.new_state(sized.quantity)
                open_positions[sig.symbol] = _Open(
                    signal=sig, sized=sized, state=st, ladder=ladder,
                    entry_time=session.index[i + 1], entry_index=i + 1,
                    quantity_open=sized.quantity)
                deployed += sized.deployed_inr
                traded_today += 1
                stop_pcts.append(sized.stop_pct)

        # ---- nothing survives the session -------------------------------
        for symbol, pos in list(open_positions.items()):
            session = per_symbol[symbol][0]
            last_i = len(session) - 1
            _close(pos, float(session["close"].iloc[-1]), session.index[-1],
                   last_i, "session_end", cfg, costs, result)
        if not open_positions:
            result.stats.days_flat += 1

    # ---- report -------------------------------------------------------
    result.metrics = compute_metrics(result.trades, cfg.capital.reward_risk_ratio)
    result.typical_stop_pct = float(np.median(stop_pcts)) if stop_pcts else 0.006
    result.pot_note = sizing.pot_note(result.typical_stop_pct, cfg)
    if stop_pcts:
        entry = 1000.0
        stop = entry * (1 - result.typical_stop_pct)
        qty = max(1, int(cfg.capital.risk_inr("intraday_equity") / (entry - stop)))
        result.friction_r = costs.friction_r(entry, stop, qty)
    return result


def _manage(pos: _Open, bar, ts, i: int, ladder: ExitLadder, trail_r: float,
            cfg, costs, result) -> bool:
    """
    One bar of position management. Returns True when the position closed.

    THE ENTRY BAR IS INCLUDED - see the module docstring. The fill is that
    bar's open, so its low is reachable and a same-bar stop is a real stop.
    """
    ie = cfg.intraday_equity
    entry = pos.sized.entry
    rp = pos.sized.risk_points
    if rp <= 0:
        return False

    best_r = (float(bar["high"]) - entry) / rp
    worst_r = (float(bar["low"]) - entry) / rp
    mark_r = (float(bar["close"]) - entry) / rp

    # A bar that covers both the stop and the target is AMBIGUOUS, and the
    # ladder resolves it pessimistically - the stop wins, at the level the
    # bar opened with.
    hit_stop = worst_r <= pos.state.stop_r
    hit_target = best_r >= cfg.capital.reward_risk_ratio
    ambiguous = hit_stop and hit_target

    # The square-off is on the bar's CLOSE time, not its stamp.
    closing = scanner._bar_close_time(ts, ie.bar_interval_minutes)
    if closing >= ie.force_exit:
        _close(pos, float(bar["close"]), ts, i, "force_exit", cfg, costs,
               result, ambiguous=ambiguous)
        return True

    decisions = ladder.advance(pos.state, mark_r, best_r=best_r,
                               worst_r=worst_r, trail_distance_r=trail_r)
    for d in decisions:
        if d.kind in (ExitKind.STOPPED_OUT, ExitKind.TARGET_EXIT):
            _close(pos, entry + d.exit_r * rp, ts, i,
                   "stop" if d.kind is ExitKind.STOPPED_OUT else "target",
                   cfg, costs, result, ambiguous=ambiguous)
            return True
        if d.kind is ExitKind.PARTIAL_EXIT:
            pos.realised_r += d.exit_r * (d.exit_lots / pos.sized.quantity)
            pos.quantity_open -= d.exit_lots
            pos.legs += 1
            pos.partial_banked = True
    return pos.state.closed


def _close(pos: _Open, price: float, ts, i: int, reason: str, cfg, costs,
           result, ambiguous: bool = False) -> None:
    """Book the trade, net of friction charged PER LEG."""
    entry = pos.sized.entry
    rp = pos.sized.risk_points
    qty = pos.sized.quantity

    gross = (price - entry) * pos.quantity_open + pos.realised_r * rp * qty
    friction = (costs.buy_cost(entry, qty) + costs.slippage(entry, qty))
    friction += (costs.sell_cost(price, pos.quantity_open)
                 + costs.slippage(price, pos.quantity_open))
    if pos.partial_banked:
        # A banked partial is a THIRD leg, and it pays its own charges.
        banked_qty = qty - pos.quantity_open
        friction += (costs.sell_cost(entry + rp * 2.0, banked_qty)
                     + costs.slippage(entry, banked_qty))

    net = gross - friction
    risk_inr = rp * qty
    r = net / risk_inr if risk_inr > 0 else 0.0
    if ambiguous:
        # One bar covering both stop and target is recorded and counted a
        # LOSS, exactly as the swing book does.
        r = min(r, -1.0)

    result.trades.append(IntradayTrade(
        entry_time=pos.entry_time, exit_time=ts, strategy=pos.signal.strategy,
        direction=pos.signal.direction, regime=pos.signal.regime,
        symbol=pos.signal.symbol, entry_underlying=entry,
        exit_underlying=price, stop_points=rp, outcome=reason,
        bars_held=max(0, i - pos.entry_index), r_multiple=r, reason=reason,
        quantity=qty, net_pnl=net, exit_legs=pos.legs,
        partial_banked=pos.partial_banked, ambiguous=ambiguous))
    pos.state.lots_remaining = 0


__all__ = ["run", "Result", "IntradayTrade", "DayStats", "SignalCache",
           "signal_signature", "BOOK_ONLY_FIELDS", "intraday_trade_cfg",
           "CAVEATS"]
