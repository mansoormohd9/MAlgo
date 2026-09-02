"""
One session, one symbol, every bar: which signals fired and why the rest did
not.

THE FUNCTION THE BACKTEST AND THE LIVE RUNNER BOTH CALL

`evaluate_session()` is pure in `(symbol, session bars, signal config)`. It
does NOT read the book - not open positions, not remaining slots, not the
day's P&L. That is the same extraction `swing/scanner.evaluate_symbol` made
and for the same reason: a backtest that reimplements the decision logic is
measuring a different system. It is also what makes three other things
possible at once - caching a session's signals across ladder variants,
sharding the work across processes, and running the identical code live.

Everything book-dependent - ranking against other symbols, the concurrency
cap, heat, capital, the ladder - happens in the CONSUMER.

THE GATES ARE ORDERED CHEAP TO EXPENSIVE

    universe -> prefiltered? -> session integrity -> warm-up -> entry window
      -> regime -> strategy signals -> stop bounds -> R:R -> (consumer ranks)

A symbol that was never going to qualify should not cost an indicator, and a
bar outside the entry window should not cost a strategy call.

THE REJECTION LEDGER ACCOUNTS FOR EVERY SYMBOL

"Nothing fired" and "the data was not there" are different facts, and on the
first run of a new book the ledger is usually the *finding*. It also drives
the live page's "why nothing fired" table, which is the only thing that makes
a quiet day legible.

WHAT THE WARM-UP MEANS HERE, AND WHY IT IS NOT A BUG

`Context.bars` is TODAY'S SESSION ONLY - `engine.py:165` slices it that way
and this module matches it exactly. `BaseStrategy._preflight` needs
`max(atr_period, min_bars)` bars, which at 5 minutes is 125 minutes past the
open. So nothing can fire before ~11:20 and the effective window is roughly
11:20 to `entry_cutoff`.

Do not "fix" that by prepending yesterday's bars. `signals.opening_range`
reads the first bars OF THE FRAME with no day-awareness, so a multi-day frame
silently yields YESTERDAY's opening range - and `orb_retest`, plus every
level built from the opening range, would trade against the wrong number with
no error. `warmseed` is the pre-registered variant that tests widening the
window properly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time, timedelta

import pandas as pd

from .. import regime as regime_mod
from ..data.csv_feed import BarReplayer
from ..strategies.registry import build_enabled
from ..strategy import Context
from . import precompute as pre
from . import ranking, sizing

# ---- rejection stages, cheap to expensive -----------------------------
STAGE_NO_DATA = "no_data"
STAGE_INTEGRITY = "integrity"
STAGE_NOT_PREFILTERED = "not_in_watchlist"
STAGE_LIQUIDITY = "session_liquidity"
STAGE_ATR_BAND = "atr_band"
STAGE_WARMUP = "warming_up"
STAGE_WINDOW = "outside_entry_window"
STAGE_REGIME = "regime"
STAGE_NO_SETUP = "no_setup"
STAGE_CONFIDENCE = "below_min_confidence"
STAGE_DIRECTION = "short_not_allowed"
STAGE_STOP = "stop_bounds"
STAGE_RR = "reward_risk"

STAGES = (STAGE_NO_DATA, STAGE_INTEGRITY, STAGE_NOT_PREFILTERED,
          STAGE_LIQUIDITY, STAGE_ATR_BAND, STAGE_WARMUP, STAGE_WINDOW,
          STAGE_REGIME, STAGE_NO_SETUP, STAGE_CONFIDENCE, STAGE_DIRECTION,
          STAGE_STOP, STAGE_RR)

LONG = "long"


@dataclass(frozen=True)
class BarSignal:
    """
    One strategy firing on one bar.

    Everything here is decided from bars <= `bar_index`. Fields the CONSUMER
    needs in order to rank, gate or size are carried rather than recomputed,
    which is what lets one scan pass serve every ladder variant.

    `stop` and `stop_pct` are provisional - resolved against the signal bar's
    close. The book re-resolves them against the actual fill, because a gap
    can move a signal across the 5% cap between the two.
    """
    symbol: str
    day: date
    bar_index: int
    bar_time: pd.Timestamp
    strategy: str
    direction: str
    confidence: float
    reason: str
    ref_close: float          # the close the signal was read at
    atr: float                # ATR at the SIGNAL bar; ATR[i+1] is look-ahead
    stop: float
    stop_pct: float
    regime: str
    #: ranking inputs, computed from the truncated window only
    volume_ratio: float | None = None
    session_position: float | None = None

    @property
    def risk_points(self) -> float:
        return self.ref_close - self.stop


@dataclass
class SessionScan:
    """One symbol, one session: what fired and what did not."""
    symbol: str
    day: date
    signals: list = field(default_factory=list)
    #: (stage, reason, count)
    rejections: dict = field(default_factory=dict)
    bars_evaluated: int = 0
    first_decision_bar: pd.Timestamp | None = None

    def reject(self, stage: str, detail: str = "") -> None:
        key = (stage, detail)
        self.rejections[key] = self.rejections.get(key, 0) + 1

    def stage_counts(self) -> dict:
        out: dict = {}
        for (stage, _), n in self.rejections.items():
            out[stage] = out.get(stage, 0) + n
        return out


def _bar_close_time(ts: pd.Timestamp, interval_minutes: int) -> time:
    """
    Kite stamps a bar at its OPEN, so the bar covering 15:10-15:15 is stamped
    15:10. Every window gate in this book is on the CLOSE, because acting on
    the open stamp squares the book off five minutes late and hands the
    result a free window of drift that raises no error.
    """
    return (ts + timedelta(minutes=interval_minutes)).time()


def evaluate_session(symbol: str, session: pd.DataFrame, cfg,
                     strategies: dict | None = None,
                     prev_high: float = 0.0, prev_low: float = 0.0,
                     prev_close: float = 0.0) -> SessionScan:
    """
    Every signal one symbol produced on one session.

    Pure in its arguments and blind to the book - see the module docstring.
    `strategies` is built from the registry when not supplied, so this book
    cannot drift from what the UI and the option engine enumerate.
    """
    ie = cfg.intraday_equity
    day = session.index[-1].date() if len(session) else None
    scan = SessionScan(symbol=symbol, day=day)

    if session is None or session.empty:
        scan.reject(STAGE_NO_DATA)
        return scan

    if strategies is None:
        keys = list(ie.enabled_strategies) or None
        strategies = build_enabled(keys, cfg)

    # One vectorised pass, then slice. Without this the whole backtest is
    # 33 hours rather than minutes - see `precompute.py`.
    pack = pre.build_pack(session, symbol, cfg)
    framed = pre.attach(session.copy(), pack)

    # `min_bars` is a BaseStrategy attribute; the two adapters in
    # registry.py wrap `strategy.LevelBreakStrategy` directly and do not have
    # it, so it is read defensively rather than assumed. The 25 matches
    # `BaseStrategy.min_bars` and `LevelBreakStrategy`'s own warm-up check.
    warmup = max(cfg.signal.atr_period,
                 max((getattr(s, "min_bars", 25) for s in strategies.values()),
                     default=25))
    if len(framed) <= warmup:
        scan.reject(STAGE_WARMUP, f"session has {len(framed)} bars")
        return scan

    replayer = BarReplayer(framed, warmup=warmup, max_window=10_000)
    reading = None

    for i in range(warmup, len(framed)):
        ts = framed.index[i]
        closing = _bar_close_time(ts, ie.bar_interval_minutes)

        # The entry window, on the bar's CLOSE. Cheap, so it comes before
        # any strategy call.
        if closing < ie.entry_start or closing > ie.entry_cutoff:
            scan.reject(STAGE_WINDOW)
            continue

        window = replayer.window_at(i)
        scan.bars_evaluated += 1
        if scan.first_decision_bar is None:
            scan.first_decision_bar = ts

        atr = pack.atr_at(i, cfg.signal.atr_period)
        if atr <= 0:
            scan.reject(STAGE_WARMUP, "zero atr")
            continue

        ref_close = float(window["close"].iloc[-1])
        atr_pct = atr / ref_close if ref_close > 0 else 0.0
        if not (ie.min_atr_pct <= atr_pct <= ie.max_atr_pct):
            scan.reject(STAGE_ATR_BAND, f"{atr_pct:.2%}")
            continue

        ctx = Context(
            bars=window,
            now=ts.time(),
            prev_day_high=prev_high,
            prev_day_low=prev_low,
            prev_day_close=prev_close,
        )

        # The regime READING is computed once per bar and stored on every
        # signal; whether to OBEY it is the consumer's call, which is what
        # makes `enforce_regime_gate` free to sweep.
        reading = regime_mod.classify(window, prev_close, cfg)

        fired_this_bar = False
        for key, strat in strategies.items():
            sigl = strat.on_bar(ctx)
            if not sigl.direction:
                continue
            fired_this_bar = True

            if sigl.direction != LONG and not ie.allow_short:
                scan.reject(STAGE_DIRECTION, key)
                continue

            stop, why = sizing.resolve_stop(ref_close, atr, cfg)
            if why:
                # The 5% cap REJECTS, never clamps. See sizing.py.
                scan.reject(STAGE_STOP, why)
                continue

            scan.signals.append(BarSignal(
                symbol=symbol, day=day, bar_index=i, bar_time=ts,
                strategy=key, direction=sigl.direction,
                confidence=float(sigl.confidence), reason=sigl.reason,
                ref_close=ref_close, atr=atr, stop=stop,
                stop_pct=(ref_close - stop) / ref_close,
                regime=reading.regime.value if reading else "unknown",
                volume_ratio=ranking.volume_ratio(window),
                session_position=ranking.session_position(window),
            ))

        if not fired_this_bar:
            scan.reject(STAGE_NO_SETUP)

    return scan


def eligible(sig: BarSignal, cfg) -> bool:
    """
    The consumer-side filters, applied at replay rather than at scan time.

    `min_confidence` and `enforce_regime_gate` are deliberately NOT baked
    into the cached signal: they are the two knobs most worth sweeping, and
    putting them inside the cached unit would make them the most expensive
    ones to move. Stored on the signal, applied here.
    """
    ie = cfg.intraday_equity
    if sig.confidence < ie.min_confidence:
        return False
    return True


def rank_signals(signals: list, cfg, morning: dict | None = None,
                 top_n: int | None = None,
                 held: set | None = None,
                 sector_of: dict | None = None) -> list:
    """
    Choose between signals that fired on the SAME bar.

    The cross-sectional layer. Reads only what each `BarSignal` already
    carries, all of which was computed from bars at or before its own
    decision bar - so this cannot reach forward even though it compares
    across symbols. That is trap T1, and `morning` is the prior-sessions-only
    rank from `ranking.morning_ranks`.

    ORDERING IS BY SCORE, NEVER BY DICT ORDER. Iterating the universe in file
    order under a concurrency cap would give alphabetically early names
    systematic priority and make the result depend on how the CSV is sorted.
    """
    held = held or set()
    morning = morning or {}
    rr = cfg.capital.reward_risk_ratio

    scored = []
    for s in signals:
        if s.symbol in held:
            continue
        if not eligible(s, cfg):
            continue
        total, parts = ranking.bar_score(
            confidence=s.confidence, reward_risk=rr,
            morning=morning.get(s.symbol),
            volume_ratio=s.volume_ratio,
            session_position=s.session_position, cfg=cfg)
        scored.append((total, s, parts))

    # Deterministic: score first, then symbol as a stable tiebreak so two
    # runs over a reshuffled universe produce an identical trade list.
    scored.sort(key=lambda r: (-r[0], r[1].symbol, r[1].strategy))

    out, seen_symbols, sector_count = [], set(), {}
    limit = cfg.intraday_equity.top_n if top_n is None else top_n
    for total, s, parts in scored:
        if len(out) >= limit:
            break
        if s.symbol in seen_symbols:
            continue                       # one ticket per name per bar
        if sector_of:
            sec = sector_of.get(s.symbol)
            if sec is not None:
                if sector_count.get(sec, 0) >= cfg.intraday_equity.max_per_sector:
                    continue
                sector_count[sec] = sector_count.get(sec, 0) + 1
        seen_symbols.add(s.symbol)
        out.append((total, s, parts))
    return out


__all__ = ["BarSignal", "SessionScan", "evaluate_session", "rank_signals",
           "eligible", "STAGES", "LONG"] + [
    n for n in dir() if n.startswith("STAGE_")]
