"""
E5. Does ANY signal on 5-minute Nifty 100 bars produce excess forward return?

WHY THIS EXISTS, AND WHY IT IS NOT A BACKTEST

`intraday_equity/backtest.py` answered "do THESE four strategies work" and the
answer was -0.524R, worse than a randomised-entry null. `diagnostics.D8` then
isolated the entry and found every horizon's excess inside its 95% interval.
Both of those measured the strategies the book happens to ship.

This module asks the prior question the book never asked: is there excess
forward return after ANY simple signal on these bars, in ANY part of the
session - including the 09:30-11:45 window the engine structurally cannot
reach, because `Context.bars` is today's session only and `_preflight` needs
its warm-up inside the day (`config.py:683-694`, `warmup_sessions = 0`).

A DIAGNOSTIC OVER RAW BARS NEEDS NONE OF THAT MACHINERY. There is no scanner,
no ladder, no sizing, no cost model and no engine here, so the morning window
is observable for free. Warm-up seeding is only worth engineering if this
measurement says the morning holds something - measure first, build second.

THE THREE THINGS THAT MAKE THIS A TEST RATHER THAN A SEARCH

  TWO CONTROLS, STACKED. Every forward return has the benchmark's move over
  the identical bar window subtracted, and is then de-meaned by the
  LEAVE-ONE-OUT mean of that excess for the same (symbol, clock). The first
  removes market drift; the second removes per-symbol intraday seasonality.
  After both, each (symbol, clock) column sums to EXACTLY zero, so a cell can
  only score by picking WHICH SESSIONS to fire on. That is the only thing a
  signal could legitimately be doing.

  A FAMILY-WISE NULL, NOT A PER-CELL ONE. The grid is 13 signals x 3 windows x
  6 horizons = 234 cells. At a 0.05 per-cell threshold roughly twelve of them
  are expected to look significant on pure noise, and the repo has already
  been bitten by counting multiplicity per book instead of across the search
  (the best p anywhere across ~30 variants in four books is 0.077, which is
  WORSE than the ~0.032 that noise is expected to produce). So the statistic
  reported is the MAXIMUM t across the entire grid, scored against the
  distribution of that same maximum under the null.

  THE NULL IS A CIRCULAR SHIFT OF THE SIGNAL ALONG SESSIONS, applied to every
  symbol at once. It breaks the association between a signal and the returns
  that followed it while preserving everything else that makes these
  observations dependent: the signal's own autocorrelation, its time-of-day
  distribution, and - because the shift is shared across symbols - the
  cross-sectional clustering of firings against a cross-sectionally
  correlated market. Shuffling per symbol would destroy that clustering and
  produce a null far too narrow, which is how a diagnostic like this
  manufactures a discovery.

  With 739 sessions there are 738 non-trivial shifts, so the test is run
  EXHAUSTIVELY over all of them rather than sampled - an exact randomisation
  test, and deterministic. It is affordable because the statistic at every
  shift is a circular cross-correlation, computed for all shifts at once by
  FFT (`_xcorr_all_shifts`). The naive loop is ~10^12 operations; this is
  seconds.

ONE-SIDED, BECAUSE THE BOOK IS LONG-ONLY. A cell with a large NEGATIVE excess
describes a short edge this book cannot express, so the family-wise maximum is
taken over SIGNED t. Negative cells are printed and are not tradeable.

THE SIGNAL BAR NEVER COUNTS. A mask computed from bars through clock c is
shifted one bar forward before it is measured, matching `backtest.py`'s rule
that you cannot buy the close of the bar you are still deciding on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Callable

import numpy as np
import pandas as pd

#: Forward horizons in BARS, matching `diagnostics.DEFAULT_HORIZONS` so a
#: result here can be read beside D8's without a units conversion.
DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 4, 8, 12, 20)

#: (label, start, end) on BAR-CLOSE time, half-open on the right.
#:
#: The split is structural, not fitted. `morning` is the window the engine
#: cannot currently reach at all; `book` is what it actually trades today;
#: `late` is what remains before the 15:10 square-off. Three pre-registered
#: windows, and the family-wise null pays for all three.
WINDOWS: tuple[tuple[str, time, time], ...] = (
    ("morning", time(9, 30), time(11, 45)),
    ("book", time(11, 45), time(14, 30)),
    ("late", time(14, 30), time(15, 10)),
)

#: A cell needs at least this many firings before its t is allowed to compete
#: for the family-wise maximum. A cell that fires 40 times has been sampled,
#: not measured, and its t is the noisiest thing in the grid - letting it win
#: would make the whole test a search for the rarest signal.
MIN_FIRINGS = 500


# --------------------------------------------------------------- the panel

@dataclass
class Panel:
    """
    Every symbol's bars aligned onto one (symbol, session, clock) grid.

    The (session x clock) shape is what makes this tractable, and it is the
    same reason `diagnostics._session_matrix` uses it: "h bars later in the
    same session" becomes a column shift, which cannot silently reach into
    the next session the way a positional shift on a flat frame would.
    """
    symbols: list[str]
    dates: np.ndarray               # (D,) of datetime.date
    clocks: list                    # (C,) of datetime.time
    close: np.ndarray               # (S, D, C) float32, NaN where no bar
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    bench_close: np.ndarray         # (D, C) float32

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.close.shape

    def window_mask(self, start: time, end: time) -> np.ndarray:
        """(C,) boolean over clocks, half-open [start, end)."""
        return np.array([start <= c < end for c in self.clocks], dtype=bool)


def _pivot(frame: pd.DataFrame, column: str, dates, clocks) -> np.ndarray:
    """One symbol's column as a (D, C) grid, NaN where the bar is absent."""
    flat = pd.DataFrame({
        "v": frame[column].to_numpy(dtype=float),
        "d": frame.index.date,
        "t": frame.index.time,
    })
    mat = flat.pivot_table(index="d", columns="t", values="v", aggfunc="last")
    return mat.reindex(index=dates, columns=clocks).to_numpy(dtype=np.float32)


def build_panel(barset, symbols=None) -> Panel:
    """
    Align an `IntradayBarSet` onto one dense grid.

    Sessions and clocks are the UNION across symbols, so a symbol that halted
    for an afternoon becomes NaN rather than shifting every later bar by one
    position. Nothing downstream indexes positionally into a session.
    """
    syms = sorted(symbols if symbols is not None else barset.bars)
    syms = [s for s in syms if s in barset.bars and len(barset.bars[s])]
    if not syms:
        raise ValueError("no symbols with bars - nothing to measure")

    dates, clocks = set(), set()
    for s in syms:
        idx = barset.bars[s].index
        dates.update(idx.date)
        clocks.update(idx.time)
    if barset.benchmark is not None and len(barset.benchmark):
        clocks.update(barset.benchmark.index.time)
    dates = np.array(sorted(dates))
    clocks = sorted(clocks)

    def stack(col):
        return np.stack([_pivot(barset.bars[s], col, dates, clocks)
                         for s in syms])

    bench = (_pivot(barset.benchmark, "close", dates, clocks)
             if barset.benchmark is not None and len(barset.benchmark)
             else np.full((len(dates), len(clocks)), np.nan, dtype=np.float32))

    return Panel(symbols=syms, dates=dates, clocks=clocks,
                 close=stack("close"), high=stack("high"), low=stack("low"),
                 volume=stack("volume"), bench_close=bench)


# ------------------------------------------------------- forward, controlled

def _forward(close: np.ndarray, h: int) -> np.ndarray:
    """Return over the next `h` bars WITHIN the session. NaN past the close."""
    nxt = np.full_like(close, np.nan)
    if h < close.shape[-1]:
        nxt[..., :-h] = close[..., h:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return nxt / close - 1.0


def excess_forward(panel: Panel, h: int) -> np.ndarray:
    """
    (S, D, C) forward return, benchmark-subtracted then de-meaned per
    (symbol, clock) by a LEAVE-ONE-OUT mean.

    THE LEAVE-ONE-OUT IS NOT A REFINEMENT. Including a session in the control
    it is scored against shrinks the very excess being measured, and with ~739
    sessions per column the shrinkage is small enough to be invisible and
    large enough to matter across 234 cells.

    After both controls each (symbol, clock) column sums to exactly zero, so a
    cell scores only by choosing WHICH sessions to fire on. That is what makes
    the circular-shift null exactly centred rather than approximately so.
    """
    raw = _forward(panel.close, h)
    bench = _forward(panel.bench_close, h)[None, :, :]
    with np.errstate(invalid="ignore"):
        ex = raw - bench

    valid = np.isfinite(ex)
    filled = np.where(valid, ex, 0.0).astype(np.float64)
    n = valid.sum(axis=1, keepdims=True).astype(np.float64)   # per (S, 1, C)
    total = filled.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (total - filled) / np.maximum(n - 1.0, 1.0)
    out = np.where(valid & (n > 1), filled - loo, np.nan)
    return out.astype(np.float64)


# ------------------------------------------------------------- the signals

def _roll_mean(a: np.ndarray, n: int) -> np.ndarray:
    """Trailing mean over the last `n` bars along the clock axis, exclusive."""
    c = np.cumsum(np.nan_to_num(a, nan=0.0), axis=-1)
    k = np.cumsum(np.isfinite(a).astype(np.float64), axis=-1)
    out = np.full(a.shape, np.nan)
    out[..., n:] = ((c[..., n - 1:-1] - np.concatenate(
        [np.zeros(a.shape[:-1] + (1,)), c[..., :-n - 1]], axis=-1))
        / np.maximum(k[..., n - 1:-1] - np.concatenate(
            [np.zeros(a.shape[:-1] + (1,)), k[..., :-n - 1]], axis=-1), 1.0))
    return out


def _roll_std(a: np.ndarray, n: int) -> np.ndarray:
    """Trailing std of `a` over the last `n` bars, exclusive of the current."""
    m = _roll_mean(a, n)
    m2 = _roll_mean(a * a, n)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _bar_return(close: np.ndarray) -> np.ndarray:
    prev = np.full_like(close, np.nan)
    prev[..., 1:] = close[..., :-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return close / prev - 1.0


def _k_bar_return(close: np.ndarray, k: int) -> np.ndarray:
    prev = np.full_like(close, np.nan)
    if k < close.shape[-1]:
        prev[..., k:] = close[..., :-k]
    with np.errstate(invalid="ignore", divide="ignore"):
        return close / prev - 1.0


def _session_vwap(panel: Panel) -> np.ndarray:
    """Cumulative VWAP within each session. Stocks have real volume."""
    tp = (panel.high + panel.low + panel.close) / 3.0
    v = np.nan_to_num(panel.volume, nan=0.0)
    pv = np.cumsum(np.nan_to_num(tp, nan=0.0) * v, axis=-1)
    cv = np.cumsum(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cv > 0, pv / np.maximum(cv, 1e-9), np.nan)


def _prior_high(high: np.ndarray, n: int) -> np.ndarray:
    """Highest high of the `n` bars BEFORE the current one."""
    S, D, C = high.shape
    out = np.full(high.shape, np.nan)
    for i in range(n, C):
        out[..., i] = np.nanmax(high[..., i - n:i], axis=-1)
    return out


def _sigma_move(panel: Panel, k: int, mult: float, up: bool) -> np.ndarray:
    """A k-bar move beyond `mult` trailing standard deviations."""
    r = _k_bar_return(panel.close, k)
    sd = _roll_std(_bar_return(panel.close), 20) * np.sqrt(k)
    with np.errstate(invalid="ignore"):
        return (r >= mult * sd) if up else (r <= -mult * sd)


def _xs_decile(panel: Panel, top: bool) -> np.ndarray:
    """
    Cross-sectional decile of return-since-open, ranked across the universe
    at each (session, clock).

    THE ONLY FAMILY THIS BOOK HAS NEVER TESTED, and the one with a prior: the
    repo's single positive out-of-sample number came from a cross-sectional
    book (`swing`, +0.114R), while every time-series signal it has tried on
    intraday bars has been at or below zero.
    """
    open_px = panel.close[..., :1]
    with np.errstate(invalid="ignore", divide="ignore"):
        since = panel.close / open_px - 1.0
    finite = np.isfinite(since)

    # NaN SORTS LAST, WHICH IS THE WHOLE TRICK. A symbol that did not trade
    # this bar must not occupy a rank: pushing it to the end makes a finite
    # symbol's sorted position exactly the count of finite values below it.
    # Filling with -inf instead - as this did - hands every absent symbol the
    # BOTTOM ranks. That leaves `xs_top` roughly right and silently shifts
    # every genuine laggard out of `xs_bot`, so the one family this grid
    # exists to test would have been half measured.
    order = np.argsort(np.where(finite, since, np.nan), axis=0)
    S = since.shape[0]
    rank = np.empty(since.shape, dtype=np.float64)
    np.put_along_axis(
        rank, order,
        np.broadcast_to(np.arange(S, dtype=np.float64)[:, None, None],
                        since.shape).copy(), axis=0)
    live = finite.sum(axis=0, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = rank / np.maximum(live - 1.0, 1.0)
    keep = (pct >= 0.9) if top else (pct <= 0.1)
    return keep & finite & (live >= 20)


@dataclass(frozen=True)
class Signal:
    """One pre-registered cell of the grid."""
    key: str
    why: str
    build: Callable[[Panel], np.ndarray]


#: THE GRID. Thirteen signals, fixed before the run, spanning five families.
#: Every one is computable from raw bars alone - no strategy, no config, no
#: fitted threshold. Adding a signal after seeing the table would invalidate
#: the family-wise null, which is the whole point of writing them down here.
SIGNALS: tuple[Signal, ...] = (
    # --- continuation: the family the shipped strategies belong to
    Signal("cont_3", "3-bar move >= 1.5 sigma up",
           lambda p: _sigma_move(p, 3, 1.5, True)),
    Signal("cont_6", "6-bar move >= 1.5 sigma up",
           lambda p: _sigma_move(p, 6, 1.5, True)),
    Signal("cont_12", "12-bar move >= 1.5 sigma up",
           lambda p: _sigma_move(p, 12, 1.5, True)),
    Signal("break_12", "close above the prior 12-bar high",
           lambda p: p.close > _prior_high(p.high, 12)),
    Signal("break_24", "close above the prior 24-bar high",
           lambda p: p.close > _prior_high(p.high, 24)),

    # --- reversal: does buying weakness pay where buying strength did not
    Signal("rev_3", "3-bar move <= -1.5 sigma",
           lambda p: _sigma_move(p, 3, 1.5, False)),
    Signal("rev_6", "6-bar move <= -1.5 sigma",
           lambda p: _sigma_move(p, 6, 1.5, False)),
    Signal("rev_12", "12-bar move <= -1.5 sigma",
           lambda p: _sigma_move(p, 12, 1.5, False)),
    Signal("below_vwap", "close 1.5 sigma below session VWAP",
           lambda p: (p.close - _session_vwap(p))
           <= -1.5 * _roll_std(_bar_return(p.close), 20) * p.close * np.sqrt(6)),

    # --- volatility and flow
    Signal("range_exp", "bar range > 2x its trailing 20-bar mean",
           lambda p: (p.high - p.low) > 2.0 * _roll_mean(p.high - p.low, 20)),
    Signal("vol_surge", "volume > 3x its trailing 20-bar mean",
           lambda p: p.volume > 3.0 * _roll_mean(p.volume, 20)),

    # --- cross-sectional: never tested on this book
    Signal("xs_top", "top decile of return-since-open across the universe",
           lambda p: _xs_decile(p, True)),
    Signal("xs_bot", "bottom decile of return-since-open across the universe",
           lambda p: _xs_decile(p, False)),
)


def _executable(mask: np.ndarray) -> np.ndarray:
    """
    Shift a signal one bar forward. THE SIGNAL BAR NEVER COUNTS.

    A mask built from bars through clock c is only actionable from c+1, which
    is the same rule `backtest.py` enforces by filling at the next bar's open.
    Measuring the forward return from the signal bar itself would be a
    one-bar look-ahead - small, free to avoid, and exactly the bias that
    flatters rather than errors.
    """
    out = np.zeros_like(mask, dtype=bool)
    out[..., 1:] = mask[..., :-1]
    return out


# ------------------------------------------------------------ the statistic

def _xcorr_all_shifts(mask: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    `C[k] = sum over (s, d, c) of mask[s,d,c] * y[s,(d+k) mod D,c]`, for EVERY
    shift k at once.

    A circular cross-correlation along the session axis, so the FFT gives all
    D shifts for the price of one. The alternative - rolling the mask and
    re-reducing 5.4 million cells 738 times per cell of a 234-cell grid - is
    ~10^12 operations and is the reason exhaustive randomisation is usually
    replaced by a sampled one. `test_signal_grid.py` checks this against a
    brute-force loop on a small panel, because an off-by-one or a conjugate on
    the wrong operand here would produce a plausible null rather than an
    error.
    """
    D = mask.shape[1]
    mf = np.fft.rfft(mask.astype(np.float64), axis=1)
    yf = np.fft.rfft(y, axis=1)
    prod = np.conj(mf) * yf
    return np.fft.irfft(prod, n=D, axis=1).sum(axis=(0, 2))


@dataclass
class GridCell:
    """One (signal, window, horizon) result, with its whole null in hand."""
    signal: str
    window: str
    horizon: int
    n: int
    mean_excess: float
    t: float
    #: t at every non-trivial circular shift. Kept so the family-wise maximum
    #: can be assembled across cells afterwards rather than per cell.
    null_t: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def judgeable(self) -> bool:
        return self.n >= MIN_FIRINGS


def _cell_stats(mask: np.ndarray, e: np.ndarray, valid: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    `(n[k], t[k])` for every circular shift of `mask` against `e`.

    A firing contributes only where the SHIFTED position has a finite excess,
    so `n` is recomputed per shift from `valid` rather than assumed constant -
    otherwise late-session firings, whose forward return runs past the close,
    would inflate the denominator at some shifts and not others.
    """
    filled = np.where(valid, e, 0.0)
    n = _xcorr_all_shifts(mask, valid.astype(np.float64))
    s1 = _xcorr_all_shifts(mask, filled)
    s2 = _xcorr_all_shifts(mask, filled * filled)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = (s2 - s1 * s1 / n) / np.maximum(n - 1.0, 1.0)
        t = mean / np.sqrt(np.maximum(var, 1e-300) / n)
    return n, np.where(np.isfinite(t) & (n >= 2), t, 0.0)


def run_grid(panel: Panel, horizons=DEFAULT_HORIZONS, signals=SIGNALS,
             windows=WINDOWS, progress=None) -> list[GridCell]:
    """
    Every cell, each carrying the full distribution of its own null.

    Ordered horizon-outermost because `excess_forward` is the expensive part
    and is shared by every signal and window at that horizon.
    """
    masks = {}
    for sig in signals:
        m = sig.build(panel)
        masks[sig.key] = _executable(np.asarray(m, dtype=bool)
                                     & np.isfinite(panel.close))

    cells: list[GridCell] = []
    total = len(horizons) * len(signals) * len(windows)
    done = 0
    for h in horizons:
        e = excess_forward(panel, h)
        valid = np.isfinite(e)
        e = np.where(valid, e, 0.0)
        for wname, lo, hi in windows:
            wcol = panel.window_mask(lo, hi)[None, None, :]
            for sig in signals:
                m = masks[sig.key] & wcol
                n_all, t_all = _cell_stats(m, e, valid)
                cells.append(GridCell(
                    signal=sig.key, window=wname, horizon=h,
                    n=int(round(n_all[0])),
                    mean_excess=float(
                        _xcorr_all_shifts(m, e)[0] / max(n_all[0], 1.0)),
                    t=float(t_all[0]), null_t=t_all[1:].copy()))
                done += 1
                if progress and done % 20 == 0:
                    progress(done, total, f"h={h} {wname} {sig.key}")
    return cells


@dataclass
class GridResult:
    cells: list
    observed_max_t: float
    best: GridCell | None
    family_p: float
    n_shifts: int
    n_judgeable: int


def family_wise(cells: list) -> GridResult:
    """
    One p-value for the entire search.

    THE STATISTIC IS THE MAXIMUM t ACROSS THE GRID, and the null is the
    distribution of that same maximum - recomputed shift by shift, taking the
    max across every cell at each shift. Comparing each cell's t against its
    own null and then reporting the smallest p would be the multiple-testing
    error this module was built to avoid.

    One-sided on the positive tail: the book is long-only, so a large negative
    excess is a short edge it cannot express.
    """
    ok = [c for c in cells if c.judgeable and len(c.null_t)]
    if not ok:
        return GridResult(cells, float("nan"), None, float("nan"), 0, 0)

    obs = np.array([c.t for c in ok], dtype=float)
    null = np.vstack([c.null_t for c in ok])          # (cells, shifts)
    observed_max = float(obs.max())
    null_max = null.max(axis=0)                       # (shifts,)
    beat = int((null_max >= observed_max).sum())
    p = (1.0 + beat) / (1.0 + len(null_max))
    return GridResult(cells=cells, observed_max_t=observed_max,
                      best=ok[int(np.argmax(obs))], family_p=p,
                      n_shifts=len(null_max), n_judgeable=len(ok))


# ----------------------------------------------------------------- reporting

#: Round-trip friction on this book, as a fraction of price. Measured in E4
#: (-0.16% of price at a 2.0x ATR stop, flat across a 3x range of stop
#: multiples) and derived from TRADES rather than from config, per the repo's
#: own rule. An excess below this is real and untradeable.
#:
#: THIS NUMBER IS FROZEN BECAUSE IT WAS PRE-COMMITTED. `costs_intraday`
#: independently gives 0.183% with modelled slippage and 0.083% statutory-only,
#: so the pre-registered gate is the GENEROUS one and moving it now could only
#: make a failing result pass. `friction_reference()` reports the model's
#: figures beside it; the gate stays where it was written down.
FRICTION_PCT = 0.0016


def friction_reference() -> dict:
    """
    What the cost model says, so the frozen gate can be sanity-checked
    against it without becoming it.

    Reported as fractions of notional for a round trip. The statutory-only
    figure matters more than it looks: it is the floor no execution skill can
    get under, so an excess below THAT is untradeable under any assumption
    about fills at all.
    """
    from .costs_intraday import IntradayEquityCostModel
    c = IntradayEquityCostModel()
    px, q = 1000.0, 100.0
    notional = px * q
    return {
        "statutory_only": c.round_trip(px, px, q) / notional,
        "with_slippage": c.friction(px, px, q) / notional,
        "gate": FRICTION_PCT,
    }


def report(res: GridResult, top: int = 12) -> str:
    """The table, the gate, and nothing that decides anything."""
    lines = [
        "E5 - does ANY signal on 5-minute Nifty 100 bars produce excess "
        "forward return?",
        "",
        "  Excess = forward return, minus NIFTY over the identical bars,",
        "  minus the leave-one-out mean for the same (symbol, clock).",
        "  Each column sums to zero, so a cell scores only by choosing which",
        "  SESSIONS to fire on. Signal bar excluded: the mask is shifted one",
        "  bar forward, matching the backtest's next-bar fill.",
        "",
    ]
    if res.best is None:
        return "\n".join(lines + ["  no judgeable cell - nothing fired "
                                  f"{MIN_FIRINGS}+ times"])

    ranked = sorted([c for c in res.cells if c.judgeable],
                    key=lambda c: c.t, reverse=True)
    lines.append(f"  {'signal':<12}{'window':>9}{'h':>4}{'n':>9}"
                 f"{'excess %':>11}{'t':>8}")
    for c in ranked[:top]:
        lines.append(f"  {c.signal:<12}{c.window:>9}{c.horizon:>4}{c.n:>9,}"
                     f"{c.mean_excess * 100:>11.4f}{c.t:>8.2f}")
    if len(ranked) > top:
        worst = ranked[-1]
        lines.append(f"  {'...':<12}{'':>9}{'':>4}{'':>9}{'':>11}{'':>8}")
        lines.append(f"  {worst.signal:<12}{worst.window:>9}{worst.horizon:>4}"
                     f"{worst.n:>9,}{worst.mean_excess * 100:>11.4f}"
                     f"{worst.t:>8.2f}")

    # WINDOW ROLLUP. The pre-registered question behind `morning` was whether
    # the 11:45 constraint costs the book anything measurable. It is answered
    # by the best excess per window, not by the best t - t rewards sample size,
    # and the morning window has fewer bars by construction.
    lines.append("")
    lines.append(f"  {'window':<10}{'cells':>7}{'best excess %':>16}"
                 f"{'best t':>9}  best cell")
    for wname, _, _ in WINDOWS:
        g = [c for c in res.cells if c.window == wname and c.judgeable]
        if not g:
            continue
        top_x = max(g, key=lambda c: c.mean_excess)
        top_t = max(g, key=lambda c: c.t)
        lines.append(f"  {wname:<10}{len(g):>7}{top_x.mean_excess * 100:>16.4f}"
                     f"{top_t.t:>9.2f}  {top_x.signal}/h={top_x.horizon}")

    lines.append("")
    lines.append(f"  {'family':<14}{'best excess %':>16}{'best t':>9}")
    for fam, keys in (("continuation", ("cont_3", "cont_6", "cont_12",
                                        "break_12", "break_24")),
                      ("reversal", ("rev_3", "rev_6", "rev_12",
                                    "below_vwap")),
                      ("vol/flow", ("range_exp", "vol_surge")),
                      ("cross-section", ("xs_top", "xs_bot"))):
        g = [c for c in res.cells if c.signal in keys and c.judgeable]
        if not g:
            continue
        lines.append(f"  {fam:<14}"
                     f"{max(c.mean_excess for c in g) * 100:>16.4f}"
                     f"{max(c.t for c in g):>9.2f}")

    b = res.best
    lines += [
        "",
        f"  grid: {len(res.cells)} cells, {res.n_judgeable} with "
        f"{MIN_FIRINGS}+ firings",
        f"  best: {b.signal} / {b.window} / h={b.horizon}  "
        f"t = {res.observed_max_t:+.2f}, excess {b.mean_excess * 100:+.4f}%",
        f"  FAMILY-WISE p = {res.family_p:.4f}  (exact, over "
        f"{res.n_shifts} circular shifts of the session axis)",
        "",
        "  THE GATE, committed before the run:",
    ]
    sig_ok = res.family_p <= 0.05
    trade_ok = b.mean_excess > FRICTION_PCT
    lines += [
        f"    1. family-wise p <= 0.05 ......... "
        f"{'PASS' if sig_ok else 'FAIL'}  (p = {res.family_p:.4f})",
        f"    2. excess > {FRICTION_PCT:.2%} of price ...... "
        f"{'PASS' if trade_ok else 'FAIL'}  "
        f"(best = {b.mean_excess * 100:+.4f}%)",
        "",
    ]
    ref = friction_reference()
    lines += [
        f"  friction reference (from costs_intraday, NOT the gate):",
        f"    statutory only ... {ref['statutory_only']:.4%} of notional "
        f"-> best cell is {ref['statutory_only'] / max(b.mean_excess, 1e-12):.1f}x short",
        f"    with slippage .... {ref['with_slippage']:.4%} of notional "
        f"-> best cell is {ref['with_slippage'] / max(b.mean_excess, 1e-12):.1f}x short",
        "    The statutory figure is the floor no execution skill gets under.",
        "",
    ]
    if sig_ok and trade_ok:
        lines.append("  BOTH PASS - this cell has earned a backtest, and "
                     "nothing else in the grid has.")
    elif not sig_ok:
        lines.append("  GATE 1 FAILED. The intraday equity book is closed - "
                     "the category, not a variant.")
    else:
        lines.append("  GATE 2 FAILED. The effect survives the null and does "
                     "not survive the friction. Real, and untradeable.")
    lines.append("  A cell added after reading this table would invalidate "
                 "the family-wise null.")
    return "\n".join(lines)


def write_ledger(res: GridResult, directory="data/experiments",
                 prefix: str = "e5_signal_grid"):
    """
    One parquet per invocation, matching `experiment_core.write_ledger`.

    A result that cannot be re-read is not a result - and this run costs 35
    minutes, so re-deriving a column by re-running it is the kind of friction
    that ends with the table being quoted from a terminal buffer instead. The
    per-shift null is NOT stored: it is 738 floats x 234 cells, it is exactly
    reproducible from the panel, and the one number it produces (`family_p`)
    is on every row.
    """
    from pathlib import Path
    from datetime import datetime
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{
        "signal": c.signal, "window": c.window, "horizon": c.horizon,
        "n": c.n, "mean_excess": c.mean_excess, "t": c.t,
        "judgeable": c.judgeable,
        "family_p": res.family_p, "observed_max_t": res.observed_max_t,
        "n_shifts": res.n_shifts,
    } for c in res.cells])
    path = directory / f"{prefix}_{datetime.now():%Y%m%d-%H%M%S}.parquet"
    frame.to_parquet(path, index=False)
    return path


__all__ = ["Panel", "build_panel", "excess_forward", "Signal", "SIGNALS",
           "WINDOWS", "DEFAULT_HORIZONS", "MIN_FIRINGS", "FRICTION_PCT",
           "GridCell", "GridResult", "run_grid", "family_wise", "report",
           "friction_reference", "write_ledger"]


# ----------------------------------------------------------------- the CLI

def _main(argv=None) -> int:      # pragma: no cover
    import argparse
    import sys
    import time as _clock

    from ..config import DEFAULT
    from ..swing import markets as markets_mod
    from ..swing.universe import load_universe
    from . import bars as bars_mod

    p = argparse.ArgumentParser(description="E5 - the signal grid.")
    p.add_argument("--market", default="india")
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    market = markets_mod.get(cfg, args.market)
    wanted = [s.symbol for s in load_universe(market.universe_csv)]
    loaded = bars_mod.load_cached(
        market.cache_suffix, args.interval, wanted, market.benchmark_ticker,
        cache_dir=cfg.intraday_equity.cache_dir)
    if loaded is None or not loaded.bars:
        print("No intraday cache. Run:\n"
              f"  python scripts/fetch_intraday_history.py "
              f"--market {args.market} --years 3")
        return 2

    # The reporting channels are findings, not noise - a universe that quietly
    # shrank has already cost this repo a backtest of unknown vintage.
    print(loaded.note(), flush=True)
    if loaded.benchmark is None or not len(loaded.benchmark):
        print("NO BENCHMARK - the market-drift control cannot be applied. "
              "Refusing to run: an uncontrolled excess is not an excess.")
        return 2

    panel = build_panel(loaded)
    S, D, C = panel.shape
    print(f"panel {S} symbols x {D} sessions x {C} clocks "
          f"({panel.dates[0]} -> {panel.dates[-1]})", flush=True)
    print(f"grid  {len(SIGNALS)} signals x {len(WINDOWS)} windows x "
          f"{len(DEFAULT_HORIZONS)} horizons = "
          f"{len(SIGNALS) * len(WINDOWS) * len(DEFAULT_HORIZONS)} cells, "
          f"null = {D - 1} exact circular shifts", flush=True)

    started = _clock.time()

    def progress(n, total, label):
        print(f"  [{n}/{total}] {label}", flush=True)

    cells = run_grid(panel, progress=progress)
    res = family_wise(cells)
    print(f"\ndone in {_clock.time() - started:.0f}s\n", flush=True)
    print(report(res, top=args.top))
    try:
        print("")
        print(f"ledger -> {write_ledger(res)}")
    except Exception as e:
        print(f"could not write the ledger ({e}) - the table above stands")
    return 0


if __name__ == "__main__":        # pragma: no cover
    raise SystemExit(_main())
