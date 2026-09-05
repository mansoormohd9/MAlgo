"""
Choosing between rule sets without fooling yourself.

WHY THIS EXISTS. The swing book returned -0.077R over 282 trades, and four
plausible fixes were available. Trying them one at a time against the whole
history and keeping whichever printed the best number is not evidence - it is
fitting, and with 282 trades and eight variants it would produce a positive
expectancy by luck alone roughly as often as by skill. So every variant is
measured the same way: chosen on a TRAIN window, scored on the TEST window
that follows it, and reported with both.

WHAT IT ACTUALLY ANSWERS. Not "which variant is best" - that question has no
honest answer from one sample. It answers "if you had picked by train
expectancy at each fold and traded that pick, what would you have got", which
is the only version of the question a live book can act on.

THE THREE RULES THIS FILE ENFORCES

  1. The out-of-sample column is the result. In-sample is printed beside it
     because the GAP between them is the most informative number here: a
     variant that is wonderful in train and ordinary in test is telling you
     how much of it was fitting.
  2. A cell with too few trades does not vote. `MIN_TRADES_TO_JUDGE` is not a
     statistical threshold, it is an admission that below it the number is
     noise wearing a decimal point.
  3. Nothing here tunes anything automatically. It prints a table. Choosing
     is a decision with money behind it and stays a human one - the same
     reason `engine.confirm_entry()` exists.

COST. One scan pass is shared by every variant that does not change the
signals (see `backtest.ScanCache`), so the regime, breakeven and capital
levers are nearly free. Variants that change stops or which archetypes fire
re-scan, because they genuinely produce different signals.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd

from ..config import Config
from . import backtest as bt
from . import markets as markets_mod
from . import prices as prices_mod
from .universe import load_universe

#: Below this many trades in a window, a cell is reported but never used to
#: choose. The swing book takes roughly 90 trades a year across 100 names, so
#: a two-month test window holds about 15 - which is why selection happens on
#: the six-month train window and not on the test one.
MIN_TRADES_TO_JUDGE = 20


# ------------------------------------------------------------------ variants

@dataclass(frozen=True)
class Variant:
    """One named change to the config, with the hypothesis it is testing."""
    key: str
    label: str
    why: str
    apply: Callable[[Config], None]

    def configure(self, base: Config) -> Config:
        cfg = _clone(base)
        self.apply(cfg)
        return cfg


def _clone(cfg: Config) -> Config:
    """
    A deep copy, because variants mutate nested dataclasses.

    `dataclasses.replace` is shallow: two variants would end up sharing one
    SwingConfig and the second would silently inherit the first's changes -
    a plausible-answer failure, not a crash.
    """
    import copy
    return copy.deepcopy(cfg)


def _set(**fields):
    def apply(cfg: Config) -> None:
        for name, value in fields.items():
            setattr(cfg.swing, name, value)
    return apply


#: The four levers from the diagnosis, plus the baseline they are measured
#: against. Every one of these is real config the live scanner honours.
VARIANTS: tuple[Variant, ...] = (
    Variant("baseline", "Baseline (as shipped)",
            "the -0.077R book, unchanged, so every other row has a comparison",
            _set()),

    # L1 - the market itself
    Variant("regime200", "Regime: benchmark > 200DMA",
            "a long-only breakout book has no defence against a downtrend; "
            "regime.py calls this the highest-leverage filter in the system",
            _set(regime_ma_days=200)),
    Variant("regime50", "Regime: benchmark > 50DMA",
            "the same idea at a horizon closer to the 8-day holding period",
            _set(regime_ma_days=50)),

    # L2 - the breakeven rung
    Variant("be15", "Breakeven at +1.5R",
            "+1R is inside daily noise; the shift may be converting winners "
            "into scratches",
            _set(breakeven_at_r=1.5)),
    Variant("be20", "Breakeven at +2R",
            "de-risk only where the partial is banked anyway",
            _set(breakeven_at_r=2.0)),
    Variant("beoff", "No breakeven shift (trail only)",
            "the upper bound of the same hypothesis: never de-risk early",
            _set(breakeven_at_r=99.0)),

    # L3 - the stop band (re-scans: stops change every R:R and every ranking)
    Variant("stopwide", "Stop band 1.5-2.5 ATR",
            "a 1 ATR daily stop under a buy-stop entry is a noise-width stop "
            "on a strength entry; wider stop, smaller size, same rupee risk",
            _set(swing_atr_stop_min_multiple=1.5, swing_atr_stop_multiple=2.5)),

    # L4 - the setup mix (re-scans: detect() reads enabled_setups)
    Variant("breakoutonly", "Breakout only",
            "breakout is 74% of trades; the other three may be diluting or "
            "may be carrying it",
            _set(enabled_setups=("breakout",))),
    Variant("pullbackonly", "Pullback only",
            "the cheapest entry of the four, and the one that needs the trend "
            "to be real rather than recent",
            _set(enabled_setups=("pullback",))),

    # L5 - the horizon constraint. Not a tuning knob: a book held for ten
    # trading days is a different PRODUCT from one that trails for two months,
    # and the two cannot be compared on expectancy alone. This also answers
    # how much of the book's return came from holds longer than ten days.
    Variant("hold10", "Hard time stop at 10 trading days",
            "the stated horizon, encoded - and a measurement of what the long "
            "tail of holds was actually contributing",
            _set(max_hold_days=10)),

    # L6 - a TIGHTER stop, tested rather than assumed. It runs three ways
    # against the book: cash per ticket is risk/stop% so a tighter stop needs
    # MORE money per position, friction in R is 0.284%/stop% so it gets worse,
    # and at 0.75x ATR the stop sits inside one day's normal range.
    # `stopwide` (1.5-2.5x) was already measured and rejected at +0.043R.
    Variant("stoptight", "Stop band 0.75-1.0 ATR",
            "the tighter stop the horizon seems to want, priced against the "
            "three things that make it expensive",
            _set(swing_atr_stop_min_multiple=0.75, swing_atr_stop_multiple=1.0)),

    # L7 - the proposed exit: no target, no partial, no breakeven rung, an ATR
    # stop trailed from entry, hard exit at 10 days.
    #
    # `trail_from_r` is what makes this expressible at all. Raising
    # `partial_exit_at_r` alone removes the target AND every trailing update,
    # because `LadderMode.TRAIL` was reachable only through the partial rung -
    # the position would sit on its initial stop and nothing would say so.
    #
    # Two of its three components were already measured alone and both were
    # WORSE than the baseline (`beoff` and `be20`, +0.058R against +0.114R),
    # and that baseline loses to a random-scoring null over six years. This is
    # pre-registered because it was asked for and is cheap, not because the
    # prior is good.
    Variant("puretrail", "ATR trail from entry, no target, 10-day cap",
            "bank whatever the move gives instead of waiting for 2R, and stop "
            "holding once the idea has had ten sessions to work",
            _set(trail_from_r=0.0, breakeven_at_r=99.0,
                 partial_exit_at_r=99.0, max_hold_days=10)),

    # The two most promising levers together, which is the only combination
    # worth pre-registering: everything else would be a search.
    Variant("regime200_be15", "Regime 200DMA + breakeven at +1.5R",
            "if both levers work they should not overlap - one filters days, "
            "the other manages trades",
            _set(regime_ma_days=200, breakeven_at_r=1.5)),
)


def variant(key: str) -> Variant:
    for v in VARIANTS:
        if v.key == key:
            return v
    raise KeyError(f"unknown variant {key!r}; have {[v.key for v in VARIANTS]}")


# ------------------------------------------------------------------- results

@dataclass
class Cell:
    """One variant, one window, one phase (train or test)."""
    variant: str
    fold: int
    phase: str
    start: date
    end: date
    trades: int
    win_rate: float
    breakeven_win_rate: float
    expectancy_r: float
    total_r: float
    max_drawdown_r: float
    capital_blocks: int
    regime_blocks: int
    #: Recorded because a run with settlement and one without are different
    #: experiments that would otherwise be identical rows, and `read_ledger`
    #: de-duplicates on (variant, fold, phase).
    settle_days: int = 60

    @property
    def judgeable(self) -> bool:
        return self.trades >= MIN_TRADES_TO_JUDGE

    def as_row(self) -> dict:
        return {
            "variant": self.variant, "fold": self.fold, "phase": self.phase,
            "start": self.start, "end": self.end, "trades": self.trades,
            "win_rate": round(self.win_rate, 4),
            "breakeven_win_rate": round(self.breakeven_win_rate, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "total_r": round(self.total_r, 3),
            "max_drawdown_r": round(self.max_drawdown_r, 3),
            "capital_blocks": self.capital_blocks,
            "regime_blocks": self.regime_blocks,
            "judgeable": self.judgeable,
            "settle_days": self.settle_days,
        }


def _cell(v: Variant, window: bt.Window, phase: str,
          result: bt.SwingBacktestResult, settle_days: int = 60) -> Cell:
    m = result.metrics
    start, end = ((window.train_start, window.train_end) if phase == "train"
                  else (window.test_start, window.test_end))
    return Cell(
        variant=v.key, fold=window.index, phase=phase, start=start, end=end,
        trades=m.trades, win_rate=m.win_rate,
        breakeven_win_rate=m.breakeven_win_rate,
        expectancy_r=m.expectancy_r, total_r=m.total_r,
        max_drawdown_r=m.max_drawdown_r,
        capital_blocks=result.day_stats.capital_blocks,
        regime_blocks=result.day_stats.regime_blocks,
        settle_days=settle_days,
    )


@dataclass
class Sweep:
    cells: list[Cell] = field(default_factory=list)
    windows: list[bt.Window] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_row() for c in self.cells])

    def phase(self, phase: str, key: str) -> list[Cell]:
        return [c for c in self.cells if c.phase == phase and c.variant == key]

    def pooled(self, phase: str, key: str) -> tuple[int, float]:
        """Trade-weighted expectancy across folds, and the trade count."""
        cells = self.phase(phase, key)
        trades = sum(c.trades for c in cells)
        if not trades:
            return 0, 0.0
        return trades, sum(c.total_r for c in cells) / trades

    def selected_by_train(self) -> list[tuple[bt.Window, Optional[str], Cell | None]]:
        """
        What picking on train and trading the pick would actually have paid.

        This is the only line in the whole sweep that a live book could have
        acted on: at each fold, the variant with the best TRAIN expectancy
        among those with enough trades to judge, scored on the test window it
        had not seen.
        """
        out = []
        for w in self.windows:
            trained = [c for c in self.cells
                       if c.phase == "train" and c.fold == w.index
                       and c.judgeable]
            if not trained:
                out.append((w, None, None))
                continue
            best = max(trained, key=lambda c: c.expectancy_r)
            test = next((c for c in self.cells
                         if c.phase == "test" and c.fold == w.index
                         and c.variant == best.variant), None)
            out.append((w, best.variant, test))
        return out


# --------------------------------------------------------------------- sweep

def sweep(cfg: Config, market, bars: dict, benchmark=None, stocks=None,
          variants: tuple[Variant, ...] = VARIANTS,
          train_months: Optional[int] = None,
          test_months: Optional[int] = None,
          window_start: Optional[date] = None,
          settle_days: int = 60,
          use_cache: bool = True,
          progress: Optional[Callable] = None) -> Sweep:
    """
    Run every variant across every walk-forward window, train and test.

    Variants sharing a scan signature share one `ScanCache`, so the levers
    that do not change the signals cost one pass between them. The cache
    raises rather than falls back on a signature mismatch, so a variant that
    unexpectedly changes the scan is a loud failure and not a quiet one.
    """
    out = Sweep()
    # Computed ONCE for the whole sweep. `run()` would otherwise union a
    # hundred DatetimeIndexes on every one of several hundred calls, which is
    # the same answer every time and a large share of the wall clock.
    sessions_index = bt.all_sessions(bars)
    sessions = list(sessions_index)
    if len(sessions) < 2:
        out.warnings.append("fewer than two sessions of data - nothing to sweep")
        return out

    train_months = train_months or cfg.backtest.train_months
    test_months = test_months or cfg.backtest.test_months
    # Bars before `window_start` stay loaded as warm-up; folds never reach
    # back into them, so no session is scored on half-converged indicators.
    first = sessions[0].date()
    if window_start is not None and window_start > first:
        first = window_start
    out.windows = bt.fold_windows(first, sessions[-1].date(),
                                  train_months, test_months)
    if not out.windows:
        out.warnings.append(
            f"{len(sessions)} sessions is not enough for a "
            f"{train_months}/{test_months}-month walk-forward")
        return out

    out.warnings.append(
        "Each window starts flat: a position open when a window ends is "
        "excluded, so long holds are under-counted at every boundary.")
    if benchmark is None and any(
            v.configure(cfg).swing.regime_ma_days > 0 for v in variants):
        out.warnings.append(
            "NO BENCHMARK was supplied and regime variants are in this sweep. "
            "The filter fails closed, so those rows will show no trades - "
            "that is a missing input, not a finding about the filter.")

    caches: dict[str, bt.ScanCache] = {}
    total = len(variants) * len(out.windows) * 2
    done = 0
    for v in variants:
        variant_cfg = v.configure(cfg)
        signature = bt.scan_signature(variant_cfg, market)
        if use_cache:
            cache = caches.setdefault(signature, bt.ScanCache(signature=signature))
        else:
            cache = None

        for w in out.windows:
            for phase, (lo, hi) in (("train", (w.train_start, w.train_end)),
                                    ("test", (w.test_start, w.test_end))):
                if progress:
                    progress(done, total, f"{v.key} {phase} f{w.index}")
                result = bt.run(variant_cfg, market, bars, benchmark,
                                stocks=stocks, start=lo, end=hi,
                                scan_cache=cache,
                                sessions_index=sessions_index,
                                settle_days=settle_days)
                out.cells.append(_cell(v, w, phase, result, settle_days))
                done += 1
    return out


# --------------------------------------------------- the null, walk-forward

def walk_forward_null(cfg: Config, market, bars: dict, benchmark=None,
                      stocks=None, n_seeds: int = 40, seed0: int = 20260904,
                      train_months: Optional[int] = None,
                      test_months: Optional[int] = None,
                      window_start: Optional[date] = None,
                      settle_days: int = 60,
                      variant: Optional[Variant] = None,
                      progress: Optional[Callable] = None):
    """
    S1. Rank the book inside a distribution of random-scoring nulls, per
    out-of-sample window.

    THE QUESTION THIS BOOK HAS NEVER BEEN ASKED. All twelve variants in
    `VARIANTS` compare against `baseline`; none asks whether the SCORING beats
    chance. On the intraday equity book the equivalent null BEAT the four
    strategies on four metrics, and nobody has checked whether the same is true
    on daily bars.

    Only TEST windows are run. Train windows exist to choose between variants,
    and nothing is being chosen here - the question is whether one fixed rule
    beats noise consistently.

    ONE SCAN PASS SERVES EVERYTHING. `random_seed` is in
    `BOOK_ONLY_SWING_FIELDS`, so the book and all `n_seeds` nulls hash to the
    same scan signature and share a single `ScanCache`. That is what makes a
    40-draw null affordable at all.
    """
    from .. import experiment_core as ec

    sessions_index = bt.all_sessions(bars)
    sessions = list(sessions_index)
    if len(sessions) < 2:
        return ec.WalkForward(folds=[], slippage_pct=0.0)

    train_months = train_months or cfg.backtest.train_months
    test_months = test_months or cfg.backtest.test_months
    first = sessions[0].date()
    if window_start is not None and window_start > first:
        first = window_start
    windows = bt.fold_windows(first, sessions[-1].date(),
                              train_months, test_months)

    base_cfg = (variant or VARIANTS[0]).configure(cfg)
    signature = bt.scan_signature(base_cfg, market)
    cache = bt.ScanCache(signature=signature)

    def expectancy(run_cfg, w) -> tuple:
        res = bt.run(run_cfg, market, bars, benchmark, stocks=stocks,
                     start=w.test_start, end=w.test_end, scan_cache=cache,
                     sessions_index=sessions_index, settle_days=settle_days)
        m = res.metrics
        return ((m.expectancy_r if m else 0.0), res.trades)

    folds = []
    for i, w in enumerate(windows):
        if progress:
            progress(i + 1, len(windows), f"{w.test_start}..{w.test_end}")
        real, trades = expectancy(base_cfg, w)
        nulls = []
        for j in range(n_seeds):
            c = copy.deepcopy(base_cfg)
            c.swing.random_seed = seed0 + i * 1000 + j
            nulls.append(expectancy(c, w)[0])
        # `trades` rides along because `bars_held` is the number the horizon
        # question turns on, and because a window too thin to mean anything
        # must be excluded rather than averaged in.
        folds.append(ec.Fold(index=i, start=w.test_start, end=w.test_end,
                             score=real, nulls=nulls, trades=trades))
    return ec.WalkForward(folds=folds)


def hold_stats(wf) -> str:
    """
    How long the book actually held, and what each exit reason earned.

    Asked for explicitly and computed nowhere else: "average days to target"
    cannot be read off an expectancy table, and a book pitched at a ten-day
    horizon that in fact trails for six weeks is a different product from the
    one described.
    """
    trades = [t for f in getattr(wf, "folds", []) for t in getattr(f, "trades", [])]
    if not trades:
        return "  no trades - nothing to describe"
    lines = [f"  {'exit':<12}{'n':>6}{'share':>8}{'mean R':>9}"
             f"{'mean days':>11}{'median days':>13}"]
    import statistics as _st
    by: dict = {}
    for t in trades:
        by.setdefault(t.outcome, []).append(t)
    for reason, group in sorted(by.items(), key=lambda kv: -len(kv[1])):
        held = [t.bars_held for t in group]
        lines.append(
            f"  {reason:<12}{len(group):>6}{len(group)/len(trades):>8.0%}"
            f"{sum(t.r_multiple for t in group)/len(group):>9.3f}"
            f"{sum(held)/len(held):>11.1f}{_st.median(held):>13.0f}")
    allheld = [t.bars_held for t in trades]
    lines.append(f"  {'ALL':<12}{len(trades):>6}{1.0:>8.0%}"
                 f"{sum(t.r_multiple for t in trades)/len(trades):>9.3f}"
                 f"{sum(allheld)/len(allheld):>11.1f}"
                 f"{_st.median(allheld):>13.0f}")
    return "\n".join(lines)


# ------------------------------------------------------------------- ledger

def read_ledger(directory: str = "data/experiments") -> pd.DataFrame:
    """
    Every sweep ever run, as one frame.

    Signature groups are run as separate parallel invocations - seven variants
    share a scan pass, each signal-changing variant needs its own - so the
    results of one experiment arrive in several files and have to be read back
    together or half of it is invisible.

    Re-runs of the same (variant, fold, phase) keep the NEWEST file, because
    the reason to re-run a cell is that the earlier one was wrong. Filenames
    carry a second-resolution timestamp, so sorting them sorts by age.
    """
    from pathlib import Path
    paths = sorted(Path(directory).glob("sweep_*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path).assign(source=path.name) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    # `settle_days` is part of the identity of a cell, not a detail of it: a
    # run with the settlement tail and one without are different experiments
    # over the same variant and fold. Leaving it out of the key made the
    # deliberate bias-measurement run silently overwrite the good rows.
    key = ["variant", "fold", "phase"]
    if "settle_days" in frame.columns:
        frame["settle_days"] = frame["settle_days"].fillna(60)
        key.append("settle_days")
    return frame.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def report(frame: pd.DataFrame) -> None:
    """Print the pooled table and the walk-forward selection from a ledger."""
    if frame.empty:
        print("No sweep results found. Run the sweep first.")
        return

    # Only the correctly-settled rows are reported. Rows from a
    # `--settle-days 0` run exist to measure the boundary bias, and averaging
    # them together with the real ones would bury it instead.
    if "settle_days" in frame.columns and frame["settle_days"].nunique() > 1:
        best = frame["settle_days"].max()
        others = sorted(set(frame["settle_days"].unique()) - {best})
        print(f"  ledger also holds rows at settle_days={others} - reporting "
              f"settle_days={int(best)} only; use the ledger directly to "
              f"compare them")
        frame = frame[frame["settle_days"] == best]

    folds = sorted(frame["fold"].unique())
    # `source` exists only on a frame read back from parquet; a live sweep
    # hands its own cells straight in.
    files = frame["source"].nunique() if "source" in frame.columns else 1
    print(f"  {len(folds)} folds · {frame['variant'].nunique()} variants · "
          f"{files} ledger file(s)")
    print()
    head = (f"  {'variant':<16} {'train R':>8} {'n':>5}   "
            f"{'TEST R':>8} {'n':>5}   {'gap':>7}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    for key in [v.key for v in VARIANTS if v.key in set(frame["variant"])]:
        rows = frame[frame["variant"] == key]
        out = {}
        for phase in ("train", "test"):
            part = rows[rows["phase"] == phase]
            n = int(part["trades"].sum())
            out[phase] = (n, float(part["total_r"].sum()) / n if n else 0.0)
        (tr_n, tr_e), (te_n, te_e) = out["train"], out["test"]
        flag = "" if te_n >= MIN_TRADES_TO_JUDGE else "  (thin)"
        print(f"  {key:<16} {tr_e:>+8.3f} {tr_n:>5}   "
              f"{te_e:>+8.3f} {te_n:>5}   {tr_e - te_e:>+7.3f}{flag}")

    print()
    print("  Picking on train at each fold, scored on the test window:")
    total_r, total_n = 0.0, 0
    for fold in folds:
        trained = frame[(frame["fold"] == fold) & (frame["phase"] == "train")
                        & frame["judgeable"]]
        if trained.empty:
            print(f"    fold {fold:<2}  no variant had enough train trades")
            continue
        best = trained.loc[trained["expectancy_r"].idxmax()]
        test = frame[(frame["fold"] == fold) & (frame["phase"] == "test")
                     & (frame["variant"] == best["variant"])]
        if test.empty:
            continue
        row = test.iloc[0]
        total_r += float(row["total_r"])
        total_n += int(row["trades"])
        print(f"    fold {fold:<2}  chose {best['variant']:<16} -> "
              f"{row['expectancy_r']:+.3f}R over {int(row['trades']):>3} trades")
    if total_n:
        print()
        print(f"    Walk-forward expectancy: {total_r / total_n:+.4f}R "
              f"over {total_n} trades")
        print("    Compare against the baseline's TEST column above - not its "
              "train column, and not the best row in the table.")


# ----------------------------------------------------------------------- CLI

def _print(out: Sweep, cfg: Config) -> None:
    """
    The caveats, then the same table `--report` prints from the ledger.

    One implementation of the arithmetic, not two: a live run and a re-read of
    its own parquet must agree, and the only way to be sure of that is for
    them to share the code that computes it.
    """
    print("=" * 78)
    print("READ THIS FIRST")
    print("=" * 78)
    for c in bt.CAVEATS:
        print(f"  * {c}")
    for w in out.warnings:
        print(f"  * {w}")
    print(f"  * The TEST column is the result. TRAIN is shown so you can see "
          f"how much of a variant was fitting.")
    print(f"  * A cell under {MIN_TRADES_TO_JUDGE} trades is printed but never "
          f"used to choose.")
    print(f"  * {cfg.backtest.train_months}m train / "
          f"{cfg.backtest.test_months}m test")
    print()

    if not out.cells:
        print("Nothing ran.")
        return
    report(out.frame())


def _main() -> None:                                       # pragma: no cover
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Walk-forward sweep of the swing book's rule variants.")
    parser.add_argument("--market", default=cfg.swing.default_market,
                        choices=markets_mod.keys(cfg))
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--capital", type=float, default=None,
                        help="swing pot in rupees; defaults to config")
    parser.add_argument("--variants", default="",
                        help="comma-separated subset; default is all")
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--out", default="data/experiments",
                        help="directory for the parquet ledger")
    parser.add_argument("--no-cache", action="store_true",
                        help="re-scan for every variant (slow; proves the cache)")
    parser.add_argument("--settle-days", type=int, default=60,
                        help="sessions after a window closes in which its "
                             "open trades may still finish; 0 reproduces the "
                             "winner-deleting boundary bias, for measuring it")
    parser.add_argument("--report", action="store_true",
                        help="read the ledger and print it; run nothing")
    args = parser.parse_args()

    if args.report:
        report(read_ledger(args.out))
        return

    if args.capital:
        cfg.capital.swing_capital_inr = args.capital
    market = markets_mod.get(cfg, args.market)

    chosen = VARIANTS
    if args.variants:
        chosen = tuple(variant(k.strip()) for k in args.variants.split(","))

    stocks = load_universe(market.universe_csv)
    tickers = {s.symbol: s.yf_ticker for s in stocks}
    cfg.swing.history_days = max(int(args.years * 365) + 260, 400)
    print(f"Loading {len(tickers)} symbols, ~{args.years:g} years ...")
    price_set = prices_mod.load_prices(tickers, cfg, market)

    # `--years` sizes the TEST WINDOW, not just the fetch. The cache is
    # allowed to hold more history than was asked for - `fetch_swing_history`
    # pulls six years - and for a long time it silently did: a run labelled
    # "3 years" tested every session in the parquet, which was 5.4. The extra
    # bars are still loaded and still used, but only as WARM-UP: indicators
    # need history before the first scored session, and starting the window
    # at the first cached bar would score sessions whose EMAs were half
    # converged.
    window_start = date.today() - timedelta(days=int(args.years * 365))

    out = sweep(cfg, market, price_set.bars, price_set.benchmark,
                stocks=stocks, variants=chosen, window_start=window_start,
                settle_days=args.settle_days,
                train_months=args.train_months, test_months=args.test_months,
                use_cache=not args.no_cache,
                # flush: without it Python buffers a redirected stdout
                # and a 90-minute sweep shows nothing until it ends.
                progress=lambda d, t, l: print(f"  [{d}/{t}] {l:<28}",
                                               end="\r", flush=True))
    print(" " * 70, end="\r")
    _print(out, cfg)

    frame = out.frame()
    if not frame.empty:
        from datetime import datetime
        from pathlib import Path
        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        # Second-resolution, because signature groups are run as separate
        # parallel invocations - a date-only name meant four concurrent
        # sweeps silently overwrote one another's ledger, leaving whichever
        # finished last looking like the whole experiment.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        names = "-".join(v.key for v in chosen)[:40]
        path = directory / f"sweep_{market.key}_{stamp}_{names}.parquet"
        frame.to_parquet(path)
        print(f"\n  {len(frame)} rows -> {path}")


if __name__ == "__main__":                                 # pragma: no cover
    _main()
