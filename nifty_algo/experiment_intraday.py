"""
Which rule set for the intraday option book, chosen out of sample.

    python -u -m nifty_algo.experiment_intraday --variants gapdaily,novwap --years 4

WHY THIS EXISTS. Four years of real 5-minute NIFTY bars (993 sessions) say the
book has no measurable edge: `level_break` is +0.006R over 1,364 trades with a
95% interval of [-0.064, +0.075], which is tight enough to rule out any edge
above ~0.075R. That is a proven negative, not an unproven positive - and it is
measured in `Mode.UNDERLYING`, which ignores theta and vega, so it is the
GENEROUS case. A strategy at zero on the underlying is negative once premium
decay is paid.

Answering that took two throwaway scripts and a killed 17-minute job. The
levers below are the ones worth testing next, and this module exists so that
testing them costs one command instead of a script.

EVERY VARIANT IS PRE-REGISTERED WITH ITS HYPOTHESIS, and the hypothesis is
written before the number exists. That is not ceremony: the alternative is
running ten configurations, keeping the best, and reporting a fitted result as
a discovery. `report()` prints the TEST column and the fold-level sign test
precisely so that a variant which wins on pooled R but not on folds is visible
as what it is.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from . import experiment_core as core
from .backtest import Backtester, Mode
from .config import Config, DEFAULT
from .data.csv_feed import CsvFeed
from .experiment_core import Variant, set_on

LEDGER_PREFIX = "sweep_intraday"


def _drop_strategy(key: str):
    """Disable one registered strategy, leaving the rest at their defaults."""
    def apply(cfg) -> None:
        from .strategies.registry import default_enabled_keys
        keep = [k for k in default_enabled_keys() if k != key]
        cfg.backtest.strategy_keys = tuple(keep)
    return apply


VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="baseline",
        label="as shipped",
        why="Every number this project has published was produced here. It is "
            "the control, and it is what a variant has to beat.",
        apply=lambda cfg: None,
    ),
    Variant(
        key="gapdaily",
        label="gap measured against a DAILY range",
        why="THE LARGEST UNTESTED LEVER. 89% of all trades over four years are "
            "tagged gap_day, and warm-up does not move it. `regime.py:62,67-68` "
            "divides the overnight gap by the 14-period FIVE-MINUTE ATR, so "
            "`gap_atr_multiple = 0.5` fires on ~8 index points - 0.03% of "
            "NIFTY. The filter `regime.py` calls 'the highest-leverage filter "
            "in this whole system' is therefore inert: on a gap day it removes "
            "two strategies out of ten. Raising the multiple is the crude "
            "proxy for the right denominator, and it is testable today.",
        apply=set_on("regime", gap_atr_multiple=12.0),
    ),
    Variant(
        key="novwap",
        label="drop vwap_reclaim",
        why="-30.4R over 315 trades, the single largest loss in the book, and "
            "mechanically explained rather than merely observed: the NIFTY "
            "index reports zero volume, so `signals.vwap` degrades to a "
            "session TWAP (`has_traded_volume` -> equal weights). A strategy "
            "whose entire premise is a VWAP cross with volume confirmation is "
            "trading neither. Its docstring promises something the data cannot "
            "supply.",
        apply=_drop_strategy("vwap_reclaim"),
    ),
    Variant(
        key="warm1",
        label="one warm-up session",
        why="Already built and already measured: it opens 09:30-11:44, which "
            "the book was structurally blind to, and it won 10 of 21 folds "
            "against a coin-flip expectation of 10.5. Carried so the harness "
            "REPRODUCES A KNOWN RESULT. If the sweep disagrees with that "
            "number, the harness is wrong, not the earlier finding.",
        apply=set_on("session", warmup_sessions=1),
    ),
    Variant(
        key="ladder1r",
        label="full exit at +1R, no runner",
        why="The 22-trade sample that started this said 45% of trades reached "
            "+1R and round-tripped to breakeven, converting only 29%, which "
            "made 'just take +1R' look worth +0.3R. On 1,272 trades the "
            "scratch rate is 23% and conversion is 53%. Registered so it is "
            "killed by measurement rather than by argument.",
        apply=set_on("trade", enable_runner=False, partial_exit_at_r=1.0),
    ),
    Variant(
        key="noregime",
        label="regime gate off",
        why="A FALSIFIABLE CHECK ON THE DIAGNOSIS, not a candidate rule. If "
            "89% gap_day really means the gate is inert, switching it off "
            "should change almost nothing. A large move here would mean the "
            "gate matters after all and the inertness reading is wrong.",
        apply=set_on("regime", enforce_regime_gate=False),
    ),
)


def variant(key: str) -> Variant:
    for v in VARIANTS:
        if v.key == key:
            return v
    raise KeyError(f"unknown variant {key!r} - registered: "
                   f"{', '.join(v.key for v in VARIANTS)}")


def sweep(cfg: Config, bars: pd.DataFrame,
          variants: tuple[Variant, ...] = VARIANTS,
          train_months: int | None = None,
          test_months: int | None = None,
          window_start: date | None = None,
          mode: Mode = Mode.UNDERLYING,
          progress=None) -> core.Sweep:
    """
    Every variant over every fold, TRAIN and TEST.

    Both phases are run explicitly through `Backtester.run(start=, end=)`.
    `run()`'s own fold machinery executes test windows only - the train dates
    are carried as labels - so without this there is no train column and
    `selected_by_train` cannot be computed at all.
    """
    from .swing.backtest import fold_windows

    sessions = sorted({d for d in bars.index.date})
    if window_start:
        sessions = [d for d in sessions if d >= window_start]
    if not sessions:
        return core.Sweep()

    tm = train_months or cfg.backtest.train_months
    sm = test_months or cfg.backtest.test_months
    windows = fold_windows(sessions[0], sessions[-1], tm, sm)

    out = core.Sweep()
    for v in variants:
        vcfg = v.configure(cfg)
        keys = list(getattr(vcfg.backtest, "strategy_keys", ()) or []) or None
        for w in windows:
            for phase, lo, hi in (("train", w.train_start, w.train_end),
                                  ("test", w.test_start, w.test_end)):
                if progress:
                    progress(v.key, w.index, phase, len(windows))
                bt = Backtester(vcfg)
                res = bt.run(bars, keys, mode, start=lo, end=hi)
                # Read off the Backtester, not the result: `_skipped_days` is
                # engine state. `getattr(res, ...)` returned [] every time,
                # which is exactly the confident zero this repo keeps refusing
                # to print elsewhere.
                out.cells.append(core.cell_from_metrics(
                    v.key, w.index, phase, lo, hi, res.metrics,
                    skipped_days=len(bt._skipped_days),
                ))
    return out


def load_bars(path: str) -> pd.DataFrame:
    """
    Every bar in the file. Trimming happens at the WINDOW, never here.

    `--years` names the span that gets scored; the older bars stay loaded so
    the first scored session starts on converged indicators and with a real
    prior session. Trimming the frame would fix the label and quietly measure
    the first sessions on half-warm EMAs - the mistake `CLAUDE.md` records the
    swing CLIs having made.
    """
    return CsvFeed(path).get_bars(lookback_days=0)


def _main() -> None:
    p = argparse.ArgumentParser(
        description="Variant sweep for the intraday NIFTY option book.")
    p.add_argument("--bars", default=DEFAULT.data.csv_path,
                   help="5-minute CSV (scripts/fetch_history.py writes it)")
    p.add_argument("--years", type=float, default=None,
                   help="TEST WINDOW length, not the fetch - older bars stay "
                        "loaded as warm-up")
    p.add_argument("--variants", default="",
                   help="comma-separated keys; default is all registered")
    p.add_argument("--train-months", type=int, default=None)
    p.add_argument("--test-months", type=int, default=None)
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--out", default=str(core.DEFAULT_LEDGER_DIR))
    p.add_argument("--report", action="store_true",
                   help="print the table from the existing ledger and exit")
    args = p.parse_args()

    if args.report:
        frame = core.read_ledger(args.out, LEDGER_PREFIX)
        if frame.empty:
            print("no ledger found - run a sweep first")
            return
        sw = core.Sweep()
        fields = ("variant", "fold", "phase", "start", "end", "trades",
                  "expectancy_r", "total_r", "win_rate",
                  "breakeven_win_rate", "max_drawdown_r")
        sw.cells = [core.Cell(**{k: r[k] for k in fields if k in r})
                    for _, r in frame.iterrows()]
        print(core.report(sw))
        return

    cfg = Config()
    if args.capital:
        cfg.capital.starting_capital = args.capital

    bars = load_bars(args.bars)
    days = sorted({d for d in bars.index.date})
    window_start = None
    if args.years:
        window_start = (pd.Timestamp(days[-1])
                        - pd.Timedelta(days=int(args.years * 365))).date()

    chosen = (tuple(variant(k.strip()) for k in args.variants.split(",") if k.strip())
              if args.variants else VARIANTS)

    print(f"{len(bars):,} bars | {len(days)} sessions | {days[0]} -> {days[-1]}")
    print(f"variants: {', '.join(v.key for v in chosen)}")
    for v in chosen:
        print(f"  {v.key:<12} {v.label}")
    print(flush=True)

    def progress(key, fold, phase, total):
        print(f"  [{key:<12}] fold {fold + 1}/{total} {phase:<5}",
              end="\r", flush=True)

    sw = sweep(cfg, bars, chosen, args.train_months, args.test_months,
               window_start, progress=progress)
    print(" " * 78, end="\r")

    path = core.write_ledger(sw, args.out, LEDGER_PREFIX,
                             "-".join(v.key for v in chosen))
    print(core.report(sw))
    if path:
        print(f"\nledger -> {path}")


if __name__ == "__main__":
    _main()
