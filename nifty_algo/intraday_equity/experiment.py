"""
Which rule set, chosen out-of-sample.

`backtest.run()` is one in-sample pass and a description of the past.
`fold_windows()` + `sweep()` are how a choice actually gets made: each variant
is measured on a TRAIN window and scored on the TEST window that follows it,
because picking the best of eight variants on the data that measured them is
fitting rather than evidence.

THREE RULES, CARRIED OVER FROM THE SWING SWEEP BECAUSE THEY WERE LEARNED THE
EXPENSIVE WAY

  1. VARIANTS ARE PRE-REGISTERED. Every entry in `VARIANTS` below carries a
     `why` written before the number existed. Adding a variant after seeing a
     result, to explain that result, is how a sweep becomes a story.

  2. NOTHING AUTO-TUNES. This prints a table and stops. `selected_by_train`
     reports what walk-forward selection WOULD have paid, which on the swing
     book was worse than simply leaving the baseline alone - 6-month train
     expectancy had no predictive power over the following 2 months. That is
     a finding worth reproducing here rather than assuming away.

  3. READ THE TEST COLUMN. The train-test gap is how much of a variant was
     fitting.

WHAT IS DIFFERENT HERE: NO SETTLEMENT

The swing sweep must carry `settle_days` in a cell's IDENTITY, because a
window that ends while positions are open deletes winners preferentially - a
measured 0.176R bias. Every position in this book closes by 15:10, so a fold
boundary is a session boundary and there is nothing to settle. The parameter
is absent rather than passed as zero, so nobody re-derives it.

BUDGET THE SWEEP BY SIGNATURE GROUP, NOT BY VARIANT COUNT

Variants that only change the ladder, the capital rules or the consumer-side
filters share one `signal_signature`, so they share ONE scan pass. Only the
variants that change what a signal IS - the stop multiple, the strategy set,
the entry window - pay for another. Seven cache-sharing variants cost about
what one costs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..backtest import compute_metrics
from ..config import Config
from ..swing.backtest import Window, fold_windows
from . import backtest as bt

#: Below this many trades a cell is not judged. A 4-trade expectancy is a
#: number, not a measurement.
MIN_TRADES_TO_JUDGE = 20


def _clone(cfg: Config) -> Config:
    """
    A DEEP copy. `dataclasses.replace` is shallow, so two variants would end
    up mutating one shared `IntradayEquityConfig` and the whole sweep would
    silently measure the last variant applied, several times.
    """
    return copy.deepcopy(cfg)


def _set(**fields) -> Callable[[Config], None]:
    def apply(cfg: Config) -> None:
        for name, value in fields.items():
            setattr(cfg.intraday_equity, name, value)
    return apply


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    why: str
    apply: Callable[[Config], None]

    def configure(self, base: Config) -> Config:
        cfg = _clone(base)
        self.apply(cfg)
        return cfg


#: PRE-REGISTERED. Every `why` was written before any expectancy existed.
VARIANTS: tuple[Variant, ...] = (
    Variant("baseline", "baseline", "the shipped configuration", lambda c: None),

    # --- L1: the stop multiple, which the friction arithmetic says is the
    #     most consequential number in the book. Both directions, because the
    #     tension is genuine: friction wants a wide stop, the session's finite
    #     range wants a narrow one.
    Variant("stoptight", "stop 1.0x ATR",
            "cheaper target (needs 25% of the daily range) but friction eats "
            "0.556R and it needs 6.5x leverage to be risk-bound",
            _set(atr_stop_multiple=1.0)),
    Variant("stopwide", "stop 3.0x ATR",
            "friction falls to 0.204R, but the +2R partial then needs 76% of "
            "the median stock's entire daily range",
            _set(atr_stop_multiple=3.0)),

    # --- L2: the ladder. Shares a scan signature with the baseline.
    Variant("trailonly", "no +2R partial",
            "the partial pays a third leg of brokerage; does banking half at "
            "+2R beat simply trailing the whole position",
            _set(partial_exit_fraction=0.0)),
    Variant("be15", "breakeven at +1.5R",
            "a later breakeven shift gives the trade more room before it is "
            "reduced to a scratch",
            _set(breakeven_at_r=1.5)),

    # --- L3: the cross-sectional layer. `norank` is the honest test of the
    #     prefilter, which is otherwise assumed rather than measured.
    Variant("norank", "no RS prefilter",
            "the prefilter is a 5x saving in both backtest and live cost; "
            "this measures what it costs in expectancy",
            _set(rs_prefilter_n=0)),

    # --- L4: the warm-up window (finding 3c). Changes the signals.
    Variant("conf40", "min confidence 0.40",
            "the registry's own min_confidence is 0.25; does demanding more "
            "conviction pay for the trades it forgoes",
            _set(min_confidence=0.40)),
    Variant("regimeoff", "no regime gate",
            "the intraday regime gate was calibrated for option buying, "
            "where a range day bleeds theta; cash equity has no theta",
            _set(enforce_regime_gate=False)),
)


def variant(key: str) -> Variant:
    for v in VARIANTS:
        if v.key == key:
            return v
    raise KeyError(f"unknown variant {key!r} - registered: "
                   f"{', '.join(v.key for v in VARIANTS)}")


@dataclass
class Cell:
    """One (variant, fold, phase) result."""
    variant: str
    fold: int
    phase: str                     # "train" | "test"
    start: date
    end: date
    trades: int
    expectancy_r: float
    total_r: float
    win_rate: float
    breakeven_win_rate: float
    friction_r: float

    @property
    def judgeable(self) -> bool:
        return self.trades >= MIN_TRADES_TO_JUDGE


@dataclass
class Sweep:
    cells: list = field(default_factory=list)
    scan_passes: int = 0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(c) for c in self.cells])

    def phase(self, phase: str) -> pd.DataFrame:
        df = self.frame()
        return df[df["phase"] == phase] if not df.empty else df

    def pooled(self) -> pd.DataFrame:
        """
        Trade-weighted expectancy per variant per phase.

        Weighted by trade count rather than averaged across folds: a fold with
        3 trades and one with 60 are not equally informative, and a plain mean
        would let the thin fold swing the answer.
        """
        df = self.frame()
        if df.empty:
            return df
        rows = []
        for (v, ph), g in df.groupby(["variant", "phase"]):
            n = int(g["trades"].sum())
            r = float((g["expectancy_r"] * g["trades"]).sum())
            rows.append({"variant": v, "phase": ph, "trades": n,
                         "expectancy_r": r / n if n else 0.0,
                         "total_r": float(g["total_r"].sum())})
        return pd.DataFrame(rows)

    def bootstrap_ci(self, key: str, phase: str = "test",
                     draws: int = 2000, seed: int = 7) -> tuple:
        """
        95% interval on a variant's out-of-sample expectancy, by resampling
        FOLDS rather than trades.

        Folds, because trades within a fold are not independent - they share a
        regime, a volatility level and often a session. Resampling trades
        would produce a reassuringly narrow interval that means nothing. This
        is the number that decides whether a result clears zero.
        """
        df = self.frame()
        if df.empty:
            return (float("nan"), float("nan"))
        g = df[(df["variant"] == key) & (df["phase"] == phase)]
        g = g[g["trades"] > 0]
        if len(g) < 3:
            return (float("nan"), float("nan"))
        exp = g["expectancy_r"].to_numpy()
        wts = g["trades"].to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(draws):
            idx = rng.integers(0, len(exp), len(exp))
            w = wts[idx]
            out.append(float((exp[idx] * w).sum() / w.sum()) if w.sum() else 0.0)
        return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

    def selected_by_train(self) -> pd.DataFrame:
        """
        What walk-forward SELECTION would have paid.

        At each fold, take the variant with the best train expectancy among
        judgeable cells and score it on that fold's untouched test window.
        On the swing book this paid WORSE than the baseline, which is itself
        the finding - it says train expectancy had no predictive power.
        """
        df = self.frame()
        if df.empty:
            return df
        rows = []
        for fold, g in df[df["phase"] == "train"].groupby("fold"):
            ok = g[g["trades"] >= MIN_TRADES_TO_JUDGE]
            if ok.empty:
                continue
            pick = ok.sort_values("expectancy_r", ascending=False).iloc[0]
            test = df[(df["fold"] == fold) & (df["phase"] == "test")
                      & (df["variant"] == pick["variant"])]
            if test.empty:
                continue
            t = test.iloc[0]
            rows.append({"fold": fold, "picked": pick["variant"],
                         "train_r": pick["expectancy_r"],
                         "test_r": t["expectancy_r"], "test_trades": t["trades"]})
        return pd.DataFrame(rows)


def sweep(cfg: Config, bars: dict, benchmark=None, variants=VARIANTS,
          train_months: int | None = None, test_months: int | None = None,
          window_start: date | None = None, universe_key: str = "",
          use_cache: bool = True, progress=None) -> Sweep:
    """
    Every variant over every fold, train and test.

    Variants sharing a `signal_signature` share ONE scan pass - which is the
    whole reason the cache exists, and why a ladder sweep is nearly free while
    a stop-multiple sweep is not.
    """
    sessions = sorted({ts.date() for df in bars.values() for ts in df.index})
    if window_start:
        sessions = [d for d in sessions if d >= window_start]
    if not sessions:
        return Sweep()

    tm = train_months or cfg.backtest.train_months
    sm = test_months or cfg.backtest.test_months
    windows = fold_windows(sessions[0], sessions[-1], tm, sm)

    out = Sweep()
    caches: dict = {}
    for v in variants:
        vcfg = v.configure(cfg)
        sig = bt.signal_signature(vcfg, universe_key)
        if use_cache and sig not in caches:
            caches[sig] = bt.SignalCache(sig)
            out.scan_passes += 1
        cache = caches.get(sig) if use_cache else None

        for w in windows:
            for phase, lo, hi in (("train", w.train_start, w.train_end),
                                  ("test", w.test_start, w.test_end)):
                if progress:
                    progress(v.key, w.index, phase)
                res = bt.run(vcfg, bars, benchmark=benchmark, start=lo, end=hi,
                             cache=cache, universe_key=universe_key)
                m = res.metrics
                out.cells.append(Cell(
                    variant=v.key, fold=w.index, phase=phase, start=lo, end=hi,
                    trades=m.trades if m else 0,
                    expectancy_r=m.expectancy_r if m else 0.0,
                    total_r=m.total_r if m else 0.0,
                    win_rate=m.win_rate if m else 0.0,
                    breakeven_win_rate=m.breakeven_win_rate if m else 0.0,
                    friction_r=res.friction_r))
    return out


def report(sw: Sweep) -> str:
    """The table, and nothing that decides anything."""
    lines = []
    pooled = sw.pooled()
    if pooled.empty:
        return "no cells - not enough history for even one fold"

    lines.append(f"{'variant':<14} {'train R':>9} {'n':>6} {'TEST R':>9} "
                 f"{'n':>6} {'gap':>8} {'95% CI (test)':>22}")
    tr = pooled[pooled["phase"] == "train"].set_index("variant")
    te = pooled[pooled["phase"] == "test"].set_index("variant")
    for key in te.sort_values("expectancy_r", ascending=False).index:
        t = tr.loc[key] if key in tr.index else None
        e = te.loc[key]
        gap = (t["expectancy_r"] - e["expectancy_r"]) if t is not None else float("nan")
        lo, hi = sw.bootstrap_ci(key)
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo == lo else "n/a"
        lines.append(
            f"{key:<14} {t['expectancy_r'] if t is not None else 0:+9.3f} "
            f"{int(t['trades']) if t is not None else 0:6d} "
            f"{e['expectancy_r']:+9.3f} {int(e['trades']):6d} "
            f"{gap:+8.3f} {ci:>22}")

    sel = sw.selected_by_train()
    lines.append("")
    if not sel.empty:
        paid = float((sel["test_r"] * sel["test_trades"]).sum()
                     / max(sel["test_trades"].sum(), 1))
        base = te.loc["baseline", "expectancy_r"] if "baseline" in te.index else 0.0
        lines.append(f"walk-forward SELECTION paid {paid:+.3f}R over "
                     f"{int(sel['test_trades'].sum())} trades, against a "
                     f"baseline of {base:+.3f}R.")
        lines.append("If selection is worse, train expectancy has no "
                     "predictive power here and the honest move is to stop "
                     "choosing and run the baseline.")
    lines.append("")
    lines.append(f"scan passes: {sw.scan_passes} (variants sharing a signal "
                 f"signature shared one)")
    lines.append("READ THE TEST COLUMN. The gap is how much of a variant was "
                 "fitting.")
    return "\n".join(lines)


def _main(argv=None):     # pragma: no cover
    import argparse
    import sys

    from ..config import DEFAULT
    from ..swing import markets as markets_mod
    from ..swing.universe import load_universe
    from . import bars as bars_mod

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = DEFAULT
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", default=cfg.intraday_equity.market,
                   choices=markets_mod.keys(cfg))
    p.add_argument("--years", type=float, default=None)
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--leverage", type=float, default=None)
    p.add_argument("--interval", type=int,
                   default=cfg.intraday_equity.bar_interval_minutes)
    p.add_argument("--variants", default="",
                   help="comma-separated subset; default all")
    p.add_argument("--train-months", type=int, default=None)
    p.add_argument("--test-months", type=int, default=None)
    p.add_argument("--out", default="data/experiments")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)

    if args.capital is not None:
        cfg.capital.intraday_equity_capital_inr = args.capital
    if args.leverage is not None:
        cfg.intraday_equity.mis_leverage = args.leverage

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

    chosen = VARIANTS
    if args.variants:
        keys = [k.strip() for k in args.variants.split(",") if k.strip()]
        chosen = tuple(variant(k) for k in keys)

    start = None
    if args.years:
        sess = loaded.all_sessions()
        if sess:
            start = sess[-1] - timedelta(days=int(args.years * 365))

    print(bt.CAVEATS)

    def progress(key, fold, phase):
        print(f"\r  {key:<14} fold {fold} {phase:<5}", end="", flush=True)

    sw = sweep(cfg, loaded.bars, benchmark=loaded.benchmark, variants=chosen,
               train_months=args.train_months, test_months=args.test_months,
               window_start=start, universe_key=market.universe_csv,
               use_cache=not args.no_cache, progress=progress)
    print()
    print(report(sw))

    try:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        names = "-".join(v.key for v in chosen)[:40]
        sw.frame().to_parquet(outdir / f"ie_sweep_{market.key}_{stamp}_{names}.parquet")
    except Exception as e:
        print(f"(could not write the ledger: {e})")
    return 0


if __name__ == "__main__":     # pragma: no cover
    raise SystemExit(_main())
