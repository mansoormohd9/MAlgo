"""
Which factor rule set, chosen out of sample.

Runs on `experiment_core` - the same `Variant`, `Cell`, `Sweep`, trade-weighted
pooling, fold-resampled interval, sign test, `selected_by_train` and ledger the
option book uses. This module owns only its VARIANTS table and the loop that
drives its own backtester, which is the whole point of that split: three copies
of the statistics is how two books end up unable to compare a result.

THE UNIT IS A MONTHLY RETURN, NOT AN R. READ THIS BEFORE THE TABLE.

`Cell.expectancy_r` is populated with the sleeve's **mean return per rebalance
period**, and `Cell.trades` with the **number of rebalance periods**. Those two
columns are what `pooled`, `bootstrap_ci` and `sign_test` are defined over, so
reusing them buys the correct statistics - period-weighted pooling, an interval
that resamples FOLDS rather than periods, and the sign test that discriminates.

It does NOT make a factor row comparable to an option row. A month of an
equal-weight sleeve is not a multiple of a per-trade stop, and the two must
never be read side by side merely because they share a column name. Every
report this module prints says "monthly return" and never "R", and the
factor-native figures - CAGR, Sharpe, max drawdown, recovery, turnover - travel
in `Cell.extra`, which exists precisely so a third book need not widen the
shared type.

WHAT THE SWEEP IS FOR, GIVEN THE BASE RATES

65% of 452 published anomalies fail a t>1.96 hurdle and 82% fail at t>2.78;
post-publication returns run about half of in-sample ones. So the default
expectation is that no variant beats `random`, and the sweep is built to make
that outcome legible rather than to avoid it. `random` is a first-class
variant, not a footnote, and the SIGN TEST against it is the gate - a pooled
average can be carried by one good stretch, as the option book's warm-up
variant demonstrated by improving pooled expectancy while winning 10 of 21
windows against a coin-flip 10.5.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .. import experiment_core as ec
from ..config import DEFAULT, Config
from ..swing.backtest import fold_windows
from . import backtest as fb

#: The null is pre-registered as a VARIANT, alongside the real ones, so it is
#: run and reported on exactly the same footing.
NULL_KEY = "random"

VARIANTS: tuple = (
    ec.Variant("baseline", "12-1, whole universe",
               "the standard formation on every eligible name",
               lambda c: None),

    # --- L1: the liquidity band. The reason this book exists: the swing
    #     book's universe cannot express it (its Rs 25cr floor rejects none
    #     of the Nifty 100, whose least liquid name trades Rs 69cr).
    ec.Variant("liquid", "top-quintile turnover",
               "Indian evidence says this band pays least - it is the control, "
               "not the candidate",
               ec.set_on("factor", band="liquid")),
    ec.Variant("midliquid", "turnover quintiles 2-3",
               "the band a small account reaches and a fund cannot",
               ec.set_on("factor", band="midliquid")),
    ec.Variant("illiquid", "lower tradable quintile",
               "highest reported alpha, worst fills - the trade-off this "
               "sweep exists to price",
               ec.set_on("factor", band="illiquid")),

    # --- L2: the formation window. Two, because the literature uses two.
    ec.Variant("mom6", "6-1 formation",
               "a shorter window reacts faster and holds more reversal",
               ec.set_on("factor", formation="mom6_1")),

    # --- L3: rebalance frequency, the friction/decay trade-off.
    ec.Variant("hold3", "quarterly rebalance",
               "a third of the turnover; does the signal decay faster than "
               "the costs it saves",
               ec.set_on("factor", hold_months=3)),

    # --- L4: the survivorship bound. Not a fix.
    ec.Variant("listedonly", "names listed before the window",
               "a LOWER bound on survivorship inflation, not a correction",
               ec.set_on("factor", listed_only=True)),

    # --- the null.
    ec.Variant(NULL_KEY, "random selection",
               "identical book over noise. Given the replication base rates "
               "this is the default expectation, not a formality",
               ec.set_on("factor", random_seed=20260904)),
)


@dataclass
class PeriodStats:
    """A fold's worth of rebalance-period returns."""
    periods: int
    mean_return: float
    total_return: float
    win_rate: float


def _period_returns(result: fb.FactorResult) -> np.ndarray:
    """Return per rebalance period, from the equity curve."""
    s = result.series()
    if len(s) < 2:
        return np.array([], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(s) / s[:-1]
    return r[np.isfinite(r)]


def _stats(result: fb.FactorResult, hold_months: int = 1) -> PeriodStats:
    """
    Per-period returns, converted to a MONTHLY-EQUIVALENT.

    `hold3` rebalances quarterly, so its period return covers three months.
    Comparing that against a monthly variant's period return - which the sign
    test and the pooled table both do, column for column - reported `hold3` at
    +9.28% against a baseline of +2.90% and made a lower-turnover variant look
    three times better than it was. Geometric conversion puts every variant in
    the same unit; `periods` still counts actual rebalances, so the weighting
    and the interval keep reflecting how many independent observations there
    really were.
    """
    r = _period_returns(result)
    if not len(r):
        return PeriodStats(0, 0.0, 0.0, 0.0)
    if hold_months > 1:
        r = np.sign(1.0 + r) * np.abs(1.0 + r) ** (1.0 / hold_months) - 1.0
    return PeriodStats(
        periods=len(r),
        mean_return=float(np.mean(r)),
        total_return=float(np.prod(1.0 + r) - 1.0),
        win_rate=float(np.mean(r > 0)))


def _run_window(cfg: Config, bars: dict, capital: float,
                start: date, end: date, universe=None,
                listed_before: date | None = None) -> fb.FactorResult:
    f = cfg.factor
    return fb.run(
        bars, capital, top_n=f.top_n, band=f.band, formation=f.formation,
        hold_months=f.hold_months, min_price=f.min_price,
        min_turnover=f.min_turnover_inr, min_history=f.min_history_sessions,
        listed_only=f.listed_only, seed=f.random_seed,
        start=start, end=end, universe=universe,
        listed_before=listed_before)


def sweep(base: Config, bars: dict, capital: float,
          variants=VARIANTS, train_months: int = 24, test_months: int = 6,
          progress=None) -> ec.Sweep:
    """
    Every variant on every fold, train and test.

    Train windows are EXECUTED rather than carried as labels, because
    selecting on train and scoring on test is the only thing that separates a
    choice made out of sample from a description of the past.
    """
    sessions = sorted({d for f in bars.values()
                       for d in (t.date() if hasattr(t, "date") else t
                                 for t in f.index)})
    if len(sessions) < 400:
        return ec.Sweep()

    windows = fold_windows(sessions[0], sessions[-1], train_months, test_months)
    # Built ONCE. Independent of window, variant and seed.
    shared = fb.FactorUniverse(bars)
    # The survivorship anchor is the start of the EVALUATION period, not
    # the first row of the raw data - nothing is listed before that, so
    # anchoring there restricts the universe to nobody and the variant
    # silently reports a flat zero.
    out = ec.Sweep()
    total = len(variants) * len(windows) * 2
    n = 0
    for v in variants:
        cfg = v.configure(base)
        for i, w in enumerate(windows):
            for phase, lo, hi in (("train", w.train_start, w.train_end),
                                  ("test", w.test_start, w.test_end)):
                n += 1
                if progress:
                    progress(n, total, f"{v.key} fold{i} {phase}")
                res = _run_window(cfg, bars, capital, lo, hi,
                                  universe=shared,
                                  listed_before=windows[0].test_start)
                st = _stats(res, cfg.factor.hold_months)
                out.cells.append(ec.Cell(
                    variant=v.key, fold=i, phase=phase, start=lo, end=hi,
                    trades=st.periods,
                    # The shared columns, populated with MONTHLY RETURNS so the
                    # harness's statistics apply. Never an R - see the module
                    # docstring.
                    expectancy_r=st.mean_return,
                    total_r=st.total_return,
                    win_rate=st.win_rate,
                    max_drawdown_r=abs(res.max_drawdown()),
                    extra={
                        "cagr": res.cagr(capital),
                        "sharpe": res.sharpe(),
                        "max_dd_pct": res.max_drawdown(),
                        "recovery_months": res.longest_drawdown_months(),
                        "turnover": res.avg_turnover(),
                        "costs_inr": res.costs_paid,
                        "universe_median": (
                            float(np.median(res.universe_size))
                            if res.universe_size else 0.0),
                    }))
    return out


def report(sw: ec.Sweep, baseline_key: str = "baseline") -> str:
    """
    The shared table, relabelled in the sleeve's own units, plus the gate.

    The sign test is printed against the NULL rather than against the
    baseline, because the question this book has to answer first is not "is
    12-1 better than 6-1" but "is any of this better than picking names out of
    a hat".
    """
    lines = [fb.CAVEATS, "",
             "UNIT IS A MEAN MONTHLY RETURN, NOT AN R. A factor month is not a",
             "multiple of a per-trade stop; do not read these beside the option",
             "or intraday books merely because the column names match.", ""]
    lines.append(ec.report(sw, baseline_key=baseline_key)
                 .replace("expectancy_r", "monthly_ret"))

    lines += ["", "SIGN TEST AGAINST THE NULL - the gate that actually "
                  "discriminates", ""]
    frame = sw.frame()
    if frame.empty:
        return "\n".join(lines + ["no cells"])

    # NO CAGR COLUMN. A mean of per-fold ANNUALISED returns is upward
    # biased - a +20% half-year annualises to +44% while -20% annualises to
    # only -36% - and it read +65.6% where a single continuous run over the
    # same window gives +30.6%. Annualise the pooled monthly figure, or run
    # the window continuously; never average annualised short windows.
    lines.append(f"  {'variant':<12}{'monthly':>10}{'annualised':>12}"
                 f"{'95% CI (test)':>24}{'vs random':>12}{'p':>8}{'worst DD':>10}")
    pooled = sw.pooled()
    test = pooled[pooled["phase"] == "test"].set_index("variant")
    for v in VARIANTS:
        if v.key not in test.index:
            continue
        lo, hi = sw.bootstrap_ci(v.key, "test")
        sign = sw.sign_test(v.key, against=NULL_KEY, phase="test")
        g = frame[(frame["variant"] == v.key) & (frame["phase"] == "test")]
        dd = float(g["max_dd_pct"].min()) if "max_dd_pct" in g else float("nan")
        wins = (f"{sign['wins']}/{sign['n']}" if sign["n"] else "-")
        p = f"{sign['p']:.3f}" if sign["n"] else "-"
        monthly = float(test.loc[v.key, "expectancy_r"])
        annual = (1.0 + monthly) ** 12 - 1.0
        lines.append(
            f"  {v.key:<12}{monthly:>+10.4%}{annual:>+12.1%}"
            f"   [{lo:>+8.4%},{hi:>+8.4%}]{wins:>12}{p:>8}{dd:>10.1%}")

    lines += ["",
              "  A variant is interesting only if it beats `random` on the SIGN",
              "  TEST, not merely on pooled return. Pooled averages are carried",
              "  by single stretches - the option book's warm-up variant improved",
              "  pooled expectancy and then won 10 of 21 windows against a",
              "  coin-flip 10.5."]
    return "\n".join(lines)


def _main(argv=None) -> int:     # pragma: no cover
    p = argparse.ArgumentParser(description="Factor sleeve sweep.")
    p.add_argument("--capital", type=float, default=500_000.0)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--train-months", type=int, default=24)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--variants", default="",
                   help="comma-separated subset; the null is always included")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    cfg.capital.factor_capital_inr = args.capital
    if args.top_n:
        cfg.factor.top_n = args.top_n

    from pathlib import Path
    path = Path(cfg.factor.cache_dir) / cfg.factor.cache_name
    if not path.exists():
        print(f"No factor cache at {path}.\n"
              f"Run: python scripts/fetch_factor_history.py --years 10")
        return 2

    raw = pd.read_parquet(path)
    bars = {}
    for symbol, g in raw.groupby("symbol"):
        if str(symbol).startswith("__"):
            continue
        bars[str(symbol)] = g.drop(columns=["symbol"]).sort_index()
    print(f"{len(bars):,} symbols loaded from {path}")

    chosen = VARIANTS
    if args.variants:
        want = {k.strip() for k in args.variants.split(",")} | {NULL_KEY}
        chosen = tuple(v for v in VARIANTS if v.key in want)

    def progress(n, total, label):
        if n % 10 == 0 or n == total:
            print(f"  [{n}/{total}] {label}", flush=True)

    sw = sweep(cfg, bars, args.capital, variants=chosen,
               train_months=args.train_months, test_months=args.test_months,
               progress=progress)
    if not sw.cells:
        print("not enough history for a walk-forward split")
        return 1

    print()
    print(report(sw))
    try:
        out = ec.write_ledger(sw, ec.DEFAULT_LEDGER_DIR, "sweep_factor",
                              "-".join(v.key for v in chosen))
        print(f"\nledger -> {out}")
    except Exception as e:
        print(f"\ncould not write the ledger ({e}) - the table above stands")
    return 0


__all__ = ["VARIANTS", "NULL_KEY", "sweep", "report", "PeriodStats"]


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(_main())
