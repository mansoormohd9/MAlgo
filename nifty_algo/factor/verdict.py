"""
F1. Is the momentum sleeve distinguishable from a size-tilted random portfolio?

WHY A SECOND DRIVER, WHEN `experiment.py` ALREADY SWEEPS

`experiment.py` answers "which variant", walk-forward, in mean-monthly-return
units. This answers the prior question - "is any of it real" - and it has to be
a CONTINUOUS run to do so, because the two defects it exists to measure are
both invisible fold by fold:

  CONCENTRATION only appears over a long hold. A name that compounds from 5% to
  30% of the book needs years to do it; inside a 6-month fold every arm looks
  equally diversified, and the let-winners-run overlay that `reweight` removes
  has no room to express itself.

  THE GEOMETRIC/ARITHMETIC GAP only appears when returns are chained.
  `experiment.report` prints `(1 + mean monthly)^12 - 1`, which read +41.0% for
  the baseline where a continuous run gives +30.6% - about 10 points of pure
  volatility drag, and MORE for the higher-variance arm, so it widened the gap
  to the null by roughly 4 points as well. Every number here is
  `FactorResult.cagr`, which is `(final/start)^(1/years) - 1` and cannot drift.

THE FOUR THINGS THIS FIXES, ALL OF WHICH RAN ONE WAY

  1. THE NULL WAS ONE SEED. Momentum was compared against a single draw from
     the null distribution rather than against the distribution. Now `n_seeds`
     draws, and the p-value is the empirical rank.
  2. THE NULL WAS HANDICAPPED ON COST. Measured from the ledger, `random`
     turns 1.78x the book per rebalance against momentum's 0.634x - and the
     backtest charged NO slippage at all, because `run` called
     `buy_cost`/`sell_cost` rather than `friction`. Free fills subsidise the
     high-turnover arm nearly three times as much. Both arms are now costed
     identically at `slippage_pct` per leg.
  3. THE BOOK NEVER REWEIGHTED, so it was momentum plus an unregistered
     let-winners-run overlay. Both arms are run.
  4. THE SURVIVORSHIP BOUND TESTED THE WRONG HALF. `listed_only` bounds
     look-ahead about which companies came to EXIST, not which SURVIVED, and
     the surviving half is the one that matters for a long-only book whose
     mechanism is holding the tail. There is no fix inside this data - the
     delisted names are absent from the universe AND the history. What there
     is, is a better bound, and it is reported rather than corrected: see
     `SURVIVORSHIP_NOTE`.

THE GATE IS COMMITTED BEFORE THE RUN AND IS NOT MOVED AFTERWARDS. Both criteria
below were written down in the plan before any of this executed. A threshold
adjusted once the numbers are visible is not a threshold.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from ..experiment_core import Fold, WalkForward, walk_forward_report
from ..swing.costs_equity import DEFAULT_EQUITY_COSTS
from . import backtest as fb

#: Beat this many of `n_seeds` and the signal clears its own null at p<=0.05.
#: Expressed as a quantile so it survives a change in `n_seeds`.
NULL_QUANTILE = 0.95

#: Kill 2. If reweighting gives up more than this share of the excess CAGR
#: over the median null seed, the result was CONCENTRATION rather than
#: momentum - a different product, with different capacity and different odds
#: of repeating - and it is dead as a momentum claim whatever the seed test says.
MAX_EXCESS_GIVEN_UP = 1.0 / 3.0

SURVIVORSHIP_NOTE = """\
THE SURVIVORSHIP BOUND, AND WHY `listedonly` IS NOT IT
  `listed_only` restricts to names already listed when the window opened. That
  bounds look-ahead about which companies came to EXIST. It says nothing about
  which SURVIVED, and survival is the half that matters here: the universe is
  today's Kite dump, so a company delisted mid-window is absent from the symbol
  list AND from every bar. A long-only book holding the top of a wide mid-cap
  cross-section is holding precisely where suspensions and collapses live.
  The measured `listedonly` gap (~1.7pp of CAGR) therefore bounds the SMALL
  half and must never be quoted as the survivorship correction.

  THE BETTER BOUND IS THE LIQUIDITY BAND. A top-turnover-quintile name
  essentially cannot delist inside a decade, so `band="liquid"` is the closest
  thing to a survivorship-free measurement this data can produce. Measured as
  GEOMETRIC CAGR at 0.25%/leg over the full window:

      all         +18.8%   maxDD -57.2%   recovery 35m
      liquid      +15.5%   maxDD -43.0%   recovery 30m    <- the bound
      midliquid   +18.1%   maxDD -52.1%   recovery 35m
      illiquid    +16.0%   maxDD -52.8%   recovery 38m
      listedonly  +15.6%   maxDD -54.5%   recovery 40m    <- wrong half

  So the haircut is about 3.3pp, or 17% of the headline - and **not** the 33%
  an earlier draft of this note claimed. That 33% came from comparing
  ARITHMETIC MEAN MONTHLY returns in the walk-forward ledger (1.94% against
  2.90%), which is the same volatility-drag-inflated statistic this module
  exists to stop quoting. Corrected by measuring both as CAGR. The lesson
  survived its own author by about a day.

  The bands also land within 3pp of each other, so the edge is NOT confined to
  names too illiquid to trade - which was the other way this result could have
  been untradeable and is not.
"""


@dataclass
class Arm:
    """One configuration, run continuously over the whole window."""
    label: str
    cagr: float
    sharpe: float
    max_dd: float
    recovery_months: float
    turnover: float
    costs_inr: float
    trades: int
    concentration: dict = field(default_factory=dict)

    @classmethod
    def of(cls, label: str, res: fb.FactorResult, capital: float) -> "Arm":
        return cls(label=label, cagr=res.cagr(capital), sharpe=res.sharpe(),
                   max_dd=res.max_drawdown(),
                   recovery_months=res.longest_drawdown_months(),
                   turnover=res.avg_turnover(), costs_inr=res.costs_paid,
                   trades=res.trades, concentration=res.concentration())


@dataclass
class Verdict:
    slippage_pct: float
    drift: Arm
    reweighted: Arm
    nulls: list                       # list[Arm]
    capital: float

    # ---------------- the null distribution ----------------

    @property
    def null_cagrs(self) -> list:
        return sorted(a.cagr for a in self.nulls)

    @property
    def null_median(self) -> float:
        return statistics.median(self.null_cagrs) if self.nulls else 0.0

    def beaten(self, cagr: float) -> int:
        return sum(1 for c in self.null_cagrs if cagr > c)

    def empirical_p(self, cagr: float) -> float:
        """
        One-sided rank of `cagr` in the null distribution.

        `(1 + #{null >= observed}) / (1 + n)` - the standard permutation form,
        which cannot return zero and so never claims more certainty than the
        number of draws supports.
        """
        n = len(self.null_cagrs)
        if not n:
            return float("nan")
        ge = sum(1 for c in self.null_cagrs if c >= cagr)
        return (1.0 + ge) / (1.0 + n)

    # ---------------- the two gates ----------------

    @property
    def gate1_pass(self) -> bool:
        return self.empirical_p(self.drift.cagr) <= (1.0 - NULL_QUANTILE)

    @property
    def excess_drift(self) -> float:
        return self.drift.cagr - self.null_median

    @property
    def excess_reweighted(self) -> float:
        return self.reweighted.cagr - self.null_median

    @property
    def retained(self) -> float:
        """Share of the excess that survives equal-weight rebalancing."""
        if self.excess_drift <= 0:
            return float("nan")
        return self.excess_reweighted / self.excess_drift

    @property
    def gate2_pass(self) -> bool:
        r = self.retained
        return r == r and r >= (1.0 - MAX_EXCESS_GIVEN_UP)


def run_arms(bars: dict, capital: float, cfg_factor, n_seeds: int = 40,
             slippage_pct: float = 0.0025, seed0: int = 20260904,
             progress=None) -> Verdict:
    """
    Both momentum arms and `n_seeds` null draws, all over the same window.

    The universe is built ONCE and shared. It is a pure function of the bars -
    independent of window, variant and seed - and rebuilding it per call is
    how an earlier sweep spent 24 of its 25 minutes recomputing one object.
    """
    universe = fb.FactorUniverse(bars)

    def one(label, *, seed=None, reweight=False):
        if progress:
            progress(label)
        res = fb.run(
            bars, capital, top_n=cfg_factor.top_n, band=cfg_factor.band,
            formation=cfg_factor.formation,
            hold_months=cfg_factor.hold_months,
            min_price=cfg_factor.min_price,
            min_turnover=cfg_factor.min_turnover_inr,
            min_history=cfg_factor.min_history_sessions,
            costs=DEFAULT_EQUITY_COSTS, seed=seed, universe=universe,
            reweight=reweight, slippage_pct=slippage_pct)
        return Arm.of(label, res, capital)

    drift = one("momentum (drift weights)")
    reweighted = one("momentum (equal weight)", reweight=True)
    nulls = [one(f"null seed {i}", seed=seed0 + i) for i in range(n_seeds)]
    return Verdict(slippage_pct=slippage_pct, drift=drift,
                   reweighted=reweighted, nulls=nulls, capital=capital)


def report(v: Verdict) -> str:
    """The table and the two gates. Nothing here decides anything."""
    lines = [
        f"F1 - the factor sleeve against its own null, "
        f"at {v.slippage_pct:.2%}/leg slippage",
        "",
        "  All figures are GEOMETRIC CAGR from a single continuous run.",
        "  Never `(1 + mean monthly)^12 - 1`: that is arithmetic, it carries",
        "  volatility drag, and it inflates the higher-variance arm more.",
        "",
        f"  {'arm':<26}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>8}"
        f"{'recov':>7}{'turn':>7}{'costs Rs':>12}",
    ]
    for a in (v.drift, v.reweighted):
        lines.append(
            f"  {a.label:<26}{a.cagr:>+9.1%}{a.sharpe:>8.2f}{a.max_dd:>8.1%}"
            f"{a.recovery_months:>6.0f}m{a.turnover:>7.0%}{a.costs_inr:>12,.0f}")

    cs = v.null_cagrs
    if cs:
        lines += [
            f"  {'null (n=' + str(len(cs)) + ')':<26}"
            f"{v.null_median:>+9.1%}{'median':>8}",
            f"  {'  null range':<26}{cs[0]:>+9.1%} .. {cs[-1]:+.1%}"
            f"   p10 {np.percentile(cs, 10):+.1%}"
            f"  p90 {np.percentile(cs, 90):+.1%}",
        ]

    c = v.drift.concentration
    if c:
        lines += [
            "",
            f"  CONCENTRATION (drift arm): top name = "
            f"{c['top1_share_of_gains']:.0%} of all gains, "
            f"top 3 = {c['topn_share_of_gains']:.0%}, "
            f"across {c['names']} names traded.",
        ]

    p = v.empirical_p(v.drift.cagr)
    lines += [
        "",
        "  THE GATE, committed before the run:",
        f"    1. beat >= {NULL_QUANTILE:.0%} of null seeds ..... "
        f"{'PASS' if v.gate1_pass else 'FAIL'}  "
        f"(beat {v.beaten(v.drift.cagr)}/{len(cs)}, empirical p = {p:.3f})",
        f"    2. equal-weight keeps >= "
        f"{1 - MAX_EXCESS_GIVEN_UP:.0%} of the excess ... "
        f"{'PASS' if v.gate2_pass else 'FAIL'}  "
        f"(excess {v.excess_drift:+.1%} -> {v.excess_reweighted:+.1%}, "
        f"retained {v.retained:.0%})",
        "",
    ]
    if v.gate1_pass and v.gate2_pass:
        lines.append("  BOTH PASS - the sleeve is distinguishable from its "
                     "null and is not carried by concentration.")
    elif not v.gate1_pass:
        lines.append("  GATE 1 FAILED. Not distinguishable from picking names "
                     "out of a hat. The sleeve is dead.")
    else:
        lines.append("  GATE 2 FAILED. The excess is CONCENTRATION, not "
                     "momentum - dead as a momentum claim.")
    lines += ["", SURVIVORSHIP_NOTE]
    return "\n".join(lines)


__all__ = ["Arm", "Verdict", "run_arms", "report", "NULL_QUANTILE",
           "MAX_EXCESS_GIVEN_UP", "SURVIVORSHIP_NOTE"]


def _main(argv=None) -> int:      # pragma: no cover
    import argparse
    import sys
    import time as _clock
    from pathlib import Path

    import pandas as pd

    from ..config import DEFAULT

    p = argparse.ArgumentParser(description="F1 - the factor sleeve's gate.")
    p.add_argument("--capital", type=float, default=500_000.0)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--slippage", type=float, default=0.0025,
                   help="per leg, charged to EVERY arm identically")
    p.add_argument("--top-n", type=int, default=None)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    if args.top_n:
        cfg.factor.top_n = args.top_n

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
    print(f"{len(bars):,} symbols loaded from {path}", flush=True)
    print(fb.CAVEATS, flush=True)

    started = _clock.time()
    done = [0]
    total = args.seeds + 2

    def progress(label):
        done[0] += 1
        print(f"  [{done[0]}/{total}] {label}", flush=True)

    v = run_arms(bars, args.capital, cfg.factor, n_seeds=args.seeds,
                 slippage_pct=args.slippage, progress=progress)
    print(f"\ndone in {_clock.time() - started:.0f}s\n", flush=True)
    print(report(v))

    try:
        out = Path("data/experiments")
        out.mkdir(parents=True, exist_ok=True)
        rows = [{"arm": a.label, "kind": k, "cagr": a.cagr,
                 "sharpe": a.sharpe, "max_dd": a.max_dd,
                 "turnover": a.turnover, "costs_inr": a.costs_inr,
                 "trades": a.trades, "slippage_pct": v.slippage_pct}
                for k, arms in (("momentum", [v.drift, v.reweighted]),
                                ("null", v.nulls))
                for a in arms]
        stamp = _clock.strftime("%Y%m%d-%H%M%S")
        f = out / f"f1_verdict_{stamp}_slip{int(v.slippage_pct*10000)}bp.parquet"
        pd.DataFrame(rows).to_parquet(f, index=False)
        print(f"\nledger -> {f}")
    except Exception as e:
        print(f"\ncould not write the ledger ({e}) - the table above stands")
    return 0


if __name__ == "__main__":        # pragma: no cover
    raise SystemExit(_main())


# ------------------------------------------- the walk-forward driver

# `Fold`, `WalkForward` and `walk_forward_report` moved to `experiment_core`
# when the swing book wanted the same statistic. What stays here is the loop
# that drives THIS book's backtester - the split the shared module documents.

def walk_forward(bars: dict, capital: float, cfg_factor, windows,
                 n_seeds: int = 40, slippage_pct: float = 0.0025,
                 seed0: int = 20260904, universe=None,
                 progress=None) -> WalkForward:
    """
    Momentum against a null DISTRIBUTION in every out-of-sample window.

    Only test windows are run. Train windows exist to choose between variants,
    and nothing is being chosen here - the question is whether one fixed rule
    beats noise consistently, not which rule to use.
    """
    if universe is None:
        universe = fb.FactorUniverse(bars)
    out: list = []
    for i, w in enumerate(windows):
        def one(seed=None):
            res = fb.run(
                bars, capital, top_n=cfg_factor.top_n, band=cfg_factor.band,
                formation=cfg_factor.formation,
                hold_months=cfg_factor.hold_months,
                min_price=cfg_factor.min_price,
                min_turnover=cfg_factor.min_turnover_inr,
                min_history=cfg_factor.min_history_sessions,
                costs=DEFAULT_EQUITY_COSTS, seed=seed, universe=universe,
                reweight=cfg_factor.reweight, slippage_pct=slippage_pct,
                start=w.test_start, end=w.test_end)
            # Chained period return over the window. NOT annualised: a +20%
            # half-year annualises to +44% and a -20% to only -36%, so a mean
            # of annualised short windows is upward biased - the bug this
            # book already shipped once.
            s = res.series()
            return float(s[-1] / s[0] - 1.0) if len(s) >= 2 else 0.0

        if progress:
            progress(i + 1, len(windows), f"{w.test_start}..{w.test_end}")
        out.append(Fold(index=i, start=w.test_start, end=w.test_end,
                        score=one(None),
                        nulls=[one(seed0 + i * 1000 + j)
                               for j in range(n_seeds)]))
    return WalkForward(folds=out, slippage_pct=slippage_pct)


__all__ += ["Fold", "WalkForward", "walk_forward", "walk_forward_report"]
