"""
F2a. Three drawdown instruments for the factor sleeve, on ONE axis.

The sleeve is the only book in this repo that beats its own null, and its
problem is not expectancy - it is a -57.2% peak-to-trough with a 35-month
recovery, measured on month-end marks. Three instruments could address that,
and they are usually argued about rather than measured against each other:

  a per-name STOP        sell a holding that falls far enough intra-month
  a sleeve-level REGIME  hold cash while the benchmark is below its average
  a static CASH BLEND    simply hold less of it

THE AXIS IS DRAWDOWN REDUCTION PER POINT OF CAGR GIVEN UP, not CAGR. Any of
the three will cut the drawdown if it cuts enough of the return with it; the
only question worth asking is which one buys the most protection per point
surrendered, and whether any of them beats the trivial answer of holding less.

WHY THE STOP HAS TWO BASES, AND WHY THAT WAS DECIDED BEFORE THE RUN. A stop
measured from the original fill drifts far below the market on a name held for
months, so it stops firing - running only that would have produced "stops do
nothing" as an artefact of a rule that never triggers. The rebalance basis
re-anchors every month and is the strict form. Both are in the family, and the
family is EIGHT arms, declared here rather than grown to fit the answer.

TWO THINGS RUN IN THE STOP'S FAVOUR, AND BOTH ARE LEFT THAT WAY ON PURPOSE.
It fills at the close of the session that breached, having read that same close -
half a session ahead of the rebalance convention, which decides on prior data -
so a real stop would fill at the next open and, in a fall, lower. And cash
released by a stop earns nothing until the next rebalance, worth roughly 0.08%
a year against it. The arms lost anyway; a generous fill that still loses is a
stronger result than a fair one, and `FactorUniverse` stores closes only, so
filling at the next open is not available without widening it.

WHAT THIS CANNOT DO IS SELECT. Eight arms on one window is a search, and this
repo has already measured how far a naive statistic on a search is inflated. A
winner here is a CANDIDATE; it is confirmed or killed on India 2005-2016, a
window no parameter in this package has seen, through `--start`/`--end`.
"""
from __future__ import annotations

import argparse
import sys
import time as _clock
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DEFAULT
from ..experiment_core import two_sided_sign_p
from ..swing.costs_equity import DEFAULT_EQUITY_COSTS
from . import backtest as fb

#: The gate, committed before the run. `MIN_DD_CUT_PP` is what would make an
#: instrument worth its complexity at all; `MAX_EXCESS_GIVEN_UP` is F1's
#: existing threshold, reused rather than reinvented, applied here to the
#: sleeve's excess over the benchmark instead of over the null.
MIN_DD_CUT_PP = 10.0
MAX_EXCESS_GIVEN_UP = 1.0 / 3.0

#: India's short-term risk-free rate, for the cash arm only. VERIFY against
#: what the account actually earns on idle cash - a liquid fund and a sweep FD
#: are not the same number, and this arm is linear in it.
DEFAULT_RF = 0.065


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    why: str
    #: kwargs handed to `backtest.run`. Empty means the shipped sleeve.
    kwargs: dict = field(default_factory=dict)
    #: Non-zero only for the arm that is arithmetic on the baseline curve
    #: rather than a re-run: the fraction of the pot held in the sleeve.
    blend: float = 0.0


ARMS: tuple[Arm, ...] = (
    Arm("none", "as shipped",
        "the -57.2% baseline every other row is read against"),

    Arm("stop15e", "stop -15% from entry",
        "the rule as usually proposed: a hard floor under each holding",
        {"stop_pct": 0.15, "stop_basis": "entry"}),
    Arm("stop25e", "stop -25% from entry",
        "the same rule, loose enough to sit outside ordinary single-name noise",
        {"stop_pct": 0.25, "stop_basis": "entry"}),

    Arm("stop15r", "stop -15% from last mark",
        "the strict form: re-anchored monthly, so it keeps firing on a winner",
        {"stop_pct": 0.15, "stop_basis": "rebalance"}),
    Arm("stop25r", "stop -25% from last mark",
        "the same, at a width a one-month holding period can plausibly survive",
        {"stop_pct": 0.25, "stop_basis": "rebalance"}),

    Arm("regime200", "cash below the 200DMA",
        "the same market call the per-name stops make, priced as ONE decision "
        "a month instead of twenty",
        {"regime_ma_days": 200}),
    Arm("regime50", "cash below the 50DMA",
        "the faster version, closer to the one-month holding period",
        {"regime_ma_days": 50}),

    Arm("cash70", "70% sleeve / 30% cash",
        "the trivial answer, and the one the other five have to beat: hold "
        "less of it and earn the risk-free rate on the rest",
        blend=0.70),
)


def arm(key: str) -> Arm:
    for a in ARMS:
        if a.key == key:
            return a
    raise KeyError(f"unknown arm {key!r}; have {[a.key for a in ARMS]}")


@dataclass
class Row:
    key: str
    label: str
    cagr: float
    max_dd: float
    recovery_months: float
    turnover: float
    trades: int
    stops: int
    regime_flat: int
    vol: float

    @property
    def dd_pct(self) -> float:
        return abs(self.max_dd) * 100.0


def _annual_vol(series: np.ndarray) -> float:
    if len(series) < 3:
        return float("nan")
    r = np.diff(series) / series[:-1]
    return float(np.std(r, ddof=1)) * float(np.sqrt(12.0))


def _stats(key: str, label: str, res, capital: float) -> Row:
    return Row(key=key, label=label, cagr=res.cagr(capital),
               max_dd=res.max_drawdown(),
               recovery_months=res.longest_drawdown_months(),
               turnover=res.avg_turnover(), trades=res.trades,
               stops=res.stops_fired, regime_flat=res.regime_flat,
               vol=_annual_vol(res.series()))


def _recovery_months(curve: np.ndarray, dates: list) -> float:
    """The same definition `FactorResult.longest_drawdown_months` uses."""
    peak, peak_at, worst = curve[0], dates[0], 0.0
    for value, when in zip(curve, dates):
        if value >= peak:
            worst = max(worst, (when - peak_at).days / 30.44)
            peak, peak_at = value, when
    return max(worst, (dates[-1] - peak_at).days / 30.44)


def _blended(base, capital: float, weight: float, rf: float) -> Row:
    """
    The cash arm, computed on the baseline's own month-end curve.

    Re-running the sleeve at 70% size would answer a different and less
    interesting question - it would also re-fill every position at a different
    integer share count and mix a sizing effect into the comparison. The
    arithmetic here is exact, and the cash leg is compounded over each
    interval's actual length rather than assumed to be a calendar month.
    """
    series = base.series()
    dates = [d for d, _ in base.equity]
    r = np.diff(series) / series[:-1]
    months = np.array([(b - a).days for a, b in zip(dates, dates[1:])]) / 30.44
    cash_leg = (1.0 + rf) ** (months / 12.0) - 1.0
    blend = weight * r + (1.0 - weight) * cash_leg
    curve = np.concatenate([[capital], capital * np.cumprod(1.0 + blend)])
    peak = np.maximum.accumulate(curve)
    years = (dates[-1] - dates[0]).days / 365.25
    return Row(key="cash70",
               label=f"{weight:.0%} sleeve / {1 - weight:.0%} cash",
               cagr=(curve[-1] / capital) ** (1 / years) - 1.0,
               max_dd=float(np.min(curve / peak - 1.0)),
               recovery_months=_recovery_months(curve, dates),
               turnover=base.avg_turnover() * weight, trades=base.trades,
               stops=0, regime_flat=0,
               vol=float(np.std(blend, ddof=1)) * float(np.sqrt(12.0)))


def _on_marks(benchmark, marks):
    """The benchmark's closes on the sleeve's OWN mark dates."""
    close = benchmark["close"]
    close = close[~close.index.duplicated()]
    idx = pd.to_datetime([pd.Timestamp(d) for d in marks])
    return close.reindex(close.index.union(idx)).ffill().reindex(idx).dropna()


def benchmark_stats(benchmark, marks) -> tuple[float, float, float]:
    """
    The index's CAGR, volatility and max drawdown on the sleeve's mark dates.

    Resampling the benchmark to calendar month-ends instead would compare two
    different date grids and silently drop a third of the months to the join.
    """
    v = _on_marks(benchmark, marks)
    if len(v) < 3:
        return float("nan"), float("nan"), float("nan")
    years = (v.index[-1] - v.index[0]).days / 365.25
    r = v.pct_change().dropna()
    return (float((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1.0),
            float(r.std()) * float(np.sqrt(12.0)),
            float((v / v.cummax() - 1.0).min()))


def run_arms(bars: dict, capital: float, cfg_factor, benchmark=None,
             slippage_pct: float = 0.0025, rf: float = DEFAULT_RF,
             start=None, end=None, arms: tuple[Arm, ...] = ARMS,
             progress=None):
    """
    Every arm over one window, on a universe built once and shared.

    Returns `(rows, results)`. The raw results come back too because the
    caller needs the baseline's own mark dates to put the benchmark on the
    same grid, and because the month-by-month consistency check works on the
    curves rather than on the summary rows.
    """
    universe = fb.FactorUniverse(bars)
    rows: list[Row] = []
    results: dict = {}
    base = None
    for a in arms:
        if progress:
            progress(a.label)
        if a.blend:
            if base is None:
                raise ValueError(
                    "the blended arm is arithmetic on the baseline curve, so "
                    "`none` has to run ahead of it")
            rows.append(_blended(base, capital, a.blend, rf))
            continue
        # The config supplies the defaults and the ARM overrides them, so a
        # `FactorConfig` field cannot sit there doing nothing while its name
        # claims otherwise - the failure `partial_exit_at_r` produced in the
        # swing sweep, where a variant renamed the book and changed none of it.
        settings = {"stop_pct": cfg_factor.stop_pct,
                    "stop_basis": cfg_factor.stop_basis,
                    "regime_ma_days": cfg_factor.regime_ma_days}
        settings.update(a.kwargs)
        res = fb.run(
            bars, capital, top_n=cfg_factor.top_n, band=cfg_factor.band,
            formation=cfg_factor.formation, hold_months=cfg_factor.hold_months,
            min_price=cfg_factor.min_price,
            min_turnover=cfg_factor.min_turnover_inr,
            min_history=cfg_factor.min_history_sessions,
            costs=DEFAULT_EQUITY_COSTS, universe=universe,
            reweight=cfg_factor.reweight, slippage_pct=slippage_pct,
            start=start, end=end, benchmark=benchmark, **settings)
        if a.key == "none":
            base = res
        results[a.key] = res
        rows.append(_stats(a.key, a.label, res, capital))
    return rows, results


def report(rows: list[Row], bench_cagr: float, bench_vol: float,
           bench_dd: float, rf: float, window: str) -> str:
    base = next(r for r in rows if r.key == "none")
    excess = base.cagr - bench_cagr
    budget_pp = excess * MAX_EXCESS_GIVEN_UP * 100.0
    out = [
        f"F2a - drawdown instruments for the factor sleeve, {window}",
        "",
        f"  benchmark over the same marks: CAGR {bench_cagr:+.2%}  "
        f"vol {bench_vol:.1%}  maxDD {bench_dd:.1%}",
        f"  sleeve excess over it {excess * 100:+.2f}pp, so the budget is one "
        f"third of that = {budget_pp:.2f}pp of CAGR",
        f"  the cash arm earns rf = {rf:.2%} (VERIFY against your own "
        f"idle-cash rate; that arm is linear in it)",
        "",
        f"  {'arm':<26}{'CAGR':>9}{'vol':>7}{'maxDD':>8}{'recov':>7}"
        f"{'turn':>7}{'DDcut':>8}{'gave up':>9}{'per pp':>8}  gate",
    ]
    for r in rows:
        dd_cut = base.dd_pct - r.dd_pct
        gave = (base.cagr - r.cagr) * 100.0
        per = dd_cut / gave if gave > 0.05 else float("inf")
        per_txt = "     inf" if per == float("inf") else f"{per:>8.2f}"
        ok = dd_cut >= MIN_DD_CUT_PP and gave <= budget_pp
        gate = "-" if r.key == "none" else ("PASS" if ok else "fail")
        out.append(
            f"  {r.label:<26}{r.cagr:>+9.2%}{r.vol:>7.1%}{r.max_dd:>8.1%}"
            f"{r.recovery_months:>6.0f}m{r.turnover:>7.0%}"
            f"{dd_cut:>+8.1f}{gave:>+9.2f}{per_txt}  {gate}")
    out += [
        "",
        f"  THE GATE, committed before the run: cut max drawdown by at least "
        f"{MIN_DD_CUT_PP:.0f}pp while giving up no more than {budget_pp:.2f}pp "
        f"of CAGR. Ties go to the LOWER-turnover arm, because turnover is the "
        f"one cost that is certain.",
        "",
        "  'per pp' is drawdown removed per point of CAGR surrendered - the "
        "axis. A higher CAGR with an unchanged drawdown is not a result here, "
        "and neither is a flat curve bought by giving up the return.",
    ]
    acted = [r for r in rows if r.stops or r.regime_flat]
    if acted:
        out += ["", "  how often each instrument actually acted:"]
        for r in acted:
            what = (f"{r.stops:,} stop sells" if r.stops
                    else f"{r.regime_flat} rebalances held in cash")
            out.append(f"    {r.label:<26}{what}")
    return "\n".join(out)


def cash_frontier(base, capital: float, rf: float,
                  weights=None) -> list[Row]:
    """
    The cash blend at a range of weights - the frontier every active arm has
    to beat.

    THE GATE ABOVE HAS AN ARBITRARY THRESHOLD IN IT, and on the calibration
    window that threshold decided the answer: the 70% blend missed the CAGR
    budget by 0.02pp while having the best drawdown-per-point ratio in the
    table. A verdict that turns on where a round number was put is not a
    verdict. This is the same question asked without one - at the drawdown
    each active instrument actually achieved, what does simply holding less
    of the sleeve pay? An instrument is worth its machinery only if it beats
    this line, and the line costs nothing to run and cannot break.
    """
    if weights is None:
        weights = [w / 100.0 for w in range(40, 101, 5)]
    return [_blended(base, capital, w, rf) for w in weights]


def frontier_report(rows: list[Row], base_result, capital: float,
                    rf: float) -> str:
    """Each active arm against the cash blend at ITS OWN drawdown."""
    line = cash_frontier(base_result, capital, rf,
                         [w / 200.0 for w in range(80, 201)])
    out = ["  MATCHED-DRAWDOWN COMPARISON - every active arm against simply",
           "  holding less of the sleeve, at the SAME drawdown:",
           "",
           f"    {'arm':<26}{'maxDD':>8}{'its CAGR':>10}"
           f"{'cash at that DD':>17}{'verdict':>10}"]
    for r in rows:
        if r.key in ("none", "cash70"):
            continue
        nearest = min(line, key=lambda c: abs(c.dd_pct - r.dd_pct))
        beats = r.cagr > nearest.cagr
        out.append(
            f"    {r.label:<26}{r.max_dd:>8.1%}{r.cagr:>+10.2%}"
            f"{nearest.cagr:>+17.2%}{'  BEATS IT' if beats else '  loses':>10}")
    out += ["",
            "    Cash is the null hypothesis for a drawdown instrument the "
            "way random scoring is",
            "    the null for a signal. An arm that does not beat this line "
            "is machinery that",
            "    reproduces what a smaller position size already gives you, "
            "and pays turnover",
            "    for the privilege."]
    return "\n".join(out)


def consistency(base_result, arm_result) -> dict:
    """
    Month by month, did the instrument actually help - or did one stretch?

    THE TABLE ABOVE CANNOT ANSWER THIS, AND ON THE CALIBRATION WINDOW IT GAVE
    THE OPPOSITE ANSWER. Every stop arm beat the matched-drawdown cash line
    there, and the strict -15% arm added 1.64pp of CAGR while cutting 6.3pp of
    drawdown - which reads like a finding. It helped in 55 of 120 months, its
    median monthly difference was 0.000%, and removing its three best months
    turned the mean negative: the whole decade of advantage was 2023.

    So this is the discriminating statistic here, exactly as the fold sign
    test is for the other books. A CAGR difference can be one year. A month
    count cannot.

    `top_k_drop` is the robustness leg: an edge that evaporates when its best
    few months are removed was those months.
    """
    a = np.array([v for _, v in base_result.equity], dtype=float)
    b = np.array([v for _, v in arm_result.equity], dtype=float)
    n = min(len(a), len(b))
    if n < 4:
        return {"n": 0}
    ra = np.diff(a[:n]) / a[:n - 1]
    rb = np.diff(b[:n]) / b[:n - 1]
    delta = rb - ra
    wins = int((delta > 0).sum())
    ties = int((delta == 0).sum())
    scored = len(delta) - ties
    keep = np.sort(delta)[:-3] if len(delta) > 3 else delta
    return {
        "n": len(delta), "wins": wins, "ties": ties, "scored": scored,
        "win_rate": wins / scored if scored else float("nan"),
        "p": two_sided_sign_p(wins, scored),
        "mean_pp": float(np.mean(delta)) * 100.0,
        "median_pp": float(np.median(delta)) * 100.0,
        "compounded_pct": (float(np.prod(1.0 + rb) / np.prod(1.0 + ra)) - 1.0)
        * 100.0,
        "top_k_drop_mean_pp": float(np.mean(keep)) * 100.0,
    }


def consistency_report(results: dict) -> str:
    """`{label: FactorResult}` including `none`, against `none`, month by month."""
    base = results.get("none")
    if base is None:
        return "  (no baseline to compare against)"
    out = ["  MONTH-BY-MONTH CONSISTENCY against the shipped sleeve - the "
           "statistic that decides.",
           "",
           f"    {'arm':<26}{'months won':>12}{'sign p':>9}{'median':>9}"
           f"{'total':>9}{'minus best 3':>14}"]
    for label, res in results.items():
        if label == "none":
            continue
        c = consistency(base, res)
        if not c.get("n"):
            continue
        out.append(
            f"    {label:<26}{c['wins']:>5}/{c['scored']:<6}{c['p']:>9.3f}"
            f"{c['median_pp']:>+9.3f}{c['compounded_pct']:>+9.1f}%"
            f"{c['top_k_drop_mean_pp']:>+13.3f}%")
    out += ["",
            "    'total' compounds the whole window; 'minus best 3' is the "
            "MEAN monthly difference",
            "    with the three best months removed. An arm that is positive "
            "in the first column and",
            "    negative in the last was those three months, and a sign p "
            "near 0.5 says the same",
            "    thing before you look."]
    return "\n".join(out)


#: How much worse than the measured figure to plan for. Two distinct reasons,
#: multiplied rather than argued about: the curve is marked at MONTH-ENDS, so
#: the true intra-month trough is deeper than anything in it; and one decade is
#: one realisation, not a distribution. Neither is a forecast - they are the
#: reason a measured -57% is not a planning number.
DRAWDOWN_HAIRCUT = 1.15


def sizing_report(base_result, capital: float, costs=DEFAULT_EQUITY_COSTS,
                  tolerances=(0.10, 0.15, 0.20, 0.25)) -> str:
    """
    F2b. What share of net worth this sleeve can have, and the floor under it.

    RUIN HERE IS NOT MATHEMATICAL. Long-only, unlevered, twenty names, no
    margin - the sleeve cannot be wiped out, so gambler's-ruin arithmetic is
    the wrong model. The two ways it actually ends are abandonment at the
    trough and needing the money before the recovery, and both are governed by
    the same lever: how much of the net worth is in it.

    POSITION COUNT IS NOT THAT LEVER. At a mean pairwise monthly correlation
    of 0.28 across NSE names, twenty positions are ~3.2 independent bets and
    thirty are ~3.4. Diversification inside the sleeve is spent.

    Everything below is arithmetic on the measured drawdown. Nothing is fitted,
    so there is nothing here to overfit - which is also why the whole
    recommendation moves if the drawdown measured on unseen data is worse.
    """
    series = base_result.series()
    dates = [d for d, _ in base_result.equity]
    years = (dates[-1] - dates[0]).days / 365.25
    measured = abs(base_result.max_drawdown())
    planning = min(0.95, measured * DRAWDOWN_HAIRCUT)
    peak = np.maximum.accumulate(series)
    under = series / peak - 1.0
    sells_per_year = base_result.sells / years if years else 0.0
    dp_per_year = sells_per_year * costs.dp_charge

    out = [
        "F2b - how much of the net worth this sleeve can have",
        "",
        f"  measured max drawdown {measured:.1%} on MONTH-END marks over "
        f"{years:.1f} years;",
        f"  planning figure {planning:.0%} (x{DRAWDOWN_HAIRCUT:.2f}: the true "
        f"intra-month trough is deeper, and one decade is one sample)",
        f"  time underwater: {(under < -0.10).mean():.0%} of marks more than "
        f"10% below peak, {(under < -0.25).sum()} below 25%;",
        f"  longest recovery {base_result.longest_drawdown_months():.0f} months",
        "",
        "  ALLOCATION - the sleeve's share, and what a repeat costs the whole "
        "portfolio:",
        f"    {'you can hold through':<24}{'sleeve share':>14}"
        f"{'at measured':>14}{'at planning':>14}",
    ]
    for tol in tolerances:
        share = tol / planning
        out.append(f"    {tol:<24.0%}{share:>13.0%} {tol * measured / planning:>13.1%}"
                   f"{tol:>14.1%}")
    out += [
        "",
        "  FLOOR, from the one cost that does not shrink with the position:",
        f"    {sells_per_year:.0f} sell legs a year x Rs {costs.dp_charge:.2f} "
        f"DP = Rs {dp_per_year:,.0f} a year, whatever the pot holds",
    ]
    for pot in (200_000.0, 300_000.0, 500_000.0, 1_000_000.0, 2_000_000.0):
        out.append(f"    pot Rs {pot:>11,.0f}   ticket Rs {pot / 20:>9,.0f}"
                   f"   DP alone {dp_per_year / pot * 100.0:>5.2f}%/yr")
    out += [
        "",
        "  The variable charges and slippage are a percentage and do not care "
        "about the pot;",
        "  only this flat one does, which is why there is a floor at all "
        "rather than a taper.",
        "",
        "  SELLS ARE COUNTED, NOT INFERRED FROM TURNOVER. `avg_turnover` is "
        "the fraction of book",
        "  value that changed hands and every rebalance trades BOTH legs, so "
        "reading sells off it",
        "  doubles them and doubles this floor. The reweighting arm really "
        "does sell about twice",
        "  as often, and pays twice this.",
    ]
    return "\n".join(out)


def load(cfg_factor):
    """
    Bars and benchmark, read-only.

    Never through `prices.load_prices`, which re-downloads `history_days` and
    rewrites the parquet on a miss - the same reason `run_s1_swing_null.py`
    reads its cache directly.
    """
    path = Path(cfg_factor.cache_dir) / cfg_factor.cache_name
    if not path.exists():
        raise FileNotFoundError(
            f"No factor cache at {path}. "
            f"Run: python scripts/fetch_factor_history.py --years 10")
    raw = pd.read_parquet(path)
    bars, bench = {}, None
    for symbol, g in raw.groupby("symbol"):
        frame = g.drop(columns=["symbol"]).sort_index()
        if str(symbol).startswith("__"):
            bench = frame
        else:
            bars[str(symbol)] = frame
    return bars, bench


def _main() -> int:                                        # pragma: no cover
    p = argparse.ArgumentParser(description="F2a - drawdown instruments.")
    p.add_argument("--capital", type=float, default=500_000.0)
    p.add_argument("--slippage", type=float, default=0.0025)
    p.add_argument("--rf", type=float, default=DEFAULT_RF)
    p.add_argument("--start", default=None, help="YYYY-MM-DD, scored from")
    p.add_argument("--end", default=None, help="YYYY-MM-DD, scored to")
    p.add_argument("--arms", default="", help="comma-separated subset")
    p.add_argument("--cache", default="",
                   help="read a different parquet from the cache directory - "
                        "how the kill test points at the 2005 fetch without "
                        "disturbing the file the recorded results came from")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT.factor
    if args.cache:
        cfg.cache_name = args.cache
    bars, bench = load(cfg)
    print(f"{len(bars):,} symbols; benchmark "
          f"{'present' if bench is not None else 'MISSING'}", flush=True)
    if bench is None:
        print("Refusing to run: the regime arms fail closed without one, so "
              "two of the eight would silently reproduce the baseline.")
        return 2
    print(fb.CAVEATS, flush=True)

    start = pd.Timestamp(args.start).date() if args.start else None
    end = pd.Timestamp(args.end).date() if args.end else None
    chosen = tuple(arm(k) for k in args.arms.split(",")) if args.arms else ARMS

    started = _clock.time()
    rows, results = run_arms(bars, args.capital, cfg, benchmark=bench,
                             slippage_pct=args.slippage, rf=args.rf,
                             start=start, end=end, arms=chosen,
                             progress=lambda label: print(f"  [{label}]",
                                                          flush=True))
    print(f"\ndone in {_clock.time() - started:.0f}s\n", flush=True)

    base = results["none"]
    marks = [d for d, _ in base.equity]
    bc, bv, bdd = benchmark_stats(bench, marks)
    window = f"{marks[0]} -> {marks[-1]}, {len(marks)} marks"
    print(report(rows, bc, bv, bdd, args.rf, window))
    print()
    print(frontier_report(rows, base, args.capital, args.rf))
    print()
    labelled = {("none" if k == "none" else arm(k).label): v
                for k, v in results.items()}
    print(consistency_report(labelled))
    print()
    print(sizing_report(base, args.capital))
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(_main())
