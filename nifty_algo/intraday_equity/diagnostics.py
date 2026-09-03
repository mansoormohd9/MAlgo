"""
Why did the book lose? Descriptive statistics over a finished trade list.

The headline - "-0.524R over 2,162 trades" - is a verdict, not a diagnosis. It
cannot distinguish the two explanations that call for opposite responses:

  THE MOVE NEVER CAME. Entries are not predictive; the trades went nowhere and
  died at their stops. Nothing about holding period, exit rungs or the
  square-off matters, and the answer is that these setups do not work on
  5-minute equity bars.

  THE MOVE CAME AND WE DID NOT KEEP IT. Trades reached +1R or +1.5R and were
  then handed back by a 15:10 square-off that fires before the +2R rung. The
  entries are fine and the exit economics are wrong for the holding period.

`capture_ratio` and `mfe_attainment` separate those two, which is the whole
reason `IntradayTrade` now carries `mfe_r` / `mae_r`.

EVERY FUNCTION HERE IS PURE AND READS A FINISHED TRADE LIST. Nothing imports
the scanner, the ladder or the config, so a diagnostic cannot influence a
decision even by accident - and each is testable without a backtest. That
separation is the point: these are instruments, and an instrument that can
change the thing it measures is not one.

THEY DESCRIBE, THEY DO NOT SELECT. Nothing here should be used to choose a
subset of strategies, symbols or hours to keep. That is the fitting the
option-book chart invited and it is named in the plan's overfitting section.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: The rungs the ladder cares about, so attainment is reported against the
#: design rather than against arbitrary round numbers.
DEFAULT_RUNGS = (0.5, 1.0, 1.5, 2.0, 3.0)

#: Attainment thresholds in PERCENT OF ENTRY PRICE, for comparing runs that
#: use different stop multiples. See `attainment_pct`.
FIXED_PCT_RUNGS = (0.0025, 0.005, 0.010, 0.015)

#: Forward horizons in BARS. 20 bars is 100 minutes, about half the tradeable
#: window, and past that most entries have hit the square-off anyway.
DEFAULT_HORIZONS = (1, 2, 4, 8, 12, 20)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class Bucket:
    """One row of a census."""
    label: str
    n: int = 0
    total_r: float = 0.0
    bars: int = 0
    mfe: float = 0.0
    wins: int = 0

    # NOTE there is deliberately no `share` property: a bucket does not know
    # the population it came from, and one that returned a plausible 0.0
    # would be read as "0% of trades" rather than "not computed".

    @property
    def mean_r(self) -> float:
        return self.total_r / self.n if self.n else 0.0

    @property
    def mean_bars(self) -> float:
        return self.bars / self.n if self.n else 0.0

    @property
    def mean_mfe(self) -> float:
        return self.mfe / self.n if self.n else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0


def _census(trades, key) -> list[Bucket]:
    out: dict[str, Bucket] = {}
    for t in trades:
        b = out.setdefault(str(key(t)), Bucket(label=str(key(t))))
        b.n += 1
        b.total_r += t.r_multiple
        b.bars += t.bars_held
        b.mfe += getattr(t, "mfe_r", 0.0)
        if t.r_multiple > 0:
            b.wins += 1
    return sorted(out.values(), key=lambda b: -b.n)


def exit_census(trades) -> list[Bucket]:
    """
    D1 + D2. Count, share, mean R, mean bars and TOTAL R by exit reason.

    Total R is carried alongside the mean because they answer different
    questions: the mean says which exit is worst per trade, the total says
    which one actually spent the money. A rare, terrible exit and a common,
    mildly bad one look the same in a mean.
    """
    return _census(trades, lambda t: t.reason or t.outcome)


def strategy_census(trades) -> list[Bucket]:
    return _census(trades, lambda t: t.strategy)


def strategy_by_reason(trades) -> dict:
    """
    D5. `{strategy: {reason: Bucket}}`.

    Answers whether four setups are failing the same way or four different
    ways. Identical exit profiles across strategies point at a common factor
    - friction, holding period - rather than at four separate defects, and
    that distinction decides whether per-strategy work is worth anything.
    """
    out: dict = defaultdict(dict)
    for t in trades:
        reason = t.reason or t.outcome
        b = out[t.strategy].setdefault(reason, Bucket(label=reason))
        b.n += 1
        b.total_r += t.r_multiple
        b.bars += t.bars_held
        b.mfe += getattr(t, "mfe_r", 0.0)
        if t.r_multiple > 0:
            b.wins += 1
    return dict(out)


def time_census(trades, bucket_minutes: int = 30) -> list[Bucket]:
    """
    D3. Expectancy and MFE by entry clock time.

    Supports the late-start hypothesis only if BOTH expectancy and mean MFE
    decline as entry time advances - a trade entered at 14:25 has 45 minutes
    before the square-off and one entered at 11:45 has 205. If expectancy is
    flat across the session then time is not the binding constraint, and
    building a warm-up seeding mechanism would be machinery on a hunch.
    """
    def key(t):
        ts = t.entry_time
        minute = (ts.hour * 60 + ts.minute) // bucket_minutes * bucket_minutes
        return f"{minute // 60:02d}:{minute % 60:02d}"

    return sorted(_census(trades, key), key=lambda b: b.label)


def mfe_attainment(trades, rungs=DEFAULT_RUNGS) -> dict:
    """
    D6. What fraction of trades ever reached +0.5R, +1R, +1.5R, +2R, +3R.

    The single most diagnostic table available. The ladder shifts to breakeven
    at +1R and banks half at +2R, so attainment at those two rungs says
    directly whether the ladder's rungs are reachable inside a session that
    ends at 15:10.
    """
    n = len(trades)
    if not n:
        return {r: 0.0 for r in rungs}
    return {r: sum(1 for t in trades if getattr(t, "mfe_r", 0.0) >= r) / n
            for r in rungs}


def capture_ratio(trades) -> float:
    """
    D4. Realised R as a fraction of the R that was on the table.

    `sum(r_multiple) / sum(mfe_r)`. Near 1.0 means the book kept what it was
    offered and the entries are simply wrong. Well below it - or negative -
    means the moves happened and the exits gave them back.

    Returns 0.0 when nothing was ever offered, which is itself the finding.
    """
    offered = sum(max(0.0, getattr(t, "mfe_r", 0.0)) for t in trades)
    if offered <= 0:
        return 0.0
    return sum(t.r_multiple for t in trades) / offered


def squareoff_pessimism(trades, target_r: float = 2.0) -> dict:
    """
    D7. Force-exits whose own excursion had already reached the target.

    `_manage` tests the square-off BEFORE `ladder.advance()`, so a bar that
    touches +2R on the 15:10 bar exits at that bar's close rather than banking
    the rung. That is deliberate pessimism; this counts what it costs.

    A large number here is a finding about the RULE, not about the market, and
    would justify reordering those two checks - which is a rule change and
    belongs in an experiment, not in a diagnostic.

    `partial_banked` IS THE WHOLE CORRECTION. A first version counted every
    force-exit whose MFE reached the target and reported 79 trades - but 254
    of the 266 trades that ever touched +2R had already BANKED there, and the
    force-exit was of the runner afterwards. That is the ladder working, not
    the rule costing anything. Excluding banked runners takes the real figure
    from 79 to 11. A diagnostic that overstates by 7x is worse than none.
    """
    forced = [t for t in trades if (t.reason or t.outcome) == "force_exit"]
    reached = [t for t in forced
               if getattr(t, "mfe_r", 0.0) >= target_r
               and not getattr(t, "partial_banked", False)]
    return {
        "force_exits": len(forced),
        "reached_target_anyway": len(reached),
        "share": len(reached) / len(forced) if forced else 0.0,
        "mean_r_of_those": _mean(t.r_multiple for t in reached),
    }


def summary(trades) -> str:
    """Every diagnostic as one printable block."""
    if not trades:
        return "no trades"
    n = len(trades)
    lines = [f"{n} trades", "", "exit census (D1/D2):",
             f"  {'reason':<14}{'n':>6}{'share':>8}{'mean R':>9}"
             f"{'total R':>10}{'bars':>7}{'mean MFE':>10}"]
    for b in exit_census(trades):
        lines.append(f"  {b.label:<14}{b.n:>6}{b.n/n:>8.1%}{b.mean_r:>9.3f}"
                     f"{b.total_r:>10.1f}{b.mean_bars:>7.1f}{b.mean_mfe:>10.2f}")

    lines += ["", "MFE attainment (D6) - how far trades ever got:"]
    for rung, share in mfe_attainment(trades).items():
        lines.append(f"  reached +{rung:.1f}R{share:>8.1%}")

    lines += ["", f"capture ratio (D4): {capture_ratio(trades):+.3f}"
                  "   (realised R / R offered)"]

    sp = squareoff_pessimism(trades)
    lines += ["", f"square-off pessimism (D7): {sp['reached_target_anyway']} "
                  f"of {sp['force_exits']} force-exits ({sp['share']:.1%}) had "
                  f"already touched +2R"]

    lines += ["", "entry time (D3):",
              f"  {'bucket':<10}{'n':>6}{'mean R':>9}{'mean MFE':>10}{'win':>7}"]
    for b in time_census(trades):
        lines.append(f"  {b.label:<10}{b.n:>6}{b.mean_r:>9.3f}"
                     f"{b.mean_mfe:>10.2f}{b.win_rate:>7.1%}")

    lines += ["", "strategy x reason (D5):"]
    for strat, by_reason in sorted(strategy_by_reason(trades).items()):
        total = sum(b.n for b in by_reason.values())
        parts = ", ".join(f"{r} {b.n/total:.0%}"
                          for r, b in sorted(by_reason.items(),
                                             key=lambda kv: -kv[1].n))
        lines.append(f"  {strat:<18} n={total:<6} {parts}")
    return "\n".join(lines)


# ------------------------------------------------- stop-invariant units
#
# R IS NOT COMPARABLE ACROSS STOP MULTIPLES, and comparing stop multiples is
# exactly what the E4 experiment does. Halving `atr_stop_multiple` halves
# `risk_points`, so "+1R" becomes half the price move: attainment at every R
# rung rises, the win rate moves because the barrier moved, and friction
# measured in R roughly doubles - all with no change whatever in what price
# did after the entry.
#
# Everything below is expressed as a fraction of the ENTRY PRICE, which does
# not move when the stop does. `mfe_r` is R off the initial risk and
# `stop_points` is that risk in rupees, so the conversion is exact rather
# than approximate, and needs no new instrumentation.


def mfe_pct(t) -> float:
    """Maximum favourable excursion as a fraction of the entry price."""
    entry = float(getattr(t, "entry_underlying", 0.0) or 0.0)
    if entry <= 0:
        return 0.0
    return float(getattr(t, "mfe_r", 0.0)) * float(t.stop_points) / entry


def mae_pct(t) -> float:
    """Maximum adverse excursion as a fraction of the entry price."""
    entry = float(getattr(t, "entry_underlying", 0.0) or 0.0)
    if entry <= 0:
        return 0.0
    return float(getattr(t, "mae_r", 0.0)) * float(t.stop_points) / entry


def stop_pct(t) -> float:
    entry = float(getattr(t, "entry_underlying", 0.0) or 0.0)
    return float(t.stop_points) / entry if entry > 0 else 0.0


def attainment_pct(trades, rungs=FIXED_PCT_RUNGS) -> dict:
    """
    How often price ran a fixed PERCENTAGE in favour, whatever the stop was.

    The stop-invariant twin of `mfe_attainment`. If two stop multiples produce
    the same numbers here but different R-rung attainment, the R difference is
    the denominator and nothing else.
    """
    n = len(trades)
    if not n:
        return {r: 0.0 for r in rungs}
    vals = [mfe_pct(t) for t in trades]
    return {r: sum(1 for v in vals if v >= r) / n for r in rungs}


def _median(xs) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def invariant_summary(trades) -> dict:
    """
    The block that may be compared across stop multiples, and nothing else.

    `net_pnl` is included because the risk budget is a constant rupee amount,
    so total rupees compare directly - PROVIDED sizing stayed risk-bound. A
    variant that goes capital-bound risks less per trade, and then neither
    rupees nor R mean what they did; `median_risk_inr` is reported so that
    condition is checkable rather than assumed.
    """
    if not trades:
        return {}
    risks = [float(t.stop_points) * float(t.quantity) for t in trades]
    return {
        "n": len(trades),
        "net_inr": sum(float(t.net_pnl) for t in trades),
        "net_inr_per_trade": _mean(float(t.net_pnl) for t in trades),
        "median_stop_pct": _median(stop_pct(t) for t in trades),
        "median_mfe_pct": _median(mfe_pct(t) for t in trades),
        "median_mae_pct": _median(mae_pct(t) for t in trades),
        "mean_mfe_pct": _mean(mfe_pct(t) for t in trades),
        "mean_mae_pct": _mean(mae_pct(t) for t in trades),
        "median_risk_inr": _median(risks),
        "attainment_pct": attainment_pct(trades),
    }


def overlap(baseline, variant) -> dict:
    """
    Trades both runs took, matched on (symbol, entry bar).

    THE CLEANEST COMPARISON AVAILABLE. On a matched trade the price path is
    identical by construction, so MFE% and MAE% are identical too and every
    difference in R or in rupees comes from stop placement alone - selection
    removed. A variant that looks better only on its own trade population
    changed WHICH signals qualified, which is a different claim from "the
    stop was better placed", and the two are worth separating.
    """
    def key(t):
        return (t.symbol, pd.Timestamp(t.entry_time))

    b = {key(t): t for t in baseline}
    v = {key(t): t for t in variant}
    shared = sorted(set(b) & set(v), key=lambda k: (k[1], k[0]))
    if not shared:
        return {"n": 0}
    return {
        "n": len(shared),
        "share_of_baseline": len(shared) / len(baseline) if baseline else 0.0,
        "share_of_variant": len(shared) / len(variant) if variant else 0.0,
        "baseline_r": _mean(b[k].r_multiple for k in shared),
        "variant_r": _mean(v[k].r_multiple for k in shared),
        "baseline_inr": sum(float(b[k].net_pnl) for k in shared),
        "variant_inr": sum(float(v[k].net_pnl) for k in shared),
    }


# ------------------------------------------------- D8: the entry itself


def _session_matrix(frame, horizons):
    """
    `{horizon: DataFrame[date x clock-time]}` of forward returns.

    Pivoting to (date x clock) makes "h bars later in the same session" a
    column shift, which is both fast and structurally correct: shifting past
    the last column yields NaN rather than silently reaching into the next
    session. A positional shift on the raw intraday frame would happily
    return tomorrow morning's price for a trade entered near the close.
    """
    close = frame["close"]
    flat = pd.DataFrame({
        "close": close.to_numpy(dtype=float),
        "d": frame.index.date,
        "t": frame.index.time,
    })
    mat = flat.pivot_table(index="d", columns="t", values="close",
                           aggfunc="last")
    return {h: mat.shift(-h, axis=1) / mat - 1.0 for h in horizons}


def forward_returns(trades, bars, horizons=DEFAULT_HORIZONS, benchmark=None):
    """
    D8. What price did after each entry - no stop, no target, no ladder.

    This is the only measurement in the package that isolates the ENTRY. Every
    other number is entangled with where the stop was, when the square-off
    fired and how the ladder behaved; this one asks solely whether price went
    up after the book bought.

    TWO CONTROLS, because a raw positive number would prove nothing:

      BENCHMARK. A long-only book in a rising market collects drift and
      mistakes it for skill. Subtracting NIFTY's move over the identical bar
      window removes it.

      TIME OF DAY, leave-one-out. The same symbol's average move over the same
      clock window on all OTHER sessions. Removes intraday seasonality - and
      leave-one-out because including the trade's own session in its own
      control shrinks the very excess being measured.

    Returns `{horizon: {...}}` with the raw mean, both excesses, and a 95%
    interval on the benchmark excess. An excess inside its interval is
    reported as such rather than rounded into a conclusion.
    """
    out: dict = {}
    if not trades:
        return out

    by_symbol: dict = {}
    for symbol in {t.symbol for t in trades}:
        frame = bars.get(symbol)
        if frame is not None and len(frame):
            by_symbol[symbol] = _session_matrix(frame, horizons)
    bench = (_session_matrix(benchmark, horizons)
             if benchmark is not None and len(benchmark) else None)

    for h in horizons:
        raw, ex_bench, ex_tod = [], [], []
        for t in trades:
            mats = by_symbol.get(t.symbol)
            if mats is None:
                continue
            ts = pd.Timestamp(t.entry_time)
            d, clock = ts.date(), ts.time()
            fwd = mats[h]
            if d not in fwd.index or clock not in fwd.columns:
                continue
            r = fwd.at[d, clock]
            if not np.isfinite(r):
                continue
            raw.append(float(r))

            if bench is not None:
                bf = bench[h]
                if d in bf.index and clock in bf.columns:
                    br = bf.at[d, clock]
                    if np.isfinite(br):
                        ex_bench.append(float(r) - float(br))

            col = fwd[clock].to_numpy(dtype=float)
            col = col[np.isfinite(col)]
            if len(col) > 1:
                # Leave-one-out: this trade's own session must not sit inside
                # the control it is being measured against.
                loo = (col.sum() - float(r)) / (len(col) - 1)
                ex_tod.append(float(r) - loo)

        n = len(raw)
        se = (float(np.std(ex_bench, ddof=1)) / np.sqrt(len(ex_bench))
              if len(ex_bench) > 1 else 0.0)
        mean_b = _mean(ex_bench)
        out[h] = {
            "n": n,
            "raw_mean": _mean(raw),
            "raw_median": _median(raw),
            "excess_bench_mean": mean_b,
            "excess_bench_ci95": (mean_b - 1.96 * se, mean_b + 1.96 * se),
            "excess_tod_mean": _mean(ex_tod),
            "n_bench": len(ex_bench),
            "n_tod": len(ex_tod),
        }
    return out


def forward_returns_table(fr: dict) -> str:
    """`forward_returns` as a readable block, with the verdict per horizon."""
    if not fr:
        return "no forward returns"
    lines = [f"  {'bars':>5}{'n':>7}{'raw %':>9}{'vs bench %':>12}"
             f"{'95% CI':>22}{'vs t-o-d %':>12}  verdict"]
    for h, d in sorted(fr.items()):
        lo, hi = d["excess_bench_ci95"]
        verdict = ("POSITIVE" if lo > 0 else
                   "NEGATIVE" if hi < 0 else "inside CI")
        lines.append(
            f"  {h:>5}{d['n']:>7}{d['raw_mean']*100:>9.4f}"
            f"{d['excess_bench_mean']*100:>12.4f}"
            f"   [{lo*100:>+7.4f},{hi*100:>+7.4f}]"
            f"{d['excess_tod_mean']*100:>12.4f}  {verdict}")
    return "\n".join(lines)


__all__ = ["Bucket", "exit_census", "strategy_census", "strategy_by_reason",
           "time_census", "mfe_attainment", "capture_ratio",
           "squareoff_pessimism", "summary", "DEFAULT_RUNGS",
           "FIXED_PCT_RUNGS", "DEFAULT_HORIZONS",
           "mfe_pct", "mae_pct", "stop_pct", "attainment_pct",
           "invariant_summary", "overlap",
           "forward_returns", "forward_returns_table"]
