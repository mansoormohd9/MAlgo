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

#: The rungs the ladder cares about, so attainment is reported against the
#: design rather than against arbitrary round numbers.
DEFAULT_RUNGS = (0.5, 1.0, 1.5, 2.0, 3.0)


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


__all__ = ["Bucket", "exit_census", "strategy_census", "strategy_by_reason",
           "time_census", "mfe_attainment", "capture_ratio",
           "squareoff_pessimism", "summary", "DEFAULT_RUNGS"]
