"""
V0: does the Shariah constraint, on its own, cost return or pay it?

STAGE 0 OF THE VALUE/QUALITY BOOK, AND IT IS DELIBERATELY NOT ABOUT THE BOOK.
No scoring, no screen, no sizing, no costs - two published index pairs, the
constraint being the only difference between the members of each pair. If the
constrained universe lags its parent persistently, no overlay built on top of
it starts from zero, it starts from behind, and that is worth knowing before a
line of strategy code exists.

WHY THE INDEX PAIR IS THE RIGHT NULL HERE. L1 from the sleeve post-mortem: the
null is never zero, it is the cheapest thing you could buy instead. For "should
I run a Shariah book at all", the cheapest alternative is the compliant index
itself - so the question Stage 0 asks is the one that decides whether Stage 1
is worth running, and the answer "buy the index fund" is a real outcome rather
than a failure to find one.

THE STATISTICS ARE THE SLEEVE'S OWN. `drawdown.consistency` gives the monthly
sign test and the drop-the-three-best-months check; CAGR, drawdown and recovery
come off `FactorResult`. The one thing added is `indices.excess_sharpe`,
because `FactorResult.sharpe` subtracts no risk-free rate and this report
promises one.

WHAT THIS CANNOT ANSWER, stated here rather than at the bottom:
  - An index is not a portfolio. These are cap-weighted, rebalanced by the
    administrator, and free of the tax, brokerage and tracking error a holder
    pays. The comparison BETWEEN two of them is clean because both carry the
    same omissions; the level of either is optimistic.
  - Gross TRI on both sides - see `indices.TRI_BASIS_NOTE`.
  - The Shariah indices begin at their 2006-12-29 base date, so this window
    contains one crash (2008) and no other. A 19-year window is one
    realisation, and the sleeve's kill test is the standing reminder of what
    a single window is worth.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifty_algo.factor import drawdown as fd                 # noqa: E402
from nifty_algo.factor import indices as fi                  # noqa: E402

CAPITAL = 100.0          # a rebased index level; nothing here is in rupees


def _fmt_pair_row(label: str, res, capital: float, rf: float,
                  daily: tuple) -> str:
    """One index. Drawdown and recovery are the DAILY figures, not the marks."""
    dd, rec = daily
    return ("    " + label.ljust(20)
            + ("%+.2f%%" % (res.cagr(capital) * 100)).rjust(9)
            + ("%.1f%%" % (fd._annual_vol(res.series()) * 100)).rjust(9)
            + ("%.2f" % fi.excess_sharpe(res, rf)).rjust(9)
            + ("%.1f%%" % (dd * 100)).rjust(10)
            + ("%.0f" % rec).rjust(10)
            + ("%.1f%%" % (res.max_drawdown() * 100)).rjust(11))


def _headline(pair: fi.Pair, rf: float, daily: dict) -> str:
    dc, dp = daily[pair.name], daily[pair.parent_name]
    out = ["  " + pair.name + "  vs  " + pair.parent_name,
           "    " + str(pair.start) + " .. " + str(pair.end)
           + "  (" + ("%.1f" % pair.years) + " years, "
           + str(len(pair.marks)) + " month-end marks)",
           "",
           "    " + "index".ljust(20) + "CAGR".rjust(9) + "vol".rjust(9)
           + "Sharpe".rjust(9) + "maxDD".rjust(10) + "recovery".rjust(10)
           + "(on marks)".rjust(12),
           _fmt_pair_row(pair.parent_name, pair.parent, pair.capital, rf, dp),
           _fmt_pair_row(pair.name, pair.constrained, pair.capital, rf, dc)]
    ex = (pair.constrained.cagr(pair.capital)
          - pair.parent.cagr(pair.capital)) * 100
    dd = (abs(dc[0]) - abs(dp[0])) * 100
    out += ["    " + "excess".ljust(20) + ("%+.2fpp" % ex).rjust(9)
            + "".rjust(9) + "".rjust(9) + ("%+.1fpp" % dd).rjust(10)
            + ("%+.0f" % (dc[1] - dp[1])).rjust(10),
            "",
            "    maxDD and recovery are DAILY. The last column is the SAME "
            "drawdown seen only at",
            "    month-ends - shallower, friendlier, and the grid the sleeve "
            "was sized off."]
    return "\n".join(out)


def _consistency(pair: fi.Pair) -> str:
    """`consistency(base, arm)` -> delta is constrained MINUS parent."""
    c = fd.consistency(pair.parent, pair.constrained)
    if not c.get("n"):
        return "    (too few marks to test)"
    return "\n".join([
        "    months won by " + pair.name + ": "
        + str(c["wins"]) + "/" + str(c["scored"])
        + "   (" + ("%.0f%%" % (c["win_rate"] * 100)) + ")",
        "    two-sided sign p:           " + ("%.4f" % c["p"]),
        "    median monthly difference:  " + ("%+.3f%%" % c["median_pp"]),
        "    compounded over the window: " + ("%+.1f%%" % c["compounded_pct"])
        + "   (wealth ratio, not a difference of returns)",
        "    mean monthly diff:          " + ("%+.3f%%" % c["mean_pp"]),
        "    ... minus its best 3 months:" + ("%+.3f%%"
                                              % c["top_k_drop_mean_pp"]),
    ])


def _rolling(pair: fi.Pair, window: int) -> str:
    roll = fi.rolling_excess(pair, window_months=window)
    if roll.empty:
        return ("    (window shorter than " + str(window)
                + " months - no rolling periods)")
    ex = roll["excess_pp"].to_numpy()
    share = float((ex > 0).mean())
    lines = [
        "    " + str(len(ex)) + " overlapping " + str(window // 12)
        + "-year windows, " + str(int((ex > 0).sum())) + " positive ("
        + ("%.0f%%" % (share * 100)) + ")",
        "    excess pp:  min " + ("%+.2f" % ex.min())
        + "   p25 " + ("%+.2f" % np.percentile(ex, 25))
        + "   median " + ("%+.2f" % np.median(ex))
        + "   p75 " + ("%+.2f" % np.percentile(ex, 75))
        + "   max " + ("%+.2f" % ex.max()),
        "",
        "    worst window : " + str(roll["start"].iloc[int(ex.argmin())])
        + " .. " + str(roll["end"].iloc[int(ex.argmin())]) + "   "
        + ("%+.2fpp" % ex.min()),
        "    best window  : " + str(roll["start"].iloc[int(ex.argmax())])
        + " .. " + str(roll["end"].iloc[int(ex.argmax())]) + "   "
        + ("%+.2fpp" % ex.max()),
    ]
    # The overlap is the reason not to put a p-value on this line.
    lines += ["",
              "    These windows OVERLAP by construction - adjacent ones share "
              "59 of 60 months -",
              "    so the count of positive windows is a shape, not a sample "
              "size, and no sign test",
              "    is run on it. The monthly sign test above is the one with "
              "independent draws."]
    return "\n".join(lines)


#: Calendar five-year blocks plus the two halves.
#:
#: THESE WERE NOT PRE-REGISTERED, and saying so is the point. They were run
#: once, ad hoc, and then written down here - so they are a DESCRIPTION of
#: where the excess sat, not a test that the excess replicates. The boundaries
#: are round calendar dates rather than chosen ones, which is the only reason
#: this is worth printing at all; had they been nudged to make a block look
#: better it would be exactly the fitting the sleeve's kill test exists to
#: catch, done by eye. A real out-of-sample test on this data is Stage 2's job,
#: on a window the user picks before anyone looks at it.
SUBPERIODS = (
    ("2006-12-29", "2016-12-31", "first half"),
    ("2016-12-31", "2026-09-04", "second half"),
    ("2006-12-29", "2011-12-31", "2007-2011"),
    ("2011-12-31", "2016-12-31", "2012-2016"),
    ("2016-12-31", "2021-12-31", "2017-2021"),
    ("2021-12-31", "2026-09-04", "2022-2026"),
)


def _subperiods(series: dict, name: str, parent_name: str) -> str:
    """
    DOES THE EXCESS REPLICATE? The one question a pooled CAGR cannot answer.

    L2 from the sleeve post-mortem: pre-register the falsification quantity,
    not just the test. A whole-window excess is an average, and the sleeve's
    +7.94pp was an average that fell to +0.60pp the moment it was asked to
    hold up somewhere else. So the window is split before the verdict is read.
    """
    out = ["    " + "window".ljust(14) + "constrained".rjust(12)
           + "parent".rjust(10) + "excess".rjust(11) + "months won".rjust(13)]
    for start_s, end_s, label in SUBPERIODS:
        p = fi.pair(series, name, parent_name, capital=CAPITAL,
                    start=date.fromisoformat(start_s),
                    end=date.fromisoformat(end_s))
        ca = p.constrained.cagr(p.capital)
        cb = p.parent.cagr(p.capital)
        c = fd.consistency(p.parent, p.constrained)
        out.append("    " + label.ljust(14)
                   + ("%+.2f%%" % (ca * 100)).rjust(12)
                   + ("%+.2f%%" % (cb * 100)).rjust(10)
                   + ("%+.2fpp" % ((ca - cb) * 100)).rjust(11)
                   + (str(c["wins"]) + "/" + str(c["scored"])).rjust(13))
        if label == "second half":
            out.append("")
    return "\n".join(out)


def _verdict(pairs: list) -> str:
    out = ["  WHAT THIS SAYS ABOUT WHETHER TO CONTINUE", ""]
    for p in pairs:
        c = fd.consistency(p.parent, p.constrained)
        ex = (p.constrained.cagr(p.capital) - p.parent.cagr(p.capital)) * 100
        roll = fi.rolling_excess(p, 60)
        share = float((roll["excess_pp"] > 0).mean()) if not roll.empty else float("nan")
        drift = ("pays" if ex > 0 else "costs")
        out.append("    " + p.name + ": the constraint " + drift + " "
                   + ("%.2fpp" % abs(ex)) + " a year against "
                   + p.parent_name + ",")
        out.append("      wins " + ("%.0f%%" % (c["win_rate"] * 100))
                   + " of months (sign p " + ("%.3f" % c["p"]) + ") and "
                   + ("%.0f%%" % (share * 100))
                   + " of rolling 5-year windows.")
        out.append("")
    return "\n".join(out)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V0: the cost of the constraint.")
    ap.add_argument("--cache", default=str(fi.DEFAULT_PATH))
    ap.add_argument("--rf", type=float, default=fi.DEFAULT_RF)
    ap.add_argument("--window", type=int, default=60,
                    help="rolling excess window, in months")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args(argv)

    series = fi.load(args.cache)
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    print("=" * 78)
    print("V0 - DOES THE SHARIAH CONSTRAINT COST RETURN OR PAY IT?")
    print("=" * 78)
    print("  Source: niftyindices.com (the index administrator), gross TRI.")
    print("  Risk-free for Sharpe: " + ("%.2f%%" % (args.rf * 100))
          + "  (stated, not measured - VERIFY)")
    print()
    print("  " + fi.TRI_BASIS_NOTE.replace("\n", "\n  "))
    print()

    pairs = []
    for name, parent_name in fi.PARENT_OF.items():
        p = fi.pair(series, name, parent_name, capital=CAPITAL,
                    start=start, end=end)
        # Daily, over the PAIR's window, so both sides see the same sessions.
        daily = {n: fi.daily_drawdown(series[n], start=p.start, end=p.end)
                 for n in (name, parent_name)}
        pairs.append(p)
        print("-" * 78)
        print(_headline(p, args.rf, daily))
        print()
        print("  MONTH-BY-MONTH - the statistic that decides")
        print(_consistency(p))
        print()
        print("  ROLLING " + str(args.window // 12) + "-YEAR EXCESS")
        print(_rolling(p, args.window))
        print()
        print("  WHERE THE EXCESS SAT - calendar blocks, NOT pre-registered")
        print(_subperiods(series, name, parent_name))
        print()

    print("=" * 78)
    print(_verdict(pairs))
    print("  LIMITS, so no number above is read without them:")
    print("    - THE 2022-2026 BLOCK IS THE ONE TO ARGUE WITH. Both pairs")
    print("      turn sharply negative there, and the obvious explanation -")
    print("      Indian financials led that rally and the screen excludes")
    print("      them - is a CONJECTURE. Nothing in this data tests it: an")
    print("      index level carries no sector attribution. Treat it as the")
    print("      hypothesis to check, not as the reason.")
    print("    - An index is not a portfolio: no tax, no brokerage, no")
    print("      tracking error, and the administrator rebalances for free.")
    print("      The comparison between two of them is clean because both")
    print("      carry the same omissions; the LEVEL of either is optimistic.")
    print("    - " + ("%.0f" % pairs[0].years) + " years is one realisation")
    print("      containing one crash. The sleeve's +7.94pp became +0.60pp")
    print("      on a window it had not seen.")
    print("    - Both Shariah indices start at their 2006-12-29 base date,")
    print("      so nothing here says anything about 1995-2006.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
