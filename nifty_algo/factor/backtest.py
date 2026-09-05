"""
Does the factor sleeve work? A monthly rebalance, priced honestly.

WHY THIS IS NOT JUDGED IN R

The other three books are. This one deliberately is not, and that is a
statement about what it measures rather than a convenience. R is a multiple of
a per-trade stop, and this book has no per-trade stop - it rebalances. Forcing
it into R would let its numbers sit in the same table as the intraday book's
while measuring a different thing, and someone would eventually compare them.
CAGR, Sharpe, max drawdown, time-to-recover and turnover instead.

THE PESSIMISM RULES

  RANK ON `day`, FILL ON `day`'s CLOSE. Formation reads bars strictly before
  the rebalance date; the trade fills at that date's close. Ranking on a close
  and filling at the same close would be a look-ahead of one bar, which on a
  monthly book is small and free to avoid.

  A NAME WITH NO BAR ON THE REBALANCE DATE IS NOT TRADED. Not carried, not
  filled at a stale price. It simply does not enter, and the slot goes unused
  rather than being handed to the next candidate - filling from a deeper rank
  is a decision the live book could not have made either.

  COSTS ARE CHARGED PER LEG, BOTH SIDES, FROM `swing/costs_equity.py`. That is
  the delivery rate card: STT on BOTH legs, and the flat DP charge per scrip
  per sell that does not shrink with position size. On a small position the
  flat charge dominates, which is exactly the regime a Rs 1 lakh pot trades in.

  EQUAL WEIGHT, NOT SCORE WEIGHT. Weighting by signal strength is a second
  free parameter and the literature's standard construction is equal weight.

WHAT IT CANNOT SEE, printed above every result and repeated here so it cannot
be lost: survivorship (Kite's dump is today's listed set, and for a long-only
momentum book that bias runs one way), and no point-in-time fundamentals, so
no halal screen is replayed - the same limit the swing backtest carries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..swing.costs_equity import DEFAULT_EQUITY_COSTS
from . import momentum as mom
from .universe import FactorUniverse, month_ends

CAVEATS = """\
WHAT THIS BACKTEST CANNOT SEE
  1. SURVIVORSHIP, AND IT RUNS ONE WAY. The symbol list is TODAY'S listed set,
     so companies delisted inside the window are absent from the universe AND
     from the history. Momentum's mechanism is losers continuing to lose, and
     the worst losers are the names that stopped being listed, so this
     flatters a long-only momentum book by an unknown amount. Re-run with
     `listed_only=True` for a lower bound on the inflation.
  2. NO HALAL SCREEN. No point-in-time fundamentals exist, so the screen
     cannot be replayed. This tests a LARGER universe than a live book trades.
  3. NO CORPORATE-ACTION ADJUSTMENT beyond what the feed supplies. A split
     shows up as a return, and the return filter below drops the session
     rather than trading it.
  4. PUBLISHED MOMENTUM FIGURES ARE NOT A BENCHMARK FOR THIS. Post-publication
     returns run about half of in-sample ones, so a result matching a paper is
     evidence of a bug or of survivorship before it is evidence of an edge.
"""

#: A one-session move beyond this is treated as a corporate action rather than
#: a return. Unadjusted bars make a 1:10 split look like a -90% day, and a
#: long-only book would otherwise buy the "crash".
MAX_SESSION_MOVE = 0.5


@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_day: date


@dataclass
class FactorResult:
    equity: list = field(default_factory=list)          # (date, value)
    holdings_log: list = field(default_factory=list)
    rebalances: int = 0
    trades: int = 0
    costs_paid: float = 0.0
    turnover_sum: float = 0.0
    skipped_no_bar: int = 0
    skipped_corp_action: int = 0
    rejections: dict = field(default_factory=dict)
    start: date | None = None
    end: date | None = None
    universe_size: list = field(default_factory=list)
    #: {symbol: net rupees contributed}. Cash out minus cash in, plus what is
    #: still held at the final mark. Exists to answer ONE question that the
    #: headline CAGR cannot: how much of the result is one or two names.
    contribution: dict = field(default_factory=dict)

    @property
    def final_value(self) -> float:
        return self.equity[-1][1] if self.equity else 0.0

    def series(self) -> np.ndarray:
        return np.array([v for _, v in self.equity], dtype=float)

    def cagr(self, starting: float) -> float:
        if len(self.equity) < 2 or starting <= 0:
            return 0.0
        years = (self.equity[-1][0] - self.equity[0][0]).days / 365.25
        if years <= 0 or self.final_value <= 0:
            return 0.0
        return (self.final_value / starting) ** (1 / years) - 1.0

    def max_drawdown(self) -> float:
        s = self.series()
        if len(s) < 2:
            return 0.0
        peak = np.maximum.accumulate(s)
        return float(np.min(s / peak - 1.0))

    def longest_drawdown_months(self) -> float:
        """
        Time from a peak to recovering it - the number that decides whether a
        strategy is actually holdable. A 14% CAGR through a 65-month recovery
        is a different product from the same CAGR without one.
        """
        s, dates = self.series(), [d for d, _ in self.equity]
        if len(s) < 2:
            return 0.0
        peak, peak_at, worst = s[0], dates[0], 0.0
        for value, when in zip(s, dates):
            if value >= peak:
                worst = max(worst, (when - peak_at).days / 30.44)
                peak, peak_at = value, when
        worst = max(worst, (dates[-1] - peak_at).days / 30.44)
        return worst

    def sharpe(self, periods_per_year: float = 12.0) -> float:
        s = self.series()
        if len(s) < 3:
            return 0.0
        r = np.diff(s) / s[:-1]
        sd = float(np.std(r, ddof=1))
        if sd <= 0:
            return 0.0
        return float(np.mean(r)) / sd * math.sqrt(periods_per_year)

    def avg_turnover(self) -> float:
        return self.turnover_sum / self.rebalances if self.rebalances else 0.0

    def concentration(self, top: int = 3) -> dict:
        """
        How much of the net result came from the biggest contributors.

        A momentum sleeve that never trims can have its whole terminal wealth
        carried by a couple of positions, and that is a different product from
        one whose edge is spread across the book - same CAGR, entirely
        different capacity and entirely different odds of repeating.
        """
        if not self.contribution:
            return {}
        vals = sorted(self.contribution.values(), reverse=True)
        total = sum(vals)
        gains = sum(v for v in vals if v > 0)
        return {
            "names": len(vals),
            "top1_share_of_gains": (vals[0] / gains) if gains > 0 else 0.0,
            "topn_share_of_gains": (sum(vals[:top]) / gains)
            if gains > 0 else 0.0,
            "net_total": total,
        }

    def headline(self, starting: float) -> str:
        return (f"CAGR {self.cagr(starting):+.2%}  "
                f"Sharpe {self.sharpe():.2f}  "
                f"maxDD {self.max_drawdown():.1%}  "
                f"longest recovery {self.longest_drawdown_months():.0f} months  "
                f"turnover {self.avg_turnover():.0%}/rebalance  "
                f"costs Rs {self.costs_paid:,.0f}")


def _tradeable_price(universe: FactorUniverse, symbol: str, day: date,
                     result: FactorResult) -> float | None:
    """The close ON `day`, rejected if it implies a corporate action."""
    price = universe.price_on(symbol, day)
    if price is None or price <= 0:
        result.skipped_no_bar += 1
        return None
    prior = universe.price_at(symbol, day)
    if prior and prior > 0:
        if abs(price / prior - 1.0) > MAX_SESSION_MOVE:
            result.skipped_corp_action += 1
            return None
    return price


def _mark(universe: FactorUniverse, symbol: str, day: date,
          pos: "Position") -> float:
    """
    What a held name is worth on `day`.

    THE FALLBACK ORDER IS THE POINT. Today's close if it traded; otherwise its
    LAST KNOWN close; and only if neither exists, the entry price.

    Falling straight back to the entry price - as this did - values a holding
    that stopped trading at whatever we paid for it, forever. A name bought at
    100 that slid to 10 and then went quiet would be carried at 100, which is
    a private little survivorship bias inside a book already fighting a real
    one. It fires on 0.21% of valuations in the whole universe and 0.72% in
    the illiquid band, so it does not explain the headline - but it runs in
    the flattering direction, which is reason enough.
    """
    price = universe.price_on(symbol, day)
    if price is not None and price > 0:
        return price
    last = universe.price_at(symbol, day)
    if last is not None and last > 0:
        return last
    return pos.entry_price


def _charge(costs, price: float, shares: float, sell: bool,
            slippage_pct: float) -> float:
    """
    One leg's total cost: the statutory rate card plus slippage.

    `run` used to call `costs.buy_cost`/`sell_cost` directly, and those two
    deliberately EXCLUDE slippage - `EquityCostModel.friction` is the method
    that includes it. So every fill in every factor result before this landed
    happened at the close for free. It is not a small omission on a book that
    turns 63% of itself a month, and it is not a NEUTRAL one: the random null
    turns 1.78x per rebalance, so free fills subsidise the null nearly three
    times as much as they subsidise momentum.
    """
    leg = (costs.sell_cost(price, shares) if sell
           else costs.buy_cost(price, shares))
    return leg + price * shares * slippage_pct


def run(bars: dict, starting_capital: float, top_n: int = 20,
        band: str = "all", formation: str = "mom12_1",
        hold_months: int = 1, min_price: float = 20.0,
        min_turnover: float = 1.0e7, min_history: int = 300,
        costs=DEFAULT_EQUITY_COSTS, listed_only: bool = False,
        seed: int | None = None, start: date | None = None,
        end: date | None = None, progress=None,
        universe: FactorUniverse | None = None,
        listed_before: date | None = None,
        reweight: bool = False,
        slippage_pct: float = 0.0) -> FactorResult:
    """
    Replay the sleeve month by month.

    `seed` switches the signal to `momentum.random_scores` - the null. Every
    other mechanism is untouched, so a comparison isolates the signal.

    `universe` accepts a PREBUILT `FactorUniverse`. Building it costs 5.6s on
    2,437 symbols and it does not depend on the window, the variant or the
    seed - so a sweep that rebuilds it per call spends 24 minutes of its 25
    recomputing one object. Same lesson as `ranking.DailyHistory` in the
    intraday book; the sweep builds it once and passes it here.
    """
    if universe is None:
        universe = FactorUniverse(bars)
    sessions = sorted({d for sd in universe.symbols.values() for d in sd.dates})
    if start:
        sessions = [d for d in sessions if d >= start]
    if end:
        sessions = [d for d in sessions if d <= end]
    dates = month_ends(sessions, hold_months)
    result = FactorResult()
    if len(dates) < 2:
        return result

    result.start, result.end = dates[0], dates[-1]
    # THE SURVIVORSHIP BOUND MUST BE ANCHORED TO THE STUDY, NOT THE WINDOW.
    # Anchoring it to `dates[0]` - the first rebalance of whatever window is
    # being run - excludes only names that listed after that window opened,
    # which inside a 6-month walk-forward fold is almost nobody. The variant
    # then silently reproduces the baseline to four decimal places, which is
    # exactly what it did before this was fixed.
    anchor = listed_before or dates[0]
    restrict = universe.listed_before(anchor) if listed_only else None
    rng = np.random.default_rng(seed) if seed is not None else None

    cash = float(starting_capital)
    held: dict[str, Position] = {}

    for n, day in enumerate(dates, 1):
        if progress:
            progress(n, len(dates), day)

        elig = universe.eligible_at(day, min_price, min_turnover,
                                    min_history, band, restrict_to=restrict)
        for reason, count in elig.rejections.items():
            result.rejections[reason] = result.rejections.get(reason, 0) + count
        result.universe_size.append(len(elig.symbols))

        scores = (mom.random_scores(elig.symbols, rng) if rng is not None
                  else mom.score_universe(universe, elig.symbols, day,
                                          formation))
        wanted = set(mom.top_n(scores, top_n))

        # ---- sell what is no longer wanted -----------------------------
        for symbol in sorted(set(held) - wanted):
            price = _tradeable_price(universe, symbol, day, result)
            if price is None:
                continue                       # cannot trade it; keep holding
            pos = held.pop(symbol)
            gross = price * pos.shares
            charge = _charge(costs, price, pos.shares, True, slippage_pct)
            cash += gross - charge
            result.costs_paid += charge
            result.trades += 1
            result.turnover_sum += gross
            result.contribution[symbol] = (
                result.contribution.get(symbol, 0.0) + gross - charge)

        # ---- mark the book, then buy into the free slots ---------------
        marked = cash
        for symbol, pos in held.items():
            marked += _mark(universe, symbol, day, pos) * pos.shares

        budget = marked / max(top_n, 1) if marked > 0 else 0.0

        # ---- trim winners and top up laggards back to equal weight -----
        # WITHOUT THIS THE SLEEVE IS TWO STRATEGIES. A name held through
        # several rebalances is never trimmed, so its weight compounds and
        # terminal wealth can be carried by one or two positions - which is
        # momentum plus a let-winners-run overlay that was never registered
        # as a hypothesis. Turned on, every held name is pulled back to
        # `marked / top_n`, and every share traded pays both legs.
        if reweight and budget > 0:
            for symbol in sorted(held):
                price = _tradeable_price(universe, symbol, day, result)
                if price is None:
                    continue
                pos = held[symbol]
                target = int(budget // price)
                delta = target - pos.shares
                if delta == 0:
                    continue
                if delta > 0:
                    cost = price * delta
                    charge = _charge(costs, price, delta, False, slippage_pct)
                    if cost + charge > cash:
                        continue          # no cash to top up; leave it light
                    cash -= cost + charge
                    flow = -(cost + charge)
                else:
                    shed = -delta
                    charge = _charge(costs, price, shed, True, slippage_pct)
                    cash += price * shed - charge
                    flow = price * shed - charge
                result.contribution[symbol] = (
                    result.contribution.get(symbol, 0.0) + flow)
                result.costs_paid += charge
                result.trades += 1
                result.turnover_sum += price * abs(delta)
                held[symbol] = Position(symbol, target, pos.entry_price,
                                        pos.entry_day)

        buys = sorted(wanted - set(held))
        if buys and budget > 0:
            for symbol in buys:
                price = _tradeable_price(universe, symbol, day, result)
                if price is None:
                    continue
                shares = int(budget // price)
                if shares <= 0:
                    continue
                cost = price * shares
                charge = _charge(costs, price, shares, False, slippage_pct)
                if cost + charge > cash:
                    continue                   # no cash; the slot goes unused
                cash -= cost + charge
                result.costs_paid += charge
                result.trades += 1
                result.turnover_sum += cost
                result.contribution[symbol] = (
                    result.contribution.get(symbol, 0.0) - cost - charge)
                held[symbol] = Position(symbol, shares, price, day)

        value = cash + sum(_mark(universe, s, day, p) * p.shares
                           for s, p in held.items())
        result.equity.append((day, value))
        result.holdings_log.append((day, sorted(held)))
        result.rebalances += 1

    # CLOSE THE CONTRIBUTION LEDGER. Everything still held at the end is
    # marked and credited to its own name, so the contributions sum to
    # (final value - starting capital) rather than to the realised half only.
    # Without this a name that is still open - which for a momentum sleeve is
    # exactly the biggest winner - shows only its purchases and reads as the
    # single worst position in the book.
    if result.equity:
        last_day = result.equity[-1][0]
        for symbol, pos in held.items():
            result.contribution[symbol] = (
                result.contribution.get(symbol, 0.0)
                + _mark(universe, symbol, last_day, pos) * pos.shares)

    # `turnover_sum` is accumulated in RUPEES above and converted here into
    # units of book value, so `avg_turnover` can divide by the rebalance count
    # and report the fraction of the book that changed hands each month.
    if result.rebalances and result.equity:
        avg_value = float(np.mean([v for _, v in result.equity]))
        if avg_value > 0:
            result.turnover_sum /= avg_value
    return result


__all__ = ["run", "FactorResult", "Position", "CAVEATS", "MAX_SESSION_MOVE"]
