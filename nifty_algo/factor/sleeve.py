"""
The monthly factor sleeve, live: what to hold this month, and why.

THE SCAN IS THE BACKTEST'S OWN FUNCTIONS CALLED ON TODAY. `eligible_at`,
`score_universe` and `top_n` are imported and used unchanged, exactly as
`swing/backtest.py` calls `scanner.evaluate_symbol`. There is no `if live:`
anywhere in the path, which is the only thing that makes the recorded numbers
a description of what this page recommends.

TWO RULES SHAPE EVERYTHING HERE, AND BOTH ARE ABOUT RESTRAINT.

  1. THE SLEEVE TRADES ONCE A MONTH. Turnover is its largest measured cost -
     2.54%/yr of average equity - and its measured edge over a random-scoring
     null exists partly BECAUSE it turns 0.56x per rebalance where the null
     turns 1.75x. So the diff is computed every day and marked `provisional`
     until the rebalance date. Acting on a provisional diff is not a small
     deviation from the tested book; it is the mechanism by which that book's
     edge gets spent on brokerage.

  2. NEWS NEVER TOUCHES THE RANKING. `swing/scanner._score` folds news into a
     rank because the swing book was measured with it there. This book was not.
     A headline that reorders the top 20 makes the live sleeve a different and
     untested strategy while every figure on the page still claims to describe
     the tested one. News is attached to a pick for a human to read, and
     `NewsResult.available` keeps "nothing notable" distinct from "feed down".

WHAT THIS PAGE MUST NEVER LET YOU FORGET. F1 cleared its gate on 2016-2026:
+18.79% CAGR, beat 500/500 null seeds, p = 0.002. F2's kill test on the
never-used 2005-2016 window returned +0.60pp over the index, a -78.5% drawdown
and an 81-month recovery. `RECORDS` carries both and is rendered above the
picks rather than in a footnote, because a console that shows the first number
without the second is a machine for forgetting the second.

AND THE RECORD IS PER UNIVERSE. F4 measured that restricting to the Nifty 500
is free while restricting to the Nifty 100 costs 10pp, and that the naive
restricted backtests are inflated by 13.69pp and 11.87pp of look-ahead. Showing
the unrestricted +18.79% above a restricted book would quote numbers for a
universe nobody is trading - the same failure in a new costume - so
`record_for` returns the measured record for the selected universe, or None,
and None means the caller says so rather than borrowing.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..config import DEFAULT, Config
from ..swing import markets as markets_mod
from ..swing.universe import Stock
from . import backtest as fb
from . import membership as mb
from . import momentum as mom
from . import restriction as restr
from .universe import BANDS, FactorUniverse, month_ends

#: Column headings for every record below.
HEADLINE: tuple[str, ...] = ("window", "sleeve", "Nifty", "excess",
                             "worst fall")


@dataclass(frozen=True)
class Record:
    """
    What a backtest actually says about ONE universe.

    THE RECORD IS PER UNIVERSE BECAUSE THE NUMBERS ARE. Showing the
    unrestricted book's +18.79% above a Nifty-500-restricted book would quote
    figures for a universe that is not being traded - the same failure this
    page exists to prevent, in a new costume.

    `rows` is always what to EXPECT. Where a published-membership universe was
    measured, its naive figure goes in `naive` as a single caption line rather
    than a table row: it is not a caveat on the expectation, it is simply the
    wrong number for planning, and putting it in the table would make it the
    one a reader remembers.
    """
    universe: str
    describes: str
    rows: tuple
    caution: str
    naive: str | None = None


#: F1/F2 for the shipped book; F4's size-ranked CONTROLS for the restricted
#: ones, never their naive figures. A universe absent from here has not been
#: measured, and `record_for` returns None so the caller refuses rather than
#: borrowing someone else's numbers.
_UNRESTRICTED_ROWS = (
    ("2016-2026, where it was measured", "+18.79%", "+10.85%", "+7.94pp",
     "-57.2% / 35 months"),
    ("2005-2016, never used to build it", "+12.39%", "+11.79%", "+0.60pp",
     "-78.5% / 81 months"),
)

RECORDS: dict[str, Record] = {
    "all": Record(
        universe="all",
        describes="every listed NSE name - the book F1, F2 and F3 measured",
        rows=_UNRESTRICTED_ROWS,
        caution="Read both rows. The upper is the window the strategy was "
                "built and measured on; the lower is a decade it had never "
                "seen, where it beat the index by 0.60pp and fell 78.5% "
                "taking 81 months to recover. Survivorship flatters both."),

    "size500": Record(
        universe="size500",
        describes="top 500 by point-in-time size - the CONTROL, and the "
                  "honest estimate for any large-cap-restricted sleeve",
        rows=(("2016-2026", "+17.60%", "+10.85%", "+6.75pp",
               "-48.6% / 34 months"),
              ("2005-2016", "+13.55%", "+11.79%", "+1.76pp",
               "-74.3% / 79 months")),
        caution="Restricting by size costs essentially nothing (-0.41pp and "
                "-0.22pp against the unrestricted book) and buys nothing "
                "either - the drawdown is unchanged."),

    "nifty500": Record(
        universe="nifty500",
        describes="the Nifty 500, priced with its size-ranked control - "
                  "because today's membership cannot be applied to 2016",
        rows=(("2016-2026", "+17.60%", "+10.85%", "+6.75pp",
               "-48.6% / 34 months"),
              ("2005-2016", "+13.55%", "+11.79%", "+1.76pp",
               "-74.3% / 79 months")),
        caution="These are the size-ranked control's numbers, and they are "
                "what to expect. Restricting to the Nifty 500 is free - it "
                "costs about 0.3pp - but it does NOT reduce the drawdown.",
        naive="A backtest restricted to TODAY'S Nifty 500 reports +31.29% "
              "and -32.9%. Ignore it: +13.69pp of that is look-ahead, "
              "because today's membership can only hold companies that grew "
              "INTO the index."),

    "size100": Record(
        universe="size100",
        describes="top 100 by point-in-time size - the control for a Nifty "
                  "100 mandate",
        rows=(("2016-2026", "+7.94%", "+10.85%", "-2.91pp",
               "-33.8% / 25 months"),
              ("2005-2016", "+10.62%", "+11.79%", "-1.17pp",
               "-65.5% / 78 months")),
        caution="A 100-name universe LOSES to the index on both windows. A "
                "top-20 book is a fifth of it, so there is almost no "
                "cross-section left to rank - which is the whole mechanism."),

    "nifty100": Record(
        universe="nifty100",
        describes="the Nifty 100, priced with its size-ranked control",
        rows=(("2016-2026", "+7.94%", "+10.85%", "-2.91pp",
               "-33.8% / 25 months"),
              ("2005-2016", "+10.62%", "+11.79%", "-1.17pp",
               "-65.5% / 78 months")),
        caution="DO NOT RUN THE SLEEVE HERE. Against its control a Nifty 100 "
                "restriction costs 10.07pp of CAGR and lands BELOW the index "
                "on both windows. Momentum needs a cross-section to rank "
                "within and 100 names is not one.",
        naive="A backtest restricted to TODAY'S Nifty 100 reports +19.81% "
              "and -23.9%. Ignore it: +11.87pp of that is look-ahead."),
}


def record_for(universe_key: str) -> Record | None:
    """
    The measured record for a universe, or None if nothing measured it.

    None is a REFUSAL, not a fallback. A caller that quietly substituted
    another universe's numbers would be doing the exact thing this whole
    structure exists to stop, so adding a key to `restriction.UNIVERSES`
    without measuring it makes the page say so instead.
    """
    return RECORDS.get((universe_key or "all").lower())

#: Plan for this, not for the number the good decade showed. Measured on
#: month-end marks over one realisation; `drawdown.DRAWDOWN_HAIRCUT` is why the
#: planning figure is worse again.
MEASURED_DRAWDOWN = 0.785
MEASURED_RECOVERY_MONTHS = 81

CAVEATS: tuple[str, ...] = (
    "SURVIVORSHIP, and it runs one way: the universe is TODAY'S listed NSE "
    "set, so companies delisted inside the window are absent. For a long-only "
    "momentum book that flatters the record by an unknown amount.",
    "The halal screen is applied LIVE but could not be replayed historically - "
    "no point-in-time fundamentals exist - so the recorded returns describe an "
    "UNSCREENED book unless a screened backtest says otherwise.",
    "News is context for you, not an input to the ranking. It changed no "
    "order on this page.",
    "Nothing here places an order. It is a checklist you execute yourself.",
)

#: Below this many returns a volatility number is noise wearing a decimal point.
MIN_SESSIONS_FOR_VOL = 30


@dataclass
class SleevePick:
    """One name the sleeve wants, and everything needed to argue with it."""
    symbol: str
    rank: int
    score: float
    momentum_12_1: float
    price: float
    vol_3m: float = float("nan")
    vol_12m: float = float("nan")
    from_52w_high: float = float("nan")
    turnover_inr: float = 0.0
    liquidity_band: str = ""
    index_band: str = mb.UNKNOWN

    halal: object | None = None
    news: object | None = None

    held_qty: float = 0.0
    cost_basis: float | None = None
    market_value: float = 0.0
    pnl_pct: float | None = None

    target_value_inr: float = 0.0
    target_qty: int = 0
    #: True when the pot ran out before this name. NOT a rounding detail: the
    #: backtested book held all `top_n` names on only 9 of 121 rebalances,
    #: because `budget = marked / top_n` leaves nothing over for charges and
    #: the last buys are refused for cash. A console that ignored that would
    #: hand you a checklist of orders the broker rejects.
    unfunded: bool = False

    @property
    def is_held(self) -> bool:
        return self.held_qty > 0

    @property
    def halal_ok(self) -> bool:
        """A missing verdict is NOT a pass - it is an unscreened name."""
        return bool(self.halal is not None
                    and getattr(self.halal, "eligible", False))


@dataclass
class RegimeState:
    """
    Whether the benchmark is above its own average, as a FACT.

    F2 measured this instrument rather than assuming it: over 2016-2026 it cost
    17.0% compounded, and in 2008 alone it was worth +203%. It is crash
    insurance with a real premium, so the console states both and acts on
    neither.
    """
    ma_days: int
    above: bool | None = None
    note: str = "not computed"

    @property
    def known(self) -> bool:
        return self.above is not None


@dataclass
class SleeveScan:
    as_of: date
    picks: list = field(default_factory=list)
    universe_size: int = 0
    eligible: int = 0
    rejections: dict = field(default_factory=dict)
    screened_out: list = field(default_factory=list)
    shortlist_short: bool = False

    next_rebalance: date | None = None
    is_rebalance_day: bool = False
    sessions_to_rebalance: int | None = None

    regime: RegimeState = field(default_factory=lambda: RegimeState(0))
    membership_note: str = ""
    pot_inr: float = 0.0
    top_n: int = 20
    #: Which liquidity band `eligible_at` was asked for. On the row so a page
    #: full of micro caps says WHY it is a page full of micro caps.
    band: str = "all"
    #: Which slice of the NSE was rankable, and the one-line reason.
    universe_key: str = "all"
    universe_note: str = ""

    holdings_available: bool = False
    holdings_note: str = "not read"
    #: What the pot could not deploy after charges.
    cash_left_inr: float = 0.0

    caveats: tuple = CAVEATS

    @property
    def ticket_inr(self) -> float:
        return self.pot_inr / self.top_n if self.top_n else 0.0

    @property
    def funded(self) -> bool:
        return self.pot_inr > 0


@dataclass
class Action:
    kind: str            # BUY | TOP_UP | HOLD | TRIM | SELL
    symbol: str
    qty: int
    value_inr: float
    reason: str
    provisional: bool = True

    @property
    def is_trade(self) -> bool:
        return self.kind != "HOLD" and self.qty > 0


# ---------------------------------------------------------------- metrics

def _returns(closes: np.ndarray) -> np.ndarray:
    if len(closes) < 2:
        return np.array([], dtype=float)
    return np.diff(closes) / closes[:-1]


def annual_vol(closes: np.ndarray, sessions: int) -> float:
    """Annualised realised volatility over the last `sessions` bars."""
    window = closes[-(sessions + 1):]
    r = _returns(window)
    if len(r) < MIN_SESSIONS_FOR_VOL:
        return float("nan")
    return float(np.std(r, ddof=1)) * float(np.sqrt(252.0))


def from_52w_high(closes: np.ndarray) -> float:
    window = closes[-252:]
    if len(window) < MIN_SESSIONS_FOR_VOL:
        return float("nan")
    peak = float(np.max(window))
    return (float(window[-1]) / peak - 1.0) if peak > 0 else float("nan")


def liquidity_band(turnover: float, all_turnover: np.ndarray) -> str:
    """
    Which pre-registered band this name's turnover falls in.

    Reported, never used to filter here - `eligible_at` already applied
    whatever band the config asked for. This exists so a pick that is only
    tradeable in theory says so on its own row.
    """
    if turnover <= 0 or len(all_turnover) == 0:
        return "unknown"
    pct = float((all_turnover <= turnover).mean())
    for name, (lo, hi) in BANDS.items():
        if name == "all":
            continue
        if lo <= pct < hi or (hi >= 1.0 and pct >= lo):
            return f"{name} ({pct:.0%})"
    return f"below the bands ({pct:.0%})"


# ------------------------------------------------------------ the calendar

def rebalance_calendar(universe: FactorUniverse, hold_months: int,
                       today: date):
    """
    `(next rebalance, is it today, sessions until it)`.

    THE CADENCE IS THE STRATEGY, so `month_ends` is the same function the
    backtest rebalances on - "the next one" here and "the next one there"
    cannot drift apart.

    Past the last mark the cache holds, the date is ESTIMATED from the calendar
    and `sessions_to` comes back None. That distinction is the point: the cache
    ending before month end is normal (it ends on the last session fetched),
    but a rebalance date read off a stale cache would be a real date for the
    wrong month. Estimated means "this is when it is due, and the data cannot
    yet confirm the exact session".
    """
    sessions = sorted({d for sd in universe.symbols.values() for d in sd.dates})
    if not sessions:
        return None, False, None
    marks = month_ends(sessions, hold_months)
    upcoming = [m for m in marks if m >= today]
    if upcoming:
        nxt = upcoming[0]
        return nxt, nxt == today, len([d for d in sessions if today <= d < nxt])
    return _calendar_month_end(today), False, None


def _calendar_month_end(today: date) -> date:
    """
    The last day of this month - a placeholder for a session we cannot see.

    Never treated as a rebalance day: `is_rebalance_day` stays False on this
    path, so an estimated date can move the countdown but can never arm the
    action panel.
    """
    import calendar as _cal
    return date(today.year, today.month,
                _cal.monthrange(today.year, today.month)[1])


def regime_state(benchmark, ma_days: int, today: date) -> RegimeState:
    if not ma_days:
        return RegimeState(0, None, "off - the sleeve runs as it was tested")
    gate = fb._regime_gate(benchmark, ma_days)
    if gate is None:
        return RegimeState(ma_days, None, "no benchmark - cannot say")
    above = gate(today)
    return RegimeState(
        ma_days, above,
        (f"Nifty is above its {ma_days}-day average" if above
         else f"Nifty is BELOW its {ma_days}-day average - the gate would "
              f"hold cash (it cost 17% over 2016-2026 and paid +203% in 2008)"))


# ------------------------------------------------------------- the screen

def stock_for(symbol: str, market, fundamentals=None) -> Stock:
    """
    A `Stock` the halal screen can actually read.

    `activity_failure` and `_is_classified` match on `stock.industry` and
    `stock.sector`, NOT on the fundamentals object - so for this universe,
    which is Kite's instrument dump and carries no industry column at all,
    Yahoo's labels have to be copied ONTO the stock or every name arrives
    unclassified. Unclassified is a reject, so without this the sleeve would
    quietly find nothing eligible and look like a strict screen rather than a
    broken one.
    """
    sector = getattr(fundamentals, "yahoo_sector", None) or ""
    industry = getattr(fundamentals, "yahoo_industry", None) or ""
    return Stock(symbol=symbol, name=symbol, sector=sector, industry=industry,
                 yf_ticker=f"{symbol}{market.yf_suffix}")


def screen_symbols(symbols: list, cfg: Config, market, progress=None) -> dict:
    """
    `{symbol: HalalVerdict}` for a shortlist, fetching fundamentals for it.

    Only ever called on the shortlist, never the universe: fundamentals are one
    slow network request per name, and paying for 2,400 of them to rank 20 is
    exactly the cost the swing scanner's cheap-to-expensive gate ordering
    exists to avoid.
    """
    from ..swing import fundamentals as fund_mod
    from ..swing import halal

    bare = [stock_for(s, market) for s in symbols]
    if progress:
        progress(f"fundamentals for {len(bare)} shortlisted names")
    facts = fund_mod.load_fundamentals(bare, cfg, market)

    overrides, _ = halal.load_overrides(cfg.swing.halal.overrides_csv)
    out = {}
    for symbol in symbols:
        f = facts.get(symbol)
        out[symbol] = halal.screen(stock_for(symbol, market, f), f, cfg,
                                   overrides=overrides, market=market)
    return out


# ---------------------------------------------------------------- the scan

def scan(cfg: Config, bars: dict, benchmark=None, today: date | None = None,
         holdings=None, universe: FactorUniverse | None = None,
         with_news: bool = False, progress=None) -> SleeveScan:
    """
    This month's book, and everything needed to argue with it.

    `holdings` is a `PortfolioSnapshot` or None. None means NOT READ, which is
    a different thing from an empty account and is reported as such - an empty
    read presented as "you hold nothing" would turn every position into a BUY.
    """
    fcfg = cfg.factor
    today = today or date.today()
    market = markets_mod.factor_market(cfg)
    universe = universe or FactorUniverse(bars, adv_window=fcfg.adv_window)

    # The SAME resolver the backtest uses. Live there is no look-ahead in a
    # published membership list - today's constituents are a fact today - but
    # the code path is shared so the two cannot drift into different universes
    # while quoting each other's numbers.
    restrict_fn, universe_note = restr.resolver(
        cfg, fcfg.universe, bars, universe)
    restrict = restrict_fn(today)

    elig = universe.eligible_at(today, fcfg.min_price, fcfg.min_turnover_inr,
                                fcfg.min_history_sessions, fcfg.band,
                                restrict_to=restrict)
    scores = mom.score_universe(universe, elig.symbols, today, fcfg.formation)

    out = SleeveScan(
        as_of=today, universe_size=len(universe.symbols),
        eligible=len(elig.symbols), rejections=dict(elig.rejections),
        pot_inr=cfg.capital.capital_inr(market.capital_pool),
        top_n=fcfg.top_n, band=fcfg.band, universe_key=fcfg.universe,
        universe_note=universe_note)

    out.next_rebalance, out.is_rebalance_day, out.sessions_to_rebalance = (
        rebalance_calendar(universe, fcfg.hold_months, today))
    out.regime = regime_state(benchmark, fcfg.regime_ma_days, today)

    members = mb.load()
    out.membership_note = members.note()

    # --- rank first, then screen DOWN the ranking, as the backtest does ---
    if fcfg.halal_screened:
        ranked = mom.top_n(scores, max(fcfg.halal_shortlist, fcfg.top_n))
        verdicts = screen_symbols(ranked, cfg, market, progress=progress)
        chosen = []
        for symbol in ranked:
            if len(chosen) >= fcfg.top_n:
                break
            v = verdicts.get(symbol)
            if v is not None and v.eligible:
                chosen.append(symbol)
            else:
                out.screened_out.append(
                    (symbol, getattr(v, "reason", "not screened")))
        out.shortlist_short = len(chosen) < fcfg.top_n
    else:
        chosen = mom.top_n(scores, fcfg.top_n)
        verdicts = {}

    all_turnover = np.array([elig.turnover.get(s, 0.0) for s in elig.symbols],
                            dtype=float)
    held_map, out.holdings_available, out.holdings_note = _holdings_map(holdings)

    for i, symbol in enumerate(chosen, 1):
        closes = universe.closes_before(symbol, today)
        price = float(closes[-1]) if len(closes) else 0.0
        pick = SleevePick(
            symbol=symbol, rank=i,
            score=float(scores.get(symbol, float("nan"))),
            momentum_12_1=float(scores.get(symbol, float("nan"))),
            price=price,
            vol_3m=annual_vol(closes, 63), vol_12m=annual_vol(closes, 252),
            from_52w_high=from_52w_high(closes),
            turnover_inr=float(elig.turnover.get(symbol, 0.0)),
            liquidity_band=liquidity_band(elig.turnover.get(symbol, 0.0),
                                          all_turnover),
            index_band=members.band_of(symbol),
            halal=verdicts.get(symbol))
        pick.target_value_inr = out.ticket_inr

        pos = held_map.get(symbol)
        if pos is not None:
            pick.held_qty = float(getattr(pos, "quantity", 0.0) or 0.0)
            avg = getattr(pos, "average_price", None)
            pick.cost_basis = float(avg) if avg else None
            pick.market_value = pick.held_qty * price
            if pick.cost_basis:
                pick.pnl_pct = price / pick.cost_basis - 1.0
        out.picks.append(pick)

    _size_sequentially(out)
    if with_news and out.picks:
        _attach_news(out.picks, cfg, market, progress)
    return out


def _size_sequentially(out: SleeveScan,
                       costs=None) -> None:
    """
    Fill the book in rank order against a running balance, charges included.

    THIS MIRRORS `backtest.run`'s BUY LOOP RATHER THAN DIVIDING THE POT.
    `budget = marked / top_n` spends the whole pot on paper and leaves nothing
    for brokerage, so the loop there skips the last buys for cash - which is
    why the tested book held all 20 names on 9 of 121 rebalances and 15-19 on
    almost every other one. Sizing here by simple division would produce a
    checklist whose final orders the broker refuses, and would also describe a
    book the backtest never ran.
    """
    from ..swing.costs_equity import DEFAULT_EQUITY_COSTS
    costs = costs or DEFAULT_EQUITY_COSTS
    cash = out.pot_inr
    for pick in out.picks:
        if pick.price <= 0 or out.ticket_inr <= 0:
            pick.unfunded = out.ticket_inr > 0
            continue
        qty = int(out.ticket_inr // pick.price)
        while qty > 0:
            cost = qty * pick.price
            charge = costs.buy_cost(pick.price, qty)
            if cost + charge <= cash:
                cash -= cost + charge
                pick.target_qty = qty
                break
            qty -= 1
        pick.unfunded = pick.target_qty == 0
    out.cash_left_inr = cash


def _holdings_map(holdings):
    """
    `{symbol: Position}` from a snapshot, and whether it may be believed.

    A FAILED READ IS NOT AN EMPTY ACCOUNT. `ConnectorResult.available` exists
    because an empty list read as "you hold nothing" turns a fully invested
    book into twenty BUYs - the single most expensive mistake this page could
    make.
    """
    if holdings is None:
        return {}, False, "holdings NOT READ - every position shows as new"
    positions = {str(getattr(p, "symbol", "")).upper(): p
                 for p in getattr(holdings, "positions", [])}
    if not getattr(holdings, "complete", False):
        try:
            note = "; ".join(holdings.caveats()) or "a connector failed"
        except Exception:
            note = "a connector failed"
        return positions, False, f"holdings INCOMPLETE - {note}"
    return positions, True, "holdings read"


def _attach_news(picks: list, cfg: Config, market, progress=None) -> None:
    """News for the finalists only, and it changes no order. See rule 2."""
    from ..swing import news as news_mod
    try:
        stocks = [stock_for(p.symbol, market) for p in picks]
        if progress:
            progress(f"news for {len(stocks)} picks")
        results = news_mod.fetch_for(stocks, cfg)
    except Exception as e:                                 # pragma: no cover
        results = news_mod.unavailable([p.symbol for p in picks], str(e))
    for p in picks:
        p.news = results.get(p.symbol)


# ------------------------------------------------- between rebalances

@dataclass(frozen=True)
class Flag:
    """
    One holding worth a human look, between rebalance dates.

    NOTHING HERE SELLS ANYTHING. F5 measured the alternative: an ATR trail on
    this book, five multiples, two decades. It costs 1.2-14.7pp of CAGR and
    fails the drawdown gate on both windows, because a book that re-ranks
    monthly buys back most of what a stop sold - so a stop is a delay that
    pays two round trips. What a stop cannot see, and you can, is a company
    that has stopped being the company you bought: a fraud, a delisting, an
    auditor walking out. That is what this surfaces.
    """
    symbol: str
    kind: str              # drawdown | atr | halal | news
    detail: str
    severity: str = "watch"    # watch | review


def review(cfg: Config, scan_result: SleeveScan, holdings=None,
           universe: FactorUniverse | None = None,
           today: date | None = None) -> list:
    """
    What to look at before the next rebalance. Advice to a human, never an order.

    Ordered most-serious first, and every flag says what it is FOR - a
    drawdown flag is not evidence of anything, it is a prompt to go and read.
    """
    fcfg = cfg.factor
    today = today or scan_result.as_of
    held, available, _ = _holdings_map(holdings)
    by_symbol = {p.symbol: p for p in scan_result.picks}
    flags: list[Flag] = []

    for symbol, pos in sorted(held.items()):
        qty = float(getattr(pos, "quantity", 0.0) or 0.0)
        paid = float(getattr(pos, "average_price", 0.0) or 0.0)
        if qty <= 0 or paid <= 0:
            continue
        pick = by_symbol.get(symbol)
        price = pick.price if pick else float(
            getattr(pos, "last_price", 0.0) or 0.0)
        if price <= 0:
            continue

        move = price / paid - 1.0
        if move <= -abs(fcfg.review_drawdown_pct):
            flags.append(Flag(
                symbol, "drawdown",
                f"{move:+.1%} against what you paid "
                f"(Rs {paid:,.2f} -> Rs {price:,.2f}). Go and read why.",
                "review"))

        if universe is not None and fcfg.review_atr_multiple:
            sd = universe.symbols.get(symbol)
            atr = sd.atr_before(today) if sd is not None else None
            if atr:
                level = paid - fcfg.review_atr_multiple * atr
                if price <= level:
                    flags.append(Flag(
                        symbol, "atr",
                        f"below {fcfg.review_atr_multiple:g}x ATR from your "
                        f"entry (Rs {level:,.2f}). A trail would have sold "
                        f"here - F5 measured that rule and it costs more than "
                        f"it saves, so this is information, not a signal.",
                        "watch"))

        if pick is not None and pick.halal is not None and not pick.halal_ok:
            flags.append(Flag(
                symbol, "halal",
                f"no longer passes the screen: {pick.halal.summary}",
                "review"))

        if pick is not None and pick.news is not None:
            n = pick.news
            items = getattr(n, "items", []) or []
            if getattr(n, "available", False) and items:
                flags.append(Flag(
                    symbol, "news",
                    f"{len(items)} recent headline"
                    f"{'s' if len(items) != 1 else ''} - "
                    f"{getattr(items[0], 'title', '')[:90]}",
                    "watch"))

    order = {"review": 0, "watch": 1}
    flags.sort(key=lambda f: (order.get(f.severity, 9), f.symbol, f.kind))
    if not available and flags:
        flags.append(Flag(
            "-", "holdings",
            "the holdings read was incomplete, so this list may be missing "
            "positions entirely.", "review"))
    return flags


# --------------------------------------------------------------- the call

def decide(scan_result: SleeveScan, holdings=None) -> list:
    """
    The diff between what is held and what the sleeve wants.

    PROVISIONAL UNTIL THE REBALANCE DATE. Every action carries the flag and
    nothing here hides it - a page that renders a provisional BUY as an
    ordinary one has removed the only guard against turning a monthly book into
    a daily one.
    """
    provisional = not scan_result.is_rebalance_day
    held, available, _ = _holdings_map(holdings)
    wanted = {p.symbol for p in scan_result.picks}
    actions: list[Action] = []

    for p in scan_result.picks:
        if not scan_result.funded:
            actions.append(Action(
                "HOLD", p.symbol, 0, 0.0,
                "the factor pot is zero - set it before any size exists",
                provisional))
            continue
        if p.target_qty <= 0:
            actions.append(Action(
                "HOLD", p.symbol, 0, 0.0,
                "the pot ran out before this rank - the tested book is "
                "routinely lighter than top_n for exactly this reason",
                provisional))
            continue
        # ONE SOURCE OF TRUTH FOR WHAT IS HELD. `pick.held_qty` is whatever
        # `scan` was given; `decide` may be handed a fresher snapshot, and two
        # answers to "do I own this" is how a page recommends buying something
        # twice.
        pos = held.get(p.symbol)
        owned = int(float(getattr(pos, "quantity", 0.0) or 0.0)) if pos else 0
        if owned <= 0:
            actions.append(Action(
                "BUY", p.symbol, p.target_qty, p.target_qty * p.price,
                f"rank {p.rank} of {scan_result.top_n}, not held", provisional))
            continue
        delta = p.target_qty - owned
        if delta > 0:
            actions.append(Action("TOP_UP", p.symbol, delta, delta * p.price,
                                  "still ranked; below its equal weight",
                                  provisional))
        elif delta < 0:
            actions.append(Action("TRIM", p.symbol, -delta, -delta * p.price,
                                  "still ranked; above its equal weight",
                                  provisional))
        else:
            actions.append(Action("HOLD", p.symbol, 0, 0.0,
                                  "still ranked, already at weight",
                                  provisional))

    for symbol, pos in sorted(held.items()):
        if symbol in wanted:
            continue
        qty = int(float(getattr(pos, "quantity", 0.0) or 0.0))
        if qty <= 0:
            continue
        price = float(getattr(pos, "last_price", 0.0) or 0.0)
        actions.append(Action(
            "SELL", symbol, qty, qty * price,
            f"no longer in the top {scan_result.top_n}", provisional))

    if not available:
        for a in actions:
            a.reason += " [holdings unverified]"
    return actions


# ------------------------------------------------------------- the report

def _record_lines(universe_key: str) -> list:
    """
    The measured record for THIS universe, or a refusal.

    Shared with the page through `record_for`, so the CLI and the console
    cannot end up quoting different numbers for the same book.
    """
    rec = record_for(universe_key)
    if rec is None:
        return [f"  NO BACKTEST DESCRIBES THE UNIVERSE {universe_key!r}.",
                "  Nothing here has measured it, so no return figure is "
                "shown - borrowing another",
                "  universe's numbers would be worse than showing none."]
    out = [f"  What a backtest says about THIS universe ({rec.describes}):",
           f"    {HEADLINE[0]:<36}{HEADLINE[1]:>9}{HEADLINE[2]:>10}"
           f"{HEADLINE[3]:>9}  {HEADLINE[4]}"]
    for row in rec.rows:
        out.append(f"    {row[0]:<36}{row[1]:>9}{row[2]:>10}{row[3]:>9}  "
                   f"{row[4]}")
    out.append("")
    out += ["  " + line for line in textwrap.wrap(rec.caution, 74)]
    if rec.naive:
        out.append("")
        out += ["  " + line for line in textwrap.wrap(rec.naive, 74)]
    return out


def report(scan_result: SleeveScan, actions: list, flags: list = ()) -> str:
    s = scan_result
    lines = ["THE MONTHLY FACTOR SLEEVE", ""]
    lines += _record_lines(s.universe_key)
    lines += ["", f"  as of {s.as_of}   universe {s.universe_size:,}   "
                  f"eligible {s.eligible:,}   holding {len(s.picks)}"]

    if s.next_rebalance is None:
        lines.append("  NEXT REBALANCE UNKNOWN - the price cache holds no "
                     "sessions at all.")
    elif s.sessions_to_rebalance is None:
        lines.append(f"  next rebalance ~{s.next_rebalance} (ESTIMATED from "
                     f"the calendar - the cache ends before it, so refresh "
                     f"before acting). Everything below is PROVISIONAL.")
    elif s.is_rebalance_day:
        lines.append(f"  REBALANCE DAY ({s.next_rebalance}) - the actions "
                     f"below are live.")
    else:
        lines.append(f"  next rebalance {s.next_rebalance} "
                     f"({s.sessions_to_rebalance} sessions away) - everything "
                     f"below is PROVISIONAL and should not be traded")

    lines.append(f"  universe: {s.universe_note}")
    lines.append(f"  regime: {s.regime.note}")
    lines.append(f"  {s.membership_note}")
    lines.append(f"  {s.holdings_note}")
    if not s.funded:
        lines.append("  THE POT IS ZERO, so every ticket sizes to zero. Set "
                     "capital.factor_capital_inr.")
    else:
        lines.append(f"  pot Rs {s.pot_inr:,.0f} over {s.top_n} names = "
                     f"Rs {s.ticket_inr:,.0f} a ticket")
    if s.shortlist_short:
        lines.append(f"  THE SHORTLIST RAN OUT: fewer than {s.top_n} names "
                     f"passed the screen, so the book is lighter than designed.")

    lines += ["", f"  {'#':>3} {'symbol':<13}{'mom12-1':>9}{'vol12m':>7}"
                  f"{'off high':>9}{'ADV Rs cr':>10}  {'index band':<22}"
                  f"{'halal':<6}{'held':>6}{'P&L':>8}"]
    for p in s.picks:
        halal = "-" if p.halal is None else ("pass" if p.halal_ok else "FAIL")
        pnl = "-" if p.pnl_pct is None else f"{p.pnl_pct:+.1%}"
        lines.append(
            f"  {p.rank:>3} {p.symbol:<13}{p.momentum_12_1:>+9.1%}"
            f"{p.vol_12m:>7.0%}{p.from_52w_high:>9.1%}"
            f"{p.turnover_inr / 1e7:>10.1f}  "
            f"{p.index_band:<22}{halal:<6}"
            f"{(int(p.held_qty) if p.is_held else 0):>6}{pnl:>8}")

    outside = sum(1 for p in s.picks if p.index_band.startswith("Outside"))
    if outside and s.universe_key == "all":
        lines += [
            "",
            f"  {outside} of {len(s.picks)} picks sit OUTSIDE THE NIFTY 500. "
            f"That is what band={s.band!r} selects and it",
            "  is what was backtested - but it is also where the fills are "
            "worst. F1 measured the liquid",
            "  band at +15.5% against +18.8% for all, and a ticket you cannot "
            "fill is a return you did not get.",
        ]

    if s.screened_out:
        lines += ["", f"  screened out on the way down the ranking "
                      f"({len(s.screened_out)}):"]
        for symbol, why in s.screened_out[:10]:
            lines.append(f"    {symbol:<14}{why}")

    trades = [a for a in actions if a.is_trade]
    header = "  THE CALL"
    if trades and trades[0].provisional:
        header += "   (PROVISIONAL - do not trade before the rebalance date)"
    lines += ["", header]
    if not trades:
        lines.append("    nothing to do")
    for a in trades:
        lines.append(f"    {a.kind:<8}{a.symbol:<14}{a.qty:>8} sh  "
                     f"Rs {a.value_inr:>11,.0f}   {a.reason}")

    if flags:
        lines += ["", "  BETWEEN REBALANCES - things to LOOK at, never orders:"]
        for f in flags:
            mark = "!" if f.severity == "review" else " "
            lines.append(f"   {mark} {f.symbol:<12}{f.kind:<10}{f.detail}")
        lines += ["",
                  "    F5 measured the automatic version - an ATR trail at "
                  "five multiples over two",
                  "    decades - and it costs 1.2-14.7pp of CAGR while "
                  "failing the drawdown gate on",
                  "    both windows. A monthly book re-buys what a stop sold. "
                  "These are prompts to read."]

    lines += ["", "  What this cannot see:"]
    for c in s.caveats:
        lines.append(f"    - {c}")
    lines += ["",
              f"  SIZE IT OFF THE DRAWDOWN, NOT THE CAGR: the measured worst "
              f"fall is {MEASURED_DRAWDOWN:.1%} and took "
              f"{MEASURED_RECOVERY_MONTHS} months to recover.",
              "  There is deliberately no stop-loss control here. F2 measured "
              "one across two widths, two",
              "  bases and two decades: in a book that re-buys monthly a stop "
              "is a delay, not an exit."]
    return "\n".join(lines)


# ------------------------------------------------------------------- CLI

def load_bars(cfg: Config):
    """Read-only, and never through a loader that rewrites the cache."""
    from .drawdown import load
    return load(cfg.factor)


def _main() -> int:                                        # pragma: no cover
    p = argparse.ArgumentParser(description="This month's factor sleeve.")
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--news", action="store_true",
                   help="fetch headlines for the picks (context only)")
    p.add_argument("--halal", action="store_true",
                   help="apply the Shariah screen down the ranking")
    p.add_argument("--regime-ma", type=int, default=50,
                   help="report the regime gate's state; 0 to hide it")
    p.add_argument("--cache", default="", help="read a different parquet")
    p.add_argument("--universe", default=None,
                   choices=list(restr.UNIVERSES),
                   help="which slice of the NSE may be ranked")
    p.add_argument("--asof", default=None, help="YYYY-MM-DD, for replay")
    p.add_argument("--no-holdings", action="store_true")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    if args.capital is not None:
        cfg.capital.factor_capital_inr = args.capital
    if args.cache:
        cfg.factor.cache_name = args.cache
    if args.universe:
        cfg.factor.universe = args.universe
    cfg.factor.halal_screened = args.halal
    cfg.factor.regime_ma_days = args.regime_ma

    bars, bench = load_bars(cfg)
    today = date.fromisoformat(args.asof) if args.asof else date.today()
    print(f"{len(bars):,} symbols loaded", flush=True)

    holdings = None
    if not args.no_holdings:
        try:
            from ..portfolio import aggregate
            holdings = aggregate.load(cfg)
        except Exception as e:
            print(f"  holdings unavailable ({e})", flush=True)

    result = scan(cfg, bars, bench, today=today, holdings=holdings,
                  with_news=args.news,
                  progress=lambda m: print(f"  [{m}]", flush=True))
    print()
    print(report(result, decide(result, holdings),
                 review(cfg, result, holdings, universe=None)))
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(_main())
