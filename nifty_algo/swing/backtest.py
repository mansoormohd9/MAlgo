"""
Walk-forward backtest for the daily swing book.

======================================================================
READ THIS BEFORE READING ANY NUMBER THIS MODULE PRODUCES
======================================================================

[backtest.py](../backtest.py) opens with the same warning for the option book,
and for the same reason: a backtest's failure mode is not being wrong, it is
being FLATTERING and wrong. Three distortions here are structural, cannot be
fixed with more code, and are printed above every result rather than buried:

  1. SURVIVORSHIP. `data/nifty100.csv` is TODAY's index. Testing 2021-2026
     against it includes only companies that were still in the Nifty 100 in
     2026 - every name that collapsed out of it is absent. Real point-in-time
     index membership is not available free, and constructing it from
     memory would be worse than admitting this.

  2. NO HALAL SCREEN AND NO EARNINGS BLACKOUT. Both need fundamentals, and
     the only fundamentals available are TODAY's - a company that failed the
     debt test in 2022 and passes now would be wrongly included for 2022, and
     an earnings date from 2026 says nothing about a blackout in 2023. So
     neither gate runs at all here, which means this tests a LARGER universe
     than the live scan would ever trade. Running them on today's data would
     have been worse: a look-ahead dressed as rigour.

  3. NEWS CANNOT BE REPLAYED. Google News RSS serves recent items only, so
     the 10% news weight has no historical value. This one is genuinely
     handled rather than merely disclosed: `_score()` already redistributes
     that weight across the remaining components when news is unavailable,
     and the backtest goes down that exact path. The scan does the same thing
     on a day the feed is unreachable.

What it CAN measure honestly: whether these setups, sized this way and managed
by this ladder, reach their targets more often than their stops - and what the
delivery charges take out of the difference. That is the question worth asking
before any money moves, and it is the one the repo could not answer at all.

STRUCTURE. Every decision comes from `scanner.evaluate_symbol()` and
`scanner.rank_and_size()` - the same functions `scan()` calls, not copies.
`setup.detect()` is called completely unchanged on a progressively truncated
frame, which its own docstring already promises is safe: it reads only the last
bar of what it is given. There is no `if backtest:` anywhere in the path.

PESSIMISM, as everywhere else here. A daily bar covering both the stop and the
next rung is scored as a stop, at the stop's PRE-BAR level. The ladder does
this itself - `positions.ExitLadder.advance()` - so the backtest and the live
book cannot disagree about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..backtest import Metrics, compute_metrics
from ..config import Config
from ..positions import ExitKind, ExitLadder
from . import markets as markets_mod
from . import news as news_mod
from . import scanner as scanner_mod
from . import setup as setup_mod
from .book import swing_trade
from .costs_equity import DEFAULT_EQUITY_COSTS, EquityCostModel

#: Printed above every result. Not a footnote - see the module docstring.
CAVEATS = (
    "SURVIVORSHIP: the universe file is today's index. Names that fell out of "
    "it are absent, so this result is flattered by an unknown amount.",
    "NO HALAL SCREEN, NO EARNINGS BLACKOUT: both need point-in-time "
    "fundamentals, and only today's exist. Neither gate runs, so this tests a "
    "LARGER universe than the live scan would trade.",
    "NEWS: cannot be replayed - the feed serves recent items only. Its 10% "
    "weight is redistributed over the other components, which is the same "
    "path a live scan takes when the feed is unreachable.",
    "FILLS: assumed at the trigger price. Real slippage is measured per fill "
    "by the position book, and is a placeholder until you have twenty of them.",
    "This is not evidence to trade live capital. It is evidence about whether "
    "the idea is worth paper-trading.",
)


@dataclass
class SwingTrade:
    """
    One completed trade.

    Field names match `backtest.BacktestTrade` where they overlap, so
    `compute_metrics()` consumes this directly rather than through an adapter.
    """
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    strategy: str                  # the setup key, for per-setup expectancy
    sector: str
    entry: float
    exit_price: float
    stop: float
    target: float
    quantity: float
    outcome: str
    bars_held: int
    r_multiple: float              # NET of charges, in R
    gross_r: float
    net_pnl: float
    friction: float
    partial_banked: bool = False
    reason: str = ""


@dataclass
class SwingDayStats:
    """What the BOOK did, as opposed to what the setups did."""
    days: int = 0
    days_flat: int = 0
    max_concurrent: int = 0
    max_open_risk_r: float = 0.0
    avg_deployed_inr: float = 0.0
    heat_blocks: int = 0           # entries refused by the open-risk cap
    capital_blocks: int = 0        # entries refused for want of cash
    peak_deployed_inr: float = 0.0

    def as_dict(self) -> dict:
        return {
            "Sessions": self.days,
            "Days flat": f"{self.days_flat} ({self.days_flat / self.days:.0%})"
                         if self.days else "0",
            "Max concurrent positions": self.max_concurrent,
            "Peak open risk (R)": f"{self.max_open_risk_r:.2f}",
            "Avg deployed": f"₹{self.avg_deployed_inr:,.0f}",
            "Peak deployed": f"₹{self.peak_deployed_inr:,.0f}",
            "Entries blocked by heat cap": self.heat_blocks,
            "Entries blocked by cash": self.capital_blocks,
        }


@dataclass
class SwingBacktestResult:
    market: str = markets_mod.INDIA
    trades: list[SwingTrade] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    by_setup: dict[str, Metrics] = field(default_factory=dict)
    day_stats: SwingDayStats = field(default_factory=SwingDayStats)
    warnings: list[str] = field(default_factory=list)
    caveats: tuple = CAVEATS
    start: Optional[date] = None
    end: Optional[date] = None
    symbols: int = 0

    @property
    def equity_curve_r(self) -> pd.Series:
        if not self.trades:
            return pd.Series(dtype=float)
        ordered = sorted(self.trades, key=lambda t: t.exit_time)
        return pd.Series(
            np.cumsum([t.r_multiple for t in ordered]),
            index=[t.exit_time for t in ordered],
        )

    def headline(self) -> str:
        m = self.metrics
        if not m.trades:
            return "No trades. Either the gates are too tight or the window is too short."
        verdict = ("above" if m.win_rate > m.breakeven_win_rate else "below")
        return (f"{m.trades} trades · {m.win_rate:.0%} won ({verdict} the "
                f"{m.breakeven_win_rate:.0%} breakeven) · expectancy "
                f"{m.expectancy_r:+.3f}R · total {m.total_r:+.1f}R · "
                f"max drawdown {m.max_drawdown_r:.1f}R")


# --------------------------------------------------------------- open trade

@dataclass
class _Open:
    """A position the simulation is carrying."""
    symbol: str
    sector: str
    setup_key: str
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    quantity: float
    state: object                  # LadderState
    bars_held: int = 0
    realised: float = 0.0
    partial_banked: bool = False

    @property
    def risk_points(self) -> float:
        return max(self.entry - self.stop, 0.0)

    def r_of(self, price: float) -> float:
        rp = self.risk_points
        return (price - self.entry) / rp if rp > 0 else 0.0

    def price_of(self, r: float) -> float:
        return self.entry + r * self.risk_points

    @property
    def open_risk_r(self) -> float:
        per_unit = max(-self.state.stop_r, 0.0)
        total = float(self.state.lots_total) or 1.0
        return per_unit * (self.state.lots_remaining / total)


# --------------------------------------------------------------- the runner

def run(cfg: Config, market, bars: dict[str, pd.DataFrame],
        benchmark: Optional[pd.DataFrame] = None,
        stocks: Optional[list] = None,
        start: Optional[date] = None, end: Optional[date] = None,
        costs: EquityCostModel = DEFAULT_EQUITY_COSTS,
        progress: Optional[Callable] = None) -> SwingBacktestResult:
    """
    Walk the daily bars forward, one session at a time.

    `bars` is `{symbol: daily DataFrame}` - the same shape `prices.load_prices`
    returns, so a backtest reads the cache the live scan already wrote.
    `stocks` supplies sector labels for the sector cap; without it every name
    is treated as its own sector and the cap cannot bite.
    """
    result = SwingBacktestResult(market=market.key)
    if not bars:
        result.warnings.append("no bars supplied - nothing to test")
        return result

    ladder = ExitLadder(cfg, trade=swing_trade(cfg))
    sectors = {s.symbol: s.sector for s in (stocks or [])}
    by_symbol = {s.symbol: s for s in (stocks or [])}
    budget = cfg.capital.risk_inr(market.capital_pool)
    capital = cfg.capital.capital_inr(market.capital_pool)
    if budget <= 0:
        result.warnings.append(
            f"the {cfg.capital.pool_label(market.capital_pool)} is ₹0, so "
            f"every ticket would size to zero. Set "
            f"{cfg.capital.pool_field(market.capital_pool)} before reading "
            f"anything below.")
        return result

    sessions = _sessions(bars, start, end)
    if len(sessions) < 2:
        result.warnings.append("fewer than two sessions in range")
        return result

    result.start, result.end = sessions[0].date(), sessions[-1].date()
    result.symbols = len(bars)

    warmup = setup_mod.min_bars(cfg)
    open_trades: dict[str, _Open] = {}
    stats = SwingDayStats()
    deployed_total = 0.0

    for i, day in enumerate(sessions):
        if progress and i % 20 == 0:
            progress(i, len(sessions), f"{day:%b %Y}")

        # ---- 1. MANAGE WHAT IS OPEN, FIRST ----
        # Same ordering rule as `engine.run_once()`: money already at risk
        # outranks a new opportunity. Evaluating entries first would let a
        # fourth position open on a day the third had already stopped out.
        _manage(open_trades, bars, day, ladder, cfg, costs, result, sectors)

        stats.days += 1
        if not open_trades:
            stats.days_flat += 1
        stats.max_concurrent = max(stats.max_concurrent, len(open_trades))
        heat = sum(t.open_risk_r for t in open_trades.values())
        stats.max_open_risk_r = max(stats.max_open_risk_r, heat)
        today_deployed = sum(t.entry * t.state.lots_remaining
                             for t in open_trades.values())
        deployed_total += today_deployed
        stats.peak_deployed_inr = max(stats.peak_deployed_inr, today_deployed)

        # ---- 2. can anything new be opened at all? ----
        room = cfg.swing.top_n - len(open_trades)
        if room <= 0:
            continue
        if heat >= cfg.swing.max_open_risk_r - 1e-9:
            stats.heat_blocks += 1
            continue

        # ---- 3. the scan, on bars up to and including today ----
        scored = _scan_day(bars, benchmark, day, cfg, market, warmup,
                           open_trades, by_symbol, sectors)
        if not scored:
            continue

        picks = scanner_mod.rank_and_size(
            scored, cfg, market, _unit_rate(), day.date(), top_n=room)

        # ---- 4. open them, at the trigger, on the NEXT session ----
        for pick in picks:
            _try_open(pick, bars, sessions, i, open_trades, ladder, cfg,
                      stats, capital)

    # Anything still open at the end is NOT counted as a win or a loss. It is
    # an unfinished trade, and scoring it at the last close would let a long
    # holding period masquerade as an outcome.
    for t in open_trades.values():
        result.warnings.append(
            f"{t.symbol} was still open at the end of the window and is "
            f"excluded from the statistics.")

    stats.avg_deployed_inr = deployed_total / stats.days if stats.days else 0.0
    result.day_stats = stats
    result.metrics = compute_metrics(result.trades, cfg.capital.reward_risk_ratio)
    result.by_setup = _by_setup(result.trades, cfg.capital.reward_risk_ratio)
    return result


# --------------------------------------------------------------- internals

def _sessions(bars: dict, start: Optional[date],
              end: Optional[date]) -> list[pd.Timestamp]:
    """Every trading day any symbol has a bar for, in order."""
    index = None
    for df in bars.values():
        index = df.index if index is None else index.union(df.index)
    if index is None:
        return []
    days = pd.DatetimeIndex(sorted(set(index)))
    if start is not None:
        days = days[days.date >= start]
    if end is not None:
        days = days[days.date <= end]
    return list(days)


def _scan_day(bars, benchmark, day, cfg, market, warmup, open_trades,
              by_symbol, sectors):
    """
    Run the live gates over every symbol, using bars up to `day` inclusive.

    The truncation is the whole trick, and it is safe only because
    `setup.detect` reads the last bar of what it is handed and nothing after
    it - a property `tests/test_swing_setup.py` asserts by truncating a frame
    and checking the ticket does not move.
    """
    bench = benchmark.loc[benchmark.index <= day] if benchmark is not None else None
    scored = []

    for symbol, df in bars.items():
        if symbol in open_trades:
            continue                      # the duplicate-name guard
        frame = df.loc[df.index <= day]
        if len(frame) < warmup:
            continue

        found, metrics, rejection = scanner_mod.evaluate_symbol(
            symbol, frame, bench, cfg, market)
        if rejection is not None:
            continue

        # News is not replayable. `unavailable` is the SAME object a live scan
        # builds when the feed is down, so `_score` redistributes the weight
        # rather than scoring a neutral 0.5 that nobody measured.
        n = news_mod.unavailable(
            [symbol], "news cannot be replayed historically")[symbol]
        total, parts = scanner_mod._score(found, metrics, n, cfg)

        stock = by_symbol.get(symbol) or _StubStock(
            symbol, sectors.get(symbol, symbol))
        scored.append((stock, found, metrics, n, total, parts))

    return scored


def _try_open(pick, bars, sessions, i, open_trades, ladder, cfg, stats,
              capital: float):
    """
    Enter on the NEXT session, and only if it trades through the trigger.

    Never on the signal day's own bar: the scan is read off that day's CLOSE,
    so its own range is not information you had. This is the same rule
    `tracker._evaluate` applies to a live pick - `df.index.date > scanned_on`.

    THE CASH GATE. `scanner._size` caps ONE ticket at the whole pot, because
    that is all it can see; three tickets opened on three different days can
    therefore ask for three times the money. Live, the third buy is simply
    rejected for want of funds. A backtest without this constraint quietly
    trades a larger account than you have and reports its returns against the
    smaller one - which inflates every R-per-rupee figure downstream. The
    first version of this file did exactly that: average deployment came out
    at Rs 128,000 against a Rs 100,000 pot.
    """
    if i + 1 >= len(sessions):
        return
    symbol = pick.symbol
    df = bars.get(symbol)
    if df is None or symbol in open_trades:
        return

    committed = sum(t.entry * t.state.lots_remaining
                    for t in open_trades.values())
    if committed + float(pick.setup.entry) * float(pick.quantity) > capital:
        stats.capital_blocks += 1
        return

    entry = float(pick.setup.entry)
    forward = df.loc[df.index > sessions[i]]
    if forward.empty:
        return

    # NOTE: this scans forward to find the fill, so a position is registered
    # in `open_trades` on the SIGNAL day even when it fills several sessions
    # later. `_manage` skips any bar at or before `entry_time`, so nothing is
    # managed early - but the concurrency, heat and cash gates do see it, and
    # therefore block other entries slightly sooner than reality would. The
    # bias runs toward FEWER trades, never more, which is the direction an
    # error here has to run.
    window = forward.iloc[:cfg.swing.valid_for_days]
    for ts, bar in window.iterrows():
        if float(bar["high"]) < entry:
            continue
        # A gap straight through the trigger fills at the OPEN, not at the
        # trigger - you cannot buy below where the market opened. Ignoring
        # this is a small, systematic, always-favourable error.
        fill = max(entry, float(bar["open"]))

        # ...and the cash check has to be re-run against that FILL. Checking
        # only the planned entry lets a gap-up quietly deploy more than the
        # pot holds: a real run against Nifty 100 data peaked at Rs 30,229
        # against a Rs 30,000 account before this was here. Live, the broker
        # would simply have refused the third buy.
        quantity = float(pick.quantity)
        if committed + fill * quantity > capital:
            stats.capital_blocks += 1
            return

        open_trades[symbol] = _Open(
            symbol=symbol,
            sector=getattr(pick, "sector", ""),
            setup_key=pick.setup.key,
            entry_time=ts, entry=fill,
            stop=float(pick.setup.stop), target=float(pick.setup.target),
            quantity=quantity,
            state=ladder.new_state(int(quantity)),
        )
        return


def _manage(open_trades, bars, day, ladder, cfg, costs, result, sectors):
    """Advance every open position through today's bar."""
    for symbol in list(open_trades):
        t = open_trades[symbol]
        df = bars.get(symbol)
        if df is None or day not in df.index:
            continue                       # no bar: do not mark, do not act
        if day <= t.entry_time:
            continue                       # the entry bar itself is not managed

        bar = df.loc[day]
        t.bars_held += 1
        atr = _atr_at(df, day, cfg.swing.atr_period)
        rp = t.risk_points
        trail_r = (atr * cfg.swing.trail_atr_multiple / rp) if rp > 0 else 0.0

        decisions = ladder.advance(
            t.state,
            mark_r=t.r_of(float(bar["close"])),
            best_r=t.r_of(float(bar["high"])),
            worst_r=t.r_of(float(bar["low"])),
            trail_distance_r=trail_r,
        )
        for d in decisions:
            if not d.exit_lots:
                continue
            fill = t.price_of(d.exit_r)
            t.realised += (fill - t.entry) * d.exit_lots
            if d.kind is ExitKind.PARTIAL_EXIT:
                t.partial_banked = True
            if t.state.closed:
                result.trades.append(
                    _close(t, day, fill, d, costs, sectors))
                del open_trades[symbol]
                break


def _close(t: _Open, day, fill: float, decision, costs, sectors) -> SwingTrade:
    """
    Book the finished trade, charges included.

    Friction is charged on the WHOLE position and converted into R, which is
    the only form in which it is comparable to the 2R the ladder aims at. The
    flat DP fee means a small ticket keeps proportionally less of its win, and
    a percentage-only model would hide exactly that.
    """
    risk_inr = t.risk_points * t.quantity
    # The DP charge is per scrip PER SELL DAY, so a position that banked a
    # partial and then exited pays it twice. Charging one round trip flatters
    # exactly the trades the ladder is built to produce - `costs.py` already
    # charges the option book per leg for the same reason.
    legs = 2 if t.partial_banked else 1
    friction = costs.friction(t.entry, fill, t.quantity)
    friction += (legs - 1) * costs.dp_charge
    gross_r = t.realised / risk_inr if risk_inr > 0 else 0.0
    net_pnl = t.realised - friction
    net_r = net_pnl / risk_inr if risk_inr > 0 else 0.0

    outcome = {
        ExitKind.STOPPED_OUT: "stop",
        ExitKind.TARGET_EXIT: "target",
    }.get(decision.kind, decision.kind.value if decision.kind else "closed")

    return SwingTrade(
        entry_time=t.entry_time, exit_time=day,
        symbol=t.symbol, strategy=t.setup_key,
        sector=t.sector or sectors.get(t.symbol, ""),
        entry=t.entry, exit_price=fill, stop=t.stop, target=t.target,
        quantity=t.quantity, outcome=outcome, bars_held=t.bars_held,
        r_multiple=net_r, gross_r=gross_r, net_pnl=net_pnl,
        friction=friction, partial_banked=t.partial_banked,
        reason=decision.detail,
    )


def _atr_at(df: pd.DataFrame, day, period: int) -> float:
    """ATR as of `day`, computed on bars up to and including it only."""
    frame = df.loc[df.index <= day]
    if len(frame) < period + 1:
        return 0.0
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    value = tr.rolling(period).mean().iloc[-1]
    return 0.0 if pd.isna(value) else float(value)


def _by_setup(trades: list[SwingTrade], reward_risk: float) -> dict:
    out: dict[str, Metrics] = {}
    for key in sorted({t.strategy for t in trades}):
        out[key] = compute_metrics([t for t in trades if t.strategy == key],
                                   reward_risk)
    return out


class _StubStock:
    """Sector labels for a symbol the universe file did not describe."""

    def __init__(self, symbol: str, sector: str):
        self.symbol = symbol
        self.name = symbol
        self.sector = sector
        self.industry = ""
        self.yf_ticker = symbol


class _unit_rate:
    """
    An FX rate of exactly 1.0.

    The backtest is India-only for now, and India's rate IS 1.0 - but
    `rank_and_size` takes a `Rate` because a foreign market's sizing crosses a
    currency boundary. Passing a real `fx.Rate` would fetch today's quote and
    apply it to a trade from 2022, mixing a currency move into a strategy
    result. Refusing to do that is the point.
    """
    currency = "INR"
    inr_per_unit = 1.0
    from_cache = False

    def note(self) -> str:
        return "no conversion (INR)"


def _main() -> None:                                      # pragma: no cover
    import argparse
    import sys

    from ..config import DEFAULT as cfg
    from . import prices as prices_mod
    from .universe import load_universe

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of the daily swing book.")
    parser.add_argument("--market", default=cfg.swing.default_market,
                        choices=markets_mod.keys(cfg))
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--capital", type=float, default=None,
                        help="swing pot in rupees; defaults to config")
    args = parser.parse_args()

    if args.capital:
        cfg.capital.swing_capital_inr = args.capital
    market = markets_mod.get(cfg, args.market)

    print("=" * 70)
    print("READ THIS FIRST")
    print("=" * 70)
    for c in CAVEATS:
        print(f"  * {c}")
    print()

    stocks = load_universe(market.universe_csv)
    tickers = {s.symbol: s.yf_ticker for s in stocks}
    cfg.swing.history_days = max(int(args.years * 365) + 260, 400)
    print(f"Loading {len(tickers)} symbols, ~{args.years:g} years ...")
    price_set = prices_mod.load_prices(tickers, cfg, market)

    result = run(cfg, market, price_set.bars, price_set.benchmark,
                 stocks=stocks,
                 progress=lambda d, t, l: print(f"  [{d}/{t}] {l}", end="\r"))
    print(" " * 60, end="\r")

    for w in result.warnings[:5]:
        print(f"  ! {w}")
    print()
    print(result.headline())
    print()
    for k, v in result.metrics.as_dict().items():
        print(f"  {k:<26} {v}")
    print()
    for k, v in result.day_stats.as_dict().items():
        print(f"  {k:<26} {v}")
    if result.by_setup:
        print("\n  Per setup:")
        for key, m in result.by_setup.items():
            print(f"    {key:<12} {m.trades:>4} trades  "
                  f"{m.win_rate:>5.0%}  {m.expectancy_r:+.3f}R")


if __name__ == "__main__":                                # pragma: no cover
    _main()
