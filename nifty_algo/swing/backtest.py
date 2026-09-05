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

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..backtest import Metrics, compute_metrics
from ..config import Config
from ..positions import ExitKind, ExitLadder
from . import market_regime as regime_mod
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
    regime_blocks: int = 0         # days the regime gate refused an entry
                                   # that had room and heat to be taken - not
                                   # every day the market was below its
                                   # average, because a full book never asked
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
            "Entry days blocked by regime": self.regime_blocks,
        }


#: Where a trade actually ended, in R. `outcome` cannot answer this: `_close`
#: maps every ExitKind.STOPPED_OUT to the word "stop", so a full -1R stop and
#: a 0R exit at a stop already shifted to breakeven are the same string. They
#: are opposite facts about the ladder. The boundaries are in R because that
#: is the only unit both books share.
EXIT_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("Stopped (< -0.5R)", float("-inf"), -0.5),
    ("Scratched (-0.5..+0.5R)", -0.5, 0.5),
    ("Small win (+0.5..+1.5R)", 0.5, 1.5),
    ("Target zone (+1.5..+2.5R)", 1.5, 2.5),
    ("Runner (> +2.5R)", 2.5, float("inf")),
)


def exit_buckets(trades: list["SwingTrade"]) -> dict[str, int]:
    """
    How the ladder ended each trade, bucketed by realised R.

    The bucket that matters is `Scratched`: those are trades that reached +1R,
    moved the stop to breakeven, and then died there. They cost nothing, and
    they are also the winners the book was supposed to have - a book whose win
    rate sits just under its breakeven is usually losing them here, and no
    other field in the result says so.
    """
    counts = {label: 0 for label, _, _ in EXIT_BUCKETS}
    for t in trades:
        for label, low, high in EXIT_BUCKETS:
            if low <= t.r_multiple < high:
                counts[label] += 1
                break
    return counts


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
    def exit_buckets(self) -> dict[str, int]:
        return exit_buckets(self.trades)

    @property
    def partials_banked(self) -> int:
        return sum(1 for t in self.trades if t.partial_banked)

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
                f"{m.breakeven_win_rate:.0%} realised breakeven) · expectancy "
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


# ------------------------------------------------------------- walk-forward

@dataclass(frozen=True)
class Window:
    """One train/test split. Dates only - it holds no data and no result."""
    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @property
    def label(self) -> str:
        return f"fold {self.index} · test {self.test_start:%b %Y}-{self.test_end:%b %Y}"


def fold_windows(start: date, end: date, train_months: int,
                 test_months: int) -> list[Window]:
    """
    Rolling train/test splits across `start`..`end`.

    WHY THE SWING BOOK NEEDS THIS. Until now `run()` was a single in-sample
    pass, which is fine for "what did these rules do" and useless for "which
    of these rule sets should I use" - choosing the best of eight variants on
    the same data that measured them is not evidence, it is the definition of
    fitting. The option backtester has had folds since the start; this is the
    same idea, reusing `cfg.backtest.train_months` / `test_months` rather than
    inventing a second set of numbers to drift apart.

    Windows roll forward by the TEST length, so every session outside the
    first train window is scored exactly once out-of-sample.
    """
    if train_months <= 0 or test_months <= 0:
        raise ValueError("train_months and test_months must both be positive")

    windows: list[Window] = []
    train_start = pd.Timestamp(start)
    hard_end = pd.Timestamp(end)
    i = 0
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if train_end >= hard_end:
            break
        # A truncated final test window is still a window - it is simply
        # shorter, and dropping it would throw away the most recent data,
        # which is the part least likely to be stale.
        windows.append(Window(
            index=i,
            train_start=train_start.date(),
            train_end=(train_end - pd.Timedelta(days=1)).date(),
            test_start=train_end.date(),
            test_end=min(test_end - pd.Timedelta(days=1), hard_end).date(),
        ))
        i += 1
        train_start = train_start + pd.DateOffset(months=test_months)
    return windows


# --------------------------------------------------------------- scan cache

#: Fields of `SwingConfig` the SCAN cannot see. Everything else is assumed to
#: change the signals and therefore invalidates a cached scan.
#:
#: The asymmetry is deliberate and load-bearing. Forgetting to list a
#: book-only field here costs a re-scan, which is slow. Failing to notice that
#: a NEW field affects the scan would serve a stale answer to a different
#: question - the benchmark-cache bug in a new costume. So the default is
#: "invalidates", and only fields provably read after `evaluate_symbol` are
#: exempt: sizing, ranking, the ladder, the fill window and the sector cap all
#: run outside the cached unit.
BOOK_ONLY_SWING_FIELDS = frozenset({
    "top_n", "max_open_risk_r", "trail_atr_multiple", "partial_exit_fraction",
    "valid_for_days", "max_per_sector",
    # The ladder runs after the scan, and the regime gate is a property of the
    # day rather than of any symbol - `_scan_day` reads neither. `breakeven_at_r`
    # and `regime_ma_days` are exactly the two levers this cache exists to
    # sweep cheaply. `enabled_setups` is NOT here: `setup.detect` reads it.
    "breakeven_at_r", "regime_ma_days",
    # `random_seed` replaces the SCORE, which `rank_and_size` reads at t[4]
    # AFTER the cache is consulted - so the null and the real book share one
    # scan pass. `max_hold_days` is read only by `_manage`. Both are provably
    # outside the cached unit, which is the bar this list sets.
    "random_seed", "max_hold_days",
    # never read by the backtest at all
    "markets", "default_market", "cache_dir", "history_days", "halal",
    "price_cache_hours", "fundamentals_cache_days",
    "earnings_blackout_days", "news_finalists", "news_lookback_hours",
    "news_max_items",
})


def scan_signature(cfg: Config, market) -> str:
    """
    Identity of everything the per-symbol scan reads.

    A `ScanCache` carries this and `run` refuses to use one that disagrees.
    Reusing a cache across a changed stop band would answer yesterday's
    question with today's label, and would do it without raising.
    """
    payload = [(k, repr(v)) for k, v in sorted(vars(cfg.swing).items())
               if k not in BOOK_ONLY_SWING_FIELDS]
    payload.append(("reward_risk_ratio", repr(cfg.capital.reward_risk_ratio)))
    payload.append(("market", repr((market.key, market.min_price,
                                    market.min_avg_turnover,
                                    market.benchmark_ticker,
                                    market.price_divisor))))
    return hashlib.sha1(repr(payload).encode()).hexdigest()[:16]


@dataclass
class ScanCache:
    """
    Every symbol's evaluation for every session, computed once.

    WHY THIS EXISTS. A 3-year India pass is ~135,000 `evaluate_symbol` calls
    and over ten minutes, and a sweep of eight variants across six walk-forward
    folds would be a day of compute - which in practice means the sweep does
    not get run, and parameters get chosen by argument instead of evidence.

    WHY IT IS SAFE. `evaluate_symbol` is a pure function of (symbol, bars up
    to today, scan config, market). It does not read the book: the only
    book-dependent step in `_scan_day` is skipping names already held, and
    that filter moves to the consumer, where it belongs. Ranking, sizing, the
    sector cap and the ladder all run outside this cache, which is exactly why
    the ladder and regime variants can replay it.

    `test_a_replayed_variant_is_identical_to_a_direct_run` is the guard.
    """
    signature: str
    days: dict = field(default_factory=dict)

    @property
    def sessions_cached(self) -> int:
        return len(self.days)


# --------------------------------------------------------------- the runner

def run(cfg: Config, market, bars: dict[str, pd.DataFrame],
        benchmark: Optional[pd.DataFrame] = None,
        stocks: Optional[list] = None,
        start: Optional[date] = None, end: Optional[date] = None,
        costs: EquityCostModel = DEFAULT_EQUITY_COSTS,
        progress: Optional[Callable] = None,
        scan_cache: Optional[ScanCache] = None,
        sessions_index: Optional[pd.DatetimeIndex] = None,
        settle_days: int = 60) -> SwingBacktestResult:
    """
    Walk the daily bars forward, one session at a time.

    `bars` is `{symbol: daily DataFrame}` - the same shape `prices.load_prices`
    returns, so a backtest reads the cache the live scan already wrote.
    `stocks` supplies sector labels for the sector cap; without it every name
    is treated as its own sector and the cap cannot bite.

    `end` is the last day a new position may be OPENED, not the last day the
    book is managed. After it, `settle_days` further sessions are walked with
    entries switched off, so a position opened inside the window is carried to
    its natural exit.

    WHY THAT MATTERS. Anything still open when a window closes is excluded
    from the statistics, and what is still open is not a random sample - a
    loser hits its stop in days while a winner trails for weeks. Cutting at
    the boundary therefore deletes winners preferentially. Measured on the
    30-fold India sweep before this existed: the same ~261 trades scored
    -67.8R tiled against -21.6R run continuously, and every variant looked
    about 0.2R worse than it was.

    `scan_cache` is an optional `ScanCache` shared across variant runs. It is
    filled on first use and read afterwards, and it RAISES rather than falls
    back when its signature disagrees with this config - a silently stale scan
    is the one failure mode that would make every number here plausible and
    wrong.
    """
    if scan_cache is not None:
        expected = scan_signature(cfg, market)
        if scan_cache.signature != expected:
            raise ValueError(
                f"scan cache was built for signature {scan_cache.signature} "
                f"and this config hashes to {expected}; a cache may only be "
                f"shared between runs whose scan parameters are identical")

    result = SwingBacktestResult(market=market.key)
    if not bars:
        result.warnings.append("no bars supplied - nothing to test")
        return result

    ladder = ExitLadder(cfg, trade=swing_trade(cfg))
    # ONE generator per backtest, so the null is reproducible from its seed and
    # consecutive sessions draw different numbers. Seeding per day would hand
    # every session the same permutation - noise with a pattern in it.
    rng = (np.random.default_rng(cfg.swing.random_seed)
           if cfg.swing.random_seed is not None else None)
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

    sessions = _sessions(bars, start, end, sessions_index)
    if len(sessions) < 2:
        result.warnings.append("fewer than two sessions in range")
        return result

    result.start, result.end = sessions[0].date(), sessions[-1].date()
    result.symbols = len(bars)

    # The regime filter fails closed, which is right, and which means a
    # missing benchmark produces a book that never trades. Without this line
    # that reads as "the filter kills the book" rather than "the filter never
    # ran" - two opposite conclusions from one empty table.
    if cfg.swing.regime_ma_days > 0:
        probe = regime_mod.benchmark_state(benchmark, cfg.swing.regime_ma_days,
                                           as_of=sessions[0])
        if benchmark is None:
            result.warnings.append(
                f"the {cfg.swing.regime_ma_days}-day regime filter is on but "
                f"no benchmark was supplied, so EVERY day is blocked and no "
                f"trade below is a judgement about the setups")
        elif not probe.ok and probe.moving_average is None:
            result.warnings.append(
                f"the benchmark has fewer than {cfg.swing.regime_ma_days} bars "
                f"at the start of this window, so the first sessions are "
                f"blocked for want of history rather than by the market")

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

        # ---- 2b. is the market itself worth being long in? ----
        # Off by default (`regime_ma_days = 0`). Placed after the heat check
        # and before the scan for the same cheap-to-expensive reason the live
        # scanner orders its gates: a day the whole book stands down should
        # not cost 100 symbol evaluations.
        regime = regime_mod.benchmark_state(
            benchmark, cfg.swing.regime_ma_days, as_of=day)
        if not regime.ok:
            stats.regime_blocks += 1
            continue

        # ---- 3. the scan, on bars up to and including today ----
        scored = _scan_day(bars, benchmark, day, cfg, market, warmup,
                           open_trades, by_symbol, sectors, scan_cache)
        if not scored:
            continue

        scored = _apply_null(scored, cfg, day, rng)

        picks = scanner_mod.rank_and_size(
            scored, cfg, market, _unit_rate(), day.date(), top_n=room)

        # ---- 4. open them, at the trigger, on the NEXT session ----
        for pick in picks:
            _try_open(pick, bars, sessions, i, open_trades, ladder, cfg,
                      stats, capital)

    # ---- settlement: manage to the exit, open nothing new ----
    # The entry window is closed; these sessions exist only to let positions
    # opened inside it finish. Day stats deliberately do not advance here -
    # they describe the window that was traded, not the tail that settled it.
    if open_trades and settle_days > 0:
        every = _sessions(bars, None, None, sessions_index)
        tail = [d for d in every if d > sessions[-1]][:settle_days]
        for day in tail:
            if not open_trades:
                break
            _manage(open_trades, bars, day, ladder, cfg, costs, result, sectors)

    # Anything still open after settlement is NOT counted as a win or a loss.
    # It is an unfinished trade, and scoring it at the last close would let a
    # long holding period masquerade as an outcome.
    for t in open_trades.values():
        result.warnings.append(
            f"{t.symbol} was still open {settle_days} sessions after the "
            f"window closed and is excluded from the statistics.")

    stats.avg_deployed_inr = deployed_total / stats.days if stats.days else 0.0
    result.day_stats = stats
    result.metrics = compute_metrics(result.trades, cfg.capital.reward_risk_ratio)
    result.by_setup = _by_setup(result.trades, cfg.capital.reward_risk_ratio)
    return result


# --------------------------------------------------------------- internals

def all_sessions(bars: dict) -> pd.DatetimeIndex:
    """
    Every trading day any symbol has a bar for, in order.

    Split out and exposed because a sweep calls `run()` hundreds of times over
    the same bars, and unioning a hundred DatetimeIndexes each time was a
    meaningful share of the wall clock for work whose answer never changes.
    """
    index = None
    for df in bars.values():
        index = df.index if index is None else index.union(df.index)
    if index is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(set(index)))


def _sessions(bars: dict, start: Optional[date], end: Optional[date],
              precomputed: Optional[pd.DatetimeIndex] = None
              ) -> list[pd.Timestamp]:
    """The sessions in range. `precomputed` is `all_sessions(bars)`, reused."""
    days = all_sessions(bars) if precomputed is None else precomputed
    if len(days) == 0:
        return []
    if start is not None:
        days = days[days.date >= start]
    if end is not None:
        days = days[days.date <= end]
    return list(days)


def _scan_day(bars, benchmark, day, cfg, market, warmup, open_trades,
              by_symbol, sectors, cache: Optional[ScanCache] = None):
    """
    Run the live gates over every symbol, using bars up to `day` inclusive.

    The truncation is the whole trick, and it is safe only because
    `setup.detect` reads the last bar of what it is handed and nothing after
    it - a property `tests/test_swing_setup.py` asserts by truncating a frame
    and checking the ticket does not move.
    """
    if cache is not None and day in cache.days:
        # The duplicate-name guard is applied HERE rather than during
        # evaluation, because which names are held is a property of the book
        # being simulated and not of the market on that day. Moving it is what
        # makes one scan serve every variant.
        return [row for row in cache.days[day] if row[0].symbol not in open_trades]

    bench = benchmark.loc[benchmark.index <= day] if benchmark is not None else None
    scored = []

    for symbol, df in bars.items():
        # Without a cache, skip held names before doing the work - the result
        # is the same and the pass is faster. With one, evaluate everything
        # once so later variants holding different names still hit it.
        if cache is None and symbol in open_trades:
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

    if cache is None:
        return scored
    cache.days[day] = scored
    return [row for row in scored if row[0].symbol not in open_trades]


def _apply_null(scored, cfg, day, rng):
    """
    THE NULL. Replace the score with noise, and change nothing else.

    `rank_and_size` ranks on `t[4]`, so overwriting that one element hands the
    book a random ordering of exactly the same candidates - same gates, same
    triggers, same stops, same sizing, same sector cap, same ladder, same
    charges. A run with a seed and a run without differ in the SCORING and in
    nothing else, which is what makes the comparison a test of the scoring.

    APPLIED AFTER THE CACHE IS READ, deliberately. `_scan_day` stores the
    scored rows, so randomising inside it would either poison the cache for
    the real book or force the null to pay its own full scan pass. Here the
    null is nearly free and `scan_signature` stays identical - see
    `BOOK_ONLY_SWING_FIELDS`.

    THE RNG IS THREADED THROUGH `run`, not created here. One generator per
    backtest means the null is reproducible from its seed and that consecutive
    sessions draw different numbers; a generator seeded per day would give
    every session the same permutation, which is a very orderly kind of noise.
    """
    if rng is None:
        return scored
    return [(stock, found, metrics, n, float(rng.normal()), parts)
            for stock, found, metrics, n, _total, parts in scored]


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


class _TimeStopKind:
    """`.value` is all `_close` reads off an unmapped exit kind."""
    value = "time_stop"


TIME_STOP_KIND = _TimeStopKind()


@dataclass
class _TimeStop:
    """
    A decision-shaped stand-in for the time stop.

    `_close` reads `.kind` and `.detail` off whatever the ladder produced. The
    time stop is not a ladder rung - it is a constraint applied from outside -
    so rather than teach `ExitLadder` about calendars, or teach `_close` a
    second exit vocabulary, it presents the same two attributes.

    The kind carries its own `.value` so the trade is labelled `time_stop`
    rather than the generic `closed`. That distinction is the entire point of
    the variant: "how many trades did the clock end, and what did they earn"
    is unanswerable if they are pooled with every other non-stop exit.
    """
    kind: object = TIME_STOP_KIND
    detail: str = "time stop"


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
        closed_here = False
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
                closed_here = True
                break

        # THE TIME STOP, AND IT RUNS LAST ON PURPOSE. The ladder is given the
        # bar first, so a position that reached its stop or its target on the
        # final permitted day is booked at THAT level rather than at the close.
        # Running the clock first would convert real stops into time exits and
        # quietly improve the loss distribution - a flattering rewrite of what
        # happened, with no error anywhere.
        limit = cfg.swing.max_hold_days
        if limit and not closed_here and t.bars_held >= limit:
            fill = float(bar["close"])
            remaining = t.state.lots_remaining
            t.realised += (fill - t.entry) * remaining
            result.trades.append(
                _close(t, day, fill, _TimeStop(), costs, sectors))
            del open_trades[symbol]


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

    # `--years` sizes the TEST WINDOW, not just the fetch. The cache is
    # allowed to hold more history than was asked for - `fetch_swing_history`
    # pulls six years - and for a long time it silently did: a run labelled
    # "3 years" tested every session in the parquet, which was 5.4. The extra
    # bars are still loaded and still used, but only as WARM-UP: indicators
    # need history before the first scored session, and starting the window
    # at the first cached bar would score sessions whose EMAs were half
    # converged.
    window_start = date.today() - timedelta(days=int(args.years * 365))

    result = run(cfg, market, price_set.bars, price_set.benchmark,
                 stocks=stocks, start=window_start,
                 progress=lambda d, t, l: print(f"  [{d}/{t}] {l}",
                                                end="\r", flush=True))
    print(" " * 60, end="\r")

    for w in result.warnings[:5]:
        print(f"  ! {w}")
    print()
    print(f"{result.start} to {result.end}"
          f"{' - ' + str(result.symbols) + ' symbols' if result.symbols else ''}")
    print(result.headline())
    print()
    for k, v in result.metrics.as_dict().items():
        print(f"  {k:<26} {v}")
    print()
    for k, v in result.day_stats.as_dict().items():
        print(f"  {k:<26} {v}")
    print("\n  Where trades ended:")
    for label, count in result.exit_buckets.items():
        share = count / result.metrics.trades if result.metrics.trades else 0.0
        print(f"    {label:<26} {count:>4}  {share:>5.0%}")
    print(f"    {'Banked a partial':<26} {result.partials_banked:>4}")

    if result.by_setup:
        print("\n  Per setup:")
        for key, m in result.by_setup.items():
            print(f"    {key:<12} {m.trades:>4} trades  "
                  f"{m.win_rate:>5.0%}  {m.expectancy_r:+.3f}R")


if __name__ == "__main__":                                # pragma: no cover
    _main()
