"""
Central configuration. Every tunable number lives here - never hardcode
a parameter inside strategy logic, or you cannot sweep it in a backtest.
"""
from dataclasses import dataclass, field
from datetime import time

from .swing.markets import default_markets as _default_markets


@dataclass
class CapitalConfig:
    starting_capital: float = 100_000.0

    # --- Session governors (your stated rules) ---
    session_target_pct: float = 0.10   # +10% of capital -> stop for the day
    session_stop_pct: float = 0.05     # -5%  of capital -> stop for the day
    max_entries_per_session: int = 3   # 3 orders -> stop for the day

    # --- Give-back ratchet ---
    # "If I have a positive of 2% in the first order I would like to trail my
    # stoploss from 5% to 3%, and likewise."
    #
    # Expressed as a give-back cap off the day's PEAK realised P&L, so it keeps
    # working for any sequence of trades instead of needing a lookup table:
    #
    #     floor = max(-session_stop_pct, peak - ratchet_giveback_pct)
    #
    # With a 5% give-back that reproduces your rule exactly, and keeps going:
    #
    #     peak      floor     the day stop now reads
    #     +0%       -5%       5%   <- opening budget
    #     +2%       -3%       3%   <- your stated rule
    #     +4%       -1%       1%
    #     +6%       +1%       day can no longer finish red
    #     +8%       +3%
    #
    # The "3%" is not a second parameter - it is what a 5% give-back from a
    # +2% peak reads as on the day's P&L. The floor is monotonic because the
    # peak is; it never loosens.
    #
    # Note that the give-back equals the opening budget, so you always have
    # exactly three trades' worth of risk (3 x 1.667%) of room below the peak,
    # which is what keeps this consistent with max_entries_per_session.
    ratchet_giveback_pct: float = 0.05   # max give-back from the day's peak
    ratchet_arm_at_pct: float = 0.0      # peak needed before trailing starts;
                                         # 0.0 trails from the first rupee banked,
                                         # 0.02 would wait for your +2% before arming

    # --- Derived per-trade levels ---
    # Deliberately derived, not independently set, so the three governors
    # above stay mathematically consistent with each other.
    @property
    def risk_per_trade_pct(self) -> float:
        return self.session_stop_pct / self.max_entries_per_session

    @property
    def reward_per_trade_pct(self) -> float:
        return self.session_target_pct / self.max_entries_per_session

    @property
    def risk_per_trade_rupees(self) -> float:
        return self.starting_capital * self.risk_per_trade_pct

    @property
    def reward_per_trade_rupees(self) -> float:
        return self.starting_capital * self.reward_per_trade_pct

    @property
    def reward_risk_ratio(self) -> float:
        """
        Reward:risk, derived rather than set - so it cannot contradict the
        governors it is computed from.

        Guarded because `risk_per_trade_pct` is itself `session_stop_pct /
        max_entries_per_session`, and a zero session stop makes this a
        division by zero. That took the WHOLE APP down rather than just the
        page that read it, because `app.py` prints this in the sidebar on
        every render. A ratio of 0.0 with no risk defined is the honest
        answer, and it leaves the misconfiguration visible instead of fatal.
        """
        risk = self.risk_per_trade_pct
        return self.reward_per_trade_pct / risk if risk else 0.0

    # --- the other pots ---
    # THREE POOLS, ONE FORMULA. Each of these is a separate BALANCE, never a
    # separate formula: `risk_per_trade_pct` above stays the only place the
    # session governors are turned into a per-trade number, and every pool
    # runs through it. Adding a second formula is how two books end up with
    # two versions of the same risk rule that drift apart.
    #
    #   starting_capital     the intraday NIFTY option account
    #   swing_capital_inr    Indian cash equity held for days - a DIFFERENT
    #                        pot from the option account even though both are
    #                        rupees at the same broker, because one balance
    #                        funding two books means a drawdown in one
    #                        silently shrinks the size of the other
    #   foreign_capital_inr  money remitted under LRS, in another broker
    #                        entirely and unreachable from a domestic trade
    #
    # The last two are 0 until you fund them, and a scan against an unfunded
    # pool STANDS THE MARKET DOWN rather than quietly sizing off another pot.
    swing_capital_inr: float = 0.0
    foreign_capital_inr: float = 0.0

    @property
    def swing_risk_per_trade_inr(self) -> float:
        return self.swing_capital_inr * self.risk_per_trade_pct

    @property
    def swing_reward_per_trade_inr(self) -> float:
        return self.swing_capital_inr * self.reward_per_trade_pct

    @property
    def foreign_risk_per_trade_inr(self) -> float:
        return self.foreign_capital_inr * self.risk_per_trade_pct

    @property
    def foreign_reward_per_trade_inr(self) -> float:
        return self.foreign_capital_inr * self.reward_per_trade_pct

    #: pool key -> (attribute holding the balance, human label). Spelled once
    #: so the balance, the label and the error message cannot disagree. The
    #: keys are `swing.markets.POOL_*`, imported by value rather than by name
    #: to keep `config` free of a cycle back into `swing`.
    POOLS = {
        "home": ("starting_capital", "intraday option account"),
        "swing_india": ("swing_capital_inr", "Indian swing pot"),
        "foreign": ("foreign_capital_inr", "foreign (LRS) pool"),
    }

    def _pool(self, pool: str) -> tuple[str, str]:
        try:
            return self.POOLS[pool]
        except KeyError:
            # Deliberately loud. The previous version returned the domestic
            # balance for any unrecognised pool, so a typo sized a trade off
            # the wrong account and produced an entirely plausible ticket.
            raise KeyError(
                f"unknown capital pool {pool!r} - registered pools are "
                f"{', '.join(sorted(self.POOLS))}"
            ) from None

    def capital_inr(self, pool: str) -> float:
        """The balance backing `pool`, in rupees."""
        return float(getattr(self, self._pool(pool)[0]))

    def risk_inr(self, pool: str) -> float:
        """Per-trade risk budget in rupees, for any pool."""
        return self.capital_inr(pool) * self.risk_per_trade_pct

    def reward_inr(self, pool: str) -> float:
        """Per-trade reward target in rupees, for any pool."""
        return self.capital_inr(pool) * self.reward_per_trade_pct

    def pool_label(self, pool: str) -> str:
        """How to name this pot to a human, e.g. in a stand-down message."""
        return self._pool(pool)[1]

    def pool_field(self, pool: str) -> str:
        """The config attribute to point someone at when a pot is unfunded."""
        return f"capital.{self._pool(pool)[0]}"


@dataclass
class InstrumentConfig:
    """
    VERIFY THESE ON THE NSE SITE BEFORE GOING LIVE.
    Lot sizes have been revised three times since Nov 2024.
    """
    symbol: str = "NIFTY"
    lot_size: int = 65               # Nifty, as revised Jan 2026
    tick_size: float = 0.05
    strike_step: int = 50

    # Liquidity gates - refuse to trade an illiquid contract
    max_spread_pct_of_premium: float = 0.005   # 0.5%
    min_open_interest: int = 100_000
    min_premium: float = 30.0        # below this, spread dominates
    max_premium: float = 400.0       # above this, one lot breaks the budget


@dataclass
class SessionConfig:
    market_open: time = time(9, 15)
    entry_start: time = time(9, 30)    # skip opening auction noise
    entry_cutoff: time = time(14, 30)  # no new entries late in the day
    force_exit: time = time(15, 10)    # flat before close, always
    market_close: time = time(15, 30)

    opening_range_minutes: int = 15
    bar_interval_minutes: int = 5

    # Theta on expiry day is brutal for buyers. Start with it off.
    trade_on_expiry_day: bool = False


@dataclass
class SignalConfig:
    atr_period: int = 14
    atr_stop_multiple: float = 1.0     # underlying stop = 1 x ATR

    # Level detection
    pivot_lookback: int = 5            # bars each side for a swing point
    level_cluster_atr_frac: float = 0.3
    min_level_touches: int = 2

    # Breakout confirmation
    breakout_buffer_atr_frac: float = 0.15
    volume_surge_multiple: float = 1.5
    min_body_to_range: float = 0.50    # conviction candle

    # Doji reversal variant
    max_doji_body_to_range: float = 0.15
    doji_confirm_bars: int = 1

    # Strike selection
    max_delta: float = 0.45            # hard ceiling regardless of math
    min_delta: float = 0.20            # below this, gamma/theta risk too high

    # News / event blackout (minutes around a scheduled event)
    event_blackout_minutes: int = 30


@dataclass
class StrategyConfig:
    """
    Parameters for the strategies added on top of LevelBreakStrategy.
    Grouped by strategy so a sweep can vary one family at a time.
    """
    # --- Trendline break ---
    trendline_min_r2: float = 0.75      # below this it is not a line, it is hope
    trendline_min_points: int = 3
    trendline_break_atr_frac: float = 0.15

    # --- VWAP ---
    vwap_band_sigma: float = 1.0
    vwap_reclaim_atr_frac: float = 0.10   # how far beyond VWAP counts as a reclaim
    vwap_min_distance_atr: float = 0.25   # must have been meaningfully away first

    # --- Opening range breakout ---
    orb_retest_max_bars: int = 6          # retest must arrive within this window
    orb_retest_tolerance_atr: float = 0.25

    # --- Failed breakout / liquidity sweep ---
    sweep_min_wick_atr: float = 0.35      # wick must genuinely exceed the level
    sweep_max_close_back_atr: float = 0.20

    # --- Trend pullback ---
    ema_fast: int = 9
    ema_slow: int = 21
    pullback_max_distance_atr: float = 0.35
    pullback_min_trend_bars: int = 5

    # --- Volatility squeeze ---
    squeeze_lookback: int = 7             # NR7
    squeeze_range_percentile: float = 0.25
    squeeze_expansion_atr_frac: float = 0.20

    # --- Gap playbook ---
    gap_min_atr: float = 0.5
    gap_max_atr: float = 3.0              # beyond this it is news, stay out

    # --- Shared ---
    min_confidence: float = 0.25          # below this, do not alert at all


@dataclass
class RegimeConfig:
    """
    Day classification. Option BUYING needs expansion; in a range, theta wins.
    """
    expansion_or_atr_multiple: float = 1.2   # OR width >= this x ATR -> expansion
    range_or_atr_multiple: float = 0.6       # OR width <= this x ATR -> range
    vwap_slope_lookback: int = 12
    gap_atr_multiple: float = 0.5
    enforce_regime_gate: bool = True         # off = every strategy may fire


@dataclass
class TradeManagementConfig:
    """
    What happens AFTER the entry. Everything here is denominated in R, where
    1R is `CapitalConfig.risk_per_trade_rupees`, so the exit ladder speaks the
    same units as the entry stop.
    """
    enable_runner: bool = True

    breakeven_at_r: float = 1.0        # +1R -> stop to entry, trade cannot lose
    partial_exit_at_r: float = 2.0     # +2R -> bank part of the position
    partial_exit_lots: int = 1         # how many lots to release at that point

    # A SHARE book cannot use `partial_exit_lots`. One NIFTY lot is 65 and is
    # indivisible, so "bank one lot" is the only expressible partial there;
    # a swing ticket is 6 shares and wants to bank 3. When this is set it
    # replaces `partial_exit_lots` with `floor(lots_total * fraction)`.
    # None everywhere except the swing book, so the option ladder is
    # untouched - see `ExitLadder.partial_units`.
    partial_exit_fraction: float | None = None

    trail_atr_multiple: float = 1.0    # runner trails 1 ATR behind the underlying

    # NSE requires order quantities in exact multiples of the lot size, so a
    # partial exit is impossible on a single lot. Size two by default and fall
    # back to one - with the runner disabled - when two will not fit the risk
    # budget or the free capital.
    preferred_lots: int = 2
    min_lots: int = 1


@dataclass
class BrokerConfig:
    """
    Order placement. `dry_run` is the only thing standing between this system
    and your money; it defaults to True and nothing in the code path flips it
    automatically.
    """
    name: str = "kite"
    dry_run: bool = True

    exchange: str = "NFO"
    product: str = "MIS"               # intraday; this system is never overnight
    order_type: str = "LIMIT"          # never MARKET on an option book
    limit_buffer_ticks: int = 2        # cross the spread by this much to fill
    variety: str = "regular"


@dataclass
class EquityBrokerConfig:
    """
    Cash-equity order placement for the swing book.

    A SIBLING of BrokerConfig, not a replacement. That one is NFO/MIS and its
    docstring is right that "this system is never overnight" - which is
    exactly why it cannot describe a book that holds for days. Two profiles,
    two products, no `if market == ...` anywhere.

    `dry_run` defaults True and nothing in the code path flips it, same as the
    option side.

    THE STOP LIVES AT THE BROKER HERE, and that is the opposite of
    `kite_orders.modify_stop()` - deliberately. An intraday option stop is
    recomputed every 5-minute bar, so a resting order would need modifying on
    every one of them. A swing stop moves at most once a day, so a resting GTT
    is both cheap and the only version of the stop that survives the laptop
    being shut.
    """
    dry_run: bool = True

    exchange: str = "NSE"
    product: str = "CNC"               # delivery. MIS would square off at close
    order_type: str = "LIMIT"          # never MARKET
    variety: str = "regular"
    tick_size: float = 0.05

    # A GTT places a LIMIT order when it triggers, and if that limit does not
    # fill the same day the GTT is CANCELLED rather than retried. So the limit
    # has to sit past the trigger: above it on a buy, below it on a sell.
    # Zerodha's own guidance is that this "acts like a market order with the
    # protection of your limit set".
    gtt_limit_buffer_pct: float = 0.003

    tag: str = "swingbook"             # <=20 alphanumeric chars, Kite's cap

    # Zerodha's account cap is far higher; this is a sanity ceiling so a bug
    # that re-arms in a loop runs out of room long before it runs out of API.
    max_active_gtts: int = 50

    # Whether DDPI (or the older POA) is active on the account. When False,
    # every delivery SELL needs a CDSL TPIN authorisation that is valid for
    # ONE TRADING DAY - so a resting stop GTT placed on Monday is REJECTED on
    # Wednesday unless the account was re-authorised that morning after 07:00.
    # It still displays as active in Kite either way, which is why this is a
    # setting the UI reads rather than an assumption the code makes.
    ddpi_active: bool = False


@dataclass
class DataConfig:
    provider: str = "csv"              # "csv" | "kite" | "yfinance" | "fyers" | "dhan"
    symbol: str = "NIFTY"
    yfinance_ticker: str = "^NSEI"
    interval_minutes: int = 5
    lookback_days: int = 5
    csv_path: str = "data/nifty_5m.csv"
    vix_csv_path: str = "data/india_vix_daily.csv"
    poll_seconds: int = 30
    max_data_gap_bars: int = 3         # exceed this -> kill switch


@dataclass
class AlertConfig:
    enable_inapp: bool = True
    enable_telegram: bool = False
    enable_desktop: bool = False
    enable_email: bool = False

    # A 5-minute bar is re-evaluated on every refresh. Without these two you
    # get the same alert twenty times.
    dedupe_window_minutes: int = 30
    per_strategy_cooldown_minutes: int = 15

    force_exit_reminder: bool = True   # non-negotiable #2, at session.force_exit


@dataclass
class BacktestConfig:
    train_months: int = 6
    test_months: int = 2
    mode: str = "underlying"           # "underlying" | "synthetic_premium"
    assumed_iv: float = 0.14           # fallback when no VIX reading for the day
    use_vix_iv: bool = True            # calibrate per-day IV from India VIX
    vix_to_iv_scale: float = 1.0       # VIX/100 is already an annualised sigma
    risk_free_rate: float = 0.065
    max_bars_in_trade: int = 60        # exit at this many bars if neither hit
    apply_trade_management: bool = True  # simulate the exit ladder, not a flat 2:1
    apply_session_governors: bool = True  # simulate the day rules, not just signals


@dataclass
class HalalConfig:
    """
    Shariah screening thresholds.

    Denominator is TOTAL ASSETS, not market cap. Both are defensible - AAOIFI
    and Dow Jones use market cap - but a market-cap denominator makes the
    verdict move with the share price, so a stock can be compliant on Monday
    and non-compliant on Friday having filed nothing. For a screen that reruns
    every day that is noise, not information. Total assets only moves when a
    new balance sheet lands, which is when the answer should actually change.

    WHAT THIS CANNOT DO: the non-compliant-revenue test (haram income <= 5% of
    total revenue) and the purification amount need a line-item read of the
    annual report. Neither is derivable from any free data source, so neither
    is attempted here. Every verdict this module produces says so.
    """
    # --- FTSE / Yasaar: denominator is TOTAL ASSETS. This is what HLAL tracks,
    # and it is the primary screen for the reason in the docstring above.
    debt_to_assets_max: float = 0.33
    cash_and_interest_to_assets_max: float = 0.33
    receivables_to_assets_max: float = 0.49

    # --- AAOIFI / S&P: denominator is MARKET CAP. This is what SPUS tracks.
    # Computed alongside the primary screen and reported, never gating, so
    # that "compliant on FTSE, fails AAOIFI" is visible as the real state of
    # affairs rather than resolved silently in one standard's favour. It is
    # nearly free - fundamentals.py already reads market cap.
    aaoifi_debt_max: float = 0.30
    aaoifi_cash_and_interest_max: float = 0.30
    aaoifi_receivables_max: float = 0.49

    #: Which standard decides `eligible`. Changing this changes which stocks
    #: are tradeable, so it is one field in one place rather than a flag
    #: threaded through the scan.
    primary_method: str = "ftse"
    #: Which standards are computed at all. Both, by default: the second one
    #: costs no extra request and its disagreements are the useful output.
    compute_methods: tuple = ("ftse", "aaoifi")

    # --- contested activity categories (GICS/Yahoo vocabulary only) ---
    # Each names the standard it comes from. SPUS (AAOIFI/S&P) excludes the
    # first two; HLAL (FTSE/Yasaar) does not. Neither is a mistake, so neither
    # is hardcoded.
    exclude_defence: bool = True             # aerospace & defence
    exclude_exchanges_and_data: bool = True  # SPGI, MSCI, ICE, CME
    exclude_shell_companies: bool = True     # no operating business to screen

    def methods(self) -> tuple:
        """
        The methodologies to compute, primary first.

        Primary first so a reader of `verdicts` sees the gating standard at
        the top, and so a misconfigured `compute_methods` that omits the
        primary still computes it.
        """
        rest = [m for m in self.compute_methods if m != self.primary_method]
        return (self.primary_method, *rest)

    # Fail closed. A missing balance sheet is not evidence of compliance, and
    # the cost of the two errors is not symmetric: wrongly excluding a halal
    # stock costs you one candidate out of a hundred, wrongly including a
    # haram one defeats the entire purpose of the screen.
    exclude_on_missing_data: bool = True

    overrides_csv: str = "data/halal_overrides.csv"


@dataclass
class SwingConfig:
    """
    Daily Nifty 100 swing scanner.

    A DIFFERENT BOOK FROM THE REST OF THIS SYSTEM. Everything else here is
    intraday NIFTY options, flat by 15:10. This is cash equity held for days.

    Capital and reward:risk are deliberately ABSENT - they are read from
    CapitalConfig, so both books size off one set of governors and cannot
    drift apart. `top_n` is 3 because `max_entries_per_session` is 3.
    """
    # The universe file, benchmark, price floor and turnover floor were the
    # four numbers in here that were secretly India-only. They now live per
    # market in `markets.py`; everything remaining in this class is genuinely
    # market-agnostic and applies unchanged to a New York or London bar.
    markets: dict = field(default_factory=_default_markets)
    default_market: str = "india"
    cache_dir: str = "data/cache"
    top_n: int = 3
    history_days: int = 400            # ~1 trading year plus the 200d warm-up

    # --- trend / setup, on DAILY bars ---
    # 20/50 rather than the intraday 9/21: a 9-day EMA on daily bars is a
    # two-week average, which is noise at this horizon.
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    # The stop is placed at structure and then CLAMPED into this band, so it
    # is never tighter than daily noise nor wider than the risk budget can
    # size. A stop is only a real stop if it sits where the idea is wrong.
    swing_atr_stop_min_multiple: float = 1.0
    swing_atr_stop_multiple: float = 1.5   # wider than intraday - overnight gaps
    stop_structure_bars: int = 10          # swing low searched over this window
    measured_move_bars: int = 40           # base height, when there is no
                                           # resistance left overhead
    target_min_atr: float = 1.0            # a target closer than this is noise
    # ...and a target further than this is not reachable inside the holding
    # window. Over ~10 sessions a stock covers roughly sqrt(10) ATR, so a
    # level 8 ATR away is a real level and a fictional target. The ticket
    # shows the reachable figure and says when it has been capped.
    target_max_atr: float = 4.0
    max_entry_distance_atr: float = 1.0    # trigger further away than this is
                                           # not today's trade
    breakout_buffer_atr_frac: float = 0.15
    pivot_lookback: int = 5
    level_cluster_atr_frac: float = 0.5
    min_level_touches: int = 2
    volume_surge_multiple: float = 1.5
    squeeze_lookback: int = 7
    pullback_max_distance_atr: float = 1.0
    sweep_min_wick_atr: float = 0.35
    sweep_max_close_back_atr: float = 0.20

    # --- tradeability ---
    # `min_price` and the turnover floor are per-market (see markets.py) - they
    # are denominated in the exchange's own currency and cannot be one number.
    # These two are ratios, so they travel.
    min_atr_pct: float = 0.008          # below this there is no move to catch
    max_atr_pct: float = 0.070          # above this the 1.5-ATR stop is unaffordable

    # --- relative strength ---
    rs_short_days: int = 21
    rs_long_days: int = 63

    # --- gates ---
    earnings_blackout_days: int = 5     # results due within N sessions -> stand aside
    max_per_sector: int = 1             # three picks in one sector is one bet
    news_finalists: int = 15            # only fetch news for this many candidates
    news_lookback_hours: int = 72
    news_max_items: int = 25            # newest N vote; an unbounded
                                        # count lets one widely-covered
                                        # name outweigh a quiet one
    valid_for_days: int = 5             # an untriggered entry expires after this

    # --- which archetypes may fire ---
    # `setup.detect` tries every builder listed here and nothing else. It is a
    # config field rather than a backtest-only filter because a backtest that
    # can trade a subset the live scanner cannot is no longer measuring the
    # system being traded (invariant 1). Measured finding that made it worth
    # having: breakout was 209 of 282 trades, pullback 71, reclaim 2, squeeze
    # ZERO - the book calls itself four archetypes and trades as one.
    enabled_setups: tuple = ("breakout", "pullback", "squeeze", "reclaim")

    # --- market regime ---
    # Days of the benchmark moving average new longs must be above. 0 = off,
    # which is how every result before this existed was produced. See
    # market_regime.py for why a long-only breakout book wants one.
    regime_ma_days: int = 0

    # --- after the entry ---
    # The runner trails this many ATRs behind the peak. Wider than the
    # intraday book's 1.0 for the same reason `swing_atr_stop_multiple` is
    # 1.5: an overnight hold has to survive gaps that a 15:10 flat never sees.
    trail_atr_multiple: float = 1.5
    # Where the stop moves to breakeven, in R. None inherits the intraday
    # book's `TradeConfig.breakeven_at_r`. It is overridable here for the same
    # reason `trail_atr_multiple` is: +1R on a 5-minute bar and +1R on a daily
    # bar are different distances relative to the noise that has to be
    # survived, and a shift that fires inside daily noise converts winners
    # into scratches. Set it high (99) to disable the rung and trail only.
    breakeven_at_r: float | None = None
    # Half the position is banked at +2R and the rest runs. Whole shares, so
    # a 1-share ticket cannot split and takes a full exit at +2R instead -
    # the same fallback the option book uses on a single lot.
    partial_exit_fraction: float = 0.5
    # TOTAL open risk, in R, across every open position at once. `top_n` caps
    # how many a single scan proposes; nothing capped how many could be open
    # simultaneously, so three scans on three days could put nine positions
    # on. 3.0 keeps the worst case - everything stops out together - at
    # exactly the session stop the governors are built around.
    max_open_risk_r: float = 3.0

    # --- composite score weights (must sum to 1.0) ---
    w_setup: float = 0.30
    w_relative_strength: float = 0.25
    w_reward_risk: float = 0.15
    w_volume: float = 0.10
    w_position_52w: float = 0.10
    w_news: float = 0.10

    # --- cache lifetimes ---
    price_cache_hours: int = 12         # daily bars change once a day
    fundamentals_cache_days: int = 7    # balance sheets change once a quarter

    halal: HalalConfig = field(default_factory=HalalConfig)

    def score_weights(self) -> dict[str, float]:
        return {
            "setup": self.w_setup,
            "relative_strength": self.w_relative_strength,
            "reward_risk": self.w_reward_risk,
            "volume": self.w_volume,
            "position_52w": self.w_position_52w,
            "news": self.w_news,
        }


@dataclass
class PortfolioConfig:
    """
    Which brokers this account actually uses, for `nifty_algo/portfolio/`.

    THIS LIST IS A CLAIM ABOUT YOUR ACCOUNTS, and it decides how a missing
    answer is read. A connector NOT in it is never called and never counted -
    which is what keeps the registered-but-unimplemented IBKR stub from
    marking every snapshot incomplete. A connector that IS in it and cannot
    answer makes the snapshot incomplete, and portfolio percentages are then
    withheld rather than computed against a partial book.

    So adding "ibkr" here before it is implemented is not a no-op: it is you
    saying "I hold things there", and every report will correctly refuse to
    quote a weight until it can see them.
    """
    connectors: tuple = ("manual", "kite")
    manual_path: str = "data/manual_positions.csv"

    # Sessions of daily bars behind the correlation matrix and the beta
    # figures in the risk report. ~1 trading year: long enough that a single
    # earnings gap does not dominate, short enough to describe the regime you
    # are actually in. The report prints the window and the sample size,
    # because a correlation without an n is a decoration.
    correlation_days: int = 250
    #: Below this many overlapping sessions, a pair's correlation is reported
    #: as unavailable rather than computed. Two names that share 12 bars will
    #: happily produce a 0.9.
    min_correlation_sessions: int = 60


@dataclass
class Config:
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    trade: TradeManagementConfig = field(default_factory=TradeManagementConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    equity_broker: EquityBrokerConfig = field(default_factory=EquityBrokerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)


DEFAULT = Config()
