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

    #: The short-premium book's own entry count - see `POOL_ENTRIES`. Six
    #: rather than three because that book runs a PORTFOLIO of strategies
    #: concurrently and each leg of a strangle is an entry; at six the
    #: per-trade risk is 0.833% and the session floor is unchanged at 5%.
    short_premium_max_entries_per_session: int = 6

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
    def entries_for(self, pool: str) -> int:
        """How many entries `pool`'s session allows. Validates the key."""
        self._pool(pool)          # raise on a typo, exactly as capital_inr does
        attr = self.POOL_ENTRIES.get(pool)
        n = getattr(self, attr) if attr else self.max_entries_per_session
        return max(int(n), 1)

    def risk_pct_for(self, pool: str) -> float:
        """
        THE formula, and the only place it is written.

        `risk_per_trade_pct` below is this function applied to the option
        account, kept as a property because the engine, the alerts and the
        sidebar all read it by that name. Adding a second expression of
        `session_stop_pct / entries` is how two books end up with two
        versions of the same risk rule that drift apart.
        """
        return self.session_stop_pct / self.entries_for(pool)

    @property
    def risk_per_trade_pct(self) -> float:
        return self.risk_pct_for("home")

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
    #   intraday_equity_capital_inr
    #                        cash equity traded INTRADAY on MIS and flat by
    #                        15:10. A fourth pot rather than a share of the
    #                        swing one for the same reason the swing pot is
    #                        not a share of the option one: a drawdown in a
    #                        book you check once a day must not silently
    #                        shrink the size of a book that trades every bar.
    #
    #   short_premium_capital_inr
    #                        intraday NIFTY option SELLING. A fifth pot, and
    #                        the one whose separation matters most: a short
    #                        premium book is margin-constrained rather than
    #                        cash-constrained, so its balance is consumed by
    #                        SPAN + exposure blocked at the broker, not by
    #                        the debit paid. Sharing it with the option
    #                        BUYING account would let one book's blocked
    #                        margin silently size the other book's trade.
    #
    # The last four are 0 until you fund them, and a scan against an unfunded
    # pool STANDS THE MARKET DOWN rather than quietly sizing off another pot.
    swing_capital_inr: float = 0.0
    foreign_capital_inr: float = 0.0
    intraday_equity_capital_inr: float = 0.0
    short_premium_capital_inr: float = 0.0

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
        "intraday_equity": ("intraday_equity_capital_inr",
                            "intraday equity (MIS) pot"),
        "short_premium": ("short_premium_capital_inr",
                          "intraday short-premium pot"),
    }

    #: pool key -> the attribute holding ITS entry count. Everything not
    #: listed uses `max_entries_per_session`, so this dict is empty of the
    #: four books that share the three-entry day.
    #:
    #: ONE FORMULA, NOT ONE NUMBER. `risk_pct_for` below is still the only
    #: expression of `session_stop_pct / entries` anywhere in the system;
    #: this only lets a book say how many entries its day has. A seller
    #: takes more positions per session than a buyer of three lottery
    #: tickets, and forcing it to 3 would either oversize every ticket or
    #: leave two thirds of the day's budget unused.
    POOL_ENTRIES = {
        "short_premium": "short_premium_max_entries_per_session",
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
        return self.capital_inr(pool) * self.risk_pct_for(pool)

    def reward_inr(self, pool: str) -> float:
        """Per-trade reward target in rupees, for any pool."""
        return self.capital_inr(pool) * (
            self.session_target_pct / self.entries_for(pool))

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

    #: Prior trading days prepended to `Context.bars` so indicators are
    #: already warm at 09:15.
    #:
    #: WHY THIS EXISTS. `Context.bars` is one session (engine.py:165,
    #: backtest.py:413), so a 30-bar warm-up is spent INSIDE the day: the
    #: first decidable bar is 11:45 and `entry_start = 09:30` is dead config.
    #: The book cannot see the open, the opening range break, or the morning
    #: trend - the most volatile hours of the session, and the ones an option
    #: BUYER needs, since a long option is long vega and short theta. It also
    #: means the regime is classified 31 bars in, against a mature 5-minute
    #: ATR of ~15 points, which is what makes `gap_atr_multiple = 0.5` fire on
    #: an 8-point overnight move and tag almost every day GAP_DAY.
    #:
    #: WHY IT DEFAULTS TO 0. Every result this project has ever produced was
    #: produced at 0, and turning it on changes what the book trades rather
    #: than how it sizes. It is a PRE-REGISTERED VARIANT in the sense
    #: `swing.experiment` means it - measure it against 0 on out-of-sample
    #: data and keep whichever wins, the same way `regime_ma_days = 0`
    #: preserves the pre-regime-filter baseline.
    #:
    #: SAFE ONLY BECAUSE the session-anchored reads are now day-aware -
    #: `signals.last_session` / `signals.session_open`. Raising this before
    #: that change would have made `opening_range` return a finished day's
    #: range, silently.
    warmup_sessions: int = 0

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
class IntradayEquityBrokerConfig:
    """
    Cash-equity order placement for the INTRADAY book. The third profile.

    `BrokerConfig` is NFO/MIS, `EquityBrokerConfig` is NSE/CNC/GTT, and this
    is NSE/MIS. Three profiles, three products, no `if book == ...` anywhere.

    `dry_run` defaults True and nothing in the code path flips it.

    NO ddpi_active, AND NO GTT FIELDS - both absences are the point. A CDSL
    TPIN authorisation is required for a DELIVERY sell, because the shares
    leave your demat account. An MIS position is squared off intraday and
    never reaches demat, so no TPIN is involved and the whole "looks armed
    and is not" failure that `EquityBrokerConfig` documents cannot occur
    here. Do not copy `ddpi_active` across; it would be a field that is
    always irrelevant and therefore always ignored.

    THE STOP IS A PAIR, and that is a third answer rather than a repeat of
    either sibling's. `kite_orders.modify_stop()` rests NOTHING because an
    intraday option stop is recomputed every 5-minute bar. `kite_equity`
    rests a two-leg GTT because a swing stop moves at most once a day. Here
    the working stop is recomputed every bar like the option book's - but the
    consequence ("the stop exists only while the runner runs") is worse on
    cash equity, because a crashed runner at 10:00 leaves an unstopped
    position until the broker squares it off around 15:20. So the app holds
    the working ATR stop AND a plain stop order rests at the wide disaster
    level. That one never moves, so it costs no modify-per-bar, and it
    survives the laptop being shut.

    SL-M is chosen deliberately over a Bracket or Cover order: Zerodha
    withdrew Bracket Orders in March 2020, and Cover Orders carry their own
    leverage constraints. A plain stop order depends on neither.
    """
    dry_run: bool = True

    exchange: str = "NSE"
    product: str = "MIS"               # intraday. Squared off by the broker.
    order_type: str = "LIMIT"          # never MARKET on the entry
    stop_order_type: str = "SL-M"      # the resting disaster stop
    variety: str = "regular"
    tick_size: float = 0.05
    limit_buffer_ticks: int = 2        # cross the spread by this much to fill

    tag: str = "dayequity"             # <=20 alphanumeric chars, Kite's cap

    #: Stop asking the broker for new entries after this. Distinct from
    #: `SessionConfig.force_exit`: entries stop before exits do.
    square_off_at: time = time(15, 10)

    #: Sanity ceiling so a bug that re-arms in a loop runs out of room long
    #: before it runs out of API. Same reasoning as `max_active_gtts`.
    max_open_positions: int = 5


@dataclass
class IntradayEquityConfig:
    """
    Intraday cash equity on the Nifty 100. The THIRD book.

    Neither of the other two describes it. The option book is intraday but
    trades one instrument and selects strikes; the swing book is cash equity
    but holds for days and rests its stop at the broker. This one sweeps a
    universe every 5-minute bar and is flat by 15:10.

    Capital and reward:risk are ABSENT for the same reason they are absent
    from SwingConfig - they are read from CapitalConfig, so all three books
    size off one set of governors. `top_n` is 3 because
    `max_entries_per_session` is 3.

    The MARKET is borrowed from `swing.markets` as a DATA DESCRIPTOR only -
    universe file, benchmark ticker, currency symbol, yfinance suffix. This
    book deliberately does NOT read `Market.capital_pool` or
    `Market.min_avg_turnover`: its pot is its own, and its liquidity floor is
    an intraday one far above a daily swing floor. Registering a second India
    `Market` instead would pollute `markets.keys(cfg)`, which is the
    `--market` choice list for the swing CLIs.
    """
    market: str = "india"
    cache_dir: str = "data/cache"
    top_n: int = 3                     # concurrent positions, = max entries/day

    # --- bars ---
    bar_interval_minutes: int = 5
    history_years: float = 3.0

    # --- the entry window ---
    # NOTE these are the OUTER bounds, not the effective ones. A strategy
    # cannot fire until `BaseStrategy._preflight` has its 25 bars, and
    # `Context.bars` is TODAY'S SESSION ONLY (engine.py:165), so warm-up
    # happens inside the session: 25 x 5min = 125 minutes, i.e. nothing fires
    # before ~11:20. That is a real property of the live book rather than a
    # backtest artefact, and `warmseed` is the pre-registered variant that
    # tests widening it. Do NOT "fix" it by prepending prior-day bars - that
    # breaks `signals.opening_range`, which reads the first bars OF THE FRAME
    # with no day-awareness at all, and would silently return yesterday's
    # opening range on every session.
    entry_start: time = time(9, 30)
    entry_cutoff: time = time(14, 30)
    force_exit: time = time(15, 10)

    # --- the stop, and the two bounds on it ---
    # THE STOP MULTIPLE IS THE MOST CONSEQUENTIAL NUMBER IN THIS BOOK, and
    # 1.0 - the option book's SignalConfig default - is the wrong value here.
    # Two forces pull against each other, and both are arithmetic rather than
    # opinion. Measured 5-minute ATR is 0.257% of price and the median stock's
    # whole daily high-low range is 2.04%:
    #
    #   xATR  stop%   friction  breakeven  lev needed  2R move (% of day's range)
    #    1.0  0.257%   0.556R     51.9%       6.5x      0.51%   (25%)
    #    2.0  0.514%   0.292R     43.1%       3.2x      1.03%   (50%)
    #    3.0  0.771%   0.204R     40.1%       2.2x      1.54%   (76%)
    #    4.0  1.028%   0.160R     38.7%       1.6x      2.06%   (101%)
    #
    # FRICTION FORCES A WIDE STOP: costs and slippage are roughly fixed in
    # percent, so at 1x ATR they eat 0.556R and the book must win 52% of a
    # 2:1 payoff merely to break even. THE SESSION'S FINITE RANGE FORCES A
    # NARROW ONE: at 4x ATR the +2R partial needs the trade to travel the
    # stock's ENTIRE median daily range in one direction, from the entry,
    # before 15:10.
    #
    # 2.0 is chosen because it is the tightest multiple whose leverage
    # requirement (3.2x) fits inside Zerodha's ~5x MIS allowance, so sizing is
    # decided by RISK rather than by the size of the pot. This was derived
    # from measured volatility and published costs BEFORE any expectancy was
    # computed - it is a pre-registered design choice, not a tuned one - and
    # `stoptight`/`stopwide` sweep it out-of-sample.
    atr_period: int = 14
    atr_stop_multiple: float = 2.0

    # A 5% stop was the original brief. Measured on the cached Nifty 100
    # history, the MEDIAN stock's median daily high-low range is 2.04% and
    # the whole day's range exceeds 5% on only 3.7% of sessions - so a 5%
    # intraday stop is unreachable on ~96% of days. It is not a tight risk
    # control, it is NO control, and it would make 1R a 5% move so the
    # +1R/+2R ladder could never fire once. So it is a CEILING: a signal
    # whose ATR stop is wider than this is REJECTED, never clamped. Clamping
    # would cap the denominator of R while leaving the numerator free, which
    # manufactures enormous R multiples on exactly the most volatile names -
    # and looks like risk control while doing the opposite.
    max_stop_pct: float = 0.05
    # And a floor, because a stop tighter than the spread plus slippage is
    # not a stop, it is a guaranteed scratch. Set against the modelled round
    # trip: slippage is 0.05% a leg, so 0.10% there and back, and this floor
    # is 1.5x that. It also sits below the p10 5-minute ATR of 0.213%, so it
    # rejects only genuinely untradeable stops rather than quietly excluding
    # the calmer half of the universe.
    min_stop_pct: float = 0.0015

    # Refuse to chase a gap this far past the signal bar's close. The fill is
    # the NEXT bar's open; if that opened far above the trigger the setup is
    # gone, and filling at the signal bar's close instead would be fiction.
    max_chase_pct: float = 0.004

    # --- tradeability, in intraday terms ---
    # THESE ARE 5-MINUTE NUMBERS AND MUST NOT BE COPIED FROM SwingConfig.
    # `SwingConfig.min_atr_pct` is 0.008 because a DAILY bar of a Nifty 100
    # name has a measured median ATR of 2.22%. A 5-minute bar does not: by
    # sqrt-of-time over the 75 bars in an NSE session, the implied 5m ATR is
    #
    #     median 0.257%,  p10 0.213%,  p90 0.330%,  max 0.438%
    #
    # measured across the 98 symbols in data/cache/daily_prices_india.parquet.
    # A floor of 0.004 - which is what a daily-shaped band looks like -
    # admitted ONE symbol of 98, so the scan returned nothing and read as
    # tight gates rather than as a misconfigured band. The floor below sits
    # well under p10 so it only catches a genuinely dead tape, and the ceiling
    # well over the daily-scaled maximum so it only catches a halt or a
    # corporate action the integrity gate missed.
    min_session_turnover_inr: float = 5.0e7   # Rs 5 crore in the session so far
    min_atr_pct: float = 0.0012
    max_atr_pct: float = 0.0150

    # AND THESE ARE THE DAILY ONES, WHICH ARE A DIFFERENT QUANTITY.
    # The morning prefilter ranks on PRIOR SESSIONS, so its ATR is a daily
    # ATR - measured median 2.22%, p10 1.85%, p90 2.86%. The pair above is a
    # 5-minute ATR, measured median 0.257%. They differ by a factor of ~8.7
    # (sqrt of the 75 bars in a session) and sharing one field between them
    # silently filters the universe against the wrong scale: a daily 2.2%
    # tested against a 1.5% intraday ceiling rejects EVERY name, and the scan
    # then reports no candidates rather than a misconfiguration.
    min_daily_atr_pct: float = 0.010
    max_daily_atr_pct: float = 0.060
    #: Never take more than this share of the fill bar's volume. Without it,
    #: risk-based sizing will happily buy a third of a 5-minute bar on a
    #: mid-cap, and the backtest will report that fill as free.
    participation_cap_pct: float = 0.02

    # --- the cross-sectional layer ---
    #: How many names survive the morning RS cut. Computed from PRIOR
    #: sessions only, so it is knowable at 09:15 - which is what makes it
    #: legitimate live and affordable in a backtest (a 5x saving in both).
    #: `norank` is the pre-registered variant that measures its cost.
    rs_prefilter_n: int = 20
    rs_short_days: int = 5
    rs_long_days: int = 21
    max_per_sector: int = 1

    # --- score weights ---
    w_setup: float = 0.35
    w_relative_strength: float = 0.25
    w_reward_risk: float = 0.15
    w_volume: float = 0.15
    w_session_position: float = 0.10

    # --- sizing ---
    #: MIS leverage MULTIPLIES BUYING POWER, NOT RISK APPETITE. It appears in
    #: exactly one expression - the affordability cap - and never in the
    #: numerator of the quantity. Default 1.0 so neither the backtest nor the
    #: live sizer ever spends margin the account might not be granted;
    #: raising it is a deliberate act.
    mis_leverage: float = 1.0

    #: Long only. Short selling is available on MIS, but the swing book is
    #: long-only and this keeps the two consistent.
    allow_short: bool = False

    #: The live scan screens; the BACKTEST CANNOT - there is no point-in-time
    #: fundamental data, exactly as documented for the swing backtest. So the
    #: backtest tests a LARGER universe than the live book trades, and says
    #: so in its caveats rather than quietly implying they are the same.
    halal_screen: bool = True

    #: Strategies from `strategies.registry`. An empty tuple means "whatever
    #: is default_enabled there", so this book cannot drift from the registry.
    enabled_strategies: tuple = ()

    #: Obey `regime.classify()` / `Strategy.allowed_regimes`. Stored on the
    #: cached signal and applied by the consumer, so sweeping it is free.
    enforce_regime_gate: bool = True
    min_confidence: float = 0.0

    # --- post-entry (the ladder) ---
    trail_atr_multiple: float = 1.0
    partial_exit_fraction: float = 0.5
    breakeven_at_r: float | None = None   # None inherits TradeManagementConfig
    max_open_risk_r: float = 3.0

    # --- caches ---
    instruments_cache_days: int = 1

    def score_weights(self) -> dict:
        return {
            "setup": self.w_setup,
            "relative_strength": self.w_relative_strength,
            "reward_risk": self.w_reward_risk,
            "volume": self.w_volume,
            "session_position": self.w_session_position,
        }


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
class ShortPremiumConfig:
    """
    Intraday NIFTY option SELLING. The FOURTH book.

    Every other book in this repo is long an instrument. This one is short
    premium, and almost every number below is the inverse of the option
    book's equivalent - not a tweak of it. Read `risk.py:193-230` first:
    that engine picks the HIGHEST delta that fits a debit budget, because a
    buyer wants directional capture per rupee of theta paid. A seller wants
    the lowest delta that still pays a credit worth collecting, and is
    constrained by MARGIN rather than by the premium it can afford.

    Capital, reward:risk and the session governors are ABSENT here for the
    same reason they are absent from SwingConfig and IntradayEquityConfig -
    they come from CapitalConfig, which gained a fifth POOL and no second
    formula.

    THE HONESTY CONSTRAINT. There is no historical option chain at any
    price (`backtest.py:6-11`, `data/kite_feed.py:8-16`), and
    `SYNTHETIC_PREMIUM` fixes IV for a trade's life with zero skew
    (`pricing.py:175`). For a buyer that is a caveat; for a seller it prices
    the entire edge AND the entire risk at zero. Nothing in this book may be
    validated against it. The route to a real backtest is `recorder.py`,
    which starts accumulating the day you run it and not before.
    """
    pool: str = "short_premium"

    # --- strike selection: the inverse of risk.viable_strikes ------------
    #: The delta band a seller lives in. The option book's floor is 0.20 and
    #: its ceiling 0.45; every strike in THIS band is refused there twice
    #: over, which is the clearest statement that these are two engines.
    min_delta: float = 0.05
    max_delta: float = 0.20

    #: Credit floors, both of them. The absolute one keeps you out of strikes
    #: where the tick and the spread dominate; the multiple keeps you out of
    #: strikes where the round trip eats the edge. A 3-rupee credit against a
    #: ~1.5-rupee round trip is not a trade, it is a donation with extra
    #: steps - and it is the mistake a delta filter alone will not catch.
    min_credit: float = 8.0
    min_credit_to_cost_multiple: float = 4.0

    #: Wider than the buyer's 0.5% (`InstrumentConfig`) in absolute terms
    #: because OTM wings genuinely quote wider - but it bites harder, since a
    #: seller crosses it on the way in AND on the way out, on a smaller
    #: premium. Expressed against the mid, like `OptionQuote.spread_pct`.
    max_spread_pct_of_premium: float = 0.020

    min_open_interest: int = 200_000
    min_volume: int = 1_000
    strikes_each_side: int = 15

    # --- exits, in R ------------------------------------------------------
    #: 1R is defined by this and nothing else: the stop is
    #: `credit x stop_at_credit_multiple`, so per-unit risk is
    #: `credit x (multiple - 1)`. At 2.0 the premium doubling is exactly -1R
    #: and buying back at zero is exactly +1R.
    #:
    #: THIS IS WHY THE LADDER IS RE-PARAMETERISED RATHER THAN REUSED AS
    #: CONFIGURED. `TradeManagementConfig.partial_exit_at_r` is 2.0, and a
    #: short can never reach +2R at any credit multiple <= 3. Left at the
    #: default the partial rung never fires, the runner never arms, and the
    #: trail is dead code - silently, with no error anywhere.
    stop_at_credit_multiple: float = 2.0
    breakeven_at_r: float = 0.25
    partial_exit_at_r: float = 0.50
    partial_exit_fraction: float = 0.5

    #: A CONSTANT fraction of R, not `atr x trail_atr_multiple / stop_points`
    #: the way the option book trails (`positions.py:405-406`). The
    #: underlying's ATR is not the risk driver for a theta position, and
    #: dividing by a premium-derived stop distance mixes two units.
    trail_distance_r: float = 0.15

    # --- the three stops --------------------------------------------------
    #: A premium stop alone is a VEGA stop: an IV spike with spot unchanged
    #: fires it at the worst possible price. An underlying stop alone is
    #: blind to vol. Both, whichever comes first, is the only honest answer -
    #: and the time stop (SessionConfig.force_exit) is the only one that is
    #: always available.
    #:
    #: THE UNDERLYING STOP IS ANCHORED TO THE SHORT STRIKE, NOT TO ENTRY SPOT.
    #: Anchoring it to spot at entry - "stop if spot moves 1.5 ATR against
    #: me" - looks reasonable and is wrong twice over. Selling a 0.15-delta
    #: call ~300 points OTM with a 100-point ATR puts that stop 150 points
    #: away: it fires on noise while the position is still deeply OTM and
    #: perfectly healthy, and it sits INSIDE the strike, so the strike is
    #: never reached and the strike-touch branch becomes dead code. Distance
    #: from the strike is what a seller's risk actually scales with, and it
    #: is the number that stays sane across strikes, expiries and vol.
    #:
    #: 0.5 means "exit when spot comes within half an ATR of the strike I am
    #: short". 0.0 means "exit on touch".
    underlying_stop_buffer_atr: float = 0.5

    #: `SessionConfig.trade_on_expiry_day` is False and its comment reasons
    #: it out FOR BUYERS - "theta on expiry day is brutal for buyers". A
    #: seller is on the other side of exactly that sentence and inherits
    #: vicious gamma with it. So this book states its own position rather
    #: than inheriting one argued for the opposite trade.
    trade_on_expiry_day: bool = False

    # --- the macro second check ------------------------------------------
    #: The gate FAILS CLOSED: an unavailable check blocks the entry, exactly
    #: as `swing/fx.py` stands a market down rather than guessing a rate.
    require_macro_gate: bool = True
    vix_percentile_lookback: int = 252
    #: Selling the bottom decile of the IV distribution is being paid nothing
    #: for real risk. Percentile, not level, because a VIX of 12 means
    #: something different in different years.
    min_vix_percentile: float = 0.20
    #: A rising-vol day is the one that runs a seller over.
    max_vix_1d_change_pct: float = 0.10
    max_overnight_gap_atr: float = 1.0

    # --- the recorder -----------------------------------------------------
    chain_dir: str = "data/chain"
    record_interval_seconds: int = 60
    record_expiries: int = 2

    #: Strategies from `short_premium.registry`. Empty means "whatever is
    #: default_enabled there", so this book cannot drift from its registry.
    enabled_strategies: tuple = ()
    min_confidence: float = 0.25

    def ladder_settings(self) -> "TradeManagementConfig":
        """
        The seller's rungs, as the object `ExitLadder` already accepts.

        `ExitLadder(cfg, trade=...)` takes a settings override - that is how
        the swing book gets its own partial size without a second ladder
        (`positions.py:104-116`). Returning a real TradeManagementConfig
        rather than a look-alike means the ladder cannot be handed an object
        missing a field it reads.
        """
        return TradeManagementConfig(
            enable_runner=True,
            breakeven_at_r=self.breakeven_at_r,
            partial_exit_at_r=self.partial_exit_at_r,
            partial_exit_lots=1,
            partial_exit_fraction=self.partial_exit_fraction,
            trail_atr_multiple=0.0,     # unused: the trail is in R, see above
            preferred_lots=1,
            min_lots=1,
        )


@dataclass
class ShortPremiumBrokerConfig:
    """
    THE THIRD SWITCH.

    `BrokerConfig.dry_run` guards the option book and
    `EquityBrokerConfig.dry_run` guards the swing book, deliberately as two
    flags so that going live on one is not a decision about the other.
    Selling naked premium is a third such decision, and the most consequential
    of them: it is the only book here whose loss is unbounded.
    """
    name: str = "kite"
    dry_run: bool = True
    exchange: str = "NFO"

    #: MIS, and here that is the RIGHT answer for the reason it is the wrong
    #: one in `kite_equity.py`. That book holds for days, so a broker
    #: square-off at 15:20 would turn every swing trade into an intraday one
    #: without telling you. This book is flat by 15:10 anyway, so the broker's
    #: own square-off sits BELOW our force exit as a backstop rather than
    #: above it as a surprise - and MIS is what makes the margin affordable.
    product: str = "MIS"

    #: LIMIT only. NSE disabled market orders in options, so this is not a
    #: preference; and a limit set past the trigger is Zerodha's own
    #: description of how to get market-like behaviour with protection.
    order_type: str = "LIMIT"
    limit_buffer_ticks: int = 2
    variety: str = "regular"

    #: A RESTING DISASTER STOP, which `kite_orders.modify_stop()` deliberately
    #: does not place. That module trails an option stop every 5-minute bar,
    #: so a resting order would need modifying on every one of them - each
    #: modify a request that can fail, be rejected, or race the fill - and
    #: with a BOUNDED loss that is a fair trade to refuse.
    #:
    #: A naked short's loss is not bounded. So the active management stop
    #: stays in-process, and a single WIDE stop rests at the broker at a level
    #: it should never reach, priced so that a disconnected laptop is a bad
    #: day rather than an account. The two conclusions differ because the
    #: payoffs differ, not because the code drifted.
    rest_disaster_stop: bool = True
    disaster_stop_credit_multiple: float = 3.0
    #: The resting stop is an SL, and its limit is set PAST the trigger by
    #: this fraction - same reasoning as the swing book's
    #: `gtt_limit_buffer_pct`: a stop order that does not fill is not a stop.
    disaster_stop_limit_buffer_pct: float = 0.03


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
    intraday_equity: IntradayEquityConfig = field(
        default_factory=IntradayEquityConfig)
    intraday_equity_broker: IntradayEquityBrokerConfig = field(
        default_factory=IntradayEquityBrokerConfig)
    short_premium: ShortPremiumConfig = field(
        default_factory=ShortPremiumConfig)
    short_premium_broker: ShortPremiumBrokerConfig = field(
        default_factory=ShortPremiumBrokerConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)


DEFAULT = Config()
