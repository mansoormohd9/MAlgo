"""
Central configuration. Every tunable number lives here - never hardcode
a parameter inside strategy logic, or you cannot sweep it in a backtest.
"""
from dataclasses import dataclass, field
from datetime import time


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
        return self.reward_per_trade_pct / self.risk_per_trade_pct


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
    debt_to_assets_max: float = 0.33
    cash_and_interest_to_assets_max: float = 0.33
    receivables_to_assets_max: float = 0.49

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
    universe_csv: str = "data/nifty100.csv"
    cache_dir: str = "data/cache"
    benchmark_ticker: str = "^NSEI"
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
    min_price: float = 50.0
    min_avg_turnover_cr: float = 25.0   # 20-day average traded value, rupees crore
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
class Config:
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    trade: TradeManagementConfig = field(default_factory=TradeManagementConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)


DEFAULT = Config()
