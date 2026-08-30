"""
The scan: a universe in, at most three tickets out.

ONE PIPELINE, THREE MARKETS. Which exchange is being scanned is a `Market`
(see markets.py) threaded through this function - it supplies the universe
file, the benchmark, the price and turnover floors, the halal taxonomy and the
currency. Nothing below branches on a market key; if you find yourself writing
`if market.key == "us"`, the thing you want belongs on the dataclass.

SIZING CROSSES A CURRENCY BOUNDARY AND MUST SAY SO. The risk budget is in
rupees because that is where the governors live. A US stop distance is in
dollars. `_size()` converts explicitly through `fx.py`, and when no trusted
rate exists the whole foreign market stands down rather than sizing off a
guess - see `_resolve_fx`.

EVERY STAGE RECORDS WHAT IT DROPPED AND WHY. That is not bookkeeping, it is
the point. "No picks today" and "eleven setups fired and every one of them had
a target less than 2R away" look identical from the outside and mean opposite
things about the market and about this code. The rejection ledger is what
makes the difference visible, exactly as `evaluations_table` does for the
intraday book.

ORDER OF THE GATES IS DELIBERATE. The cheap, local, offline tests run first
and the expensive network calls run last, so a stock that was never going to
qualify never costs an HTTP request:

    universe -> halal -> prices -> tradeability -> setup -> R:R
             -> earnings -> news (finalists only) -> rank -> sector cap -> size

RUN IT HEADLESS:  python -m nifty_algo.swing.scanner --market india|us|uk
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from . import fundamentals as fundamentals_mod
from . import halal as halal_mod
from . import news as news_mod
from . import prices as prices_mod
from . import fx as fx_mod
from . import market_regime as regime_mod
from . import markets as markets_mod
from . import setup as setup_mod
from .universe import Stock, load_universe

# Stage names used in the rejection ledger. One place, so the page can group
# by them without matching on prose.
# Note there is no "halal" stage here: halal exclusions are their own
# list on the result, because "this company is screened out" is a
# standing fact about the stock rather than something today's tape did.
STAGE_NO_DATA = "no_price_data"
STAGE_TRADEABILITY = "tradeability"
STAGE_SETUP = "no_setup"
STAGE_REWARD_RISK = "reward_risk"
STAGE_EARNINGS = "earnings_blackout"
STAGE_NEWS = "news_veto"
STAGE_SIZING = "position_size"
STAGE_SECTOR = "sector_cap"


@dataclass
class SwingPick:
    """One tradeable candidate, fully specified and fully explained."""
    symbol: str
    name: str
    sector: str
    direction: str
    setup: setup_mod.SwingSetup
    quantity: float                  # float, not int: IBKR fills fractional
                                     # US shares and rounding a $900 stock down
                                     # to one share is a risk error, not a
                                     # tidy-up. Whole-share markets floor it.
    # Amounts in the MARKET's currency. Renamed from rupee_risk/rupee_reward:
    # a field called `rupee_risk` holding dollars is the kind of thing that is
    # read a hundred times and questioned never. The intraday book keeps its
    # own `rupee_risk` - that one really is always rupees.
    risk_amount: float
    reward_amount: float
    deployed: float
    # ...and the same three in rupees, because the risk budget is denominated
    # in rupees even when the trade is not, and the portfolio view has to add
    # a Mumbai ticket to a New York one.
    risk_inr: float = 0.0
    reward_inr: float = 0.0
    deployed_inr: float = 0.0
    market: str = markets_mod.INDIA
    currency: str = "INR"
    currency_symbol: str = "₹"
    fx_inr_per_unit: float = 1.0
    score: float = 0.0
    score_parts: dict[str, dict] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    news: news_mod.NewsResult | None = None
    halal: halal_mod.HalalVerdict | None = None
    last_close: float = 0.0
    scanned_on: date | None = None
    valid_until: date | None = None
    capital_note: str = ""

    @property
    def reward_risk(self) -> float:
        return self.setup.reward_risk

    @property
    def is_fractional(self) -> bool:
        return abs(self.quantity - round(self.quantity)) > 1e-9

    def qty_text(self) -> str:
        return (f"{self.quantity:,.4f}".rstrip("0").rstrip(".")
                if self.is_fractional else f"{self.quantity:,.0f}")

    def money(self, amount: float, dp: int = 0) -> str:
        return f"{self.currency_symbol}{amount:,.{dp}f}"

    def why(self) -> list[str]:
        """The full case for this trade, in the order a person would make it."""
        out = list(self.setup.reasons)
        rs = self.metrics.get("rs_short")
        if rs is not None:
            direction = "outperforming" if rs > 0 else "lagging"
            bench = self.metrics.get("benchmark_label", "the index")
            out.append(
                f"Relative strength: {direction} {bench} by {rs:+.1%} over "
                f"the last month ({self.metrics.get('rs_long', 0):+.1%} over "
                f"three)."
            )
        pos = self.metrics.get("pos_52w")
        if pos is not None:
            out.append(f"Sitting at {pos:.0%} of its 52-week range.")
        if self.news is not None and self.news.available:
            out.append(f"News: {self.news.summary()}.")
        elif self.news is not None:
            out.append(f"News: {self.news.summary()} - it was excluded from "
                       f"the ranking rather than counted as neutral.")
        if self.halal is not None:
            out.append(f"Halal screen: {self.halal.summary}.")
        return out

    def to_record(self) -> dict:
        """A flat, JSON-safe snapshot for the journal."""
        return {
            "symbol": self.symbol, "name": self.name, "sector": self.sector,
            "market": self.market, "currency": self.currency,
            "fx_inr_per_unit": round(self.fx_inr_per_unit, 6),
            "direction": self.direction, "setup": self.setup.key,
            "setup_label": self.setup.label,
            "entry": round(self.setup.entry, 2),
            "stop": round(self.setup.stop, 2),
            "target": round(self.setup.target, 2),
            "quantity": round(self.quantity, 4),
            "risk_amount": round(self.risk_amount, 2),
            "reward_amount": round(self.reward_amount, 2),
            "risk_inr": round(self.risk_inr, 2),
            "reward_inr": round(self.reward_inr, 2),
            "deployed_inr": round(self.deployed_inr, 2),
            # Legacy keys. The journal is append-only and every record written
            # before this change spells them this way, so the tracker has to
            # read both; emitting both keeps one reader working on both eras.
            # For India they are identical to the new fields by construction.
            "rupee_risk": round(self.risk_inr, 2),
            "rupee_reward": round(self.reward_inr, 2),
            "deployed": round(self.deployed, 2),
            "reward_risk": round(self.reward_risk, 2),
            "score": round(self.score, 4),
            "last_close": round(self.last_close, 2),
            "scanned_on": self.scanned_on.isoformat() if self.scanned_on else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "news_score": round(self.news.score, 3) if self.news else None,
            "news_available": self.news.available if self.news else False,
            "halal_source": self.halal.source if self.halal else None,
        }


@dataclass
class ExcludedStock:
    """A halal exclusion, carrying enough to render a row."""
    symbol: str
    name: str
    sector: str
    verdict: halal_mod.HalalVerdict


@dataclass
class ScanResult:
    market: str = markets_mod.INDIA
    market_label: str = ""
    picks: list[SwingPick] = field(default_factory=list)
    excluded_halal: list[ExcludedStock] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    universe_size: int = 0
    eligible_size: int = 0
    scanned_on: date | None = None
    prices_note: str = ""
    fundamentals_age_days: int | None = None
    news_note: str = ""
    capital_note: str = ""
    fx_note: str = ""
    stood_down: str = ""      # non-empty means the scan did not run, and why

    def rejections_for(self, stage: str) -> list[dict]:
        return [r for r in self.rejections if r["stage"] == stage]

    def stage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rejections:
            counts[r["stage"]] = counts.get(r["stage"], 0) + 1
        return counts

    def accounts_for_everything(self) -> bool:
        """Every symbol must end up either picked, excluded, or rejected."""
        return (len(self.picks) + len(self.excluded_halal)
                + len(self.rejections)) == self.universe_size


def scan(cfg, market=None, universe: list[Stock] | None = None,
         force_prices: bool = False, force_fundamentals: bool = False,
         skip_news: bool = False, progress=None,
         today: date | None = None) -> ScanResult:
    """
    Run the whole pipeline for one market.

    `market` is a `markets.Market`, a market key, or None for the configured
    default. `progress` is an optional callable (done, total, label). Nothing
    in here imports Streamlit; the page passes a closure over its own progress
    bar.
    """
    swing = cfg.swing
    today = today or date.today()
    market = _resolve_market(cfg, market)
    result = ScanResult(scanned_on=today, market=market.key,
                        market_label=market.label)

    # The rate is settled BEFORE any work, because a foreign market with no
    # trusted rate produces no tickets at all - and finding that out after
    # three hundred HTTP requests would be an odd way to learn it.
    try:
        rate = fx_mod.rate_for_market(market, cfg)
    except fx_mod.FxUnavailable as e:
        result.stood_down = str(e)
        result.warnings.append(str(e))
        return result
    result.fx_note = rate.note() if not market.is_home else ""

    # An unfunded pot stands the market DOWN rather than falling back to
    # another pool's balance. Every pool except the option account is checked,
    # which now includes India's own swing pot: two books drawing on one
    # balance is the same error as sizing a US trade off the Mumbai account,
    # it is just harder to see because both halves are in rupees.
    if (market.capital_pool != markets_mod.POOL_HOME
            and cfg.capital.capital_inr(market.capital_pool) <= 0):
        result.stood_down = (
            f"{market.label} sizes off the "
            f"{cfg.capital.pool_label(market.capital_pool)}, which is set to "
            f"₹0. Set it on the Settings page (or "
            f"{cfg.capital.pool_field(market.capital_pool)}) - the "
            f"alternative is sizing this trade off a different pot, which "
            f"would claim money that book does not have. Every other market "
            f"is unaffected."
        )
        result.warnings.append(result.stood_down)
        return result

    stocks = universe if universe is not None else load_universe(market.universe_csv)
    result.universe_size = len(stocks)

    # ---- 1. halal screen (offline, so it runs before anything is fetched) ----
    overrides, warnings = halal_mod.load_overrides(swing.halal.overrides_csv)
    result.warnings.extend(warnings)

    if progress:
        progress(0, 1, "fundamentals for the halal screen")
    funds = fundamentals_mod.load_fundamentals(
        stocks, cfg, market, force_refresh=force_fundamentals, progress=progress)
    result.fundamentals_age_days = fundamentals_mod.cache_age_days(cfg)

    eligible: list[Stock] = []
    verdicts: dict[str, halal_mod.HalalVerdict] = {}
    for stock in stocks:
        verdict = halal_mod.screen(stock, funds.get(stock.symbol), cfg, overrides,
                                   market=market)
        verdicts[stock.symbol] = verdict
        if verdict.eligible:
            eligible.append(stock)
        else:
            result.excluded_halal.append(
                ExcludedStock(stock.symbol, stock.name, stock.sector, verdict))
    result.eligible_size = len(eligible)

    if not eligible:
        result.warnings.append(
            "Every symbol was excluded by the halal screen. If that is not "
            "what you expected, check the fundamentals cache age - a screen "
            "that cannot read balance sheets fails closed by design."
        )
        return result

    # ---- 2. prices ----
    tickers = {s.symbol: s.yf_ticker for s in eligible}
    # Yahoo already told us each symbol's quote currency while we were
    # fetching balance sheets, so the pence/pounds question is answered from
    # data rather than assumed from the exchange.
    divisors = {}
    for s in eligible:
        f = funds.get(s.symbol)
        d = f.price_divisor if f is not None else None
        if d is not None:
            divisors[s.symbol] = d
    price_set = prices_mod.load_prices(tickers, cfg, market,
                                       force_refresh=force_prices,
                                       divisors=divisors, progress=progress)
    result.prices_note = price_set.note()

    for symbol in price_set.missing:
        _reject(result, symbol, STAGE_NO_DATA,
                "yfinance returned no daily bars for this ticker")

    # ---- 2b. the market itself, before any per-symbol work ----
    # A long-only book has no answer to a market trending down; this is the
    # filter the option book has always had and this one shipped without. Off
    # unless `regime_ma_days` is set. It reads the benchmark, so it cannot run
    # before prices - but it runs before the setup, R:R, earnings and news
    # work it would otherwise waste.
    regime = regime_mod.benchmark_state(price_set.benchmark,
                                        swing.regime_ma_days)
    if not regime.ok:
        result.stood_down = (
            f"{market.label} stands down: {regime.reason}. This is the "
            f"{regime.ma_days}-day regime filter, not an absence of setups - "
            f"turn it off in settings to scan anyway."
        )
        result.warnings.append(result.stood_down)
        return result

    # ---- 3-6. tradeability, setup, reward:risk ----
    candidates: list[tuple[Stock, setup_mod.SwingSetup, dict]] = []
    by_symbol = {s.symbol: s for s in eligible}

    for symbol, df in price_set.bars.items():
        stock = by_symbol.get(symbol)
        if stock is None:
            continue
        found, metrics, rejection = evaluate_symbol(
            symbol, df, price_set.benchmark, cfg, market)
        if rejection is not None:
            _reject(result, symbol, *rejection)
            continue
        candidates.append((stock, found, metrics))

    # ---- 7. earnings blackout ----
    survivors: list[tuple[Stock, setup_mod.SwingSetup, dict]] = []
    for stock, found, metrics in candidates:
        f = funds.get(stock.symbol)
        days = f.days_to_earnings(today) if f else None
        if days is not None and 0 <= days <= swing.earnings_blackout_days:
            _reject(result, stock.symbol, STAGE_EARNINGS,
                    f"results due in {days} day(s) - a multi-day hold through "
                    f"an earnings gap is a coin flip, not a {cfg.capital.reward_risk_ratio:.0f}:1 trade")
            continue
        metrics["days_to_earnings"] = days
        survivors.append((stock, found, metrics))

    # ---- 8. news, for the finalists only ----
    survivors.sort(key=lambda t: _provisional_rank(t[1], t[2], cfg), reverse=True)
    finalists = survivors[:swing.news_finalists]
    deferred = survivors[swing.news_finalists:]

    if skip_news:
        news_by_symbol = news_mod.unavailable(
            [s.symbol for s, _, _ in survivors], "news fetch skipped")
        result.news_note = "News was skipped for this run."
    else:
        news_by_symbol = news_mod.fetch_for([s for s, _, _ in finalists], cfg,
                                            progress=progress)
        news_by_symbol.update(news_mod.unavailable(
            [s.symbol for s, _, _ in deferred],
            f"outside the top {swing.news_finalists} on price alone - "
            f"news was not fetched"))
        reachable = sum(1 for r in news_by_symbol.values() if r.available)
        result.news_note = (
            f"News read for {reachable} of {len(finalists)} finalists."
            if reachable else
            "News could not be reached for any finalist - it is excluded from "
            "the ranking rather than scored as neutral.")

    scored: list[tuple[Stock, setup_mod.SwingSetup, dict, news_mod.NewsResult, float, dict]] = []
    for stock, found, metrics in survivors:
        n = news_by_symbol.get(stock.symbol) or news_mod.NewsResult(stock.symbol)
        if n.veto:
            _reject(result, stock.symbol, STAGE_NEWS,
                    f"{n.veto} - \"{n.veto_headline}\"")
            continue
        total, parts = _score(found, metrics, n, cfg)
        scored.append((stock, found, metrics, n, total, parts))

    # ---- 9-10. rank, sector cap, size ----
    result.picks = rank_and_size(
        scored, cfg, market, rate, today, verdicts=verdicts,
        reject=lambda sym, stage, why: _reject(result, sym, stage, why))
    result.capital_note = _capital_note(result.picks, cfg, market)
    return result


# ---------------- the decision stages, callable on their own ----------------
#
# Extracted from `scan()` so the BACKTESTER can run the identical gates over
# historical bars instead of reimplementing them. A backtest that reimplements
# the decision logic is measuring a different system, however carefully it is
# copied - which is the same reason invariant #1 forbids `if backtest:` inside
# a strategy. `scan()` above calls exactly these, so there is one implementation
# to be wrong.

def evaluate_symbol(symbol: str, df, benchmark, cfg, market):
    """
    Stages 3-6 for one symbol: tradeability, setup, reward:risk.

    Returns `(found, metrics, rejection)` where `rejection` is a
    `(stage, reason)` pair or None. Reading only the last bar of `df` is
    `setup.detect`'s documented contract, which is what makes calling this on
    a progressively truncated frame a legitimate backtest rather than a
    look-ahead.
    """
    metrics, why_not = _metrics(df, benchmark, cfg, market)
    if why_not:
        return None, metrics, (STAGE_TRADEABILITY, why_not)

    found, note = setup_mod.detect(symbol, df, cfg)
    if found is None:
        return None, metrics, (STAGE_SETUP, note or "no setup")

    min_rr = cfg.capital.reward_risk_ratio
    if found.reward_risk < min_rr:
        return None, metrics, (
            STAGE_REWARD_RISK,
            f"{found.label}: target is only {found.reward_risk:.2f}R away, "
            f"below the {min_rr:.1f}R floor")

    return found, _enrich(metrics, df, benchmark, cfg), None


def rank_and_size(scored, cfg, market, rate, today, verdicts=None,
                  reject=None, top_n: int | None = None) -> list[SwingPick]:
    """
    Stages 9-10: rank on score, apply the sector cap and `top_n`, then size.

    `scored` is `[(stock, found, metrics, news, total, parts)]`.
    `reject(symbol, stage, reason)` is called for everything dropped, so the
    caller's ledger still accounts for every symbol.
    """
    swing = cfg.swing
    limit = swing.top_n if top_n is None else top_n
    verdicts = verdicts or {}
    reject = reject or (lambda *_: None)

    scored = sorted(scored, key=lambda t: t[4], reverse=True)
    per_sector: dict[str, int] = {}
    picks: list[SwingPick] = []

    for stock, found, metrics, n, total, parts in scored:
        if len(picks) >= limit:
            reject(stock.symbol, STAGE_SECTOR,
                   f"ranked below the top {limit} (score {total:.3f})")
            continue
        taken = per_sector.get(stock.sector, 0)
        if taken >= swing.max_per_sector:
            reject(stock.symbol, STAGE_SECTOR,
                   f"{stock.sector} already has {taken} pick - three tickets "
                   f"in one sector is one bet, not three")
            continue

        pick = _size(stock, found, metrics, n, total, parts,
                     verdicts.get(stock.symbol), cfg, today, market, rate)
        if pick is None:
            budget = cfg.capital.risk_inr(market.capital_pool)
            reject(stock.symbol, STAGE_SIZING,
                   f"stop is {found.risk_points:,.2f} points wide - the "
                   f"smallest tradeable size risks more than the "
                   f"₹{budget:,.0f} budget")
            continue

        per_sector[stock.sector] = taken + 1
        picks.append(pick)

    return picks


def _resolve_market(cfg, market):
    """Accept a Market, a key, or None. Never guess on a bad key."""
    if market is None:
        return markets_mod.get(cfg, cfg.swing.default_market)
    if isinstance(market, str):
        return markets_mod.get(cfg, market)
    return market


# ---------------- metrics, scoring, sizing ----------------

def _metrics(df, benchmark, cfg, market) -> tuple[dict, str]:
    """
    Public metrics plus the tradeability verdict.

    The price and turnover floors come from `market` and are denominated in
    that market's currency: $10m of tape and Rs 25 crore of tape are both
    "liquid enough to size into" and neither number means anything applied to
    the other exchange.
    """
    swing = cfg.swing
    close = float(df["close"].iloc[-1])
    a = float(setup_mod.atr(df, swing.atr_period).iloc[-1])
    atr_pct = a / close if close > 0 else 0.0
    turnover = prices_mod.avg_turnover(df, market.turnover_divisor)

    metrics = {
        "last_close": close,
        "atr": a,
        "atr_pct": atr_pct,
        "turnover": turnover,
        "turnover_unit": market.turnover_unit,
        "turnover_cr": turnover,          # legacy alias; India reads the same
        "currency_symbol": market.symbol,
        "benchmark_label": market.benchmark_label,
        "ema_fast_period": swing.ema_fast,
        "ema_slow_period": swing.ema_slow,
    }

    if close < market.min_price:
        return metrics, (f"trades at {market.money(close, 1)}, below the "
                         f"{market.money(market.min_price)} floor")
    if turnover < market.min_avg_turnover:
        return metrics, (f"20-day average turnover {market.turnover(turnover)} "
                         f"is below {market.turnover(market.min_avg_turnover)} "
                         f"- too thin to size into")
    if atr_pct < swing.min_atr_pct:
        return metrics, (f"ATR is {atr_pct:.2%} of price - there is no move here "
                         f"to catch")
    if atr_pct > swing.max_atr_pct:
        return metrics, (f"ATR is {atr_pct:.2%} of price - a {swing.swing_atr_stop_multiple}x "
                         f"ATR stop is too wide to size within the risk budget")
    return metrics, ""


def _enrich(metrics: dict, df, benchmark, cfg) -> dict:
    """
    The metrics that SCORE but never gate, added once a setup exists.

    Relative strength, position in the 52-week range and the volume ratio
    decide how a candidate ranks; none of them can reject one. Computing them
    for every symbol on every session meant the backtest paid for the ranking
    of the ~95 names in 100 that had no setup to rank - about half the cost of
    a scan. Nothing reads these on a rejected symbol: `_reject` records the
    stage and the reason, and `_score` only ever sees survivors.
    """
    swing = cfg.swing
    metrics["pos_52w"] = prices_mod.position_in_52w(df)
    metrics["rs_short"] = prices_mod.relative_strength(df, benchmark,
                                                       swing.rs_short_days)
    metrics["rs_long"] = prices_mod.relative_strength(df, benchmark,
                                                      swing.rs_long_days)
    metrics["volume_ratio"] = _volume_ratio(df)
    return metrics


def _volume_ratio(df) -> float:
    if len(df) < 21 or float(df["volume"].iloc[-20:].mean()) <= 0:
        return 1.0
    return float(df["volume"].iloc[-1] / df["volume"].iloc[-20:].mean())


def _provisional_rank(found, metrics, cfg) -> float:
    """
    A price-only score, used solely to decide which names are worth an HTTP
    request for news. Deliberately the same shape as the real score minus the
    news term, so the finalist cut is not arbitrary.
    """
    total, _ = _score(found, metrics, news_mod.NewsResult("", available=False), cfg)
    return total


def _score(found, metrics, n, cfg) -> tuple[float, dict]:
    """
    The composite, with every component's raw value and contribution kept.

    When news is unavailable its weight is REDISTRIBUTED over the other
    components rather than being scored as neutral. Scoring an unreachable
    feed as 0.5 would quietly award half the news weight to every stock at
    once, which is a ranking artefact masquerading as information.
    """
    swing = cfg.swing
    min_rr = cfg.capital.reward_risk_ratio

    rs_short = metrics.get("rs_short") or 0.0
    rs_long = metrics.get("rs_long") or 0.0
    blended_rs = 0.6 * rs_short + 0.4 * rs_long

    raw = {
        "setup": found.quality,
        "relative_strength": _clip(0.5 + blended_rs / 0.20 * 0.5),
        "reward_risk": _clip((found.reward_risk - min_rr) / max(min_rr, 0.1)),
        "volume": _clip((metrics.get("volume_ratio", 1.0) - 0.5) / 2.0),
        "position_52w": _clip(metrics.get("pos_52w", 0.5)),
        "news": n.normalised if n.available else None,
    }

    weights = dict(swing.score_weights())
    if raw["news"] is None:
        dropped = weights.pop("news")
        live = sum(weights.values())
        if live > 0:
            weights = {k: v * (1.0 + dropped / live) for k, v in weights.items()}
        raw.pop("news")

    parts: dict[str, dict] = {}
    total = 0.0
    for name, value in raw.items():
        weight = weights.get(name, 0.0)
        contribution = value * weight
        total += contribution
        parts[name] = {"raw": round(value, 4), "weight": round(weight, 4),
                       "contribution": round(contribution, 4)}
    if raw.get("news") is None and "news" not in parts:
        parts["news"] = {"raw": None, "weight": 0.0, "contribution": 0.0,
                         "note": "unavailable - weight redistributed"}

    return round(total, 4), parts


def _clip(x: float) -> float:
    if x != x:                                   # NaN
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _size(stock, found, metrics, n, total, parts, verdict, cfg,
          today: date, market, rate) -> SwingPick | None:
    """
    Turn a setup into a position.

    Quantity comes from the risk budget, not from the capital: you size so
    that the distance to the stop costs exactly one unit of risk. Capital only
    ever reduces the number, never increases it.

    THE CURRENCY CONVERSION IS THE ONE THING TO GET RIGHT HERE. `risk_points`
    is in the exchange's currency; the budget is in rupees. Dividing one by
    the other without `rate` is not a rounding error - on a US name it returns
    a size about 88x too large, and the ticket it prints looks entirely normal.
    """
    cap = cfg.capital
    risk_points = found.risk_points
    if risk_points <= 0:
        return None

    inr_per_unit = rate.inr_per_unit
    if inr_per_unit <= 0:
        return None

    # Budget, moved from rupees into the currency the stop is measured in.
    budget_local = cap.risk_inr(market.capital_pool) / inr_per_unit
    capital_local = cap.capital_inr(market.capital_pool) / inr_per_unit

    raw = budget_local / risk_points
    quantity = _round_quantity(raw, market)
    note = ""
    if quantity <= 0:
        return None

    affordable = _round_quantity(capital_local / found.entry, market)
    if quantity > affordable:
        if affordable <= 0:
            return None
        quantity = affordable
        note = (f"Capped at {affordable:,.4f} shares by capital, not by risk - "
                f"the full risk-based size would need more than "
                f"{market.money(capital_local)}. Risk on this trade is "
                f"therefore below the usual budget.")

    risk_amount = quantity * risk_points
    reward_amount = quantity * found.reward_points
    deployed = quantity * found.entry

    return SwingPick(
        symbol=stock.symbol, name=stock.name, sector=stock.sector,
        direction=setup_mod.LONG, setup=found, quantity=quantity,
        risk_amount=risk_amount,
        reward_amount=reward_amount,
        deployed=deployed,
        risk_inr=risk_amount * inr_per_unit,
        reward_inr=reward_amount * inr_per_unit,
        deployed_inr=deployed * inr_per_unit,
        market=market.key, currency=market.currency,
        currency_symbol=market.symbol, fx_inr_per_unit=inr_per_unit,
        score=total, score_parts=parts, metrics=metrics, news=n,
        halal=verdict, last_close=metrics.get("last_close", 0.0),
        scanned_on=today,
        valid_until=today + timedelta(days=cfg.swing.valid_for_days),
        capital_note=note,
    )


def _round_quantity(raw: float, market) -> float:
    """
    Whole shares, unless the market fills fractions.

    Rounding a $900 stock down to one share does not make the position safe -
    it silently changes the risk on the ticket, usually downward, sometimes to
    zero, and the reason never appears anywhere. Where the broker supports
    fractions the honest answer is the fraction.
    """
    if raw <= 0:
        return 0.0
    if not market.allow_fractional:
        return float(math.floor(raw))
    # Four decimals is finer than any broker's minimum increment and keeps the
    # arithmetic from carrying float noise into the ticket.
    return float(math.floor(raw * 10_000) / 10_000)


def _capital_note(picks: list[SwingPick], cfg, market) -> str:
    if not picks:
        return ""
    total_deployed = sum(p.deployed for p in picks)
    total_risk = sum(p.risk_amount for p in picks)
    capital_inr = cfg.capital.capital_inr(market.capital_pool)
    rate = picks[0].fx_inr_per_unit or 1.0
    capital_local = capital_inr / rate

    note = (f"All {len(picks)} together: {market.money(total_deployed)} "
            f"deployed, {market.money(total_risk)} at risk "
            f"({(total_risk / capital_local if capital_local else 0):.1%} of "
            f"this market's capital).")
    if not market.is_home:
        note += (f" In rupees: ₹{sum(p.deployed_inr for p in picks):,.0f} "
                 f"deployed, ₹{sum(p.risk_inr for p in picks):,.0f} at risk.")
    if total_deployed > capital_local:
        note += (f" That is more than the {market.money(capital_local)} in "
                 f"this pool - you cannot take all of them at full size "
                 f"without margin. Each ticket is sized correctly on its own; "
                 f"choosing between them is yours.")
    return note


def _reject(result: ScanResult, symbol: str, stage: str, reason: str) -> None:
    result.rejections.append({"symbol": symbol, "stage": stage, "reason": reason})


# ---------------- headless entry point ----------------

def _main() -> None:                                      # pragma: no cover
    import argparse
    import sys

    from ..config import DEFAULT

    # Windows consoles default to cp1252, which cannot encode the rupee sign
    # this report is full of. Losing the whole scan to a print is absurd.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT

    parser = argparse.ArgumentParser(description="The daily swing scan.")
    parser.add_argument("--market", default=cfg.swing.default_market,
                        choices=markets_mod.keys(cfg),
                        help="which exchange to scan")
    parser.add_argument("--skip-news", action="store_true",
                        help="do not fetch news for the finalists")
    args = parser.parse_args()

    market = markets_mod.get(cfg, args.market)
    print(f"Scanning {market.label} ...")

    def progress(done, total, label):
        print(f"  [{done}/{total}] {label}", end="\r")

    result = scan(cfg, market=market, skip_news=args.skip_news,
                  progress=progress)
    print(" " * 70, end="\r")

    if result.stood_down:
        # Not "no picks today". The scan did not run, and saying which is the
        # entire reason this branch exists.
        print(f"\n{market.label} STOOD DOWN — no scan was run.")
        print(f"  {result.stood_down}")
        return

    print(f"\nUniverse {result.universe_size} · halal-eligible "
          f"{result.eligible_size} · {result.prices_note}")
    if result.fx_note:
        print(f"FX: {result.fx_note}")
    if result.news_note:
        print(result.news_note)
    for w in result.warnings:
        print(f"  ! {w}")

    print(f"\nExcluded by the halal screen: {len(result.excluded_halal)}")
    for x in result.excluded_halal[:10]:
        print(f"  {x.symbol:<12} {x.verdict.reason[:70]}")
    if len(result.excluded_halal) > 10:
        print(f"  ... and {len(result.excluded_halal) - 10} more")

    print("\nDropped, by stage:")
    for stage, count in sorted(result.stage_counts().items(),
                               key=lambda kv: -kv[1]):
        print(f"  {stage:<18} {count}")

    print(f"\nTop {len(result.picks)}:")
    for i, p in enumerate(result.picks, 1):
        s = p.setup
        print(f"\n{i}. {p.symbol} — {p.name} ({p.sector})")
        print(f"   {s.label} · score {p.score:.3f}")
        print(f"   Entry {s.entry:,.2f}  Stop {s.stop:,.2f} "
              f"({s.stop_pct:.1%})  Target {s.target:,.2f} ({s.target_pct:.1%})")
        print(f"   Qty {p.qty_text()}  Risk {p.money(p.risk_amount)}  "
              f"Reward {p.money(p.reward_amount)}  R:R {p.reward_risk:.2f}  "
              f"Deployed {p.money(p.deployed)}")
        if p.currency != "INR":
            print(f"   In rupees: risk ₹{p.risk_inr:,.0f}  "
                  f"deployed ₹{p.deployed_inr:,.0f}")
        if p.halal is not None and p.halal.disagreement:
            print("   ! Shariah standards disagree on this name:")
            for mv in p.halal.verdicts.values():
                print(f"       {mv.summary}")
        print(f"   {s.trigger_note}")
        for line in p.why():
            print(f"   - {line}")
        if p.capital_note:
            print(f"   ! {p.capital_note}")

    if result.capital_note:
        print(f"\n{result.capital_note}")
    print(f"\n{halal_mod.DISCLAIMER}")
    print("\nEvery symbol accounted for: "
          f"{result.accounts_for_everything()}")


if __name__ == "__main__":                                # pragma: no cover
    _main()
