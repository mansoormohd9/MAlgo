"""
The scan: a hundred stocks in, at most three tickets out.

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

RUN IT HEADLESS:  python -m nifty_algo.swing.scanner
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from . import fundamentals as fundamentals_mod
from . import halal as halal_mod
from . import news as news_mod
from . import prices as prices_mod
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
    quantity: int
    rupee_risk: float
    rupee_reward: float
    deployed: float
    score: float
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

    def why(self) -> list[str]:
        """The full case for this trade, in the order a person would make it."""
        out = list(self.setup.reasons)
        rs = self.metrics.get("rs_short")
        if rs is not None:
            direction = "outperforming" if rs > 0 else "lagging"
            out.append(
                f"Relative strength: {direction} the Nifty by {rs:+.1%} over "
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
            "direction": self.direction, "setup": self.setup.key,
            "setup_label": self.setup.label,
            "entry": round(self.setup.entry, 2),
            "stop": round(self.setup.stop, 2),
            "target": round(self.setup.target, 2),
            "quantity": self.quantity,
            "rupee_risk": round(self.rupee_risk, 2),
            "rupee_reward": round(self.rupee_reward, 2),
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


def scan(cfg, universe: list[Stock] | None = None,
         force_prices: bool = False, force_fundamentals: bool = False,
         skip_news: bool = False, progress=None,
         today: date | None = None) -> ScanResult:
    """
    Run the whole pipeline.

    `progress` is an optional callable (done, total, label). Nothing in here
    imports Streamlit; the page passes a closure over its own progress bar.
    """
    swing = cfg.swing
    today = today or date.today()
    result = ScanResult(scanned_on=today)

    stocks = universe if universe is not None else load_universe(swing.universe_csv)
    result.universe_size = len(stocks)

    # ---- 1. halal screen (offline, so it runs before anything is fetched) ----
    overrides, warnings = halal_mod.load_overrides(swing.halal.overrides_csv)
    result.warnings.extend(warnings)

    if progress:
        progress(0, 1, "fundamentals for the halal screen")
    funds = fundamentals_mod.load_fundamentals(
        stocks, cfg, force_refresh=force_fundamentals, progress=progress)
    result.fundamentals_age_days = fundamentals_mod.cache_age_days(cfg)

    eligible: list[Stock] = []
    verdicts: dict[str, halal_mod.HalalVerdict] = {}
    for stock in stocks:
        verdict = halal_mod.screen(stock, funds.get(stock.symbol), cfg, overrides)
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
    price_set = prices_mod.load_prices(tickers, cfg, force_refresh=force_prices,
                                       progress=progress)
    result.prices_note = price_set.note()

    for symbol in price_set.missing:
        _reject(result, symbol, STAGE_NO_DATA,
                "yfinance returned no daily bars for this ticker")

    # ---- 3-6. tradeability, setup, reward:risk ----
    candidates: list[tuple[Stock, setup_mod.SwingSetup, dict]] = []
    by_symbol = {s.symbol: s for s in eligible}

    for symbol, df in price_set.bars.items():
        stock = by_symbol.get(symbol)
        if stock is None:
            continue

        metrics, why_not = _metrics(df, price_set.benchmark, cfg)
        if why_not:
            _reject(result, symbol, STAGE_TRADEABILITY, why_not)
            continue

        found, note = setup_mod.detect(symbol, df, cfg)
        if found is None:
            _reject(result, symbol, STAGE_SETUP, note or "no setup")
            continue

        min_rr = cfg.capital.reward_risk_ratio
        if found.reward_risk < min_rr:
            _reject(result, symbol, STAGE_REWARD_RISK,
                    f"{found.label}: target is only {found.reward_risk:.2f}R "
                    f"away, below the {min_rr:.1f}R floor")
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

    # ---- 9-10. rank, sector cap ----
    scored.sort(key=lambda t: t[4], reverse=True)
    per_sector: dict[str, int] = {}
    picks: list[SwingPick] = []

    for stock, found, metrics, n, total, parts in scored:
        if len(picks) >= swing.top_n:
            _reject(result, stock.symbol, STAGE_SECTOR,
                    f"ranked below the top {swing.top_n} (score {total:.3f})")
            continue
        taken = per_sector.get(stock.sector, 0)
        if taken >= swing.max_per_sector:
            _reject(result, stock.symbol, STAGE_SECTOR,
                    f"{stock.sector} already has {taken} pick - three tickets "
                    f"in one sector is one bet, not three")
            continue

        pick = _size(stock, found, metrics, n, total, parts,
                     verdicts.get(stock.symbol), cfg, today)
        if pick is None:
            _reject(result, stock.symbol, STAGE_SIZING,
                    f"stop is {found.risk_points:,.2f} points wide - one share "
                    f"risks more than the ₹{cfg.capital.risk_per_trade_rupees:,.0f} budget")
            continue

        per_sector[stock.sector] = taken + 1
        picks.append(pick)

    result.picks = picks
    result.capital_note = _capital_note(picks, cfg)
    return result


# ---------------- metrics, scoring, sizing ----------------

def _metrics(df, benchmark, cfg) -> tuple[dict, str]:
    """Public metrics plus the tradeability verdict."""
    swing = cfg.swing
    close = float(df["close"].iloc[-1])
    a = float(setup_mod.atr(df, swing.atr_period).iloc[-1])
    atr_pct = a / close if close > 0 else 0.0
    turnover = prices_mod.turnover_crore(df)

    metrics = {
        "last_close": close,
        "atr": a,
        "atr_pct": atr_pct,
        "turnover_cr": turnover,
        "pos_52w": prices_mod.position_in_52w(df),
        "rs_short": prices_mod.relative_strength(df, benchmark, swing.rs_short_days),
        "rs_long": prices_mod.relative_strength(df, benchmark, swing.rs_long_days),
        "volume_ratio": _volume_ratio(df),
        "ema_fast_period": swing.ema_fast,
        "ema_slow_period": swing.ema_slow,
    }

    if close < swing.min_price:
        return metrics, f"trades at ₹{close:,.1f}, below the ₹{swing.min_price:,.0f} floor"
    if turnover < swing.min_avg_turnover_cr:
        return metrics, (f"20-day average turnover ₹{turnover:,.1f} cr is below "
                         f"₹{swing.min_avg_turnover_cr:,.0f} cr - too thin to size into")
    if atr_pct < swing.min_atr_pct:
        return metrics, (f"ATR is {atr_pct:.2%} of price - there is no move here "
                         f"to catch")
    if atr_pct > swing.max_atr_pct:
        return metrics, (f"ATR is {atr_pct:.2%} of price - a {swing.swing_atr_stop_multiple}x "
                         f"ATR stop is too wide to size within the risk budget")
    return metrics, ""


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
          today: date) -> SwingPick | None:
    """
    Turn a setup into a position.

    Quantity comes from the risk budget, not from the capital: you size so
    that the distance to the stop costs exactly one unit of risk. Capital only
    ever reduces the number, never increases it.
    """
    cap = cfg.capital
    risk_points = found.risk_points
    if risk_points <= 0:
        return None

    quantity = math.floor(cap.risk_per_trade_rupees / risk_points)
    note = ""

    if quantity < 1:
        return None

    affordable = math.floor(cap.starting_capital / found.entry)
    if quantity > affordable:
        if affordable < 1:
            return None
        quantity = affordable
        note = (f"Capped at {quantity} shares by capital, not by risk - the "
                f"full risk-based size would need more than "
                f"₹{cap.starting_capital:,.0f}. Risk on this trade is therefore "
                f"below the usual budget.")

    return SwingPick(
        symbol=stock.symbol, name=stock.name, sector=stock.sector,
        direction=setup_mod.LONG, setup=found, quantity=quantity,
        rupee_risk=quantity * risk_points,
        rupee_reward=quantity * found.reward_points,
        deployed=quantity * found.entry,
        score=total, score_parts=parts, metrics=metrics, news=n,
        halal=verdict, last_close=metrics.get("last_close", 0.0),
        scanned_on=today,
        valid_until=today + timedelta(days=cfg.swing.valid_for_days),
        capital_note=note,
    )


def _capital_note(picks: list[SwingPick], cfg) -> str:
    if not picks:
        return ""
    total_deployed = sum(p.deployed for p in picks)
    total_risk = sum(p.rupee_risk for p in picks)
    cap = cfg.capital
    note = (f"All {len(picks)} together: ₹{total_deployed:,.0f} deployed, "
            f"₹{total_risk:,.0f} at risk "
            f"({total_risk / cap.starting_capital:.1%} of capital).")
    if total_deployed > cap.starting_capital:
        note += (f" That is more than your ₹{cap.starting_capital:,.0f} - you "
                 f"cannot take all three at full size without margin. Each "
                 f"ticket is sized correctly on its own; choosing between them "
                 f"is yours.")
    return note


def _reject(result: ScanResult, symbol: str, stage: str, reason: str) -> None:
    result.rejections.append({"symbol": symbol, "stage": stage, "reason": reason})


# ---------------- headless entry point ----------------

def _main() -> None:                                      # pragma: no cover
    import sys

    from ..config import DEFAULT

    # Windows consoles default to cp1252, which cannot encode the rupee sign
    # this report is full of. Losing the whole scan to a print is absurd.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = DEFAULT
    print("Scanning the Nifty 100 ...")

    def progress(done, total, label):
        print(f"  [{done}/{total}] {label}", end="\r")

    result = scan(cfg, progress=progress)
    print(" " * 70, end="\r")

    print(f"\nUniverse {result.universe_size} · halal-eligible "
          f"{result.eligible_size} · {result.prices_note}")
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
        print(f"   Qty {p.quantity}  Risk ₹{p.rupee_risk:,.0f}  "
              f"Reward ₹{p.rupee_reward:,.0f}  R:R {p.reward_risk:.2f}  "
              f"Deployed ₹{p.deployed:,.0f}")
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
