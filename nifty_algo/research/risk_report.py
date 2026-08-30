"""
The portfolio risk framework: what this book loses, and to what.

WHY THIS IS THE STRONGEST OF THE TEN BRIEFINGS. Almost all of it is
computable. Correlation, concentration, currency exposure, liquidity,
position sizing against the governors and a REPLAYED drawdown are arithmetic
over data this repo already caches. Only the last section - which hedge, which
rebalance - is judgment. That ratio is the opposite of the macro pack's, and
it is why this one carries the numbers a decision can rest on.

THREE THINGS IT REFUSES TO DO, each of which is the ordinary way this report
gets written and each of which produces a confident wrong answer:

  1. IT DOES NOT MODEL A DRAWDOWN. There is no normal distribution here, no
     parametric VaR, no simulated shock. The stress test REPLAYS the actual
     benchmark drawdown episodes in the cached history and applies this book's
     current weights to what those names actually did. A modelled 99% VaR on
     equity returns understates the tail by construction - the distribution is
     not normal, and the days that matter are precisely the ones a normal
     distribution says cannot happen.

  2. IT DOES NOT REPLAY A PORTFOLIO YOU DID NOT HOLD. The weights applied to
     history are TODAY'S weights, because this build has no position history
     before the journal. That answers "how would this book have behaved",
     never "how did it behave", and every stress figure says so.

  3. IT DOES NOT QUOTE A PERCENTAGE OF A BOOK IT COULD NOT READ.
     `PortfolioSnapshot.weight()` returns None on an incomplete snapshot, and
     concentration is a percentage or it is nothing.

THE LOOK-THROUGH IS WHAT MAKES CONCENTRATION REAL. `swing/holdings.py` knows
what SPUS and HLAL hold. A book with a direct NVDA position AND both funds is
far more concentrated in NVDA than a position table shows, and it is the only
place that fact can appear before the order is placed. Same module, same
reasoning as the pick-card badge - a badge there, a risk line here.

SURVIVORSHIP APPLIES HERE TOO. The universe file is today's index, so the
cached bars belong to companies that were still in it. Replayed drawdowns are
therefore optimistic in the same direction the swing backtest's are, and the
caveat is printed above the result rather than under it.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config, DEFAULT
from ..portfolio import aggregate as portfolio_mod
from ..swing import holdings as etf_holdings
from ..swing import markets as markets_mod
from ..swing import prices as prices_mod
from . import exposure, holdings_prices
from .base import (SOURCE_CACHE, SOURCE_COMMITTED, SOURCE_CONFIG,
                   SOURCE_PORTFOLIO, Fact, FactPack)
from .providers import macro_series as ms

REPORT = "risk"
TITLE = "Portfolio risk framework"

#: A drawdown episode has to be big enough to be an event rather than a wobble.
MIN_EPISODE_DRAWDOWN = 0.10
#: ...and this many of them, worst first, are replayed.
MAX_EPISODES = 4


def build(cfg: Config = DEFAULT, market_key: str = markets_mod.INDIA,
          snapshot=None, force_refresh: bool = False,
          progress=None) -> FactPack:
    market = markets_mod.get(cfg, market_key)
    if snapshot is None:
        _say(progress, "reading holdings")
        snapshot = portfolio_mod.load(cfg)

    pack = FactPack(report=REPORT, title=TITLE,
                    inputs={"market": market.key,
                            "market_label": market.label,
                            "correlation_days": cfg.portfolio.correlation_days})
    pack.caveats.extend(snapshot.caveats())

    held = [p for p in snapshot.positions if p.market == market.key]
    if not snapshot.positions:
        pack.stood_down = (
            "No positions were found in any enabled connector, so there is no "
            "portfolio to assess. This is not a low-risk result - it is an "
            "absent one. Check the connector notes above before reading it as "
            "either.")
        _connectors(pack, snapshot)
        return pack

    _connectors(pack, snapshot)

    if not held:
        # Positions exist, but none in the market that was asked for - which
        # is what `--market us` on a purely Indian book looks like. Every
        # market-scoped section below would then be computed over an empty
        # list, and the concentration one would raise on `rows[0]`. Saying so
        # is both the correct answer and the one that does not crash.
        _no_positions_here(pack, snapshot, market)
        _lookthrough(pack, snapshot)
        return pack

    _say(progress, "pricing your holdings")
    bars, benchmark, price_note = holdings_prices.bars_for(snapshot, cfg,
                                                           market.key,
                                                           force_refresh)
    stock_returns = holdings_prices.returns(bars, cfg.portfolio.correlation_days)
    weights = {p.symbol: snapshot.value_inr.get(p.key, 0.0) for p in held}

    _concentration(pack, snapshot, held, market)
    _lookthrough(pack, snapshot)
    _correlation(pack, cfg, stock_returns, price_note)
    _rate_sensitivity(pack, cfg, snapshot, held, market, stock_returns)
    _stress(pack, cfg, bars, benchmark, weights, market)
    _tails(pack, cfg, bars, weights)
    _liquidity(pack, snapshot, held, bars, market)
    _sizing(pack, cfg, snapshot, held, market)
    _actions(pack)

    pack.caveats.append(
        "Weights applied to history are TODAY'S weights. Every replayed "
        "figure answers how this book would have behaved, never how it did.")
    pack.caveats.append(
        "The cached universe is today's index, so names that left it are "
        "absent from every replay below - the same survivorship distortion "
        "the swing backtest prints above its results.")
    return pack


def _say(progress, message: str) -> None:
    if progress:
        progress(message)


def _no_positions_here(pack, snapshot, market) -> None:
    """Held elsewhere, not here - and the difference matters."""
    s = pack.section("Concentration")
    elsewhere = sorted({p.market for p in snapshot.positions})
    s.add(Fact.unknown(
        "Positions in this market",
        f"nothing is held in {market.label}. The snapshot did find positions "
        f"in: {', '.join(elsewhere)}. Re-run with --market set to one of "
        f"those. This is an empty MARKET, not an empty account - the two "
        f"produce very different risk reports and only one of them is a "
        f"low-risk result.",
        source=SOURCE_PORTFOLIO))


def _connectors(pack, snapshot) -> None:
    s = pack.section("Where these holdings came from")
    s.rows = [{"source": r.source, "available": r.available,
               "positions": len(r.positions), "note": r.note}
              for r in snapshot.results]
    s.add(Fact.known("Snapshot complete", snapshot.complete, SOURCE_PORTFOLIO,
                     note=("Every enabled connector answered and every line "
                           "converted." if snapshot.complete else
                           "At least one source could not be read or one "
                           "currency could not be converted, so portfolio "
                           "PERCENTAGES are withheld throughout this report. "
                           "Absolute figures are still facts about what was "
                           "seen.")))
    s.add(Fact.known("Value seen", round(snapshot.total_inr, 2),
                     SOURCE_PORTFOLIO, unit="INR",
                     note=snapshot.note()))


# ---------------- concentration ----------------

def _concentration(pack, snapshot, held, market) -> None:
    s = pack.section("Concentration")
    sectors = exposure.sector_map(market)

    rows = []
    for p in sorted(held, key=lambda x: -snapshot.value_inr.get(x.key, 0.0)):
        weight = snapshot.weight(p.key)
        rows.append({
            "symbol": p.symbol,
            "sector": sectors.get(p.symbol, "unclassified"),
            "value_inr": round(snapshot.value_inr.get(p.key, 0.0), 2),
            "weight_pct": None if weight is None else round(weight * 100.0, 2),
            "unrealised_pct": (None if p.pnl_pct is None
                               else round(p.pnl_pct * 100.0, 2)),
        })
    s.rows = rows

    if not snapshot.complete:
        s.add(Fact.unknown(
            "Largest single position",
            "the snapshot is incomplete, so every share here would be a share "
            "of a partial book", source=SOURCE_PORTFOLIO))
        s.note = ("Absolute values only. A concentration figure computed "
                  "against a denominator we could not establish is worse than "
                  "no figure, because it would be acted on.")
        return

    top = rows[0]
    s.add(Fact.known("Largest single position", top["symbol"],
                     SOURCE_PORTFOLIO,
                     note=f"{top['weight_pct']}% of the book."))

    by_sector: dict[str, float] = {}
    for r in rows:
        by_sector[r["sector"]] = by_sector.get(r["sector"], 0.0) + r["value_inr"]
    biggest = max(by_sector.items(), key=lambda kv: kv[1])
    s.add(Fact.known(
        "Largest sector", biggest[0], SOURCE_PORTFOLIO,
        note=f"{biggest[1] / snapshot.total_inr * 100.0:,.1f}% of the book. "
             f"`SwingConfig.max_per_sector` caps the SCANNER at "
             f"one pick per sector on the view that three names in one sector "
             f"is one bet; nothing caps what the account accumulates, which is "
             f"what this measures."))

    # Herfindahl: the honest one-number answer to "how many bets is this".
    shares = [r["weight_pct"] / 100.0 for r in rows if r["weight_pct"]]
    if shares:
        hhi = sum(x * x for x in shares)
        s.add(Fact.known(
            "Effective number of positions", round(1.0 / hhi, 2),
            "derived (inverse Herfindahl of the weights)",
            note=f"{len(rows)} positions held. A book of {len(rows)} equal "
                 f"weights would score {len(rows)}; the gap is how much of "
                 f"this book is really one or two names."))


def _lookthrough(pack, snapshot) -> None:
    """
    What the Shariah ETFs already own of what you hold directly.

    The largest concentration in a book like this is usually invisible in the
    position table: a direct NVDA position alongside SPUS and HLAL is adding
    to a bet, not diversifying into one.
    """
    s = pack.section("Fund look-through")
    book = etf_holdings.load_holdings()
    if not book:
        s.add(Fact.unknown(
            "ETF look-through",
            "data/etf_holdings.csv is empty or missing, so a direct position "
            "that duplicates a fund holding cannot be detected. Refresh it "
            "from the Daily picks page.", source=SOURCE_COMMITTED))
        return

    # Every market, not just the one being reported: an Indian book's
    # look-through concentration comes from US-listed funds by definition.
    baskets = [p for p in snapshot.positions if p.asset_class in ("etf", "mf")]
    funds = {p.symbol.upper() for p in baskets}
    balances = {p.symbol.upper(): snapshot.value_inr.get(p.key, 0.0)
                for p in baskets}

    rows = []
    for p in snapshot.positions:
        if p.asset_class in ("etf", "mf", "cash"):
            continue
        overlap = etf_holdings.overlap_for(p.symbol, book, balances)
        if not overlap.any:
            continue
        rows.append({
            "symbol": p.symbol,
            "direct_value_inr": round(snapshot.value_inr.get(p.key, 0.0), 2),
            "via_funds_inr": round(overlap.value_inr, 2),
            "funds": ", ".join(sorted(h.etf for h in overlap.rows)),
            "max_fund_weight_pct": round(
                max(h.weight for h in overlap.rows) * 100.0, 2),
            "note": overlap.note(),
        })
    s.rows = rows
    s.add(Fact.known(
        "Direct positions also held inside your funds", len(rows),
        SOURCE_COMMITTED,
        note=(f"Funds recognised in the book: "
              f"{', '.join(sorted(funds)) or 'none'}. Fund weights are as of "
              f"{etf_holdings.as_of(book) or 'unknown'}.")))
    s.note = (
        "This is a concentration fact, never a rejection - what you hold is "
        "your decision. What it must not be is a decision made silently.")


# ---------------- correlation ----------------

def _correlation(pack, cfg, stock_returns, price_note) -> None:
    s = pack.section("Correlation between holdings")
    rows = exposure.pairwise(stock_returns, cfg)
    s.rows = rows

    average, pairs = exposure.average_pairwise(rows)
    if average is None:
        s.add(Fact.unknown(
            "Average pairwise correlation",
            f"no pair had the {cfg.portfolio.min_correlation_sessions} "
            f"overlapping sessions this build will report a correlation on",
            source=SOURCE_CACHE))
    else:
        s.add(Fact.known(
            "Average pairwise correlation", average, SOURCE_CACHE,
            note=f"Across {pairs} pair(s) over "
                 f"{cfg.portfolio.correlation_days} sessions. THE "
                 f"diversification number: a book of six names at 0.75 is one "
                 f"position with extra brokerage, and the count of names says "
                 f"nothing about it."))
        worst = rows[0]
        if worst["correlation"] is not None:
            s.add(Fact.known(
                "Most correlated pair", f"{worst['a']} / {worst['b']}",
                SOURCE_CACHE,
                note=f"{worst['correlation']} over {worst['sessions']} "
                     f"sessions. These two are close to one position."))
    s.note = (
        f"{price_note} Correlations are of DAILY returns over one regime. "
        f"They rise in a sell-off - the diversification a book has is at its "
        f"lowest exactly when it is needed, which is why the replayed "
        f"drawdown below matters more than this table does.")


def _rate_sensitivity(pack, cfg, snapshot, held, market, stock_returns) -> None:
    s = pack.section("Interest rate and factor sensitivity")
    history = ms.history(cfg)
    moves = ms.factor_moves(history) if history is not None else pd.DataFrame()
    if moves.empty:
        s.add(Fact.unknown("Factor sensitivity",
                           "no macro history could be loaded to correlate "
                           "against", source=SOURCE_CACHE))
        return

    rows = []
    for p in sorted(held, key=lambda x: -snapshot.value_inr.get(x.key, 0.0)):
        weight = snapshot.weight(p.key)
        row = {"symbol": p.symbol,
               "weight_pct": None if weight is None else round(weight * 100.0, 2)}
        row.update(exposure.sensitivities(p.symbol, stock_returns, moves, cfg))
        rows.append(row)
    s.rows = rows

    weighted = [(r["weight_pct"], r["nifty_beta"]) for r in rows
                if r["weight_pct"] is not None and r["nifty_beta"] is not None]
    if weighted:
        total = sum(w for w, _ in weighted)
        if total > 0:
            s.add(Fact.known(
                "Book beta to the benchmark",
                round(sum(w * b for w, b in weighted) / total, 3),
                "derived from cached daily bars", unit="beta",
                note=f"Value-weighted across the {len(weighted)} position(s) "
                     f"with enough history, against the "
                     f"{market.benchmark_label}. Covers "
                     f"{total:,.1f}% of the book."))
    else:
        s.add(Fact.unknown(
            "Book beta to the benchmark",
            "no position had both a weight and enough overlapping sessions",
            source=SOURCE_CACHE))
    s.note = (
        "Rate sensitivity here is MEASURED, not assigned by sector rule. A "
        "correlation to the 10-year of -0.3 says these moved opposite each "
        "other about a third of the time over this window; it is not a "
        "duration and it does not carry a magnitude.")


# ---------------- the replayed drawdown ----------------

def episodes(benchmark: pd.DataFrame,
             min_drawdown: float = MIN_EPISODE_DRAWDOWN,
             limit: int = MAX_EPISODES) -> list[dict]:
    """
    The benchmark's real peak-to-trough episodes, deepest first.

    Found rather than hardcoded. A list of dates in the source ("Mar 2020,
    Oct 2021") is a list that is wrong the moment the cache covers a different
    span, and it would quietly replay nothing when the history is short.
    """
    if benchmark is None or benchmark.empty or "close" not in benchmark:
        return []
    close = pd.to_numeric(benchmark["close"], errors="coerce").dropna()
    if len(close) < 60:
        return []

    peak = close.cummax()
    underwater = close < peak

    out: list[dict] = []
    start = None
    for date, wet in underwater.items():
        if wet and start is None:
            start = date
        elif not wet and start is not None:
            out.append(_episode(close, start, date))
            start = None
    if start is not None:
        out.append(_episode(close, start, close.index[-1]))

    deep = [e for e in out if e and e["benchmark_drawdown_pct"] <= -min_drawdown * 100]
    return sorted(deep, key=lambda e: e["benchmark_drawdown_pct"])[:limit]


def _episode(close: pd.Series, start, end) -> dict | None:
    window = close.loc[:end]
    window = window.loc[window.index >= start]
    if window.empty:
        return None
    # The peak is the last bar BEFORE the drawdown began, which is where the
    # money actually was. Measuring from the first underwater bar understates
    # every episode by one day of the fall.
    prior = close.loc[:start]
    peak = float(prior.iloc[-2]) if len(prior) >= 2 else float(window.iloc[0])
    trough_date = window.idxmin()
    trough = float(window.min())
    return {
        "from": str(start)[:10],
        "trough": str(trough_date)[:10],
        "to": str(end)[:10],
        "sessions": int(len(window)),
        "benchmark_drawdown_pct": round((trough / peak - 1.0) * 100.0, 2),
    }


def _stress(pack, cfg, bars, benchmark, weights, market) -> None:
    s = pack.section("Recession stress test (replayed, not modelled)")
    found = episodes(benchmark)
    if not found:
        s.add(Fact.unknown(
            "Drawdown episodes replayed",
            f"the cached {market.benchmark_label} history contains no "
            f"peak-to-trough fall of {MIN_EPISODE_DRAWDOWN:.0%} or more. Run "
            f"`python scripts/fetch_swing_history.py --market {market.key} "
            f"--years 6` for a history long enough to contain one.",
            source=SOURCE_CACHE))
        return

    total_weight = sum(weights.values())
    rows = []
    for episode in found:
        moved, covered = 0.0, 0.0
        for symbol, weight in weights.items():
            change = _window_return(bars.get(symbol), episode["from"],
                                    episode["trough"])
            if change is None:
                continue
            moved += weight * change
            covered += weight
        rows.append({
            **episode,
            "book_drawdown_pct": (round(moved / covered * 100.0, 2)
                                  if covered > 0 else None),
            "coverage_pct": (round(covered / total_weight * 100.0, 1)
                             if total_weight > 0 else 0.0),
        })
    s.rows = rows

    scored = [r for r in rows if r["book_drawdown_pct"] is not None]
    if scored:
        worst = min(scored, key=lambda r: r["book_drawdown_pct"])
        s.add(Fact.known(
            "Worst replayed drawdown for this book",
            worst["book_drawdown_pct"], SOURCE_CACHE, unit="%",
            as_of=worst["trough"],
            note=(f"{worst['from']} to {worst['trough']}, when the "
                  f"{market.benchmark_label} fell "
                  f"{worst['benchmark_drawdown_pct']}%. Covers "
                  f"{worst['coverage_pct']}% of the book by value - positions "
                  f"without bars over that window are excluded and the rest "
                  f"renormalised.")))
    else:
        s.add(Fact.unknown(
            "Worst replayed drawdown for this book",
            "no position had cached bars covering any of the episodes found",
            source=SOURCE_CACHE))

    s.note = (
        "These are the benchmark's ACTUAL falls, with this book's current "
        "weights applied to what those names actually did. No distribution is "
        "assumed and no shock is simulated - a parametric VaR on equity "
        "returns understates the tail by construction, because the days that "
        "matter are the ones a normal distribution says cannot happen.")


def _window_return(df, start: str, end: str) -> float | None:
    """One symbol's return across an episode, or None if it has no bars there."""
    if df is None or df.empty or "close" not in df:
        return None
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    window = close.loc[(close.index >= start) & (close.index <= end)]
    if len(window) < 2:
        return None
    first = float(window.iloc[0])
    return None if first == 0 else float(window.iloc[-1]) / first - 1.0


# ---------------- tails, liquidity, sizing ----------------

def _tails(pack, cfg, bars, weights) -> None:
    s = pack.section("Tail risk (historical, not modelled)")
    daily = holdings_prices.returns(bars, 0)          # 0 = every cached session
    series = exposure.portfolio_returns(daily, weights)
    if series.empty or len(series) < cfg.portfolio.min_correlation_sessions:
        s.add(Fact.unknown(
            "Historical daily moves",
            f"fewer than {cfg.portfolio.min_correlation_sessions} sessions of "
            f"book history could be assembled", source=SOURCE_CACHE))
        return

    sessions = int(len(series))
    s.add(Fact.known("Sessions of book history", sessions, SOURCE_CACHE,
                     note="At today's weights, over every cached session."))
    for label, q in (("1-in-20 day (5th percentile)", 0.05),
                     ("1-in-100 day (1st percentile)", 0.01)):
        s.add(Fact.known(label, round(float(series.quantile(q)) * 100.0, 2),
                         SOURCE_CACHE, unit="%",
                         note=f"An empirical percentile of {sessions} actual "
                              f"sessions, not a modelled quantile. Its own "
                              f"tail is only as deep as this history is long."))
    s.add(Fact.known("Worst single session",
                     round(float(series.min()) * 100.0, 2),
                     SOURCE_CACHE, unit="%", as_of=str(series.idxmin())[:10]))
    s.add(Fact.known("Annualised volatility",
                     round(float(series.std()) * (252 ** 0.5) * 100.0, 2),
                     SOURCE_CACHE, unit="%",
                     note="Standard deviation scaled by sqrt(252). Quoted "
                          "because it is expected, not because equity returns "
                          "are normal - they are not, which is why the "
                          "percentiles above are empirical."))
    s.rows = [{"session": str(d)[:10], "return_pct": round(float(v) * 100, 2)}
              for d, v in series.nsmallest(10).items()]
    s.judgment = [
        "Probability estimates for tail scenarios. This section counts what "
        "HAS happened over the cached window; attaching a forward probability "
        "to a scenario is a judgment and must be labelled as one.",
    ]


def _liquidity(pack, snapshot, held, bars, market) -> None:
    s = pack.section("Liquidity")
    rows = []
    for p in sorted(held, key=lambda x: -snapshot.value_inr.get(x.key, 0.0)):
        df = bars.get(p.symbol)
        turnover = (prices_mod.avg_turnover(df, divisor=1.0)
                    if df is not None and not df.empty else None)
        value = snapshot.value_inr.get(p.key, 0.0)
        rows.append({
            "symbol": p.symbol,
            "value_inr": round(value, 2),
            "avg_daily_traded_value": (None if not turnover
                                       else round(float(turnover), 0)),
            "turnover_display": (
                market.turnover(turnover / market.turnover_divisor)
                if turnover else "unknown"),
            # The number that matters is not "is it liquid" but "how much of a
            # day's volume is my exit". Expressed as a share of one day rather
            # than as days-to-exit, because on a large cap the latter rounds
            # to 0.0 for every position and stops distinguishing anything.
            "pct_of_one_days_volume": (
                None if not turnover else round(value / turnover * 100.0, 4)),
        })
    s.rows = rows

    unknown = [r["symbol"] for r in rows if r["avg_daily_traded_value"] is None]
    s.add(Fact.known("Positions with a turnover figure",
                     len(rows) - len(unknown), SOURCE_CACHE,
                     note=(f"No cached bars for {', '.join(unknown)}."
                           if unknown else "All of them.")))
    s.note = (
        f"Turnover is the 20-session average traded VALUE from the cached "
        f"bars. A position worth a fraction of one percent of a day's volume "
        f"exits in minutes; one worth 10% or more IS the price. The floor for "
        f"this market is {market.turnover(market.min_avg_turnover)} a day and "
        f"it gates NEW positions; nothing re-checks it on one you hold.")


def _sizing(pack, cfg, snapshot, held, market) -> None:
    s = pack.section("Position sizing against the governors")
    pool = market.capital_pool
    try:
        pot = cfg.capital.capital_inr(pool)
        risk = cfg.capital.risk_inr(pool)
    except Exception as e:
        s.add(Fact.unknown("Per-trade risk budget",
                           f"unknown capital pool: {e}", source=SOURCE_CONFIG))
        return

    s.add(Fact.known("Capital pool", cfg.capital.pool_label(pool),
                     SOURCE_CONFIG,
                     note=f"Set at {cfg.capital.pool_field(pool)}."))
    s.add(Fact.known("Pot", round(pot, 2), SOURCE_CONFIG, unit="INR"))
    s.add(Fact.known(
        "Per-trade risk budget", round(risk, 2), SOURCE_CONFIG, unit="INR",
        note=f"session_stop_pct / max_entries_per_session = "
             f"{cfg.capital.risk_per_trade_pct:.2%} of the pot. This is the "
             f"ONE sizing formula in this repo; a briefing that proposes a "
             f"different one is proposing a second set of governors."))

    if pot <= 0:
        s.add(Fact.unknown(
            "Position value against the pot",
            f"the {cfg.capital.pool_label(pool)} pot is zero, so there is "
            f"nothing to size against. An unfunded pool stands a market down "
            f"rather than borrowing another pot's balance.",
            source=SOURCE_CONFIG))
        return

    s.rows = [{
        "symbol": p.symbol,
        "value_inr": round(snapshot.value_inr.get(p.key, 0.0), 2),
        "pct_of_pot": round(
            snapshot.value_inr.get(p.key, 0.0) / pot * 100.0, 2),
        "loss_at_a_10pct_fall": round(
            snapshot.value_inr.get(p.key, 0.0) * 0.10, 2),
    } for p in sorted(held, key=lambda x: -snapshot.value_inr.get(x.key, 0.0))]

    deployed = sum(snapshot.value_inr.get(p.key, 0.0) for p in held)
    s.add(Fact.known(
        "Deployed against the pot", round(deployed / pot * 100.0, 2),
        SOURCE_CONFIG, unit="%",
        note=f"{deployed:,.0f} of a {pot:,.0f} pot. Over 100% means the pot "
             f"recorded in config is smaller than the account actually "
             f"holding these - which makes every risk figure derived from it "
             f"wrong, in the reassuring direction."))
    s.note = (
        f"`SwingConfig.max_open_risk_r` caps TOTAL open risk across positions "
        f"at {cfg.swing.max_open_risk_r}R, which is exactly the session stop "
        f"the governors are built around. That cap is enforced when a ticket "
        f"is armed, not on holdings acquired any other way.")


def _actions(pack) -> None:
    s = pack.section("Hedges and rebalancing")
    s.note = (
        "Deliberately empty of computed conclusions. The correlation table, "
        "the concentration figures, the replayed drawdown and the sizing "
        "arithmetic above are the evidence; choosing a hedge is a judgment "
        "about cost and about a view, and this repo does not hold one.")
    s.judgment = [
        "Which three risks matter most here, and what would actually reduce "
        "them - noting that the cheapest reduction is usually selling the "
        "concentration rather than buying a hedge against it.",
        "Specific rebalancing percentages. Any target must respect the sizing "
        "section above: per-trade risk is session_stop_pct divided by "
        "max_entries_per_session, applied to the pool that pays.",
        "Whether the US-situs estate exposure in `swing/crossborder.py` "
        "belongs in the top three. It is structural and unrecoverable, "
        "unlike every market risk above.",
    ]
