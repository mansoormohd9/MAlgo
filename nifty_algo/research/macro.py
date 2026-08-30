"""
The macro impact assessment: what the world is charging, and what your book
is exposed to.

WHAT THIS MODULE COMPUTES AND WHAT IT REFUSES TO. It computes the priced
facts - yields, the curve, the dollar, the rupee, crude, gold, volatility, the
indices - and it computes YOUR EXPOSURE to each of them, which is the half a
generic macro summary always leaves out. What it does not do is forecast. The
Fed's next move, geopolitics, trade policy and supply chains have no series;
they are declared in `Section.judgment` so that whatever writes the prose is
told, explicitly and in the data, that it is being asked for an opinion rather
than handed an answer.

THE SENSITIVITY TABLE IS THE POINT. "Rates are up 41bps over twelve months" is
in every newspaper. "Rates are up 41bps and the four positions that are 62% of
your book have a -0.31 correlation to that move over 250 sessions" is a fact
about you, and it is the one thing this repo is positioned to produce, because
the daily bars for your holdings are already cached for the swing scanner.

CORRELATION IS NOT CAUSATION AND THE OUTPUT SAYS SO. A 250-session window
across one regime measures that regime. Every figure carries its `n`, anything
under `PortfolioConfig.min_correlation_sessions` is reported unavailable
rather than computed, and the caveat sits above the table rather than under it.

WHY THE PORTFOLIO CAN BE ABSENT. The macro half stands alone and is worth
reading with no broker connected at all. If the snapshot is incomplete the
sections that need it say so - and portfolio percentages are withheld
entirely, because `PortfolioSnapshot.weight()` returns None on a partial book
rather than dividing by a denominator it does not have.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config, DEFAULT
from ..portfolio import aggregate as portfolio_mod
from ..swing import markets as markets_mod
from . import exposure, holdings_prices
from .base import SOURCE_CONFIG, SOURCE_PORTFOLIO, Fact, FactPack
from .providers import macro_series as ms

REPORT = "macro"
TITLE = "Macro impact assessment"



def build(cfg: Config = DEFAULT, market_key: str = markets_mod.INDIA,
          snapshot=None, force_refresh: bool = False,
          progress=None) -> FactPack:
    """
    The whole pack. `snapshot` defaults to reading every enabled connector.
    """
    market = markets_mod.get(cfg, market_key)
    if snapshot is None:
        _say(progress, "reading holdings")
        snapshot = portfolio_mod.load(cfg)

    _say(progress, "downloading macro series")
    readings = ms.load(cfg, force_refresh=force_refresh)
    manual = ms.load_manual(cfg)

    pack = FactPack(report=REPORT, title=TITLE,
                    inputs={"market": market.key,
                            "market_label": market.label,
                            "correlation_days": cfg.portfolio.correlation_days})
    pack.caveats.extend(snapshot.caveats())
    pack.caveats.append(
        "Every series below is MARKET-PRICED, not forecast. A yield is what "
        "borrowing costs today, not what anyone expects next quarter.")

    _rates(pack, readings, manual)
    _inflation(pack, readings, manual)
    _growth(pack, readings, manual)
    _currency(pack, readings, snapshot, market)
    _employment(pack, manual)
    _risk(pack, readings)

    _say(progress, "pricing your holdings")
    _book(pack, cfg, snapshot, market, readings, progress=progress)
    _rotation(pack)
    return pack


def _say(progress, message: str) -> None:
    if progress:
        progress(message)


# ---------------- the priced series ----------------

def _fact(reading: ms.Reading, label: str | None = None) -> Fact:
    """One `Reading` as a `Fact`, availability and all."""
    if not reading.available:
        return Fact.unknown(label or reading.series.label, reading.note,
                            source=reading.source)
    changes = ", ".join(
        f"{window} {value:+,.1f}{reading.change_unit}"
        for window, value in (("1m", reading.change_1m),
                              ("3m", reading.change_3m),
                              ("12m", reading.change_12m))
        if value is not None)
    return Fact.known(label or reading.series.label, round(reading.last, 4),
                      reading.source, as_of=reading.as_of,
                      unit=reading.series.unit,
                      note=f"{changes}. {reading.series.why}" if changes
                           else reading.series.why)


def _rates(pack, readings, manual) -> None:
    s = pack.section("Rate environment")
    for key in ("us_10y", "us_3m"):
        s.add(_fact(readings[key]))
    s.add(_fact(manual["india_repo"]))

    ten, three = readings["us_10y"], readings["us_3m"]
    if ten.available and three.available:
        slope = ten.last - three.last
        s.add(Fact.known(
            "US curve slope, 10y minus 3m", round(slope, 3),
            "derived from ^TNX and ^IRX", as_of=ten.as_of, unit="%",
            note=("Inverted - the front end pays more than the long end, "
                  "which historically leads a slowdown by quarters rather "
                  "than weeks." if slope < 0 else
                  "Positively sloped - lending long pays more than lending "
                  "short, the ordinary state.")))
    else:
        s.add(Fact.unknown("US curve slope, 10y minus 3m",
                           "needs both ^TNX and ^IRX, and one is unavailable"))

    s.note = (
        "The long yield is the discount rate on distant earnings, so a rise "
        "in it costs a growth multiple more than a value one arithmetically, "
        "before any view about the economy. The direction of the last twelve "
        "months matters more here than the level.")
    s.judgment = [
        "Where policy goes over the next 6-12 months. Nothing in this repo "
        "forecasts a central bank; the curve above is what the market is "
        "PAYING, which is evidence but not a projection.",
        "Whether the growth/value split in this book is intentional or "
        "incidental. The sensitivity table below measures it; only you know "
        "if it was chosen.",
    ]


def _inflation(pack, readings, manual) -> None:
    s = pack.section("Inflation and input costs")
    s.add(_fact(manual["india_cpi_yoy"]))
    s.add(_fact(manual["us_cpi_yoy"]))
    for key in ("crude", "gold"):
        s.add(_fact(readings[key]))
    s.note = (
        "India imports roughly 85% of its crude, so the oil price is not one "
        "input among many - it moves headline inflation, the current account "
        "and the rupee together, and it does it across whole sectors rather "
        "than single names. Where the published CPI figures are unavailable, "
        "crude is the best free proxy this build has for the direction of "
        "the Indian print, and it is a proxy rather than the number.")
    s.judgment = [
        "Which sectors benefit and which suffer. The correlation table below "
        "gives the measured answer for the names YOU hold; the general case "
        "(oil marketing versus upstream, importers versus exporters) is "
        "domain knowledge this repo does not encode.",
    ]


def _growth(pack, readings, manual) -> None:
    s = pack.section("Growth and corporate earnings")
    s.add(_fact(manual["india_gdp_yoy"]))
    for key in ("nifty", "nifty_bank", "sp500"):
        s.add(_fact(readings[key]))

    nifty, bank = readings["nifty"], readings["nifty_bank"]
    if nifty.available and bank.available and None not in (nifty.change_12m,
                                                           bank.change_12m):
        spread = bank.change_12m - nifty.change_12m
        s.add(Fact.known(
            "Banks versus the market, 12 months", round(spread, 2),
            "derived from ^NSEBANK and ^NSEI", as_of=nifty.as_of, unit="%",
            note=("Banks are the most rate-sensitive index on the exchange, "
                  "so this spread is the cleanest free read on how the rate "
                  "environment is actually landing on earnings - as opposed "
                  "to how it is expected to.")))
    s.note = (
        "Index levels are not earnings. They are earnings multiplied by what "
        "the market will pay for them, and over twelve months the multiple "
        "usually moves more than the earnings do.")
    s.judgment = [
        "The earnings forecast itself. This build carries no consensus "
        "estimates for Indian names - Yahoo's coverage of them is sparse to "
        "absent - so any EPS growth figure in the briefing must be labelled "
        "as the writer's own and sourced.",
    ]


def _currency(pack, readings, snapshot, market) -> None:
    s = pack.section("Currency exposure")
    for key in ("usdinr", "dxy"):
        s.add(_fact(readings[key]))

    # The exposure half: how much of the book is not in rupees at all.
    foreign = {c: rows for c, rows in snapshot.by_currency().items()
               if c != "INR"}
    if not snapshot.complete:
        s.add(Fact.unknown(
            "Share of the book in foreign currency",
            "the holdings snapshot is incomplete, so this would be a share of "
            "a partial book - see the caveats above",
            source=SOURCE_PORTFOLIO))
    elif snapshot.total_inr <= 0:
        s.add(Fact.unknown("Share of the book in foreign currency",
                           "no positions with a value were found",
                           source=SOURCE_PORTFOLIO))
    else:
        value = sum(snapshot.value_inr.get(p.key, 0.0)
                    for rows in foreign.values() for p in rows)
        s.add(Fact.known(
            "Share of the book in foreign currency",
            round(value / snapshot.total_inr * 100.0, 2),
            SOURCE_PORTFOLIO, unit="%",
            note=(f"Currencies held: {', '.join(sorted(foreign)) or 'none'}. "
                  f"An unhedged foreign holding is two bets - the asset and "
                  f"the cross - and the rupee leg is usually the one nobody "
                  f"decided to take.")))

    s.rows = [{"currency": currency,
               "positions": len(rows),
               "value_inr": round(sum(snapshot.value_inr.get(p.key, 0.0)
                                      for p in rows), 2),
               "rate": snapshot.rate_notes.get(currency, "not converted")}
              for currency, rows in sorted(snapshot.by_currency().items())]
    s.note = (
        "Values are in rupees throughout, because that is what this account "
        "spends and reports in - the market being scanned does not change "
        "that. Every line above in another currency carries an FX exposure "
        "whether or not it was chosen, and `swing/fx.py` refuses to convert "
        "at a guessed rate, so anything listed as 'not converted' is excluded "
        "from every total here rather than estimated.")


def _employment(pack, manual) -> None:
    s = pack.section("Employment and the consumer")
    s.add(_fact(manual["us_unemployment"]))
    s.note = (
        "This section is almost entirely unavailable by design rather than by "
        "accident. India's labour data (PLFS, CMIE) has no free machine "
        "endpoint, and the US series needs a FRED key. Rather than fill the "
        "gap with a proxy that would read like a measurement, the section "
        "reports what it does not have.")
    s.judgment = [
        "Employment trends and what they imply for consumer spending. There "
        "is no series behind this in the pack - any figure quoted must be "
        "attributed to the writer, with its source and date.",
    ]


def _risk(pack, readings) -> None:
    s = pack.section("Risk and volatility")
    s.add(_fact(readings["india_vix"]))
    vix = readings["india_vix"]
    if vix.available:
        s.add(Fact.known(
            "What India VIX implies for the next month", round(vix.last / 3.46, 2),
            "derived from ^INDIAVIX", as_of=vix.as_of, unit="% (1 sd)",
            note=("VIX is an annualised one-standard-deviation move; dividing "
                  "by sqrt(12) puts it on a monthly footing. It is priced, "
                  "not predicted, and one standard deviation is exceeded "
                  "about a third of the time by construction.")))
    s.note = (
        "A low VIX is not the same as low risk. It is a low PRICE for "
        "insurance, which is a statement about positioning and recent "
        "realised volatility, and it is lowest immediately before the moves "
        "that make it high.")
    s.judgment = [
        "Geopolitics, trade policy and supply chains. There is no series for "
        "any of these in this pack. They belong in the briefing, clearly "
        "marked as the writer's assessment rather than as measurement.",
    ]


# ---------------- the half that is about you ----------------

def _book(pack, cfg, snapshot, market, readings, progress=None) -> None:
    s = pack.section("Your book against these factors")
    held = [p for p in snapshot.positions if p.market == market.key]

    if not held:
        s.note = (f"No {market.label} positions were found, so there is "
                  f"nothing to measure against the series above. The macro "
                  f"sections stand on their own.")
        s.add(Fact.unknown("Positions measured",
                           "no holdings in this market", source=SOURCE_PORTFOLIO))
        return

    bars, _benchmark, price_note = holdings_prices.bars_for(
        snapshot, cfg, market.key)
    days = cfg.portfolio.correlation_days
    stock_returns = holdings_prices.returns(bars, days)

    history = ms.history(cfg)
    moves = ms.factor_moves(history) if history is not None else pd.DataFrame()

    sectors = exposure.sector_map(market)
    rows = []
    for p in sorted(held, key=lambda x: -snapshot.value_inr.get(x.key, 0.0)):
        weight = snapshot.weight(p.key)
        row = {
            "symbol": p.symbol,
            "key": p.key,
            "sector": sectors.get(p.symbol, "unclassified"),
            "value_inr": round(snapshot.value_inr.get(p.key, 0.0), 2),
            # None, not 0.0 - a partial book has no honest percentage, and a
            # zero here would be read as "a rounding-error position".
            "weight_pct": None if weight is None else round(weight * 100.0, 2),
        }
        row.update(exposure.sensitivities(p.symbol, stock_returns,
                                          moves, cfg))
        rows.append(row)
    s.rows = rows

    measured = sum(1 for r in rows if r.get("nifty_beta") is not None)
    s.add(Fact.known("Positions measured", len(rows), SOURCE_PORTFOLIO,
                     note=f"{measured} of them had enough overlapping "
                          f"sessions to correlate."))
    s.add(Fact.known("Correlation window", days, SOURCE_CONFIG, unit="sessions",
                     note=f"PortfolioConfig.correlation_days. Anything with "
                          f"fewer than {cfg.portfolio.min_correlation_sessions} "
                          f"overlapping sessions is reported as null rather "
                          f"than computed."))

    _sector_weights(pack, rows, snapshot)

    s.note = (
        f"{price_note} Beta is against the {market.benchmark_label}; the "
        f"factor columns are correlations of daily returns, not betas, and a "
        f"correlation of -0.3 to the 10-year says these move opposite each "
        f"other about a third of the time, not that rates cause the move.")
    pack.caveats.append(
        f"Correlations are measured over {days} sessions - one regime. They "
        f"describe how this book behaved recently and are not a forecast of "
        f"how it will behave in a different one.")


def _sector_weights(pack, rows, snapshot) -> None:
    """
    Sector concentration - its own section, not a second table hidden inside
    the position one. A consumer that has to filter rows by a marker column to
    find out which table it is reading will eventually not bother.
    """
    s = pack.section("Sector concentration")
    totals: dict[str, float] = {}
    for r in rows:
        totals[r["sector"]] = totals.get(r["sector"], 0.0) + r["value_inr"]

    book = snapshot.total_inr
    can_weight = snapshot.complete and book > 0
    s.rows = [
        {"sector": sector, "positions": sum(1 for r in rows
                                            if r["sector"] == sector),
         "value_inr": round(value, 2),
         "weight_pct": round(value / book * 100.0, 2) if can_weight else None}
        for sector, value in sorted(totals.items(), key=lambda kv: -kv[1])
    ]

    if not can_weight:
        s.add(Fact.unknown(
            "Sector weights",
            "the holdings snapshot is incomplete, so a sector share would be "
            "a share of a partial book. Absolute values are still shown.",
            source=SOURCE_PORTFOLIO))
        return

    top = s.rows[0] if s.rows else None
    if top:
        s.add(Fact.known("Largest sector", top["sector"], SOURCE_PORTFOLIO,
                         note=f"{top['weight_pct']}% of the book across "
                              f"{top['positions']} position(s)."))
    s.note = (
        "The swing scanner caps itself at one pick per sector "
        "(`SwingConfig.max_per_sector`) on the view that three names in one "
        "sector is one bet. That cap governs what the scanner PROPOSES; it "
        "says nothing about what the account has accumulated, which is what "
        "this table measures.")


def _rotation(pack) -> None:
    s = pack.section("Sector rotation and portfolio actions")
    s.note = (
        "Deliberately empty of computed conclusions. Everything needed to "
        "reach one is above - the priced series, their twelve-month "
        "direction, this book's sector weights and its measured sensitivity "
        "to each factor - but the step from those to 'rotate into X' is a "
        "judgment about a cycle, and this repo does not encode one.")
    s.judgment = [
        "Where in the economic cycle we are, and the sector rotation that "
        "follows from it.",
        "Specific portfolio adjustments. Any that are proposed must respect "
        "the risk governors in CapitalConfig - per-trade risk is "
        "session_stop_pct / max_entries_per_session applied to the pool that "
        "is paying, and no briefing may invent a second sizing rule.",
        "Timing. None of the series above carries a lead time, and attaching "
        "one to them is a forecast rather than a reading.",
    ]
