"""
The macro series a briefing can actually stand on, and the ones it cannot.

TWO CATEGORIES, AND THE SPLIT IS THE POINT.

MARKET-PRICED SERIES are free, daily and honest: yields, the dollar, the
rupee, crude, gold, volatility, the indices. Yahoo serves all of them, they
need no key, and they are what a market is currently *paying* rather than what
a forecaster expects. Everything in `SERIES` is one of these.

PUBLISHED STATISTICS are not. India's CPI, GDP, IIP and the repo rate come
from MoSPI and the RBI, neither of which offers a free machine endpoint this
build can rely on; the US equivalents need a FRED key. So they are declared in
`MANUAL_SERIES` and read from an optional CSV you fill in yourself. Absent,
they are `Fact.unknown` with the reason attached - never a blank cell and
never a zero, because "inflation: 0.0%" is a sentence a briefing would
cheerfully build a sector rotation on.

WHAT IS NOT HERE AT ALL, and must be labelled judgment rather than data: the
Fed's next move, geopolitics, trade policy, supply chains. There is no series
for those. `macro.py` puts them in `Section.judgment` so the writer of the
prose is explicitly told it is being asked for an opinion.

CHANGES ARE COMPUTED IN THE RIGHT UNIT. A yield that goes 4.0 -> 4.4 has risen
40 BASIS POINTS, not 10%, and an index that goes 24000 -> 26400 has risen 10%,
not 2400 points. Reporting either in the other's unit is the kind of number
that survives review because it is dimensionally plausible.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

CACHE_NAME = "macro_series.parquet"
MANUAL_NAME = "macro_manual.csv"

#: How a level moves, and therefore how a change must be quoted.
RATE = "rate"              # quoted in percent; changes are basis points
LEVEL = "level"            # index, price or cross; changes are percent


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    ticker: str
    kind: str
    unit: str
    why: str                # why this series is in a portfolio briefing


#: Market-priced, free, daily. Ordered the way a briefing reads them.
SERIES: tuple[Series, ...] = (
    Series("us_10y", "US 10-year Treasury yield", "^TNX", RATE, "%",
           "The global discount rate. It sets what a rupee of distant "
           "earnings is worth, which is why it moves growth harder than "
           "value."),
    Series("us_3m", "US 13-week T-bill yield", "^IRX", RATE, "%",
           "The front end - the closest free proxy for where policy actually "
           "is, as opposed to where anyone says it is going. Against the "
           "10-year it gives the curve slope."),
    Series("dxy", "US dollar index", "DX-Y.NYB", LEVEL, "index",
           "Dollar strength is a headwind for emerging-market equities and "
           "for the unhedged rupee value of foreign holdings."),
    Series("usdinr", "USD/INR", "USDINR=X", LEVEL, "INR",
           "Converts every foreign line in the book, and is itself an "
           "exposure for an Indian resident holding US-domiciled funds."),
    Series("crude", "Brent-equivalent crude (WTI front month)", "CL=F", LEVEL,
           "USD/bbl",
           "India imports around 85% of its oil. This is the single largest "
           "external input to Indian inflation, the current account and the "
           "rupee, and it moves whole sectors rather than single names."),
    Series("gold", "Gold", "GC=F", LEVEL, "USD/oz",
           "A read on real rates and on risk appetite, and a large household "
           "asset in India specifically."),
    Series("india_vix", "India VIX", "^INDIAVIX", LEVEL, "index",
           "What the options market is charging for the next month of Nifty "
           "risk. The one forward-looking number here that is priced rather "
           "than forecast."),
    Series("nifty", "Nifty 50", "^NSEI", LEVEL, "index",
           "The benchmark every Indian position is measured against."),
    Series("nifty_bank", "Nifty Bank", "^NSEBANK", LEVEL, "index",
           "Banks are the most rate-sensitive index in the market, so this "
           "against the Nifty is the cleanest read on how rates are landing."),
    Series("sp500", "S&P 500", "^GSPC", LEVEL, "index",
           "The benchmark for the foreign side of the book."),
)

#: No free machine endpoint. Recorded by hand or reported unavailable.
MANUAL_SERIES: tuple[Series, ...] = (
    Series("india_cpi_yoy", "India CPI inflation, YoY", "", RATE, "%",
           "MoSPI publishes this monthly. It is what the RBI targets, so it "
           "is the number the repo rate is a function of."),
    Series("india_repo", "RBI repo rate", "", RATE, "%",
           "The policy rate itself. The RBI publishes it; there is no free "
           "API this build will depend on."),
    Series("india_gdp_yoy", "India real GDP growth, YoY", "", RATE, "%",
           "The top line corporate earnings ultimately track."),
    Series("us_cpi_yoy", "US CPI inflation, YoY", "", RATE, "%",
           "Needs a FRED key. Drives the Fed, which drives the 10-year, "
           "which is already priced above."),
    Series("us_unemployment", "US unemployment rate", "", RATE, "%",
           "The other half of the Fed's mandate, and the read on US consumer "
           "spending."),
)


@dataclass
class Reading:
    """One series as it stands today, with its own history behind it."""
    series: Series
    last: float | None = None
    as_of: str = ""
    change_1m: float | None = None
    change_3m: float | None = None
    change_12m: float | None = None
    sessions: int = 0
    available: bool = False
    note: str = ""
    source: str = ""

    @property
    def change_unit(self) -> str:
        """bps for a rate, % for a level. See the module docstring."""
        return "bps" if self.series.kind == RATE else "%"

    def summary(self) -> str:
        if not self.available:
            return f"{self.series.label}: unavailable - {self.note}"
        bits = [f"{self.last:,.2f}{self.series.unit}"]
        for window, value in (("1m", self.change_1m), ("3m", self.change_3m),
                              ("12m", self.change_12m)):
            if value is not None:
                bits.append(f"{window} {value:+,.1f}{self.change_unit}")
        return f"{self.series.label}: " + ", ".join(bits)


def load(cfg, force_refresh: bool = False, series=None) -> dict[str, Reading]:
    """
    Every market-priced series, from cache when it is fresh enough.

    A series that fails is `available=False` WITH A REASON and the others still
    return. One dead ticker must not cost the whole macro picture - but it must
    also not silently vanish from a briefing that then reads as complete.
    """
    wanted = list(series or SERIES)
    frame = _frame(cfg, wanted, force_refresh)
    if frame is None:
        return {s.key: Reading(s, available=False,
                               note="no macro history could be downloaded and "
                                    "no usable cache exists",
                               source="yfinance")
                for s in wanted}
    return {s.key: _read(s, frame) for s in wanted}


def load_manual(cfg) -> dict[str, Reading]:
    """
    The published statistics, from `data/macro_manual.csv` if you keep one.

    Columns: `key,value,as_of,source`. Anything absent comes back unavailable
    with the reason - which for these is a fact about the world (no free
    endpoint) rather than an outage, and the note says so.
    """
    path = Path(cfg.swing.cache_dir).parent / MANUAL_NAME
    recorded: dict[str, dict] = {}
    if path.exists():
        try:
            import csv
            with path.open("r", encoding="utf-8", newline="") as f:
                lines = [ln for ln in f if not ln.lstrip().startswith("#")]
            for row in csv.DictReader(lines):
                key = (row.get("key") or "").strip()
                if key:
                    recorded[key] = row
        except Exception as e:
            # Unreadable is not empty - same rule as the manual position file.
            return {s.key: Reading(s, available=False,
                                   note=f"{path} exists but could not be read "
                                        f"({e})")
                    for s in MANUAL_SERIES}

    out: dict[str, Reading] = {}
    for s in MANUAL_SERIES:
        row = recorded.get(s.key)
        value = _num(row.get("value")) if row else None
        if value is None:
            out[s.key] = Reading(
                s, available=False,
                note=f"not published on any free machine endpoint this build "
                     f"will depend on. Record it in {path} as "
                     f"`{s.key},<value>,<YYYY-MM-DD>,<source>` to have it "
                     f"appear here.")
            continue
        out[s.key] = Reading(s, last=value, available=True,
                             as_of=(row.get("as_of") or "").strip(),
                             source=(row.get("source") or "recorded by hand"),
                             note="recorded by hand - not verified by this build")
    return out


# ---------------- download and cache ----------------

def _frame(cfg, wanted: list[Series], force_refresh: bool) -> pd.DataFrame | None:
    """
    Daily closes for every ticker, one column each, from cache when fresh.

    Cached exactly like `swing/prices.py` does it and for the same reason:
    Streamlit re-runs the whole script on every interaction, so an uncached
    fetch means re-downloading the world to scroll a page.
    """
    path = Path(cfg.swing.cache_dir) / CACHE_NAME
    tickers = [s.ticker for s in wanted if s.ticker]

    if not force_refresh:
        cached = _read_cache(path, cfg.swing.price_cache_hours)
        if cached is not None and all(t in cached.columns for t in tickers):
            return cached

    downloaded = _download(tickers)
    if downloaded is None or downloaded.empty:
        # Stale beats absent: an old rate quoted WITH ITS DATE is a fact, and
        # `_read` puts the date on every reading. A missing one is nothing.
        return _read_cache(path, max_age_hours=None)

    _write_cache(path, downloaded)
    return downloaded


def _download(tickers: list[str]) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:                                   # pragma: no cover
        return None
    try:
        raw = yf.download(tickers, period="2y", interval="1d",
                          auto_adjust=False, progress=False,
                          group_by="ticker", threads=True)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None

    out = pd.DataFrame(index=raw.index)
    for ticker in tickers:
        try:
            column = (raw[ticker]["Close"] if isinstance(raw.columns,
                                                         pd.MultiIndex)
                      else raw["Close"])
        except Exception:
            continue
        out[ticker] = pd.to_numeric(column, errors="coerce")
    return out.dropna(how="all")


def _read_cache(path: Path, max_age_hours: int | None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if max_age_hours is not None:
        age = (datetime.now()
               - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
        if age > max_age_hours * 3600:
            return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    except Exception:
        pass          # a cache that cannot be written must not fail the read


# ---------------- one series ----------------

def _read(s: Series, frame: pd.DataFrame) -> Reading:
    if s.ticker not in frame.columns:
        return Reading(s, available=False, source="yfinance",
                       note=f"{s.ticker} was not in the download - Yahoo "
                            f"sometimes drops a symbol from a batch")
    column = frame[s.ticker].dropna()
    if column.empty:
        return Reading(s, available=False, source="yfinance",
                       note=f"{s.ticker} returned no closes")

    last = float(column.iloc[-1])
    reading = Reading(s, last=last, available=True, source=f"yfinance {s.ticker}",
                      sessions=int(len(column)),
                      as_of=str(column.index[-1])[:10])
    # ~21 / 63 / 252 sessions. Positional rather than date-based so a holiday
    # cannot silently shorten a window to nothing.
    for attr, back in (("change_1m", 21), ("change_3m", 63),
                       ("change_12m", 252)):
        if len(column) > back:
            setattr(reading, attr, _change(s, float(column.iloc[-1 - back]), last))
    return reading


def _change(s: Series, then: float, now: float) -> float | None:
    """Basis points for a rate, percent for a level. Never the other way."""
    if s.kind == RATE:
        return (now - then) * 100.0
    if then == 0:
        return None
    return (now / then - 1.0) * 100.0


def _num(raw) -> float | None:
    text = str(raw or "").strip().replace(",", "").replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def history(cfg, force_refresh: bool = False) -> pd.DataFrame | None:
    """
    The raw daily frame behind `load()`, columns keyed by SERIES key.

    Exposed because the portfolio sections need to correlate a holding's
    returns against a factor's, and re-downloading the same two years to do it
    would be the third copy of this data in one run.
    """
    frame = _frame(cfg, list(SERIES), force_refresh)
    if frame is None:
        return None
    renamed = {s.ticker: s.key for s in SERIES if s.ticker in frame.columns}
    return frame.rename(columns=renamed)[list(renamed.values())]


def factor_moves(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Each factor's daily move, in the unit that factor actually moves in.

    A RATE series moves in absolute points (a yield going 4.0 -> 4.1 is the
    event), a LEVEL series in percent. Taking percent changes of a yield makes
    a move from 0.5% to 0.6% look twenty times larger than the same 10bp move
    at 5%, which would rank every rate sensitivity in the book by nothing more
    than where yields happened to be.
    """
    kinds = {s.key: s.kind for s in SERIES}
    out = {}
    for key in frame.columns:
        column = pd.to_numeric(frame[key], errors="coerce")
        out[key] = (column.diff() if kinds.get(key) == RATE
                    else column.pct_change())
    return pd.DataFrame(out).dropna(how="all")
