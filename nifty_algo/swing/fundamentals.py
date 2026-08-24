"""
Balance-sheet lines and the earnings date, per symbol.

These feed two gates: the halal financial-ratio screen and the earnings
blackout. Both are slow to fetch - there is no batch endpoint, so it is one
request per symbol - and both change rarely: a balance sheet moves once a
quarter and an earnings date once a quarter. So this module caches hard, for
a week by default, and refreshes on demand rather than on every scan.

WHAT YAHOO ACTUALLY GIVES YOU, AND WHY THE CODE LOOKS DEFENSIVE

Yahoo's line-item names for Indian companies are inconsistent between sectors
and change between yfinance releases. "Total Debt" is sometimes absent and has
to be rebuilt from long- and short-term borrowings; receivables appear under
at least three labels. Every field here is therefore looked up through a list
of candidate names and may legitimately come back None.

None is not zero. A missing "Total Debt" does not mean a debt-free company,
and the halal screen treats absent data as a failure to verify rather than as
a pass - see `halal.py`.

THE CACHE IS KEYED `{market}:{SYMBOL}`, NOT BY SYMBOL. Bare tickers collide
across exchanges, and a collision here is not a crash - it is one company
screened against another company's balance sheet, which reads as a perfectly
ordinary verdict. The returned dict is still keyed by bare symbol because the
caller is already scoped to one market.

`currency` is recorded because the LSE quotes most of its shares in pence and
a few in pounds. `prices.py` has to divide the first group by 100 and must not
divide the second; the `info` call that happens here anyway is the only free
place to learn which is which.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

CACHE_NAME = "fundamentals.json"

# Candidate Yahoo line items, best first. The first one present wins.
_TOTAL_ASSETS = ("Total Assets",)
_TOTAL_DEBT = ("Total Debt",)
_LONG_TERM_DEBT = ("Long Term Debt", "Long Term Debt And Capital Lease Obligation")
_SHORT_TERM_DEBT = ("Current Debt", "Short Term Debt",
                    "Current Debt And Capital Lease Obligation")
_CASH = ("Cash Cash Equivalents And Short Term Investments",
         "Cash And Cash Equivalents")
_SHORT_TERM_INVESTMENTS = ("Other Short Term Investments",
                           "Short Term Investments")
_RECEIVABLES = ("Accounts Receivable", "Receivables",
                "Gross Accounts Receivable")


@dataclass
class Fundamentals:
    """
    One symbol's screening inputs. Every numeric field may be None, and None
    always means "Yahoo did not give us this", never "zero".
    """
    symbol: str
    total_assets: float | None = None
    total_debt: float | None = None
    cash_and_investments: float | None = None
    receivables: float | None = None
    market_cap: float | None = None
    balance_sheet_date: str | None = None      # ISO date of the statement used
    next_earnings_date: str | None = None      # ISO date, if Yahoo knows one
    yahoo_sector: str | None = None
    yahoo_industry: str | None = None
    currency: str | None = None                # as Yahoo reports it: GBp != GBP
    fetched_at: str | None = None
    error: str | None = None

    @property
    def has_balance_sheet(self) -> bool:
        return (self.total_assets is not None and self.total_assets > 0
                and self.total_debt is not None
                and self.cash_and_investments is not None
                and self.receivables is not None)

    def days_to_earnings(self, today: date | None = None) -> int | None:
        """Sessions-agnostic calendar days until results, or None if unknown."""
        if not self.next_earnings_date:
            return None
        try:
            when = date.fromisoformat(self.next_earnings_date)
        except ValueError:
            return None
        return (when - (today or date.today())).days

    @property
    def price_divisor(self) -> float | None:
        """
        What `prices.py` must divide this symbol's quotes by, or None if Yahoo
        did not say. "GBp" is pence and "GBP" is pounds - one character apart,
        two orders of magnitude apart.
        """
        if not self.currency:
            return None
        return 100.0 if self.currency == "GBp" else 1.0

    def age_days(self, now: datetime | None = None) -> int | None:
        if not self.fetched_at:
            return None
        try:
            then = datetime.fromisoformat(self.fetched_at)
        except ValueError:
            return None
        return ((now or datetime.now()) - then).days


def load_fundamentals(stocks: Iterable, cfg, market, force_refresh: bool = False,
                      progress=None) -> dict[str, Fundamentals]:
    """
    Fundamentals for every stock, fetching only what is missing or stale.

    `stocks` is an iterable of `universe.Stock`; `market` is a `markets.Market`
    and scopes the cache keys. `progress` is an optional callable
    (done, total, label) so a UI can show progress without this module knowing
    what Streamlit is.

    Keyed `{market}:{SYMBOL}` on disk, bare symbol in the return value - see
    the module docstring for why the first half of that matters.
    """
    stocks = list(stocks)
    cache_path = Path(cfg.swing.cache_dir) / CACHE_NAME
    cache = _read_cache(cache_path)
    max_age = timedelta(days=cfg.swing.fundamentals_cache_days)

    keyed = {s.symbol: market.qualified(s.symbol) for s in stocks}

    if force_refresh:
        stale = list(stocks)
    else:
        stale = [s for s in stocks if _is_stale(cache.get(keyed[s.symbol]), max_age)]

    out: dict[str, Fundamentals] = {
        s.symbol: cache[keyed[s.symbol]]
        for s in stocks if keyed[s.symbol] in cache
    }

    for i, stock in enumerate(stale):
        if progress:
            progress(i, len(stale), f"fundamentals: {stock.symbol}")
        out[stock.symbol] = _fetch_one(stock)

    if progress and stale:
        progress(len(stale), len(stale), "fundamentals complete")

    if stale:
        # Merge back into the whole cache rather than replacing it: another
        # market's entries live in the same file and must survive this write.
        cache.update({keyed[sym]: f for sym, f in out.items()})
        _write_cache(cache_path, cache)
    return out


def cache_age_days(cfg) -> int | None:
    """How old the fundamentals cache is, for the freshness strip."""
    path = Path(cfg.swing.cache_dir) / CACHE_NAME
    if not path.exists():
        return None
    return (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days


# ---------------- one symbol ----------------

def _fetch_one(stock) -> Fundamentals:
    f = Fundamentals(symbol=stock.symbol,
                     fetched_at=datetime.now().isoformat(timespec="seconds"))
    try:
        import yfinance as yf
    except ImportError:                                   # pragma: no cover
        f.error = "yfinance is not installed"
        return f

    try:
        t = yf.Ticker(stock.yf_ticker)
    except Exception as e:
        f.error = f"could not open ticker: {e}"
        return f

    _read_balance_sheet(t, f)
    _read_market_cap(t, f)
    _read_earnings_date(t, f)
    _read_classification(t, f)
    return f


def _read_balance_sheet(t, f: Fundamentals) -> None:
    try:
        bs = t.balance_sheet
    except Exception as e:
        f.error = f"balance sheet unavailable: {e}"
        return
    if bs is None or getattr(bs, "empty", True):
        f.error = "Yahoo returned no balance sheet"
        return

    # Columns are statement dates, most recent first.
    try:
        col = bs.columns[0]
        f.balance_sheet_date = str(getattr(col, "date", lambda: col)())[:10]
    except Exception:
        col = bs.columns[0]

    f.total_assets = _line(bs, col, _TOTAL_ASSETS)

    debt = _line(bs, col, _TOTAL_DEBT)
    if debt is None:
        # Rebuild it. A company with borrowings reported only as long- and
        # short-term lines is not a company with no debt, and treating it as
        # one would hand it a free pass through the leverage screen.
        parts = [_line(bs, col, _LONG_TERM_DEBT), _line(bs, col, _SHORT_TERM_DEBT)]
        present = [p for p in parts if p is not None]
        debt = float(sum(present)) if present else None
    f.total_debt = debt

    # The screen asks about interest-bearing assets, so short-term
    # investments belong in this number alongside cash. Yahoo's combined line
    # already includes them; its narrow line does not, and adding them to the
    # combined one would double-count.
    cash = _line(bs, col, _CASH)
    if cash is not None and _found_label(bs, _CASH) == "Cash And Cash Equivalents":
        extra = _line(bs, col, _SHORT_TERM_INVESTMENTS)
        if extra is not None:
            cash += extra
    f.cash_and_investments = cash

    f.receivables = _line(bs, col, _RECEIVABLES)


def _read_market_cap(t, f: Fundamentals) -> None:
    # fast_info avoids the slow, frequently-throttled .info round trip.
    for getter in (lambda: t.fast_info["market_cap"],
                   lambda: t.info.get("marketCap")):
        try:
            value = getter()
            if value:
                f.market_cap = float(value)
                return
        except Exception:
            continue


def _read_earnings_date(t, f: Fundamentals) -> None:
    try:
        cal = t.calendar
    except Exception:
        return
    when = None
    if isinstance(cal, dict):
        dates = cal.get("Earnings Date") or cal.get("earningsDate")
        if isinstance(dates, (list, tuple)) and dates:
            when = dates[0]
        elif dates:
            when = dates
    elif cal is not None and hasattr(cal, "loc"):
        try:
            when = cal.loc["Earnings Date"].iloc[0]
        except Exception:
            when = None
    if when is None:
        return
    try:
        f.next_earnings_date = str(getattr(when, "date", lambda: when)())[:10]
    except Exception:
        f.next_earnings_date = str(when)[:10]


def _read_classification(t, f: Fundamentals) -> None:
    """
    Yahoo's own sector/industry, kept only as a cross-check against the
    committed universe file. The universe file is authoritative for the halal
    activity screen - Yahoo classifies several Indian names oddly - but a
    disagreement is worth being able to see.
    """
    try:
        info = t.info
    except Exception:
        return
    if isinstance(info, dict):
        f.yahoo_sector = info.get("sector")
        f.yahoo_industry = info.get("industry")
        currency = info.get("currency")
        # Kept verbatim, case included: "GBp" and "GBP" are different units.
        f.currency = str(currency) if currency else None


# ---------------- helpers ----------------

def _line(bs, col, candidates: tuple[str, ...]) -> float | None:
    for name in candidates:
        if name in bs.index:
            try:
                value = bs.loc[name, col]
            except Exception:
                continue
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value == value:                     # not NaN
                return value
    return None


def _found_label(bs, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in bs.index:
            return name
    return None


def _is_stale(f: Fundamentals | None, max_age: timedelta) -> bool:
    if f is None or not f.fetched_at:
        return True
    try:
        then = datetime.fromisoformat(f.fetched_at)
    except ValueError:
        return True
    return datetime.now() - then > max_age


def _read_cache(path: Path) -> dict[str, Fundamentals]:
    if not path.exists():
        return {}
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, Fundamentals] = {}
    for key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        known = {k: v for k, v in payload.items()
                 if k in Fundamentals.__dataclass_fields__}
        # The dict key may be qualified (`us:AAPL`); the dataclass wants the
        # bare symbol, and the stored payload is the authority on it.
        known["symbol"] = payload.get("symbol") or key.rsplit(":", 1)[-1]
        try:
            out[key] = Fundamentals(**known)
        except TypeError:
            continue
    return out


def _write_cache(path: Path, data: dict[str, Fundamentals]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in data.items()}
        path.write_text(json.dumps(payload, indent=1, default=str),
                        encoding="utf-8")
    except Exception:
        pass
