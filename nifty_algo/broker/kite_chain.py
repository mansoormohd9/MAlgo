"""
Real NIFTY option chain from Kite Connect.

This is what finally makes the liquidity gates in `risk.py` mean something.
On a synthetic chain the open interest is 500,000 and the spread is 0.4%
because `pricing.SyntheticChainSpec` invented numbers chosen to pass - so
`select_strike()` has never once rejected a contract for illiquidity. Against
a real chain those gates start doing the job they were written for.

Two things here are worth understanding before trusting the output:

DELTA IS DERIVED, NOT GIVEN. Kite returns a premium; strike selection needs a
delta. So each strike's premium is inverted to an implied volatility and the
delta computed from that (`pricing.implied_vol` -> `pricing.bs_delta`). Doing
it per strike means the volatility SKEW is respected. Assuming one flat IV
across the chain - which is what the synthetic path does - gets PE deltas
materially wrong, and a wrong delta buys the wrong strike.

EXPIRIES COME FROM THE INSTRUMENT DUMP. `chain.next_weekly_expiry()` hardcodes
a guess at the weekly expiry weekday, and its own docstring flags that the day
has moved twice recently. The dump states the real expiry for every contract,
so when this provider is live nothing has to guess. A wrong expiry means a
wrong time-to-expiry means a wrong delta, silently.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import Config, DEFAULT
from ..pricing import bs_delta, implied_vol, time_to_expiry_years
from ..risk import OptionQuote
from .kite_auth import KiteSession, NotAuthenticated

INSTRUMENT_CACHE = Path(".kite_instruments.csv")

# Kite accepts up to 500 instruments per quote() call. A chain of 12 strikes
# each side is 25 contracts, so batching is not a real constraint - but the
# cap is enforced here anyway rather than discovered in production.
MAX_QUOTE_BATCH = 500


@dataclass
class ChainRow:
    """One contract, before it becomes an OptionQuote."""
    tradingsymbol: str
    strike: int
    option_type: str
    expiry: date
    last_price: float
    bid: float
    ask: float
    open_interest: int
    volume: int
    iv: Optional[float]
    delta: Optional[float]

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0 and self.ask >= self.bid:
            return (self.bid + self.ask) / 2.0
        return self.last_price

    @property
    def spread_pct(self) -> float:
        ref = self.mid
        if ref <= 0 or self.bid <= 0 or self.ask <= 0:
            return 1.0
        return (self.ask - self.bid) / ref


class KiteChain:
    """Loads the NFO instrument dump once a day and quotes strikes on demand."""

    def __init__(self, session: KiteSession | None = None, cfg: Config = DEFAULT,
                 cache_file: Path | str = INSTRUMENT_CACHE):
        self.session = session or KiteSession()
        self.cfg = cfg
        self.cache_file = Path(cache_file)
        self._instruments: Optional[pd.DataFrame] = None
        self._loaded_on: Optional[date] = None

    # ---------------- instrument dump ----------------

    def instruments(self, refresh: bool = False) -> pd.DataFrame:
        """
        Every live NIFTY option contract: tradingsymbol, strike, expiry, type.

        The dump is a few megabytes and changes once a day when new contracts
        list, so it is cached to disk and reloaded on date change. Downloading
        it per bar would burn the rate limit for no benefit.
        """
        today = date.today()
        if not refresh and self._instruments is not None and self._loaded_on == today:
            return self._instruments

        df = None
        if not refresh and self.cache_file.exists():
            try:
                cached = pd.read_csv(self.cache_file, parse_dates=["expiry"])
                stamp = datetime.fromtimestamp(self.cache_file.stat().st_mtime).date()
                if stamp == today and not cached.empty:
                    df = cached
            except Exception:
                df = None                      # corrupt cache, just refetch

        if df is None:
            df = self._download()
            try:
                df.to_csv(self.cache_file, index=False)
            except OSError:
                pass                           # cache is an optimisation, not a requirement

        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        self._instruments = df
        self._loaded_on = today
        return df

    def _download(self) -> pd.DataFrame:
        kite = self.session.client()
        raw = kite.instruments("NFO")
        df = pd.DataFrame(raw)
        if df.empty:
            raise NotAuthenticated("Kite returned an empty NFO instrument dump")

        df = df[(df["name"] == self.cfg.instrument.symbol)
                & (df["segment"] == "NFO-OPT")
                & (df["instrument_type"].isin(["CE", "PE"]))]
        if df.empty:
            raise NotAuthenticated(
                f"no {self.cfg.instrument.symbol} options in the NFO dump - "
                f"check InstrumentConfig.symbol"
            )
        return df[["tradingsymbol", "instrument_token", "strike",
                   "expiry", "instrument_type", "lot_size"]].copy()

    # ---------------- expiries ----------------

    def expiries(self) -> list[date]:
        """Every listed expiry, ascending. The real ones, not a weekday guess."""
        df = self.instruments()
        return sorted({e for e in df["expiry"]})

    def nearest_expiry(self, today: date | None = None) -> Optional[date]:
        today = today or date.today()
        future = [e for e in self.expiries() if e >= today]
        return future[0] if future else None

    def lot_size(self, expiry: date | None = None) -> Optional[int]:
        """
        The exchange's lot size, straight from the dump.

        `InstrumentConfig.lot_size` is a hardcoded 65 with a comment warning
        that it has been revised three times since Nov 2024. When the dump is
        available it is the authority; the config value is the offline fallback.
        """
        df = self.instruments()
        if expiry is not None:
            df = df[df["expiry"] == expiry]
        if df.empty:
            return None
        return int(df["lot_size"].mode().iloc[0])

    # ---------------- quotes ----------------

    def get_chain(self, spot: float, option_type: str,
                  today: date | None = None,
                  expiry: date | None = None,
                  strikes_each_side: int = 12) -> tuple[list[OptionQuote], date]:
        """
        Live quotes for strikes around ATM, as OptionQuote objects that
        `RiskEngine.select_strike()` can consume unchanged.

        Strikes whose premium cannot be inverted to an implied volatility are
        DROPPED, not defaulted. An un-invertible premium means the quote is
        stale, crossed, or below intrinsic - all of which are reasons not to
        select that strike, and none of which are improved by substituting a
        guessed volatility.
        """
        today = today or date.today()
        expiry = expiry or self.nearest_expiry(today)
        if expiry is None:
            raise NotAuthenticated("no future expiry found in the instrument dump")

        df = self.instruments()
        df = df[(df["expiry"] == expiry) & (df["instrument_type"] == option_type)]
        if df.empty:
            raise NotAuthenticated(f"no {option_type} contracts for expiry {expiry}")

        step = self.cfg.instrument.strike_step
        atm = round(spot / step) * step
        lo = atm - strikes_each_side * step
        hi = atm + strikes_each_side * step
        df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
        if df.empty:
            raise NotAuthenticated(
                f"no {option_type} strikes between {lo} and {hi} for {expiry}")

        symbols = [f"NFO:{s}" for s in df["tradingsymbol"].tolist()[:MAX_QUOTE_BATCH]]
        kite = self.session.client()
        try:
            quoted = kite.quote(symbols)
        except Exception as e:
            raise NotAuthenticated(f"Kite quote() failed: {e}") from e

        t_years = time_to_expiry_years(today, expiry)
        if t_years <= 0:
            # Expiry day: hours, not days. Black-Scholes at t=0 returns
            # intrinsic only, which would hand every OTM strike a delta of 0.
            t_years = 0.5 / 365.0

        rows = self._build_rows(df, quoted, spot, option_type, expiry, t_years)
        quotes = [
            OptionQuote(
                strike=r.strike,
                option_type=r.option_type,
                premium=round(r.mid, 2),
                delta=round(r.delta, 4),
                bid=round(r.bid, 2),
                ask=round(r.ask, 2),
                open_interest=r.open_interest,
                iv=r.iv,
                volume=r.volume,
            )
            for r in rows if r.delta is not None
        ]
        return quotes, expiry

    def _build_rows(self, df: pd.DataFrame, quoted: dict, spot: float,
                    option_type: str, expiry: date, t_years: float) -> list[ChainRow]:
        r = self.cfg.backtest.risk_free_rate
        rows: list[ChainRow] = []

        for _, inst in df.iterrows():
            key = f"NFO:{inst['tradingsymbol']}"
            q = quoted.get(key)
            if not q:
                continue

            depth = q.get("depth") or {}
            buys, sells = depth.get("buy") or [], depth.get("sell") or []
            bid = float(buys[0]["price"]) if buys else 0.0
            ask = float(sells[0]["price"]) if sells else 0.0
            last = float(q.get("last_price") or 0.0)

            row = ChainRow(
                tradingsymbol=str(inst["tradingsymbol"]),
                strike=int(inst["strike"]),
                option_type=option_type,
                expiry=expiry,
                last_price=last,
                bid=bid,
                ask=ask,
                open_interest=int(q.get("oi") or 0),
                volume=int(q.get("volume") or 0),
                iv=None,
                delta=None,
            )

            row.iv = implied_vol(row.mid, spot, row.strike, t_years, r, option_type)
            if row.iv is not None:
                row.delta = bs_delta(spot, row.strike, t_years, row.iv, r, option_type)
            rows.append(row)

        return rows

    def rows_for_brief(self, spot: float, option_type: str,
                       today: date | None = None,
                       strikes_each_side: int = 10) -> tuple[list[ChainRow], date]:
        """
        The same data as get_chain() but WITHOUT dropping un-invertible rows,
        because the daily brief needs to show you the strikes that failed and
        why, not silently omit them.
        """
        today = today or date.today()
        expiry = self.nearest_expiry(today)
        if expiry is None:
            raise NotAuthenticated("no future expiry found in the instrument dump")

        df = self.instruments()
        df = df[(df["expiry"] == expiry) & (df["instrument_type"] == option_type)]
        step = self.cfg.instrument.strike_step
        atm = round(spot / step) * step
        df = df[(df["strike"] >= atm - strikes_each_side * step)
                & (df["strike"] <= atm + strikes_each_side * step)]

        symbols = [f"NFO:{s}" for s in df["tradingsymbol"].tolist()[:MAX_QUOTE_BATCH]]
        kite = self.session.client()
        quoted = kite.quote(symbols)

        t_years = max(time_to_expiry_years(today, expiry), 0.5 / 365.0)
        rows = self._build_rows(df, quoted, spot, option_type, expiry, t_years)
        return sorted(rows, key=lambda x: x.strike), expiry
