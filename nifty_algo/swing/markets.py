"""
The markets this book can be pointed at.

WHY THIS MODULE EXISTS. The swing scanner was written for the Nifty 100 and
four of its numbers were secretly India-only: the universe file, the benchmark
ticker, the price floor and the turnover floor. Everything else in `swing/` -
the EMAs, the ATR multiples, the score weights, the whole of `signals.py` - is
already currency-blind, because a daily bar is a daily bar. So the entire
difference between scanning Mumbai, New York and London is the contents of one
dataclass.

Registry, not a pile of if-statements: same reason `data/factory.py` is the
only place a feed is constructed and `strategies/registry.py` is the only place
a strategy is enumerated. A market that is scannable but has no benchmark, or
tunable in the UI but absent from the CLI, is the bug this shape prevents.

THREE THINGS HERE ARE LOAD-BEARING AND EASY TO MISS:

`price_divisor` - the London Stock Exchange quotes most of its shares in PENCE
and yfinance passes that through untouched. SHEL.L comes back as 2500 meaning
GBP 25.00. Unhandled, every UK price floor, turnover figure, stop distance and
position size is wrong by 100x, and none of it looks wrong.

`capital_pool` - money remitted under LRS is a different pot from the domestic
account. Sizing a US trade off the Indian balance would claim capital that is
not in that broker.

`taxonomy` - the halal activity screen matches on classification labels, and
NSE's vocabulary ("Non Banking Financial Company (NBFC)") shares nothing with
Yahoo's GICS-derived one ("Credit Services"). One table cannot serve both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Cash-account markets, keyed the way the CLI and the UI both spell them.
INDIA = "india"
US = "us"
UK = "uk"

#: Which pot of money sizes a trade. `home` is the domestic account; `foreign`
#: is whatever you have remitted under the Liberalised Remittance Scheme.
POOL_HOME = "home"
POOL_FOREIGN = "foreign"

#: Which classification vocabulary the halal activity screen should match on.
TAXONOMY_NSE = "nse"
TAXONOMY_GICS = "gics"


class UnknownMarket(KeyError):
    """Asked for a market that is not registered. Never guess a default."""


@dataclass
class Market:
    """
    One exchange's identity and its tradeability thresholds.

    Mutable rather than frozen because these are tunables and invariant #3
    says tunables live in config - `SwingConfig.markets` holds the instances,
    so a test or the Settings page can move a floor without monkeypatching a
    module global.
    """
    key: str
    label: str
    universe_csv: str
    benchmark_ticker: str
    benchmark_label: str

    # --- money ---
    currency: str                     # ISO code: INR | USD | GBP
    symbol: str                       # what to print in front of a number
    fx_pair: str | None               # yfinance pair quoting INR per unit;
                                      # None for the home currency itself
    capital_pool: str = POOL_HOME
    allow_fractional: bool = False    # IBKR fills fractional US shares; NSE
                                      # and the LSE order book do not

    # --- quote conventions ---
    price_divisor: float = 1.0        # 100.0 where the exchange quotes minor
                                      # units (LSE pence). Applied ONCE, at
                                      # ingest, so nothing downstream repeats it.

    # --- tradeability floors, in this market's own currency ---
    min_price: float = 0.0
    min_avg_turnover: float = 0.0
    turnover_unit: str = ""           # how to label it: "cr", "m"
    turnover_divisor: float = 1.0     # what to divide traded value by

    # --- screening ---
    taxonomy: str = TAXONOMY_GICS

    @property
    def is_home(self) -> bool:
        return self.capital_pool == POOL_HOME

    @property
    def cache_suffix(self) -> str:
        """Per-market cache filenames. See `prices.py` for why this matters."""
        return self.key

    def money(self, amount: float, dp: int = 0) -> str:
        return f"{self.symbol}{amount:,.{dp}f}"

    def turnover(self, amount: float) -> str:
        return f"{self.symbol}{amount:,.1f} {self.turnover_unit}".strip()

    def qualified(self, symbol: str) -> str:
        """
        `{market}:{SYMBOL}` - the key for anything cached or journalled.

        Bare symbols collide across exchanges, and a collision here silently
        screens one company with another's balance sheet.
        """
        return f"{self.key}:{symbol.upper()}"


def default_markets() -> dict[str, Market]:
    """
    The registry. Returned fresh so each `Config` owns its own instances and
    a test that moves a threshold cannot leak into the next test.
    """
    return {
        INDIA: Market(
            key=INDIA, label="India · Nifty 100",
            universe_csv="data/nifty100.csv",
            benchmark_ticker="^NSEI", benchmark_label="Nifty 50",
            currency="INR", symbol="\u20b9", fx_pair=None,
            capital_pool=POOL_HOME, allow_fractional=False,
            price_divisor=1.0,
            min_price=50.0,
            # 20-day average traded value. Rupees crore, as every Indian
            # screen quotes it.
            min_avg_turnover=25.0, turnover_unit="cr", turnover_divisor=1e7,
            taxonomy=TAXONOMY_NSE,
        ),
        US: Market(
            key=US, label="US · SPUS + HLAL constituents",
            universe_csv="data/us_halal.csv",
            benchmark_ticker="^GSPC", benchmark_label="S&P 500",
            currency="USD", symbol="$", fx_pair="USDINR=X",
            capital_pool=POOL_FOREIGN, allow_fractional=True,
            price_divisor=1.0,
            # A US large cap under $5 has usually been left for dead; the
            # universe is S&P 500 constituents, so this floor rarely bites and
            # exists to catch a delisting in progress.
            min_price=5.0,
            min_avg_turnover=10.0, turnover_unit="m", turnover_divisor=1e6,
            taxonomy=TAXONOMY_GICS,
        ),
        UK: Market(
            key=UK, label="UK · FTSE 100",
            universe_csv="data/ftse100.csv",
            benchmark_ticker="^FTSE", benchmark_label="FTSE 100",
            currency="GBP", symbol="\u00a3", fx_pair="GBPINR=X",
            capital_pool=POOL_FOREIGN, allow_fractional=False,
            # THE PENCE TRAP. See the module docstring.
            price_divisor=100.0,
            min_price=1.0,
            min_avg_turnover=3.0, turnover_unit="m", turnover_divisor=1e6,
            taxonomy=TAXONOMY_GICS,
        ),
    }


def get(cfg, key: str) -> Market:
    """
    Look one up, or say which ones exist.

    Raises rather than falling back to India: a typo that silently scans the
    wrong exchange would produce a page full of plausible, wrong tickets.
    """
    markets = cfg.swing.markets
    try:
        return markets[key]
    except KeyError:
        raise UnknownMarket(
            f"unknown market {key!r} - registered markets are "
            f"{', '.join(sorted(markets))}"
        ) from None


def keys(cfg) -> list[str]:
    """Registered market keys, in a stable display order."""
    order = {INDIA: 0, US: 1, UK: 2}
    return sorted(cfg.swing.markets, key=lambda k: (order.get(k, 99), k))
