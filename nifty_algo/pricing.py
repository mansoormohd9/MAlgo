"""
Black-Scholes for European index options.

Two jobs, both auxiliary - never a signal source:

  1. Synthesise an option chain when no broker chain is available, so
     `RiskEngine.approve()` always has OptionQuote objects to select from
     and the alert can name a strike.
  2. Price the backtester's optional SYNTHETIC_PREMIUM mode.

READ THIS BEFORE TRUSTING ANY NUMBER OUT OF HERE.

A synthesised premium assumes a single flat IV, no skew, no smile, no bid-ask,
and infinite depth. Real index options have all four. Per the README's own
Phase 3 note, backtest results built on synthetic premiums are optimistic by
15-25%. The synthetic chain also fabricates OI and a tight spread, which means
the liquidity gates in risk.py cannot reject it the way they would reject a
real illiquid contract.

Use this to develop and to reason about strike selection. Do not use it to
decide that a strategy is profitable.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from math import log, sqrt, exp

from scipy.stats import norm

from .config import Config, DEFAULT
from .risk import OptionQuote

TRADING_DAYS_PER_YEAR = 252.0


def _d1_d2(spot: float, strike: float, t_years: float,
           iv: float, r: float) -> tuple[float, float]:
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0
    vol_t = iv * sqrt(t_years)
    d1 = (log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / vol_t
    return d1, d1 - vol_t


def bs_price(spot: float, strike: float, t_years: float, iv: float,
             r: float, option_type: str) -> float:
    """Black-Scholes premium. `option_type` is "CE" or "PE"."""
    if t_years <= 0:
        intrinsic = (spot - strike) if option_type == "CE" else (strike - spot)
        return max(intrinsic, 0.0)
    d1, d2 = _d1_d2(spot, strike, t_years, iv, r)
    disc = exp(-r * t_years)
    if option_type == "CE":
        return spot * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    return strike * disc * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_delta(spot: float, strike: float, t_years: float, iv: float,
             r: float, option_type: str) -> float:
    """
    Delta. Returned POSITIVE for both CE and PE.

    risk.py compares against `abs(q.delta)`, and the whole system expresses a
    bearish view by buying a PE rather than by carrying a negative sign. Keeping
    delta positive here means the strike-selection maths reads the same for both
    sides.
    """
    if t_years <= 0:
        itm = (spot > strike) if option_type == "CE" else (spot < strike)
        return 1.0 if itm else 0.0
    d1, _ = _d1_d2(spot, strike, t_years, iv, r)
    return norm.cdf(d1) if option_type == "CE" else norm.cdf(-d1)


def time_to_expiry_years(today: date, expiry: date) -> float:
    """Calendar-day count over 365. Intraday theta is not modelled here."""
    return max((expiry - today).days, 0) / 365.0


# Volatility bounds for the IV solver. 1% and 500% annualised: wide enough for
# anything a real index option prints, including expiry-day gamma silliness,
# narrow enough that a nonsense premium fails to bracket rather than converging
# on a nonsense answer.
MIN_IV = 0.01
MAX_IV = 5.00


def implied_vol(price: float, spot: float, strike: float, t_years: float,
                r: float, option_type: str) -> float | None:
    """
    Back out implied volatility from a REAL market premium.

    This is the piece that lets a live chain drive strike selection. A broker
    gives you a premium, not a delta - and `RiskEngine.select_strike()` selects
    on delta, because delta is what converts the underlying stop into a rupee
    loss. So: premium -> IV -> delta, per strike, from live quotes.

    Doing it per strike is also what makes the SKEW visible. `synthetic_chain`
    assumes one flat IV across every strike; real index options are priced with
    OTM puts materially bid over equidistant calls. Under a flat-IV assumption
    the PE deltas are simply wrong, and the wrong delta buys the wrong strike.

    Returns None when the premium cannot be bracketed - typically a stale or
    crossed quote, or a premium below intrinsic. None means "do not trust this
    quote", and callers should drop the strike rather than guess a fallback IV.
    """
    if price <= 0 or spot <= 0 or strike <= 0 or t_years <= 0:
        return None

    # No volatility can price an option below its own intrinsic value; such a
    # quote is stale or crossed, not cheap.
    intrinsic = max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
    if price < intrinsic - 0.01:
        return None

    def diff(iv: float) -> float:
        return bs_price(spot, strike, t_years, iv, r, option_type) - price

    lo, hi = diff(MIN_IV), diff(MAX_IV)
    if lo > 0 or hi < 0:
        # Price sits outside what the model can produce at any volatility.
        return None

    from scipy.optimize import brentq
    try:
        return float(brentq(diff, MIN_IV, MAX_IV, xtol=1e-6, maxiter=100))
    except (ValueError, RuntimeError):
        return None


@dataclass
class SyntheticChainSpec:
    """How wide a chain to fabricate around spot."""
    strikes_each_side: int = 12
    assumed_iv: float = 0.14
    risk_free_rate: float = 0.065
    fake_open_interest: int = 500_000
    fake_spread_pct: float = 0.004      # just inside risk.py's 0.5% gate


def synthetic_chain(spot: float, t_years: float, option_type: str,
                    cfg: Config = DEFAULT,
                    spec: SyntheticChainSpec | None = None) -> list[OptionQuote]:
    """
    Build a plausible chain around spot.

    Every quote produced here is FICTION. The engine tags alerts built from a
    synthetic chain so the UI can render them differently, and the journal
    records which chain source was used.
    """
    spec = spec or SyntheticChainSpec()
    step = cfg.instrument.strike_step
    atm = int(round(spot / step) * step)

    quotes: list[OptionQuote] = []
    for i in range(-spec.strikes_each_side, spec.strikes_each_side + 1):
        strike = atm + i * step
        if strike <= 0:
            continue
        premium = bs_price(spot, strike, t_years, spec.assumed_iv,
                           spec.risk_free_rate, option_type)
        if premium < 0.05:
            continue
        delta = bs_delta(spot, strike, t_years, spec.assumed_iv,
                         spec.risk_free_rate, option_type)
        half_spread = premium * spec.fake_spread_pct / 2.0
        quotes.append(OptionQuote(
            strike=strike,
            option_type=option_type,
            premium=round(premium, 2),
            delta=round(delta, 4),
            bid=round(max(premium - half_spread, 0.05), 2),
            ask=round(premium + half_spread, 2),
            open_interest=spec.fake_open_interest,
            iv=spec.assumed_iv,       # flat by construction - there is no skew here
        ))
    return quotes
