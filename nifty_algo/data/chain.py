"""
Option chain sourcing.

`RiskEngine.approve()` needs a list of OptionQuote to select a strike from.
Without a chain there is no strike, no premium, no stop, and therefore no
actionable alert - only a direction, which is not a trade.

Two sources:

  BROKER    real quotes. Liquidity gates in risk.py do real work here:
            spread, OI, and premium bounds can genuinely reject a contract.

  SYNTHETIC Black-Scholes premiums around spot. Available always, needs no
            credentials, and is how the whole system runs on day one.

The synthetic chain fabricates open interest and a tight spread, which means
`risk.py`'s liquidity gates CANNOT reject it - they will pass every time
because the inputs were invented to pass. A synthetic-chain alert tells you
the correct strike given the maths; it tells you nothing about whether that
contract is actually tradeable at that price.

Every quote carries its source so the alert, the UI, and the journal can all
say which one produced the number.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta

from ..config import Config, DEFAULT
from ..pricing import synthetic_chain, SyntheticChainSpec, time_to_expiry_years
from ..risk import OptionQuote


@dataclass
class ChainResult:
    quotes: list[OptionQuote]
    source: str            # "synthetic" | "broker"
    expiry: date
    is_synthetic: bool
    note: str


def next_weekly_expiry(today: date | None = None) -> date:
    """
    Next Nifty weekly expiry.

    Currently Tuesday, changed from Thursday in 2025. VERIFY THIS ON THE NSE
    CIRCULAR BEFORE RELYING ON IT - the expiry day has moved twice recently
    and a wrong expiry means a wrong time-to-expiry means a wrong premium and
    a wrong delta, which corrupts strike selection silently.
    """
    today = today or date.today()
    expiry_weekday = 1                      # Monday=0, so 1 = Tuesday
    days_ahead = (expiry_weekday - today.weekday()) % 7
    return today + timedelta(days=days_ahead)


class ChainProvider:
    """Falls back to synthetic whenever a broker chain is unavailable."""

    def __init__(self, cfg: Config = DEFAULT, prefer_broker: bool = True,
                 broker_chain=None):
        self.cfg = cfg
        self.prefer_broker = prefer_broker
        # Injected rather than constructed here so the engine can run without
        # kiteconnect installed, and so tests can supply a recorded chain.
        self._broker_chain = broker_chain
        self._broker_failure: str = ""

    def get_chain(self, spot: float, option_type: str,
                  today: date | None = None) -> ChainResult:
        today = today or date.today()
        expiry = self.resolve_expiry(today)
        t_years = time_to_expiry_years(today, expiry)

        # Expiry-day options have ~zero time value; Black-Scholes with t=0
        # returns intrinsic only, which would produce nonsense strikes. Give
        # it a small floor representing the hours left in the session.
        if t_years <= 0:
            t_years = 0.5 / 365.0

        if self.prefer_broker:
            broker = self._try_broker_chain(spot, option_type, expiry)
            if broker is not None:
                return broker

        spec = SyntheticChainSpec(assumed_iv=self.cfg.backtest.assumed_iv,
                                  risk_free_rate=self.cfg.backtest.risk_free_rate)
        quotes = synthetic_chain(spot, t_years, option_type, self.cfg, spec)
        why = f" Broker chain unavailable: {self._broker_failure}." \
            if self._broker_failure else ""
        return ChainResult(
            quotes=quotes,
            source="synthetic",
            expiry=expiry,
            is_synthetic=True,
            note=(f"SYNTHETIC chain: Black-Scholes at {spec.assumed_iv:.0%} flat IV, "
                  f"fabricated OI and spread. Strike maths is right; tradeability "
                  f"is unverified. Confirm the live quote before acting.{why}"),
        )

    def resolve_expiry(self, today: date) -> date:
        """
        The real expiry when a broker chain is attached, the weekday guess
        otherwise.

        `next_weekly_expiry()` assumes Tuesday and its own docstring flags that
        the day has moved twice recently. The instrument dump states the actual
        expiry of every listed contract, so prefer it whenever it is there.
        """
        if self._broker_chain is not None:
            try:
                real = self._broker_chain.nearest_expiry(today)
                if real is not None:
                    return real
            except Exception:
                pass                     # fall through to the guess
        return next_weekly_expiry(today)

    def _try_broker_chain(self, spot: float, option_type: str,
                          expiry: date) -> ChainResult | None:
        """
        Real quotes, when a broker chain has been attached.

        Any failure returns None so the engine keeps running on synthetic
        quotes rather than going silent - but the reason is recorded and
        printed in the synthetic chain's note, because a system that quietly
        downgrades from real data to fabricated data without telling you is
        worse than one that stops.
        """
        if self._broker_chain is None:
            return None
        try:
            quotes, real_expiry = self._broker_chain.get_chain(
                spot, option_type, expiry=expiry)
        except Exception as e:
            self._broker_failure = f"{type(e).__name__}: {e}"
            return None

        if not quotes:
            self._broker_failure = "broker returned no usable quotes"
            return None

        self._broker_failure = ""
        return ChainResult(
            quotes=quotes,
            source="broker",
            expiry=real_expiry,
            is_synthetic=False,
            note="",
        )
