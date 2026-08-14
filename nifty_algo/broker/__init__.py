"""
Broker integration.

Everything in this package can touch a real account, which is why it is
separated from `nifty_algo.data` (read-only market data) rather than living
alongside it. The split is deliberate: importing a feed can never place an
order, because the feed module has no access to the order module.

Order placement is guarded by `BrokerConfig.dry_run`, which defaults to True.
"""
from .kite_auth import KiteSession, NotAuthenticated

__all__ = ["KiteSession", "NotAuthenticated"]
