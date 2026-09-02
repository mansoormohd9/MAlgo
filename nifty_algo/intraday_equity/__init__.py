"""
The THIRD book: intraday cash equity on the Nifty 100.

Neither of the other two describes it, which is why it is a package rather
than a flag on one of them:

  the option book (`nifty_algo/`, outside `swing/`) is intraday, but it
  trades ONE instrument and its whole risk engine is strike selection;

  the swing book (`nifty_algo/swing/`) is cash equity on a universe, but it
  HOLDS FOR DAYS, which is what makes a resting GTT the right stop and a
  daily bar the right resolution.

This one sweeps a universe every 5-minute bar and is flat by 15:10.

WHAT IT SHARES, AND WHY THAT IS THE POINT

`signals.py` unchanged - every indicator there is pure `bars in, values out`
with no bar-size assumption, which `swing/setup.py` already proved by running
them on daily bars. `positions.ExitLadder` unchanged, so there is exactly one
implementation of the exit state machine to be wrong. `governor.py` unchanged,
so the day rules are the same day rules. `CapitalConfig` unchanged, so the
per-trade risk formula stays in one place - this book adds a fourth POOL, not
a fourth formula. And `strategies/registry.py` unchanged: the eight intraday
setups written for the index are reused verbatim, because a 5-minute stock
bar is a 5-minute bar.

THE ONE THING THAT GENUINELY CHANGES WHEN YOU MOVE TO STOCKS

The index reports zero volume, so `volume_surge`, `vwap` and
`underlying_liquidity_ok` currently fall back to range and TWAP proxies.
Stocks have real volume, so `has_traded_volume()` is True and all three take
their real branch. That is a BEHAVIOUR CHANGE, not a port - the strategies
were calibrated against the degraded version - and every result this package
prints has to say so.

WHAT IS DELIBERATELY NOT SHARED

Not `engine.TradingEngine`: it is single-symbol by construction (one feed,
one `EngineState.bars`, one `Context`), and a hundred-symbol book cannot be
expressed by it. `runner.py` reuses the PIECES instead.

Not `swing/costs_equity.py`: that is delivery, and intraday MIS differs on
STT, brokerage and the DP charge by enough to change a verdict. See
`costs_intraday.py`.

Not `swing/prices.py`: it is `interval="1d"` by explicit argument.

Not the GTT machinery, and not `ddpi_active` - an intraday sell never leaves
the demat account, so no CDSL TPIN is involved and the swing book's
"looks armed and is not" hazard cannot occur here.
"""
