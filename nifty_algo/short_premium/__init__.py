"""
The FOURTH book: intraday NIFTY option SELLING.

Every other book here is long an instrument. This one is short premium, and
that is not a direction - it is a different payoff, a different constraint and
a different definition of R. A flag on the option book would have been wrong
in all three ways:

  PAYOFF. A buyer's profit is unbounded and their loss is capped at the debit.
  A seller's profit is capped at the credit and their loss is not bounded at
  all. `TradeManagementConfig.partial_exit_at_r` is 2.0; a short at a 2x-credit
  stop can never reach +2R, so left alone the partial rung never fires, the
  runner never arms and the trail is dead code - silently.

  CONSTRAINT. `RiskEngine.approve()` gates on `premium x lot x lots <=
  free_capital` (risk.py:348) because a buyer pays a debit. A seller RECEIVES
  the premium and is constrained by SPAN + exposure margin, which appears
  nowhere in the option path. Sizing off the debit would approve a position
  the account cannot carry, and it would look entirely ordinary.

  SELECTION. `risk.select_strike()` returns the HIGHEST delta that fits the
  budget - "best directional capture per rupee of theta paid" (risk.py:223).
  A seller wants the lowest delta that still pays a credit worth collecting.
  With `min_delta=0.20` and `min_premium=30` the buying engine refuses every
  strike this book trades, twice over.

WHAT IT SHARES, AND WHY THAT IS THE POINT

`positions.ExitLadder` unchanged - it is pure arithmetic in R with no prices
in it, and it already accepts a settings override, which is how the swing book
gets its own rungs. `ShortPremiumConfig.ladder_settings()` hands it a real
`TradeManagementConfig`, so there is still exactly one exit state machine to
be wrong. `signals.py` unchanged. `governor.py` unchanged, instantiated
separately so this book's day is its own. `CapitalConfig` unchanged except for
a fifth POOL and a generalised entry count - one formula, still, in
`risk_pct_for`. `journal.py` unchanged.

WHAT IS DELIBERATELY NOT SHARED

Not `Signal`: five fields, and `option_type` hardcoded as a function of
`direction` ("CE for long, PE for short", strategy.py:35). It cannot express a
side, a leg or a credit. See `signal.ShortSignal`.

Not `risk.RiskEngine`: see SELECTION and CONSTRAINT above.

Not `positions.ManagedPosition`: its `r_of`, `pnl_for` and `premium_of`
(positions.py:308-322) are sign-locked long. A winning short reports negative
R there and stops out on the first bar. See `position.ShortPosition`.

Not `costs.py` as-is: a seller pays STT on the ENTRY leg and stamp duty on the
exit, which is the reverse of what `entry_friction`/`exit_friction` assume.
See `costs_sell.py`, which reuses the same rate table rather than restating it.

Not `backtest.py`'s SYNTHETIC_PREMIUM, ever. It fixes IV for a trade's life
and assumes zero skew (pricing.py:175). For a buyer that is a caveat; for a
seller it prices the entire edge AND the entire risk at zero. There is no
historical option chain at any price, so the only honest route to a backtest
is `recorder.py` - which starts accumulating the day it is first run, and
answers nothing before then.
"""
