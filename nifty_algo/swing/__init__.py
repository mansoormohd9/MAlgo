"""
Daily Nifty 100 swing scanner - a second, separate book.

Everything else in this package trades NIFTY index options intraday and is
flat by 15:10. This subsystem does something different: once a day it sweeps
the Nifty 100 constituents, applies a halal screen, and produces at most three
cash-equity LONG swing candidates with a full ticket - entry, stop, target,
quantity - held for days rather than hours.

WHAT IT SHARES WITH THE REST OF THE SYSTEM, DELIBERATELY:

  nifty_algo.signals   every indicator. They are pure `bars in, values out`
                       functions and work unchanged on daily bars.
  CapitalConfig        risk per trade and the 2:1 reward:risk floor. NOT
                       re-declared here - two books sizing off two copies of
                       the same governor is how they drift apart.
  Journal              the same append-only record.

WHAT IT DOES NOT SHARE: the engine, the risk engine's option-strike machinery,
the session governors. Those encode "intraday, three entries, flat by close",
which is not what this is.

Like `nifty_algo.ui`, nothing in here renders. The page is a viewer.
"""
