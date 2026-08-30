# Superseded: measured before the settlement fix

These two sweeps were run when `swing/backtest.run()` DELETED any position
still open when its window closed. What is open at a boundary is not a random
sample - a loser hits its stop in days, a winner trails for weeks - so the
harness removed winners preferentially and every variant read about 0.2R worse
than it was. The same ~261 baseline trades scored -67.8R tiled against -21.6R
run continuously.

`run(..., settle_days=60)` now carries a window's trades to their natural exit
with entries switched off. Kept rather than deleted because the gap between
these numbers and the ones beside them is the measurement of the bias.
