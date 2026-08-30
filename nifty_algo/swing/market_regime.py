"""
Is the market itself worth being long in today?

NOT [regime.py](../regime.py). That one classifies the intraday character of a
single session - opening-range width, VWAP slope, gap size - to decide which
option strategy may speak. This one asks a slower and cruder question about
the index itself: is it above its own long moving average. Different horizon,
different inputs, different book, so a different file - the same reasoning
that keeps `costs.py` and `costs_equity.py` apart.

WHY IT EXISTS. The swing book is long-only and 74% of its trades are
breakouts. A long-only breakout book has no defence at all against a market
that trends down for eight months: it fires into every failed rally and pays
friction on each one. The option book has had `regime.py` gating it from the
start, and that module's own docstring calls the idea the highest-leverage
filter in the system. The swing book shipped without one.

DEFAULT OFF. `regime_ma_days = 0` disables this entirely, which is the
behaviour every result recorded before it existed was produced under. Turning
it on is an experiment with a number attached, not a silent improvement -
`swing/experiment.py` measures it out-of-sample.

Pure functions, no I/O, no config import beyond the value passed in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class RegimeReading:
    """Whether new longs are allowed, and the arithmetic behind the answer."""
    ok: bool
    ma_days: int
    close: Optional[float] = None
    moving_average: Optional[float] = None
    reason: str = ""

    @property
    def enabled(self) -> bool:
        return self.ma_days > 0


#: Returned when the filter is switched off. `ok` is True because "no filter"
#: must mean "every day passes" - a disabled gate that blocks would be a
#: silent behaviour change for every existing result.
DISABLED = RegimeReading(ok=True, ma_days=0, reason="regime filter is off")


def benchmark_state(benchmark: Optional[pd.DataFrame], ma_days: int,
                    as_of=None) -> RegimeReading:
    """
    Read the benchmark's trend as of `as_of` (default: its last bar).

    Only bars up to and including `as_of` are used, so this is safe to call
    inside the walk-forward loop - the same discipline `setup.detect` follows.

    A MISSING OR SHORT BENCHMARK BLOCKS. There is no honest way to say "the
    market is fine" without the data that would show it was not, and every
    other unavailable input in this book fails closed too - `fx.py` stands the
    market down, the halal screen refuses to pass what it cannot verify. The
    alternative is a filter that quietly stops filtering on exactly the days
    the feed breaks.
    """
    if ma_days <= 0:
        return DISABLED

    if benchmark is None or "close" not in getattr(benchmark, "columns", []):
        return RegimeReading(
            ok=False, ma_days=ma_days,
            reason="no benchmark bars, so the market trend cannot be read")

    frame = benchmark if as_of is None else benchmark.loc[benchmark.index <= as_of]
    if len(frame) < ma_days:
        return RegimeReading(
            ok=False, ma_days=ma_days,
            reason=(f"only {len(frame)} benchmark bars, need {ma_days} for the "
                    f"{ma_days}-day average"))

    close = float(frame["close"].iloc[-1])
    ma = float(frame["close"].rolling(ma_days).mean().iloc[-1])
    if pd.isna(ma):
        return RegimeReading(
            ok=False, ma_days=ma_days,
            reason=f"the {ma_days}-day average is not computable yet")

    ok = close > ma
    gap = (close / ma - 1.0) * 100.0
    reason = (
        f"benchmark {close:,.1f} is {abs(gap):.1f}% "
        f"{'above' if ok else 'below'} its {ma_days}-day average ({ma:,.1f})"
    )
    if not ok:
        reason += " - a long-only book stands aside"
    return RegimeReading(ok=ok, ma_days=ma_days, close=close,
                         moving_average=ma, reason=reason)
