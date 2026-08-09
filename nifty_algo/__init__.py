"""
Nifty intraday option-buying system.

Layering, outermost to innermost:

    data/ ──► signals ──► strategies/ ──► risk ──► alerts/ ──► journal
                                          ▲
                                     hard gate

`risk.RiskEngine.approve()` is the only path from a strategy signal to an
actionable alert. Nothing in strategies/ may bypass it.
"""

__version__ = "0.2.0"
