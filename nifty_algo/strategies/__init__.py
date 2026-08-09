"""
Strategy library.

Every strategy in here subclasses the `Strategy` ABC from ../strategy.py
unchanged: it receives a Context and returns a Signal. None of them touch a
broker, a clock, a file, or the risk engine. That separation is what lets the
backtester and the live runner import identical code.
"""
from .registry import all_strategies, build_enabled, STRATEGY_CLASSES  # noqa: F401
