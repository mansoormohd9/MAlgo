"""
Institutional-style research briefings, as deterministic fact packs.

The arithmetic lives here; the prose does not. See `base.py` for the contract
every pack keeps - most importantly that a fact this build could not establish
is carried as `available=False` WITH A REASON, and never as a blank or a zero.
"""
from .base import Fact, FactPack, Section

__all__ = ["Fact", "FactPack", "Section"]
