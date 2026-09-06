"""
Which slice of the NSE the sleeve is allowed to rank inside.

The shipped book ranks all ~2,400 listed names, and on real data 17 of its top
20 sit outside the Nifty 500 - one of them trading Rs 2.5 cr a day. Restricting
the universe is therefore a real lever, and this module resolves the three ways
of expressing it into one shape: `day -> set of symbols, or None`.

WHY A CALLABLE AND NOT A SET. `size500` is a different set on every rebalance,
because it ranks by a size series that moves. Handing `run()` one static set
computed from the whole history would be look-ahead of a second kind - the same
trap `eligible_at`'s `< day` filter exists to prevent, arriving through the
back door.

THE LOOK-AHEAD IN `nifty500`, STATED PLAINLY. `data/nifty500.csv` is TODAY'S
membership. Applying it to 2016 means the book may only ever hold companies
that GREW INTO the index - selection on the outcome, running hard in the
flattering direction, and worse than the plain survivorship bias
`factor/universe.py` already documents. It is the literal question the mandate
asks, so it is measurable here; it is not a number to plan against on its own.

WHICH IS WHY `size500` EXISTS. The Nifty 500 selects on free-float market cap,
so the honest control ranks by size directly: today's share count (implied by
`Fundamentals.market_cap` over today's close) applied to each date's close. Its
error is share ISSUANCE and buybacks - mechanical, bounded, and unrelated to
whether a company later joined an index. `nifty500` minus `size500` is an
estimate of the look-ahead, and reporting the pair is the point of having both.

THE CONTROL DEGRADES THE FURTHER BACK IT REACHES, in a known direction. A 2008
rebalance is priced with a share count measured in 2026 - eighteen years of
issuance later - so a company that has diluted heavily has its PAST size
overstated, and growth names dilute most. That pushes exactly those names into
the early-window top 500. So `size500` is a better control than `nifty500` and
is not a clean one: over a long window it errs the same way, far less. Read the
look-ahead estimate as a LOWER bound.

TURNOVER IS NOT A SUBSTITUTE FOR EITHER. CUPID passes `band="liquid"` - its
turnover blew out WITH the momentum, Rs 774 cr a day on a micro cap - so a
liquidity stand-in readmits exactly the names a size mandate means to exclude.
Size and traded value are different facts and this module does not conflate
them.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from . import membership as mb

#: The registered universes. `all` is the shipped book and the only one every
#: recorded F1/F2/F3 number describes.
UNIVERSES: tuple[str, ...] = ("all", "nifty500", "nifty100",
                              "size500", "size100")

#: Each index restriction gets a size-ranked twin of the same width, so no
#: published-membership arm is ever reported without a control beside it. An
#: uncontrolled `nifty100` would be the same trap as an uncontrolled
#: `nifty500`, just on a shorter list - and a shorter list is a STRONGER
#: filter, because "grew into the Nifty 100 by 2026" selects harder than
#: "grew into the Nifty 500".
SIZE_RANKS: dict[str, int] = {"size500": 500, "size100": 100}

#: Kept for callers that predate the pair. Same value it always had.
SIZE_RANK_N = 500


class UnknownUniverse(KeyError):
    """Asked for a universe that is not registered. Never guess a default."""


def _implied_shares(bars: dict, market_caps: dict) -> dict:
    """
    `{symbol: shares}` implied by today's market cap over today's close.

    Yahoo gives a market cap and not a share count, and a share count is what
    turns a price history into a size history. Dividing one by the other is
    exact today and drifts backwards only with issuance and buybacks - which is
    the whole reason this is a defensible control: the error does not know
    whether the company later joined an index.
    """
    out = {}
    for symbol, cap in market_caps.items():
        frame = bars.get(symbol)
        if frame is None or frame.empty or not cap or cap <= 0:
            continue
        try:
            last = float(frame["close"].iloc[-1])
        except Exception:
            continue
        if last > 0:
            out[symbol] = float(cap) / last
    return out


def load_market_caps(cfg, market_key: str = "factor_india") -> dict:
    """
    `{symbol: market cap}` from the fundamentals cache FILE. No network.

    Deliberately not `load_fundamentals`, which fetches whatever is missing or
    stale - handing it a universe fires a thousand requests as a side effect of
    a measurement. Same reasoning as `run_f3_screened.verdict_map` and
    `run_s1_swing_null`'s direct parquet read.
    """
    path = Path(cfg.swing.cache_dir) / "fundamentals.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    prefix = f"{market_key}:"
    out = {}
    for key, payload in raw.items():
        if not key.startswith(prefix) or not isinstance(payload, dict):
            continue
        cap = payload.get("market_cap")
        if cap:
            out[key[len(prefix):]] = float(cap)
    return out


class SizeRank:
    """
    Top `n` by point-in-time size, recomputed per rebalance.

    A NAME WITH NO MARKET CAP IS EXCLUDED, NOT RANKED AT ZERO. Ranking it at
    zero would put it at the bottom of every list and quietly guarantee it is
    never held, which reads identically to "it was small" - a missing fact
    wearing a measurement's clothes. Excluding it is the same rule
    `HalalVerdict` applies to an absent balance sheet, and the count is
    reported so the exclusion is visible.
    """

    def __init__(self, universe, shares: dict, n: int = SIZE_RANK_N):
        self.n = n
        self.shares = shares
        self.universe = universe
        self.symbols = [s for s in shares if s in universe.symbols]
        self.skipped = sorted(set(universe.symbols) - set(shares))

    def at(self, day: date) -> set:
        sizes = []
        for symbol in self.symbols:
            price = self.universe.price_at(symbol, day)   # strictly before day
            if price is None or price <= 0:
                continue
            sizes.append((self.shares[symbol] * price, symbol))
        if not sizes:
            return set()
        sizes.sort(reverse=True)
        return {symbol for _size, symbol in sizes[:self.n]}


def resolver(cfg, universe_key: str, bars: dict, factor_universe,
             root: str | Path = "."):
    """
    `day -> set | None` for one universe key, plus a one-line description.

    Returns `(fn, note)`. `None` from `fn` means no restriction at all, which
    is what keeps `universe="all"` a true no-op rather than a filter that
    happens to pass everything.
    """
    key = (universe_key or "all").lower()
    if key not in UNIVERSES:
        raise UnknownUniverse(
            f"unknown universe {universe_key!r} - registered are "
            f"{', '.join(UNIVERSES)}")

    if key == "all":
        return (lambda _day: None), "all listed NSE equity (as tested)"

    if key in ("nifty500", "nifty100"):
        members = mb.load(root)
        band = "nifty500" if key == "nifty500" else "nifty100"
        names = members.sets.get(band)
        if not names:
            raise UnknownUniverse(
                f"{key} needs data/{band}.csv - run "
                f"`python -m nifty_algo.factor.membership --refresh`")
        note = (f"{key}: {len(names):,} names from TODAY'S published list. "
                f"In a backtest this is LOOK-AHEAD - it can only hold "
                f"companies that grew into the index.")
        return (lambda _day, n=frozenset(names): set(n)), note

    n = SIZE_RANKS[key]
    caps = load_market_caps(cfg)
    shares = _implied_shares(bars, caps)
    rank = SizeRank(factor_universe, shares, n)
    note = (f"{key}: top {n} by point-in-time size, from "
            f"{len(rank.symbols):,} names with a market cap; "
            f"{len(rank.skipped):,} excluded for having none.")
    return rank.at, note


def compose(*fns):
    """
    Intersect several restrictions, treating None as "no opinion".

    `listed_only` already produces a restriction of its own, and two
    restrictions must narrow rather than replace - a universe key silently
    overriding the survivorship bound would answer a different question with
    the same label.
    """
    live = [f for f in fns if f is not None]
    if not live:
        return None

    def apply(day):
        out = None
        for fn in live:
            got = fn(day)
            if got is None:
                continue
            out = set(got) if out is None else (out & set(got))
        return out

    return apply


def static(names) -> object:
    """Wrap a fixed set as a resolver, for `listed_only` and for tests."""
    if names is None:
        return None
    frozen = frozenset(names)
    return lambda _day: set(frozen)


__all__ = ["UNIVERSES", "UnknownUniverse", "SizeRank", "resolver", "compose",
           "static", "load_market_caps", "SIZE_RANK_N"]
