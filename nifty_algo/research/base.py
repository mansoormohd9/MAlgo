"""
The envelope every research briefing is delivered in.

WHAT THIS PACKAGE IS FOR, AND WHAT IT DELIBERATELY IS NOT. The ten briefings
this repo is growing - macro impact, portfolio risk, DCF, technicals and the
rest - are each part arithmetic and part judgment. The arithmetic belongs
here, in Python, deterministic and testable. The judgment (a Fed policy
outlook, a moat rating, a SWOT) belongs to whatever writes the prose, and this
package's job is to hand that writer a set of facts it cannot get wrong.

So a `FactPack` is not a report. It is the evidence a report is allowed to
cite, and its central rule is the one `news.NewsResult.available` and
`halal.py` already follow throughout this repo:

    A MISSING FACT IS NEVER A BENIGN VALUE.

`Fact.available` is False by default, `Fact.unknown()` demands a reason, and a
`Fact` that is not available carries no value at all - so a consumer cannot
read a zero out of a field Yahoo simply did not return. This matters more here
than anywhere else in the codebase, because roughly a third of what these
briefings ask for does not exist for Indian equities on any free source:
there is no short-interest disclosure in India at all, insider dealing is a
SEBI PIT filing rather than an API, and analyst consensus is sparse. Rendered
as 0% or as a blank cell, every one of those reads as good news.

WHY THE FACT PACK IS THE PRODUCT. Being JSON with an explicit availability
flag on every number is what lets a Claude Code skill write a McKinsey-voiced
briefing without inventing anything: the skill's rule is that no figure may
appear in the prose that is not in the pack, and every `available: false` must
be named as unavailable rather than quietly skipped. `unavailable()` below
exists so that rule is one call to check rather than a hunt through the tree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

#: Where a number came from. Free text, but always specific enough to re-fetch
#: by hand - "yfinance ^TNX" and not "the internet".
SOURCE_YAHOO = "yfinance"
SOURCE_CACHE = "data/cache (local parquet)"
SOURCE_PORTFOLIO = "portfolio connectors"
SOURCE_CONFIG = "nifty_algo/config.py"
SOURCE_COMMITTED = "committed repo data"


@dataclass(frozen=True)
class Fact:
    """
    One number, with everything needed to trust or discard it.

    `available=False` carries `value=None` and always a `note`. There is no
    third state and no sentinel value - a fact we do not have must be
    impossible to accidentally arithmetic with.
    """
    label: str
    value: Any = None
    unit: str = ""
    source: str = ""
    as_of: str = ""
    available: bool = False
    note: str = ""

    @classmethod
    def known(cls, label: str, value: Any, source: str, as_of: str = "",
              unit: str = "", note: str = "") -> "Fact":
        return cls(label=label, value=value, unit=unit, source=source,
                   as_of=as_of, available=True, note=note)

    @classmethod
    def unknown(cls, label: str, note: str, source: str = "") -> "Fact":
        """
        The only way to build an unavailable fact, and it requires a reason.

        The reason is the point: "Yahoo returned no balance sheet" and "India
        publishes no short interest" are different findings, and a reader who
        is told neither will assume the first.
        """
        return cls(label=label, value=None, source=source, available=False,
                   note=note)

    def display(self) -> str:
        if not self.available:
            return f"unavailable - {self.note}"
        if isinstance(self.value, float):
            # Thousands separators above 1000, significant figures below it.
            # A one-rule format prints an index level as 2.409e+04 or a
            # correlation as 0.58, and both are unreadable in the wrong place.
            body = (f"{self.value:,.2f}" if abs(self.value) >= 1000
                    else f"{self.value:,.4g}")
            return f"{body}{(' ' + self.unit) if self.unit else ''}"
        return f"{self.value}{(' ' + self.unit) if self.unit else ''}"

    def to_dict(self) -> dict:
        return {"label": self.label, "value": self.value, "unit": self.unit,
                "source": self.source, "as_of": self.as_of,
                "available": self.available, "note": self.note}


@dataclass
class Section:
    """
    One heading of a briefing: some scalar facts, optionally a table.

    `rows` is a plain list of dicts rather than a DataFrame so the pack
    serialises without pandas in the loop and a skill can read it directly.
    `judgment` names what this section CANNOT answer from data - it is the
    explicit handover to whatever writes the prose, and it is a field rather
    than a comment so the handover is machine-readable.
    """
    name: str
    facts: list[Fact] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    note: str = ""
    judgment: list[str] = field(default_factory=list)

    def add(self, fact: Fact) -> Fact:
        self.facts.append(fact)
        return fact

    def unavailable(self) -> list[Fact]:
        return [f for f in self.facts if not f.available]

    def to_dict(self) -> dict:
        return {"name": self.name,
                "facts": [f.to_dict() for f in self.facts],
                "rows": self.rows, "note": self.note,
                "judgment_required": self.judgment}


@dataclass
class FactPack:
    """
    Everything one briefing is allowed to cite.

    `caveats` are the distortions that make the numbers below mean less than
    they appear to - survivorship in the cached universe, a partial portfolio,
    a stale balance sheet. They are printed ABOVE the result, the way
    `swing/backtest.py` prints its three structural distortions, because a
    caveat under a table is one nobody reads.
    """
    report: str
    title: str = ""
    generated_at: str = ""
    inputs: dict = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    stood_down: str = ""          # non-empty: the report could not be produced

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat(timespec="seconds")

    def section(self, name: str, **kwargs) -> Section:
        s = Section(name=name, **kwargs)
        self.sections.append(s)
        return s

    def unavailable(self) -> list[dict]:
        """
        Every fact this run could not establish, flattened.

        The skill contract depends on this: naming what is missing is not
        optional politeness, it is the difference between "no insider selling"
        and "India does not publish insider dealing this way".
        """
        return [{"section": s.name, **f.to_dict()}
                for s in self.sections for f in s.unavailable()]

    def judgment_required(self) -> list[dict]:
        """What the data cannot decide, and which section is waiting on it."""
        return [{"section": s.name, "question": q}
                for s in self.sections for q in s.judgment]

    def to_dict(self) -> dict:
        return {
            "report": self.report,
            "title": self.title,
            "generated_at": self.generated_at,
            "inputs": self.inputs,
            "stood_down": self.stood_down,
            "caveats": self.caveats,
            "sections": [s.to_dict() for s in self.sections],
            "unavailable": self.unavailable(),
            "judgment_required": self.judgment_required(),
        }

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_encode)


def _encode(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    # numpy scalars and pandas Timestamps both answer to item()/isoformat()
    for attr in ("isoformat", "item"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return str(value)
