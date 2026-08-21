"""
The phrase table the news scorer reads.

A LEXICON, NOT A MODEL. It matches substrings in headlines and adds up
weights. It cannot read tone, it does not know when something is already
priced in, and it will occasionally score a headline about the wrong company
in the same sector. That ceiling is the price of a scorer that needs no API
key, no model, and no per-scan cost - and it is why news is weighted at 10%
of the composite score rather than being allowed to drive a pick on its own.

The vetoes are different in kind from the weights. A weight nudges a ranking;
a veto removes a candidate entirely regardless of how good the chart looks.
They are deliberately conservative: a false veto costs you one candidate out
of a hundred, and holding a swing position through a fraud investigation
costs considerably more.
"""
from __future__ import annotations

# ---------------- weighted phrases ----------------
# Longer, more specific phrases first: the scorer stops at the first match
# within a family so "target price cut" is not also counted as "target price".

POSITIVE: tuple[tuple[str, float], ...] = (
    # Broker actions
    ("target price raised", 0.9),
    ("raises target", 0.9),
    ("hikes target", 0.9),
    ("upgrades to buy", 1.0),
    ("upgrade", 0.8),
    ("initiates coverage with buy", 0.7),
    ("outperform", 0.6),
    ("top pick", 0.6),

    # Results
    ("beats estimates", 0.9),
    ("beats street", 0.9),
    ("record profit", 0.9),
    ("profit jumps", 0.8),
    ("profit rises", 0.7),
    ("profit surges", 0.8),
    ("revenue jumps", 0.6),
    ("strong quarter", 0.7),
    ("margin expansion", 0.6),

    # Business wins
    ("bags order", 0.9),
    ("wins order", 0.9),
    ("wins contract", 0.9),
    ("secures contract", 0.8),
    ("order win", 0.8),
    ("order book", 0.5),
    ("new plant", 0.5),
    ("capacity expansion", 0.5),
    ("capex", 0.4),
    ("acquires", 0.5),
    ("partnership", 0.4),
    ("tie-up", 0.4),

    # Regulatory green lights
    ("usfda approval", 0.9),
    ("drug approval", 0.8),
    ("receives approval", 0.7),
    ("gets approval", 0.7),
    ("clearance", 0.5),

    # Ownership and index events
    ("buyback", 0.7),
    ("bonus issue", 0.6),
    ("stock split", 0.4),
    ("promoter buys", 0.7),
    ("raises stake", 0.6),
    ("index inclusion", 0.7),
    ("msci inclusion", 0.7),
    ("dividend", 0.3),
)

NEGATIVE: tuple[tuple[str, float], ...] = (
    # Broker actions
    ("target price cut", -0.9),
    ("cuts target", -0.9),
    ("slashes target", -0.9),
    ("downgrades to sell", -1.0),
    ("downgrade", -0.8),
    ("underperform", -0.6),

    # Results
    ("misses estimates", -0.9),
    ("profit falls", -0.8),
    ("profit declines", -0.8),
    ("profit drops", -0.8),
    ("loss widens", -0.9),
    ("posts loss", -0.9),
    ("weak quarter", -0.7),
    ("margin pressure", -0.6),
    ("guidance cut", -0.9),

    # Operations
    ("plant shutdown", -0.7),
    ("production halt", -0.7),
    ("recall", -0.6),
    ("workers strike", -0.6),
    ("fire at", -0.5),

    # Regulatory friction (short of a veto)
    ("usfda observation", -0.8),
    ("form 483", -0.8),
    ("import alert", -0.9),
    ("warning letter", -0.8),
    ("show cause notice", -0.6),
    ("tax demand", -0.5),
    ("gst notice", -0.5),
    ("penalty", -0.5),

    # Ownership and people
    ("offer for sale", -0.6),
    ("stake sale", -0.5),
    ("promoter sells", -0.7),
    ("block deal", -0.4),
    ("steps down", -0.5),
    ("resigns", -0.5),
    ("layoffs", -0.5),
)

# ---------------- hard vetoes ----------------
# A match here kills the candidate outright. The headline that did it is
# always shown, so a false positive is visible and arguable rather than an
# unexplained absence.

VETOES: tuple[tuple[str, str], ...] = (
    ("fraud", "fraud allegation"),
    ("forensic audit", "forensic audit ordered"),
    ("accounting irregularit", "accounting irregularities"),
    ("whistleblower", "whistleblower complaint"),
    ("auditor resigns", "auditor resignation"),
    ("auditor quits", "auditor resignation"),
    ("sebi probe", "SEBI investigation"),
    ("sebi bars", "SEBI action"),
    ("sebi order against", "SEBI action"),
    ("insider trading", "insider trading case"),
    ("money laundering", "money laundering case"),
    ("enforcement directorate", "ED action"),
    ("ed raid", "ED raid"),
    ("income tax raid", "tax raid"),
    ("cbi probe", "CBI investigation"),
    ("cbi raid", "CBI raid"),
    ("defaults on", "debt default"),
    ("loan default", "debt default"),
    ("insolvency", "insolvency proceedings"),
    # NOT a bare "nclt". The tribunal handles routine subsidiary
    # amalgamations alongside insolvency, and "NCLT approves amalgamation of
    # Inzpera Healthsciences with Cipla" is a housekeeping notice, not a
    # reason to refuse the trade - it vetoed Cipla on live data before this
    # was narrowed. The insolvency wording above catches the real cases.
    ("admitted to nclt", "NCLT insolvency admission"),
    ("nclt petition against", "NCLT petition"),
    ("nclt plea against", "NCLT petition"),
    ("pledge invoked", "promoter pledge invoked"),
    ("invokes pledged shares", "promoter pledge invoked"),
    ("trading halt", "trading halted"),
    ("delisting", "delisting"),
)


def score_headline(title: str) -> tuple[float, list[str]]:
    """
    Score one headline in [-1, +1] and list the phrases that did it.

    Capped at +/-1 so a headline stuffed with five bullish words cannot
    outvote five separate stories - the aggregate is meant to measure how
    many things are being said, not how loudly one of them is said.
    """
    text = title.lower()
    total = 0.0
    matched: list[str] = []
    for phrase, weight in POSITIVE + NEGATIVE:
        if phrase in text:
            total += weight
            matched.append(f"{phrase} ({weight:+.1f})")
    return max(-1.0, min(1.0, total)), matched


def veto_reason(title: str) -> str | None:
    """The veto this headline trips, or None."""
    text = title.lower()
    for phrase, label in VETOES:
        if phrase in text:
            return label
    return None
