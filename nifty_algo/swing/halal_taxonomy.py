"""
The prohibited-activity tables, one per classification vocabulary.

WHY TWO TABLES AND NOT ONE BIGGER ONE. The activity screen matches substrings
against a stock's `industry` and `sector` labels, and those labels come from
whoever classified the exchange. NSE says "Non Banking Financial Company
(NBFC)"; Yahoo says "Credit Services". NSE says "Breweries & Distilleries";
Yahoo says "Beverages - Wineries & Distilleries". The two vocabularies share
almost no strings, so one merged table would be a list of needles most of
which can never match, and a gap in either half would be invisible.

WHAT MOVED, AND WHAT DID NOT. The NSE table is the original from `halal.py`,
unchanged - India's verdicts must not shift because a second market was added.
The GICS table is new and is deliberately NOT a transliteration of it, because
two of the India entries are wrong in Yahoo's vocabulary:

  "gaming" - correct for NSE, where it means Delta Corp's casinos. In Yahoo it
  also matches "Electronic Gaming & Multimedia", which is video games. Video
  games are not gambling and no mainstream Shariah index excludes them, so the
  GICS table names the gambling terms explicitly instead.

  "holding company" - a genuine flag on NSE. In Yahoo, "Shell Companies" and
  conglomerate labels catch operating businesses that happen to be structured
  as holdcos, so it is left to the ratio screen, which is what it is for.

THE THREE CONTESTED CATEGORIES ARE TOGGLES, NOT RULINGS. The two US ETFs in
this user's portfolio disagree about them: SPUS (AAOIFI / S&P) excludes
aerospace & defence and financial exchanges & data; HLAL (FTSE / Yasaar) does
not. Neither is a mistake. Hardcoding either one would be this module quietly
picking a madhhab, so they are configuration with the standard named in the
comment, and the pick card says which toggle removed a name.
"""
from __future__ import annotations

from . import markets as markets_mod

# --------------------------------------------------------------------------
# NSE / Indian exchange labels. Verbatim from the original halal.py table.
# --------------------------------------------------------------------------
NSE_ACTIVITIES: tuple[tuple[str, str], ...] = (
    # Interest-based finance - the largest single block in the Nifty 100.
    ("private sector bank", "conventional banking (interest-based)"),
    ("public sector bank", "conventional banking (interest-based)"),
    ("non banking financial company", "interest-based lending (NBFC)"),
    ("nbfc", "interest-based lending (NBFC)"),
    ("housing finance", "interest-based lending"),
    ("financial institution", "interest-based finance"),
    ("investment company", "conventional investment holding"),
    ("holding company", "conventional financial holding"),
    ("asset management", "conventional asset management"),
    ("stockbroking", "conventional brokerage"),
    ("depositories", "conventional financial services"),
    ("bank", "conventional banking (interest-based)"),

    # Insurance.
    ("life insurance", "conventional insurance"),
    ("general insurance", "conventional insurance"),
    ("insurance", "conventional insurance"),

    # Intoxicants and tobacco.
    ("breweries", "alcohol"),
    ("distilleries", "alcohol"),
    ("alcohol", "alcohol"),
    ("liquor", "alcohol"),
    ("tobacco", "tobacco"),
    ("cigarette", "tobacco"),

    # Gambling and conventional entertainment.
    ("gambling", "gambling"),
    ("casino", "gambling"),
    ("lottery", "gambling"),
    ("gaming", "gambling / conventional gaming"),
    ("film production", "conventional entertainment"),
    ("movies & entertainment", "conventional entertainment"),
    ("cinema", "conventional entertainment"),
    ("multiplex", "conventional entertainment"),

    # Pork and non-halal meat.
    ("pork", "pork products"),

    # Hotels: the objection is the bar and banquet revenue, which for Indian
    # listed hotel groups is material and is not separately disclosed in any
    # free feed. Excluded here, and an obvious candidate for an override if
    # you check a specific company's accounts and disagree.
    ("hotels & resorts", "hotel bar/banquet revenue not separable"),
    ("hotels", "hotel bar/banquet revenue not separable"),
)

# --------------------------------------------------------------------------
# Yahoo / GICS-derived labels, as returned for US and UK listings.
#
# Yahoo renders the separator as an em dash in some releases and a hyphen in
# others ("Banks-Diversified" vs "Banks - Diversified"), so every needle here
# is a fragment that survives both.
# --------------------------------------------------------------------------
GICS_ACTIVITIES: tuple[tuple[str, str], ...] = (
    # --- interest-based finance ---
    ("mortgage finance", "interest-based lending"),
    ("reit - mortgage", "interest-bearing mortgage assets"),
    ("reit-mortgage", "interest-bearing mortgage assets"),
    ("mortgage reit", "interest-bearing mortgage assets"),
    # Visa, Mastercard and PayPal are classified here alongside Amex, Discover
    # and Capital One, who are unambiguously lenders. The card networks are
    # the classic override candidate - see the note in halal.py.
    ("credit services", "interest-based lending / card credit"),
    ("capital markets", "conventional brokerage and dealing"),
    ("asset management", "conventional asset management"),
    ("banks", "conventional banking (interest-based)"),
    ("bank", "conventional banking (interest-based)"),
    ("savings & cooperative", "conventional banking (interest-based)"),
    ("thrift", "conventional banking (interest-based)"),

    # --- insurance ---
    ("insurance", "conventional insurance"),
    ("reinsurance", "conventional insurance"),

    # --- intoxicants and tobacco ---
    ("brewers", "alcohol"),
    ("wineries & distilleries", "alcohol"),
    ("distilleries", "alcohol"),
    ("beverages - alcoholic", "alcohol"),
    ("tobacco", "tobacco"),

    # --- gambling ---
    # Named explicitly rather than via "gaming": see the module docstring.
    ("gambling", "gambling"),
    ("casino", "gambling"),
    ("resorts & casinos", "gambling"),
    ("lottery", "gambling"),
    ("betting", "gambling"),

    # --- conventional entertainment and adult content ---
    ("entertainment", "conventional entertainment"),
    ("broadcasting", "conventional broadcasting"),

    # --- hotels: the same bar-revenue objection as the Indian table ---
    ("lodging", "hotel bar revenue not separable"),
)

#: Contested categories, each behind a config toggle. The value is
#: (needles, reason, config attribute on HalalConfig).
GICS_TOGGLED: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("aerospace & defense", "aerospace & defence", "defense", "defence"),
        "aerospace & defence (excluded by AAOIFI/S&P, permitted by FTSE)",
        "exclude_defence",
    ),
    (
        ("financial data & stock exchanges", "financial exchanges",
         "financial data"),
        "financial exchanges & data (excluded by AAOIFI/S&P, permitted by FTSE)",
        "exclude_exchanges_and_data",
    ),
    (
        ("shell companies",),
        "shell company - no operating business to screen",
        "exclude_shell_companies",
    ),
)

TABLES = {
    markets_mod.TAXONOMY_NSE: NSE_ACTIVITIES,
    markets_mod.TAXONOMY_GICS: GICS_ACTIVITIES,
}


def table_for(taxonomy: str) -> tuple[tuple[str, str], ...]:
    """
    The activity table for a vocabulary.

    Unknown taxonomies raise rather than defaulting: screening a US stock
    against NSE's vocabulary would match nothing at all and report every bank
    on the S&P 500 as activity-permissible.
    """
    try:
        return TABLES[taxonomy]
    except KeyError:
        raise KeyError(
            f"no activity table for taxonomy {taxonomy!r} - known tables are "
            f"{', '.join(sorted(TABLES))}"
        ) from None


def toggled_for(taxonomy: str, hcfg) -> tuple[tuple[str, str], ...]:
    """The contested entries currently switched on, flattened into needles."""
    if taxonomy != markets_mod.TAXONOMY_GICS:
        return ()
    out: list[tuple[str, str]] = []
    for needles, reason, attr in GICS_TOGGLED:
        if getattr(hcfg, attr, False):
            out.extend((needle, reason) for needle in needles)
    return tuple(out)
