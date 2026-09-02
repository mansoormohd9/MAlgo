"""
NSE token resolution, and the rename diagnosis that made it useful.

The behaviour under test is not "does it map a symbol to a number" - that is
a dict lookup. It is the reporting contract around the lookup:

  - a symbol that does not resolve is NAMED, never skipped, so a universe
    cannot quietly shrink (`accounts_for`);
  - a symbol that does not resolve comes back with a LEAD, because the first
    real run printed `UNRESOLVED: TATAMOTORS, LTIM` and both were renames -
    the companies were listed the whole time under new tradingsymbols;
  - and a lead is never applied. `tokens` must never contain a symbol that
    was only suggested.

Everything here runs offline against a hand-built dump, in the style of
`conftest.py`: engineer the exact input rather than hunt for one in real data.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from nifty_algo.data import instruments as instr


# The shape Kite's `instruments("NSE")` returns, cut down to the columns this
# module reads. TMPV and LTM stand in for the two real renames.
DUMP = [
    {"segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "RELIANCE",
     "instrument_token": 738561, "name": "RELIANCE INDUSTRIES LTD"},
    {"segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "TMPV",
     "instrument_token": 884737, "name": "TATA MOTORS PASSENGER VEHICLES LTD"},
    {"segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "LTM",
     "instrument_token": 4561409, "name": "LTIMINDTREE LIMITED"},
    {"segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "TATASTEEL",
     "instrument_token": 895745, "name": "TATA STEEL LIMITED"},
    # Deliberate noise: an ETF and a non-EQ series must never resolve, or the
    # book could take an MIS position in a fund by accident.
    {"segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "NIFTYBEES",
     "instrument_token": 130305, "name": "NIPPON INDIA ETF NIFTY 50 BEES"},
    {"segment": "NSE", "instrument_type": "SG", "tradingsymbol": "GS2030",
     "instrument_token": 111, "name": "GOVERNMENT SECURITY"},
    {"segment": "NFO-OPT", "instrument_type": "CE", "tradingsymbol": "NIFTY25000CE",
     "instrument_token": 222, "name": "NIFTY"},
]

#: The universe file's `name` column for the two names that no longer resolve.
UNIVERSE_NAMES = {"TATAMOTORS": "Tata Motors", "LTIM": "LTIMindtree"}


class FakeKite:
    def __init__(self, rows=DUMP, fail=False):
        self.rows = rows
        self.fail = fail
        self.calls = 0

    def instruments(self, exchange):
        self.calls += 1
        if self.fail:
            raise RuntimeError("network down")
        return list(self.rows)


class FakeSession:
    """Stands in for `KiteSession` without touching `.kite_session.json`."""

    def __init__(self, kite=None):
        self.kite = kite or FakeKite()
        self.authenticated = True

    def client(self):
        return self.kite


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path)


# --------------------------------------------------------------- resolution


def test_only_nse_eq_rows_resolve(cache_dir):
    """An ETF is EQ and resolves; a government security and an option do not."""
    got = instr.resolve(["RELIANCE", "GS2030", "NIFTY25000CE"],
                        FakeSession(), cache_dir=cache_dir)
    assert got.get("RELIANCE") == 738561
    assert got.get("GS2030") is None
    assert got.get("NIFTY25000CE") is None


def test_an_unresolved_symbol_is_named_never_skipped(cache_dir):
    """
    The contract the whole class exists for.

    A universe of four that resolves to two must say which two are gone. A
    scan that is quietly 2% blind reports those names as having had no setup.
    """
    wanted = ["RELIANCE", "TATAMOTORS", "LTIM", "TATASTEEL"]
    got = instr.resolve(wanted, FakeSession(), cache_dir=cache_dir,
                        names=UNIVERSE_NAMES)

    assert sorted(got.missing) == ["LTIM", "TATAMOTORS"]
    assert instr.accounts_for(got, wanted)
    assert len(got) + len(got.missing) == len(wanted)


# --------------------------------------------------------------- suggestions


def test_a_renamed_symbol_is_suggested_by_company_name(cache_dir):
    """
    The regression for the run that started all this.

    TATAMOTORS and TMPV share NOT ONE CHARACTER, so no amount of ticker
    matching finds this. It is only reachable through the dump's `name`
    column, which this module used to discard.
    """
    got = instr.resolve(["TATAMOTORS", "LTIM"], FakeSession(),
                        cache_dir=cache_dir, names=UNIVERSE_NAMES)

    assert got.suggestions["TATAMOTORS"][0][0] == "TMPV"
    assert got.suggestions["LTIM"][0][0] == "LTM"


def test_a_suggestion_is_never_applied_automatically(cache_dir):
    """
    A lead is printed for a human, never substituted.

    Silently resolving TATAMOTORS to TMPV would screen one company against
    another's balance sheet - the failure `{market}:{SYMBOL}` keying exists
    to prevent - and it would do it without raising anything.
    """
    got = instr.resolve(["TATAMOTORS"], FakeSession(), cache_dir=cache_dir,
                        names=UNIVERSE_NAMES)

    assert "TATAMOTORS" in got.missing
    assert got.get("TATAMOTORS") is None
    assert "TMPV" not in got.tokens
    assert len(got) == 0


def test_a_weak_prefix_is_not_offered_as_a_lead(cache_dir):
    """
    "TATA" is four characters of ten and matches every Tata company.

    Without the share floor those fill all three suggestion slots and bury
    the one candidate that matched on name.
    """
    got = instr.resolve(["TATAMOTORS"], FakeSession(), cache_dir=cache_dir,
                        names=UNIVERSE_NAMES)
    offered = [sym for sym, _ in got.suggestions["TATAMOTORS"]]
    assert offered == ["TMPV"]
    assert "TATASTEEL" not in offered


def test_an_unrecognisable_symbol_gets_no_lead_and_says_so(cache_dir):
    """
    "No candidate" is a finding - it is evidence of a delisting rather than
    a rename - so it gets a line of its own instead of silence.
    """
    got = instr.resolve(["ZZZDELISTED"], FakeSession(), cache_dir=cache_dir,
                        names={"ZZZDELISTED": "Some Vanished Company"})
    assert got.suggestions.get("ZZZDELISTED") is None
    line = " ".join(got.suggestion_lines())
    assert "ZZZDELISTED" in line and "no candidate" in line and "delisting" in line


def test_without_the_name_map_a_ticker_rename_is_still_found(cache_dir):
    """
    `names` is optional. Prefix matching alone cannot find TATAMOTORS ->
    TMPV, and must still find a ticker that was merely extended.
    """
    rows = DUMP + [{"segment": "NSE", "instrument_type": "EQ",
                    "tradingsymbol": "RELIANCEPP", "instrument_token": 9,
                    "name": "RELIANCE PARTLY PAID"}]
    got = instr.resolve(["RELIANCEX"], FakeSession(FakeKite(rows)),
                        cache_dir=cache_dir)
    assert [s for s, _ in got.suggestions["RELIANCEX"]] == ["RELIANCE", "RELIANCEPP"]


# --------------------------------------------------------------- the cache


def test_names_are_cached_alongside_tokens(cache_dir):
    instr.resolve(["RELIANCE"], FakeSession(), cache_dir=cache_dir)
    raw = json.loads(instr.cache_path(cache_dir).read_text(encoding="utf-8"))

    assert raw["tokens"]["RELIANCE"] == 738561
    assert raw["names"]["TMPV"] == "TATA MOTORS PASSENGER VEHICLES LTD"


def test_an_old_cache_without_names_still_resolves(cache_dir):
    """
    `names` arrived after `tokens`. A cache written before it must resolve
    normally - it simply cannot suggest - rather than being treated as torn
    and forcing a re-download on a path that may have no network.
    """
    instr.cache_path(cache_dir).parent.mkdir(parents=True, exist_ok=True)
    instr.cache_path(cache_dir).write_text(json.dumps({
        "fetched_on": date.today().isoformat(),
        "tokens": {"RELIANCE": 738561},
    }), encoding="utf-8")

    kite = FakeKite()
    got = instr.resolve(["RELIANCE", "TATAMOTORS"], FakeSession(kite),
                        cache_dir=cache_dir, names=UNIVERSE_NAMES)

    assert kite.calls == 0                     # served from cache
    assert got.get("RELIANCE") == 738561
    assert got.missing == ["TATAMOTORS"]
    assert got.suggestions == {}               # degraded, not wrong


def test_a_stale_cache_serves_when_the_download_fails(cache_dir):
    """
    Permanent tokens do not rot, so a month-old map is still correct for
    every name it holds. It is only missing what listed since, which comes
    back as unresolved rather than hidden.
    """
    old = (date.today() - timedelta(days=30)).isoformat()
    instr.cache_path(cache_dir).parent.mkdir(parents=True, exist_ok=True)
    instr.cache_path(cache_dir).write_text(json.dumps({
        "fetched_on": old,
        "tokens": {"RELIANCE": 738561},
        "names": {"RELIANCE": "RELIANCE INDUSTRIES LTD"},
    }), encoding="utf-8")

    got = instr.resolve(["RELIANCE", "TMPV"],
                        FakeSession(FakeKite(fail=True)), cache_dir=cache_dir)

    assert got.from_cache and got.fetched_on == date.fromisoformat(old)
    assert got.get("RELIANCE") == 738561
    assert got.missing == ["TMPV"]


def test_a_torn_cache_costs_a_redownload_not_a_crash(cache_dir):
    instr.cache_path(cache_dir).parent.mkdir(parents=True, exist_ok=True)
    instr.cache_path(cache_dir).write_text("{not json", encoding="utf-8")

    kite = FakeKite()
    got = instr.resolve(["RELIANCE"], FakeSession(kite), cache_dir=cache_dir)

    assert kite.calls == 1
    assert got.get("RELIANCE") == 738561


def test_no_cache_and_no_network_raises_rather_than_returning_empty(cache_dir):
    """
    An empty token map read as "nothing resolved" would report the entire
    universe as delisted. It has to raise.
    """
    with pytest.raises(instr.InstrumentsUnavailable):
        instr.resolve(["RELIANCE"], FakeSession(FakeKite(fail=True)),
                      cache_dir=cache_dir)


def test_the_note_names_the_unresolved_symbols(cache_dir):
    got = instr.resolve(["RELIANCE", "TATAMOTORS"], FakeSession(),
                        cache_dir=cache_dir, names=UNIVERSE_NAMES)
    note = got.note()
    assert "TATAMOTORS" in note and "UNRESOLVED" in note
