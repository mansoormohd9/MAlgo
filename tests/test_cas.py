"""
Mutual funds from a CAS. `casparser` itself is not exercised - its output is
the input here - so the suite needs neither the package nor a real statement.
"""
from __future__ import annotations

import json

from nifty_algo.portfolio import cas
from nifty_algo.portfolio.base import FIXED_INCOME, GOLD, MUTUAL_FUND

#: The shape `casparser.read_cas_pdf(..., output="dict")` returns, trimmed to
#: the fields this connector reads.
PARSED = {
    "statement_period": {"from": "01-Jan-2020", "to": "31-Aug-2026"},
    "folios": [
        {"folio": "123/45", "schemes": [
            {"scheme": "Parag Parikh Flexi Cap Fund - Direct Growth",
             "isin": "INF879O01027", "amfi": "122639", "type": "EQUITY",
             "close": 1500.123,
             "valuation": {"date": "2026-08-29", "nav": 90.0,
                           "value": 135011.07, "cost": 100000.0}},
            {"scheme": "Some Liquid Fund - Direct Growth",
             "isin": "INF000000001", "type": "DEBT", "close": 10.0,
             "valuation": {"date": "2026-08-29", "nav": 3000.0,
                           "value": 30000.0, "cost": 29000.0}},
            {"scheme": "Closed Out Fund", "isin": "INF000000002",
             "type": "EQUITY", "close": 0.0,
             "valuation": {"date": "2026-08-29", "nav": 10.0, "value": 0.0}},
        ]},
        {"folio": "999", "schemes": [
            {"scheme": "SBI Gold Fund - Direct Growth", "isin": "INF200K01XX1",
             "type": None, "close": 100.0,
             "valuation": {"date": "2026-08-30", "nav": 25.0, "value": 2500.0}},
        ]},
    ],
}


def test_normalise_keeps_closing_balances_only():
    out = cas.normalise(PARSED)
    assert out["statement_date"] == "2026-08-30"      # the latest valuation
    names = [h["name"] for h in out["holdings"]]
    assert "Closed Out Fund" not in names             # zero units dropped
    assert len(out["holdings"]) == 3
    assert not any("transactions" in h for h in out["holdings"])


def test_scheme_types_map_to_allocation_classes():
    by_name = {h["name"]: cas.to_position(h, "2026-08-30")
               for h in cas.normalise(PARSED)["holdings"]}
    assert by_name["Parag Parikh Flexi Cap Fund - Direct Growth"].asset_class \
        == MUTUAL_FUND
    assert by_name["Some Liquid Fund - Direct Growth"].asset_class == FIXED_INCOME
    assert by_name["SBI Gold Fund - Direct Growth"].asset_class == GOLD
    flexi = by_name["Parag Parikh Flexi Cap Fund - Direct Growth"]
    assert flexi.value_native == 135011.07
    assert flexi.key == "india:INF879O01027"
    assert flexi.currency == "INR"


def test_enabled_but_never_imported_is_unavailable(tmp_path):
    result = cas.CasConnector(path=tmp_path / "none.json").fetch()
    assert result.available is False
    assert "no CAS imported" in result.note


def test_an_import_is_read_back_with_its_age(tmp_path):
    path = tmp_path / "cas.json"
    path.write_text(json.dumps(cas.normalise(PARSED)), encoding="utf-8")
    result = cas.CasConnector(path=path).fetch()
    assert result.available
    assert len(result.positions) == 3
    assert "2026-08-30" in result.note and "days old" in result.note


def test_decimal_strings_from_casparser_are_numbers_and_zero_is_zero():
    """casparser 1.x returns Decimals, which serialise to strings - and the
    string "0.000" is truthy. A closed scheme must still be dropped."""
    parsed = {"folios": [{"folio": "1", "schemes": [
        {"scheme": "Closed", "isin": "INF1", "close": "0.000",
         "valuation": {"date": "2026-08-01", "nav": "10.5", "value": "0.00"}},
        {"scheme": "Open", "isin": "INF2", "close": "1,234.500",
         "valuation": {"date": "2026-08-01", "nav": "10.0",
                       "value": "12345.00", "cost": "10000"}},
    ]}], "parse_warnings": ["INF2: unit balance mismatch"]}
    out = cas.normalise(parsed)
    assert [h["name"] for h in out["holdings"]] == ["Open"]
    assert out["holdings"][0]["units"] == 1234.5
    assert out["holdings"][0]["value"] == 12345.0
    assert out["parse_warnings"]


def test_parse_warnings_reach_the_note(tmp_path):
    path = tmp_path / "cas.json"
    data = cas.normalise(PARSED)
    data["parse_warnings"] = ["INF879O01027: unit balance mismatch"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "parse warning" in cas.CasConnector(path=path).fetch().note
