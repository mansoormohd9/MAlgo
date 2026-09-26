"""
The IBKR connector, read-only through the Flex Web Service.

No test here reaches IBKR: the parser is fed a committed fixture, and the
two-request protocol is driven through an injected fetcher. `conftest` blanks
`IBKR_FLEX_TOKEN` for the whole suite, so a test that wants the connector
configured sets it itself with monkeypatch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nifty_algo.portfolio import ibkr
from nifty_algo.portfolio.base import CASH, EQUITY, ETF

FIXTURE = Path(__file__).parent / "fixtures" / "ibkr_flex_statement.xml"

SEND_OK = ("<FlexStatementResponse><Status>Success</Status>"
           "<ReferenceCode>REF1</ReferenceCode>"
           "<Url>https://example.invalid/GetStatement</Url>"
           "</FlexStatementResponse>")
GENERATING = ("<FlexStatementResponse><Status>Warn</Status>"
              "<ErrorCode>1019</ErrorCode>"
              "<ErrorMessage>Statement generation in progress</ErrorMessage>"
              "</FlexStatementResponse>")


def _by_symbol(positions):
    return {p.symbol: p for p in positions}


# ---------------------------------------------------------------- parsing

def test_the_statement_parses_into_native_currency_positions():
    positions, notes = ibkr.parse_statement(FIXTURE.read_text(encoding="utf-8"))
    got = _by_symbol(positions)

    assert got["SPUS"].quantity == pytest.approx(120.5)    # fractional kept
    assert got["SPUS"].currency == "USD"                   # never converted here
    assert got["SPUS"].asset_class == ETF
    assert got["SPUS"].account == "U1234567"
    assert got["AAPL"].asset_class == EQUITY
    assert got["ISDW"].market == "uk"                      # LSE listing


def test_lot_rows_and_options_are_not_counted_twice_or_at_all():
    positions, notes = ibkr.parse_statement(FIXTURE.read_text(encoding="utf-8"))
    assert [p.symbol for p in positions].count("AAPL") == 1
    assert not any("CALL" in p.name for p in positions)
    assert any("skipped" in n for n in notes)


def test_cash_is_counted_once_not_with_its_base_summary():
    positions, _ = ibkr.parse_statement(FIXTURE.read_text(encoding="utf-8"))
    cash = [p for p in positions if p.asset_class == CASH]
    assert len(cash) == 1
    assert cash[0].value_native == pytest.approx(1500.0)


# ---------------------------------------------------------------- protocol

def test_the_two_requests_poll_while_the_statement_generates():
    body = FIXTURE.read_text(encoding="utf-8")
    replies = iter([SEND_OK, GENERATING, body])
    calls = []

    def fetch(url, params):
        calls.append((url, params))
        return next(replies)

    out = ibkr.request_statement(fetch, "tok", "999", sleep=lambda s: None)
    assert out == body
    assert calls[0][1]["q"] == "999"          # the query id
    assert calls[1][1]["q"] == "REF1"         # then the reference code


def test_a_refused_request_raises_rather_than_returning_nothing():
    refused = ("<FlexStatementResponse><Status>Fail</Status>"
               "<ErrorMessage>Token has expired.</ErrorMessage>"
               "</FlexStatementResponse>")
    with pytest.raises(RuntimeError, match="expired"):
        ibkr.request_statement(lambda u, p: refused, "tok", "999")


# ---------------------------------------------------------------- connector

@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv(ibkr.TOKEN_ENV, "tok")
    monkeypatch.setenv(ibkr.QUERY_ENV, "999")


def test_a_failed_fetch_is_unavailable_not_an_empty_account(configured, tmp_path):
    def boom(url, params):
        raise ConnectionError("no route")
    conn = ibkr.IbkrConnector(cache_path=tmp_path / "flex.xml", fetcher=boom)
    result = conn.fetch()
    assert result.available is False
    assert "not as an empty account" in result.note


def test_a_fresh_statement_is_cached_and_reused(configured, tmp_path):
    body = FIXTURE.read_text(encoding="utf-8")
    replies = iter([SEND_OK, body])
    calls = []

    def fetch(url, params):
        calls.append(url)
        return next(replies)

    path = tmp_path / "flex.xml"
    first = ibkr.IbkrConnector(cache_path=path, fetcher=fetch).fetch()
    assert first.available and len(first.positions) == 5
    assert path.exists()

    def never(url, params):
        raise AssertionError("the cache should have answered")
    second = ibkr.IbkrConnector(cache_path=path, fetcher=never).fetch()
    assert second.available and "cached" in second.note
    assert len(calls) == 2
