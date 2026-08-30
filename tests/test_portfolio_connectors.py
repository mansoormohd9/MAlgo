"""
The holdings layer, and the one failure it exists to make impossible.

THE BUG THIS FILE IS ABOUT. `BrokerTransport._read` returns an empty default
when a broker read FAILS, which is right for a reconciler and catastrophic for
a risk report: an empty list read as "you hold nothing" produces a report with
no concentration, no correlation and no single-stock risk, on a fully invested
account. A clean bill of health manufactured by an expired token is the exact
shape of failure this repo refuses elsewhere (`protection_state`,
`news.available`), and most of the assertions below are about that one
distinction.

Nothing here touches the network or the real session file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nifty_algo.config import Config
from nifty_algo.portfolio import aggregate, base, registry
from nifty_algo.portfolio.base import ConnectorResult, Position
from nifty_algo.portfolio.kite import KiteConnector
from nifty_algo.portfolio.manual import ManualConnector


# ---------------------------------------------------------------- fakes

class _Broker:
    """
    Stands in for `KiteEquity`. `fail=True` reproduces the real failure shape:
    an empty list AND an incremented `read_failures`, exactly as `_read` does.
    """

    def __init__(self, rows=None, fail=False, authenticated=True):
        self.rows = rows or []
        self.fail = fail
        self.read_failures = 0
        self.session = type("S", (), {"authenticated": authenticated})()

    def holdings(self):
        if self.fail:
            self.read_failures += 1
            return []
        return self.rows


class _Holding:
    """The `HoldingView` shape the connector reads."""

    def __init__(self, symbol, quantity=10, t1=0, avg=100.0, last=110.0,
                 exchange="NSE"):
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.t1_quantity = t1
        self.average_price = avg
        self.last_price = last
        self.pnl = 0.0

    @property
    def total_quantity(self):
        return self.quantity + self.t1_quantity


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.swing.cache_dir = str(tmp_path)
    cfg.portfolio.manual_path = str(tmp_path / "manual_positions.csv")
    return cfg


def _pos(key="india:INFY", qty=10.0, last=110.0, ccy="INR", **kw) -> Position:
    market, symbol = key.split(":")
    return Position(key=key, symbol=symbol, market=market, quantity=qty,
                    average_price=kw.pop("avg", 100.0), last_price=last,
                    currency=ccy, source=kw.pop("source", "manual"), **kw)


# ------------------------------------------------- the failed-read distinction

def test_a_failed_holdings_read_is_not_an_empty_portfolio(tmp_path):
    """
    THE test in this file. A broker read that failed and a broker read that
    returned nothing produce the same empty list, and must not produce the
    same result.
    """
    connector = KiteConnector(cfg=_cfg(tmp_path), broker=_Broker(fail=True))
    result = connector.fetch()

    assert result.available is False
    assert result.positions == []
    assert "token" in result.note.lower()


def test_an_account_with_no_holdings_is_available_and_empty(tmp_path):
    connector = KiteConnector(cfg=_cfg(tmp_path), broker=_Broker(rows=[]))
    result = connector.fetch()

    assert result.available is True
    assert result.positions == []
    assert "not a failed read" in result.note


def test_unavailable_cannot_be_built_without_a_reason():
    """`available` defaults False, and the only constructor for it demands a
    note - so a falsy result can never reach a report unexplained."""
    assert ConnectorResult(source="x").available is False
    with pytest.raises(TypeError):
        ConnectorResult.unavailable("x")            # no reason given


def test_a_failed_source_makes_the_whole_snapshot_incomplete(tmp_path):
    cfg = _cfg(tmp_path)
    snapshot = aggregate.PortfolioSnapshot(
        positions=[_pos()],
        results=[ConnectorResult.ok("manual", [_pos()]),
                 ConnectorResult.unavailable("kite", "token expired")],
        value_inr={"india:INFY": 1100.0},
    )
    assert snapshot.complete is False
    # ...and the percentage is withheld rather than computed on half a book.
    assert snapshot.weight("india:INFY") is None
    assert any("kite" in c for c in snapshot.caveats())


def test_an_enabled_but_unconfigured_connector_is_unavailable(tmp_path):
    """
    Kite is unconfigured most weekdays - its token dies overnight. A book that
    silently shrank to the manual file on those days would report weights
    against a fraction of the account.
    """
    cfg = _cfg(tmp_path)
    snapshot = aggregate.load(cfg, connectors=["kite"])

    assert snapshot.complete is False
    assert snapshot.failed_sources
    assert "not configured" in snapshot.failed_sources[0].note


# ---------------------------------------------------------------- the registry

def test_an_unknown_connector_raises_rather_than_defaulting():
    """A typo that silently returned the Indian broker would produce a report
    that is internally consistent and about the wrong account."""
    with pytest.raises(registry.UnknownConnector):
        registry.get("zerodah")


def test_every_registered_connector_satisfies_the_protocol():
    for key in registry.keys():
        connector = registry.get(key)
        assert isinstance(connector, base.PortfolioConnector)
        assert connector.key == key


def test_ibkr_is_registered_and_says_what_it_needs():
    """The stub exists so the protocol is proven against two brokers. It must
    answer without raising, and it must name what implementing it needs."""
    result = registry.get("ibkr").fetch()
    assert result.available is False
    assert "ib_insync" in result.note
    assert "manual_positions.csv" in result.note


def test_a_registered_but_unenabled_connector_is_not_a_failure(tmp_path):
    """
    Otherwise the IBKR stub would mark every snapshot in the repo incomplete
    forever, and `complete` would stop meaning anything.
    """
    snapshot = aggregate.load(_cfg(tmp_path), connectors=["manual"])
    assert snapshot.complete is True
    assert "ibkr" in snapshot.skipped
    assert "kite" in snapshot.skipped


# ---------------------------------------------------------------- kite mapping

def test_t1_shares_are_counted(tmp_path):
    """The day after a fill, `quantity` is 0 and `t1_quantity` holds it all -
    which reads exactly like the buy never happened."""
    broker = _Broker(rows=[_Holding("INFY", quantity=0, t1=6)])
    result = KiteConnector(cfg=_cfg(tmp_path), broker=broker).fetch()
    assert result.positions[0].quantity == 6


def test_a_closed_out_zero_quantity_row_is_not_a_position(tmp_path):
    broker = _Broker(rows=[_Holding("INFY", quantity=0, t1=0)])
    result = KiteConnector(cfg=_cfg(tmp_path), broker=broker).fetch()
    assert result.positions == []
    assert result.available is True


def test_a_zero_last_price_falls_back_to_cost(tmp_path):
    """Before the first tick of the day Kite can report 0. Valued at zero the
    line drops out of every weight in the report without an error anywhere."""
    broker = _Broker(rows=[_Holding("INFY", quantity=5, avg=100.0, last=0.0)])
    result = KiteConnector(cfg=_cfg(tmp_path), broker=broker).fetch()
    assert result.positions[0].last_price == 100.0


def test_kite_positions_are_market_qualified(tmp_path):
    broker = _Broker(rows=[_Holding("INFY")])
    result = KiteConnector(cfg=_cfg(tmp_path), broker=broker).fetch()
    assert result.positions[0].key == "india:INFY"


# ---------------------------------------------------------------- manual file

def test_a_missing_manual_file_is_a_real_answer(tmp_path):
    """Nothing recorded is true and reportable. It is not a failed read."""
    result = ManualConnector(path=tmp_path / "nope.csv").fetch()
    assert result.available is True
    assert result.positions == []


def test_an_unreadable_manual_file_is_not_an_empty_portfolio(tmp_path, monkeypatch):
    path = tmp_path / "manual_positions.csv"
    path.write_text("market,symbol\n", encoding="utf-8")

    def boom(*a, **kw):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "open", boom)
    result = ManualConnector(path=path).fetch()
    assert result.available is False
    assert "empty portfolio" in result.note


def test_the_value_form_records_a_fund_balance(tmp_path):
    """A fund balance has no share count anybody knows, and inventing one
    would put a fictional quantity into a concentration table."""
    path = tmp_path / "m.csv"
    path.write_text(
        "market,symbol,quantity,average_price,last_price,value,cost,"
        "currency,asset_class\n"
        "us,SPUS,,,,12000,10000,USD,etf\n", encoding="utf-8")

    position = ManualConnector(path=path).fetch().positions[0]
    assert position.value_native == 12000.0
    assert position.cost_native == 10000.0
    assert position.asset_class == "etf"


def test_a_bad_row_is_counted_out_loud_not_dropped(tmp_path):
    """A line that vanishes silently is a position missing from every weight."""
    path = tmp_path / "m.csv"
    path.write_text(
        "market,symbol,quantity,last_price,currency\n"
        "india,INFY,10,110,INR\n"
        "india,TCS,10,110,\n", encoding="utf-8")

    result = ManualConnector(path=path).fetch()
    assert len(result.positions) == 1
    assert "skipped" in result.note and "TCS" in result.note


def test_a_row_with_neither_quantity_nor_value_is_refused(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("market,symbol,currency\nindia,INFY,INR\n", encoding="utf-8")
    result = ManualConnector(path=path).fetch()
    assert result.positions == []
    assert "quantity" in result.note


# ------------------------------------------------------------- keys and merging

def test_the_same_ticker_on_two_exchanges_is_two_positions(tmp_path):
    """A bare ticker collision is not a crash - it is one company's weight
    computed from another company's price."""
    path = tmp_path / "m.csv"
    path.write_text(
        "market,symbol,quantity,last_price,currency\n"
        "india,INFY,10,1500,INR\n"
        "us,INFY,10,20,USD\n", encoding="utf-8")

    keys = {p.key for p in ManualConnector(path=path).fetch().positions}
    assert keys == {"india:INFY", "us:INFY"}


def test_one_company_in_two_accounts_is_one_exposure():
    """Concentration asks how much of a name you own, not how many brokers
    you own it through."""
    merged = aggregate._combine(
        _pos(qty=10.0, last=110.0, avg=100.0, source="kite"),
        _pos(qty=10.0, last=110.0, avg=120.0, source="manual"),
    )
    assert merged.quantity == 20.0
    assert merged.average_price == pytest.approx(110.0)   # blended, not either
    assert "kite" in merged.source and "manual" in merged.source


# ---------------------------------------------------------------- fx, closed

def test_an_unconvertible_currency_is_excluded_and_named(tmp_path, monkeypatch):
    """
    `fx.py` refuses to guess a rate, and that contract is kept here: the line
    is left out of the rupee total, named, and it makes the snapshot
    incomplete. Without conversion a US line is ~88x too large and looks
    entirely ordinary.
    """
    from nifty_algo.swing import fx as fx_mod

    cfg = _cfg(tmp_path)
    path = Path(cfg.portfolio.manual_path)
    path.write_text(
        "market,symbol,quantity,last_price,currency\n"
        "india,INFY,10,1500,INR\n"
        "us,AAPL,10,200,USD\n", encoding="utf-8")

    real = fx_mod.rate_inr_per          # bound before patching, or `refuse`
                                        # would call itself

    def refuse(currency, config, force_refresh=False):
        if currency == "INR":
            return real("INR", config)
        raise fx_mod.FxUnavailable("no rate today")

    monkeypatch.setattr(aggregate.fx_mod, "rate_inr_per", refuse)
    snapshot = aggregate.load(cfg, connectors=["manual"])

    assert snapshot.total_inr == pytest.approx(15_000.0)   # INR line only
    assert "USD" in snapshot.unconvertible
    assert snapshot.complete is False
    assert snapshot.weight("india:INFY") is None
    assert any("USD" in c for c in snapshot.caveats())


def test_weights_are_available_once_the_book_is_whole(tmp_path):
    cfg = _cfg(tmp_path)
    Path(cfg.portfolio.manual_path).write_text(
        "market,symbol,quantity,last_price,currency\n"
        "india,INFY,10,1000,INR\n"
        "india,TCS,10,3000,INR\n", encoding="utf-8")

    snapshot = aggregate.load(cfg, connectors=["manual"])
    assert snapshot.complete is True
    assert snapshot.weight("india:INFY") == pytest.approx(0.25)
    assert snapshot.weight("india:TCS") == pytest.approx(0.75)
