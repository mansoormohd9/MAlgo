"""
The deployed sleeve: what ships, and what must NOT fall back.

`data/cache/` is gitignored, so a clone - Streamlit Cloud clones from GitHub -
arrives with no bars and no balance sheets. Measured on such a clone before
this existed: the sleeve raised on the bars, and with the halal screen on it
rejected all 60 shortlisted names for "cannot verify" and produced a 0-name
book, which reads as a very strict screen rather than as starving.

The tests here pin the two halves that make deployment safe rather than merely
working: the LIVE path degrades to the committed slice and says so, and the
BACKTEST still refuses.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from nifty_algo.config import DEFAULT
from nifty_algo.factor import deployed_bars as dep
from nifty_algo.factor import drawdown as fd
from nifty_algo.factor import sleeve as sl
from nifty_algo.swing import fundamentals as fund

_ROOT = Path(__file__).resolve().parent.parent


def _frame(symbols, sessions=500, start=1000.0):
    days = pd.bdate_range("2024-01-01", periods=sessions)
    parts = []
    for i, sym in enumerate(symbols):
        close = [start + i * 10 + j for j in range(sessions)]
        parts.append(pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": [1_000_000] * sessions, "symbol": sym},
            index=days))
    return pd.concat(parts)


# ----------------------------------------------------------- the slice itself

def test_the_slice_keeps_only_what_the_live_scan_reads():
    """
    `FactorUniverse` reads `close` and `volume` and nothing else; ATR is not
    allocated unless asked for. Carrying open/high/low would double the file
    to ship a figure the live sleeve never computes.
    """
    sliced = dep.build(_frame(["A", "B"]), sessions=100)
    assert set(sliced.columns) == {"close", "volume", "symbol"}
    assert len(sliced.index.unique()) == 100
    assert sliced["close"].dtype == "float32"


def test_the_slice_takes_the_MOST_RECENT_sessions():
    full = _frame(["A"], sessions=500)
    sliced = dep.build(full, sessions=100)
    assert sliced.index.max() == full.index.max()
    assert sliced.index.min() > full.index.min()


def test_a_short_frame_is_not_padded_or_truncated():
    sliced = dep.build(_frame(["A"], sessions=30), sessions=400)
    assert len(sliced.index.unique()) == 30


def test_the_slice_round_trips_with_its_benchmark(tmp_path):
    frame = _frame(["A", "B", "__BENCHMARK__:^NSEI"], sessions=120)
    dep.write(frame, tmp_path / "slice.parquet", sessions=100)
    bars, bench, info = dep.load(tmp_path / "slice.parquet")
    assert set(bars) == {"A", "B"}
    assert bench is not None, "the regime gate needs the benchmark"
    assert info.sessions == 100
    assert info.symbols == 2
    assert "deployed slice" in info.describe()


def test_a_missing_slice_returns_None_rather_than_raising(tmp_path):
    """The caller has already failed to find the full cache and wants to
    report both misses in one message, not the second one only."""
    assert dep.load(tmp_path / "absent.parquet") == (None, None, None)


# ------------------------------------------- what ships is actually committed

def test_the_committed_slice_exists_and_carries_the_benchmark():
    path = _ROOT / dep.SLICE_PATH
    if not path.exists():
        pytest.skip("deployed slice not built on this machine")
    bars, bench, info = dep.load(path)
    assert bars and bench is not None
    assert info.sessions >= 300, (
        f"{info.sessions} sessions is below min_history_sessions - every name "
        f"would be rejected for short history on a deploy")


# ------------------------------------- the live path falls back, the backtest does NOT

def test_the_live_loader_falls_back_to_the_slice_and_names_it(tmp_path,
                                                              monkeypatch):
    frame = _frame(["A", "B", "__BENCHMARK__:^NSEI"], sessions=120)
    slice_file = tmp_path / "slice.parquet"
    dep.write(frame, slice_file, sessions=100)

    cfg = replace(DEFAULT, factor=replace(
        DEFAULT.factor, cache_dir=str(tmp_path), cache_name="absent.parquet"))
    monkeypatch.setattr(dep, "SLICE_PATH", str(slice_file))

    bars, bench = sl.load_bars(cfg)
    assert set(bars) == {"A", "B"}
    assert bench is not None
    assert "deployed slice" in sl.LAST_BARS_SOURCE, (
        "a scan on the slice must be able to say so on screen")


def test_the_BACKTEST_still_refuses_when_only_the_slice_exists(tmp_path,
                                                               monkeypatch):
    """
    THE MOST IMPORTANT TEST IN THIS FILE.

    `drawdown.load` feeds F1-F5. A backtest that quietly ran on 400 sessions
    would report a complete, plausible, badly wrong CAGR - and every recorded
    number in CLAUDE.md would silently stop being reproducible. The fallback
    lives in `sleeve.load_bars`, the LIVE path, and must never reach here.
    """
    frame = _frame(["A", "__BENCHMARK__:^NSEI"], sessions=120)
    dep.write(frame, tmp_path / "slice.parquet", sessions=100)
    monkeypatch.setattr(dep, "SLICE_PATH", str(tmp_path / "slice.parquet"))

    missing = replace(DEFAULT.factor, cache_dir=str(tmp_path),
                      cache_name="absent.parquet")
    with pytest.raises(FileNotFoundError):
        fd.load(missing)


def test_both_misses_are_reported_in_one_message(tmp_path, monkeypatch):
    cfg = replace(DEFAULT, factor=replace(
        DEFAULT.factor, cache_dir=str(tmp_path), cache_name="absent.parquet"))
    monkeypatch.setattr(dep, "SLICE_PATH", str(tmp_path / "no_slice.parquet"))
    with pytest.raises(FileNotFoundError) as excinfo:
        sl.load_bars(cfg)
    message = str(excinfo.value)
    assert "No factor cache" in message
    assert "no deployed slice" in message
    assert "deployed_bars" in message


def test_the_full_cache_wins_when_both_exist(tmp_path, monkeypatch):
    """Locally nothing changes: the slice is never read when the real cache
    is there, so a stale slice cannot quietly replace current bars."""
    full = _frame(["A", "B", "C", "__BENCHMARK__:^NSEI"], sessions=500)
    full.to_parquet(tmp_path / "full.parquet")
    dep.write(_frame(["A", "__BENCHMARK__:^NSEI"], sessions=120),
              tmp_path / "slice.parquet", sessions=100)
    monkeypatch.setattr(dep, "SLICE_PATH", str(tmp_path / "slice.parquet"))

    cfg = replace(DEFAULT, factor=replace(
        DEFAULT.factor, cache_dir=str(tmp_path), cache_name="full.parquet"))
    bars, _bench = sl.load_bars(cfg)
    assert set(bars) == {"A", "B", "C"}
    assert sl.LAST_BARS_SOURCE == "full cache"


# ---------------------------------------- a failed refresh keeps the old sheet

def _sheet(symbol, fetched_at):
    return fund.Fundamentals(
        symbol=symbol, total_assets=1000.0, total_debt=100.0,
        cash_and_investments=50.0, receivables=25.0, market_cap=5000.0,
        balance_sheet_date="2026-03-31", yahoo_sector="Industrials",
        yahoo_industry="Widgets", fetched_at=fetched_at)


def test_a_failed_refresh_keeps_the_cached_balance_sheet(tmp_path,
                                                         monkeypatch):
    """
    STALE DATA IS NOT THE SAME AS NO DATA.

    This used to assign the fetch result unconditionally, so an unreachable
    Yahoo replaced a usable sheet with an empty one carrying an `error` - and
    the halal screen treats absent data as CANNOT VERIFY, which is a reject.
    Measured on a fresh clone with the network blocked: 60 shortlisted names,
    60 rejected, 0 picks. A sheet carries `balance_sheet_date`, so an old one
    is visible AS old; nothing is visible about one that was thrown away.
    """
    from nifty_algo.swing.universe import Stock
    from nifty_algo.swing import markets as markets_mod

    old = (datetime.now() - timedelta(days=90)).isoformat(timespec="seconds")
    market = markets_mod.factor_market(DEFAULT)
    fund._write_cache(tmp_path / fund.CACHE_NAME,
                      {market.qualified("ACME"): _sheet("ACME", old)})

    cfg = replace(DEFAULT, swing=replace(DEFAULT.swing,
                                         cache_dir=str(tmp_path)))
    monkeypatch.setattr(fund, "_fetch_one", lambda stock: fund.Fundamentals(
        symbol=stock.symbol, error="network blocked"))

    stock = Stock(symbol="ACME", name="ACME", sector="", industry="",
                  yf_ticker="ACME.NS")
    got = fund.load_fundamentals([stock], cfg, market)

    assert got["ACME"].has_balance_sheet, (
        "the good cached sheet was discarded by a failed refresh")
    assert got["ACME"].yahoo_sector == "Industrials"
    assert got["ACME"].error is None


def test_a_SUCCESSFUL_refresh_still_replaces_the_cached_sheet(tmp_path,
                                                              monkeypatch):
    """The fallback must not freeze the cache - a good fetch always wins."""
    from nifty_algo.swing.universe import Stock
    from nifty_algo.swing import markets as markets_mod

    old = (datetime.now() - timedelta(days=90)).isoformat(timespec="seconds")
    market = markets_mod.factor_market(DEFAULT)
    fund._write_cache(tmp_path / fund.CACHE_NAME,
                      {market.qualified("ACME"): _sheet("ACME", old)})

    fresh = _sheet("ACME", datetime.now().isoformat(timespec="seconds"))
    fresh.yahoo_sector = "Technology"
    cfg = replace(DEFAULT, swing=replace(DEFAULT.swing,
                                         cache_dir=str(tmp_path)))
    monkeypatch.setattr(fund, "_fetch_one", lambda stock: fresh)

    stock = Stock(symbol="ACME", name="ACME", sector="", industry="",
                  yf_ticker="ACME.NS")
    got = fund.load_fundamentals([stock], cfg, market)
    assert got["ACME"].yahoo_sector == "Technology"


def test_the_committed_fundamentals_are_used_only_when_the_cache_is_absent(
        tmp_path, monkeypatch):
    """
    NEVER MERGED. A half-populated live cache topped up from a committed file
    would make "when was this balance sheet read" unanswerable, and the next
    write would persist the committed rows as though just fetched.
    """
    committed = tmp_path / "committed.json"
    fund._write_cache(committed, {"factor_india:SHIPPED": _sheet(
        "SHIPPED", datetime.now().isoformat(timespec="seconds"))})
    monkeypatch.setattr(fund, "DEPLOYED_NAME", str(committed))

    # No live cache -> the committed copy is read.
    assert "SHIPPED" in {k.split(":")[-1] for k in
                         fund._read_cache_with_fallback(tmp_path / "none.json")}

    # A live cache that exists -> used ALONE, committed rows absent.
    live = tmp_path / "live.json"
    fund._write_cache(live, {"factor_india:LIVE": _sheet(
        "LIVE", datetime.now().isoformat(timespec="seconds"))})
    keys = {k.split(":")[-1] for k in fund._read_cache_with_fallback(live)}
    assert keys == {"LIVE"}
