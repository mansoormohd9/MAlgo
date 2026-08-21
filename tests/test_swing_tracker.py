"""
The follow-up on past picks.

This is the part that makes the scanner falsifiable, so the thing most worth
testing is that it cannot flatter itself: the scan day's own bar must not be
allowed to trigger the trade it was derived from, and a bar that covers both
the stop and the target has to be counted as a loss rather than resolved in
the pick's favour.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from nifty_algo.journal import Journal
from nifty_algo.swing import tracker

SCAN_DAY = date(2026, 8, 10)
TODAY = date(2026, 8, 25)


def bars(rows, start=datetime(2026, 8, 10)) -> pd.DataFrame:
    """Daily bars from (high, low) pairs, one per day from `start`."""
    idx = [start + timedelta(days=i) for i in range(len(rows))]
    return pd.DataFrame(
        [{"open": (h + l) / 2, "high": h, "low": l, "close": (h + l) / 2,
          "volume": 1_000.0} for h, l in rows],
        index=pd.DatetimeIndex(idx))


def pick(**kw) -> dict:
    base = {
        "symbol": "ACME", "setup": "breakout", "setup_label": "Level breakout",
        "entry": 100.0, "stop": 95.0, "target": 115.0, "quantity": 10,
        "scanned_on": SCAN_DAY.isoformat(),
        "valid_until": (SCAN_DAY + timedelta(days=5)).isoformat(),
    }
    base.update(kw)
    return base


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal")


def record(journal, picks, day=SCAN_DAY):
    journal.write(tracker.EVENT,
                  {"scanned_on": day.isoformat(), "picks": picks}, day=day)


def only(journal, frame) -> tracker.PickOutcome:
    summary = tracker.open_picks(journal, {"ACME": frame}, lookback_days=90,
                                 today=TODAY)
    assert len(summary.outcomes) == 1
    return summary.outcomes[0]


# ---------------------------------------------------------------- outcomes

def test_entry_never_reached_before_expiry(journal):
    record(journal, [pick()])
    # Six sessions, none of which trades up to 100.
    out = only(journal, bars([(99, 96)] * 6))
    assert out.outcome == tracker.OUTCOME_EXPIRED
    assert "never reached" in out.note


def test_target_hit_pays_the_full_reward(journal):
    record(journal, [pick()])
    out = only(journal, bars([(99, 96), (101, 98), (116, 110)]))
    assert out.outcome == tracker.OUTCOME_TARGET
    assert out.triggered_on == date(2026, 8, 11)
    assert out.r_multiple == pytest.approx(3.0)      # (115-100)/(100-95)
    assert out.rupees == pytest.approx(150.0)        # 3R x 5 points x 10 shares


def test_stopped_out_is_exactly_minus_one_r(journal):
    record(journal, [pick()])
    out = only(journal, bars([(99, 96), (101, 98), (99, 94)]))
    assert out.outcome == tracker.OUTCOME_STOPPED
    assert out.r_multiple == pytest.approx(-1.0)
    assert out.rupees == pytest.approx(-50.0)


def test_a_bar_covering_both_levels_is_counted_as_a_loss(journal):
    """
    Daily data cannot say which was touched first. Resolving the coin flip in
    the pick's favour is exactly how a record flatters itself.
    """
    record(journal, [pick()])
    out = only(journal, bars([(99, 96), (120, 90)]))
    assert out.outcome == tracker.OUTCOME_AMBIGUOUS
    assert out.r_multiple == pytest.approx(-1.0)
    assert "cannot say which came first" in out.note


def test_an_open_position_is_marked_to_the_last_close(journal):
    record(journal, [pick()])
    out = only(journal, bars([(99, 96), (101, 98), (105, 102)]))
    assert out.outcome == tracker.OUTCOME_OPEN
    assert out.triggered_on == date(2026, 8, 11)
    assert out.r_multiple == pytest.approx((103.5 - 100.0) / 5.0)


def test_waiting_for_an_entry_inside_the_window_is_open_not_expired(journal):
    recent = TODAY - timedelta(days=2)
    record(journal, [pick(scanned_on=recent.isoformat(),
                          valid_until=(recent + timedelta(days=5)).isoformat())],
           day=recent)
    frame = bars([(99, 96), (98, 95)],
                 start=datetime(recent.year, recent.month, recent.day))
    out = only(journal, frame)
    assert out.outcome == tracker.OUTCOME_OPEN
    assert "waiting for" in out.note


# ---------------------------------------------------------------- honesty

def test_the_scan_days_own_bar_cannot_trigger_the_pick(journal):
    """
    The ticket was derived from that day's close. Letting the same bar fill it
    would be a look-ahead dressed up as a result.
    """
    record(journal, [pick()])
    # Day one - the scan day itself - trades right through entry and target.
    out = only(journal, bars([(200, 90), (99, 97)]))
    assert out.outcome != tracker.OUTCOME_TARGET
    assert out.triggered_on is None


def test_the_same_pick_recorded_twice_counts_once(journal):
    """Re-running the scan in an afternoon must not double the position."""
    record(journal, [pick()])
    record(journal, [pick()])
    summary = tracker.open_picks(journal, {"ACME": bars([(99, 96)] * 6)},
                                 lookback_days=90, today=TODAY)
    assert len(summary.outcomes) == 1


def test_a_symbol_with_no_bars_is_reported_not_guessed(journal):
    record(journal, [pick()])
    summary = tracker.open_picks(journal, {}, lookback_days=90, today=TODAY)
    assert summary.outcomes[0].outcome == tracker.OUTCOME_UNKNOWN
    assert summary.outcomes[0].r_multiple is None


def test_records_outside_the_lookback_are_not_read(journal):
    record(journal, [pick()], day=date(2025, 1, 6))
    summary = tracker.open_picks(journal, {"ACME": bars([(99, 96)] * 6)},
                                 lookback_days=30, today=TODAY)
    assert summary.outcomes == []
    assert "No scans recorded" in summary.note


def test_an_empty_journal_is_not_an_error(journal):
    summary = tracker.open_picks(journal, {}, today=TODAY)
    assert summary.outcomes == []


# ---------------------------------------------------------------- summary

def test_the_headline_counts_only_closed_trades(journal):
    record(journal, [pick(symbol="ACME")])
    summary = tracker.open_picks(journal, {"ACME": bars([(99, 96), (116, 98)])},
                                 lookback_days=90, today=TODAY)
    assert summary.wins == 1
    assert len(summary.closed) == 1
    assert summary.net_r == pytest.approx(3.0)
    assert "1 closed" in summary.headline()


def test_the_headline_says_so_when_nothing_has_closed(journal):
    record(journal, [pick()])
    summary = tracker.open_picks(journal, {"ACME": bars([(101, 98), (102, 99)])},
                                 lookback_days=90, today=TODAY)
    assert "too early to say" in summary.headline()


def test_record_scan_writes_what_the_replay_can_read(journal, tmp_path):
    """The two halves of this module have to agree on the record shape."""
    from nifty_algo.swing.scanner import ScanResult

    result = ScanResult(scanned_on=SCAN_DAY, universe_size=100,
                        eligible_size=40, prices_note="stub")
    tracker.record_scan(journal, result)

    written = [r for r in journal.read_day(SCAN_DAY)
               if r.get("event") == tracker.EVENT]
    assert len(written) == 1
    assert written[0]["scanned_on"] == SCAN_DAY.isoformat()
    assert written[0]["picks"] == []
