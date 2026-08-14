"""
The chain view as a thing that states a suggestion, not just a table.

The brief used to fill entry / target / stop on exactly one row and leave the
other twenty-four blank, render CE and PE with no indication of which (if
either) was actionable, and say nothing at all about whether the session was
even open. All three made a reference price look like an instruction.
"""
from __future__ import annotations
from datetime import date

from nifty_algo.brief import build_chain_view, review
from nifty_algo.config import Config
from nifty_algo.data.chain import ChainProvider
from nifty_algo.journal import Journal
from nifty_algo.risk import ApprovedOrder, RiskEngine


def _view(**kwargs):
    cfg = Config()
    return build_chain_view(
        spot=26_000.0, option_type="CE", stop_points=30.0, cfg=cfg,
        chain_provider=ChainProvider(cfg), risk=RiskEngine(cfg),
        today=date(2026, 3, 10), **kwargs)


def test_every_buyable_row_says_what_buying_it_would_cost():
    view = _view()
    buyable = [r for r in view.rows if r.viable]
    assert buyable, "the synthetic chain should offer at least one tradeable strike"

    for r in buyable:
        assert r.entry is not None and r.target is not None and r.stop is not None
        assert r.rupee_risk and r.rupee_reward and r.outlay
        assert r.lots >= 1 and r.quantity >= 1
        # Sanity: a long option's stop is below the entry and target above it.
        assert r.stop < r.entry < r.target

    for r in view.rows:
        if not r.viable:
            assert r.gates, "a row with no entry must say which gate refused it"
            assert r.entry is None


def test_the_row_numbers_are_the_risk_engines_own():
    """
    Not a parallel calculation. Every row comes back through approve(), so the
    pick's row and the decision object cannot disagree.
    """
    view = _view()
    d = view.approved
    assert isinstance(d, ApprovedOrder)

    pick = view.pick
    assert pick is not None and pick.strike == d.quote.strike
    assert pick.entry == d.entry_premium
    assert pick.target == d.premium_target
    assert pick.stop == d.premium_stop
    assert pick.rupee_risk == d.rupee_risk


def test_runner_ups_are_tradeable_losers_ranked_by_delta():
    view = _view()
    ups = view.runner_ups
    assert all(r.viable and not r.selected for r in ups)
    assert len(ups) <= 3
    assert ups == sorted(ups, key=lambda r: abs(r.delta), reverse=True)
    if view.approved:
        assert all(abs(r.delta) <= abs(view.approved.quote.delta) for r in ups)


def test_a_blocked_session_is_never_described_as_a_thing_to_do():
    """
    approve() knows about strikes and budgets and nothing about the clock. It
    will size an order perfectly at 15:20 on expiry day, so the view has to
    carry the answer or a reference price reads as an instruction.
    """
    open_label, _ = _view().status()
    assert open_label == "REFERENCE ONLY"

    blocked = _view(entry_permitted=False, halt_reason="outside_entry_window")
    label, explanation = blocked.status()
    assert label == "BLOCKED"
    assert "outside entry window" in explanation
    # The pricing is still correct - it just cannot be acted on.
    assert isinstance(blocked.decision, ApprovedOrder)


def test_the_headline_is_a_sentence_not_an_enum():
    view = _view()
    headline = view.headline()
    assert headline.startswith("BUY ")
    assert str(view.approved.quote.strike) in headline
    assert "qty" in headline
    # The CLI prints this on a Windows console; non-ASCII mojibakes there.
    headline.encode("ascii")

    empty = _view(entry_permitted=True)
    empty.decision = None
    assert empty.headline().startswith("Nothing to buy")


def test_the_frame_keeps_numbers_numeric_so_it_can_be_sorted():
    frame = _view().to_frame()
    assert frame["Spread%"].dtype.kind == "f"
    assert frame["Strike"].dtype.kind in "iu"
    assert "Why not" in frame.columns and "Action" in frame.columns


# ---------------------------------------------------------------- review

def test_review_shows_rejections(tmp_path):
    """
    `review()` filtered on the event name "rejection" while the journal writes
    "rejected", so the REJECTED block the module docstring advertises never
    rendered once.
    """
    journal = Journal(directory=str(tmp_path))
    journal.rejection("level_break", "no_viable_strike", "delta ceiling 0.19")

    day = date.today()
    lines = review(day, journal_dir=str(tmp_path))
    text = "\n".join(lines)

    assert "REJECTED" in text
    assert "level_break" in text
    assert "no_viable_strike" in text
    assert "delta ceiling 0.19" in text


def test_review_reads_the_flattened_journal_record(tmp_path):
    """`Journal.write` flattens payload to the top level - there is no
    'payload' key, and the code used to look one up."""
    journal = Journal(directory=str(tmp_path))
    journal.write("entry_confirmed", {
        "strike": 26_000, "option_type": "CE", "lots": 2, "entry": 120.0,
        "stop": 94.4, "target": 171.3, "runner": True, "dry_run": True,
    })

    text = "\n".join(review(date.today(), journal_dir=str(tmp_path)))
    assert "26000CE" in text
    assert "2 lot(s) @ 120.0" in text
    assert "[DRY RUN]" in text
