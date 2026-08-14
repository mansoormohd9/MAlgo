"""
Alert de-duplication, the risk gate, the kill switch, and pricing.

The dedupe tests matter operationally: a 5-minute bar re-evaluated every 15
seconds is 20 identical alerts per setup per channel. If suppression breaks,
the system becomes unusable long before it becomes wrong.
"""
from __future__ import annotations
from datetime import datetime, timedelta

import pytest

from nifty_algo.config import Config
from nifty_algo.alerts.base import TradeAlert, AlertKind, Notifier
from nifty_algo.alerts.dispatcher import AlertDispatcher
from nifty_algo.alerts.channels import InAppNotifier
from nifty_algo.journal import Journal
from nifty_algo.pricing import bs_price, bs_delta, synthetic_chain
from nifty_algo.risk import RiskEngine, RejectedOrder, ApprovedOrder


class CountingNotifier(Notifier):
    name = "inapp"

    def __init__(self):
        self.count = 0

    @property
    def configured(self) -> bool:
        return True

    def send(self, alert):
        self.count += 1
        return True, "ok"


class BrokenNotifier(Notifier):
    name = "telegram"

    @property
    def configured(self) -> bool:
        return True

    def send(self, alert):
        raise RuntimeError("channel exploded")


def entry_alert(ts: datetime, strategy="squeeze", strike=26_100) -> TradeAlert:
    return TradeAlert(
        kind=AlertKind.ENTRY, timestamp=ts,
        strategy_key=strategy, strategy_label=strategy,
        direction="long", option_type="CE", strike=strike,
        entry_premium=100.0, target_premium=151.3, stop_premium=74.4,
        quantity=65, lots=1, rupee_risk=1667.0, rupee_reward=3333.0,
    )


# ---------------------------------------------------------------- dedupe

def test_identical_alert_is_sent_once(cfg=None):
    cfg = Config()
    n = CountingNotifier()
    d = AlertDispatcher([n], cfg)
    ts = datetime(2026, 3, 10, 10, 30)

    for _ in range(20):                     # 20 refreshes inside one bar
        d.dispatch(entry_alert(ts))

    assert n.count == 1, "the same setup was sent more than once in one bar"


def test_cooldown_blocks_a_different_setup_from_the_same_strategy():
    cfg = Config()
    cfg.alerts.per_strategy_cooldown_minutes = 15
    n = CountingNotifier()
    d = AlertDispatcher([n], cfg)

    base = datetime(2026, 3, 10, 10, 30)
    d.dispatch(entry_alert(base, strike=26_100))
    d.dispatch(entry_alert(base + timedelta(minutes=5), strike=26_200))
    assert n.count == 1, "cooldown did not suppress a second alert 5 minutes later"

    d.dispatch(entry_alert(base + timedelta(minutes=20), strike=26_300))
    assert n.count == 2, "alert should be allowed once the cooldown expires"


def test_different_strategies_are_not_cross_suppressed():
    cfg = Config()
    n = CountingNotifier()
    d = AlertDispatcher([n], cfg)
    ts = datetime(2026, 3, 10, 10, 30)

    d.dispatch(entry_alert(ts, strategy="squeeze"))
    d.dispatch(entry_alert(ts, strategy="vwap_reclaim"))
    assert n.count == 2


def test_kill_switch_alert_is_never_suppressed():
    """A rate-limited kill switch is a kill switch that did not fire."""
    cfg = Config()
    n = CountingNotifier()
    d = AlertDispatcher([n], cfg)
    ts = datetime(2026, 3, 10, 10, 30)

    for _ in range(3):
        d.dispatch(TradeAlert(kind=AlertKind.KILL_SWITCH, timestamp=ts,
                              message="data gap"))
    assert n.count == 3


def test_suppression_is_distinguishable_from_having_no_channels():
    """
    Both cases used to return {}, and the engine could not tell them apart, so
    with every channel switched off it dropped genuine alerts from the UI feed
    and labelled them 'duplicate or within cooldown' - which was a lie about
    what had happened.
    """
    cfg = Config()
    n = CountingNotifier()
    d = AlertDispatcher([n], cfg)
    ts = datetime(2026, 3, 10, 10, 30)

    first = d.dispatch(entry_alert(ts))
    assert first == {"inapp": (True, "ok")}

    assert d.dispatch(entry_alert(ts)) is None, \
        "a suppressed alert must be distinguishable from a delivered one"

    cfg.alerts.enable_inapp = False
    d.reset_suppression()
    assert d.dispatch(entry_alert(ts)) == {}, \
        "no enabled channel is not the same thing as suppression"
    assert d.enabled_names() == []


def test_a_broken_channel_cannot_break_dispatch():
    cfg = Config()
    cfg.alerts.enable_telegram = True
    good, bad = CountingNotifier(), BrokenNotifier()
    d = AlertDispatcher([good, bad], cfg)

    results = d.dispatch(entry_alert(datetime(2026, 3, 10, 10, 30)))
    assert good.count == 1, "a failing channel stopped a working one"
    assert results["telegram"][0] is False
    assert "exploded" in results["telegram"][1]


def test_dedupe_key_distinguishes_bars():
    a = entry_alert(datetime(2026, 3, 10, 10, 30))
    b = entry_alert(datetime(2026, 3, 10, 10, 35))
    assert a.dedupe_key != b.dedupe_key


def test_alert_text_carries_the_actionable_numbers():
    text = entry_alert(datetime(2026, 3, 10, 10, 30)).as_text()
    for token in ("ENTRY", "TARGET", "STOP", "QTY", "never places an order"):
        assert token in text


# ---------------------------------------------------------------- risk gate

def test_stop_width_drives_the_delta_ceiling():
    """The core constraint: a wider stop forces a lower-delta strike."""
    e = RiskEngine(Config())
    tight = e.required_max_delta(40)
    wide = e.required_max_delta(120)
    assert tight > wide
    assert e.required_max_delta(60) == pytest.approx(1666.67 / 65 / 60, rel=1e-3)


def test_risk_engine_rejects_when_no_strike_fits():
    """A very wide stop needs a delta so low nothing qualifies."""
    cfg = Config()
    e = RiskEngine(cfg)
    chain = synthetic_chain(26_000, 5 / 365, "CE", cfg)
    decision = e.approve(chain, "CE", underlying_stop_points=900,
                         free_capital=100_000)
    assert isinstance(decision, RejectedOrder)


def test_approved_order_has_a_two_to_one_reward_risk():
    cfg = Config()
    e = RiskEngine(cfg)
    chain = synthetic_chain(26_000, 5 / 365, "CE", cfg)
    decision = e.approve(chain, "CE", underlying_stop_points=60,
                         free_capital=100_000)
    assert isinstance(decision, ApprovedOrder)

    risk = decision.entry_premium - decision.premium_stop
    reward = decision.premium_target - decision.entry_premium
    assert reward / risk == pytest.approx(2.0, rel=0.05)


def test_session_governors_stop_the_day_after_three_entries():
    from datetime import date, time
    cfg = Config()
    e = RiskEngine(cfg)
    e.start_day(date(2026, 3, 10))
    chain = synthetic_chain(26_000, 5 / 365, "CE", cfg)

    for _ in range(3):
        assert e.check_halt(now=time(10, 30)).value == "none"
        order = e.approve(chain, "CE", 60, e.capital)
        assert isinstance(order, ApprovedOrder)
        e.register_entry(order, "long")
        e.register_exit(e.session.open_positions[-1], -1667.0)

    # Three losses of 1R land on -Rs 5,001, a rupee past the day's opening
    # floor - which is the whole point of deriving risk-per-trade from the
    # session stop. Either governor tripping first is correct.
    assert e.check_halt(now=time(11, 0)).value in (
        "max_entries_reached", "give_back_floor_hit")


# ---------------------------------------------------------------- pricing

def test_black_scholes_delta_is_positive_for_both_types():
    """
    risk.py compares abs(delta) and expresses a bearish view by buying a PE,
    so both sides must come back positive or strike selection breaks.
    """
    ce = bs_delta(26_000, 26_000, 5 / 365, 0.14, 0.065, "CE")
    pe = bs_delta(26_000, 26_000, 5 / 365, 0.14, 0.065, "PE")
    assert 0 < ce < 1 and 0 < pe < 1
    assert ce + pe == pytest.approx(1.0, abs=0.01)


def test_otm_option_is_cheaper_and_lower_delta():
    atm = bs_price(26_000, 26_000, 5 / 365, 0.14, 0.065, "CE")
    otm = bs_price(26_000, 26_300, 5 / 365, 0.14, 0.065, "CE")
    assert otm < atm
    assert (bs_delta(26_000, 26_300, 5 / 365, 0.14, 0.065, "CE")
            < bs_delta(26_000, 26_000, 5 / 365, 0.14, 0.065, "CE"))


def test_synthetic_chain_quotes_pass_the_spread_gate():
    cfg = Config()
    for q in synthetic_chain(26_000, 5 / 365, "CE", cfg):
        assert q.spread_pct <= cfg.instrument.max_spread_pct_of_premium
        assert q.ask > q.bid


# ---------------------------------------------------------------- journal

def test_journal_is_append_only(tmp_path):
    j = Journal(tmp_path)
    j.write("signal", {"strategy": "squeeze"})
    j.write("rejected", {"strategy": "squeeze", "reason": "no_viable_strike"})

    records = j.read_day()
    assert [r["event"] for r in records] == ["signal", "rejected"]

    j.write("halt", {"reason": "max_entries_reached"})
    assert len(j.read_day()) == 3, "an earlier record was overwritten"


def test_journal_survives_unserialisable_payloads(tmp_path):
    from nifty_algo.risk import HaltReason
    j = Journal(tmp_path)
    j.write("halt", {"reason": HaltReason.KILL_SWITCH,
                     "when": datetime(2026, 3, 10, 10, 30)})
    rec = j.read_day()[0]
    assert rec["reason"] == "kill_switch"
