"""
The whole loop: alert -> confirm -> managed -> exited -> day over.

These are the tests that would have caught the two structural gaps the
management layer was built to close:

  1. The day target only ever gated NEW ENTRIES. If a runner alone carried the
     day to +10%, that is precisely the moment a position is still open - and
     nothing closed it.
  2. Nothing called register_exit in live code, so realised P&L stayed at zero
     and the governors could never trip at all.
"""
from datetime import date, datetime, time

import pytest

from nifty_algo.alerts.base import AlertKind, TradeAlert
from nifty_algo.alerts.dispatcher import AlertDispatcher
from nifty_algo.config import Config
from nifty_algo.data.base import StaticFrameFeed
from nifty_algo.engine import TradingEngine
from nifty_algo.governor import GovernorAction
from nifty_algo.journal import Journal
from nifty_algo.positions import ExitKind
from nifty_algo.risk import ApprovedOrder, OptionQuote, RiskEngine

from conftest import flat_bars


class _Collector:
    """A notifier that records instead of sending."""
    name = "inapp"

    def __init__(self):
        self.sent: list[TradeAlert] = []

    @property
    def configured(self):
        return True

    def send(self, alert):
        self.sent.append(alert)
        return True, "collected"


@pytest.fixture
def rig(tmp_path):
    cfg = Config()
    bars = flat_bars(60)
    collector = _Collector()
    dispatcher = AlertDispatcher([collector], cfg)
    engine = TradingEngine(StaticFrameFeed(bars), dispatcher, cfg,
                           journal=Journal(directory=str(tmp_path)))
    engine.risk.start_day(date(2026, 3, 10))
    return engine, collector, cfg


def _order(cfg, lots=2, premium=120.0):
    qty = lots * cfg.instrument.lot_size
    risk = cfg.capital.risk_per_trade_rupees
    quote = OptionQuote(strike=26_000, option_type="CE", premium=premium,
                        delta=0.42, bid=premium - 0.2, ask=premium + 0.2,
                        open_interest=500_000)
    return ApprovedOrder(
        quote=quote, lots=lots, quantity=qty, entry_premium=premium,
        premium_stop=premium - risk / qty,
        premium_target=premium + 2 * risk / qty,
        underlying_stop_points=30.0, rupee_risk=risk,
        rupee_reward=cfg.capital.reward_per_trade_rupees,
        runner_enabled=lots > 1,
    )


def _park(engine, cfg, lots=2):
    """Put an approved order in the pending map, as _evaluate_signal does."""
    from nifty_algo.strategy import Signal
    order = _order(cfg, lots=lots)
    alert = TradeAlert(kind=AlertKind.ENTRY, timestamp=datetime(2026, 3, 10, 10, 0),
                       strategy_key="level_break", direction="long",
                       option_type="CE", strike=26_000,
                       entry_premium=order.entry_premium,
                       underlying_price=26_000.0)
    signal = Signal(direction="long", option_type="CE", stop_points=30.0,
                    reason="test", confidence=0.8)
    engine._pending[alert.dedupe_key] = (order, signal, "level_break")
    return alert, order


# ---------------------------------------------------------------- confirmation

def test_nothing_is_entered_without_confirmation(rig):
    engine, _, cfg = rig
    _park(engine, cfg)
    assert engine.positions.positions == []
    assert engine.risk.governor.entries_taken == 0


def test_confirming_opens_a_managed_position(rig):
    engine, _, cfg = rig
    alert, order = _park(engine, cfg)

    pos = engine.confirm_entry(alert)
    assert pos is not None
    assert pos.state.lots_total == 2
    assert engine.risk.governor.entries_taken == 1
    assert engine.positions.open_lots == 2


def test_an_alert_cannot_be_confirmed_twice(rig):
    """A double-click must not double the position."""
    engine, _, cfg = rig
    alert, _ = _park(engine, cfg)

    assert engine.confirm_entry(alert) is not None
    assert engine.confirm_entry(alert) is None
    assert engine.positions.open_lots == 2


def test_confirmation_is_refused_once_the_day_is_over(rig):
    engine, _, cfg = rig
    alert, _ = _park(engine, cfg)
    engine.risk.register_exit(None, 10_500.0)      # day target reached

    assert engine.confirm_entry(alert) is None
    assert engine.positions.positions == []


def test_confirmation_is_refused_when_the_kill_switch_is_tripped(rig):
    engine, _, cfg = rig
    alert, _ = _park(engine, cfg)
    engine.risk.trip_kill_switch()
    assert engine.confirm_entry(alert) is None


# ---------------------------------------------------------------- management

def _advance(engine, pos, premium, now=time(11, 0)):
    """Drive one management pass at a chosen premium."""
    key = f"{pos.quote.strike}{pos.quote.option_type}"
    actions = engine.positions.update({key: premium}, atr=30.0, now=now)
    for a in actions:
        engine._handle_exit_action(a, datetime(2026, 3, 10, 11, 0), engine.state)
    return actions


def test_a_realised_exit_reaches_the_governor(rig):
    """
    The gap that mattered: register_exit was never called from live code, so
    realised P&L stayed at zero and no governor could ever trip.
    """
    engine, _, cfg = rig
    alert, _ = _park(engine, cfg)
    pos = engine.confirm_entry(alert)

    assert engine.risk.governor.realised_pnl == 0.0
    _advance(engine, pos, pos.premium_of(2.0))     # banks one lot at +2R
    assert engine.risk.governor.realised_pnl > 0.0
    assert engine.risk.governor.peak_realised_pnl > 0.0


def test_banking_a_partial_tightens_the_day_floor(rig):
    engine, _, cfg = rig
    alert, _ = _park(engine, cfg)
    pos = engine.confirm_entry(alert)

    before = engine.risk.governor.floor
    _advance(engine, pos, pos.premium_of(2.0))
    assert engine.risk.governor.floor > before


def test_exit_alerts_are_never_suppressed(rig):
    """
    Suppressing a duplicate ENTRY costs an opportunity. Suppressing an exit
    costs the trade.
    """
    engine, collector, cfg = rig
    alert, _ = _park(engine, cfg)
    pos = engine.confirm_entry(alert)

    _advance(engine, pos, pos.premium_of(2.0))
    kinds = [a.kind for a in collector.sent]
    assert AlertKind.PARTIAL_EXIT in kinds
    assert AlertKind.MANAGE in kinds


def test_a_runner_reaching_the_day_target_closes_everything(rig):
    """
    THE CASE THE ORIGINAL DESIGN COULD NOT HANDLE.

    check_halt() gated entries only. If trade one's runner alone carried the
    day past +10%, the position was still open at that instant and nothing
    closed it.
    """
    engine, collector, cfg = rig
    alert, _ = _park(engine, cfg)
    pos = engine.confirm_entry(alert)

    # Run to +14R (banks one lot at 2R, trails the runner to 13R), then give
    # back into the trail so the runner actually exits. Half a position closed
    # at +13R is 6.5R of the day's P&L - comfortably past the 10% target.
    _advance(engine, pos, pos.premium_of(14.0))
    assert engine.positions.open_lots == 1
    _advance(engine, pos, pos.premium_of(12.5))
    assert engine.positions.open_lots == 0

    verdict = engine.risk.governor.evaluate()
    assert verdict.action is GovernorAction.CLOSE_ALL
    assert engine.risk.governor.realised_pnl >= engine.risk.governor.target

    # And the engine's own pass must flatten and announce it.
    engine._manage_open_positions(engine.feed.get_bars(),
                                  datetime(2026, 3, 10, 11, 5), engine.state)
    assert engine.positions.open_lots == 0


def test_force_exit_time_flattens_everything(rig):
    engine, collector, cfg = rig
    alert, _ = _park(engine, cfg)
    pos = engine.confirm_entry(alert)

    actions = _advance(engine, pos, pos.premium_of(0.5), now=time(15, 10))
    assert [a.kind for a in actions] == [ExitKind.FORCE_EXIT]
    assert engine.positions.open_lots == 0


def test_a_single_lot_position_has_no_runner(rig):
    """The fallback path: 65 quantity cannot be halved, so 2R is a full exit."""
    engine, _, cfg = rig
    alert, order = _park(engine, cfg, lots=1)
    pos = engine.confirm_entry(alert)

    actions = _advance(engine, pos, pos.premium_of(2.0))
    assert ExitKind.TARGET_EXIT in [a.kind for a in actions]
    assert engine.positions.open_lots == 0


# ---------------------------------------------------------------- failure modes

class _BrokenFeed(StaticFrameFeed):
    def __init__(self, df, exc):
        super().__init__(df)
        self._exc = exc

    def get_bars(self, lookback_days: int = 5):
        raise self._exc


def _engine_with(feed, tmp_path):
    cfg = Config()
    return TradingEngine(feed, AlertDispatcher([_Collector()], cfg), cfg,
                         journal=Journal(directory=str(tmp_path)))


def test_a_missing_credential_blocks_but_does_not_latch(tmp_path):
    """
    An expired Kite token arrives every single morning. Latching the kill
    switch for it would mean a manual re-arm daily, which trains you to click
    through the warning that exists to stop you.
    """
    from nifty_algo.data.base import NotConfigured

    engine = _engine_with(
        _BrokenFeed(flat_bars(40), NotConfigured("token expired, run kite_login")),
        tmp_path)
    state = engine.run_once()

    assert not state.kill_switch
    assert not engine.risk.session.halted
    assert "token expired" in state.last_error


def test_a_real_feed_failure_still_latches_the_kill_switch(tmp_path):
    """The distinction has to cut both ways, or it is not a distinction."""
    from nifty_algo.data.base import FeedError

    engine = _engine_with(
        _BrokenFeed(flat_bars(40), FeedError("connection reset")), tmp_path)
    state = engine.run_once()

    assert state.kill_switch
    assert engine.risk.session.halted


def test_a_configuration_error_is_announced_once_not_every_poll(tmp_path):
    from nifty_algo.data.base import NotConfigured

    collector = _Collector()
    cfg = Config()
    engine = TradingEngine(
        _BrokenFeed(flat_bars(40), NotConfigured("no credentials")),
        AlertDispatcher([collector], cfg), cfg,
        journal=Journal(directory=str(tmp_path)))

    for _ in range(5):
        engine.run_once()
    assert len(collector.sent) == 1


def test_a_new_day_clears_positions_and_pending_orders(rig):
    engine, _, cfg = rig
    alert, _ = _park(engine, cfg)
    engine.confirm_entry(alert)
    _park(engine, cfg)

    engine._day = date(2026, 3, 9)
    engine.run_once(datetime(2026, 3, 10, 10, 0))
    assert engine.positions.positions == []
    assert engine._pending == {}
