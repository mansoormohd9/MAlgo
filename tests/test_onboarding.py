"""
`onboarding.steps` - every "done" is computed from state, never ticked.
"""
from __future__ import annotations

from nifty_algo import onboarding
from nifty_algo.config import Config


def _ready(cfg):
    cfg.plan.monthly_expenses_inr = 40_000
    cfg.plan.w_india_equity = 100.0
    return cfg


def test_a_fresh_install_starts_at_connect():
    steps = onboarding.steps(Config())
    assert [s.page for s in steps] == [
        onboarding.CONNECT, onboarding.PLAN, onboarding.ALLOCATION,
        onboarding.SLEEVE, onboarding.FOREIGN]
    assert onboarding.next_step(steps).page == onboarding.CONNECT


def test_not_read_yet_is_not_the_same_as_holding_nothing():
    unread = onboarding.steps(Config(), holdings_count=None)[0]
    empty = onboarding.steps(Config(), holdings_count=0)[0]
    assert not unread.done and not empty.done
    assert unread.detail != empty.detail


def test_a_kite_login_or_a_holding_completes_connect():
    assert onboarding.steps(Config(), kite_authenticated=True)[0].done
    assert onboarding.steps(Config(), holdings_count=3)[0].done


def test_goals_need_expenses_and_a_100pct_split():
    cfg = Config()
    cfg.plan.monthly_expenses_inr = 40_000
    goals = onboarding.steps(cfg, holdings_count=1)[1]
    assert not goals.done and "target split" in goals.detail
    cfg.plan.w_india_equity = 100.0
    assert onboarding.steps(cfg, holdings_count=1)[1].done


def test_setup_is_complete_without_the_optional_steps():
    steps = onboarding.steps(_ready(Config()), holdings_count=1)
    assert steps[2].done                       # allocation
    assert not steps[3].done and steps[3].optional   # unfunded sleeve
    assert onboarding.next_step(steps) is None


def test_a_foreign_target_makes_the_lrs_pool_required():
    cfg = _ready(Config())
    cfg.plan.w_india_equity = 80.0
    cfg.plan.w_foreign_equity = 20.0
    steps = onboarding.steps(cfg, holdings_count=1)
    assert not steps[4].optional and not steps[4].done
    assert onboarding.next_step(steps).page == onboarding.FOREIGN
    cfg.capital.foreign_capital_inr = 100_000
    assert onboarding.next_step(onboarding.steps(cfg, holdings_count=1)) is None
