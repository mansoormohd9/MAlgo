"""
The stepwise console: Connect -> Money & goals -> Allocation, the research lab
behind a toggle, and every pot edited in exactly one place.

The assertions are about the consolidation itself, because that is what
regressed silently before: the pot existed on four pages under three save
rules, and the one that never saved leaked into whichever page saved next.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import seed_offline_broker, sign_in
from streamlit.testing.v1 import AppTest

from nifty_algo import onboarding
from nifty_algo.config import Config
from nifty_algo.ui import nav, page_connect, page_plan

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _no_real_settings_writes(monkeypatch):
    """Every save seam these pages use, pointed nowhere - see CLAUDE.md,
    'A page test must never write data/settings.json'."""
    for mod in (nav, page_plan, page_connect):
        monkeypatch.setattr(mod, "save_settings", lambda: None)


def _cfg(tmp_path, lines: str = "") -> Config:
    cfg = Config()
    cfg.portfolio.connectors = ("manual",)          # nothing that dials out
    cfg.portfolio.manual_path = str(tmp_path / "manual.csv")
    if lines:
        (tmp_path / "manual.csv").write_text(
            "market,symbol,name,quantity,average_price,last_price,value,"
            "cost,currency,asset_class,account\n" + lines, encoding="utf-8")
    return cfg


def _open(cfg, page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    sign_in(at)
    at.session_state["cfg"] = cfg
    seed_offline_broker(at, cfg)
    at.run()
    assert not at.exception, _why(at)
    if page:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, _why(at)
    return at


def _why(at) -> str:
    return "; ".join(str(e.value)[:400] for e in (at.exception or []))


# ---------------------------------------------------------------- navigation

def test_a_fresh_install_lands_on_connect_with_the_lab_hidden(tmp_path):
    at = _open(_cfg(tmp_path))
    radio = at.sidebar.radio[0]
    assert radio.value == onboarding.CONNECT
    assert radio.options[:5] == [onboarding.CONNECT, onboarding.PLAN,
                                 onboarding.ALLOCATION, onboarding.SLEEVE,
                                 onboarding.FOREIGN]
    for lab_page in nav.LAB_PAGES:
        assert lab_page not in radio.options
    assert at.title[0].value == "Connect your accounts"


def test_the_lab_toggle_adds_the_lab_pages(tmp_path):
    cfg = _cfg(tmp_path)
    at = _open(cfg)
    at.sidebar.toggle(key="show_lab").set_value(True).run()
    assert not at.exception, _why(at)
    assert cfg.ui.show_research_lab is True
    assert "Live alerts" in at.sidebar.radio[0].options

    at.sidebar.radio[0].set_value("Strategies").run()
    at.sidebar.toggle(key="show_lab").set_value(False).run()
    # Switching the lab off while on a lab page must not strand the radio.
    assert not at.exception, _why(at)
    assert at.sidebar.radio[0].value in nav.main_pages()


def test_every_main_page_renders(tmp_path):
    cfg = _cfg(tmp_path, "india,EPF,EPF,,,,300000,,INR,fixed_income,\n")
    at = _open(cfg)
    for page in nav.main_pages():
        if page == "Research":        # runs a fact pack over the network
            continue
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page}: {_why(at)}"


# ---------------------------------------------------------------- one editor

def test_no_page_but_money_and_goals_edits_a_pot(tmp_path):
    """The four editors are now one. Walk every page and look."""
    cfg = _cfg(tmp_path)
    cfg.ui.show_research_lab = True
    at = _open(cfg)
    pot_keys = {"cap_factor", "cap_foreign", "cap_swing", "cap_option"}
    for page in nav.options(cfg):
        if page in ("Research", "Backtest", "Live alerts", "Daily brief"):
            continue    # network, or the option engine's feed
        at.sidebar.radio[0].set_value(page).run()
        labels = [(w.key or "", (w.label or "").lower())
                  for w in at.number_input]
        pots = [k for k, lab in labels if k in pot_keys
                or ("capital" in lab or "pot" in lab)]
        if page == onboarding.PLAN:
            assert set(pots) >= pot_keys
        else:
            assert not pots, f"{page} still edits a pot: {pots}"


def test_the_sleeve_pot_survives_two_changes_and_saves_on_save(tmp_path):
    """
    The regression that used to live in `test_ui_page_sleeve.py`, moved with
    the control: two changes in succession is the smallest sequence that
    fails without an explicit widget key.
    """
    cfg = _cfg(tmp_path)
    at = _open(cfg, onboarding.PLAN)
    for pot in (200_000.0, 225_000.0, 250_000.0):
        at.number_input(key="cap_factor").set_value(pot).run()
        assert at.number_input(key="cap_factor").value == pot
    assert cfg.capital.factor_capital_inr == 0.0          # not yet committed
    at.button(key="save_plan").click().run()
    assert not at.exception, _why(at)
    assert cfg.capital.factor_capital_inr == 250_000.0


def test_the_starting_point_fills_a_100pct_split_only_when_asked(tmp_path):
    cfg = _cfg(tmp_path)
    at = _open(cfg, onboarding.PLAN)
    assert at.number_input(key="plan_w_india").value == 0.0
    at.button(key="plan_starting_point").click().run()
    at.button(key="save_plan").click().run()
    assert not at.exception, _why(at)
    assert sum(cfg.plan.targets().values()) == pytest.approx(100.0)


def test_switching_halal_on_turns_the_sleeve_screen_on(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.factor.halal_screened is False
    at = _open(cfg, onboarding.PLAN)
    at.toggle(key="plan_halal").set_value(True).run()
    at.button(key="save_plan").click().run()
    assert cfg.plan.halal_only is True
    assert cfg.factor.halal_screened is True


def test_strategies_no_longer_leaks_a_pot_into_the_next_save(tmp_path):
    """
    Strategies wrote `starting_capital` straight onto the shared config with
    no save, so the next save anywhere persisted it. It shows the pot
    read-only now; nothing on it can move a persisted field.
    """
    cfg = _cfg(tmp_path)
    cfg.ui.show_research_lab = True
    cfg.capital.starting_capital = 100_000.0
    at = _open(cfg, "Strategies")
    for w in at.number_input:
        assert "capital" not in (w.label or "").lower()
    assert cfg.capital.starting_capital == 100_000.0


# ---------------------------------------------------------------- allocation

def test_allocation_reads_the_manual_balances_and_plans_new_money(tmp_path):
    cfg = _cfg(tmp_path,
               "india,NIFTYBEES,Nifty ETF,,,,600000,,INR,etf,\n"
               "india,EPF,EPF,,,,300000,,INR,fixed_income,\n"
               "india,SAVINGS,Bank,,,,100000,,INR,cash,\n")
    cfg.plan.w_india_equity = 70.0
    cfg.plan.w_fixed_income = 20.0
    cfg.plan.w_cash = 10.0
    cfg.plan.monthly_investment_inr = 50_000.0
    cfg.plan.monthly_expenses_inr = 25_000.0
    at = _open(cfg, onboarding.ALLOCATION)
    body = " ".join(str(m.value) for m in at.metric)
    assert "1,000,000" in body                              # net worth
    tables = [df.value for df in at.dataframe]
    assert any("Put in (₹)" in t.columns for t in tables)
    # 4 months of cash against 6 months of expenses: says so.
    text = " ".join(m.value for m in at.markdown)
    assert "4.0 months" in text


def test_the_manual_editor_writes_a_file_the_connector_reads(tmp_path):
    import pandas as pd
    from nifty_algo.portfolio.manual import ManualConnector

    path = tmp_path / "manual.csv"
    df = pd.DataFrame([{"market": "india", "symbol": "PPF", "name": "PPF",
                        "value": 250000.0, "currency": "",
                        "asset_class": "fixed_income"}])
    page_connect._save_manual(path, df)
    result = ManualConnector(path=path).fetch()
    assert result.available
    (ppf,) = result.positions
    assert ppf.value_native == 250_000.0
    assert ppf.currency == "INR"                 # filled for an Indian line
    assert ppf.asset_class == "fixed_income"
