"""
Step 2 - Money & goals: THE ONE PLACE ANY MONEY NUMBER IS EDITED.

WHY THIS PAGE EXISTS. Before it, the option pot could be edited on Settings
(saved on a button) and on Strategies (never saved, then persisted by
whichever page saved next, because every page writes the same `DEFAULT`); the
foreign pot on Settings (button) and Portfolio (saved on every keystroke); the
sleeve pot only on the sleeve page (saved on Run scan); and Backtest kept a
fourth copy. Four editors, three save rules, one number each.

Now every pot, the investor profile and the target split are edited here and
committed together by one Save button. Every other page shows them read-only
and links back.

THE DEFERRED ASSIGNMENT IS KEPT, and for the reason Settings first had it:
`save_settings()` serialises the whole config, so any other control that saves
- the DDPI box, the research-lab toggle - would otherwise persist a number
you were only halfway through typing here. So the widgets hold the typed
values, `cfg` changes only on Save, and anything previewing a typed number
reads the WIDGET, not `cfg` (the source-of-truth split that once made this
page divide by zero).

EVERY WIDGET HAS AN EXPLICIT `key`. Streamlit hashes `value=` into the element
id otherwise, and a control that reads config for its default then loses every
second change - see CLAUDE.md, "EVERY WIDGET ON THIS PAGE CARRIES AN EXPLICIT
`key`".
"""
from __future__ import annotations

import streamlit as st

from .components import banner
from .state import (get_config, get_equity_broker, peek_snapshot,
                    save_settings, settings_notes)
from .theme import get_palette
from .. import onboarding
from ..planning import allocation as alloc

#: (bucket, PlanConfig field, widget key)
TARGET_FIELDS = (
    (alloc.INDIA_EQUITY, "w_india_equity", "plan_w_india"),
    (alloc.FOREIGN_EQUITY, "w_foreign_equity", "plan_w_foreign"),
    (alloc.BUCKET_GOLD, "w_gold", "plan_w_gold"),
    (alloc.BUCKET_FIXED, "w_fixed_income", "plan_w_fixed"),
    (alloc.BUCKET_CASH, "w_cash", "plan_w_cash"),
)


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Money & goals")
    st.caption(f"Step 2 of 5. What the money is for, how it should be split, "
               f"and how much each book may use. Nothing here is applied "
               f"until you press **Save**.")

    for note in settings_notes():
        banner(note, p.warning, "⚠")

    typed = {}
    typed.update(_profile(cfg))
    typed.update(_targets(cfg, p))
    typed.update(_pots(cfg, typed, p))

    changed = _differs(cfg, typed)
    if changed:
        st.caption("Unsaved changes — press Save to apply them.")
    if st.button("Save", key="save_plan",
                 type="primary" if changed else "secondary"):
        _commit(cfg, typed)
        save_settings()
        st.success("Saved to `data/settings.json` — gitignored, and it "
                   "survives a restart.")
        st.rerun()

    st.divider()
    _orders(cfg, p)


# ---------------------------------------------------------------- you

def _profile(cfg) -> dict:
    st.subheader("You")
    plan = cfg.plan
    c1, c2, c3, c4 = st.columns(4)
    out = {
        "age": int(c1.number_input(
            "Age", min_value=0, max_value=100, step=1, value=int(plan.age),
            key="plan_age")),
        "monthly_expenses_inr": float(c2.number_input(
            "Monthly expenses (₹)", min_value=0.0, step=5_000.0,
            value=float(plan.monthly_expenses_inr), key="plan_expenses",
            help="Sets the emergency fund. Include rent/EMI.")),
        "emergency_months": float(c3.number_input(
            "Emergency fund (months)", min_value=0.0, max_value=24.0,
            step=1.0, value=float(plan.emergency_months), key="plan_em_months",
            help="Held in cash or a liquid account before anything is "
                 "invested. Six is the usual floor for a salaried job.")),
        "monthly_investment_inr": float(c4.number_input(
            "Invest each month (₹)", min_value=0.0, step=5_000.0,
            value=float(plan.monthly_investment_inr), key="plan_monthly_inv",
            help="Your SIP/deposit. Allocation shows where it should go.")),
    }
    c1, c2, c3 = st.columns(3)
    out["tolerated_drawdown_pct"] = c1.slider(
        "Whole-portfolio fall you would sit through (%)", 5, 50,
        int(round(plan.tolerated_drawdown_pct * 100)), 1,
        key="plan_tolerance",
        help="Not the sleeve's fall - the WHOLE net worth's. This is what "
             "sizes the factor sleeve below.") / 100.0
    out["rebalance_band_pp"] = float(c2.number_input(
        "Rebalance band (± pp)", min_value=1.0, max_value=20.0, step=1.0,
        value=float(plan.rebalance_band_pp), key="plan_band",
        help="A class is flagged only when it drifts further than this from "
             "target. Tighter bands mean more trading and more tax."))
    out["halal_only"] = bool(c3.toggle(
        "Halal-only investing", value=bool(plan.halal_only), key="plan_halal",
        help="Turns the Shariah screen on for the monthly sleeve when saved, "
             "and flags interest-bearing classes in your split."))
    return out


# ---------------------------------------------------------------- the split

def _targets(cfg, p) -> dict:
    st.subheader("Target split")
    st.caption("Percent of net worth. These are yours to choose — the page "
               "will not fill them in unless you ask for a starting point.")

    if st.button("Use a starting point", key="plan_starting_point"):
        for bucket, _field, key in TARGET_FIELDS:
            st.session_state[key] = float(alloc.STARTING_POINT[bucket])
        st.rerun()

    cols = st.columns(len(TARGET_FIELDS))
    out = {}
    for col, (bucket, field, key) in zip(cols, TARGET_FIELDS):
        if key not in st.session_state:
            st.session_state[key] = float(getattr(cfg.plan, field))
        out[field] = float(col.number_input(
            alloc.LABELS[bucket], min_value=0.0, max_value=100.0, step=5.0,
            key=key))
    total = sum(out.values())
    if abs(total - 100.0) < 0.1:
        st.caption(f"Sums to **100%**.")
    else:
        banner(f"Sums to <b>{total:.1f}%</b>. Allocation will not compute "
               f"drift until the split is exactly 100%.", p.warning, "⚠")
    st.caption(
        "The starting point (55 / 20 / 10 / 10 / 5) is a common shape for "
        "someone in their early 30s with a long horizon: mostly equity, some "
        "diversification abroad and into gold, a small stable leg. It is "
        "arithmetic about a typical profile, not advice about yours.")
    return out


# ---------------------------------------------------------------- the pots

def _pots(cfg, typed: dict, p) -> dict:
    st.subheader("Pots")
    st.caption(
        "Money each book may use. A pot at ₹0 stands its book down rather "
        "than borrowing another pot's balance.")
    cap = cfg.capital
    c1, c2 = st.columns(2)
    out = {
        "factor_capital_inr": float(c1.number_input(
            "Monthly sleeve (₹)", min_value=0.0, step=25_000.0,
            value=float(cap.factor_capital_inr), key="cap_factor",
            help="The momentum sleeve's pot. Sized against the share of net "
                 "worth your tolerance allows - see the line below.")),
        "foreign_capital_inr": float(c2.number_input(
            "Foreign / LRS (₹)", min_value=0.0, step=50_000.0,
            value=float(cap.foreign_capital_inr), key="cap_foreign",
            help="Remitted under LRS and sitting in the foreign broker.")),
    }
    _sleeve_cap_note(out["factor_capital_inr"],
                     typed.get("tolerated_drawdown_pct",
                               cfg.plan.tolerated_drawdown_pct), p)

    if cfg.ui.show_research_lab:
        with st.expander("Research lab pots", expanded=False):
            st.caption("The option and swing books lost to their own nulls "
                       "and live in the research lab. Their pots stay here "
                       "so nothing about money is edited anywhere else.")
            c1, c2 = st.columns(2)
            out["starting_capital"] = float(c1.number_input(
                "Intraday options (₹)", min_value=0.0, step=10_000.0,
                value=float(cap.starting_capital), key="cap_option"))
            out["swing_capital_inr"] = float(c2.number_input(
                "Indian swing (₹)", min_value=0.0, step=5_000.0,
                value=float(cap.swing_capital_inr), key="cap_swing"))
            _pot_note(cfg, out["swing_capital_inr"], p)
    return out


def _sleeve_cap_note(pot: float, tolerance: float, p) -> None:
    """
    The recommended ceiling, from F2b's arithmetic and a real net worth.

    Shown, never applied: the tolerance is yours and the pot is yours. What the
    page refuses to do is let a pot above the ceiling go by without saying so.
    """
    share = alloc.sleeve_share(tolerance)
    dd = alloc.planning_drawdown()
    snapshot = peek_snapshot()
    net = snapshot.total_inr if (snapshot is not None
                                 and snapshot.complete) else None
    cap = alloc.sleeve_cap(net, tolerance)
    line = (f"At a {tolerance:.0%} tolerated fall and the sleeve's "
            f"{dd:.0%} planning drawdown, the sleeve may be at most "
            f"**{share:.0%} of net worth**")
    if cap is None:
        st.caption(line + ". Net worth is unknown until every connected "
                   "account answers — open Allocation to read it.")
        return
    st.caption(line + f": **₹{cap:,.0f}** of ₹{net:,.0f}.")
    if pot > cap:
        banner(f"The sleeve pot is ₹{pot:,.0f}, above the ₹{cap:,.0f} your "
               f"tolerance supports. A repeat of the measured drawdown would "
               f"take the whole portfolio down about "
               f"{pot / net * dd:.0%}.", p.warning, "⚠")


def _pot_note(cfg, swing: float, p) -> None:
    """
    What the typed swing pot buys - the note Settings used to carry.

    THE COUNT DOES NOT DEPEND ON THE POT. Cash per position is `risk / stop%`
    and risk is a fixed fraction of the pot, so the positions a pot funds is
    `stop% / risk%`. Reads the TYPED pot, never `cfg` - see module docstring.
    """
    if swing <= 0:
        banner("The Indian swing pot is ₹0, so that scan will <b>stand "
               "down</b> rather than size off another pot.", p.warning, "⚠")
        return
    pct = cfg.capital.risk_per_trade_pct
    risk = swing * pct
    if risk <= 0:
        banner("Risk per trade works out to <b>0%</b> of the pot - check "
               "<code>session_stop_pct</code> and "
               "<code>max_entries_per_session</code> in "
               "<code>config.py</code>.", p.critical, "⛔")
        return
    n = cfg.swing.top_n
    st.caption(
        f"₹{swing:,.0f} gives **₹{risk:,.0f} of risk per trade** "
        f"({pct:.2%} of the pot). How many of the {n} tickets it can pay for "
        f"depends on the stop, not the pot: a **tighter stop buys more "
        f"shares**.")
    rows = ["| Stop | Deployed per position | Tickets funded |",
            "|---|---|---|"]
    for stop_pct in (0.03, 0.05, 0.08):
        per_position = risk / stop_pct
        fits = swing / per_position
        verdict = (f"**all {n}**" if fits >= n
                   else f"{fits:.1f} of {n} — the rest refused for want of cash")
        rows.append(f"| {stop_pct:.0%} | ₹{per_position:,.0f} | {verdict} |")
    st.markdown("\n".join(rows))


# ---------------------------------------------------------------- commit

_PLAN_KEYS = ("age", "monthly_expenses_inr", "emergency_months",
              "monthly_investment_inr", "tolerated_drawdown_pct",
              "rebalance_band_pp", "halal_only", "w_india_equity",
              "w_foreign_equity", "w_gold", "w_fixed_income", "w_cash")
_CAPITAL_KEYS = ("factor_capital_inr", "foreign_capital_inr",
                 "starting_capital", "swing_capital_inr")


def _differs(cfg, typed: dict) -> bool:
    for k in _PLAN_KEYS:
        if k in typed and typed[k] != getattr(cfg.plan, k):
            return True
    for k in _CAPITAL_KEYS:
        if k in typed and typed[k] != getattr(cfg.capital, k):
            return True
    return False


def _commit(cfg, typed: dict) -> None:
    turning_halal_on = typed.get("halal_only") and not cfg.plan.halal_only
    for k in _PLAN_KEYS:
        if k in typed:
            setattr(cfg.plan, k, typed[k])
    for k in _CAPITAL_KEYS:
        if k in typed:
            setattr(cfg.capital, k, typed[k])
    # One direction only, and only at the moment the preference is switched
    # on: the sleeve keeps its own toggle, and a later deliberate change there
    # is respected rather than overwritten on every save.
    if turning_halal_on:
        cfg.factor.halal_screened = True


# ---------------------------------------------------------------- orders

def _orders(cfg, p) -> None:
    """DDPI and the live-order switch, moved here from Settings unchanged."""
    st.subheader("Zerodha account")
    eq = cfg.equity_broker
    broker = get_equity_broker()

    ddpi = st.checkbox(
        "DDPI (or POA) is active on my Zerodha account",
        value=eq.ddpi_active, key="ddpi_active",
        help="Console → Profile. Without it, every delivery sell needs a "
             "CDSL TPIN authorisation that expires nightly.")
    if ddpi != eq.ddpi_active:
        eq.ddpi_active = ddpi
        # Persisted immediately: `settings_store.apply_to` refuses to restore
        # live orders on an account without DDPI, and reads this flag.
        save_settings()

    if not ddpi:
        banner(
            "<b>Without DDPI a resting stop cannot execute on its own.</b> "
            "Every delivery sell needs a CDSL TPIN authorisation valid for one "
            "trading day, and Kite shows a stale trigger as active either way. "
            "Enabling DDPI is a free one-time e-sign.", p.critical, "⛔")

    st.markdown("**Live order placement (swing book, research lab)**")
    if eq.dry_run:
        st.caption("Dry run. Every payload is journalled exactly as it would "
                   "be sent. The monthly sleeve never places orders either "
                   "way — it produces a checklist.")
        typed = st.text_input(
            "Type GO LIVE to enable real orders", key="go_live_confirm",
            help="Deliberately awkward. This is the only thing between the "
                 "app and your money.")
        if typed.strip().upper() == "GO LIVE":
            if st.button("Enable live orders", type="primary",
                         key="enable_live"):
                eq.dry_run = False
                save_settings()
                st.rerun()
    else:
        st.error("Live. Arming a ticket places a real order at Zerodha.")
        if st.button("Back to dry run", key="disable_live"):
            eq.dry_run = True
            save_settings()
            st.rerun()
    st.caption(f"{broker.mode_label}.")


def next_link() -> None:
    """Used by other pages: a read-only pot plus the way back here."""
    st.caption(f"Edit on **{onboarding.PLAN}**.")
