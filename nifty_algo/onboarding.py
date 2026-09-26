"""
Where you are in setting the console up, and what to do next.

WHY THIS EXISTS. The console grew one book at a time, so the order things must
happen in was written down nowhere: the Kite login lived inside Trade book, the
sleeve pot on the sleeve page, the foreign pot on two pages with two save
rules. A first-time user had to know the app to use it.

The steps are the order the money actually flows - connect the accounts that
hold it, say what it is for and how to split it, see where it stands against
that split, then fund the books. Each step's `done` is computed from state the
app already has, never from a "completed" flag you tick: a checklist you can
tick without doing the thing is decoration.

Pure: the caller hands in whether Kite is authenticated and the snapshot, so
this module imports no Streamlit and touches no network.
"""
from __future__ import annotations

from dataclasses import dataclass

from .planning import allocation as alloc

# Page labels, spelled once. `app.py` builds its navigation from these, so a
# "next step" link can never point at a page that was renamed.
CONNECT = "1 · Connect"
PLAN = "2 · Money & goals"
ALLOCATION = "3 · Allocation"
SLEEVE = "4 · Monthly sleeve"
FOREIGN = "5 · US / LRS"


@dataclass(frozen=True)
class Step:
    page: str
    title: str
    done: bool
    detail: str
    optional: bool = False


def steps(cfg, *, kite_authenticated: bool = False,
          holdings_count: int | None = None) -> list[Step]:
    """
    The five steps, each with whether it is done and why not.

    `holdings_count` is None when no snapshot has been read this session -
    which is different from zero, and is reported as "not checked yet" rather
    than as "you hold nothing".
    """
    plan = cfg.plan
    cap = cfg.capital

    if kite_authenticated or (holdings_count or 0) > 0:
        parts = []
        if kite_authenticated:
            parts.append("Kite logged in")
        if holdings_count:
            parts.append(f"{holdings_count} holding(s) read")
        connect = Step(CONNECT, "Connect accounts", True, ", ".join(parts))
    elif holdings_count is None:
        connect = Step(CONNECT, "Connect accounts", False,
                       "Log in to Kite or record your balances.")
    else:
        connect = Step(CONNECT, "Connect accounts", False,
                       "Nothing answered with a holding yet.")

    missing = []
    if plan.monthly_expenses_inr <= 0:
        missing.append("monthly expenses")
    if not alloc.targets_valid(plan):
        missing.append(f"target split (at {alloc.targets_total(plan):.0f}%)")
    goals = Step(PLAN, "Money & goals", not missing,
                 "Expenses and a 100% target split set." if not missing
                 else "Set " + " and ".join(missing) + ".")

    allocation = Step(
        ALLOCATION, "Check the allocation", connect.done and goals.done,
        "Drift and next-rupee plan available." if connect.done and goals.done
        else "Needs steps 1 and 2.")

    sleeve = Step(
        SLEEVE, "Monthly sleeve", cap.factor_capital_inr > 0,
        f"Pot ₹{cap.factor_capital_inr:,.0f}." if cap.factor_capital_inr > 0
        else "Optional - fund the pot in step 2 to size picks.",
        optional=True)

    wants_foreign = plan.w_foreign_equity > 0
    foreign = Step(
        FOREIGN, "US / LRS", (not wants_foreign) or cap.foreign_capital_inr > 0,
        ("No foreign target set." if not wants_foreign
         else f"Foreign pool ₹{cap.foreign_capital_inr:,.0f}."
         if cap.foreign_capital_inr > 0
         else "Your plan holds foreign equity - record the LRS pool in step 2."),
        optional=not wants_foreign)

    return [connect, goals, allocation, sleeve, foreign]


def next_step(all_steps: list[Step]) -> Step | None:
    """The first required step not done, or None when setup is complete."""
    for s in all_steps:
        if not s.done and not s.optional:
            return s
    return None
