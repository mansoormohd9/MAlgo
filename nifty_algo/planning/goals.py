"""
What the money is FOR, and whether the monthly investment gets there.

WHY THIS EXISTS. The step is called Money & goals and held no goal. At 32 the
outcome is decided by the savings rate and the horizon far more than by any
signal in this repo - `allocation.py`'s own docstring says so - and there was
nothing that turned "₹40,000 a month" into "the house in 2031 is ₹6 lakh
short".

DETERMINISTIC, IN TODAY'S RUPEES, AND NOTHING FITTED. Targets are typed in
today's money and grown at a REAL (after-inflation) return you choose, so no
inflation assumption is hidden in the arithmetic. One return, typed, not a
per-class forecast: a projection that looked precise would be read as a
prediction. `expected_real_return_pct` of 0 means NOT SET and the projection
is withheld - the code does not pick a return for you. Type 0.1 for "about
zero".

Goals live in `data/goals.csv` (gitignored - they are yours, like
`manual_positions.csv`). `project` is pure; `load` / `save` are the only file
access here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..paths import at_root

GOALS_PATH = "data/goals.csv"
COLUMNS = ("name", "target_inr", "year", "spend")
#: No projection runs further than this - beyond it the arithmetic is noise.
MAX_YEARS = 60


@dataclass(frozen=True)
class Goal:
    name: str
    #: In TODAY'S rupees - grown at the real return, never at a nominal one.
    target_inr: float
    year: int
    #: True: the money is spent that year (a house deposit) and leaves the
    #: pot. False: a corpus you need to HAVE (retirement) - checked, kept.
    spend: bool = True


@dataclass(frozen=True)
class GoalCheck:
    goal: Goal
    #: What the pot holds at the end of the goal's year, before paying it.
    projected_inr: float
    short_inr: float
    #: Extra per month from now that would close the gap on its own,
    #: ignoring how that money would also help the goals after it.
    extra_monthly_inr: float

    @property
    def on_track(self) -> bool:
        return self.short_inr <= 0


@dataclass(frozen=True)
class Projection:
    years: list[int]
    wealth_inr: list[float]
    checks: list[GoalCheck]


def _monthly_rate(real_return_pct: float) -> float:
    return (1.0 + real_return_pct / 100.0) ** (1.0 / 12.0) - 1.0


def _annuity_fv(rate: float, months: int) -> float:
    """Future value of 1 rupee a month for `months`, at a monthly rate."""
    if months <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return float(months)
    return ((1.0 + rate) ** months - 1.0) / rate


def project(start_inr: float | None, monthly_inr: float,
            real_return_pct: float, goals: list[Goal],
            this_year: int) -> Projection | None:
    """
    Month by month from now to the last goal. None when it cannot be stated:
    no net worth (an incomplete snapshot), no return chosen, or no goals.
    """
    if start_inr is None or real_return_pct == 0 or not goals:
        return None
    rate = _monthly_rate(real_return_pct)
    ordered = sorted(goals, key=lambda g: (g.year, g.name))
    last = min(max(g.year for g in ordered), this_year + MAX_YEARS)

    wealth = float(start_inr)
    years, path, checks = [this_year], [wealth], []
    months_elapsed = 0
    # Each goal is checked at the END of its year, after twelve months of
    # growth and contributions - the conservative reading of "in 2031".
    for year in range(this_year, last + 1):
        for _ in range(12):
            wealth = wealth * (1.0 + rate) + monthly_inr
            months_elapsed += 1
        for goal in (g for g in ordered if g.year == year):
            short = max(0.0, goal.target_inr - wealth)
            fv = _annuity_fv(rate, months_elapsed)
            checks.append(GoalCheck(
                goal=goal, projected_inr=wealth, short_inr=short,
                extra_monthly_inr=(short / fv) if fv else short))
            if goal.spend:
                wealth = max(0.0, wealth - goal.target_inr)
        years.append(year + 1)
        path.append(wealth)
    # Goals past the horizon are reported as unchecked rather than dropped.
    for goal in (g for g in ordered if g.year > last):
        checks.append(GoalCheck(goal=goal, projected_inr=0.0,
                                short_inr=goal.target_inr,
                                extra_monthly_inr=0.0))
    return Projection(years=years, wealth_inr=path, checks=checks)


# ---------------------------------------------------------------- the file

def load(path: str | Path = GOALS_PATH) -> tuple[list[Goal], list[str]]:
    """(goals, warnings). A bad row is a warning, never a silent drop."""
    p = at_root(path)
    if not p.exists():
        return [], []
    goals, warnings = [], []
    with p.open("r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            parsed = from_row(row)
            if parsed is None:
                if any((v or "").strip() for v in row.values()):
                    warnings.append(f"{p.name} row {i}: needs a name, a "
                                    f"target above 0 and a year - skipped")
                continue
            goals.append(parsed)
    return goals, warnings


def from_row(row: dict) -> Goal | None:
    name = str(row.get("name") or "").strip()
    try:
        target = float(str(row.get("target_inr") or "").replace(",", ""))
        year = int(float(str(row.get("year") or "")))
    except ValueError:
        return None
    # `not target > 0` also rejects NaN, which a blank editor cell becomes.
    if not name or not target > 0 or year <= 0:
        return None
    spend_raw = row.get("spend")
    spend = (str(spend_raw).strip().lower() not in ("false", "0", "no", "")
             if spend_raw not in (None, "") else True)
    if isinstance(spend_raw, bool):
        spend = spend_raw
    return Goal(name=name, target_inr=target, year=year, spend=spend)


def save(goals: list[Goal], path: str | Path = GOALS_PATH) -> None:
    p = at_root(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for g in goals:
            w.writerow([g.name, f"{g.target_inr:.0f}", g.year,
                        "true" if g.spend else "false"])
