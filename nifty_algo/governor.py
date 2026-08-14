"""
Session governors: the rules that end the trading day.

Split out of `risk.py` because these are the only rules in the system with no
opinion about markets at all. They are arithmetic on the day's P&L, they are
the rules most likely to be edited, and they are the ones that must be
provable - so they live somewhere they can be tested without constructing a
chain, a strategy, or a bar.

Three governors, "whichever reaches first":

  TARGET      realised >= +10% of capital        -> day over, close everything
  FLOOR       realised <= a ratcheting floor     -> day over, close everything
  ENTRIES     3 entries taken                    -> no more entries

The floor is the interesting one. It is not a constant -5%; it trails the
day's PEAK realised P&L, so banked profit tightens the day's remaining risk:

    floor = max(-session_stop_pct, peak - ratchet_giveback_pct)   x capital

which is what produces "5% becomes 3% once I am 2% up, and likewise".

TARGET AND FLOOR MUST CLOSE OPEN POSITIONS, NOT MERELY BLOCK NEW ENTRIES.
The original `RiskEngine.check_halt()` was only ever consulted before an
entry, which is precisely the wrong moment: if a runner alone carries the day
to +10%, the position is still open at the instant the rule fires. `evaluate()`
is therefore called after EVERY realised exit, including partials, and returns
an action rather than a boolean.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from .config import Config, DEFAULT


class GovernorAction(str, Enum):
    CONTINUE = "continue"
    BLOCK_NEW_ENTRIES = "block_new_entries"   # no more entries, keep managing open ones
    CLOSE_ALL = "close_all"                   # flatten now, day is over


class GovernorReason(str, Enum):
    NONE = "none"
    SESSION_TARGET_HIT = "session_target_hit"
    GIVE_BACK_FLOOR_HIT = "give_back_floor_hit"
    MAX_ENTRIES_REACHED = "max_entries_reached"


@dataclass
class GovernorVerdict:
    action: GovernorAction
    reason: GovernorReason
    detail: str = ""

    @property
    def day_over(self) -> bool:
        return self.action is GovernorAction.CLOSE_ALL


class SessionGovernor:
    """
    Tracks realised P&L for one trading day and reports what may still happen.

    Deliberately holds no positions and places no orders - it is handed
    numbers and returns a verdict.
    """

    def __init__(self, cfg: Config = DEFAULT, capital: float | None = None):
        self.cfg = cfg
        self.capital = capital or cfg.capital.starting_capital
        self.realised_pnl = 0.0
        self.peak_realised_pnl = 0.0
        self.entries_taken = 0
        self._floor_history: list[tuple[float, float]] = [(0.0, self.floor)]

    # ---------------- day lifecycle ----------------

    def start_day(self, capital: float | None = None) -> None:
        if capital is not None:
            self.capital = capital
        self.realised_pnl = 0.0
        self.peak_realised_pnl = 0.0
        self.entries_taken = 0
        self._floor_history = [(0.0, self.floor)]

    # ---------------- the levels ----------------

    @property
    def target(self) -> float:
        return self.capital * self.cfg.capital.session_target_pct

    @property
    def base_floor(self) -> float:
        return -abs(self.capital * self.cfg.capital.session_stop_pct)

    @property
    def floor(self) -> float:
        """
        The ratcheting give-back floor.

        Monotonic by construction: `peak_realised_pnl` only ever rises, and
        this is a non-decreasing function of it. A floor that could loosen
        would let a good morning fund a worse afternoon, which is the exact
        failure the rule exists to prevent.
        """
        c = self.cfg.capital
        arm_at = self.capital * c.ratchet_arm_at_pct
        if self.peak_realised_pnl < arm_at:
            return self.base_floor
        trailed = self.peak_realised_pnl - self.capital * c.ratchet_giveback_pct
        return max(self.base_floor, trailed)

    @property
    def floor_pct_of_capital(self) -> float:
        """The floor as the percentage a human reads it as: -0.03 means 3%."""
        return self.floor / self.capital if self.capital else 0.0

    @property
    def risk_remaining(self) -> float:
        """Rupees between here and the floor. Zero or less means the day is done."""
        return self.realised_pnl - self.floor

    # ---------------- events ----------------

    def register_entry(self) -> None:
        self.entries_taken += 1

    def register_exit(self, net_pnl: float) -> GovernorVerdict:
        """
        Book a realised P&L - a full exit or a partial - and re-evaluate.

        Partials matter here. Banking half a runner is exactly the kind of
        event that lifts the peak, tightens the floor, and can by itself carry
        the day to target while the other half is still open.
        """
        self.realised_pnl += net_pnl
        if self.realised_pnl > self.peak_realised_pnl:
            self.peak_realised_pnl = self.realised_pnl
            self._floor_history.append((self.realised_pnl, self.floor))
        return self.evaluate()

    def evaluate(self) -> GovernorVerdict:
        if self.realised_pnl >= self.target:
            return GovernorVerdict(
                GovernorAction.CLOSE_ALL, GovernorReason.SESSION_TARGET_HIT,
                f"realised Rs {self.realised_pnl:,.0f} reached the "
                f"{self.cfg.capital.session_target_pct:.0%} day target "
                f"(Rs {self.target:,.0f})",
            )
        if self.realised_pnl <= self.floor:
            return GovernorVerdict(
                GovernorAction.CLOSE_ALL, GovernorReason.GIVE_BACK_FLOOR_HIT,
                f"realised Rs {self.realised_pnl:,.0f} hit the floor of "
                f"Rs {self.floor:,.0f} ({abs(self.floor_pct_of_capital):.1%} of "
                f"capital; peak was Rs {self.peak_realised_pnl:,.0f})",
            )
        if self.entries_taken >= self.cfg.capital.max_entries_per_session:
            return GovernorVerdict(
                GovernorAction.BLOCK_NEW_ENTRIES, GovernorReason.MAX_ENTRIES_REACHED,
                f"{self.entries_taken} of "
                f"{self.cfg.capital.max_entries_per_session} entries used",
            )
        return GovernorVerdict(GovernorAction.CONTINUE, GovernorReason.NONE)

    # ---------------- reporting ----------------

    @property
    def ratchet_history(self) -> list[tuple[float, float]]:
        """(realised, floor) at each point the floor tightened. For the brief."""
        return list(self._floor_history)

    def summary(self) -> dict:
        return {
            "capital": round(self.capital, 2),
            "realised": round(self.realised_pnl, 2),
            "peak": round(self.peak_realised_pnl, 2),
            "target": round(self.target, 2),
            "floor": round(self.floor, 2),
            "floor_pct": f"{self.floor_pct_of_capital:.2%}",
            "risk_remaining": round(self.risk_remaining, 2),
            "entries_taken": self.entries_taken,
            "max_entries": self.cfg.capital.max_entries_per_session,
            "verdict": self.evaluate().action.value,
        }
