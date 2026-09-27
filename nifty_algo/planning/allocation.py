"""
Current allocation against target, and where the next rupee should go.

WHY THIS EXISTS. At 32 the decisions that move the outcome most are the split
between asset classes, the savings rate, costs and tax - not the signal inside
any one book. The console had five books and nothing that looked at the whole,
so every pot was a number typed on its own page with no reference to the
others or to the net worth they are shares of.

THREE RULES, each inherited from somewhere else in the repo:

  1. AN INCOMPLETE SNAPSHOT WITHHOLDS EVERY PERCENTAGE.
     `PortfolioSnapshot.weight()` returns None when any enabled connector
     failed, and so does everything here. A drift computed against a
     denominator that could not be established reads exactly like one that
     was, and a rebalance is an ACTION.

  2. NEW MONEY FIRST, SELLING LAST.
     Directing deposits at the underweight classes rebalances for free;
     selling realises a gain, and in India a gain held under a year is STCG.
     So `plan_contribution` fills deficits with the contribution before it
     suggests trimming anything, and a trim is only named for a class still
     outside the band AFTER the money is placed.

  3. THE SLEEVE'S SHARE IS DERIVED, NOT CHOSEN.
     `sleeve_cap` is `factor/drawdown.sizing_report`'s arithmetic - tolerated
     portfolio fall divided by the sleeve's planning drawdown - pointed at a
     real net worth. It is shown beside the pot you type and never overrides
     it: the tolerance is yours, the drawdown is measured.

Pure functions. No Streamlit, no network, no file reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..portfolio.base import (CASH, ETF, FIXED_INCOME, GOLD, MUTUAL_FUND,
                              Position)

INDIA_EQUITY = "india_equity"
FOREIGN_EQUITY = "foreign_equity"
BUCKET_GOLD = "gold"
BUCKET_FIXED = "fixed_income"
BUCKET_CASH = "cash"

BUCKETS = (INDIA_EQUITY, FOREIGN_EQUITY, BUCKET_GOLD, BUCKET_FIXED,
           BUCKET_CASH)

#: The bucket KEYS are persisted (`PlanConfig.w_gold`, `w_fixed_income`) and
#: never change; the labels say what a halal book actually keeps in each. Gold
#: is "gold & silver" because both are the same thing in an allocation - a
#: non-yielding store of value - and silver had no bucket at all, so
#: SILVERBEES was counted as Indian equity.
LABELS = {
    INDIA_EQUITY: "Indian equity",
    FOREIGN_EQUITY: "Foreign equity (LRS / international funds)",
    BUCKET_GOLD: "Gold & silver",
    BUCKET_FIXED: "Stable leg (sukuk / EPF / PPF / FD)",
    BUCKET_CASH: "Cash & emergency fund",
}

#: Kite reports every NSE line as EQUITY, including gold ETFs, so a gold ETF
#: bought through Zerodha would otherwise count as Indian equity. This is a
#: HINT, not a classifier: anything it misses can be recorded by hand with
#: `asset_class = gold`. Deliberately an explicit list plus the sovereign gold
#: bond prefix - a substring match on "GOLD" would catch jewellers' shares.
GOLD_SYMBOLS = frozenset({
    "GOLDBEES", "GOLDIETF", "GOLDCASE", "GOLD1", "GOLDETF", "HDFCGOLD",
    "AXISGOLD", "SBIGOLD", "ICICIGOLD", "KOTAKGOLD", "IVZINGOLD", "LICMFGOLD",
    "TATAGOLD", "UTIGOLD", "NIPGOLD", "QGOLDHALF", "BSLGOLDETF", "GOLDSHARE",
    "EGOLD", "AONEGOLD", "MOGOLD", "GROWWGOLD",
})

#: Silver ETFs, for the same reason and with the same caveat as the gold list.
SILVER_SYMBOLS = frozenset({
    "SILVERBEES", "SILVERIETF", "HDFCSILVER", "SBISILVER", "AXISILVER",
    "SILVERETF", "SILVERCASE", "SILVER1", "TATSILV", "SILVERADD",
})

#: Overnight / liquid ETFs. Kite reports them as EQUITY; economically they are
#: cash parked at an overnight repo rate, so they belong in the cash bucket -
#: and `planning/compliance.py` marks them interest-bearing.
LIQUID_SYMBOLS = frozenset({
    "LIQUIDBEES", "LIQUIDIETF", "LIQUIDCASE", "LIQUID1", "LIQUIDSBI",
    "LIQUIDETF", "LIQUIDADD",
})

#: NSE-listed ETFs that hold FOREIGN shares. They trade in rupees on an Indian
#: exchange, so `market == "india"` - and without this list a Nasdaq-100 ETF
#: bought on Zerodha was counted as Indian equity, understating the foreign
#: share and pointing new money at the class you already hold.
INTERNATIONAL_SYMBOLS = frozenset({
    "MON100", "MONQ50", "MAFANG", "MASPTOP50", "MAHKTECH", "HNGSNGBEES",
})

#: Words in a fund's NAME that mark an Indian fund-of-funds investing abroad
#: (CAS schemes arrive with an ISIN, not a ticker). Phrases, not single words,
#: so an Indian fund with "Global" in its marketing name is not caught by
#: accident - anything missed is fixed by recording it by hand.
INTERNATIONAL_NAME_MARKERS = (
    "NASDAQ", "S&P 500", "US EQUITY", "U.S. EQUITY", "US BLUECHIP",
    "OVERSEAS", "INTERNATIONAL", "GLOBAL EQUITY", "WORLD", "FEEDER",
    "HANG SENG", "EMERGING MARKET", "EUROPE", "JAPAN", "CHINA",
)

#: Suggested only when the user presses for one, and labelled as a starting
#: point. Never a default - see `PlanConfig`.
STARTING_POINT = {
    INDIA_EQUITY: 55.0,
    FOREIGN_EQUITY: 20.0,
    BUCKET_GOLD: 10.0,
    BUCKET_FIXED: 10.0,
    BUCKET_CASH: 5.0,
}

#: The same shape for a halal-only holder. The stable leg shrinks to what an
#: investor usually cannot avoid (EPF) or can hold as sukuk, and gold & silver
#: take its place as the non-equity ballast. Offered, never applied.
HALAL_STARTING_POINT = {
    INDIA_EQUITY: 55.0,
    FOREIGN_EQUITY: 20.0,
    BUCKET_GOLD: 15.0,
    BUCKET_FIXED: 5.0,
    BUCKET_CASH: 5.0,
}


def starting_point(plan) -> dict[str, float]:
    """The starting split to offer - the halal one when the holder asked."""
    return dict(HALAL_STARTING_POINT if getattr(plan, "halal_only", False)
                else STARTING_POINT)


def bucket_of(position: Position) -> str:
    """Which planning bucket a holding belongs to."""
    if position.asset_class == CASH:
        return BUCKET_CASH
    if position.asset_class == FIXED_INCOME:
        return BUCKET_FIXED
    if position.asset_class == GOLD:
        return BUCKET_GOLD
    symbol = position.symbol.upper()
    if (symbol in GOLD_SYMBOLS or symbol in SILVER_SYMBOLS
            or symbol.startswith("SGB")):
        return BUCKET_GOLD
    if symbol in LIQUID_SYMBOLS:
        return BUCKET_CASH
    if position.market != "india":
        return FOREIGN_EQUITY
    if symbol in INTERNATIONAL_SYMBOLS or is_international_fund(position):
        return FOREIGN_EQUITY
    return INDIA_EQUITY


def is_international_fund(position: Position) -> bool:
    """An Indian-domiciled fund whose holdings are abroad, judged by its name."""
    if position.asset_class not in (ETF, MUTUAL_FUND):
        return False
    name = (position.name or "").upper()
    return any(marker in name for marker in INTERNATIONAL_NAME_MARKERS)


@dataclass
class Row:
    bucket: str
    value_inr: float
    target_pct: float
    #: None whenever the snapshot is incomplete or holds nothing.
    current_pct: float | None
    band_pp: float

    @property
    def label(self) -> str:
        return LABELS[self.bucket]

    @property
    def drift_pp(self) -> float | None:
        if self.current_pct is None:
            return None
        return self.current_pct - self.target_pct

    @property
    def status(self) -> str:
        drift = self.drift_pp
        if drift is None:
            return "unknown"
        if drift > self.band_pp:
            return "over"
        if drift < -self.band_pp:
            return "under"
        return "in band"


@dataclass
class Allocation:
    rows: list[Row]
    total_inr: float
    complete: bool
    targets_set: bool
    notes: list[str] = field(default_factory=list)

    def row(self, bucket: str) -> Row:
        return next(r for r in self.rows if r.bucket == bucket)

    @property
    def net_worth_inr(self) -> float | None:
        """The whole, or None - a subtotal is never called a net worth."""
        return self.total_inr if self.complete else None

    @property
    def out_of_band(self) -> list[Row]:
        return [r for r in self.rows if r.status in ("over", "under")]


def targets_total(plan) -> float:
    return sum(plan.targets().values())


def targets_valid(plan) -> bool:
    """Targets sum to 100% (to the nearest 0.1pp). Zero means not set."""
    return abs(targets_total(plan) - 100.0) < 0.1


def build(snapshot, plan) -> Allocation:
    """Current against target, bucket by bucket."""
    values = {b: 0.0 for b in BUCKETS}
    for position in snapshot.positions:
        values[bucket_of(position)] += snapshot.value_inr.get(position.key, 0.0)

    total = sum(values.values())
    complete = bool(snapshot.complete)
    targets = plan.targets()
    notes: list[str] = []

    known = complete and total > 0
    if not complete:
        notes.append("Not every account answered, so percentages are "
                     "withheld - see Holdings for which one failed.")
    elif total <= 0:
        notes.append("No holdings recorded yet - connect an account or add "
                     "your balances on Connect.")
    ok_targets = targets_valid(plan)
    if not ok_targets:
        notes.append(f"Target weights sum to {targets_total(plan):.1f}%, not "
                     f"100% - drift is not computed until they do.")

    rows = [Row(bucket=b, value_inr=values[b],
                target_pct=float(targets.get(b, 0.0)),
                current_pct=(values[b] / total * 100.0) if known else None,
                band_pp=float(plan.rebalance_band_pp))
            for b in BUCKETS]
    return Allocation(rows=rows, total_inr=total, complete=complete,
                      targets_set=ok_targets, notes=notes)


@dataclass
class ContributionPlan:
    #: Rupees of the contribution each bucket should receive.
    buy: dict[str, float]
    #: Buckets still over target + band AFTER the money is placed, with the
    #: rupee trim that would bring each back to target. Empty in the usual
    #: case - which is the point of rebalancing with new money.
    trim: dict[str, float]
    reason: str


def plan_contribution(allocation: Allocation, amount: float) -> ContributionPlan | None:
    """
    Where `amount` of new money should go. None when drift is unknowable.

    Fill the deficits first, in proportion to their size; if the money covers
    every deficit, the remainder is split by target weight so the book stays
    on target rather than drifting toward whichever class was last short.
    """
    if not (allocation.complete and allocation.targets_set) or amount < 0:
        return None
    total_after = allocation.total_inr + amount
    deficits = {r.bucket: max(0.0, r.target_pct / 100.0 * total_after
                              - r.value_inr)
                for r in allocation.rows}
    need = sum(deficits.values())
    buy = {b: 0.0 for b in BUCKETS}
    if amount > 0:
        if need >= amount:
            for b, d in deficits.items():
                buy[b] = amount * d / need if need else 0.0
            reason = ("Every rupee goes to the classes below target, largest "
                      "gap first - rebalancing without selling anything.")
        else:
            left = amount - need
            for r in allocation.rows:
                buy[r.bucket] = deficits[r.bucket] + left * r.target_pct / 100.0
            reason = ("This covers every gap; the remainder is split by "
                      "target weight.")
    else:
        reason = "No new money - only the trims below would rebalance."

    trim: dict[str, float] = {}
    for r in allocation.rows:
        after = r.value_inr + buy[r.bucket]
        pct = after / total_after * 100.0 if total_after else 0.0
        if pct - r.target_pct > r.band_pp:
            trim[r.bucket] = after - r.target_pct / 100.0 * total_after
    return ContributionPlan(buy=buy, trim=trim, reason=reason)


@dataclass
class EmergencyCheck:
    needed_inr: float
    cash_inr: float
    monthly_expenses_inr: float

    @property
    def months_covered(self) -> float:
        return self.cash_inr / self.monthly_expenses_inr

    @property
    def ok(self) -> bool:
        return self.cash_inr >= self.needed_inr

    @property
    def shortfall_inr(self) -> float:
        return max(0.0, self.needed_inr - self.cash_inr)


def emergency(allocation: Allocation, plan) -> EmergencyCheck | None:
    """Cash against months x expenses, or None until expenses are entered."""
    if plan.monthly_expenses_inr <= 0:
        return None
    return EmergencyCheck(
        needed_inr=plan.monthly_expenses_inr * plan.emergency_months,
        cash_inr=allocation.row(BUCKET_CASH).value_inr,
        monthly_expenses_inr=plan.monthly_expenses_inr)


def planning_drawdown() -> float:
    """The sleeve's planning drawdown, from the SAME constants F2b uses."""
    from ..factor.drawdown import DRAWDOWN_HAIRCUT
    from ..factor.sleeve import MEASURED_DRAWDOWN
    return min(0.95, MEASURED_DRAWDOWN * DRAWDOWN_HAIRCUT)


def sleeve_share(tolerated_drawdown_pct: float) -> float:
    """Largest share of net worth the sleeve may be, for a tolerated fall."""
    if tolerated_drawdown_pct <= 0:
        return 0.0
    return min(1.0, tolerated_drawdown_pct / planning_drawdown())


def sleeve_cap(net_worth_inr: float | None,
               tolerated_drawdown_pct: float) -> float | None:
    """The recommended maximum factor pot in rupees, or None without a net worth."""
    if net_worth_inr is None or net_worth_inr <= 0:
        return None
    return net_worth_inr * sleeve_share(tolerated_drawdown_pct)


#: A typed or statement balance older than this is flagged. Six months is
#: long enough that an EPF or FD balance has visibly moved.
STALE_DAYS = 180


def stale_lines(positions, today, max_days: int = STALE_DAYS) -> list:
    """
    `[(position, age_days)]` for every dated line older than `max_days`.

    Only lines that CARRY a date can be judged; an undated manual line is
    reported by the caller as undated, not assumed fresh.
    """
    from datetime import date
    out = []
    for p in positions:
        if not p.as_of:
            continue
        try:
            age = (today - date.fromisoformat(p.as_of[:10])).days
        except ValueError:
            continue
        if age > max_days:
            out.append((p, age))
    return sorted(out, key=lambda pa: -pa[1])
