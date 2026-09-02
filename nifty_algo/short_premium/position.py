"""
A live short-premium position: the R inversion, and the three stops.

WHY THIS IS NOT `positions.ManagedPosition`. That class converts premium to R
as `(premium - entry_premium) / per_r` (positions.py:308-310) and P&L as
`(exit_premium - entry_premium) * lot_size * lots`. Both are correct for a
buyer and exactly backwards for a seller: a short whose premium is FALLING is
winning, and would report negative R there and stop out on its first bar. The
sign lives in one place here, and `ExitLadder` never sees it.

HOW R IS DEFINED, AND WHY THE LADDER HAD TO BE RE-PARAMETERISED

    per_r (per unit) = credit x (stop_at_credit_multiple - 1)
    r_of(premium)    = (credit - premium) / per_r
    max_r            = credit / per_r = 1 / (multiple - 1)

At the default 2.0 multiple the premium doubling is exactly -1R and buying
back at zero is exactly +1R. So the MAXIMUM ACHIEVABLE R IS 1.0, and
`TradeManagementConfig.partial_exit_at_r` of 2.0 could never fire - the rung
would not error, it would simply never happen, taking the runner and the trail
with it. `ShortPremiumConfig.ladder_settings()` supplies 0.25R / 0.50R
instead. This is the single most important number in the book to get right.

THE THREE STOPS, AND WHY ONE IS NOT ENOUGH

  PREMIUM stop - the premium reaching `credit x multiple`. This is a VEGA
  stop as much as a directional one: an IV spike with spot unchanged fires it,
  at the worst possible moment and the widest possible spread.

  UNDERLYING stop - spot coming within `underlying_stop_buffer_atr` ATR of
  THE SHORT STRIKE. Blind to vol, but it fires on the thing that actually
  turns a short into an unbounded loss. Anchored to the strike rather than to
  entry spot, because that is what a seller's risk scales with - see the
  config comment for the two ways the spot-anchored version is wrong.

  TIME stop - flat at `SessionConfig.force_exit`. The only one of the three
  that is always available, and the reason an intraday seller survives a night
  of headlines.

Whichever comes first. A book with only the premium stop exits every vol event
at the worst price; a book with only the underlying stop sits through a
repricing that has already lost it the trade.

GAPS FILL AT THE OPEN, NOT AT THE STOP AND NOT AT THE BAR'S WORST TICK.
`positions._to_action` fills a stopped position at the stop LEVEL
(positions.py:437-440); for a buyer with a bounded loss that is a deliberate
anti-flattering choice, but for a seller it truncates precisely the tail that
ends short-premium accounts. The fix is NOT to fill at the bar's high: a bar
that wicks through a trailing stop and recovers does not fill you at its worst
print, and pricing it that way overstates every trailing exit in the book.

So the rule is the swing backtest's rule (`swing/backtest.py`): if the bar
OPENED beyond the stop, the stop gapped and fills at the open; otherwise it
fills at the stop level. The loss is allowed to exceed 1R - that is the whole
point - but only when the market actually gapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import Config, DEFAULT
from ..positions import ExitKind, ExitLadder, LadderDecision, LadderState
from ..risk import OptionQuote

#: Why a position was closed outside the ladder's own rungs. These are not
#: ladder decisions - the ladder speaks only in R and knows nothing about spot.
STOP_UNDERLYING = "underlying_stop"
STOP_SHORT_STRIKE = "short_strike_touched"


@dataclass
class ShortAction:
    """One thing to do, priced. The premium here is a FILL, not a level."""
    kind: ExitKind
    lots: int
    exit_premium: float
    realised_r: float
    gross_pnl: float
    detail: str = ""
    new_stop_r: Optional[float] = None


@dataclass
class ShortPosition:
    quote: OptionQuote
    strategy_key: str
    lots: int
    lot_size: int
    entry_credit: float
    entry_spot: float
    #: The spot level at which the underlying stop fires. Above entry_spot for
    #: a short CE, below it for a short PE - the caller computes it from ATR
    #: because ATR belongs to the bar frame, not to the position.
    underlying_stop: float
    state: LadderState = field(default=None)  # type: ignore[assignment]
    cfg: Config = field(default_factory=lambda: DEFAULT)

    def __post_init__(self):
        if self.state is None:
            self.state = ExitLadder(
                self.cfg, trade=self.cfg.short_premium.ladder_settings()
            ).new_state(self.lots)

    # ---------------- the arithmetic, all of it signed here -------------

    @property
    def quantity(self) -> int:
        return self.lot_size * self.state.lots_remaining

    @property
    def short_strike(self) -> int:
        return self.quote.strike

    @property
    def is_call(self) -> bool:
        return self.quote.option_type == "CE"

    @property
    def per_r_premium(self) -> float:
        """Premium points that make one R. Never zero - guarded at build."""
        mult = self.cfg.short_premium.stop_at_credit_multiple
        return max(self.entry_credit * (mult - 1.0), 1e-9)

    @property
    def stop_premium(self) -> float:
        """The premium level that is -1R."""
        return self.entry_credit * self.cfg.short_premium.stop_at_credit_multiple

    @property
    def max_r(self) -> float:
        """The most this position can ever make: buying back at zero."""
        return self.entry_credit / self.per_r_premium

    def r_of(self, premium: float) -> float:
        """THE INVERSION. Premium falling is profit."""
        return (self.entry_credit - float(premium)) / self.per_r_premium

    def premium_of(self, r: float) -> float:
        """R back to a premium level. Higher R is a LOWER premium."""
        return self.entry_credit - r * self.per_r_premium

    def pnl_for(self, lots: int, exit_premium: float) -> float:
        """Gross rupees. Sold at credit, bought back at exit_premium."""
        return (self.entry_credit - float(exit_premium)) * self.lot_size * lots

    def underlying_breached(self, spot: float) -> Optional[str]:
        """
        Whether the underlying stop fired, and which side of the buffer it
        was. Touch counts, not just cross.

        Both names describe the same level: STOP_SHORT_STRIKE when spot has
        actually reached the strike, STOP_UNDERLYING when it is inside the
        buffer but not yet there. One test, two labels, so the journal can
        tell "it got to my strike" from "it got close".
        """
        if self.is_call:
            if spot >= self.underlying_stop:
                return (STOP_SHORT_STRIKE if spot >= self.short_strike
                        else STOP_UNDERLYING)
        elif spot <= self.underlying_stop:
            return (STOP_SHORT_STRIKE if spot <= self.short_strike
                    else STOP_UNDERLYING)
        return None

    # ---------------- advancing one bar --------------------------------

    def advance(self, mark_premium: float,
                spot: float,
                bar_high_premium: Optional[float] = None,
                bar_low_premium: Optional[float] = None,
                bar_open_premium: Optional[float] = None,
                adverse_spot: Optional[float] = None) -> list[ShortAction]:
        """
        One 5-minute bar, or one live tick.

        `bar_high_premium` is the ADVERSE extreme for a seller and
        `bar_low_premium` the favourable one - the inversion again, and the
        reason these are named for the bar rather than for the outcome.

        `bar_open_premium` decides whether a stop GAPPED or was merely
        touched, and so what it fills at. Omitting it prices every stop as a
        gap.

        `adverse_spot` is the bar's extreme in the direction that hurts: the
        high for a short call, the low for a short put. Defaulting it to
        `spot` measures the underlying stop on closes only, which understates
        how often it fires.
        """
        if self.state.closed:
            return []

        high = mark_premium if bar_high_premium is None else bar_high_premium
        low = mark_premium if bar_low_premium is None else bar_low_premium
        opened = mark_premium if bar_open_premium is None else bar_open_premium
        hurt = spot if adverse_spot is None else adverse_spot

        # --- the underlying stop, tested FIRST -------------------------
        # Before the ladder, for the same reason the ladder tests its own stop
        # before any promotion: a bar that breached the strike is not rescued
        # by a favourable premium print later in the same bar.
        breach = self.underlying_breached(hurt)
        if breach is not None:
            lots = self.state.lots_remaining
            self.state.lots_remaining = 0
            # Fills at the mark. We learn of the breach when we see the bar,
            # and the premium then is the mark - not the bar's worst tick,
            # which we could not have traded at.
            fill = float(mark_premium)
            return [ShortAction(
                kind=ExitKind.STOPPED_OUT, lots=lots, exit_premium=fill,
                realised_r=self.r_of(fill),
                gross_pnl=self.pnl_for(lots, fill),
                detail=(f"{breach}: spot {hurt:.1f} vs "
                        f"{'strike ' + str(self.short_strike) if breach == STOP_SHORT_STRIKE else f'stop {self.underlying_stop:.1f}'}"),
            )]

        ladder = ExitLadder(self.cfg,
                            trade=self.cfg.short_premium.ladder_settings())
        decisions = ladder.advance(
            self.state,
            mark_r=self.r_of(mark_premium),
            best_r=self.r_of(low),        # lowest premium = best for a seller
            worst_r=self.r_of(high),      # highest premium = worst
            trail_distance_r=self.cfg.short_premium.trail_distance_r,
        )
        return [self._price(d, mark_premium, opened) for d in decisions]

    def force_exit(self, mark_premium: float,
                   reason: str = "15:10 force exit") -> list[ShortAction]:
        ladder = ExitLadder(self.cfg,
                            trade=self.cfg.short_premium.ladder_settings())
        d = ladder.force_exit(self.state, self.r_of(mark_premium), reason)
        if d.kind is None:
            return []
        return [self._price(d, mark_premium, mark_premium)]

    def _price(self, d: LadderDecision, mark: float,
               bar_open: float) -> ShortAction:
        """
        Turn a decision in R into a fill.

        A STOPPED_OUT fills at the stop level unless the bar OPENED beyond it,
        in which case it gapped and fills at the open. That is the whole
        difference from `positions._to_action`, which fills at the level
        unconditionally: for a bounded-loss buyer the level is conservative,
        for a seller it caps the one outcome that is not bounded.
        """
        if d.kind is ExitKind.STOPPED_OUT:
            level = self.premium_of(d.exit_r)
            fill = max(level, float(bar_open))
        elif d.is_exit:
            fill = self.premium_of(d.exit_r)
        else:
            return ShortAction(kind=d.kind, lots=0, exit_premium=float(mark),
                               realised_r=self.r_of(mark), gross_pnl=0.0,
                               detail=d.detail, new_stop_r=d.new_stop_r)

        return ShortAction(
            kind=d.kind, lots=d.exit_lots, exit_premium=fill,
            realised_r=self.r_of(fill),
            gross_pnl=self.pnl_for(d.exit_lots, fill),
            detail=d.detail, new_stop_r=d.new_stop_r,
        )


def underlying_stop_for(option_type: str, short_strike: float, atr: float,
                        cfg: Config = DEFAULT) -> float:
    """
    Where the underlying stop sits: a buffer INSIDE the short strike.

    Anchored to the strike, never to entry spot - see
    `ShortPremiumConfig.underlying_stop_buffer_atr` for the two ways the
    spot-anchored version is wrong.
    """
    d = atr * cfg.short_premium.underlying_stop_buffer_atr
    return short_strike - d if option_type == "CE" else short_strike + d
