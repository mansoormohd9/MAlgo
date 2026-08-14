"""
The daily brief.

    python -m nifty_algo.brief              # whatever is appropriate right now
    python -m nifty_algo.brief --preopen
    python -m nifty_algo.brief --chain
    python -m nifty_algo.brief --review 2026-08-07

Three sections, for three moments in the day:

  PRE-OPEN   the frame. Prior close, gap, ATR, VIX, days to expiry, and the
             exact rupee budget the day starts with.

  CHAIN      every strike near the money with its live premium, spread, OI and
             derived delta - AND which risk gate each one fails. A table that
             shows only the winner teaches you nothing about the other
             twenty-four; seeing that six strikes failed on spread and four on
             open interest is how you learn what a tradeable chain looks like.

  REVIEW     what fired, what was rejected and why, and how the day rules moved,
             read back out of the append-only journal.

The `entry / target / stop` column is computed by calling `RiskEngine.approve()`
against the same chain the engine would see. It is not a parallel calculation
that might drift - it is the calculation.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from . import signals as sig
from .config import Config, DEFAULT
from .costs import DEFAULT_COSTS
from .data.base import DataFeed, FeedError
from .data.chain import ChainProvider
from .data.factory import build_feed
from .governor import SessionGovernor
from .regime import classify
from .risk import ApprovedOrder, RejectedOrder, RiskEngine


# ---------------------------------------------------------------- pre-open

@dataclass
class PreOpen:
    day: date
    prior_close: float
    prior_high: float
    prior_low: float
    atr: float
    gap_points: float
    gap_atr: float
    vix: Optional[float]
    expiry: Optional[date]
    days_to_expiry: Optional[int]
    expiry_blocks_trading: bool
    has_volume: bool
    feed_name: str
    feed_note: str
    budget: dict = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [
            f"PRE-OPEN  {self.day:%a %d %b %Y}",
            f"  feed            {self.feed_name}"
            f"{'  [' + self.feed_note + ']' if self.feed_note else ''}",
            f"  prior close     {self.prior_close:,.1f}"
            f"   (H {self.prior_high:,.1f} / L {self.prior_low:,.1f})",
            f"  ATR(14)         {self.atr:,.1f} pts",
        ]
        if self.gap_points:
            out.append(f"  gap             {self.gap_points:+,.1f} pts "
                       f"({self.gap_atr:+.2f} ATR)")
        if self.vix is not None:
            out.append(f"  India VIX       {self.vix:.2f}"
                       f"   -> {self.vix / 100:.1%} annualised IV")
        if self.expiry is not None:
            note = "  <- EXPIRY DAY, entries blocked" if self.expiry_blocks_trading else ""
            out.append(f"  expiry          {self.expiry:%d %b} "
                       f"({self.days_to_expiry}d){note}")
        if not self.has_volume:
            out.append("  volume          NONE (index series) - participation "
                       "gates use range expansion")

        b = self.budget
        out += [
            "",
            "  TODAY'S BUDGET",
            f"    capital       Rs {b['capital']:,.0f}",
            f"    day target    Rs {b['target']:,.0f}   (+{b['target_pct']:.0%})",
            f"    day floor     Rs {b['floor']:,.0f}   ({b['floor_pct']:.0%})",
            f"    per trade     Rs {b['risk']:,.0f} risk / "
            f"Rs {b['reward']:,.0f} reward  ({b['rr']:.1f}:1)",
            f"    entries       {b['entries']} max",
            f"    breakeven     {b['breakeven_move']:.2f} premium points just "
            f"to cover costs",
        ]
        return out


def build_preopen(cfg: Config = DEFAULT, feed: DataFeed | None = None,
                  chain_provider: ChainProvider | None = None,
                  vix: Optional[float] = None) -> PreOpen:
    feed = feed or build_feed(cfg)
    bars = feed.get_bars(cfg.data.lookback_days)
    today = bars.index[-1].date()

    prior = DataFeed.prior_session(bars, today)
    session = DataFeed.session_slice(bars, today)
    atr_series = sig.atr(bars, cfg.signal.atr_period)
    atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

    gap_points = 0.0
    if not session.empty and prior.close:
        gap_points = float(session["open"].iloc[0]) - prior.close

    provider = chain_provider or ChainProvider(cfg)
    expiry = provider.resolve_expiry(today)
    dte = (expiry - today).days if expiry else None

    if vix is None:
        vix = _latest_vix(cfg)

    gov = SessionGovernor(cfg)
    gov.start_day()
    c = cfg.capital
    lot = cfg.instrument.lot_size * cfg.trade.preferred_lots

    return PreOpen(
        day=today,
        prior_close=prior.close, prior_high=prior.high, prior_low=prior.low,
        atr=atr, gap_points=gap_points,
        gap_atr=(gap_points / atr if atr else 0.0),
        vix=vix, expiry=expiry, days_to_expiry=dte,
        expiry_blocks_trading=(expiry == today and not cfg.session.trade_on_expiry_day),
        has_volume=sig.has_traded_volume(bars),
        feed_name=feed.name,
        feed_note=feed.latency_note if feed.is_delayed else "",
        budget={
            "capital": gov.capital,
            "target": gov.target, "target_pct": c.session_target_pct,
            "floor": gov.floor, "floor_pct": gov.floor_pct_of_capital,
            "risk": c.risk_per_trade_rupees,
            "reward": c.reward_per_trade_rupees,
            "rr": c.reward_risk_ratio,
            "entries": c.max_entries_per_session,
            "breakeven_move": DEFAULT_COSTS.breakeven_move(120.0, lot),
        },
    )


def _latest_vix(cfg: Config) -> Optional[float]:
    p = Path(cfg.data.vix_csv_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
        return float(df["vix"].iloc[-1])
    except Exception:
        return None


# ---------------------------------------------------------------- the chain

@dataclass
class ChainRow:
    strike: int
    option_type: str
    premium: float
    bid: float
    ask: float
    spread_pct: float
    open_interest: int
    delta: float
    iv: Optional[float]
    gates: list[str]
    selected: bool = False
    entry: Optional[float] = None
    target: Optional[float] = None
    stop: Optional[float] = None
    lots: int = 0
    quantity: int = 0
    rupee_risk: Optional[float] = None
    rupee_reward: Optional[float] = None
    outlay: Optional[float] = None
    runner: bool = False

    @property
    def viable(self) -> bool:
        """Would `approve()` actually let you buy this one?"""
        return self.entry is not None

    @property
    def verdict(self) -> str:
        if self.selected:
            return "SELECTED"
        return "ok" if not self.gates else "; ".join(self.gates)

    @property
    def action(self) -> str:
        """What buying this row would mean, in words rather than a colour."""
        if self.selected:
            return f"BUY {self.lots} lot(s) - the engine's pick"
        if self.viable:
            return f"tradeable at {self.lots} lot(s), lower delta"
        return "not tradeable"


@dataclass
class ChainView:
    spot: float
    option_type: str
    expiry: date
    source: str
    is_synthetic: bool
    note: str
    stop_points: float
    lots: int
    delta_ceiling: float
    rows: list[ChainRow]
    decision: object = None
    days_to_expiry: int = 0
    entry_permitted: bool = True
    halt_reason: str = ""

    # ---------------- what this view is actually telling you ----------------

    @property
    def approved(self) -> Optional[ApprovedOrder]:
        return self.decision if isinstance(self.decision, ApprovedOrder) else None

    @property
    def pick(self) -> Optional[ChainRow]:
        return next((r for r in self.rows if r.selected), None)

    @property
    def runner_ups(self) -> list[ChainRow]:
        """
        Tradeable strikes that lost, best delta first.

        Shown because a single highlighted row looks like magic. Seeing the
        two strikes that were also buyable, and that they were passed over for
        less delta, is what makes the pick legible.
        """
        others = [r for r in self.rows if r.viable and not r.selected]
        return sorted(others, key=lambda r: abs(r.delta), reverse=True)[:3]

    def headline(self) -> str:
        """
        The suggestion in one plain sentence, with no jargon and no colour
        doing the talking. The UI ticket and the CLI both print this, so the
        two surfaces cannot describe the same chain differently.
        """
        d = self.approved
        if d is None:
            reason = getattr(self.decision, "reason", None)
            detail = getattr(self.decision, "detail", "")
            what = reason.value.replace("_", " ") if reason else "no viable strike"
            return (f"Nothing to buy on the {self.option_type} side - {what}"
                    f"{': ' + detail if detail else ''}.")
        return (f"BUY {d.lots} lot(s) of {self.option_type} strike "
                f"{d.quote.strike}, expiry {self.expiry:%d %b} "
                f"({self.days_to_expiry} day{'' if self.days_to_expiry == 1 else 's'})"
                f" - {d.quantity} qty at about {d.entry_premium:.2f}.")

    def status(self) -> tuple[str, str]:
        """(label, explanation) — is this a thing to do, or a thing to know?"""
        if not self.entry_permitted:
            return ("BLOCKED",
                    f"Entries are closed: {self.halt_reason.replace('_', ' ')}. "
                    f"Shown for reference only.")
        if self.approved is None:
            return ("NOTHING TO DO", "No strike on this side clears every gate.")
        return ("REFERENCE ONLY",
                f"No strategy has fired. This is the strike the engine would "
                f"pick if a {self.option_type} signal appeared right now, at a "
                f"{self.stop_points:.0f}-point stop.")

    # ---------------- rendering ----------------

    def to_frame(self) -> pd.DataFrame:
        """
        Numbers stay numbers so the table can be sorted. Formatting is the
        caller's job (`st.column_config` in the UI, `lines()` for the CLI).
        """
        return pd.DataFrame([{
            "Strike": r.strike,
            "Action": r.action,
            "Premium": round(r.premium, 2),
            "Bid": round(r.bid, 2),
            "Ask": round(r.ask, 2),
            "Spread%": r.spread_pct,
            "OI": r.open_interest,
            "IV": r.iv,
            "Delta": round(abs(r.delta), 3),
            "Lots": r.lots or None,
            "Entry": round(r.entry, 2) if r.entry is not None else None,
            "Target": round(r.target, 2) if r.target is not None else None,
            "Stop": round(r.stop, 2) if r.stop is not None else None,
            "Risk": round(r.rupee_risk) if r.rupee_risk is not None else None,
            "Reward": round(r.rupee_reward) if r.rupee_reward is not None else None,
            "Outlay": round(r.outlay) if r.outlay is not None else None,
            "Why not": "" if r.viable else "; ".join(r.gates),
        } for r in self.rows])

    def lines(self) -> list[str]:
        label, explanation = self.status()
        out = [
            f"CHAIN  {self.option_type}  spot {self.spot:,.1f}  "
            f"expiry {self.expiry:%d %b} ({self.days_to_expiry}d)  [{self.source}]",
            f"  {label}: {explanation}",
            f"  >> {self.headline()}",
            f"  stop {self.stop_points:.0f} pts at {self.lots} lot(s) "
            f"-> delta ceiling {self.delta_ceiling:.3f}",
        ]
        if self.is_synthetic:
            out.append(f"  ** {self.note}")
        out.append("")
        out.append(f"  {'Strike':>7} {'Prem':>8} {'Spr%':>7} {'OI':>10} "
                   f"{'IV':>7} {'Delta':>7}  Verdict")
        for r in self.rows:
            out.append(
                f"  {r.strike:>7} {r.premium:>8.2f} {r.spread_pct:>6.2%} "
                f"{r.open_interest:>10,} "
                f"{(format(r.iv, '.1%') if r.iv else '-'):>7} "
                f"{abs(r.delta):>7.3f}  {r.verdict}"
            )
        if isinstance(self.decision, ApprovedOrder):
            d = self.decision
            qty = d.quantity
            out += [
                "",
                f"  IF ENTERED NOW: BUY {d.lots} lot(s) x {d.quote.strike}"
                f"{d.quote.option_type} = {qty} qty",
                f"    entry  {d.entry_premium:.2f}",
                f"    target {d.premium_target:.2f}   (+Rs {d.rupee_reward:,.0f})",
                f"    stop   {d.premium_stop:.2f}   (-Rs {d.rupee_risk:,.0f})",
                f"    outlay Rs {d.entry_premium * qty:,.0f}   "
                f"costs Rs {DEFAULT_COSTS.round_trip(d.entry_premium, d.premium_target, qty):,.0f} "
                f"round trip, breakeven "
                f"+{DEFAULT_COSTS.breakeven_move(d.entry_premium, qty):.2f} pts",
                f"    {d.sizing_note}",
            ]
            if self.runner_ups:
                out.append("    also tradeable: " + ", ".join(
                    f"{r.strike}{r.option_type} (delta {abs(r.delta):.3f})"
                    for r in self.runner_ups))
        elif isinstance(self.decision, RejectedOrder):
            out += ["", f"  NO TRADEABLE STRIKE: {self.decision.reason.value} "
                        f"- {self.decision.detail}"]
        return out


def build_chain_view(spot: float, option_type: str, stop_points: float,
                     cfg: Config = DEFAULT,
                     chain_provider: ChainProvider | None = None,
                     risk: RiskEngine | None = None,
                     today: Optional[date] = None,
                     entry_permitted: bool = True,
                     halt_reason: str = "") -> ChainView:
    """
    The chain, scored by the real risk engine.

    Every gate verdict comes from `RiskEngine.gate_failures()` and the
    entry/target/stop from `RiskEngine.approve()` - the same calls the live
    engine makes. Nothing here recomputes a rule.

    `approve()` is called once for the chain to find the pick, and then once
    per quote against a single-strike chain to fill in what buying THAT strike
    would mean. The second call is pure computation over a list of one and
    reuses the rule rather than restating it, which is the only way the row
    numbers cannot drift from the engine's.

    `entry_permitted` / `halt_reason` are passed in by the caller because
    `approve()` deliberately knows nothing about the session clock - it will
    happily approve a strike at 3pm on expiry day. Carrying the answer on the
    view is what stops a reference price from reading like an instruction.
    """
    today = today or date.today()
    provider = chain_provider or ChainProvider(cfg)
    risk = risk or RiskEngine(cfg)

    result = provider.get_chain(spot, option_type, today)
    lots = risk.lots_for(stop_points)
    ceiling = min(risk.required_max_delta(stop_points, lots), cfg.signal.max_delta)

    decision = risk.approve(result.quotes, option_type, stop_points, risk.capital)
    chosen = decision.quote.strike if isinstance(decision, ApprovedOrder) else None

    rows = []
    for q in sorted(result.quotes, key=lambda x: x.strike):
        selected = q.strike == chosen
        # What this specific strike would cost you, decided by the same rule.
        own = (decision if selected
               else risk.approve([q], option_type, stop_points, risk.capital))
        buyable = isinstance(own, ApprovedOrder)
        rows.append(ChainRow(
            strike=q.strike, option_type=q.option_type, premium=q.premium,
            bid=q.bid, ask=q.ask, spread_pct=q.spread_pct,
            open_interest=q.open_interest, delta=q.delta,
            iv=q.iv,
            gates=risk.gate_failures(q, stop_points, lots),
            selected=selected,
            entry=own.entry_premium if buyable else None,
            target=own.premium_target if buyable else None,
            stop=own.premium_stop if buyable else None,
            lots=own.lots if buyable else 0,
            quantity=own.quantity if buyable else 0,
            rupee_risk=own.rupee_risk if buyable else None,
            rupee_reward=own.rupee_reward if buyable else None,
            outlay=(own.entry_premium * own.quantity) if buyable else None,
            runner=own.runner_enabled if buyable else False,
        ))

    return ChainView(
        spot=spot, option_type=option_type, expiry=result.expiry,
        source=result.source, is_synthetic=result.is_synthetic,
        note=result.note, stop_points=stop_points, lots=lots,
        delta_ceiling=ceiling, rows=rows, decision=decision,
        days_to_expiry=max((result.expiry - today).days, 0),
        entry_permitted=entry_permitted, halt_reason=halt_reason,
    )


# ---------------------------------------------------------------- review

def review(day: date, journal_dir: str = "journal") -> list[str]:
    """Read the day back out of the append-only journal."""
    path = Path(journal_dir) / f"{day.isoformat()}.jsonl"
    if not path.exists():
        return [f"REVIEW  {day}", f"  no journal at {path}"]

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    out = [f"REVIEW  {day:%a %d %b %Y}   ({len(records)} records)"]
    counts: dict[str, int] = {}
    for r in records:
        counts[r.get("event", "?")] = counts.get(r.get("event", "?"), 0) + 1
    out.append("  " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))

    entries = [r for r in records if r.get("event") == "entry_confirmed"]
    actions = [r for r in records if r.get("event") == "position_action"]
    # `Journal.rejection()` writes the event name "rejected". This filtered on
    # "rejection", so the REJECTED block this module's docstring advertises was
    # always empty - the one thing the review promised to show never rendered.
    rejects = [r for r in records if r.get("event") == "rejected"]

    if entries:
        out += ["", "  ENTRIES"]
        for r in entries:
            # `Journal.write()` flattens payload into the top-level record, so
            # the record IS the payload. There is no "payload" key to look up.
            p = r
            out.append(f"    {p.get('strike')}{p.get('option_type')}  "
                       f"{p.get('lots')} lot(s) @ {p.get('entry')}  "
                       f"stop {p.get('stop')}  target {p.get('target')}  "
                       f"{'runner' if p.get('runner') else 'no runner'}"
                       f"{'  [DRY RUN]' if p.get('dry_run') else ''}")

    if actions:
        out += ["", "  MANAGEMENT"]
        realised = 0.0
        for r in actions:
            p = r
            realised += float(p.get("net") or 0.0)
            out.append(f"    {p.get('kind'):<18} {p.get('strike')}"
                       f"{p.get('option_type')}  {p.get('detail')}"
                       f"{'  net Rs ' + format(p.get('net', 0), ',.0f') if p.get('lots') else ''}")
        out.append(f"    {'REALISED':<18} Rs {realised:,.0f}")

    if rejects:
        out += ["", "  REJECTED"]
        for r in rejects[:20]:
            p = r
            out.append(f"    {p.get('strategy', '?'):<18} {p.get('reason', '')} "
                       f"- {p.get('detail', '')}")

    return out


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default=None, help="csv | kite | yfinance | ...")
    p.add_argument("--csv", default=None, help="override DataConfig.csv_path")
    p.add_argument("--preopen", action="store_true")
    p.add_argument("--chain", action="store_true")
    p.add_argument("--review", metavar="YYYY-MM-DD")
    p.add_argument("--stop-points", type=float, default=None,
                   help="stop width for the chain view (default: 1 x ATR)")
    args = p.parse_args(argv)

    cfg = Config()
    if args.provider:
        cfg.data.provider = args.provider
    if args.csv:
        cfg.data.csv_path = args.csv

    if args.review:
        print("\n".join(review(date.fromisoformat(args.review))))
        return 0

    show_all = not (args.preopen or args.chain)

    try:
        feed = build_feed(cfg)
        pre = build_preopen(cfg, feed)
    except FeedError as e:
        print(f"\n  {e}\n")
        return 1

    if args.preopen or show_all:
        print("\n".join(pre.lines()))

    if args.chain or show_all:
        bars = feed.get_bars(cfg.data.lookback_days)
        spot = float(bars["close"].iloc[-1])
        stop_points = args.stop_points or max(pre.atr, 1.0)
        provider = ChainProvider(cfg)
        for option_type in ("CE", "PE"):
            print("")
            view = build_chain_view(spot, option_type, stop_points, cfg,
                                    chain_provider=provider)
            print("\n".join(view.lines()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
