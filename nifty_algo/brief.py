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

    @property
    def verdict(self) -> str:
        if self.selected:
            return "SELECTED"
        return "ok" if not self.gates else "; ".join(self.gates)


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

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Strike": r.strike,
            "Premium": round(r.premium, 2),
            "Bid": round(r.bid, 2),
            "Ask": round(r.ask, 2),
            "Spread%": f"{r.spread_pct:.2%}",
            "OI": r.open_interest,
            "IV": f"{r.iv:.1%}" if r.iv else "-",
            "Delta": round(abs(r.delta), 3),
            "Entry": round(r.entry, 2) if r.entry else None,
            "Target": round(r.target, 2) if r.target else None,
            "Stop": round(r.stop, 2) if r.stop else None,
            "Verdict": r.verdict,
        } for r in self.rows])

    def lines(self) -> list[str]:
        out = [
            f"CHAIN  {self.option_type}  spot {self.spot:,.1f}  "
            f"expiry {self.expiry:%d %b}  [{self.source}]",
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
            out += [
                "",
                f"  IF ENTERED NOW: BUY {d.lots} lot(s) x {d.quote.strike}"
                f"{d.quote.option_type} = {d.quantity} qty",
                f"    entry  {d.entry_premium:.2f}",
                f"    target {d.premium_target:.2f}   (+Rs {d.rupee_reward:,.0f})",
                f"    stop   {d.premium_stop:.2f}   (-Rs {d.rupee_risk:,.0f})",
                f"    {d.sizing_note}",
            ]
        elif isinstance(self.decision, RejectedOrder):
            out += ["", f"  NO TRADEABLE STRIKE: {self.decision.reason.value} "
                        f"- {self.decision.detail}"]
        return out


def build_chain_view(spot: float, option_type: str, stop_points: float,
                     cfg: Config = DEFAULT,
                     chain_provider: ChainProvider | None = None,
                     risk: RiskEngine | None = None,
                     today: Optional[date] = None) -> ChainView:
    """
    The chain, scored by the real risk engine.

    Every gate verdict comes from `RiskEngine.gate_failures()` and the
    entry/target/stop from `RiskEngine.approve()` - the same calls the live
    engine makes. Nothing here recomputes a rule.
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
        rows.append(ChainRow(
            strike=q.strike, option_type=q.option_type, premium=q.premium,
            bid=q.bid, ask=q.ask, spread_pct=q.spread_pct,
            open_interest=q.open_interest, delta=q.delta,
            iv=q.iv,
            gates=risk.gate_failures(q, stop_points, lots),
            selected=selected,
            entry=decision.entry_premium if selected else None,
            target=decision.premium_target if selected else None,
            stop=decision.premium_stop if selected else None,
            lots=decision.lots if selected else 0,
        ))

    return ChainView(
        spot=spot, option_type=option_type, expiry=result.expiry,
        source=result.source, is_synthetic=result.is_synthetic,
        note=result.note, stop_points=stop_points, lots=lots,
        delta_ceiling=ceiling, rows=rows, decision=decision,
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
    rejects = [r for r in records if r.get("event") == "rejection"]

    if entries:
        out += ["", "  ENTRIES"]
        for r in entries:
            p = r.get("payload", r)
            out.append(f"    {p.get('strike')}{p.get('option_type')}  "
                       f"{p.get('lots')} lot(s) @ {p.get('entry')}  "
                       f"stop {p.get('stop')}  target {p.get('target')}  "
                       f"{'runner' if p.get('runner') else 'no runner'}"
                       f"{'  [DRY RUN]' if p.get('dry_run') else ''}")

    if actions:
        out += ["", "  MANAGEMENT"]
        realised = 0.0
        for r in actions:
            p = r.get("payload", r)
            realised += float(p.get("net") or 0.0)
            out.append(f"    {p.get('kind'):<18} {p.get('strike')}"
                       f"{p.get('option_type')}  {p.get('detail')}"
                       f"{'  net Rs ' + format(p.get('net', 0), ',.0f') if p.get('lots') else ''}")
        out.append(f"    {'REALISED':<18} Rs {realised:,.0f}")

    if rejects:
        out += ["", "  REJECTED"]
        for r in rejects[:20]:
            p = r.get("payload", r)
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
