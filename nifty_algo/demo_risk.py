"""
Run this first: python -m nifty_algo.demo_risk

It prints the actual numbers your rules produce. Read the output before
you write another line of strategy code - if the economics do not work
here, no amount of signal cleverness will save the system.
"""
from datetime import date, time

from .config import DEFAULT
from .costs import DEFAULT_COSTS
from .governor import SessionGovernor
from .risk import RiskEngine, OptionQuote


def line(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    cfg = DEFAULT
    engine = RiskEngine(cfg)
    cap = cfg.capital
    inst = cfg.instrument

    line("YOUR RULES, RESOLVED")
    print(f"Capital                 Rs {cap.starting_capital:>10,.0f}")
    print(f"Session target  {cap.session_target_pct:>5.0%}   Rs "
          f"{cap.starting_capital * cap.session_target_pct:>10,.0f}")
    print(f"Session stop    {cap.session_stop_pct:>5.0%}   Rs "
          f"{cap.starting_capital * cap.session_stop_pct:>10,.0f}")
    print(f"Max entries             {cap.max_entries_per_session:>10}")
    print(f"  -> risk / trade {cap.risk_per_trade_pct:>5.2%}   Rs "
          f"{cap.risk_per_trade_rupees:>10,.0f}")
    print(f"  -> reward/ trade {cap.reward_per_trade_pct:>4.2%}   Rs "
          f"{cap.reward_per_trade_rupees:>10,.0f}")
    print(f"  -> reward:risk          {cap.reward_risk_ratio:>10.2f} : 1")

    be_winrate = 1 / (1 + cap.reward_risk_ratio)
    print(f"\nBreakeven win rate (before costs): {be_winrate:.1%}")
    print("Three consecutive losses hit the session stop exactly.")
    print("That is by design - 'max 3 orders' and '5% stop' are the")
    print("same constraint expressed two different ways.")

    line("THE STRIKE CONSTRAINT")
    print(f"Lot size {inst.lot_size}. Max premium loss per unit = "
          f"Rs {cap.risk_per_trade_rupees:,.0f} / {inst.lot_size} = "
          f"Rs {cap.risk_per_trade_rupees / inst.lot_size:.2f}\n")
    print(f"{'underlying stop':>16} {'max delta':>11} {'verdict':>28}")
    for stop_pts in (30, 40, 50, 60, 80, 100, 120):
        d = engine.required_max_delta(stop_pts)
        if d >= 0.45:
            verdict = "ATM affordable (stop tight)"
        elif d >= cfg.signal.min_delta:
            verdict = "OTM only"
        else:
            verdict = "REJECT - no viable strike"
        print(f"{stop_pts:>13} pts {d:>11.3f} {verdict:>28}")
    print("\nRead this the right way round: the stop chooses the strike.")
    print("A wider stop forces a lower-delta (further OTM) option.")

    line("COST DRAG PER ROUND TRIP")
    print(f"{'premium':>9} {'lot cost':>11} {'friction':>10} {'as % of risk':>14}")
    for premium in (40, 60, 80, 120, 200):
        qty = inst.lot_size
        lot_cost = premium * qty
        friction = DEFAULT_COSTS.total_friction(premium, premium, qty)
        print(f"{premium:>9.0f} {lot_cost:>11,.0f} {friction:>10,.0f} "
              f"{friction / cap.risk_per_trade_rupees:>13.1%}")
    print("\nFriction eats a real slice of every trade's risk budget.")
    print("Your backtest must subtract it or the equity curve is fiction.")

    line("WHY THE RUNNER NEEDS TWO LOTS")
    print("NSE fills whole lots only. You cannot sell 32 of a 65-quantity lot,")
    print("so 'bank half at +2R' is impossible on one lot. Sizing two halves")
    print("the delta ceiling, because risk per trade is FIXED:\n")
    print(f"{'underlying stop':>16} {'1 lot':>9} {'2 lots':>9}   {'runner?':>8}")
    for stop_pts in (25, 30, 40, 50, 64, 80):
        d1 = engine.required_max_delta(stop_pts, 1)
        d2 = engine.required_max_delta(stop_pts, 2)
        ok = "yes" if d2 >= cfg.signal.min_delta else "NO - 1 lot"
        print(f"{stop_pts:>13} pts {min(d1, cfg.signal.max_delta):>9.3f} "
              f"{min(d2, cfg.signal.max_delta):>9.3f}   {ok:>8}")
    print(f"\nBelow a delta of {cfg.signal.min_delta:.2f} no strike survives, so a wide-stop")
    print("day falls back to one lot with the runner disabled. The alert says so.")
    print("Note the cost: two lots is twice the quantity, so friction as a share")
    print("of a FIXED risk budget roughly doubles. The runner is not free.")

    line("THE GIVE-BACK RATCHET")
    print("The day stop trails the day's PEAK realised P&L:\n")
    print(f"{'peak':>10} {'floor':>10}   reads as")
    probe = SessionGovernor(cfg, capital=cap.starting_capital)
    for peak in (0, 2_000, 4_000, 6_000, 8_000):
        probe.start_day()
        probe.register_exit(peak)
        pct = probe.floor_pct_of_capital
        reads = (f"{pct:.0%} locked in" if pct > 0
                 else f"{abs(pct):.0%} day stop")
        note = ("  <- your stated rule" if peak == 2_000
                else "  <- cannot finish red" if pct > 0 else "")
        print(f"{peak:>+10,.0f} {probe.floor:>+10,.0f}   {reads}{note}")
    print("\nMonotonic: the peak only rises, so the floor never loosens.")
    print("Give-back equals the opening budget, so there is always exactly")
    print("three trades' worth of room below the peak.")

    line("SESSION GOVERNOR WALKTHROUGH")
    engine.start_day(date.today())
    chain = [
        OptionQuote(26000, "CE", 210.0, 0.50, 209.5, 210.5, 900_000),
        OptionQuote(26100, "CE", 150.0, 0.38, 149.7, 150.3, 750_000),
        OptionQuote(26200, "CE",  95.0, 0.28, 94.8,  95.2,  600_000),
        OptionQuote(26300, "CE",  55.0, 0.18, 54.8,  55.2,  400_000),
    ]

    for trade_no, pnl in enumerate([-1667, -1667, +3333], start=1):
        halt = engine.check_halt(now=time(10, 30))
        if halt.value != "none":
            print(f"Trade {trade_no}: BLOCKED - {halt.value}")
            break

        decision = engine.approve(chain, "CE", underlying_stop_points=60,
                                  free_capital=engine.capital)
        if hasattr(decision, "reason"):
            print(f"Trade {trade_no}: REJECTED - {decision.reason.value}")
            break

        q = decision.quote
        print(f"Trade {trade_no}: {q.strike}{q.option_type} @ {q.premium:.1f} "
              f"(delta {q.delta:.2f}, qty {decision.quantity})  "
              f"stop {decision.premium_stop:.2f} / target {decision.premium_target:.2f}")
        print(f"          {decision.sizing_note}")
        engine.register_entry(decision, "long")
        engine.register_exit(engine.session.open_positions[-1], pnl)
        print(f"          realised Rs {pnl:+,.0f}   "
              f"session Rs {engine.session.realised_pnl:+,.0f}   "
              f"floor Rs {engine.governor.floor:+,.0f}")

    final_halt = engine.check_halt(now=time(11, 0))
    print(f"\nAfter 3 entries: halt = {final_halt.value}")
    print("Summary:", engine.summary())
    print("\nNote the sizing note above: at a 60-point stop the two-lot delta")
    print("ceiling is 0.21 and this demo chain has nothing between 0.20 and")
    print("0.21, so it falls back to one lot and the runner is off. That is")
    print("the fallback working, not a bug - and it is why the alert says so.")


if __name__ == "__main__":
    main()
