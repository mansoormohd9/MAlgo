"""
Run this first: python -m nifty_algo.demo_risk

It prints the actual numbers your rules produce. Read the output before
you write another line of strategy code - if the economics do not work
here, no amount of signal cleverness will save the system.
"""
from datetime import date, time

from .config import DEFAULT
from .costs import DEFAULT_COSTS
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
        engine.register_entry(decision, "long")
        engine.register_exit(engine.session.open_positions[-1], pnl)
        print(f"          realised Rs {pnl:+,.0f}   "
              f"session Rs {engine.session.realised_pnl:+,.0f}")

    final_halt = engine.check_halt(now=time(11, 0))
    print(f"\nAfter 3 entries: halt = {final_halt.value}")
    print("Summary:", engine.summary())


if __name__ == "__main__":
    main()
