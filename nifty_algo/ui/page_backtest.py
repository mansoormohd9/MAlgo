"""Backtest page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .theme import get_palette
from .charts import equity_curve, expectancy_by_strategy
from .components import banner
from .state import get_config
from ..backtest import Backtester, Mode
from ..data.csv_feed import CsvFeed
from ..data.factory import build_feed
from ..strategies.registry import all_strategies, default_enabled_keys


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Backtest")

    # Two books, two backtesters, and they measure genuinely different things.
    # A single page with a mode switch would imply the numbers are comparable;
    # they are not - one is an intraday option book and the other is
    # multi-day cash equity.
    which = st.radio("Book", ["Intraday options", "Swing equity"],
                     horizontal=True, key="bt_book")
    if which == "Swing equity":
        _swing(cfg, p)
        return

    banner(
        "<b>What this can and cannot measure.</b> There are no historical option "
        "chains available at any price — Kite serves no data for expired contracts "
        "and retires their instrument tokens. <b>Underlying</b> mode measures the "
        "only question the signal layer can answer — does the setup reach +2R "
        "before −1R on the underlying, after friction. <b>Synthetic premium</b> "
        "mode prices that same move through Black-Scholes at that day's India VIX, "
        "with no skew, and per the README's Phase 3 note is <b>optimistic by "
        "15–25%</b>. Neither is evidence to trade live capital; forty paper "
        "sessions still come first.",
        p.warning, "⚠")

    infos = all_strategies()
    labels = {i.key: i.label for i in infos}

    # ---------------- inputs ----------------
    c1, c2 = st.columns([2, 1])
    source = c1.radio("Data", ["Sample / CSV file", "Current feed"],
                      horizontal=True)
    mode_label = c2.radio("Mode", ["Underlying (trustworthy)",
                                   "Synthetic premium (optimistic)"])
    mode = Mode.UNDERLYING if mode_label.startswith("Underlying") \
        else Mode.SYNTHETIC_PREMIUM

    csv_path = cfg.data.csv_path
    if source.startswith("Sample"):
        csv_path = st.text_input("CSV / Parquet path", value=cfg.data.csv_path)

    keys = st.multiselect(
        "Strategies", [i.key for i in infos],
        default=st.session_state.get("bt_keys", default_enabled_keys()),
        format_func=lambda k: labels.get(k, k))
    st.session_state.bt_keys = keys

    with st.expander("Simulation assumptions — read these once"):
        st.markdown(f"""
- **Intrabar ties go to the stop.** If one bar's range contains both the stop
  and the target, the stop is assumed to hit first. Without this a backtest
  awards itself the best of both on every volatile bar.
- **Friction is subtracted from every trade** via `costs.py`
  (`{cfg.instrument.lot_size}`-unit lot, brokerage + STT + exchange + GST + slippage).
- **Session governors apply**: max {cfg.capital.max_entries_per_session} entries
  per day, entries only between {cfg.session.entry_start:%H:%M} and
  {cfg.session.entry_cutoff:%H:%M}, flat by {cfg.session.force_exit:%H:%M}.
- **The regime gate applies** if it is enabled on the Strategies page.
- **Pivot confirmation delay is enforced** by `BarReplayer` — a strategy cannot
  see a swing pivot before the bars that confirm it have printed.
- **Walk-forward** is {cfg.backtest.train_months} months train /
  {cfg.backtest.test_months} test, rolling; only test windows are scored.
  With less history than that, a single in-sample pass runs instead and says so.
""")

    if not st.button("Run backtest", type="primary"):
        return
    if not keys:
        st.warning("Select at least one strategy.")
        return

    # ---------------- run ----------------
    try:
        if source.startswith("Sample"):
            bars = CsvFeed(csv_path).get_bars(lookback_days=0)
        else:
            bars = build_feed(cfg).get_bars(cfg.data.lookback_days)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Generate the sample file with `python -m nifty_algo.data.sample`")
        return

    with st.spinner(f"Replaying {len(bars):,} bars…"):
        result = Backtester(cfg).run(bars, keys, mode)

    st.caption(f"{len(bars):,} bars · "
               f"{pd.Series(bars.index.date).nunique()} sessions · "
               f"{bars.index.min():%Y-%m-%d} → {bars.index.max():%Y-%m-%d}")

    for w in result.warnings:
        banner(w, p.warning, "⚠")

    if not result.trades:
        st.info("No trades were generated. Either no setup fired, or the regime "
                "gate blocked every one. Widen the date range, enable more "
                "strategies, or turn the gate off on the Strategies page.")
        return

    # ---------------- headline ----------------
    m = result.metrics
    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", m.trades)
    c2.metric("Win rate", f"{m.win_rate:.1%}",
              f"realised breakeven {m.breakeven_win_rate:.1%}",
              delta_color="off")
    c3.metric("Expectancy", f"{m.expectancy_r:+.3f} R",
              "per trade, net of friction", delta_color="off")
    c4.metric("Profit factor", f"{m.profit_factor:.2f}",
              f"max DD {m.max_drawdown_r:.2f} R", delta_color="off")

    if m.win_rate < m.breakeven_win_rate:
        # The REALISED breakeven - what the payoff these trades actually
        # produced had to clear. Attributing it to the configured ratio would
        # be the conflation this figure exists to end: the ladder banks half
        # at +2R and shifts to breakeven at +1R, so it does not produce 2:1
        # trades and 1/(1+R:R) is not the number it had to beat.
        banner(f"Win rate {m.win_rate:.1%} is below the {m.breakeven_win_rate:.1%} "
               f"this payoff needed (avg win {m.avg_win_r:+.2f}R against avg "
               f"loss {-m.avg_loss_r:+.2f}R). A "
               f"{cfg.capital.reward_risk_ratio:.1f}:1 book that always ran to "
               f"target would have needed "
               f"{m.target_breakeven_win_rate:.1%}. This configuration loses "
               f"money on this data.", p.critical, "⛔")
    if mode is Mode.SYNTHETIC_PREMIUM:
        st.caption(f"Synthetic net P&L: ₹{m.net_pnl:+,.0f} — "
                   f"optimistic by 15–25%, see the banner above.")

    st.plotly_chart(equity_curve(result.equity_curve_r, p), width="stretch")

    # ---------------- the day rules ----------------
    # Trade-level R says whether the SETUPS have edge. This block says whether
    # the money management helps or hurts, which is a separate question and the
    # one the original backtester could not answer at all.
    d = result.day_stats
    if d.days:
        st.subheader("Session rules")
        st.caption("How the day governors actually behaved — the +10% target, "
                   "the ratcheting give-back floor, and the 3-entry budget. "
                   "Trade R measures the signals; this measures the rules.")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Trading days", d.days,
                  f"{d.green_days} green", delta_color="off")
        g2.metric("Hit +10% target", d.days_hit_target,
                  f"{d.days_hit_target / d.days:.0%} of days", delta_color="off")
        g3.metric("Hit give-back floor", d.days_hit_floor,
                  f"floor ratcheted on {d.days_ratcheted}", delta_color="off")
        g4.metric("Avg entries/day", f"{d.avg_entries:.2f}",
                  f"of {cfg.capital.max_entries_per_session} · "
                  f"₹{d.avg_realised:,.0f}/day", delta_color="off")

        runners = d.trades_with_partial
        if cfg.trade.enable_runner:
            st.caption(
                f"{runners} of {len(result.trades)} trades reached +2R and "
                f"banked a partial, leaving a runner. If that number is zero "
                f"the runner is not doing anything and the extra brokerage on "
                f"the second exit is pure cost."
            )

    # ---------------- per strategy ----------------
    if len(result.by_strategy) > 1:
        st.plotly_chart(
            expectancy_by_strategy(result.by_strategy, p, labels),
            width="stretch")

    st.markdown("**Per strategy** — table view")
    st.dataframe(pd.DataFrame([
        {"Strategy": labels.get(k, k), **v.as_dict()}
        for k, v in sorted(result.by_strategy.items(),
                           key=lambda kv: kv[1].expectancy_r, reverse=True)
    ]), width="stretch", hide_index=True)

    # ---------------- folds ----------------
    if result.folds:
        st.markdown("**Walk-forward folds** — edge that lives in one fold was a "
                    "good quarter, not an edge")
        st.dataframe(pd.DataFrame([
            {"Fold": f.index + 1,
             "Test window": f"{f.test_start} → {f.test_end}",
             **f.metrics.as_dict()}
            for f in result.folds
        ]), width="stretch", hide_index=True)

    # ---------------- trades ----------------
    with st.expander(f"All {len(result.trades)} trades"):
        tdf = pd.DataFrame([{
            "Entry": t.entry_time, "Exit": t.exit_time,
            "Strategy": labels.get(t.strategy, t.strategy),
            "Dir": t.direction, "Type": t.option_type, "Regime": t.regime,
            "In": round(t.entry_underlying, 1), "Out": round(t.exit_underlying, 1),
            "Outcome": t.outcome.value, "Bars": t.bars_held,
            "R": round(t.r_multiple, 3),
            "Entry prem": round(t.entry_premium, 2) or None,
            "Exit prem": round(t.exit_premium, 2) or None,
            "Net ₹": round(t.net_pnl, 0) or None,
            "Reason": t.reason,
        } for t in result.trades])
        st.dataframe(tdf, width="stretch", hide_index=True)
        st.download_button("Download trades CSV", tdf.to_csv(index=False),
                           "backtest_trades.csv", "text/csv")


# ---------------------------------------------------------------- swing book

def _swing(cfg, p) -> None:
    """
    The daily equity book's backtest.

    A separate panel rather than a mode on the option one, because the two
    measure different things and putting them behind one switch would imply
    the numbers are comparable. They are not.
    """
    from ..swing import backtest as swing_bt
    from ..swing import markets as markets_mod
    from ..swing import prices as prices_mod
    from ..swing.universe import load_universe

    banner(
        "<b>Read these before the numbers.</b> Three distortions here are "
        "structural and cannot be coded away, so they travel with every "
        "result rather than living in a docstring.",
        p.warning, "⚠")
    for c in swing_bt.CAVEATS:
        st.markdown(f"- {c}")

    market = markets_mod.get(cfg, markets_mod.INDIA)
    c1, c2, c3 = st.columns([1, 1, 2])
    years = c1.number_input("Years", min_value=1.0, max_value=10.0,
                            value=3.0, step=0.5, key="swing_bt_years")
    pot = c2.number_input("Swing pot (₹)", min_value=0.0, step=5_000.0,
                          value=float(cfg.capital.swing_capital_inr or
                                      100_000.0),
                          key="swing_bt_pot",
                          help="Sizes every ticket. The backtest refuses to "
                               "deploy more than this in total, because live "
                               "the third buy is simply rejected.")
    c3.caption(
        "Reads the daily bars already cached by the scan where it can. "
        "A longer window downloads more history the first time, which takes "
        "a few minutes for a hundred symbols."
    )

    if not st.button("Run swing backtest", type="primary",
                     key="run_swing_bt"):
        _render_swing(st.session_state.get("swing_bt_result"), cfg, p)
        return

    if pot <= 0:
        st.error("A ₹0 pot sizes every ticket to zero. Set it above, or on "
                 "the Settings page.")
        return

    # `get_config()` hands back the process-global DEFAULT, so BOTH of these
    # overrides leak into every other page for the rest of the session if they
    # are not restored - and a `history_days` left at six years would have
    # every later scan download years of bars it never reads. Restored in
    # `finally`, not on the success path.
    original_pot = cfg.capital.swing_capital_inr
    original_days = cfg.swing.history_days
    cfg.capital.swing_capital_inr = pot
    cfg.swing.history_days = max(int(years * 365) + 260, 400)

    bar = st.progress(0.0, text="loading history")
    try:
        stocks = load_universe(market.universe_csv)
        tickers = {s.symbol: s.yf_ticker for s in stocks}
        price_set = prices_mod.load_prices(
            tickers, cfg, market,
            progress=lambda d, t, l: bar.progress(
                min(1.0, d / t if t else 0.0), text=l))
        result = swing_bt.run(
            cfg, market, price_set.bars, price_set.benchmark, stocks=stocks,
            progress=lambda d, t, l: bar.progress(
                min(1.0, d / t if t else 0.0), text=f"simulating {l}"))
    except Exception as e:
        bar.empty()
        st.error(f"The backtest could not complete: {e}")
        return
    finally:
        cfg.capital.swing_capital_inr = original_pot
        cfg.swing.history_days = original_days

    bar.empty()
    st.session_state.swing_bt_result = result
    _render_swing(result, cfg, p)


def _render_swing(result, cfg, p) -> None:
    if result is None:
        st.info("Nothing run yet.")
        return

    for w in result.warnings[:6]:
        st.warning(w)

    st.subheader(result.headline())
    if result.start and result.end:
        st.caption(f"{result.start:%b %Y} to {result.end:%b %Y} · "
                   f"{result.symbols} symbols")
        # A price cache younger than `price_cache_hours` is reused whole, so
        # raising `history_days` does nothing until it expires - and a "5
        # year" run would quietly walk 400 days without saying so.
        got = (result.end - result.start).days / 365.0
        wanted = float(st.session_state.get("swing_bt_years", got))
        if got < wanted - 0.5:
            st.warning(
                f"You asked for {wanted:g} years and the cached bars only "
                f"cover {got:.1f}. The daily cache is reused until it "
                f"expires, so deep history has to be pulled deliberately: "
                f"`python scripts/fetch_swing_history.py --market india "
                f"--years {wanted:g}`"
            )

    m = result.metrics
    if not m.trades:
        return

    cols = st.columns(4)
    cols[0].metric("Trades", m.trades)
    cols[1].metric("Win rate", f"{m.win_rate:.0%}",
                   delta=f"{m.win_rate - m.breakeven_win_rate:+.0%} vs "
                         f"realised breakeven")
    cols[2].metric("Expectancy", f"{m.expectancy_r:+.3f}R")
    cols[3].metric("Max drawdown", f"{m.max_drawdown_r:.1f}R")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Performance**")
        st.dataframe(pd.DataFrame(m.as_dict().items(),
                                  columns=["", "Value"]),
                     hide_index=True, width="stretch")
    with c2:
        st.markdown("**What the book did**")
        st.dataframe(pd.DataFrame(result.day_stats.as_dict().items(),
                                  columns=["", "Value"]),
                     hide_index=True, width="stretch")

    curve = result.equity_curve_r
    if len(curve) > 1:
        st.plotly_chart(equity_curve(curve, p), width="stretch")

    if result.by_setup:
        st.markdown("**Per setup**")
        st.dataframe(
            pd.DataFrame([{
                "Setup": key,
                "Trades": x.trades,
                "Win rate": f"{x.win_rate:.0%}",
                "Expectancy (R)": f"{x.expectancy_r:+.3f}",
                "Total (R)": f"{x.total_r:+.1f}",
            } for key, x in result.by_setup.items()]),
            hide_index=True, width="stretch")
        st.caption(
            "A setup whose edge lives in one fold did not have edge, it had "
            "a good quarter. Treat a small trade count as no information."
        )

    gross = sum(t.gross_r for t in result.trades)
    net = sum(t.r_multiple for t in result.trades)
    friction = sum(t.friction for t in result.trades)
    st.markdown("**What the charges took**")
    st.caption(
        f"Gross {gross:+.1f}R became net {net:+.1f}R — ₹{friction:,.0f} of "
        f"delivery charges across {len(result.trades)} trades, about "
        f"₹{friction / len(result.trades):,.0f} each. The flat DP fee is the "
        f"same rupee amount on a ₹10,000 ticket as on a ₹100,000 one, which "
        f"is why small positions keep proportionally less of a win."
    )

    with st.expander("Every trade"):
        st.dataframe(
            pd.DataFrame([{
                "In": t.entry_time.date(),
                "Out": t.exit_time.date(),
                "Symbol": t.symbol,
                "Setup": t.strategy,
                "Entry": f"{t.entry:,.2f}",
                "Exit": f"{t.exit_price:,.2f}",
                "Qty": f"{t.quantity:g}",
                "Outcome": t.outcome,
                "Gross R": f"{t.gross_r:+.2f}",
                "Net R": f"{t.r_multiple:+.2f}",
                "Days": t.bars_held,
            } for t in sorted(result.trades, key=lambda x: x.exit_time,
                              reverse=True)]),
            hide_index=True, width="stretch")
