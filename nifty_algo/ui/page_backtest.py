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

    banner(
        "<b>What this can and cannot measure.</b> There are no historical option "
        "chains in this project. <b>Underlying</b> mode measures the only question "
        "the signal layer can answer — does the setup reach +2R before −1R on the "
        "underlying, after friction. <b>Synthetic premium</b> mode prices that same "
        "move through Black-Scholes at one flat IV, and per the README's Phase 3 "
        "note is <b>optimistic by 15–25%</b>. Neither is evidence to trade live "
        "capital; forty paper sessions still come first.",
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
              f"breakeven {m.breakeven_win_rate:.1%}", delta_color="off")
    c3.metric("Expectancy", f"{m.expectancy_r:+.3f} R",
              "per trade, net of friction", delta_color="off")
    c4.metric("Profit factor", f"{m.profit_factor:.2f}",
              f"max DD {m.max_drawdown_r:.2f} R", delta_color="off")

    if m.win_rate < m.breakeven_win_rate:
        banner(f"Win rate {m.win_rate:.1%} is below the {m.breakeven_win_rate:.1%} "
               f"needed at {cfg.capital.reward_risk_ratio:.1f}:1. This "
               f"configuration loses money on this data.", p.critical, "⛔")
    if mode is Mode.SYNTHETIC_PREMIUM:
        st.caption(f"Synthetic net P&L: ₹{m.net_pnl:+,.0f} — "
                   f"optimistic by 15–25%, see the banner above.")

    st.plotly_chart(equity_curve(result.equity_curve_r, p), width="stretch")

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
