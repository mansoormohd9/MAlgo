# Nifty Intraday Option-Buying System

A from-scratch algo trading system for NSE index options, sized for a
₹1,00,000 account under SEBI's April 2026 retail algo framework.

Run `python -m nifty_algo.demo_risk` first. It prints the actual economics
of your rules before you write any strategy logic.

**This system alerts. It never places an order.** It computes the strike,
size, target, and stop and tells you; you place the trade. There is no broker
order path anywhere in the package.

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

python -m nifty_algo.demo_risk        # the economics — read this first
python -m nifty_algo.data.sample      # fabricated sample bars, so nothing blocks
streamlit run app.py                  # the alert console
```

For alerts that reach you with the browser closed, copy `.env.example` to
`.env`, fill in the Telegram keys, and run the headless loop:

```bash
python -m nifty_algo.run_live --provider yfinance --telegram
```

Streamlit cannot notify you when the tab is shut. That is why the decision
loop lives in `engine.py` and the UI is only a viewer over it — both drivers
run identical code.

### The five pages

| Page | What it does |
|---|---|
| **Live alerts** | Session governors, alert cards with entry/target/stop/strike/size, candlestick chart with the levels, trendlines and VWAP the strategies actually used, and a *why nothing fired* table |
| **Strategies** | Toggle each of the ten setups, tune every parameter, control the regime gate |
| **Backtest** | Walk-forward run, metrics, equity curve, per-strategy expectancy, trade list |
| **Journal** | The append-only record, filterable, CSV/JSONL export |
| **Settings** | Feed choice, notification channels with per-channel test buttons, `.env` status |

---

## Your rules, resolved

| Rule | Value | Level |
|---|---|---|
| Session target | +10% (₹10,000) | Stop trading for the day |
| Session stop | −5% (₹5,000) | Stop trading for the day |
| Max entries | 3 | Stop trading for the day |
| Risk per trade | 1.67% (₹1,667) | *derived* = 5% ÷ 3 |
| Reward per trade | 3.33% (₹3,333) | *derived* = 10% ÷ 3 |
| Reward:risk | 2.00 : 1 | breakeven win rate 33.3% |

The three governors are self-consistent: three consecutive losses land
exactly on the session stop. "Max 3 orders" and "5% stop" are the same
constraint expressed twice — that's a feature, keep it.

## The constraint that shapes everything

```
max premium loss per unit = ₹1,667 / 65 = ₹25.64
required delta            ≤ 25.64 / underlying_stop_points
```

| Underlying stop | Max delta | Implication |
|---|---|---|
| 40 pts | 0.64 | ATM affordable, but stop is inside the noise |
| 60 pts | 0.43 | near-ATM |
| 80 pts | 0.32 | OTM only |
| 120 pts | 0.21 | deep OTM, gamma risk |

**The stop chooses the strike, not the reverse.** Every retail trader who
picks a strike first and then "sets a stop" has the causality backwards.

Sweet spot for this account: **60–80 point stop, 0.30–0.40 delta**, premium
₹90–150, lot cost ₹6,000–10,000. Round-trip friction there is ~₹70, about
4% of your risk budget.

---

## Architecture

```
Data ──► Signals ──► Strategy ──► RiskEngine ──► Execution ──► Journal
                                      ▲
                                 hard gate
```

`RiskEngine.approve()` is the only path to an order. The strategy proposes;
risk disposes. Everything the risk engine enforces — session governors,
strike selection, correlation cap, kill switch — is unreachable from
strategy code by design.

**The one rule above all others:** `strategy.py` is imported *unchanged* by
both the backtester and the live runner. If you ever write `if backtest:`
inside a strategy, your backtest has stopped measuring the system you
intend to trade.

### Files

| File | Role |
|---|---|
| `config.py` | Every tunable number. Nothing hardcoded elsewhere. |
| `costs.py` | Indian statutory charges + slippage. |
| `signals.py` | Pure functions: ATR, pivots, levels, trendlines, VWAP, EMA, compression, sweeps, gaps. |
| `strategy.py` | `Strategy` ABC + `LevelBreakStrategy`. **Unchanged — do not edit.** |
| `strategies/` | Eight added setups + the registry the UI and backtester share. |
| `risk.py` | Session governors, strike selection, position bookkeeping. |
| `regime.py` | Day classifier — decides which strategies may speak today. |
| `pricing.py` | Black-Scholes, for the synthetic chain and the backtest premium mode. |
| `data/` | `DataFeed` ABC, CSV replay + `BarReplayer`, yfinance, Fyers/Dhan, chain provider. |
| `alerts/` | `TradeAlert`, four channels, and the de-duplicating dispatcher. |
| `engine.py` | The decision loop. Headless; driven by both the UI and `run_live`. |
| `journal.py` | Append-only JSONL, one file per trading day. |
| `backtest.py` | Walk-forward, two modes. Read its docstring before its numbers. |
| `run_live.py` | Headless runner — alerts with the browser closed. |
| `demo_risk.py` | Prints the economics. Start here. |
| `../app.py` | Streamlit console. A viewer over the engine; decides nothing. |

### Your discretionary signals, made mechanical

| You watch | Code |
|---|---|
| Buy/sell pressure | `close_position_in_range()` — 1.0 = buyers won the bar |
| Doji | `body_to_range() ≤ 0.15` |
| Support/resistance points | `find_pivots()` → `cluster_levels()`, scored by touch count |
| A line traversing up/down | `fit_trendline()` — least squares through swings, gated on R² |
| Liquidity | spread %, OI, volume-vs-baseline gates in `risk.py` |
| News | **blackout filter, not a signal** — see roadmap |

---

## The strategy library

Ten setups, one contract: `Context` in, `Signal` out. None can reach an alert
without passing `RiskEngine.approve()`.

| Strategy | Idea | Default |
|---|---|---|
| **Level break** | Close beyond a clustered level, volume surge + conviction candle | ⭐ on |
| **Trendline break** | R²-gated line through swings; trade the *break*, not the bounce | ⭐ on |
| **VWAP reclaim / reject** | Return across session VWAP after a real excursion away from it | ⭐ on |
| **Volatility squeeze** | NR7 compression resolving into expansion | ⭐ on |
| **ORB retest** | Opening-range break, then a retest that *holds* | off |
| **Failed breakout / sweep** | Wick beyond a level, close back inside — the inverse of level break | off |
| **Trend pullback** | EMA 9/21 trend, entry on a pullback to the fast EMA | off |
| **PDH/PDL sweep reclaim** | The day's largest and most visible stop cluster, taken and reclaimed | off |
| **Gap fade / gap-and-go** | Gap > 0.5 ATR: fade toward prior close, or continue if the OR holds | off |
| **Doji reversal** | Your original variant B, now selectable on its own | off |

**Most ship off deliberately.** You get three entries a day; ten strategies
spend them before 11:00 on whichever setups fire first, and a losing week
becomes unattributable.

### Two additions worth arguing for

**The regime gate** (`regime.py`) is not a strategy — it classifies the day as
expansion, range, or gap from opening-range width versus ATR, and only lets a
strategy fire in the regimes it declares. A long option is a long-volatility,
negative-theta position: in a range the underlying goes nowhere, IV bleeds, and
theta collects against you every bar, so a breakout system fires into chop and
pays friction each time. It costs one ATR comparison and it is the
highest-leverage filter here.

**The volatility squeeze** is the only setup that buys premium *before* IV
reprices. Every other strategy enters after a move has begun — by then the
option is at its most expensive and the move must continue just to cover what
you paid. Compression means realised volatility has collapsed, dragging premium
down with it; you go long vega into a vega increase instead of being
short-changed by it. The cost is that compression resolves either way, so you
wait for the expansion bar to declare direction.

---

## Alerts

Four channels, fanned out from one dispatcher: in-app banner + sound, Telegram,
Windows toast, and email. Every alert carries the strike, entry, target, stop,
quantity, rupee risk/reward, the reason string, and — critically — the feed's
latency note, so a delayed-feed alert never reaches your phone without that
warning attached.

**De-duplication is what makes this usable.** A 5-minute bar re-evaluated every
15 seconds is twenty identical alerts per setup per channel; one breakout across
four channels would be eighty messages. Two guards: a dedupe key of
`(strategy, direction, strike, bar timestamp)` and a per-strategy cooldown. Kill
switch alerts are exempt — a rate-limited kill switch is one that did not fire.

---

## Roadmap

**Phase 1 — Data ✅ scaffolded, ⬜ real history still needed**
`DataFeed` ABC with CSV/Parquet replay, yfinance fallback, and Fyers/Dhan
adapters. `BarReplayer` enforces the pivot confirmation delay (`find_pivots`
uses a centred window — live you only know a pivot `lookback` bars later), and
`tests/test_lookahead.py` asserts that property directly. **What remains: pull
≥18 months of real Fyers or Dhan minute data.** The sample generator produces a
random walk; it exercises the pipeline and proves nothing about edge.

**Phase 2 — Backtest ✅**
Walk-forward, train 6 / test 2, rolling; only test windows scored. Reports win
rate, expectancy, max drawdown, longest losing streak, and profit factor **after
`costs.py` friction**. Intrabar ties resolve to the stop, pessimistically.

**Phase 3 — Option pricing in backtest ⚠️ partial, and honestly labelled**
There are still no historical option chains. Two modes instead:
`UNDERLYING` measures whether the setup reaches +2R before −1R on the
underlying — the only question the signal layer can answer, and trustworthy.
`SYNTHETIC_PREMIUM` prices that move through Black-Scholes at one flat IV and
is **optimistic by 15–25%**, exactly as this section originally warned. The UI
prints that sentence beside every number the mode produces. Neither is evidence
to trade live capital.

**Phase 4 — Paper trade**
Minimum 40 live-market sessions with real WebSocket data and simulated
fills. Compare realised slippage against your model. Do not skip this.

**Phase 5 — Live, minimum size**
Register the strategy with your broker for an Algo-ID. Whitelist a static
IP or VPS. Start at half the computed size for 20 sessions.

---

## Compliance checklist (mandatory since 1 April 2026)

- [ ] Broker API that is registered under the retail algo framework
      (Fyers / Dhan / Angel One / Zerodha — **not** IndMoney or Lemon)
- [ ] Strategy registered via broker → exchange Algo-ID
- [ ] Static IP or registered VPS whitelisted in the broker dev console
- [ ] Order rate held under 10 orders/sec (trivially true here)
- [ ] White-box logic, own account only — no separate SEBI registration needed

## Non-negotiables

| # | Rule | Where it is enforced |
|---|---|---|
| 1 | Kill switch on any unhandled exception, data gap, or broker error | `engine.run_once()` catches everything and calls `trip_kill_switch()`; `DataFeed.detect_gap()` finds holes in the bar stream. Re-arming is manual — a kill switch that clears itself is a warning you never read. |
| 2 | Flat before 15:10. Never carry an intraday option overnight | `engine._maybe_force_exit()` fires a force-exit alert at `session.force_exit` |
| 3 | Every order, fill, and rejection in an append-only journal | `journal.py` — JSONL, one file per day, opened `"a"`, never rewritten. **Rejections are logged as carefully as approvals**: "risk refused 14 setups because no strike fit the budget" is a finding, and it is invisible if you only log fills. |
| 4 | No live capital until paper slippage matches the model | Nothing in this package can place an order. Log paper fills from the Live page and accumulate the 40 sessions. |
| 5 | Never buy an option with a spread wider than 0.5% of premium | `RiskEngine.select_strike()` — but note this gate **cannot reject a synthetic quote**, because the synthetic chain fabricates its own spread and OI to pass. Alerts built on a synthetic chain say so. |

## Testing

```bash
pytest                    # 50 tests
```

The suite that matters most is `tests/test_lookahead.py`. Every other failure
gives you a wrong answer you can see; look-ahead bias gives you a *flattering*
one you cannot, and it invalidates every backtest number silently. Those tests
build a frame with one unmistakable swing high and assert the replayer hides it
until the confirming bars have printed.

---

*Not investment advice. SEBI data shows over 90% of retail F&O traders lose
money. This is engineering scaffolding for testing an idea, not evidence
that the idea works.*
