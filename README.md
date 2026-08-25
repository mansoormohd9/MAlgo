# Nifty Intraday Option-Buying System

A from-scratch algo trading system for NSE index options, sized for a
₹1,00,000 account under SEBI's April 2026 retail algo framework.

Run `python -m nifty_algo.demo_risk` first. It prints the actual economics
of your rules before you write any strategy logic.

**Entries need your click. Exits do not.** Nothing becomes a position on its
own — you press a button. Once you are in, the breakeven shift, the partial at
+2R and the trailing stop run automatically, because a stop that needs a human
to press a button is not a stop.

**Two books, two switches, both defaulting to dry run.** The Indian swing book
can place real cash-equity orders through Kite (`EquityBrokerConfig.dry_run`);
the intraday option book still cannot (`BrokerConfig.dry_run`). Going live on a
multi-day equity book is a different decision from going live on an intraday
option book, and a single flag would have made it one decision.

**One click arms a resting trigger — it does not buy now.** Every swing setup
puts its entry *above* the last close, because each one demands the stock prove
itself before you pay for it. So the order waits, and **Zerodha** does the
waiting: between your click and the fill, nothing of yours needs to be running.

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

python -m nifty_algo.demo_risk        # the economics — read this first
streamlit run app.py                  # the console
```

That runs on the synthetic chain with no data. To use real data:

```bash
copy .env.example .env                # add KITE_API_KEY / KITE_API_SECRET
python -m nifty_algo.broker.kite_login    # once per trading day
python scripts/fetch_history.py           # ~4 years of real 5m NIFTY bars
python scripts/fetch_vix.py               # India VIX, for backtest IV
python -m nifty_algo.brief                # the day's frame + scored chain
```

For alerts that reach you with the browser closed:

```bash
python -m nifty_algo.run_live --provider kite --telegram
```

Streamlit cannot notify you when the tab is shut. That is why the decision
loop lives in `engine.py` and the UI is only a viewer over it — both drivers
run identical code. Note that `run_live` attaches **no broker**: it will use a
real chain when Kite is authenticated, but it can never place an order, because
reading quotes and sending orders are separate permissions.

### The nine pages

| Page | What it does |
|---|---|
| **Live alerts** | Session governors with the live ratcheting floor, open positions and where their stops actually sit, alert cards with a **Place order** button, candlestick chart with the levels/trendlines/VWAP the strategies used, and a *why nothing fired* table |
| **Daily brief** | Pre-open frame (gap, ATR, VIX, expiry, the day's rupee budget); the option chain with **which gate each strike fails**; and a journal-driven review of any past day |
| **Daily picks** | A different book: once a day it sweeps one market's universe — **India** (Nifty 100), **US** (the union of SPUS + HLAL constituents) or **UK** (FTSE 100) — applies a **halal (Shariah) screen** under both the FTSE/Yasaar and AAOIFI standards, and returns at most three cash-equity LONG **swing** tickets, priced in that market's currency and in rupees, with an overlap check against the Shariah ETFs you already hold and a ledger accounting for every symbol |
| **Trade book** | **What you actually own.** The morning checklist (Kite token, CDSL authorisation, free cash), open positions with where the stop really sits, armed triggers still waiting, every disagreement between the ledger and Zerodha, and the realised results of trades you *took* — as opposed to picks the scanner made |
| **Portfolio** | The cross-border picture no single ticket can show: the **US estate-tax meter** ($60,000 non-resident exemption, and which of your holdings count toward it), US-vs-Ireland domicile comparison, what your funds actually own, and LRS/TCS arithmetic. Arithmetic with citations, not advice |
| **Strategies** | Toggle each of the ten setups, tune every parameter, control the regime gate |
| **Backtest** | Two books, two panels. Intraday: walk-forward, metrics, equity curve, **and how the day rules behaved**. Swing: the same over daily equity bars, with its survivorship and point-in-time caveats printed above every number |
| **Journal** | The append-only record, filterable, CSV/JSONL export |
| **Settings** | The three capital pots, the live-orders switch, DDPI status, feed choice, notification channels, `.env` status |

---

## Your rules, resolved

| Rule | Value | Level |
|---|---|---|
| Session target | +10% (₹10,000) | **Close everything**, stop for the day |
| Session floor | ratcheting, starts −5% | **Close everything**, stop for the day |
| Max entries | 3 | Block new entries; keep managing what is open |
| Risk per trade | 1.67% (₹1,667) | *derived* = 5% ÷ 3 |
| Reward per trade | 3.33% (₹3,333) | *derived* = 10% ÷ 3 |
| Reward:risk | 2.00 : 1 | breakeven win rate 33.3% |

The three governors are self-consistent: three consecutive losses land
exactly on the session stop. "Max 3 orders" and "5% stop" are the same
constraint expressed twice — that's a feature, keep it.

Note the difference in the *Level* column. The target and the floor **flatten
open positions**; running out of entries does not. That distinction matters
because a runner can carry the day past +10% while it is still open, and
blocking entries would do nothing about it. `governor.py` returns `CLOSE_ALL`
for exactly that case.

### The give-back ratchet

> *"If I have a positive of 2% in the first order I would like to trail my
> stoploss from 5% to 3%, and likewise."*

Implemented as a give-back cap off the day's **peak realised P&L**, so it works
for any sequence of trades rather than needing a lookup table:

```
floor = max(-5% of capital,  peak - 5% of capital)
```

| Peak realised | Floor | The day stop now reads |
|---|---|---|
| ₹0 | −₹5,000 | 5% — the opening budget |
| **+₹2,000** | **−₹3,000** | **3% — your stated rule** |
| +₹4,000 | −₹1,000 | 1% |
| +₹6,000 | +₹1,000 | the day can no longer finish red |
| +₹8,000 | +₹3,000 | |

The "3%" is not a second parameter — it is what a 5% give-back from a +2% peak
reads as. The floor is monotonic because the peak is, so it never loosens. And
because the give-back equals the opening budget, you always have exactly three
trades' worth of room below the peak, which keeps the ratchet consistent with
`max_entries_per_session = 3`. Set `ratchet_arm_at_pct = 0.02` if you prefer
the stricter reading where trailing does not begin until +2% is banked.

## The constraint that shapes everything

```
max premium loss per unit = ₹1,667 / (65 × lots)
required delta            ≤ that / underlying_stop_points
```

**Lots is in the denominator.** Risk per trade is fixed, so doubling the size
does not double the risk — it *halves the delta you may buy*.

| Underlying stop | Max delta @ 1 lot | Max delta @ 2 lots |
|---|---|---|
| 30 pts | 0.85 → capped 0.45 | 0.43 |
| 50 pts | 0.51 | 0.26 |
| 64 pts | 0.40 | 0.20 — the boundary |
| 80 pts | 0.32 | 0.16 — **no viable strike** |

**The stop chooses the strike, not the reverse.** Every retail trader who
picks a strike first and then "sets a stop" has the causality backwards.

Sweet spot for this account: **60–80 point stop, 0.30–0.40 delta**, premium
₹90–150. Round-trip friction there is ~₹70, about 4% of your risk budget.

## Trade management

| Trigger | Action |
|---|---|
| Entry | Stop −1R, target +2R |
| +1R | Stop to breakeven — the trade can no longer lose |
| +2R | **Bank one lot**; the rest becomes a runner |
| Runner | Stop trails 1 ATR behind the underlying, **ratchet only** |
| Stop touched | Exit whatever remains |
| 15:10 | Force exit, unconditional |

**The runner needs two lots.** NSE fills whole lots only, and NIFTY is 65 — you
cannot sell 32 of one lot, so "bank half" is impossible on a single lot.
`approve()` therefore sizes `preferred_lots` (2) and falls back to 1 when the
halved delta ceiling leaves no viable strike. On the fallback the runner is
**disabled** and 2R is a full exit; the alert says so in `sizing_note`, because
a trade that silently became a plain 2:1 is one you would misread all day.

`ExitLadder` in `positions.py` is written in **R and nothing else** — no
premiums, no lots, no rupees. The live engine feeds it R computed from the
option premium; the backtester feeds it R computed from the underlying. One
state machine, so there is exactly one implementation to be wrong.

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
| `costs.py` | Indian statutory charges + slippage, charged **per leg** so a scale-out pays its extra brokerage. |
| `signals.py` | Pure functions: ATR, pivots, levels, trendlines, VWAP, EMA, compression, sweeps, gaps. |
| `strategy.py` | `Strategy` ABC + `LevelBreakStrategy`. **Unchanged — do not edit.** |
| `strategies/` | Eight added setups + the registry the UI and backtester share. |
| `risk.py` | Strike selection, sizing, approval. Delegates the day rules to `governor.py`. |
| `governor.py` | The day rules: target, the ratcheting give-back floor, entry count. Pure arithmetic, separately testable. |
| `positions.py` | `ExitLadder` (unit-free, in R) + `PositionManager`. Breakeven shift, partial, trail. |
| `regime.py` | Day classifier — decides which strategies may speak today. |
| `pricing.py` | Black-Scholes, plus `implied_vol()` — premium → IV → delta, per strike, which is how skew becomes visible. |
| `data/` | `DataFeed` ABC, CSV replay + `BarReplayer`, **Kite**, yfinance, Fyers/Dhan, chain provider. |
| `swing/markets.py` | The market registry — universe file, benchmark, currency, quote divisor, taxonomy, capital pool. Add an exchange here, not with an `if`. |
| `swing/fx.py` | Rupees per foreign unit, for sizing. **Fails closed** — no trusted rate, no scan. |
| `swing/halal_taxonomy.py` | Two activity tables: NSE labels and Yahoo/GICS ones. They share almost no strings. |
| `swing/holdings.py` | What SPUS and HLAL hold, so a "new" position that you already own says so. |
| `swing/crossborder.py` | LRS/TCS, US estate-tax exposure, withholding by domicile, UK SDRT. Every rate date-stamped. |
| `broker/` | **The only package that can spend money.** Kite auth, real chain, order placement. |
| `alerts/` | `TradeAlert`, four channels, and the de-duplicating dispatcher. |
| `engine.py` | The decision loop. Headless; driven by both the UI and `run_live`. |
| `brief.py` | The daily brief — CLI and the Streamlit page share this one implementation. |
| `journal.py` | Append-only JSONL, one file per trading day. |
| `backtest.py` | Walk-forward, two modes. Read its docstring before its numbers. |
| `run_live.py` | Headless runner — alerts with the browser closed, no broker attached. |
| `demo_risk.py` | Prints the economics. Start here. |
| `../scripts/` | `fetch_history.py`, `fetch_vix.py` — one-time and daily data pulls. |
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

## Market data: what is possible, and what is not

**Zerodha Kite Connect** — ₹500/month per API key, with historical data bundled
into that price since February 2025 (it used to be a separate ₹2,000/month
add-on). Longest track record of the Indian broker APIs and the largest
ecosystem. `python -m nifty_algo.broker.kite_login` once each trading morning:
the access token is issued per login and dies overnight, there is no refresh
token, and any library claiming otherwise is scraping the login form.

**The hard limit, and it shapes the whole backtest.** Kite serves **no
historical data for expired option contracts** — once a weekly expires, its
instrument token is retired. Zerodha have said on their own developer forum
that they have no plans to add it. There is no route to a premium-level options
backtest through Kite at any subscription tier. If you want real 1-minute option
history, that is a paid vendor (TrueData, GlobalDatafeeds; roughly ₹1–3k/month),
and it is a clean swap at the feed layer rather than a rewrite.

Index tokens, by contrast, never expire — which is why `scripts/fetch_history.py`
can pull years of real NIFTY 5-minute bars, and why the backtest is built on
**real index history with modelled premiums** rather than the reverse.

**Two things the index feed does not give you.** It has no traded volume — an
index is a computed average and nothing trades in it — and Kite returns 0. Left
unhandled that was three silent failures at once: `volume_surge` returned True
on every bar (`0 >= 0 × 1.5`), `vwap` returned all-NaN so the VWAP strategy
quietly never fired, and `underlying_liquidity_ok` blocked everything. Now
`signals.has_traded_volume()` detects it and the gates fall back to range
expansion, with VWAP degrading to a session TWAP. Same claim, different
measurement — the backtester and the brief both say so out loud.

**Not Lemonn.** It has no public developer API in India; its SmartInvest product
is no-code prebuilt strategies. The `lemon.markets` API that turns up in
searches is an unrelated European company.

## Roadmap

**Phase 1 — Data ✅**
`DataFeed` ABC with CSV/Parquet replay, **Kite**, yfinance fallback, and
Fyers/Dhan adapters. `BarReplayer` enforces the pivot confirmation delay
(`find_pivots` uses a centred window — live you only know a pivot `lookback`
bars later), and `tests/test_lookahead.py` asserts that property directly.
`scripts/fetch_history.py` pulls real NIFTY 5-minute history, resumably. The
sample generator is still a random walk and `CsvFeed` will **not** silently fall
back to it.

**Phase 2 — Backtest ✅, and it now tests the rules, not just the signals**
Walk-forward, train 6 / test 2, rolling; only test windows scored. Intrabar ties
resolve to the stop, pessimistically — including for the trailing stop.

The previous version built a `RiskEngine` per simulated day and never called
`register_exit`, so realised P&L stayed at zero, the target and floor could
never fire, and the only governor with any effect was the entry count. It
measured the setups. It did not measure the system. Now every simulated exit is
booked and the result carries a `DayStats` block: days that hit the target, days
that hit the floor, days the floor ratcheted, entries used, trades that banked a
partial.

**Phase 3 — Option pricing in backtest ⚠️ partial, and honestly labelled**
There are no historical option chains, at any price (see above). Two modes:
`UNDERLYING` measures whether the setup reaches +2R before −1R on the
underlying — the only question the signal layer can answer, and trustworthy.
`SYNTHETIC_PREMIUM` prices that move through Black-Scholes at **that day's India
VIX** rather than a flat 14%, which removes a bias that ran in opposite
directions across calm and violent periods. It still has no skew and still
fills every order, and is **optimistic by 15–25%**. Neither is evidence to trade
live capital.

Your rules survive this limitation better than most would, because they are
denominated in R and in rupees off the underlying — `UNDERLYING` mode tests the
ladder and the ratchet properly. It is the premium P&L that stays approximate.

**Phase 4 — Paper trade**
Minimum 40 live-market sessions with real WebSocket data and simulated
fills. Compare realised slippage against your model. Do not skip this.

**Phase 5 — Live, minimum size**
Register the strategy with your broker for an Algo-ID. Whitelist a static
IP or VPS. Start at half the computed size for 20 sessions.

---

## The swing book, live

```
Evening   Scan runs. Up to 3 tickets. You press Arm on the ones you want.
          -> a BUY GTT rests at Zerodha: "buy N if it trades through X"
          Close the laptop.

Any day   Zerodha watches the price. Not this app. It fires unattended.

Next run  Trade book -> Run reconcile & manage:
            reads holdings, records the fill and its SLIPPAGE
            arms the exit OCO (stop + target) for anything unguarded
            advances the ladder on today's bar
            pushes any ratcheted stop to the broker with a modify
            retires entries that never triggered inside their window
```

**The one thing that can silently break this: CDSL TPIN.** Unless **DDPI** is
active on your Zerodha account, every delivery sell — not just GTT, *any* CNC
sell — needs a TPIN authorisation that is **valid for one trading day**. A stop
GTT armed on Monday is rejected on Wednesday unless you re-authorised that
morning, and **Kite still shows it as active**. Nothing on the broker's own
screen tells you the stop is decorative.

So the app refuses to call a stop protected on a day the authorisation is not
recorded, and puts that warning on *every* page while a ticket is live.
Enabling DDPI is a free one-time e-sign and removes the whole problem.

| | Without DDPI | With DDPI |
|---|---|---|
| Buy trigger fires | works | works |
| Stop / target fires | **rejected** unless authorised that morning | works |
| Your daily job | Kite login **+ authorise holdings** | Kite login |

### What the swing backtest says

Read `python -m nifty_algo.swing.backtest --market india --years 3` before
arming anything. It prints its own caveats first — survivorship (the universe
file is *today's* index), point-in-time fundamentals, and news that cannot be
replayed — and they all run in the flattering direction.

On the ~2.5 years of cached Nifty 100 data at the time of writing it returned a
**negative expectancy of roughly −0.04R a trade at a 24–26% win rate, against
the 33.3% a 2:1 payoff needs to break even.** Per-setup figures flipped sign
between two runs with different capital, which is what 30-trade samples do.
That is not a reason to abandon the idea; it is a reason not to arm it with
money yet.

### Portfolio heat, which the option book never needed

`max_entries_per_session` caps entries *in a day*. A swing book holds for days,
so three scans on three days could put nine positions on. `SwingConfig.max_open_risk_r`
caps total R at risk across everything open, and a position whose stop has
reached breakeven contributes nothing to it — which is the point of moving it
there. The Trade book shows the figure against the cap.

## Compliance checklist (mandatory since 1 April 2026)

- [ ] Broker API that is registered under the retail algo framework
      (Zerodha / Fyers / Dhan / Angel One — **not** IndMoney or Lemonn)
- [ ] Strategy registered via broker → exchange Algo-ID
- [ ] Static IP or registered VPS whitelisted in the broker dev console
- [ ] Order rate held under 10 orders/sec (trivially true here — Kite caps at 3)
- [ ] White-box logic, own account only — no separate SEBI registration needed

`confirm_entry()` is human-initiated by design, which is why this is built as
one-click confirmation rather than autonomous execution. **Check your own
obligations with Zerodha before setting `dry_run = False`.**

### One operational dependency you must know about

`kite_orders.modify_stop()` deliberately does **not** place a resting SL order
at the broker. The stop trails on the underlying's ATR, so its premium level is
recomputed every bar, and a resting order would have to be modified on every one
of those bars — each modify being a request that can fail, be rejected, or race
the fill. The engine holds the stop and sends a SELL when it triggers.

**Which means the stop only exists while the engine is running.** That is a real
dependency, and you should know it rather than discover it.

## Non-negotiables

| # | Rule | Where it is enforced |
|---|---|---|
| 1 | Kill switch on any unhandled exception, data gap, or broker error | `engine.run_once()` catches everything and calls `trip_kill_switch()`; `DataFeed.detect_gap()` finds holes in the bar stream. Re-arming is manual — a kill switch that clears itself is a warning you never read. **`NotConfigured` is the one exception**: a missing file or an expired Kite token blocks loudly but does not latch, because a manual re-arm every morning before login trains you to click through the warning. |
| 2 | Flat before 15:10. Never carry an intraday option overnight | `engine._maybe_force_exit()` alerts at `session.force_exit`, and `PositionManager.update()` force-exits every open position at that time regardless of where the ladder stands. |
| 3 | Every order, fill, and rejection in an append-only journal | `journal.py` — JSONL, one file per day, opened `"a"`, never rewritten. **Rejections are logged as carefully as approvals**: "risk refused 14 setups because no strike fit the budget" is a finding, and it is invisible if you only log fills. Every stop move, partial and exit is journalled as `position_action`. |
| 4 | No live capital until paper slippage matches the model | `BrokerConfig.dry_run` defaults to `True` and nothing in the code path flips it. Run dry for the 40 sessions and compare the modelled premium in each alert against what actually filled. |
| 5 | Never buy an option with a spread wider than 0.5% of premium | `RiskEngine.select_strike()`. Against a **real** Kite chain this gate finally does work — `tests/test_kite_and_chain.py` asserts a wide-spread quote is rejected. On a synthetic chain it still cannot bite, because the synthetic chain fabricates its spread and OI to pass, and alerts built on one say so. |
| 6 | Exit alerts are never suppressed | `dispatcher._suppressed()` exempts every `POSITION_KIND`. Suppressing a duplicate *entry* costs an opportunity; suppressing a stop move or an exit costs the trade. |

## Testing

```bash
pytest                    # 540 tests
```

The suite that matters most is `tests/test_lookahead.py`. Every other failure
gives you a wrong answer you can see; look-ahead bias gives you a *flattering*
one you cannot, and it invalidates every backtest number silently. Those tests
build a frame with one unmistakable swing high and assert the replayer hides it
until the confirming bars have printed.

| File | What it pins |
|---|---|
| `test_lookahead.py` | The replayer hides unconfirmed pivots. Read this one first. |
| `test_governor.py` | The ratchet, row by row; the floor is monotonic; three losses land exactly on −₹5,000 |
| `test_positions.py` | Breakeven at +1R, partial at +2R, **the trail never loosens**, one bar crosses every rung it passed, the stop is tested at its pre-bar level |
| `test_engine_lifecycle.py` | Confirm → manage → exit → day over. Includes the case the old design could not handle: a runner alone reaching +10% while still open. |
| `test_kite_and_chain.py` | Kite candles become tz-naive IST; IV inversion round-trips and refuses sub-intrinsic quotes; **wide spreads and thin OI are rejected**; expiry comes from the dump, not the weekday guess |
| `test_volumeless_index.py` | The three silent failures on a volume-less index series, all three fixed |

---

*Not investment advice. SEBI data shows over 90% of retail F&O traders lose
money. This is engineering scaffolding for testing an idea, not evidence
that the idea works.*
