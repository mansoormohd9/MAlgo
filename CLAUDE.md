# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Windows, `.venv` at the repo root. Use `.venv\Scripts\python.exe` (or activate first).

```bash
pip install -r requirements.txt

python -m nifty_algo.demo_risk          # prints the account economics - read before touching risk code
streamlit run app.py                    # the console (7 pages)
python -m nifty_algo.run_live --provider kite --telegram   # headless alerts, never places an order
python -m nifty_algo.brief              # the day's frame + scored chain, CLI form of the Daily brief page
python -m nifty_algo.swing.scanner --market india|us|uk   # the swing book, headless
python -m nifty_algo.swing.backtest --market india --years 3 --capital 100000   # does the swing book work?
python -m nifty_algo.swing.experiment --market india --years 3 --capital 100000  # which rule set, chosen out-of-sample
python -m nifty_algo.experiment_intraday --variants baseline,gapdaily,novwap  # same question for the option book
python -m nifty_algo.experiment_intraday --report        # the table from the existing ledger, no re-run
python -m nifty_algo.intraday_equity.signal_grid --market india  # E5: is there ANY edge on 5m equity bars? (~35 min)
python -m nifty_algo.factor.verdict --seeds 500 --slippage 0.0025   # F1: is the factor sleeve real? (~40 min)
python -m nifty_algo.factor.drawdown                        # F2: does ANY drawdown instrument beat holding less? (~45s)
python -m nifty_algo.factor.sleeve --capital 500000         # what the sleeve wants THIS month, headless
python -m nifty_algo.factor.membership --refresh            # Nifty 50/500 constituent lists from NSE
python scripts/fetch_factor_fundamentals.py --shortlist 60  # balance sheets for every name ever shortlisted (~35 min)
python scripts/run_f3_screened.py                           # F3: what does the halal screen cost the sleeve? (~5s)
python scripts/run_f4_universe.py                           # F4: what does a Nifty 500 restriction cost, and how much of it is look-ahead? (~10s)
python -m nifty_algo.factor.sleeve --universe nifty500 --halal --capital 500000  # the restricted book, live
# F5 (the ATR trail) has no CLI of its own - it is `trail_atr_multiple` on
# `factor.backtest.run`, and it lost. See "F5" below before asking for a stop.
python scripts/run_s1_swing_null.py --seeds 40 --years 6 --capital 200000  # S1: does the swing book beat chance? (~55 min)
python -m nifty_algo.intraday_equity.signal_grid --horizon sessions   # E6: do intraday signals predict DAYS? (~40 min)
python -m nifty_algo.research macro --market india          # macro fact pack; --json is what a skill reads
python -m nifty_algo.research risk  --market india --json   # portfolio risk fact pack
python scripts/fetch_swing_history.py --market india --years 6  # deep daily bars for it
python scripts/build_universe.py --market us   # rebuild data/us_halal.csv from SPUS+HLAL
python scripts/build_universe.py --market uk   # rebuild data/ftse100.csv
python -m nifty_algo.broker.kite_login  # once per trading morning; token dies overnight
python scripts/fetch_history.py         # real 5m NIFTY history (resumable)
python scripts/fetch_vix.py             # India VIX, for SYNTHETIC_PREMIUM backtests
```

Tests (977: 972 passing, 5 skipped; pytest, `pythonpath = . tests` so `nifty_algo` and the conftest helpers both import without an install step):

```bash
pytest                                                     # the only run that counts as done
pytest tests/test_governor.py                              # while iterating: one file
pytest tests/test_positions.py::test_trail_never_loosens   # one test
pytest -k ratchet
```

There is no linter or type checker configured. Match the surrounding style: `from __future__ import annotations`, dataclasses, and a module docstring that explains *why* the module exists.

## Definition of done

A change is not finished when it compiles, or when the test you were thinking
of passes. Before reporting any code change complete, follow
[.claude/skills/definition-of-done/](.claude/skills/definition-of-done/SKILL.md) -
four steps, in order:

1. **Run the whole suite** - `pytest`, not `-k`, not the one file you touched. The two books share `signals.py`, `CapitalConfig`, `ExitLadder` and the journal; a targeted run is how the other book breaks silently.
2. **Assume the solution is wrong** - re-read the diff and write down a concrete reason it is broken, with a `file:line` and the input that breaks it, *before* any reason it is right. Every failure this repo has actually suffered was plausible rather than loud.
3. **Backtest if the traded path moved** - `swing/` or `signals.py` -> the swing backtest; `strategy.py` / `risk.py` / `positions.py` / `governor.py` -> the option backtest, now `python -m nifty_algo.experiment_intraday` (or the Backtest page, or `Backtester(cfg).run(bars)` in process). UI, broker and script changes are not backtestable; say so rather than skipping quietly.
4. **Report** what ran, what failed, what was not backtestable, and any test you modified.

If the change is meant to *improve* something rather than fix it, step 3b also
applies: name the metric, the baseline and the out-of-sample window **before**
writing the change, derive every yardstick from trades rather than config, and
choose between variants with `python -m nifty_algo.swing.experiment`, which
selects on a train window and scores on the test window after it.

## The two books

This repo holds two unrelated trading systems that share indicators, risk sizing, and the journal - and nothing else.

**Intraday NIFTY option buying** (`nifty_algo/`, everything outside `swing/`) - alerts on 5-minute index bars, three entries a day, flat by 15:10.

**Daily multi-market swing** (`nifty_algo/swing/`) - once a day, sweeps one market's universe, applies a Shariah screen, returns at most three cash-equity LONG tickets held for days. Three markets are registered in [markets.py](nifty_algo/swing/markets.py): India (Nifty 100), US (the union of SPUS and HLAL constituents) and UK (FTSE 100). It reuses `signals.py` (pure `bars in, values out`, so daily bars work unchanged), `CapitalConfig`, and `Journal`. It deliberately does **not** use the engine, the option-strike machinery, or the session governors - those encode "intraday, three entries, flat by close".

Never re-declare risk numbers inside `swing/`. Two books sizing off two copies of the same governor is how they drift apart.

**The swing book can place orders; the option book still cannot.** `swing/` is
wired to Kite for NSE cash equity ([kite_equity.py](nifty_algo/broker/kite_equity.py),
[book.py](nifty_algo/swing/book.py), [daily.py](nifty_algo/swing/daily.py)),
under its own `EquityBrokerConfig.dry_run` which defaults `True`. The option
path in `kite_orders.py` is untouched and stays dry-run under the separate
`BrokerConfig.dry_run`. Two switches, deliberately - going live on a
multi-day equity book is a different decision from going live on an intraday
option book, and one flag would make it one decision.

## Where the stop lives, and why the two books disagree

The intraday stop trails on the underlying's ATR and is recomputed every
5-minute bar, so `kite_orders.modify_stop()` rests **no** order at the broker:
a resting SL would need modifying on every bar, and each modify can fail, be
rejected, or race the fill. Consequence - the stop exists only while the
engine runs.

A swing stop moves at most **once a day**, so `kite_equity` does the opposite
and rests a two-leg OCO GTT at Zerodha. That stop survives the laptop being
shut, which is the entire point of a book you check once a day. Same
reasoning, opposite conclusion, two files - not one file with a flag.

**One OCO per position, and the +2R partial is NOT at the broker.** Zerodha's
two-leg GTT sells one quantity at one of two triggers; it cannot express "bank
half here, run the rest to there". So the resting OCO carries the stop and the
structural target, and the partial is taken by `daily.run()`. The split
follows the risk: a stop prevents a loss and must never depend on the app
being open, a partial banks a profit and can wait a day.

## The constraint that shapes the whole equity path: CDSL TPIN

Unless **DDPI** (or the older POA) is active, every delivery sell at Zerodha
needs a CDSL TPIN authorisation - not just GTT, *any* CNC sell - and it is
valid for **one trading day**. A stop GTT placed on Monday is rejected on
Wednesday unless the account was re-authorised that morning after 07:00, and
**Kite still displays it as active either way**.

That is the "looks armed and is not" failure this repo refuses to tolerate
elsewhere, so: `kite_equity.protection_state()` returns three states, never
two, and can only reach `PROTECTED` with DDPI on. A recorded authorisation
reaches `UNVERIFIED` and no further, because Kite Connect exposes no endpoint
to confirm it. `app.py._protection_banner` puts the warning on **every** page
while a ticket is live - a warning confined to the Trade book page is one you
see only when you were already looking.

Buys are unaffected. Only sells need TPIN.

## Architecture: the invariants

```
Data --> Signals --> Strategy --> RiskEngine --> Execution --> Journal
                                      ^
                                 hard gate
```

1. **`strategy.py` is imported unchanged by both the backtester and the live runner.** If you write `if backtest:` inside a strategy, the backtest has stopped measuring the system being traded. The same rule holds for `risk.py`. The two original strategies in `strategy.py` are *wrapped*, not edited - see the adapters at the top of [registry.py](nifty_algo/strategies/registry.py).
2. **`RiskEngine.approve()` is the only path to an order.** The strategy proposes a `Signal`; risk disposes - strike selection, sizing, stop, target, session governors, liquidity gates. A `Signal` without an approved order is journalled and discarded. Strategy code cannot reach anything the risk engine enforces.
3. **Every tunable number lives in [config.py](nifty_algo/config.py).** Nothing hardcoded in strategy logic, or it cannot be swept in a backtest. The per-trade risk/reward levels are `@property`-derived from the three session governors so they stay mutually consistent - don't turn them into independent fields.
4. **[engine.py](nifty_algo/engine.py) is headless** - it imports nothing from Streamlit. `app.py` and `run_live.py` are two drivers over the same `TradingEngine`, which is why UI alerts and headless alerts are produced by identical code. [nifty_algo/ui/](nifty_algo/ui/) renders and decides nothing.
5. **Position management runs before signal generation** in `engine.run_once()`. Open money outranks a new opportunity; evaluating entries first would let a fourth trade be proposed on a bar where the day had already ended.
6. **`ExitLadder` in [positions.py](nifty_algo/positions.py) is written in R and nothing else** - no premiums, no lots, no rupees. Live feeds it R computed from the option premium; the backtester feeds it R computed from the underlying. One state machine, so there is exactly one implementation to be wrong.
7. **[broker/](nifty_algo/broker/) is the only package that can spend money.** `BrokerConfig.dry_run` defaults to `True` and nothing in the code path flips it. `run_live.py` attaches no broker at all.
8. **Entries require a human click** (`engine.confirm_entry()`); exits - breakeven shift, partial at +2R, trailing stop, 15:10 force exit - are automatic and must stay that way.
9. **The journal is append-only JSONL**, one file per day, opened `"a"`, never rewritten. Rejections are logged as carefully as approvals: "risk refused 14 setups because no strike fit the budget" is a finding, and it is invisible if you only log fills. The same principle drives the swing scanner's rejection ledger and the Live page's "why nothing fired" table.
10. **Registries are single sources of truth.** Strategies are enumerated from [registry.py](nifty_algo/strategies/registry.py) by the UI, engine and backtester alike; feeds are built only through [factory.py](nifty_algo/data/factory.py). A strategy must never be live-tradeable but un-backtestable, or tunable in the UI but absent from the engine.

## Domain constraints that shape the code

- **Lots is in the denominator.** `max premium loss per unit = risk_per_trade / (lot_size x lots)`, and required delta <= that / underlying stop points. Doubling size *halves the delta you may buy*. The stop chooses the strike, not the reverse.
- **The runner needs two lots.** NSE fills whole lots (NIFTY = 65), so "bank half" is impossible on one lot. `approve()` sizes `preferred_lots` = 2 and falls back to 1, which disables the runner and makes 2R a full exit - the alert says so in `sizing_note`.
- **The give-back ratchet** is `floor = max(-session_stop_pct, peak - ratchet_giveback_pct)` off the day's *peak realised* P&L - monotonic, never loosens. The "3% at +2%" rule is what a 5% give-back reads as, not a second parameter. Lives in [governor.py](nifty_algo/governor.py), which is pure arithmetic and separately testable.
- **The index carries no traded volume** - Kite returns 0. Unhandled, that silently broke three gates at once (`volume_surge` true on every bar, all-NaN VWAP, liquidity blocking everything). `signals.has_traded_volume()` detects it and the gates fall back to range expansion / session TWAP. Stocks in the swing book *do* have real volume, so the same functions behave differently there by design.
- **There is no historical option chain data at any price** (Kite retires expired instrument tokens). The backtester's `UNDERLYING` mode is the trustworthy one; `SYNTHETIC_PREMIUM` is optimistic by 15-25% and must stay labelled as such everywhere it surfaces.
- **Kite's access token is issued per login and dies overnight** - there is no refresh token. `NotConfigured` (missing file, expired token) is the one exception to the latching kill switch: it blocks loudly without requiring a manual re-arm, because a re-arm every morning trains you to click through warnings.
- **Streamlit re-runs the whole script on every interaction**, so the engine, dispatcher and caches live in `session_state` via [state.py](nifty_algo/ui/state.py). A fresh `RiskEngine` per rerun would silently reset `entries_taken` and realised P&L (governors never trip); a fresh dispatcher would re-send every alert.
- **Alert de-duplication is load-bearing**: key is `(strategy, direction, strike, bar timestamp)` plus a per-strategy cooldown. Kill-switch and every `POSITION_KIND` alert are exempt - suppressing a duplicate entry costs an opportunity, suppressing a stop move costs the trade.
- **The swing scanner orders its gates cheap-to-expensive** (`universe -> halal -> prices -> tradeability -> setup -> R:R -> earnings -> news -> rank -> sector cap -> size`) so a stock that was never going to qualify never costs an HTTP request. News is fetched for finalists only, and "nothing notable published" and "feed unreachable" are distinct facts (`available`) - conflating them would read as a clean bill of health on every stock at once.

## Multi-market: the four things that bite

The setup detection, the scoring and the whole of `signals.py` are currency-blind and needed no changes - a daily bar is a daily bar. Everything that did need changing was a number or a label that was secretly India-only, and all four failure modes below produce a *plausible* answer rather than an exception.

- **The benchmark cache is keyed by ticker, and there is one parquet per market.** It used to be one file with the benchmark under a fixed `__BENCHMARK__` key, accepted whenever it held every symbol asked for. Scan the US then India and that key is present, so India's relative strength gets computed against the S&P 500 with no error anywhere. Both halves of the fix are load-bearing. Regression: `test_a_us_cache_cannot_satisfy_an_india_scan`.
- **The LSE quotes in PENCE and yfinance passes it through.** `SHEL.L` arrives as 2500 meaning £25.00. Normalised once, at ingest, in `prices._extract` - `Market.price_divisor` is the default and `fundamentals.currency` ("GBp" vs "GBP") overrides it per symbol, because not every LSE line is in pence. Never divide the benchmark: an index is quoted in points.
- **[fx.py](nifty_algo/swing/fx.py) fails closed, and the scan stands the market DOWN.** `_size()` divides a rupee budget by a foreign stop distance; without conversion a US ticket is ~88x too large and looks entirely ordinary. There is no fallback rate - not a hardcoded 83, not the last rate seen. `ScanResult.stood_down` is a distinct state from "nothing qualified", and the page and CLI say which. The pots being divided are in **Three capital pools, one formula** below.
- **Everything cached or journalled is keyed `{market}:{SYMBOL}`.** Bare tickers collide across exchanges, and a collision is not a crash - it is one company screened against another's balance sheet.

## The halal screen across vocabularies

- **Two activity tables, not one bigger one** ([halal_taxonomy.py](nifty_algo/swing/halal_taxonomy.py)). NSE says "Non Banking Financial Company (NBFC)"; Yahoo says "Credit Services". The GICS table is deliberately *not* a transliteration: `"gaming"` is right for NSE (Delta Corp's casinos) and wrong for Yahoo, where it also matches "Electronic Gaming & Multimedia" - video games, which no mainstream Shariah index excludes.
- **Contested categories are toggles with the standard named**, because the two ETFs in this portfolio disagree: SPUS (AAOIFI/S&P) excludes aerospace & defence and financial exchanges & data, HLAL (FTSE/Yasaar) does not. Hardcoding either would be the code quietly picking a madhhab.
- **Both ratio methodologies are computed; only the primary gates.** FTSE/Yasaar divides by total assets (what HLAL tracks, and what [halal.py](nifty_algo/swing/halal.py) already did); AAOIFI/S&P divides by market cap (what SPUS tracks). `HalalVerdict.verdicts` holds both and `.disagreement` flags the split. `eligible` follows `primary_method` and nothing else - if that slips, changing what is *displayed* changes what is *tradeable*. Screening against total assets is also why a verdict changes when a balance sheet lands rather than when the share price moves.
- **A missing fact is never a pass.** The screen cannot verify the haram-revenue test at all, so every verdict carries `haram_revenue_verified = False`; absent fundamentals are a failure to verify, not a clean bill of health.
- **The US universe is the union of two professionally screened funds**, which makes the local screen a re-verification rather than the only line of defence. Expect disagreements and treat them as findings: the largest class is the debt test, because Yahoo's "Total Debt" includes capitalised operating leases while FTSE/Yasaar screens interest-bearing debt only. Lease-heavy names therefore fail here and pass there. The excluded table flags every name the funds hold. **Do not tune the threshold to make the numbers agree** - that is fitting the screen to the answer.
- **[holdings.py](nifty_algo/swing/holdings.py) is a badge, never a rejection.** If you hold SPUS and HLAL, a "new" NVDA position is not new. Concentration is your decision; what the code refuses to do is let it be made silently.
- **[crossborder.py](nifty_algo/swing/crossborder.py) is arithmetic with citations, not advice.** Every rate carries `VERIFIED_ON`. The one that matters most is not a cost but an exposure: US-domiciled ETFs and direct US shares are US-situs assets against a $60,000 non-resident estate tax exemption, with no India-US estate treaty; an Ireland-domiciled UCITS holding the same companies is outside the regime entirely.

## Three capital pools, one formula

`CapitalConfig` holds three balances - `starting_capital` (the option
account), `swing_capital_inr` (Indian cash equity) and `foreign_capital_inr`
(LRS money in another broker) - reached through `capital_inr(pool)` and
`risk_inr(pool)`. The **formula** is still in exactly one place:
`risk_per_trade_pct = session_stop_pct / max_entries_per_session`, applied to
whichever balance is paying. Adding a second formula is how two books end up
with two versions of the same rule.

An unfunded pool **stands the market down** rather than sizing off another
pot, and `_pool()` now **raises** on an unknown key - it used to return the
domestic balance for any typo, which sized a trade off the wrong account and
produced an entirely plausible ticket.

`SwingPick` carries both currencies (`risk_amount` in the market's, `risk_inr`
in rupees); the intraday book's `rupee_risk` is untouched and really is always
rupees.

`Market.is_home` means *domestic currency* and is now a stored `domestic`
field, **not** `capital_pool == POOL_HOME`. Those were two ideas that happened
to coincide while there were only two pools; India needed its own pot while
staying rupee-denominated, and deriving one from the other would have switched
on the whole LRS/estate-tax panel for a Mumbai trade.

The pot must be **big enough for `top_n` positions**. Cash per ticket is
`risk / stop%`, so at a 4% stop one position is over 40% of the pot and three
will not fit. The scan proposes them anyway and the broker refuses the third
when its trigger fires; `page_settings._pot_note` does that arithmetic for
you, and the backtest counts the refusals in `capital_blocks`.

## The swing backtest, and what it cannot measure

`swing/backtest.py` calls `scanner.evaluate_symbol()` and
`scanner.rank_and_size()` - the same functions `scan()` calls, extracted for
exactly this reason. `setup.detect()` is handed a progressively truncated
frame **unchanged**; there is no `if backtest:` in the path, and
`test_swing_backtest.py::test_truncating_the_future_changes_nothing` asserts
that cutting the data short leaves every already-finished trade byte-identical.

Three distortions are structural and are printed above **every** result rather
than living here: survivorship (the universe file is today's index),
point-in-time fundamentals (today's balance sheets applied to all history),
and news (not replayable - its weight is redistributed through `_score`'s
existing unavailable path). Treat them the way `SYNTHETIC_PREMIUM` is treated.

Two things it *does* enforce that are easy to leave out: the entry never fills
below the next session's **open** (a gap through the trigger fills at the
open), and total deployment never exceeds the pot - checked against the
**fill**, not the planned entry, because a gap-up otherwise deploys money the
account does not have.

**The pot is part of the experiment.** `swing_capital_inr` defaults to 0 and a
zero pot sizes every ticket to zero, so a run without `--capital` reports "no
trades" and reads like tight gates. At Rs 1,00,000 the India run refuses **795
entries for want of cash against 282 taken** - the pot decides *which* trades
the book gets, not merely their size, so two runs at different `--capital` are
two different experiments.

## S1: the swing book lost to its own null, and the +0.114R was a window

**Every book in this repo is now tested against a null that shares its
machinery**, and the swing book was the last to face one. `SwingConfig.random_seed`
replaces `scanner._score` with noise and changes nothing else - same gates,
trigger, stop, ladder, sizing, sector cap and charges - so a seeded run differs
in the SCORING alone. It is applied in `_apply_null` AFTER the scan cache is
read, and `random_seed` is in `BOOK_ONLY_SWING_FIELDS`, so 41 runs per fold cost
**one** scan pass. Without that the 40-draw null is unaffordable.

Result over 33 out-of-sample windows at Rs 2,00,000: **12/33 folds won against
16.5 expected by chance, mean percentile 42%, sign-test p = 0.163.** The null
did not tie the book, it beat it - the same verdict the intraday equity book
got, now on daily bars.

**THE +0.114R WAS THE LAST TWO YEARS.** Identical code and data:

| window | folds | trades | expectancy |
| --- | --- | --- | --- |
| 2024-09 -> 2026-09 | 9 | 74 | **+0.111R** |
| 2020-09 -> 2026-09 | 33 | 361 | **-0.094R** |

The fold table shows the shape: the book wins the **last seven folds straight**
(percentiles 90/60/88/92/50/82/80) and loses nearly everything from 2022-09 to
2025-07 (8-35%). Capital is not the explanation - Rs 1L and Rs 2L agree to
0.01R. A `git stash` A/B over the same window reproduced **+0.111037R over 74
trades on both sides, trade for trade**, so it is not a code change either.
This is exactly the "carried by one good stretch" failure the sign test exists
to catch, and it went unnoticed because the book had never been scored against
anything but itself.

**THE HOLDING-PERIOD ANSWER: median 5 days, mean 8.2 - and NOT ONE `target`
EXIT in 356 trades.** Every exit is the initial, breakeven or trailing stop; the
structural target is never reached. "Average days to target" is unanswerable
because there are no targets, and a 2R reward:risk that never pays 2R is a
design figure rather than an outcome.

**`walk_forward_report` takes a `unit` because the two books measure different
things.** The factor sleeve scores a window in chained period return; the swing
book in expectancy. Formatting the second as a percent printed +0.369R as
"+36.90%" - complete, plausible, a hundredfold wrong.

## S2 and E6: the exit was not the problem, and neither was the horizon

**S2 tested the exit rule directly** - an ATR stop trailed from entry, no
breakeven rung, no partial, no target, hard 10-day cap - against its own 40-draw
null on the same 33 folds:

| | baseline | `puretrail` |
| --- | --- | --- |
| folds won vs median null | 12/33 | **8/33** |
| two-sided sign p | 0.163 | **0.005, in the WRONG direction** |
| mean percentile | 42% | 41% |
| expectancy | -0.103R | -0.099R |

The exit made it **worse**. 89% of trades stop out at -0.292R in a median 3
sessions; the 11% reaching the cap show +1.423R, which is survivorship - those
are simply the trades that had not stopped, and the cap truncates them.

**"TRAIL FROM ENTRY" WAS NOT EXPRESSIBLE, AND REMOVING THE TARGET WOULD HAVE
KILLED THE TRAIL SILENTLY.** `LadderMode.TRAIL` was reachable only through the
+2R partial rung, so raising `partial_exit_at_r` removed every trailing update
and left positions on their initial stop - the trap `ShortPremiumConfig` already
records for another book. `TradeManagementConfig.trail_from_r` (None = no-op) is
the fix and `tests/test_positions.py` pins the old behaviour so it cannot
return. Note it SUPERSEDES the breakeven rung and, at 1.0 ATR trail against a
1.5 ATR stop, tightens the stop from -1.0R to -0.667R the moment a trade opens.

**A `SwingConfig` field can silently do nothing.** `partial_exit_at_r` lives on
`TradeManagementConfig`; a variant setting it on `SwingConfig` changed nothing
while its name claimed the partial was gone. Only printing the RESOLVED trade
config caught it. `test_swing_overrides_actually_reach_the_ladder` now asserts
every override arrives.

**THE SIGN TEST IS TWO-SIDED - ASSERT THE DIRECTION SEPARATELY.** 8/33 folds
scores p = 0.005 and a gate testing the p alone printed **PASS**.
"Significantly different from chance" is not "better than chance".

**E6 asked whether an intraday signal predicts DAYS**, since E5 only ever looked
100 minutes ahead. Same signals and null, shifted along the SESSION axis at
horizons of 1/3/5/10 sessions:

| | E5 (100 min) | E6 (1-10 sessions) |
| --- | --- | --- |
| best excess | +0.0154% | **+0.0829%** |
| family-wise p | 0.0108 | **0.0731 - fails** |
| toll to clear | 0.0827% MIS | **0.3378% DELIVERY** |
| shortfall | 5.4x | **4.1x** |

**The move got 5x bigger and so did the toll**: an overnight hold is CNC, so STT
lands on both legs and the flat DP charge applies. The friction argument for a
longer hold was right about the numerator and wrong to ignore the denominator.
The family-wise p also got WORSE, not better.

## Choosing between rule sets: the yardstick and the folds

**The breakeven win rate is derived from the trades, not from the config.**
`compute_metrics` computes it as `avg_loss / (avg_win + avg_loss)` and keeps
the design figure `1/(1+R:R)` beside it as `target_breakeven_win_rate`. They
are ten points apart on the current book: the ladder shifts to breakeven at
+1R, so it produces small losses and medium wins rather than 2:1 trades, and
judging it against 33.3% turned a 2.7-point miss into an apparent 10.6-point
rout. `book.Performance.yardstick()` does the same for the live book and falls
back to the design figure - saying so - under
`MIN_TRADES_FOR_REALISED_BREAKEVEN`.

**A walk-forward window must SETTLE, or it deletes winners.** `end` is the
last day a position may be *opened*; `run(..., settle_days=60)` then keeps
managing until those trades finish. Anything still open when a window closes is
excluded from the statistics, and what is open at a boundary is not a random
sample - a loser hits its stop in days, a winner trails for weeks. Measured on
the same variant over the same window with settlement as the only difference:
**-0.132R over 236 trades against +0.043R over 308**, so the bias was 0.176R
and 23% of all trades. It made every variant look worse than it was and it
raised no error at all.

**`swing/backtest.run()` is one in-sample pass; `fold_windows()` and
[experiment.py](nifty_algo/swing/experiment.py) are how a choice gets made.**
Variants are selected on a train window and scored on the test window that
follows, because picking the best of eight variants on the data that measured
them is fitting, not evidence. The out-of-sample column is the result; the
train-test gap is how much of a variant was fitting.

**`ScanCache` makes that affordable, and refuses to be wrong quietly.**
`evaluate_symbol` is a pure function of (symbol, bars to date, scan config,
market) - it does not read the book - so one scan pass serves every variant
that only changes the ladder, the regime gate or the capital rules. The
duplicate-name filter moves to the consumer, which is what makes this safe.
`scan_signature()` hashes every `SwingConfig` field **except** an explicit
book-only list, so a new field invalidates the cache by default; a mismatch
**raises** rather than falling back, because a stale scan answers a different
question fluently. `test_a_replayed_variant_is_identical_to_a_direct_run` is
the guard.

**Budget the sweep by SIGNATURE GROUP, not by variant count.** Measured on the
India universe (98 symbols, 1355 sessions): one scanned session costs ~3.6s, so
a full pass is **~80 minutes**. Variants sharing a scan signature share that
one pass; each variant that changes the signals (`stopwide`, `breakoutonly`,
`pullbackonly`) adds another. Seven cache-sharing variants cost about what one
costs; running all ten in a single invocation costs four passes. The cost is
`setup.detect` recomputing every indicator over the whole truncated frame on
every session - O(n^2) in bars, and not fixable without bounding the frame,
which would change `build_levels` and is a behaviour change rather than an
optimisation. Two exact wins have already been taken: `relative_strength` uses
an index intersection rather than `pd.concat`, and the ranking-only metrics
(RS, 52-week position, volume ratio) are computed by `scanner._enrich` **only
for symbols that pass the gates** - they score, they never reject, and paying
for the ranking of 95 names in 100 that had no setup was a third of the cost.

**`--years` names the TEST WINDOW, not the fetch.** It used to set
`history_days` only, so a run labelled "3 years" scored every session the
parquet held - `fetch_swing_history` pulls six, and the India runs were
actually 1355 sessions, about 5.4 years. Both CLIs now pass a window start to
`run()`; the older bars stay loaded as warm-up, because trimming the bars
would fix the label and score the first sessions on half-converged EMAs. The
swing backtest also prints its true date range above the headline now - a
result that cannot say which period it covers is not a result.

**Anything long-running must be unbuffered end to end.** `python -u` alone is
not enough: a `| grep` in the pipeline buffers too, and the first attempt at
this sweep ran for 101 CPU-minutes having written zero bytes. Use `python -u
... | grep --line-buffered`, and the CLIs pass `flush=True` on their progress
prints.

**The regime filter is off by default.** `swing/market_regime.py` (not
`regime.py` - that one classifies an intraday session for the option book)
gates new longs on the benchmark being above its own moving average.
`regime_ma_days = 0` disables it, which is how every result before it existed
was produced. It fails closed: a missing or too-short benchmark blocks rather
than passes.

## One harness, three books

`experiment_core.py` owns everything about MEASUREMENT rather than about a
particular book: `Variant`/`set_on`, `Cell`, `Sweep`, the pooling rule, the
interval, the sign test, `selected_by_train`, the ledger and the table. Each
book keeps only its `VARIANTS` table and the loop that drives its own
backtester. Three copies of the statistics is how two books end up unable to
compare a result.

**Pooled expectancy is TRADE-WEIGHTED**, never a mean of fold means - a fold
with 3 trades and one with 60 are not equally informative.

**The interval resamples FOLDS, not trades.** Trades inside a fold share a
regime, a volatility level and often a session. Bootstrapping trades produces a
reassuringly narrow interval that means nothing.

**The SIGN TEST is the statistic that actually discriminates**, and it is the
one none of the earlier copies had. Measured on the option book: a warm-up
variant improved pooled expectancy from -0.051R to -0.019R and nearly halved
total loss - which reads like a result - and then won **10 of 21** walk-forward
windows against a coin-flip expectation of 10.5. A pooled average can be
carried by one good stretch; consistency across independent windows cannot.
Report both and believe the second.

**The option backtester is packed.** `_run_session` attaches an
`indicator_cache` pack per session, which is measured at **4.3x** on real bars
and asserted byte-identical in `tests/test_experiment_intraday.py`. The env
kill switch is `NIFTY_ALGO_DISABLE_PACK=1`, and `_DISABLED` is read once at
import - flipping it mid-process needs `refresh_disabled()`.

**A date range bounds what is SCORED, not what is loaded.**
`Backtester.run(start=, end=)` and `_run_window(score_from=, score_to=)` keep
every earlier bar visible to `prior_session` and to the warm-up window.
Slicing the frame instead cost the first session of every window - across a
21-fold sweep run in two phases that is 42 sessions deleted, quietly, and
always the same ones.

**What four years of real bars say about the option book.** 993 sessions,
2022-09 -> 2026-09: `level_break` is +0.006R over 1,364 trades with a 95%
interval of [-0.064, +0.075]. That is tight enough to rule out any edge above
~0.075R - a proven negative, not an unproven positive. And it is
`Mode.UNDERLYING`, which ignores theta and vega, so it is the GENEROUS case: a
strategy at zero on the underlying is negative once premium decay is paid.
89% of all trades are tagged `gap_day`, which is why the regime gate is the
largest untested lever - see the `gapdaily` variant.

## Searching for an edge, and paying for the search

**E5 closed the intraday equity book as a CATEGORY**, and it is the template
for how a search gets scored here.
[intraday_equity/signal_grid.py](nifty_algo/intraday_equity/signal_grid.py)
asks the prior question a backtest cannot: is there excess forward return after
ANY simple signal on these bars? 13 pre-registered signals x 3 session windows
x 6 horizons over 100 symbols x 739 sessions, with no scanner, ladder, sizing
or cost model in the path. Because it reads raw bars it can also see the
09:30-11:45 window the engine structurally cannot reach.

The answer: a REAL effect, an order of magnitude too small.
Family-wise **p = 0.0108**; best excess **+0.0154%** against **0.0827%**
statutory friction (**5.4x short**) and **0.1827%** with slippage (**11.9x**).
The book never failed for want of a signal - the largest honest edge on these
bars is a tenth of the cost of taking it. Two priors died with it: the morning
window really does carry the biggest excess (+0.0334%, so `warmseed` was
justified) and is still 2.5x short at zero slippage; and cross-section was the
WEAKEST of the four families, not the strongest.

**THE NAIVE t ON PANEL DATA IS INFLATED ~5x, and this is measured, not
asserted.** The winning cell's t was **+10.78** - and 7 of 738 circular shifts
of the same data produced a maximum t above it. Firings cluster across symbols
and sessions, so the effective sample is a small fraction of the row count and
a per-cell `t > 1.96` screen would have returned dozens of discoveries.

**So the statistic is the MAXIMUM across the whole grid, against the
distribution of that same maximum.** Never the best cell's own p. The null
circular-shifts the SIGNAL along the session axis, the same shift for every
symbol at once - which preserves the signal's autocorrelation, its time-of-day
distribution and the cross-sectional clustering of firings, and breaks only the
alignment being tested. Shuffling per symbol destroys that clustering and gives
a null far too narrow, which is how a diagnostic manufactures a discovery. On
12 pure-noise panels it calibrates to mean p **0.476** with full spread.

**Exhaustive, not sampled, because an FFT makes it free.** The statistic at
every shift is a circular cross-correlation, so all 738 come out of one
`irfft(conj(rfft(mask)) * rfft(y))` - seconds instead of ~10^12 operations.
`test_signal_grid.py` checks it against a brute-force loop, because a conjugate
on the wrong operand yields a complete, symmetric, entirely wrong null.

**Multiplicity is counted across BOOKS, not within one.** ~30 variants have now
been measured against zero across four books; the expected minimum p over that
many near-independent tests is ~1/31 = 0.032, and the best p anywhere in the
repo is 0.077. Any new search carries its own family-wise correction or it is
not evidence.

## The factor sleeve, after its four defects were fixed

**F1 is the gate the sleeve had to clear**
([factor/verdict.py](nifty_algo/factor/verdict.py)): both momentum arms and
**500** null draws, run CONTINUOUSLY over the whole window rather than fold by
fold, because the two defects it measures are invisible inside a 6-month fold.
Result at 0.25%/leg: **+18.8% CAGR against a null median of +3.1%, beating
500/500 seeds, empirical p = 0.002** - which is the family-wise threshold, not
the per-book one. Equal-weight rebalancing RETAINS 106% of the excess, so the
result is not concentration.

**The four defects, and which way each ran:**

1. **The book never reweighted.** Winners were never trimmed, so the sleeve was
   momentum plus an unregistered let-winners-run overlay. `FactorConfig.reweight`
   runs the other arm; it turns out to *help* (+19.7% vs +18.8%), so this one
   ran AGAINST the book rather than for it.
2. **No slippage was charged at all** - `run()` called `buy_cost`/`sell_cost`,
   which exclude it, rather than `friction`. This ran hard in the null's favour:
   the null turns **1.78x** the book per rebalance against momentum's **0.56x**,
   so free fills subsidised it ~3x as much. Charging both identically at
   0.25%/leg costs momentum 1.9pp and the null 5.5pp, and **widens** the excess
   from +12.7pp to +16.3pp. The excess grows monotonically with slippage.
3. **The null was ONE SEED** - a single draw compared against, not a
   distribution. At 40 seeds the best achievable p is 1/41 = 0.024, which does
   not clear a family-wise bar; 500 seeds is what gets to 0.002.
4. **`listed_only` bounds the WRONG HALF of survivorship** - look-ahead about
   which companies came to EXIST, not which survived, and survival is the half
   that matters for a long-only book holding the tail. There is no fix inside
   this data. The usable bound is the liquidity band: **liquid +15.5% against
   all at +18.8%, a 3.3pp / 17% haircut.** All four bands land within 3pp, so
   the edge is not confined to names too illiquid to trade.

**THE WALK-FORWARD DOES NOT AGREE WITH THE POOLED TEST, AND THAT IS THE
FINDING.** `verdict.walk_forward` re-ran the 16 out-of-sample windows in the new
cost regime, ranking momentum inside a 40-draw null distribution in each:

| regime | folds won | sign p | mean pctile | combined p |
| --- | --- | --- | --- | --- |
| 0.25%/leg, drift | 11/16 | **0.210** | 69% | **0.0039** |
| 0.00%/leg, drift | 10/16 | 0.454 | 62% | 0.0477 |
| 0.25%/leg, reweight | 11/16 | **0.210** | 69% | 0.0050 |

Fixing the costs moves it in the right direction (10/16 -> 11/16, combined
0.048 -> 0.004) and still does not clear the sign test. **13 of 16 is what
p<0.05 needs.**

The two statistics disagree because the percentile distribution is BIMODAL, not
because either is wrong: 8 folds sit at or above the 92nd percentile of their
null and 5 sit at or below the 32nd. Momentum wins most windows overwhelmingly
and loses the rest badly. A sign test cannot see that; a mean percentile cannot
see that adjacent windows share a regime. **Report both.**

**The losing windows are not random - they are momentum crashes.** Fold 3
(2020-03 -> 2020-09, the COVID collapse) is the 98th percentile; fold 4
(2020-09 -> 2021-03, the junk rally off the bottom) is the **2nd**. That is the
documented failure mode of the factor, arriving exactly where the literature
says it does. It is evidence the effect is real momentum rather than noise -
and equally, evidence that the -57% drawdown is structural rather than bad luck.

**READ THE DRAWDOWN BEFORE THE CAGR.** -57.2% peak-to-trough with a **35-month**
recovery, and that is measured on MONTH-END marks, so the true intra-month
figure is worse. On a Rs 5,00,000 pot that is Rs 2.85 lakh down and nearly three
years to get back. The CAGR is the reason to look; the drawdown is the reason
most people would not hold it.

**And the headline that started this is dead.** The +30.6% widely quoted earlier
was an 8-year window with free fills. The same book over the full window with
honest fills is **+18.8%**, and the survivorship-lean bound is **+15.5%** - which
is ordinary published-Indian-momentum territory rather than double it. Window,
slippage and survivorship, in that order.

## F2: the drawdown instruments are CRASH INSURANCE, and the sleeve's edge is window-dependent

[factor/drawdown.py](nifty_algo/factor/drawdown.py) scores 8 pre-declared arms -
a per-name stop at two widths x two bases, a benchmark-MA regime gate at two
lengths, and a static cash blend - on ONE axis: drawdown removed per point of
CAGR surrendered. It was then re-run on **India 2005-2016**, a window no
parameter in this package had seen, fetched with `--out` so the file the
recorded results came from was never touched.

**THE TWO WINDOWS DISAGREE, AND THE DISAGREEMENT IS THE RESULT.**

| | sleeve CAGR | NIFTY | excess | maxDD | recovery |
| --- | --- | --- | --- | --- | --- |
| 2016-2026, where F1 was measured | +18.79% | +10.85% | **+7.94pp** | -57.2% | 35m |
| 2005-2016, never used | +12.39% | +11.79% | **+0.60pp** | **-78.5%** | **81m** |
| 2005-2026, all of it | +17.86% | +11.18% | +6.69pp | -78.5% | 81m |

**The momentum edge is 0.60pp over the index on the unseen decade.** F1's
p = 0.002 was measured entirely inside 2016-2026, and 2005-2016 is not a
confirmation of it. Pre-2016 survivorship is WORSE - 2008's casualties are
absent from today's listed set - so the sleeve is flattered there and the true
excess is likely negative. Quote both windows or neither.

**F2b's KILL TEST FIRED.** The pre-committed rule was: if the drawdown on unseen
data exceeds 65%, rebuild the allocation on the worse figure. It is **-78.5%
with an 81-month recovery** - nearly seven years underwater, against the 35
months the calibration window showed. So the planning figure is 90%, not 66%,
and every allocation drops by about a third: tolerate a 15% portfolio drawdown
and the sleeve gets **17% of net worth, not 23%**; tolerate 20% and it gets
22%, not 30%. This too is a LOWER bound.

**THE INSTRUMENTS ARE INSURANCE, WHICH IS WHY ONE WINDOW COULD NOT SCORE THEM.**
On 2016-2026 `regime50` cost **-17.0%** compounded and lost to the cash line;
on 2005-2016 it made **+39.6%** and cut the drawdown from -78.5% to -35.1%. The
per-year attribution says exactly why:

| | 2008 | every other year |
| --- | --- | --- |
| `regime50` | **+203.4%** | negative in 8 of 10 |
| `stop15r` | +89.8% | negative in 7 of 10 |

`stop15r` on 2016-2026 is the same shape with a smaller crash: **2023 is +14.5%
and the whole decade is +14.5%** - every other year nets to nothing.

**So "the three best months carry it" is a DESCRIPTION of insurance, not a
verdict on it.** The consistency check in `consistency()` is the right lens for
an edge and the wrong one for a hedge, whose payoff is concentrated by
construction. Reported together, and neither alone decides. What can be said
without a forecast: the unconditional cost of `regime50` is about 17% per calm
decade, and its payoff needs a 2008.

**By the letter of the pre-registered criterion, `regime50` SURVIVES** - it
passed on 2016-2026 and passes again on 2005-2016 and on the full 21 years. That
is not a licence to arm it. It is a statement that the gate could not separate
insurance from edge, which is a fact about the gate.

**A stop in a monthly-rebalanced book is a DELAY, not an exit.** The sleeve
re-ranks monthly and re-buys anything still in the top 20 - 430 distinct names
across 121 marks - so a stop costs a round trip and a gap, never the position.
Which is also why "stops truncate the right tail momentum depends on" is the
wrong mechanism: `reweight` cuts every winner back to 1/20 every month and
*beats* drift (+19.7% vs +18.8%). The tail comes from re-selection.

**THE GATE'S THRESHOLD DECIDED THE ANSWER ONCE, so it needs a threshold-free
companion.** On 2016-2026 `regime50` was the sole PASS while `cash70` FAILED by
0.02pp of CAGR with the best ratio in the table. `cash_frontier` asks the same
question without a round number - at the drawdown each arm actually achieved,
what does simply holding less pay? - and on that window it inverts the verdict.
**Cash is the null hypothesis for a drawdown instrument the way random scoring
is the null for a signal.**

**REJECTING THE SPLIT DAY IS NOT ENOUGH.** `MAX_SESSION_MOVE` stops a 1:10 split
being traded as a -90% session, but a stop compares against a BASIS, and after
the split the position still holds a pre-split basis against a post-split price -
so it fires on the next session instead. One day late, same wrong trade, no
longer near an obvious -90% move. `_stop_price` returns the factor and the caller
rebases with it; worth 0.8pp of CAGR on the `stop15e` arm. Regression:
`test_a_split_sized_move_does_not_trigger_the_stop`.

**Twenty positions is ~3.2 independent bets.** Mean pairwise monthly correlation
across 400 NSE names is **0.28**, so effective N = 1/(rho + (1-rho)/20) = 3.2 and
thirty names reaches only 3.4. Position count is not a survival lever - and a
per-name stop in a crash is one market call wearing twenty costumes, paying
twenty round trips for it.

**SELLS ARE COUNTED, NOT INFERRED FROM TURNOVER.** `avg_turnover` is the fraction
of book value that changed hands and every rebalance trades both legs, so reading
sell legs off it doubles them - which doubled the flat-DP floor to Rs 2,056/yr
when the counted figure is **65 sells and Rs 992**. At Rs 5,00,000 that is
0.20%/yr, not 0.41%. `FactorResult.sells` exists so it is read, not derived.

**RELATIVE PERFORMANCE COMPOUNDS AS A WEALTH RATIO, NOT AS A DIFFERENCE.**
`prod(1 + r_arm - r_base)` is not `prod(1+r_arm)/prod(1+r_base)`, and with
monthly returns of +-20% the gap is large: an ad-hoc attribution built the wrong
way reported a whole-window +2.6% where the ratio gives +14.5%. `consistency()`
uses the ratio; anything computed beside it must too.

**Sizing is arithmetic, not a fitted parameter.** Ruin here is not mathematical -
long-only, unlevered, no margin - so the failure modes are abandonment at the
trough and needing the money before the recovery, and both are governed by the
sleeve's share of net worth. `sizing_report` maps a measured drawdown onto that
share and nothing in it is fitted, which is exactly why the whole recommendation
moved when the measured drawdown did.

**And the margin over the index is 3-5 names.** Remove the top contributor and
the 2016-2026 CAGR is 15.96%; the top three, 13.30%; the top five, 11.20% -
against the index at 10.88% on identical marks. Reason to size it small and never
lever it.

## The sleeve as a console, and the restraint built into it

[factor/sleeve.py](nifty_algo/factor/sleeve.py) is the live book and
[ui/page_sleeve.py](nifty_algo/ui/page_sleeve.py) renders it. THE SCAN IS THE
BACKTEST'S OWN FUNCTIONS CALLED ON TODAY - `eligible_at`, `score_universe` and
`top_n`, imported and used unchanged - which is what makes the recorded numbers
a description of what the page recommends.
`test_the_live_scan_is_the_backtests_own_book` asserts it against
`wanted_log`, and it is the only test in that file that matters.

**THE PAGE IS BUILT AROUND A REFUSAL.** The sleeve rebalances monthly, turnover
is its largest measured cost, and its edge over the null exists partly BECAUSE
it turns 0.56x per rebalance against the null's 1.75x. So the diff is computed
daily and stamped **provisional** on every day that is not the rebalance date;
the checklist download exists only when it is armed. Acting on a provisional
diff is not a small deviation from the tested book - it is the mechanism by
which that book's edge is spent on brokerage.

**NEWS NEVER TOUCHES THE RANKING.** `swing/scanner._score` folds news into a
rank because that book was measured with it there. This one was not, so a
headline that reordered the top 20 would make the live sleeve a different and
untested strategy while every figure on the page still described the tested
one. `test_news_cannot_reach_the_ranking` pins it.

**THE BOOK HELD ALL 20 NAMES ON 9 OF 121 REBALANCES.** `budget = marked / top_n`
spends the pot with nothing left for charges, so the last buys are refused for
cash and the sleeve runs at 15-19 names most months. That is a property of the
measured result, not a bug - but a console that divided the pot by 20 would hand
you a checklist whose final orders the broker rejects. `_size_sequentially`
therefore fills in rank order against a running balance exactly as `run()` does,
and `SleevePick.unfunded` says which names the pot did not reach.

**`wanted_log` IS NOT `holdings_log`.** The first is what the ranking asked for,
the second what the account could afford, and logging only fills would hide a
systematic under-deployment behind a `top_n` that reads like a promise. Same
discipline as journalling rejections as carefully as approvals.

**THE FACTOR MARKET IS BORROWED, NOT REGISTERED.** `markets.factor_market(cfg)`
returns a `replace()` of India rather than a fourth entry in `default_markets()`,
because `markets.keys(cfg)` is the `--market` choice list for every swing CLI -
the same reasoning `IntradayEquityConfig` already records. **Taxonomy is what
forces it to exist at all:** the swing book screens the Nifty 100, whose
industries are hand-maintained in NSE's vocabulary, while the factor universe is
~2,400 names from Kite's instrument dump with no industry column, so only
Yahoo's GICS labels can classify it.

**AND THOSE LABELS HAVE TO BE COPIED ONTO THE STOCK.** `activity_failure` and
`_is_classified` match on `stock.industry`/`stock.sector`, NOT on the
fundamentals object. Without `sleeve.stock_for(symbol, market, fundamentals)`
every name arrives unclassified - and unclassified is a REJECT - so the sleeve
would find nothing eligible and look like a strict screen rather than a broken
one.

**The screen runs AFTER the ranking and is bounded.** Cheap-to-expensive, as the
swing scanner already orders its gates: rank first, then take the highest-ranked
names that pass, looking no further than `halal_shortlist`. That bound is
load-bearing - without it the book reaches as far down as it must to find twenty
passing names, which on a strict month means holding the 300th-best momentum name
and still calling it momentum. It holds FEWER names instead and records that it
did. Measured on the live top 30: **11 rejected, 37%**, so the default shortlist
of 60 fills comfortably. It also makes the screen affordable - fundamentals are
one slow request per name, and only 954 distinct names ever reached the top 60
across 121 rebalances, against 2,437 in the universe.

**F3: THE SCREEN IS ROUGHLY FREE ON RETURN AND CONSISTENTLY CUTS THE DRAWDOWN.**
`scripts/run_f3_screened.py`, both windows, `halal_screened` applied at
selection:

| window | unscreened | screened |
| --- | --- | --- |
| 2016-2026 | +18.79% / -57.2% | **+18.01% / -47.9%** |
| 2005-2016 | +12.39% / -78.5% | **+13.76% / -74.8%** |

It costs 0.78pp and saves 9.3pp of drawdown on one window, ADDS 1.37pp and saves
3.7pp on the other. So the book you would actually trade is not worse than the
one that was measured - which is the question F3 existed to answer, and the
answer could easily have gone the other way.

**Two things must be read with it.** The balance sheets are TODAY'S applied to
every rebalance - the same point-in-time distortion the swing backtest prints -
and it cannot be fixed with free data. And **26% of the 2005-2016 shortlist is
`unverifiable`** against 6% of the 2016-2026 one: the further back you look the
less Yahoo classifies, so a quarter of that window's names are excluded for a
DATA reason wearing a Shariah reason's clothes. The +1.37pp is not clean evidence
the screen helps.

**THE PLANNING DRAWDOWN STAYS AT 78.5%.** The screened figure on the good window
is -47.9%, and sizing off it would repeat exactly the mistake the kill test
caught: -57.2% also looked like the answer until an unseen decade said -78.5%.
The worst screened number is -74.8%, the worst measured anywhere is -78.5%, and
the more conservative of two numbers that close is not worth trading away.

**`halal_screened` still defaults False** so every recorded F1/F2 number
reproduces byte-identically, and `run_f3_screened.py` reads the fundamentals
cache FILE rather than calling `load_fundamentals` - that function fetches
whatever is missing, so handing it the universe fires ~1,500 requests as a side
effect of a measurement.

**Index membership is a file, not a calculation.** [factor/membership.py](nifty_algo/factor/membership.py)
resolves Nifty 50 / Next 50 / 500 / outside from committed constituent lists.
Turnover rank correlates with membership and is not it, and a band guessed from
bars would be right most of the time - the worst kind of wrong. A missing file
reports "50 / Next 50 split unavailable" rather than picking a half, the same
discipline `HalalVerdict` applies to an absent fundamental.

**What the page refuses to do:** offer a stop-loss control (F2 measured one and
it is a delay, not an exit, in a book that re-buys monthly), place an order (the
sleeve is a THIRD book and going live on it is its own decision), or show
+18.79% without +0.60pp beside it. `_record` renders both windows above the
picks, and `test_both_windows_are_on_screen_before_anything_else` keeps them
there.

**A failed holdings read is not an empty account.** `_holdings_map` returns three
states, and an incomplete snapshot marks every action "[holdings unverified]" -
because an empty list read as "you hold nothing" turns a fully invested book into
twenty BUYs, the single most expensive mistake this page could make.

## F4: the Nifty 500 restriction is FREE, and its apparent benefit is hindsight

The live sleeve's top 20 was 17 names outside the Nifty 500, one trading Rs 2.5
cr a day. [factor/restriction.py](nifty_algo/factor/restriction.py) makes the
universe a lever, and `scripts/run_f4_universe.py` measures it three ways
because measuring it ONE way produces a spectacular lie.

| window | `all` | `nifty500` | `size500` (control) | look-ahead |
| --- | --- | --- | --- | --- |
| 2016-2026 | +18.01% / -47.9% | **+31.29% / -32.9%** | +17.60% / -48.6% | **+13.69pp** |
| 2005-2016 | +13.76% / -74.8% | **+19.14% / -66.3%** | +13.55% / -74.3% | **+5.59pp** |

**RESTRICTING TO TODAY'S NIFTY 500 APPEARS TO ADD 13pp OF CAGR AND HALVE THE
DRAWDOWN. Almost all of it is hindsight.** Today's membership applied to 2016
can only hold companies that GREW INTO the index - selection on the outcome,
which is worse than the plain survivorship bias `factor/universe.py` documents
because survivorship removes losers while this also pre-selects winners.

**The control is a point-in-time size rank**, because the Nifty 500 selects on
free-float market cap: today's implied share count (`market_cap` over today's
close) applied to each date's own close, top 500 per rebalance. Its error is
share ISSUANCE - mechanical, and it does not know whether a company later joined
an index. On that control the restriction costs **-0.41pp and -0.22pp**: free,
and it buys nothing. **The drawdown relief is hindsight too** - `size500` draws
down -48.6% and -74.3%, indistinguishable from unrestricted.

**So the answer to "what if I have to be in the Nifty 500" is: do it, it is
free.** Trade it for the mandate and for fillability - the restricted book runs
Rs 22-814 cr a day - but expect `size500`'s return and `size500`'s drawdown, not
the +31% the naive backtest reports. The planning drawdown stays 78.5%.

**TURNOVER RANK IS NOT A PROXY FOR INDEX MEMBERSHIP.** CUPID passes
`band="liquid"` - its turnover blew out WITH the momentum, Rs 774 cr a day on a
micro cap - so any liquidity stand-in readmits exactly the names a size mandate
excludes. Size and traded value are different facts.

**THE RESTRICTION IS A CALLABLE, NOT A SET.** `size500` is a different set every
rebalance; passing `run()` one set computed over the whole history would be
look-ahead arriving through the back door. `restriction.compose` intersects it
with `listed_only` so a universe key cannot silently override the survivorship
bound.

**THE CONTROL IS NOT CLEAN EITHER, just cleaner - and it was wrong twice before
it was right.** First it was built only from names that had ever reached the top
60, i.e. pre-selected on the outcome it exists to be independent of; that was
worth 2.8pp on the 2016-2026 window (+2.38pp before the full-universe fetch,
-0.41pp after). Second, it prices a 2008 rebalance with a share count measured
in 2026, so heavy diluters have their past size overstated and growth names
dilute most. Every look-ahead figure it produces is therefore a LOWER bound.

**AND THE NIFTY 100 IS WORSE THAN USELESS.** Given its own control (`size100`,
top 100 by point-in-time size) the picture is:

| pair | naive | control | look-ahead | honest cost vs `all` |
| --- | --- | --- | --- | --- |
| nifty500 vs size500 | +31.29% | +17.60% | +13.69pp | **-0.41pp** |
| nifty100 vs size100 | +19.81% | **+7.94%** | +11.87pp | **-10.07pp** |

A 100-name universe LOSES to the index on both windows (+7.94% and +10.62%
against +10.85% and +11.79%). A top-20 book is a fifth of it, so there is almost
no cross-section left to rank - and cross-section is the entire mechanism. Every
published-membership arm now gets a size-ranked twin of the same width, because
an uncontrolled `nifty100` is the same trap as an uncontrolled `nifty500` on a
shorter list, and a shorter list selects HARDER.

**THE RECORD ON SCREEN IS PER UNIVERSE.** `sleeve.RECORDS` / `record_for` hold
what a backtest actually says about each one, and `report()` and
[page_sleeve.py](nifty_algo/ui/page_sleeve.py) both read it - so the CLI and the
console cannot quote different numbers for the same book. For a restricted
universe the table shows the CONTROL's figures, because those are what to
expect; the naive +31.29% appears once, in a caption, labelled inflated. It is
not a caveat on the expectation, it is the wrong number to plan with, and a
table row would make it the one you remember.

**`record_for` RETURNS None FOR AN UNMEASURED UNIVERSE, AND None IS A REFUSAL.**
Both consumers render "no backtest describes this universe" rather than borrow
another's numbers, and `test_every_registered_universe_has_been_measured` fails
the moment a key is added to `restriction.UNIVERSES` without measuring it.

**THE RECORD IS COMPUTED AFTER THE CONTROLS AND PLACED ABOVE THEM.** `_controls`
is what writes the selected universe onto the config, so rendering the record
first showed the PREVIOUS universe's numbers for one interaction - on the single
panel whose whole purpose is never to quote the wrong book. A `st.container()`
reserves the position and is filled afterwards.

**`universe` defaults to `"all"`** so every recorded F1/F2/F3 number reproduces
byte-identically, and `test_no_restriction_is_byte_identical_to_the_old_book`
pins it. Live, none of the look-ahead applies: today's membership is a fact
today, which is why the page offers the selector at all.

**The pot, universe and screen persist; the rest of the strategy does not.**
`settings_store.FIELDS` gained three entries, and the distinction is close
enough to be worth stating: `factor_universe` is not a parameter being swept, it
is which slice of the market this holder may own, and the Shariah screen is a
property of the holder rather than of the strategy. Both belong with the pots;
the formation, band, `top_n` and hold period stay in version-controlled config.
`apply_to` gained a SECOND refusal beside the `dry_run` one - every string
coerces to a string, so a hand-edited `"factor_universe": "nifty42"` would pass
the type check and raise deep inside a scan; it degrades to the unrestricted
book with a note instead.

**A page test must never write `data/settings.json`.** Persisting from `_run`
means the page writes real user state, and the first run of these tests wrote
`factor_universe: nifty500` and a Rs 5,00,000 pot into the live file - after
which three unrelated tests failed because `get_config()` applied them.
`settings_store.save` binds `DEFAULT_PATH` as a default argument at import, so
patching the module constant does not redirect it; the seam that works is the
page's own `save_settings` reference.

## F5: an ATR trail is the same crash insurance, and it is not free

A per-name trailing stop is the most-requested thing this sleeve does not have,
so it was measured rather than argued about: Wilder ATR(14), armed at every
rebalance, ratcheted daily, never loosened, on the Nifty 500 + halal book.

| trail | 2016-2026 | 2005-2016 |
| --- | --- | --- |
| none | +31.29% / -32.9% | +19.14% / -66.3% |
| 3x ATR | +19.56% / -31.4% (**-11.73pp**) | +14.96% / -57.4% (-4.18pp) |
| 4x ATR | +24.93% / -28.0% (-6.36pp) | +16.22% / -62.5% (-2.92pp) |
| 6x ATR | +28.09% / -32.0% (-3.20pp) | +17.71% / -65.9% (-1.43pp) |

**IT FAILS F2's GATE ON BOTH WINDOWS.** The best drawdown relief anywhere is
8.9pp (3x ATR on the crash window, which also cuts recovery from 57 to 36
months) - short of the 10pp the gate asks - and the SAME setting costs 11.73pp
for 1.5pp of drawdown on the calm decade. That is the crash-insurance shape F2
already found for the regime gate, arriving by a different route.

**THE RELATIONSHIP IS MONOTONIC, WHICH IS WHY NO MULTIPLE RESCUES IT.** Fixed
percentage stops on the same book trace the identical curve - 5% costs 10.7pp
and fires 992 times over 121 rebalances, 25% costs 1.3pp and fires 47 - and the
drawdown barely moves along any of it. **A 5% stop measured from entry makes the
drawdown WORSE**, -39.9% against -32.9%, because it sells low and re-buys
higher. The mechanism is the one F2 named: a book that re-ranks monthly buys
back most of what a stop sold, so a stop is a delay that pays two round trips.

**Turnover is the tell.** 39% -> 110% at a 5% stop, 39% -> 126% at 2x ATR. The
sleeve's edge over a random-scoring null exists partly BECAUSE it turns 0.56x
per rebalance against the null's 1.75x; a tight stop turns it INTO the null.

**SO THE CONSOLE FLAGS AND NEVER SELLS.** `sleeve.review()` returns `Flag`
objects - drawdown against what you paid, a break of the level a trail would
have fired at, a halal verdict that changed, a news hit - ordered most-serious
first, rendered in the "Between rebalances" panel, and carrying no quantity by
construction. A flag costs nothing and can see the thing a stop cannot: a
company that has stopped being the company you bought. The `atr` row prints the
level the measured-unprofitable rule would have sold at, labelled as
information rather than as a recommendation.

**AND THE OPERATIONAL PATH WOULD HAVE BEEN WORSE THAN THE ARITHMETIC.** Without
DDPI every delivery sell needs a CDSL TPIN authorisation valid for ONE trading
day, and Kite still displays a stale GTT as active - so a trailing stop
maintained by daily re-authorisation is precisely the "looks armed and is not"
failure `kite_equity.protection_state()` exists to prevent.

**`trail_atr_multiple` and `atr_window` default to None/0**, and the ATR array
is not even allocated unless asked for - a third full-length array over 2,400
symbols is a cost nobody asked for, and every F1-F4 number reproduces
byte-identically without it.

## Holdings, and the research briefings built on them

`nifty_algo/portfolio/` answers "what do I own", broker-agnostically.
`nifty_algo/research/` turns that plus market data into **fact packs** - JSON
evidence a briefing is allowed to cite - and `.claude/skills/macro-brief/` and
`.claude/skills/portfolio-risk/` write the prose from them. There is no LLM
call in the Python; the arithmetic is deterministic and testable, and the
judgment (Fed outlook, moat ratings, sector rotation) is handed over
explicitly through `Section.judgment`.

**`ConnectorResult.available` is the field that matters most in the whole
package.** `BrokerTransport._read` returns an empty default when a broker read
FAILS - right for a reconciler, catastrophic for a risk report, because an
empty list read as "you hold nothing" produces no concentration, no
correlation and no single-stock risk on a fully invested account. `available`
defaults False, `unavailable()` demands a reason, and the Kite connector sets
it from the delta in `read_failures` either side of the call. Regression:
`test_a_failed_holdings_read_is_not_an_empty_portfolio`.

**An incomplete snapshot withholds every percentage.**
`PortfolioSnapshot.weight()` returns **None**, not a float, when any enabled
connector failed or any currency would not convert. A share computed against a
denominator that could not be established reads exactly like one that was, and
it would be acted on. Absolute values still render - they are facts about what
was seen.

**A connector not in `PortfolioConfig.connectors` is never asked and never
counted.** That list is a claim about your accounts, and it is what keeps the
registered-but-unimplemented IBKR stub from marking every snapshot incomplete
forever. Enabling `ibkr` before it is implemented is therefore not a no-op: it
correctly makes every report refuse to quote a weight.

**The research package never writes to the price cache.** `load_prices`
re-downloads `history_days` (400) and **rewrites the parquet** on any cache
miss, and `scripts/fetch_swing_history.py` deliberately puts six years in that
same file. So `holdings_prices.bars_for` asks only for symbols already cached
and names the rest, rather than silently truncating the history every backtest
depends on.

**A fact this build could not establish is `available=False` with a reason,
never a blank and never a zero.** About a third of what these briefings ask
for does not exist for Indian equities on any free source: **India publishes
no short-interest disclosure at all**, insider dealing is a SEBI PIT filing
rather than an API, institutional trend is the quarterly shareholding pattern,
and the CPI/GDP/repo prints have no free machine endpoint (they come from
`data/macro_manual.csv` if you keep one). Rendered as 0.0, every one of those
reads as good news. `FactPack.unavailable()` is a top-level list in the JSON
so the skills can check that rule without walking the tree.

**Yields move in basis points, indices in percent.** `macro_series` carries
`RATE` vs `LEVEL` per series and `factor_moves` differences the first and
percent-changes the second. Percent-changing a yield makes a 10bp move at 0.5%
look twenty times the same move at 5%, which would rank every rate sensitivity
in the book by nothing but where yields happened to be.

**The stress test is REPLAYED, not modelled.** `risk_report.episodes()` finds
the benchmark's real peak-to-trough falls in the cached history and applies
today's weights to what those names actually did, reporting `coverage_pct`
alongside. No distribution is assumed: a parametric VaR on equity returns
understates the tail by construction, because the days that matter are the
ones a normal distribution says cannot happen. Two limits are printed above
every result - the weights are today's, not the ones held then, and the cached
universe is today's index, so the same survivorship distortion applies as in
the swing backtest.

**`exposure.portfolio_returns` renormalises per session rather than filling
with zeroes.** A holding with no bar on an early session is excluded from that
day's mean, not scored 0% at full weight - the latter damps the volatility,
the percentiles and the worst session by exactly the missing weight, in the
reassuring direction, with no error anywhere.

## Delivery charges are not option charges

`costs.py` is options. `swing/costs_equity.py` is delivery, and two
differences are large enough to change whether a trade was worth taking: STT
is charged on **both** legs, and there is a **flat Rs 15.34 DP charge per
scrip per sell** that does not shrink with the position. On a Rs 10,000 ticket
the round trip is ~Rs 48, about 0.1R of a Rs 500 risk budget. Rates carry
`VERIFIED_ON`, like `crossborder.py`.

## Testing conventions

[conftest.py](tests/conftest.py) provides `flat_bars()` / `append_bar()` / `make_context()` so a test can engineer an exact setup rather than hunt for one in real data.

- [test_lookahead.py](tests/test_lookahead.py) is the suite that matters most. `find_pivots()` uses a **centred** window, so live you only learn a pivot `lookback` bars later; `BarReplayer` enforces that delay and these tests assert it. Every other failure gives a wrong answer you can see - look-ahead bias gives a flattering one you cannot.
- Ties resolve pessimistically: intrabar, the stop wins (including for the trailing stop). A swing outcome where one daily bar covers both stop and target is recorded `ambiguous` and counted as a **loss**.
- Both books consume `signals.py`, `CapitalConfig` and `ExitLadder`, which is why the whole suite runs before anything is called done - see **Definition of done** above.

## The console is gated, and the gate is above the imports

`nifty_algo/ui/auth.py` runs in `app.py` **between** `st.set_page_config()` and
the `page_*` imports, and calls `st.stop()`. That ordering is the whole
control: an unauthenticated session never imports a page module, so it never
builds the engine, the feed, the journal, the broker or the `st.fragment(
run_every=...)` poller. Inside `main()` the gate would reject the same visitor
*after* constructing all of it. `nifty_algo/ui/__init__.py` is a bare docstring,
which is what keeps importing `auth` free.

**It fails closed and says so.** No `APP_PASSWORD` means the app refuses to
open, never that no password is needed - an unconfigured gate is invisible to
the person who deployed it, because they know the password. `APP_AUTH_DISABLED=1`
is the one escape hatch, and deciding "am I deployed?" by sniffing the hostname
was rejected: that makes the safe path depend on a guess.

**`bridge_secrets()` is why no call site changed.** There is no `.env` on
Cloud and fourteen readers use `os.getenv`, so flat `st.secrets` keys are
copied into `os.environ` once at startup, never overwriting - env stays
authoritative, and the two sources cannot disagree about which won.

**The throttle is per-client, bounded, and NOT the security control.**
`X-Forwarded-For` is client-supplied, so it is evadable by anyone who can forge
a header; it exists to make brute force expensive, not to stop it. Three things
it must never do: lock out globally (a self-DoS any stranger can trigger),
allocate a bucket on a mere page load (the bot traffic becomes the memory
leak), or let a truthy non-string out of `client_key()` - a non-`str` key can
never equal `UNKNOWN_CLIENT`, so it would silently promote an unidentified
client out of the capped shared bucket into the full lockout.

**Search-indexing is not solvable from Python here.** A `<meta robots>` through
`st.markdown` survives the server and is stripped by the frontend sanitiser,
and Community Cloud serves its own `index.html`. Use the Cloud private-app
viewer allowlist, which rejects traffic before the script runs. A tag that
reads as armed and does nothing is the failure this repo refuses elsewhere.

Tests: [test_auth.py](tests/test_auth.py). Every UI `AppTest` must call
`conftest.sign_in(at)` or it renders the login form and nothing else.

## The suite must never reach a broker, and one fixture is why

`tests/conftest._no_live_credentials` blanks every bridged credential -
`KITE_API_KEY`, `KITE_API_SECRET` and the Telegram/SMTP/Fyers/Dhan keys - for
every test in the suite.

**BLANK, NOT DELETE.** `python-dotenv` skips any key already present in
`os.environ`, so an empty string is what stops `load_dotenv()` re-supplying the
real value; deleting them leaves the door open on the very next AppTest.
`test_auth._clean_env` already used this trick for the auth keys and documents
it - the fixture applies it to the broker keys and to the whole run.

**It is sufficient because `KiteSession.configured` is
`bool(api_key and api_secret)`**, both read from the environment. A cached
`.kite_session.json` alone does NOT make it configured, so blanking those two
closes the path rather than narrowing it. `tests/test_no_live_credentials.py`
asserts the guard directly, because a guard that only works silently is one that
rots.

**What it fixes, and what that cost.** `tests/test_auth.py` drives the console
through Streamlit's `AppTest`, which runs `app.py` - so `load_dotenv()` and
`bridge_secrets()` executed and copied the real `.env` keys into `os.environ`
**for the rest of the process**. Nothing reset them, and
`tests/test_portfolio_connectors.py::test_an_enabled_but_unconfigured_connector_is_unavailable`
then asserted Kite was unconfigured, reached a genuinely configured Kite, and
**read the live account** - real positions, a fresh FX rate. It placed no orders,
but a repo whose entire design is about keeping code away from money by accident
should not have its test run reach the account at all.

    pytest tests/test_auth.py tests/test_portfolio_connectors.py   # was 1 failed

That two-file reproduction was the whole diagnosis: ORDER dependence, not the
login on its own. It passes now.

**A test that needs a credential ABSENT has to say so.** The fixture is
function-scoped, so a test's own `monkeypatch.setenv` still wins - but a blank
value is PRESENT as far as `bridge_secrets` is concerned, so
`test_the_bridge_never_overwrites_a_real_environment_variable` now deletes the
key it needs missing instead of assuming the machine has none. Depending on the
ambient environment is the habit that caused all of this.

## Older note: how that failure used to present

`pytest` was green from a cold machine and failed one test on a trading morning,
and the failing test was the messenger rather than the bug.

`tests/test_auth.py` drives the console through Streamlit's `AppTest`, which
runs `app.py` - so `load_dotenv()` and `bridge_secrets()` execute and copy the
real `.env` keys into `os.environ` **for the rest of the process**. Both are
doing exactly what they were written to do; nothing resets them afterwards.
`tests/test_portfolio_connectors.py::test_an_enabled_but_unconfigured_connector_is_unavailable`
then asserts that Kite is unconfigured, reaches a genuinely configured Kite,
and reads the **live account** - real positions, a fresh FX rate.

    pytest tests/test_portfolio_connectors.py     # 22 passed
    pytest tests/test_auth.py tests/test_portfolio_connectors.py   # 1 failed

That two-file reproduction is the whole diagnosis: it is ORDER dependence, not
the login on its own, and not any of the factor work.

Two separate things follow, and the smaller one is the test. **The suite makes
authenticated network calls to Zerodha with real credentials and pulls real
holdings into a test process.** It places no orders - `dry_run` defaults hold -
but a repo whose entire design is about keeping code away from money by
accident should not have its test run reach the account at all. The fix is
isolation at the boundary that leaks: restore `os.environ` around the AppTest
tests, or have the portfolio connectors read configuration from an injected
mapping rather than from process-wide state.

Until then a full run on a logged-in machine reports one failure that is not a
regression, which is precisely the kind of noise that trains you to skim a red
suite.

## Data and secrets

`journal/`, `.env`, `.kite_session.json`, and downloaded market data are gitignored. `data/nifty100.csv`, `data/us_halal.csv`, `data/ftse100.csv`, `data/etf_holdings.csv` and `data/halal_overrides.csv` are **source, not data** - the scan cannot run without them, and [.gitignore](.gitignore) carries explicit negations for both. The universe file is committed rather than fetched because NSE 403s unattended clients, and a failed download reads identically to a quiet market.
