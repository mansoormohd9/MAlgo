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
python -m nifty_algo.research macro --market india          # macro fact pack; --json is what a skill reads
python -m nifty_algo.research risk  --market india --json   # portfolio risk fact pack
python scripts/fetch_swing_history.py --market india --years 6  # deep daily bars for it
python scripts/build_universe.py --market us   # rebuild data/us_halal.csv from SPUS+HLAL
python scripts/build_universe.py --market uk   # rebuild data/ftse100.csv
python -m nifty_algo.broker.kite_login  # once per trading morning; token dies overnight
python scripts/fetch_history.py         # real 5m NIFTY history (resumable)
python scripts/fetch_vix.py             # India VIX, for SYNTHETIC_PREMIUM backtests
```

Tests (617: 613 passing, 4 skipped; pytest, `pythonpath = . tests` so `nifty_algo` and the conftest helpers both import without an install step):

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
3. **Backtest if the traded path moved** - `swing/` or `signals.py` -> the swing backtest; `strategy.py` / `risk.py` / `positions.py` / `governor.py` -> the option backtest, which has no CLI (the Backtest page, or `Backtester(cfg).run(bars)` in process). UI, broker and script changes are not backtestable; say so rather than skipping quietly.
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

## Data and secrets

`journal/`, `.env`, `.kite_session.json`, and downloaded market data are gitignored. `data/nifty100.csv`, `data/us_halal.csv`, `data/ftse100.csv`, `data/etf_holdings.csv` and `data/halal_overrides.csv` are **source, not data** - the scan cannot run without them, and [.gitignore](.gitignore) carries explicit negations for both. The universe file is committed rather than fetched because NSE 403s unattended clients, and a failed download reads identically to a quiet market.
