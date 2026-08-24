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
python scripts/build_universe.py --market us   # rebuild data/us_halal.csv from SPUS+HLAL
python scripts/build_universe.py --market uk   # rebuild data/ftse100.csv
python -m nifty_algo.broker.kite_login  # once per trading morning; token dies overnight
python scripts/fetch_history.py         # real 5m NIFTY history (resumable)
python scripts/fetch_vix.py             # India VIX, for SYNTHETIC_PREMIUM backtests
```

Tests (385, pytest; `pythonpath = . tests` so `nifty_algo` and the conftest helpers both import without an install step):

```bash
pytest
pytest tests/test_governor.py                              # one file
pytest tests/test_positions.py::test_trail_never_loosens   # one test
pytest -k ratchet
```

There is no linter or type checker configured. Match the surrounding style: `from __future__ import annotations`, dataclasses, and a module docstring that explains *why* the module exists.

## The two books

This repo holds two unrelated trading systems that share indicators, risk sizing, and the journal - and nothing else.

**Intraday NIFTY option buying** (`nifty_algo/`, everything outside `swing/`) - alerts on 5-minute index bars, three entries a day, flat by 15:10.

**Daily multi-market swing** (`nifty_algo/swing/`) - once a day, sweeps one market's universe, applies a Shariah screen, returns at most three cash-equity LONG tickets held for days. Three markets are registered in [markets.py](nifty_algo/swing/markets.py): India (Nifty 100), US (the union of SPUS and HLAL constituents) and UK (FTSE 100). It reuses `signals.py` (pure `bars in, values out`, so daily bars work unchanged), `CapitalConfig`, and `Journal`. It deliberately does **not** use the engine, the option-strike machinery, or the session governors - those encode "intraday, three entries, flat by close".

Never re-declare risk numbers inside `swing/`. Two books sizing off two copies of the same governor is how they drift apart.

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
- **`kite_orders.modify_stop()` deliberately places no resting SL order at the broker** - the stop trails on the underlying's ATR and is recomputed every bar. Consequence: the stop only exists while the engine is running.
- **Streamlit re-runs the whole script on every interaction**, so the engine, dispatcher and caches live in `session_state` via [state.py](nifty_algo/ui/state.py). A fresh `RiskEngine` per rerun would silently reset `entries_taken` and realised P&L (governors never trip); a fresh dispatcher would re-send every alert.
- **Alert de-duplication is load-bearing**: key is `(strategy, direction, strike, bar timestamp)` plus a per-strategy cooldown. Kill-switch and every `POSITION_KIND` alert are exempt - suppressing a duplicate entry costs an opportunity, suppressing a stop move costs the trade.
- **The swing scanner orders its gates cheap-to-expensive** (`universe -> halal -> prices -> tradeability -> setup -> R:R -> earnings -> news -> rank -> sector cap -> size`) so a stock that was never going to qualify never costs an HTTP request. News is fetched for finalists only, and "nothing notable published" and "feed unreachable" are distinct facts (`available`) - conflating them would read as a clean bill of health on every stock at once.
- **[halal.py](nifty_algo/swing/halal.py) screens against total assets, not market cap**, so a verdict changes when a balance sheet lands rather than when the share price moves. It cannot verify the haram-revenue test; every verdict carries `haram_revenue_verified = False`. Missing fundamentals data is a failure to verify, never a pass.

## Multi-market: the four things that bite

The setup detection, the scoring and the whole of `signals.py` are currency-blind and needed no changes - a daily bar is a daily bar. Everything that did need changing was a number or a label that was secretly India-only, and all four failure modes below produce a *plausible* answer rather than an exception.

- **The benchmark cache is keyed by ticker, and there is one parquet per market.** It used to be one file with the benchmark under a fixed `__BENCHMARK__` key, accepted whenever it held every symbol asked for. Scan the US then India and that key is present, so India's relative strength gets computed against the S&P 500 with no error anywhere. Both halves of the fix are load-bearing. Regression: `test_a_us_cache_cannot_satisfy_an_india_scan`.
- **The LSE quotes in PENCE and yfinance passes it through.** `SHEL.L` arrives as 2500 meaning £25.00. Normalised once, at ingest, in `prices._extract` - `Market.price_divisor` is the default and `fundamentals.currency` ("GBp" vs "GBP") overrides it per symbol, because not every LSE line is in pence. Never divide the benchmark: an index is quoted in points.
- **[fx.py](nifty_algo/swing/fx.py) fails closed, and the scan stands the market DOWN.** `_size()` divides a rupee budget by a foreign stop distance; without conversion a US ticket is ~88x too large and looks entirely ordinary. There is no fallback rate - not a hardcoded 83, not the last rate seen. `ScanResult.stood_down` is a distinct state from "nothing qualified", and the page and CLI say which.
- **Two capital pools, one formula.** LRS money is a different pot from the domestic account, so `CapitalConfig` gained `foreign_capital_inr` and `risk_inr(pool)` - the *formula* still lives in one place, per the rule above. An unfunded foreign pool stands the market down rather than sizing off the Indian balance. `SwingPick` carries both currencies (`risk_amount` in the market's, `risk_inr` in rupees); the intraday book's `rupee_risk` is untouched and really is always rupees.
- **Everything cached or journalled is keyed `{market}:{SYMBOL}`.** Bare tickers collide across exchanges, and a collision is not a crash - it is one company screened against another's balance sheet.

## The halal screen across vocabularies

- **Two activity tables, not one bigger one** ([halal_taxonomy.py](nifty_algo/swing/halal_taxonomy.py)). NSE says "Non Banking Financial Company (NBFC)"; Yahoo says "Credit Services". The GICS table is deliberately *not* a transliteration: `"gaming"` is right for NSE (Delta Corp's casinos) and wrong for Yahoo, where it also matches "Electronic Gaming & Multimedia" - video games, which no mainstream Shariah index excludes.
- **Contested categories are toggles with the standard named**, because the two ETFs in this portfolio disagree: SPUS (AAOIFI/S&P) excludes aerospace & defence and financial exchanges & data, HLAL (FTSE/Yasaar) does not. Hardcoding either would be the code quietly picking a madhhab.
- **Both ratio methodologies are computed; only the primary gates.** FTSE/Yasaar divides by total assets (what HLAL tracks, and what this repo already did); AAOIFI/S&P divides by market cap (what SPUS tracks). `HalalVerdict.verdicts` holds both and `.disagreement` flags the split. `eligible` follows `primary_method` and nothing else - if that slips, changing what is *displayed* changes what is *tradeable*.
- **The US universe is the union of two professionally screened funds**, which makes the local screen a re-verification rather than the only line of defence. Expect disagreements and treat them as findings: the largest class is the debt test, because Yahoo's "Total Debt" includes capitalised operating leases while FTSE/Yasaar screens interest-bearing debt only. Lease-heavy names therefore fail here and pass there. The excluded table flags every name the funds hold. **Do not tune the threshold to make the numbers agree** - that is fitting the screen to the answer.
- **[holdings.py](nifty_algo/swing/holdings.py) is a badge, never a rejection.** If you hold SPUS and HLAL, a "new" NVDA position is not new. Concentration is your decision; what the code refuses to do is let it be made silently.
- **[crossborder.py](nifty_algo/swing/crossborder.py) is arithmetic with citations, not advice.** Every rate carries `VERIFIED_ON`. The one that matters most is not a cost but an exposure: US-domiciled ETFs and direct US shares are US-situs assets against a $60,000 non-resident estate tax exemption, with no India-US estate treaty; an Ireland-domiciled UCITS holding the same companies is outside the regime entirely.

## Testing conventions

[conftest.py](tests/conftest.py) provides `flat_bars()` / `append_bar()` / `make_context()` so a test can engineer an exact setup rather than hunt for one in real data.

- [test_lookahead.py](tests/test_lookahead.py) is the suite that matters most. `find_pivots()` uses a **centred** window, so live you only learn a pivot `lookback` bars later; `BarReplayer` enforces that delay and these tests assert it. Every other failure gives a wrong answer you can see - look-ahead bias gives a flattering one you cannot.
- Ties resolve pessimistically: intrabar, the stop wins (including for the trailing stop). A swing outcome where one daily bar covers both stop and target is recorded `ambiguous` and counted as a **loss**.
- When touching [signals.py](nifty_algo/signals.py), remember both books consume it - run the `test_swing_*` files too.

## Data and secrets

`journal/`, `.env`, `.kite_session.json`, and downloaded market data are gitignored. `data/nifty100.csv`, `data/us_halal.csv`, `data/ftse100.csv`, `data/etf_holdings.csv` and `data/halal_overrides.csv` are **source, not data** - the scan cannot run without them, and [.gitignore](.gitignore) carries explicit negations for both. The universe file is committed rather than fetched because NSE 403s unattended clients, and a failed download reads identically to a quiet market.
