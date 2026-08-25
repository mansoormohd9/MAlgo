---
name: definition-of-done
description: The definition of done in this repo - run before reporting any code change complete. Covers the full pytest suite, an adversarial re-read of the uncommitted diff on the assumption it is wrong, and a backtest when the traded path moved. Triggers on done, finished, that should work, ready to commit, and on any edit under nifty_algo/, scripts/ or tests/.
---

# Definition of done

A change here is not finished when it compiles, and not when the test you were
thinking about passes. It is finished after the four steps below have run, in
order, and been reported.

The reason this is a written rule rather than a habit: every failure this repo
has actually suffered was **plausible**. Nothing raised. The benchmark cache
returned a number, the LSE price looked like a price, the FX-less US ticket
looked like a ticket, `_pool()` returned a balance. A change that "works when I
try it" is exactly the evidence those bugs also produced.

---

## Step 1 - run every test

```
.venv\Scripts\python.exe -m pytest
```

The whole suite. Not `-k`, not `-x`, not the one file you touched.

**Why the whole suite:** the two books share `signals.py`, `CapitalConfig`,
`positions.ExitLadder` and the journal. A targeted run is precisely how a
change to the intraday book breaks the swing book without anyone finding out
until a live scan. `CLAUDE.md` already says "when touching `signals.py`, run
the `test_swing_*` files too" - this generalises it, because the shared surface
is wider than one file.

Rules for reporting it:

- Quote the real counts from pytest's own output. "Tests pass" is not a report.
- A **collection error is a failure.** So is an unexpected skip. An import that
  breaks means the file was never checked, which reads identically to green.
- If you edited a test so it would pass, say so on its own line in the report.
  That is a change to the specification, not to the code, and it needs to be
  visible as one.
- Never report done on a red suite. If a failure is pre-existing and unrelated,
  prove it: `git stash && pytest -k <that test>` and say the result.

---

## Step 2 - assume the current solution is wrong

Re-read the change as if reviewing someone else's work that you already know
contains a bug.

```
git status
git diff
git diff --stat
```

Untracked files carry no diff - read them in full.

**The rule: write down a concrete reason the change is wrong before writing
any reason it is right.** Not a checklist tick - a hypothesis with a
`file:line` and the input that breaks it. "Looks correct" is the output of the
process, never its starting point.

Prompts, each drawn from a bug this repo actually shipped or deliberately
designed against:

- **Does it fail by producing a plausible answer rather than an exception?**
  All four multi-market bugs did: the benchmark cache keyed without a market,
  pence read as pounds, a US ticket sized without FX (~88x too large, entirely
  ordinary-looking), `_pool()` returning the domestic balance for a typo.
- **Was a number re-declared instead of derived?** `risk_per_trade_pct =
  session_stop_pct / max_entries_per_session` lives in exactly one place. A
  second copy does not fail - it drifts.
- **Is there an `if backtest:` on any strategy or risk path?** If so, the
  backtest has stopped measuring the system being traded.
- **Does anything now "look armed and not be"?** `protection_state()` returns
  three states rather than two for this reason; a recorded TPIN authorisation
  can only reach `UNVERIFIED`. A new state that collapses to a boolean is this
  bug returning.
- **Are new cache and journal keys `{market}:{SYMBOL}`?** A bare ticker
  collision is not a crash - it is one company screened against another's
  balance sheet.
- **Does a missing input fail closed?** `fx.py` stands the market down and has
  no fallback rate, not a hardcoded 83, not the last rate seen. A default that
  lets the code continue is the failure mode, not the fix for it.
- **Are rejections still journalled, not just fills?** "Risk refused 14 setups
  because no strike fit the budget" is a finding, and it is invisible if only
  approvals are logged.
- **Look-ahead: does live learn this value on the same bar the backtest does?**
  `find_pivots()` is centred, so a pivot is knowable only `lookback` bars
  later. Every other bug gives a wrong answer you can see; this one gives a
  flattering one you cannot.
- **Does an exit still happen without a human?** Entries need a click; the
  breakeven shift, the +2R partial, the trail and the 15:10 flat do not, and
  must not start needing one.

For each hypothesis, report: the claim, why it would be wrong, where, and then
either the fix or the specific evidence that kills it.

**"No problems found" is only a valid conclusion after at least one hypothesis
was written down and killed with evidence.**

---

## Step 3 - backtest when the traded path moved

Which one to run follows from what changed:

| Changed | Run |
| --- | --- |
| `nifty_algo/swing/`, `signals.py`, swing settings in `config.py` | `.venv\Scripts\python.exe -m nifty_algo.swing.backtest --market india --years 3 --capital <pot>` |
| `strategy.py`, `risk.py`, `positions.py`, `governor.py`, `signals.py`, session governors in `config.py` | the option backtest - **no CLI exists**, see below |
| `ui/`, `broker/`, `scripts/`, tests, docs only | **Do not run one.** Say "not backtestable" and why - a number that could not have moved is not evidence. |

`signals.py` appears in two rows deliberately. Both books consume it, so both
backtests run.

The option backtest runs only from the Backtest page (`streamlit run app.py`)
or in process - `Backtester` takes bars and returns metrics, exactly as
`test_volumeless_index.py` calls it:

```
.venv\Scripts\python.exe -c "from nifty_algo.config import Config; from nifty_algo.backtest import Backtester, Mode; from nifty_algo.data.csv_feed import CsvFeed; bars = CsvFeed('data/nifty_5m.csv').get_bars(lookback_days=0); r = Backtester(Config()).run(bars, mode=Mode.UNDERLYING); print(r.metrics); [print('WARN:', w) for w in r.warnings]"
```

`data/nifty_5m.csv` is gitignored and may not exist - `python
scripts/fetch_history.py` builds it. `data/sample_nifty_5m.csv` is committed
but holds only 600 bars, which is too little for the 6/2-month walk-forward:
the run says so in `r.warnings` and falls back to a single in-sample pass.
**Print `r.warnings`.** An in-sample number quoted without that line is the
flattering-and-wrong case this whole file exists to prevent. Stay in
`Mode.UNDERLYING`; `SYNTHETIC_PREMIUM` is optimistic by 15-25%.

Rules that travel with any number this produces:

- **Pass `--capital` unless `data/settings.json` exists.** `swing_capital_inr`
  defaults to 0, and a zero pot sizes every ticket to zero: the run finishes
  clean and reports **no trades**, which reads exactly like "the gates were too
  tight". The module prints "the Indian swing pot is Rs 0" above the table -
  that line is the result, not a warning to scroll past. **A zero-trade
  backtest is a configuration failure to investigate, never a finding to
  report.**
- **It takes over ten minutes** on a funded pot for India over three years.
  Run it in the background and keep working; do not shorten `--years` to make
  it finish, because a shorter window is a different experiment.
- **Reproduce the distortion warnings the module prints**, do not quote the
  headline alone. Survivorship, point-in-time fundamentals and unreplayable
  news are structural for the swing book; `SYNTHETIC_PREMIUM` is optimistic by
  15-25% for the option book. A number stripped of them is a different claim
  from the one the module made.
- **Compare against the baseline, and be suspicious of improvement.** Measured
  2026-08-25, `--market india --years 3 --capital 100000`: **282 trades, 22.7%
  won against a 33.3% breakeven, expectancy -0.077R, total -21.6R, max
  drawdown 43.4R**. A change that turns that positive is a claim to be checked -
  most often the check finds a look-ahead, not an edge.
- **The pot is part of the experiment, so quote it.** That run reports **795
  entries blocked by cash** against 282 taken: at Rs 1,00,000 the account
  refuses far more trades than it fills, so the pot decides *which* trades the
  book gets, not merely their size. Two runs at different `--capital` are two
  different experiments and their expectancies are not comparable.
- **Never adjust a threshold to improve the result.** That is fitting the
  system to the answer - the same prohibition `halal.py` carries about tuning
  the debt test until the funds agree.
- **If it cannot run, say so.** Missing `data/cache` history is a fact to
  report ("run `scripts/fetch_swing_history.py` first"), not a reason to skip
  the step silently.

---

## Step 4 - report

Four lines, always, even when everything is clean:

1. **Tests** - the command, and pytest's own counts.
2. **Adversarial review** - the hypotheses raised, and for each, fixed or
   dismissed-with-evidence.
3. **Backtest** - which one ran and what it said with its caveats, or which
   did not run and why.
4. **Caveats** - any test modified, any failure left standing, anything not
   verified.
