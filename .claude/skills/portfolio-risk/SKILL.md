---
name: portfolio-risk
description: Write the portfolio risk assessment - correlation, concentration, fund look-through, a replayed drawdown, tail risk, liquidity and sizing against the governors. Runs the deterministic fact pack first and writes only from it. Triggers on portfolio risk, risk assessment, stress test, how concentrated am I, what happens in a crash, correlation between my holdings.
---

# Portfolio risk framework

You are writing a risk assessment of this account. Unlike the macro briefing,
almost all of this one is computed - your job is mostly to read it correctly
and rank what matters.

## Step 1 - build the pack

```
.venv\Scripts\python.exe -m nifty_algo.research risk --market india --json
```

If the command fails, stop and report it. Never estimate a portfolio's risk
from memory of what it held.

## Step 2 - read `stood_down` and `caveats` FIRST

If `stood_down` is non-empty, the report could not be produced. Say so and
stop. The usual cause is that no connector returned any position - which is
**not** a low-risk account, it is an unread one.

`caveats` carries the things that make every figure below mean less than it
appears to. The two that recur:

- **The snapshot is incomplete.** A connector failed or a currency would not
  convert. Every `weight_pct` then comes through as `null` and you must not
  fill it in. Report the absolute values and say which source failed.
- **Today's weights, applied to history.** Every replayed figure answers "how
  would this book have behaved", never "how did it behave". Say it once, plainly.

## Step 3 - the two rules

**1. Cite no figure that is not in the pack.** Every correlation, drawdown,
percentile, weight and turnover figure must come from `sections[].facts` or
`sections[].rows`.

**2. Name everything in `unavailable`.** A position with no cached bars is
absent from every correlation and every replay - that is a hole in the
analysis, not a clean position, and the reader has to be told which names it
applies to.

## Step 4 - what each section is actually saying

- **Correlation.** `Average pairwise correlation` is the diversification
  number. Six names at 0.75 is one position with extra brokerage. Note that
  correlations rise in a sell-off, so this table flatters the book relative to
  the replayed drawdown - which is why the drawdown outranks it.
- **Concentration.** `Effective number of positions` is the inverse Herfindahl.
  Compare it to the count of positions; the gap is how much of the book is
  really one or two names.
- **Fund look-through.** A direct position that your ETFs also hold is adding
  to a bet, not diversifying into one. It is a fact to state, never a
  recommendation to sell - concentration is the owner's decision.
- **Recession stress test.** REPLAYED, not modelled. These are real
  peak-to-trough episodes in the cached benchmark with this book's weights
  applied to what those names actually did. Quote `coverage_pct` alongside
  every drawdown - a figure covering 60% of the book is a different claim from
  one covering 99%. Do not describe it as a VaR or as a simulation.
- **Tail risk.** Empirical percentiles of actual sessions, not modelled
  quantiles. Its own tail is only as deep as the history is long, so do not
  quote a 1-in-100 day from 200 sessions as though it were one.
- **Liquidity.** `pct_of_one_days_volume` is the useful column: a position
  worth a fraction of a percent of a day's volume exits in minutes; one worth
  10% or more *is* the price.
- **Sizing.** `Deployed against the pot` over 100% means the configured pot is
  smaller than the account actually holding these positions - which makes every
  risk figure derived from it wrong in the reassuring direction. Flag that
  loudly if you see it.

## Step 5 - the judgment you ARE being asked for

`judgment_required` asks for the top three risks, the hedges, the rebalancing
percentages, and whether the US estate-tax exposure belongs in the top three.

Rank the risks. That is the value you add, and it is genuinely a judgment:
the pack can tell you the book is 49% in one sector and 0.78 correlated in its
largest pair, but not which of those you should do something about first.

Two things to weigh honestly:

- The cheapest way to reduce a concentration is usually to sell some of it,
  not to buy a hedge against it. Price the hedge before recommending it.
- For an Indian resident holding US-domiciled funds, the US-situs estate
  exposure in `swing/crossborder.py` is *structural and unrecoverable*, unlike
  every market risk in the pack. It may well outrank them.

## Step 6 - the shape

1. **The three risks that matter, ranked**, each with the pack figure behind it.
2. **What the book is**, from the concentration and correlation sections.
3. **What it loses**, from the replayed drawdown and the tails.
4. **What it can get out of**, from liquidity.
5. **Whether it is sized right**, against the governors.
6. **Actions** - specific, and derived from
   `session_stop_pct / max_entries_per_session` applied to the paying pool.
   Never invent a second sizing rule.

Radical transparency means saying what you do not know as clearly as what you
do. The `unavailable` list is not a footnote in this briefing; it is part of
the risk. Analysis, not investment advice - say it once, at the end.
