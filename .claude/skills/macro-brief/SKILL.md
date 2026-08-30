---
name: macro-brief
description: Write the macro impact assessment - how current rate, inflation, currency, growth and volatility conditions bear on the holdings in this account. Runs the deterministic fact pack first and writes the briefing only from it. Triggers on macro, macro brief, rate environment, sector rotation, how do current conditions affect my portfolio.
---

# Macro impact assessment

You are writing a macro strategy briefing for the owner of this account. The
numbers come from a fact pack this repo generates; the reading of them is
yours.

## Step 1 - build the pack

```
.venv\Scripts\python.exe -m nifty_algo.research macro --market india --json
```

`--market` accepts `india`, `us` or `uk`. Add `--refresh` to force a fresh
download of the macro series (they cache for a few hours otherwise).

If the command fails, stop and report the failure. Do not write the briefing
from memory - a macro brief with invented levels is worse than no brief,
because every number in it is checkable and none of them would check out.

## Step 2 - the two rules

These are not style preferences. They are the reason the pack exists.

**1. Cite no figure that is not in the pack.** Every level, change,
correlation, beta and weight in your briefing must appear in `sections[].facts`
or `sections[].rows`. You may do arithmetic on pack figures if you show the
inputs. You may not supply a number from your own knowledge - not a CPI print,
not an earnings estimate, not a historical average - without labelling it
explicitly as your own and giving its source and date.

**2. Name everything the pack could not establish.** The top-level
`unavailable` list is exactly this. Each entry has a `label` and a `note`
saying *why*. Work them into the briefing as findings rather than dropping
them, because they are not all the same kind of absence:

- *"India publishes no free machine endpoint for CPI"* is a fact about the
  world. Say so, and say what you used instead.
- *"the holdings snapshot is incomplete"* means the account was only partly
  readable, so no percentage of the book is trustworthy. Say which source
  failed, from the `caveats` list.

A section that silently omits an unavailable fact reads as though the fact was
fine.

## Step 3 - the judgment you ARE being asked for

`judgment_required` lists what the data cannot decide - the Fed's path,
geopolitics, trade and supply chains, employment's read on consumer spending,
where in the cycle we are, and the sector rotation that follows.

Answer these. That is the value you add. But mark them clearly as assessment
rather than measurement - a sentence like "This is a view, not a reading:" or a
short **Judgment** label - so the reader can tell which half of the briefing
would change if the data changed and which would change if you changed your
mind.

## Step 4 - the shape of the briefing

An executive macro briefing, in this order:

1. **The call, in three sentences.** What the environment is, what it means for
   this specific book, and the one action that follows.
2. **Rates, inflation, growth, currency, employment, risk.** One short section
   each, leading with the pack's figure and its twelve-month direction. The
   twelve-month direction matters more than the level.
3. **This book against those factors.** The `Your book against these factors`
   and `Sector concentration` sections are the heart of the briefing and are
   the part no generic macro note can contain. Quote the betas and factor
   correlations with their session counts.
4. **Sector rotation.** Your judgment, labelled as such, argued from the
   weights and sensitivities above.
5. **Portfolio adjustments.** Specific, and see the constraint below.
6. **Timeline.** Also judgment. None of the series carries a lead time.

## Constraints that are not negotiable

- **Never invent a sizing rule.** Per-trade risk in this repo is
  `session_stop_pct / max_entries_per_session` applied to the pool that pays -
  one formula, in `CapitalConfig`. If you propose a position size, derive it
  from that. A second rule is a second set of governors.
- **A correlation is not a cause.** The pack's correlations are over one
  regime and carry their `n`. Say "these moved together" and not "rates drove".
- **Percentages of the book are withheld on an incomplete snapshot** and come
  through as `null`. Do not fill them in, and do not read `null` as zero.
- Say plainly that this is analysis and not investment advice, once, at the
  end. Do not repeat it in every section.
