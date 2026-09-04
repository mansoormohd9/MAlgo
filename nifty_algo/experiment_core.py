"""
Choosing between rule sets, for any book in this repo.

THREE BOOKS, ONE HARNESS. The swing book grew a variant sweep first
(`swing/experiment.py`); the intraday-equity book re-derived it
(`intraday_equity/experiment.py`); the option book needed a third. At three
copies the statistics start to drift, and the statistics are the entire point -
a sweep whose confidence interval is computed one way here and another way
there cannot be compared across books.

So this module owns everything that is about MEASUREMENT rather than about a
particular book: the cell, the sweep, the pooling rule, the interval, the sign
test, the selection rule, the ledger and the table. What stays in each book is
its `VARIANTS` table and the loop that runs its own backtester.

WHAT THE NUMBERS MEAN, AND WHY THEY ARE COMPUTED THIS WAY

POOLED expectancy is TRADE-WEIGHTED, not a mean of fold means. A fold with 3
trades and one with 60 are not equally informative and a plain average lets the
thin fold swing the answer.

THE INTERVAL RESAMPLES FOLDS, NOT TRADES. Trades inside a fold share a regime,
a volatility level and often a session, so they are not independent draws.
Bootstrapping trades produces a reassuringly narrow interval that means
nothing. This is the number that decides whether a result clears zero.

THE SIGN TEST IS THE ONE THAT ACTUALLY DISCRIMINATES, and it is the piece none
of the three copies had. Measured on the option book: a warm-up variant
improved pooled expectancy from -0.051R to -0.019R and halved total loss, which
reads like a result - and then won 10 of 21 walk-forward windows against a
coin-flip expectation of 10.5. A pooled average can be carried by one good
stretch; consistency across independent windows cannot. Report both, and
believe the second.

SELECTION IS SCORED, NOT ASSUMED. `selected_by_train` asks what picking the
best train variant at each fold would actually have paid on the untouched test
window. On the swing book that paid WORSE than the baseline, which is itself
the finding: it says train expectancy had no predictive power, and the honest
move is to stop choosing and run the baseline.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

#: Below this a cell is printed but cannot vote in a selection. A variant that
#: looks best on 6 trades has not been measured, it has been sampled.
MIN_TRADES_TO_JUDGE = 20

DEFAULT_LEDGER_DIR = Path("data/experiments")


# ------------------------------------------------------------------ variants

@dataclass(frozen=True)
class Variant:
    """One named change to the config, with the hypothesis it is testing."""
    key: str
    label: str
    why: str
    apply: Callable[[object], None]

    def configure(self, base):
        """
        A DEEP copy. `dataclasses.replace` is shallow, so a variant that
        mutates `cfg.signal` would reach through and edit the caller's config -
        every subsequent variant in the sweep would inherit it, and the sweep
        would silently measure a cumulative stack of changes rather than each
        one alone.
        """
        cfg = copy.deepcopy(base)
        self.apply(cfg)
        return cfg


def set_on(section: str, **fields) -> Callable[[object], None]:
    """
    `apply` that assigns onto one config section, e.g. `set_on("signal",
    max_delta=0.30)`.

    Takes the section by name because the option book's levers are spread
    across `cfg.signal`, `cfg.trade`, `cfg.session` and `cfg.regime` - unlike
    the swing and intraday-equity books, whose tunables live under a single
    namespace. Asserting the attribute exists turns a typo into an error at
    sweep-build time rather than a variant that silently changes nothing.
    """
    def apply(cfg) -> None:
        target = getattr(cfg, section)
        for name, value in fields.items():
            if not hasattr(target, name):
                raise AttributeError(
                    f"cfg.{section} has no field {name!r} - a variant that "
                    f"sets a non-existent field is a variant that tests nothing"
                )
            setattr(target, name, value)
    return apply


# --------------------------------------------------------------------- cells

@dataclass
class Cell:
    """One (variant, fold, phase) result."""
    variant: str
    fold: int
    phase: str                     # "train" | "test"
    start: date
    end: date
    trades: int
    expectancy_r: float
    total_r: float
    win_rate: float = 0.0
    breakeven_win_rate: float = 0.0
    max_drawdown_r: float = 0.0
    #: Book-specific columns (the swing book's `capital_blocks`, the
    #: intraday-equity book's `friction_r`). Held in a dict rather than as
    #: fields so a third book does not have to widen the shared type, and
    #: flattened into the ledger by `as_row`.
    extra: dict = field(default_factory=dict)

    @property
    def judgeable(self) -> bool:
        return self.trades >= MIN_TRADES_TO_JUDGE

    def as_row(self) -> dict:
        row = {
            "variant": self.variant, "fold": self.fold, "phase": self.phase,
            "start": self.start, "end": self.end, "trades": self.trades,
            "expectancy_r": self.expectancy_r, "total_r": self.total_r,
            "win_rate": self.win_rate,
            "breakeven_win_rate": self.breakeven_win_rate,
            "max_drawdown_r": self.max_drawdown_r,
            "judgeable": self.judgeable,
        }
        row.update(self.extra)
        return row


def cell_from_metrics(variant: str, fold: int, phase: str, start: date,
                      end: date, metrics, **extra) -> Cell:
    """Build a Cell from any book's `Metrics`. Tolerates None."""
    m = metrics
    return Cell(
        variant=variant, fold=fold, phase=phase, start=start, end=end,
        trades=getattr(m, "trades", 0) if m else 0,
        expectancy_r=getattr(m, "expectancy_r", 0.0) if m else 0.0,
        total_r=getattr(m, "total_r", 0.0) if m else 0.0,
        win_rate=getattr(m, "win_rate", 0.0) if m else 0.0,
        breakeven_win_rate=getattr(m, "breakeven_win_rate", 0.0) if m else 0.0,
        max_drawdown_r=getattr(m, "max_drawdown_r", 0.0) if m else 0.0,
        extra=extra,
    )


# -------------------------------------------------------------------- sweeps

def two_sided_sign_p(wins: int, n: int) -> float:
    """
    Exact two-sided binomial p for `wins` successes in `n` trials at p=0.5.

    Written out rather than pulled from scipy so the number in a report can be
    read off the code. Ties are the caller's problem: a sign test excludes
    them from `n` rather than splitting them, because a tie is not evidence
    either way.
    """
    if n <= 0:
        return float("nan")
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


@dataclass
class Sweep:
    cells: list = field(default_factory=list)
    scan_passes: int = 0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_row() for c in self.cells])

    def phase(self, phase: str) -> pd.DataFrame:
        df = self.frame()
        return df[df["phase"] == phase] if not df.empty else df

    def pooled(self) -> pd.DataFrame:
        """Trade-weighted expectancy per variant per phase."""
        df = self.frame()
        if df.empty:
            return df
        rows = []
        for (v, ph), g in df.groupby(["variant", "phase"]):
            n = int(g["trades"].sum())
            r = float((g["expectancy_r"] * g["trades"]).sum())
            rows.append({"variant": v, "phase": ph, "trades": n,
                         "expectancy_r": r / n if n else 0.0,
                         "total_r": float(g["total_r"].sum())})
        return pd.DataFrame(rows)

    def bootstrap_ci(self, key: str, phase: str = "test",
                     draws: int = 2000, seed: int = 7) -> tuple:
        """95% interval on a variant's expectancy, resampling FOLDS."""
        df = self.frame()
        if df.empty:
            return (float("nan"), float("nan"))
        g = df[(df["variant"] == key) & (df["phase"] == phase)]
        g = g[g["trades"] > 0]
        if len(g) < 3:
            return (float("nan"), float("nan"))
        exp = g["expectancy_r"].to_numpy()
        wts = g["trades"].to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(draws):
            idx = rng.integers(0, len(exp), len(exp))
            w = wts[idx]
            out.append(float((exp[idx] * w).sum() / w.sum()) if w.sum() else 0.0)
        return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

    def sign_test(self, key: str, against: str = "baseline",
                  phase: str = "test") -> dict:
        """
        In how many folds did `key` beat `against`, and is that more than a
        coin flip?

        The discriminating statistic. A pooled average can be carried by one
        good stretch; winning most of many independent windows cannot. Folds
        where either side has no trades are dropped, and exact ties are
        excluded from `n` rather than counted as half.
        """
        df = self.frame()
        blank = {"wins": 0, "losses": 0, "ties": 0, "n": 0,
                 "win_rate": float("nan"), "p": float("nan"),
                 "coin_flip": float("nan")}
        if df.empty:
            return blank
        a = df[(df["variant"] == key) & (df["phase"] == phase)]
        b = df[(df["variant"] == against) & (df["phase"] == phase)]
        if a.empty or b.empty:
            return blank
        a = a.set_index("fold")["expectancy_r"]
        b = b.set_index("fold")["expectancy_r"]
        common = sorted(set(a.index) & set(b.index))
        wins = sum(1 for f in common if a[f] > b[f])
        losses = sum(1 for f in common if a[f] < b[f])
        ties = len(common) - wins - losses
        n = wins + losses
        return {
            "wins": wins, "losses": losses, "ties": ties, "n": n,
            "win_rate": (wins / n) if n else float("nan"),
            "p": two_sided_sign_p(wins, n),
            "coin_flip": n / 2.0 if n else float("nan"),
        }

    def selected_by_train(self) -> pd.DataFrame:
        """What walk-forward SELECTION would actually have paid."""
        df = self.frame()
        if df.empty:
            return df
        rows = []
        for fold, g in df[df["phase"] == "train"].groupby("fold"):
            ok = g[g["trades"] >= MIN_TRADES_TO_JUDGE]
            if ok.empty:
                continue
            pick = ok.sort_values("expectancy_r", ascending=False).iloc[0]
            test = df[(df["fold"] == fold) & (df["phase"] == "test")
                      & (df["variant"] == pick["variant"])]
            if test.empty:
                continue
            t = test.iloc[0]
            rows.append({"fold": fold, "picked": pick["variant"],
                         "train_r": pick["expectancy_r"],
                         "test_r": t["expectancy_r"],
                         "test_trades": t["trades"]})
        return pd.DataFrame(rows)


# -------------------------------------------------------------------- ledger

def write_ledger(sw: Sweep, directory: Path | str, prefix: str,
                 names: str = "") -> Optional[Path]:
    """
    One parquet per sweep invocation, timestamped to the second.

    To the second because signature groups are run as separate concurrent
    processes and a coarser stamp would have them overwrite each other.
    """
    df = sw.frame()
    if df.empty:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tail = f"_{names[:40]}" if names else ""
    path = directory / f"{prefix}_{stamp}{tail}.parquet"
    df.to_parquet(path, index=False)
    return path


def read_ledger(directory: Path | str = DEFAULT_LEDGER_DIR,
                prefix: str = "sweep") -> pd.DataFrame:
    """
    Every ledger under `directory`, newest winning on a repeated cell.

    Sorted by filename, which sorts by timestamp because of how `write_ledger`
    names them, so `keep="last"` is `keep="newest"`. Re-running one variant to
    fix it therefore supersedes the old row instead of double-counting it.
    """
    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(directory.glob(f"{prefix}_*.parquet")):
        try:
            f = pd.read_parquet(path)
        except Exception:
            continue                  # a torn file is not the whole ledger
        f["source"] = path.name
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    key = [c for c in ("variant", "fold", "phase") if c in frame.columns]
    if not key:
        return frame
    return frame.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


# --------------------------------------------------------------------- table

def report(sw: Sweep, baseline_key: str = "baseline") -> str:
    """The table, and nothing that decides anything."""
    lines: list[str] = []
    pooled = sw.pooled()
    if pooled.empty:
        return "no cells - not enough history for even one fold"

    tr = pooled[pooled["phase"] == "train"].set_index("variant")
    te = pooled[pooled["phase"] == "test"].set_index("variant")
    if te.empty:
        return "no test cells - nothing was scored out of sample"

    lines.append(f"{'variant':<14} {'train R':>9} {'n':>6} {'TEST R':>9} "
                 f"{'n':>6} {'gap':>8} {'95% CI (test)':>22} "
                 f"{'vs base':>10} {'p':>7}")
    for key in te.sort_values("expectancy_r", ascending=False).index:
        t = tr.loc[key] if key in tr.index else None
        e = te.loc[key]
        gap = (t["expectancy_r"] - e["expectancy_r"]) if t is not None else float("nan")
        lo, hi = sw.bootstrap_ci(key)
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo == lo else "n/a"
        st = sw.sign_test(key, against=baseline_key)
        folds = f"{st['wins']}/{st['n']}" if st["n"] else "-"
        p = f"{st['p']:.3f}" if st["p"] == st["p"] else "-"
        if key == baseline_key:
            folds, p = "-", "-"
        lines.append(
            f"{key:<14} {t['expectancy_r'] if t is not None else 0:+9.3f} "
            f"{int(t['trades']) if t is not None else 0:6d} "
            f"{e['expectancy_r']:+9.3f} {int(e['trades']):6d} "
            f"{gap:+8.3f} {ci:>22} {folds:>10} {p:>7}")

    sel = sw.selected_by_train()
    lines.append("")
    if not sel.empty:
        paid = float((sel["test_r"] * sel["test_trades"]).sum()
                     / max(sel["test_trades"].sum(), 1))
        base = (te.loc[baseline_key, "expectancy_r"]
                if baseline_key in te.index else 0.0)
        lines.append(f"walk-forward SELECTION paid {paid:+.3f}R over "
                     f"{int(sel['test_trades'].sum())} trades, against a "
                     f"baseline of {base:+.3f}R.")
        lines.append("If selection is worse, train expectancy has no "
                     "predictive power here and the honest move is to stop "
                     "choosing and run the baseline.")
    lines.append("")
    if sw.scan_passes:
        lines.append(f"scan passes: {sw.scan_passes} (variants sharing a "
                     f"signature shared one)")
    lines.append("READ THE TEST COLUMN. The gap is how much of a variant was "
                 "fitting.")
    lines.append("'vs base' is folds won against the baseline and 'p' its "
                 "two-sided sign test - a variant that wins on pooled R but "
                 "not on folds was carried by one stretch.")
    return "\n".join(lines)
