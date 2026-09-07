"""
Where the repo is, so a relative data path cannot mean two different things.

EVERY DATA PATH IN `config.py` IS RELATIVE - `data/cache`, `data/nifty100.csv`,
`data/settings.json`, `journal/`. Relative to WHAT was never stated, and the
answer is "the current working directory", which is a property of how the
process was launched rather than of the repo. Launch the console from anywhere
but the repo root and `data/cache/factor_daily_india.parquet` resolves to a
file that does not exist.

WHAT THAT LOOKED LIKE. The Monthly sleeve reported

    No factor cache at data/cache/factor_daily_india.parquet.
    Run: python scripts/fetch_factor_history.py --years 10

with a 62 MB cache sitting in `data/cache/` the whole time - and the advice was
actively harmful, because re-running that fetch is ~35 minutes and thousands of
broker calls to rebuild a file that was already there. A missing file and an
unfindable file are different faults and only one of them is fixed by
downloading anything.

SO: `at_root` anchors a relative path to the repo, and leaves an absolute one
alone. The passthrough is what keeps `tmp_path` fixtures working - a test that
points `cache_dir` at a temporary directory means that directory and not a
subdirectory of the repo.

This is deliberately NOT a general path abstraction. It resolves one thing and
it is used at the sites that read committed or downloaded data, closest to the
`open()`, so a caller that already holds an absolute path never notices it.
"""
from __future__ import annotations

from pathlib import Path

#: The repo root - the directory holding `app.py`, `data/` and `nifty_algo/`.
#: Derived from this file's location, which is the one anchor that does not
#: depend on how the process was started.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def at_root(value) -> Path:
    """
    `value` as an absolute path: unchanged if it already is, else under the repo.

    Anything already absolute is returned as-is, so an explicit path always
    wins over the anchor. That ordering matters more than it looks: the
    alternative - always joining to the root - would silently relocate a
    test's `tmp_path` and a user's `--cache /elsewhere/x.parquet` into the
    repo, which is a wrong answer that reads like a working one.
    """
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path
