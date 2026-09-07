"""
Data paths must not depend on where the process was launched.

THIS IS A REGRESSION SUITE FOR A BUG THAT READ AS MISSING DATA. Launching the
console from any directory but the repo root made the Monthly sleeve report

    No factor cache at data/cache/factor_daily_india.parquet.
    Run: python scripts/fetch_factor_history.py --years 10

with a 62 MB cache present the whole time - and following that advice costs
~35 minutes and thousands of broker calls to rebuild a file that already
exists. Every data path in `config.py` is relative, so "relative to what" was
answered by the CWD.

THE THREE QUIETER ONES MATTER MORE THAN THE LOUD ONE. The bar cache at least
raised. From the wrong directory:

  - `membership.load()` found no constituent lists, so the sleeve reported
    "50 / Next 50 split unavailable" on a repo that commits all four CSVs, and
    a `nifty500` universe raised `UnknownUniverse`;
  - `fundamentals.read_cached` returned `{}`, so every name arrived
    unclassified - which the halal screen treats as a REJECT - while
    `load_fundamentals` refetched the whole shortlist from Yahoo one request
    at a time;
  - `halal.load_overrides` found no file and returned no rulings, so hand-made
    verdicts silently stopped applying.

None of those raise. All of them answer a different question fluently, which
is the failure this repo refuses everywhere else.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nifty_algo import paths
from nifty_algo.config import DEFAULT
from nifty_algo.factor import membership as mb
from nifty_algo.swing import fundamentals as fund
from nifty_algo.swing import halal


# ------------------------------------------------------------ the resolver

def test_the_repo_root_is_the_directory_holding_app_py():
    assert (paths.REPO_ROOT / "app.py").exists()
    assert (paths.REPO_ROOT / "nifty_algo").is_dir()
    assert (paths.REPO_ROOT / "data").is_dir()


def test_a_relative_path_is_anchored_to_the_repo():
    assert paths.at_root("data/cache") == paths.REPO_ROOT / "data" / "cache"
    assert paths.at_root(Path("data") / "x.csv") == (
        paths.REPO_ROOT / "data" / "x.csv")


def test_an_absolute_path_is_returned_UNCHANGED(tmp_path):
    """
    The passthrough is what keeps every `tmp_path` fixture honest.

    Always joining to the root would relocate a test's temporary directory,
    and a user's `--cache /elsewhere/x.parquet`, into the repo - a wrong
    answer that reads like a working one.
    """
    assert paths.at_root(tmp_path) == tmp_path
    assert paths.at_root(str(tmp_path)) == tmp_path


def test_the_resolver_does_not_care_where_it_is_called_from(monkeypatch,
                                                            tmp_path):
    before = paths.at_root("data/cache")
    monkeypatch.chdir(tmp_path)
    assert paths.at_root("data/cache") == before


# --------------------------------------------- the four readers that broke

#: The repo root computed WITHOUT `paths.at_root`, so an existence check
#: cannot be fooled by the very resolver under test. If `at_root` regressed,
#: the file is still found here and the assertion below fails - rather than
#: the test skipping and reporting "not built on this machine", which is the
#: same conflation of "absent" and "unfindable" that caused the whole bug.
_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def elsewhere(monkeypatch, tmp_path):
    """Run from a directory that is definitely not the repo root."""
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() != paths.REPO_ROOT
    return tmp_path


def test_the_factor_cache_is_found_from_any_directory(elsewhere):
    """
    The loud one. Skipped only if the cache genuinely is not built, because
    it is gitignored - and asserted rather than skipped whenever it exists,
    since "the file is absent" and "the file is unfindable" are the two states
    this test exists to keep apart.
    """
    from nifty_algo.factor import drawdown as fd

    path = _ROOT / DEFAULT.factor.cache_dir / DEFAULT.factor.cache_name
    if not path.exists():
        pytest.skip(f"{path.name} is not built on this machine")
    bars, _bench = fd.load(DEFAULT.factor)
    assert bars, "the cache resolved but produced no bars"


def test_the_error_names_an_absolute_path_and_does_not_advise_a_refetch(
        tmp_path):
    """
    The advice was the worst part of the bug: a relative path in the message
    is indistinguishable from a genuinely missing file, and it sent you to a
    35-minute fetch. The message now shows where it looked.
    """
    from dataclasses import replace
    from nifty_algo.factor import drawdown as fd

    missing = replace(DEFAULT.factor, cache_dir=str(tmp_path),
                      cache_name="nope.parquet")
    with pytest.raises(FileNotFoundError) as excinfo:
        fd.load(missing)
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "this is not the error you think it is" in message


def test_membership_is_found_from_any_directory(elsewhere):
    """
    The quietest of the four: no exception, just an empty `Membership` and a
    console saying the 50 / Next 50 split is unavailable.
    """
    members = mb.load()
    assert members.sets.get("nifty500"), "nifty500 list not found"
    assert "unavailable" not in members.note()


def test_a_nifty500_universe_resolves_from_any_directory(elsewhere):
    """This raised `UnknownUniverse` from the wrong directory."""
    from nifty_algo.factor import restriction as restr

    fn, note = restr.resolver(DEFAULT, "nifty500", {}, None)
    assert fn(None)
    assert "nifty500" in note


def test_the_fundamentals_cache_is_found_from_any_directory(elsewhere):
    """
    An empty cache is not a neutral outcome. `load_fundamentals` refetches the
    whole shortlist over the network, and `read_cached` returning `{}` leaves
    every name unclassified - which the halal screen treats as a REJECT.
    """
    if not (_ROOT / DEFAULT.swing.cache_dir / fund.CACHE_NAME).exists():
        pytest.skip("fundamentals cache is not built on this machine")

    assert fund.cache_path(DEFAULT).is_absolute()
    from nifty_algo.swing import markets as markets_mod
    cached = fund.read_cached(DEFAULT, markets_mod.factor_market(DEFAULT))
    assert cached, "the cache resolved but produced no entries"


def test_the_halal_overrides_are_found_from_any_directory(elsewhere):
    """
    `load_overrides` returns `({}, [])` for a missing file, so from the wrong
    directory your rulings stopped applying with no warning - the one outcome
    its own docstring calls worse than no ruling at all.
    """
    configured = Path(DEFAULT.swing.halal.overrides_csv)
    if not (_ROOT / configured).exists():
        pytest.skip("no overrides file committed on this machine")
    # A missing file returns ({}, []) - indistinguishable from a clean file -
    # so the observable proof is that the resolver reaches the committed one.
    assert halal.at_root(configured) == _ROOT / configured
    _overrides, warnings = halal.load_overrides(configured)
    assert not warnings


def test_an_explicit_absolute_override_path_still_wins(tmp_path, elsewhere):
    """An explicit path must never be relocated into the repo."""
    target = tmp_path / "rulings.csv"
    target.write_text(
        "symbol,verdict,note,reviewed_on\n"
        "ACME,compliant,checked by hand,2026-09-07\n", encoding="utf-8")
    overrides, warnings = halal.load_overrides(target)
    assert not warnings, warnings
    assert overrides["ACME"]["verdict"] == "compliant"


# ------------------------------------------- the console, end to end

def test_the_app_anchors_its_working_directory_before_reading_anything(
        elsewhere):
    """
    THE END-TO-END GUARD FOR THE BUG AS REPORTED.

    Everything above tests a reader in isolation. This tests the thing that
    actually broke: `streamlit run app.py` from the wrong directory. `app.py`
    chdirs to its own location before `load_dotenv()` and before the page
    imports, so by the time any page can read a file the CWD is the repo root.

    Asserting the CWD rather than clicking Run scan is deliberate - the scan
    reads a 62 MB gitignored parquet, and the readers it uses are each pinned
    above. What is unique to this test is the ORDERING: an anchor that ran
    after the first read would pass every test above and still fail in use.
    """
    from conftest import sign_in
    from streamlit.testing.v1 import AppTest

    assert Path.cwd() == elsewhere
    at = AppTest.from_file(str(paths.REPO_ROOT / "app.py"), default_timeout=180)
    sign_in(at)
    at.run()

    assert not at.exception
    assert Path.cwd() == paths.REPO_ROOT, (
        "app.py did not anchor the working directory, so every relative data "
        "path still depends on where it was launched from")
