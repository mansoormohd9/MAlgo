"""
The settings file: what persists, and the one thing that refuses to.

`data/settings.json` exists because a capital pot that vanishes on restart
silently changes what the app will do - the foreign pool used to do exactly
that, and the US scan would quietly go back to standing down with nothing
saying why.

The test that matters is the last one. A persisted "live orders ON" must not
survive a restart into an account that cannot execute a sell unattended.
"""
from __future__ import annotations

import json

import pytest

from nifty_algo import settings_store as ss
from nifty_algo.config import Config


@pytest.fixture
def path(tmp_path):
    return tmp_path / "settings.json"


def test_a_round_trip_preserves_every_pot(path):
    cfg = Config()
    cfg.capital.starting_capital = 250_000.0
    cfg.capital.swing_capital_inr = 30_000.0
    cfg.capital.foreign_capital_inr = 800_000.0
    ss.save(cfg, path)

    fresh = Config()
    ss.apply_to(fresh, path)

    assert fresh.capital.starting_capital == 250_000.0
    assert fresh.capital.swing_capital_inr == 30_000.0
    assert fresh.capital.foreign_capital_inr == 800_000.0


def test_a_missing_file_is_not_an_error(path):
    cfg = Config()
    assert ss.apply_to(cfg, path) == []
    assert cfg.capital.swing_capital_inr == 0.0


def test_a_corrupt_file_is_ignored_rather_than_fatal(path):
    """A hand-edited file should not stop the app opening."""
    path.write_text("{not json at all", encoding="utf-8")
    cfg = Config()

    ss.apply_to(cfg, path)

    assert cfg.capital.swing_capital_inr == 0.0


def test_a_wrong_type_is_skipped_and_reported(path):
    path.write_text(json.dumps({"swing_capital_inr": "thirty thousand"}),
                    encoding="utf-8")
    cfg = Config()

    notes = ss.apply_to(cfg, path)

    assert cfg.capital.swing_capital_inr == 0.0
    assert notes and "swing_capital_inr" in notes[0]


def test_only_account_facts_are_persisted():
    """
    Only YOUR account goes in here. Every strategy tunable stays in
    `config.py`, in version control, where changing one is a diff somebody can
    read rather than a JSON file you forgot you edited.

    THE THREE SLEEVE ENTRIES ARE ACCOUNT FACTS, NOT TUNABLES, and the
    distinction is worth stating because it is close. `factor_universe` is not
    a parameter someone is sweeping - it is which slice of the market this
    holder is permitted to own, and the Shariah screen is a property of the
    holder rather than of the strategy. Both belong with the pots. The
    formation, the band, `top_n` and the hold period stay in `config.py`, and
    this assertion is exact so that adding a fourth is a deliberate act.
    """
    keys = {key for _, _, key in ss.FIELDS}
    assert keys == {"option_capital_inr", "swing_capital_inr",
                    "foreign_capital_inr", "equity_dry_run", "ddpi_active",
                    "factor_capital_inr", "factor_universe",
                    "factor_halal_screened"}


def test_the_sleeve_settings_survive_a_restart(path):
    """A console opened once a month should come back configured."""
    cfg = Config()
    cfg.capital.factor_capital_inr = 750_000.0
    cfg.factor.universe = "nifty500"
    cfg.factor.halal_screened = True
    ss.save(cfg, path)

    fresh = Config()
    assert ss.apply_to(fresh, path) == []
    assert fresh.capital.factor_capital_inr == 750_000.0
    assert fresh.factor.universe == "nifty500"
    assert fresh.factor.halal_screened is True


def test_an_unregistered_universe_is_refused_rather_than_crashing_a_scan(path):
    """
    Every string coerces to a string, so a hand-edited universe would pass the
    type check and then raise `UnknownUniverse` deep inside a scan. It has to
    degrade to the unrestricted book with a note instead.
    """
    path.write_text(json.dumps({"factor_universe": "nifty42"}),
                    encoding="utf-8")
    cfg = Config()
    notes = ss.apply_to(cfg, path)
    assert cfg.factor.universe == "all"
    assert any("not a registered universe" in n for n in notes)


def test_live_orders_do_not_survive_a_restart_without_ddpi(path):
    """
    THE refusal.

    Live placement on an account where every sell needs a TPIN authorisation
    that expires nightly is the exact combination this design exists to
    prevent. You can turn it back on deliberately - what you cannot do is
    inherit it.
    """
    cfg = Config()
    cfg.equity_broker.dry_run = False
    cfg.equity_broker.ddpi_active = False
    ss.save(cfg, path)

    fresh = Config()
    notes = ss.apply_to(fresh, path)

    assert fresh.equity_broker.dry_run is True
    assert notes and "DDPI" in notes[0]


def test_live_orders_do_survive_when_ddpi_is_active(path):
    cfg = Config()
    cfg.equity_broker.dry_run = False
    cfg.equity_broker.ddpi_active = True
    ss.save(cfg, path)

    fresh = Config()
    notes = ss.apply_to(fresh, path)

    assert fresh.equity_broker.dry_run is False
    assert not notes
