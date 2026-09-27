"""
Step 1 - Connect: every account that holds your money, in one place.

WHY THIS PAGE EXISTS. Before it, the Kite login lived inside Trade book (a page
about swing tickets), the only record of IBKR was a stub note on Portfolio, and
balances with no API - EPF, PPF, NPS, FDs, gold - could be recorded only by
hand-editing a CSV you had to know existed. A first-time user could not find
the first thing they needed to do.

WHICH ACCOUNTS YOU HOLD IS A CLAIM, AND IT IS SAVED. `PortfolioConfig.connectors`
decides how a missing answer is read: a connector you enable and that cannot
answer makes the snapshot incomplete, so every percentage downstream is
withheld rather than computed against a partial book. That is correct - and it
is why enabling an account here is a statement about your money, not a
preference.

Nothing on this page places an order.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from .components import banner, kite_login_panel
from .page_settings import CREDENTIAL_GROUPS
from .state import get_config, get_kite_session, get_snapshot, save_settings
from .theme import get_palette
from .. import onboarding
from ..paths import at_root
from ..portfolio import cas as cas_mod
from ..portfolio import ibkr as ibkr_mod
from ..portfolio import manual as manual_mod
from ..portfolio import registry
from ..portfolio.base import ASSET_CLASSES

#: What each connector is, in words a first-time user can choose from.
DESCRIPTIONS = {
    "manual": "Balances no API reports: EPF, PPF, NPS, FDs, gold, cash, "
              "and anything held elsewhere.",
    "kite": "Zerodha - your Indian shares and ETFs, read live.",
    "ibkr": "Interactive Brokers - US/UK holdings bought under LRS, read from "
            "a Flex statement.",
    "cas": "Mutual funds across every AMC, from a CAMS/KFintech CAS PDF.",
}

#: The editable columns of the balances table, and the full CSV header the
#: manual connector reads - kept in the connector's own order.
CSV_COLUMNS = ("market", "symbol", "name", "quantity", "average_price",
               "last_price", "value", "cost", "currency", "asset_class",
               "account", "as_of")


def render() -> None:
    p = get_palette()
    cfg = get_config()
    st.title("Connect your accounts")
    st.caption("Step 1 of 6. Tell the console where your money is. Every "
               "later step reads what this page connects.")

    enabled = _choose(cfg)

    if "kite" in enabled:
        _kite(p)
    if "ibkr" in enabled:
        _ibkr(p)
    if "cas" in enabled:
        _cas(cfg, p)
    _manual(cfg, p)

    st.divider()
    _read(cfg, p)


# ---------------------------------------------------------------- choose

def _choose(cfg) -> list[str]:
    st.subheader("Where do you hold money?")
    options = [k for k in registry.keys() if k != "manual"]
    current = [k for k in cfg.portfolio.connectors if k in options]
    chosen = st.multiselect(
        "Accounts", options, default=current, key="connect_enabled",
        format_func=lambda k: registry.CONNECTORS[k].label
        if hasattr(registry.CONNECTORS[k], "label") else k,
        help="Enabling an account says you hold money there. If it then "
             "cannot be read, percentages are withheld rather than computed "
             "against part of your wealth.")
    for k in options:
        st.caption(f"**{k}** — {DESCRIPTIONS.get(k, '')}")
    wanted = tuple(["manual"] + [k for k in options if k in chosen])
    if wanted != tuple(cfg.portfolio.connectors):
        cfg.portfolio.connectors = wanted
        save_settings()
        st.session_state.pop("portfolio_snapshot", None)
    return list(wanted)


def _keys_present(group: str) -> list[tuple[str, bool]]:
    return [(k, bool(os.getenv(k, "").strip()))
            for k in CREDENTIAL_GROUPS.get(group, [])]


def _key_list(group: str) -> None:
    st.markdown("  \n".join(f"{'✅' if ok else '⬜'} `{k}`"
                            for k, ok in _keys_present(group)))


# ---------------------------------------------------------------- kite

def _kite(p) -> None:
    with st.container(border=True):
        st.markdown("#### Zerodha (Kite)")
        c1, c2 = st.columns([1, 2])
        with c1:
            _key_list("Kite (Zerodha)")
            st.caption("From developers.kite.trade, in `.env`. Values are "
                       "never shown.")
        with c2:
            session = get_kite_session()
            if session is not None and getattr(session, "authenticated", False):
                st.success("Logged in for today.")
                st.caption("Kite's token dies overnight — there is no refresh "
                           "token, so this is a once-a-morning step.")
            else:
                if session is not None:
                    st.warning("Not logged in today.")
                kite_login_panel(session)


# ---------------------------------------------------------------- ibkr

def _ibkr(p) -> None:
    with st.container(border=True):
        st.markdown("#### Interactive Brokers")
        c1, c2 = st.columns([1, 2])
        with c1:
            _key_list("IBKR (Flex Web Service)")
        with c2:
            if ibkr_mod.IbkrConnector().is_configured():
                st.success("Flex token and query id present.")
                st.caption(f"Statements are cached for "
                           f"{ibkr_mod.CACHE_HOURS:.0f}h — IBKR throttles "
                           f"Flex requests and the data is end-of-day.")
            else:
                st.caption(ibkr_mod.REQUIREMENTS)


# ---------------------------------------------------------------- cas

def _cas(cfg, p) -> None:
    with st.container(border=True):
        st.markdown("#### Mutual funds (CAS)")
        conn = cas_mod.CasConnector(cfg)
        if conn.is_configured():
            result = conn.fetch()
            st.caption(f"Imported: {len(result.positions)} scheme(s). "
                       f"{result.note}.")
        else:
            st.caption("No statement imported yet.")
        st.caption("Request a **detailed CAS** from camsonline.com or "
                   "kfintech.com; it arrives by email as a PDF locked with "
                   "your PAN. The PDF and password are never saved — only "
                   "the closing balance of each scheme.")
        c1, c2 = st.columns([2, 1])
        upload = c1.file_uploader("CAS PDF", type=["pdf"], key="cas_pdf")
        password = c2.text_input("PDF password", type="password",
                                 key="cas_password")
        if upload is not None and password and st.button(
                "Import statement", key="cas_import"):
            try:
                out = cas_mod.import_pdf(upload.getvalue(), password)
            except Exception as e:
                st.error(f"Import failed: {e}")
                return
            st.session_state.pop("portfolio_snapshot", None)
            st.success(f"Imported {len(out['holdings'])} scheme(s), "
                       f"statement dated {out['statement_date']}.")


# ---------------------------------------------------------------- manual

def _manual(cfg, p) -> None:
    with st.container(border=True):
        st.markdown("#### Balances you record by hand")
        st.caption(
            "EPF, PPF, NPS, FDs, gold, cash and anything without an API. "
            "Give a single **value** in the line's own currency. "
            "`fixed_income` for EPF/PPF/FD/NPS-debt; `gold` for gold; `mf` "
            "for NPS equity; `cash` for savings and the emergency fund. "
            "Put the statement date in `as_of` (YYYY-MM-DD) so an old balance "
            "shows its age. Saved to a gitignored file.")
        path = at_root(cfg.portfolio.manual_path)
        df = _load_manual(path)
        edited = st.data_editor(
            df, key="manual_editor", num_rows="dynamic", width="stretch",
            hide_index=True,
            column_config={
                "asset_class": st.column_config.SelectboxColumn(
                    "asset_class", options=list(ASSET_CLASSES), required=True),
                "market": st.column_config.SelectboxColumn(
                    "market", options=["india", "us", "uk"], required=True),
                "value": st.column_config.NumberColumn("value", min_value=0.0),
            })
        if st.button("Save balances", key="manual_save"):
            _save_manual(path, edited)
            st.session_state.pop("portfolio_snapshot", None)
            st.success(f"Saved {len(edited.dropna(how='all'))} line(s).")


def _load_manual(path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path, comment="#", dtype=str).fillna("")
        except Exception:
            df = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[list(CSV_COLUMNS)]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _save_manual(path, df: pd.DataFrame) -> None:
    """Rewrite the file with the template header, one row per edited line."""
    df = df.dropna(how="all").copy()
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[list(CSV_COLUMNS)].fillna("")
    df["currency"] = [c or ("INR" if m in ("", "india") else "")
                      for c, m in zip(df["currency"], df["market"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_csv(index=False, lineterminator="\n")
    header_comment = "".join(line + "\n" for line in
                             manual_mod.TEMPLATE.splitlines()
                             if line.startswith("#"))
    path.write_text(header_comment + body, encoding="utf-8")


# ---------------------------------------------------------------- read

def _read(cfg, p) -> None:
    st.subheader("Read my holdings")
    if st.button("Read now", key="connect_read", type="primary"):
        get_snapshot(refresh=True)
        # The sidebar checklist rendered before this button ran; rerun so it
        # reflects what was just read rather than lagging one interaction.
        st.rerun()
    snapshot = st.session_state.get("portfolio_snapshot")
    if snapshot is None:
        st.caption("Not read yet this session.")
        return
    st.dataframe(pd.DataFrame([{
        "Account": r.source,
        "Answered": "yes" if r.available else "NO",
        "Holdings": len(r.positions),
        "Detail": r.note,
    } for r in snapshot.results]), width="stretch", hide_index=True)
    if snapshot.complete:
        banner(f"Every connected account answered — ₹{snapshot.total_inr:,.0f} "
               f"across {len(snapshot.positions)} holding(s). Next: "
               f"<b>{onboarding.PLAN}</b>.", p.good, "▣")
    else:
        banner("Not every account answered, so net worth and percentages are "
               "withheld until it does. " + " ".join(snapshot.caveats()),
               p.warning, "⚠")
