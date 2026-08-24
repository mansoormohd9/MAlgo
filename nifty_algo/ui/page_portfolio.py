"""
Portfolio: the cross-border picture the pick cards cannot show.

WHY THIS PAGE EXISTS. Every other page in this console looks at one trade.
This one looks at the account, because three of the things that matter most to
an Indian resident investing abroad are properties of the WHOLE holding and
are invisible in any single ticket:

  - whether the US-situs total has crossed the $60,000 estate-tax line
  - how much of a "new" position you already own through the funds
  - what the year's remittances have done to the TCS allowance

Like the rest of `ui/`, this page decides nothing. `swing/crossborder.py` does
the arithmetic and `swing/holdings.py` reads the funds' books; this lays them
out and states the caveats.

THE INPUTS ARE TYPED IN, NOT FETCHED. There is no broker connection in this
build - the decision was screener-only, manual orders - so balances come from
you. They persist in session state for the life of the app run and are never
written to disk, because a file of account balances is not something this repo
should be creating without being asked.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .components import banner
from .state import get_config
from .theme import get_palette
from ..swing import crossborder as crossborder_mod
from ..swing import fx as fx_mod
from ..swing import holdings as holdings_mod

HOLDINGS_CSV = "data/etf_holdings.csv"

#: The funds this user actually holds, and the fact about each that the estate
#: meter turns on. Domicile is the whole question: a UCITS holding the very
#: same American companies is not a US-situs asset, and there is no
#: look-through to what it owns.
FUNDS = (
    ("SPUS", "SP Funds S&P 500 Sharia (SPUS)", "US", True,
     "US-domiciled. AAOIFI / S&P screening."),
    ("HLAL", "Wahed FTSE USA Shariah (HLAL)", "US", True,
     "US-domiciled. FTSE / Yasaar screening — the same standard this "
     "screener uses as its primary."),
    ("ISDW", "iShares MSCI World Islamic UCITS", "IE", False,
     "Ireland-domiciled UCITS. NOT a US-situs asset, so it sits outside the "
     "US estate tax regime entirely."),
)


class _Holding:
    """The shape `crossborder.estate_exposure` reads."""

    def __init__(self, label, value_usd, us_situs):
        self.label = label
        self.value_usd = value_usd
        self.us_situs = us_situs


def render() -> None:
    p = get_palette()
    cfg = get_config()

    st.title("Portfolio")
    st.caption("Cross-border exposure, costs and reporting for an Indian "
               "resident investing through IBKR under LRS.")

    banner(f"<b>Arithmetic, not advice.</b> {crossborder_mod.DISCLAIMER}",
           p.warning, "⚖")

    rate = _usd_rate(cfg, p)
    _capital_pool(cfg, rate, p)
    values = _inputs(p)

    _estate_meter(values, p)
    _domicile_comparison(p)
    _overlap_table(values, p)
    _remittance(values, rate, p)
    _reporting(p)


# ---------------------------------------------------------------- the pool

def _capital_pool(cfg, rate, p) -> None:
    """
    Fund the foreign pool.

    This lives here rather than in Settings because it is the same act as
    telling the page what you hold abroad. Until it is set, the US and UK
    scans stand down by design - sizing a dollar trade off the domestic
    balance would claim money that is not in that broker, and defaulting to
    "use the Indian capital" would be exactly that mistake made silently.
    """
    st.subheader("Foreign capital pool")
    c1, c2 = st.columns([1, 2])
    current = float(cfg.capital.foreign_capital_inr or 0.0)
    entered = c1.number_input(
        "Remitted and available abroad (₹)", min_value=0.0, step=50_000.0,
        value=current, key="foreign_capital_inr_input",
        help="What is actually in the IBKR account, in rupees. The US and UK "
             "scans size off this pool; the Indian book is unaffected.")
    cfg.capital.foreign_capital_inr = entered

    per_trade = cfg.capital.risk_inr("foreign")
    if entered <= 0:
        c2.warning(
            "The foreign pool is ₹0, so the **US and UK scans will stand "
            "down**. Set it here to enable them — the Indian book is "
            "unaffected either way.")
        return

    line = (f"Per-trade risk on the foreign book: **₹{per_trade:,.0f}** "
            f"({cfg.capital.risk_per_trade_pct:.2%} of the pool, the same "
            f"governor the Indian book uses).")
    if rate is not None:
        line += f" That is about ${per_trade / rate.inr_per_unit:,.0f}."
    c2.info(line)


# ---------------------------------------------------------------- inputs

def _inputs(p) -> dict:
    st.subheader("What you hold")
    st.caption("Entered by hand — this build has no broker connection. Values "
               "live for this app run only and are never written to disk.")

    values: dict[str, float] = {}
    cols = st.columns(len(FUNDS) + 1)
    for col, (key, label, _domicile, _situs, _note) in zip(cols, FUNDS):
        values[key] = col.number_input(
            f"{key} (USD)", min_value=0.0, step=500.0, value=0.0,
            key=f"fund_balance_usd_{key}", help=label)
        # The pick-card overlap check reads rupee balances from session state
        # under this key, so the two panels cannot disagree about what you own.
        st.session_state[f"fund_balance_{key}"] = values[key]

    values["DIRECT_US"] = cols[-1].number_input(
        "Direct US shares (USD)", min_value=0.0, step=500.0, value=0.0,
        key="direct_us_usd",
        help="Individual US stocks held at IBKR. These are US-situs assets "
             "exactly as a US-domiciled ETF is.")
    return values


def _usd_rate(cfg, p):
    try:
        return fx_mod.rate_inr_per("USD", cfg)
    except fx_mod.FxUnavailable as e:
        banner(f"<b>No USD rate.</b> {e} Figures below stay in dollars.",
               p.warning, "⚠")
        return None


# ---------------------------------------------------------------- estate

def _estate_meter(values, p) -> None:
    st.subheader("US estate tax exposure")

    holdings = [
        _Holding(label, values.get(key, 0.0), situs)
        for key, label, _dom, situs, _note in FUNDS
    ]
    holdings.append(_Holding("Direct US shares",
                             values.get("DIRECT_US", 0.0), True))

    exposure = crossborder_mod.estate_exposure(holdings)

    c1, c2, c3 = st.columns(3)
    c1.metric("US-situs assets", f"${exposure.us_situs_usd:,.0f}")
    c2.metric("Exemption", f"${crossborder_mod.US_ESTATE_EXEMPTION_USD:,.0f}")
    c3.metric("Headroom" if not exposure.breached else "Over the line",
              f"${exposure.headroom_usd:,.0f}" if not exposure.breached
              else f"${exposure.over_exemption_usd:,.0f}",
              delta=None)

    fraction = min(1.0, exposure.us_situs_usd /
                   crossborder_mod.US_ESTATE_EXEMPTION_USD) \
        if crossborder_mod.US_ESTATE_EXEMPTION_USD else 0.0
    st.progress(fraction)

    banner(exposure.note(),
           p.critical if exposure.breached else p.good,
           "⛔" if exposure.breached else "▣")

    if exposure.breached:
        st.markdown(
            "**What actually reduces this.** Holding the equivalent "
            "Ireland-domiciled UCITS instead of the US-domiciled fund — the "
            "iShares MSCI World Islamic UCITS you already hold is the model. "
            "The underlying companies are the same American ones; the "
            "*wrapper* is what the US taxes, and there is no look-through. "
            "Switching realises Indian capital gains, so this is a decision "
            "with a cost, not a free move — but it is a decision, and it "
            "cannot be made if the number is never computed."
        )


# ---------------------------------------------------------------- domicile

def _domicile_comparison(p) -> None:
    with st.expander("US-domiciled vs Ireland-domiciled, side by side"):
        st.dataframe(pd.DataFrame([
            {"": "US estate tax",
             "US-domiciled (SPUS, HLAL)":
                 f"US-situs. Exempt only to "
                 f"${crossborder_mod.US_ESTATE_EXEMPTION_USD:,.0f}, then up to "
                 f"{crossborder_mod.US_ESTATE_TOP_RATE:.0%}. No India-US "
                 f"estate treaty.",
             "Ireland UCITS (ISDW)": "Not US-situs. Outside the regime."},
            {"": "Dividend withholding",
             "US-domiciled (SPUS, HLAL)":
                 f"{crossborder_mod.US_DIVIDEND_WHT_DTAA:.0%} under the "
                 f"India-US DTAA, creditable via Form 67.",
             "Ireland UCITS (ISDW)":
                 f"{crossborder_mod.IRISH_UCITS_FUND_LEVEL_WHT:.0%} suffered "
                 f"inside the fund; "
                 f"{crossborder_mod.IRISH_WHT_TO_INDIAN_RESIDENT:.0%} Irish "
                 f"WHT to you."},
            {"": "Distributions",
             "US-domiciled (SPUS, HLAL)":
                 "Distributing. A taxable event every year whether you wanted "
                 "the cash or not.",
             "Ireland UCITS (ISDW)":
                 "An accumulating class never distributes — no annual event, "
                 "gain taxed on sale."},
            {"": "Indian capital gains",
             "US-domiciled (SPUS, HLAL)":
                 f"Same for both: {crossborder_mod.FOREIGN_LTCG_RATE:.1%} after "
                 f"{crossborder_mod.FOREIGN_LTCG_MONTHS} months, slab before.",
             "Ireland UCITS (ISDW)": "Same."},
            {"": "Cost and liquidity",
             "US-domiciled (SPUS, HLAL)":
                 "Deeper US liquidity, tighter spreads, lower expense ratios.",
             "Ireland UCITS (ISDW)":
                 "Thinner LSE liquidity, usually a higher expense ratio. This "
                 "is the real trade-off — it is not free."},
        ]), width="stretch", hide_index=True)
        st.caption(
            "The estate-tax row is the one that is structural and "
            "unrecoverable. The withholding rows are timing and paperwork. "
            "Weigh them differently."
        )


# ---------------------------------------------------------------- overlap

def _overlap_table(values, p) -> None:
    book = holdings_mod.load_holdings(HOLDINGS_CSV)
    if not book:
        st.caption("No ETF holdings committed yet — run the **Refresh ETF "
                   "holdings** button on the Daily picks page to enable the "
                   "overlap view.")
        return

    with st.expander(f"What your funds actually own "
                     f"(as of {holdings_mod.as_of(book) or 'unknown'})"):
        balances = {k: values.get(k, 0.0) for k, *_ in FUNDS}
        rows = []
        for symbol, entries in book.items():
            exposure = sum(h.weight * balances.get(h.etf, 0.0) for h in entries)
            rows.append({
                "Symbol": symbol,
                "Company": entries[0].name,
                "Funds": ", ".join(sorted(h.etf for h in entries)),
                "Max weight": f"{max(h.weight for h in entries):.2%}",
                "Your exposure (USD)": round(exposure, 2),
            })
        frame = pd.DataFrame(rows).sort_values(
            "Your exposure (USD)", ascending=False)
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(
            "A direct position in any of these ADDS to what you already hold. "
            "That is not a reason to avoid them — the funds hold them because "
            "they are the compliant large caps — but it is a reason not to "
            "call it diversification."
        )


# ---------------------------------------------------------------- remittance

def _remittance(values, rate, p) -> None:
    st.subheader("Remittance and TCS")
    c1, c2 = st.columns(2)
    already = c1.number_input(
        "Already remitted this financial year (₹)", min_value=0.0,
        step=50_000.0, value=0.0, key="lrs_used_inr",
        help="Cumulative across every non-tour LRS remittance, all banks "
             "combined. The allowance is per person per year, not per "
             "transaction.")
    planned = c2.number_input(
        "Planning to remit now (₹)", min_value=0.0, step=50_000.0,
        value=0.0, key="lrs_planned_inr")

    tcs = crossborder_mod.tcs_on(planned, already)
    banner(tcs.note(), p.good if tcs.tcs_inr <= 0 else p.warning, "▣")

    if rate is not None:
        used_usd = (already + planned) / rate.inr_per_unit
        st.caption(
            f"That is about ${used_usd:,.0f} of the "
            f"${crossborder_mod.LRS_ANNUAL_CAP_USD:,.0f} annual LRS cap, at "
            f"{rate.note()}.")


# ---------------------------------------------------------------- reporting

def _reporting(p) -> None:
    with st.expander("Reporting and capital gains"):
        st.markdown(f"**Schedule FA.** {crossborder_mod.SCHEDULE_FA_NOTE}")
        st.markdown(f"**Capital gains.** {crossborder_mod.cgt_note()}")
        st.markdown(
            "**On this book specifically.** These are swing trades held for "
            "days. Every one of them is short-term by a wide margin and is "
            "taxed at your slab rate, so the 12.5% long-term figure should "
            "never appear in an expectancy calculation for this strategy. "
            "Your ETF holdings are the part of the portfolio where the "
            "24-month clock is worth watching."
        )
        st.caption("Sources: " + " · ".join(
            f"[{label}]({url})"
            for label, url in crossborder_mod.SOURCES.values()))
