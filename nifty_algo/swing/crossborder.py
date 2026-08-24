"""
What it costs an Indian resident to hold a foreign position.

WHY THIS IS IN A TRADING REPO AT ALL. The scan ranks setups. For a domestic
trade that is nearly the whole story, because the frictions are small and
identical across candidates. For a foreign trade it is not: the same setup can
be a good idea through an Ireland-domiciled fund and a poor one held directly,
and nothing about the chart says so. The differences are structural, they are
arithmetic, and they are invisible on a broker statement - which is exactly
the combination that makes them worth computing rather than remembering.

THE ONE THAT MATTERS MOST IS NOT A COST, IT IS AN EXPOSURE. US-domiciled ETFs
and directly-held US shares are US-situs assets. For a non-resident alien the
US estate tax exemption is $60,000 - not the $13-odd million a US citizen
gets - and India has no estate tax treaty with the United States to soften it.
An Ireland-domiciled UCITS holding the identical companies is not US-situs and
is outside that regime entirely. Somebody holding SPUS and HLAL can cross that
line without a single document ever mentioning it, which is why the meter is
computed on every view rather than filed under "read later".

THIS MODULE IS ARITHMETIC AND CITATIONS. It is not tax advice, it does not
know your slab, your residency history or your treaty position, and every rate
in it is stamped with the date it was verified because all of them change.
`SOURCES` carries the reference for each number. Confirm with a CA before
filing anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: When the rates below were last checked against the cited source. Printed
#: next to every figure - a tax constant with no date on it is a liability.
VERIFIED_ON = "2026-08-21"

# --------------------------------------------------------------------------
# Liberalised Remittance Scheme, and the tax collected at source on it.
# --------------------------------------------------------------------------
LRS_ANNUAL_CAP_USD = 250_000.0
TCS_FREE_ALLOWANCE_INR = 1_000_000.0     # raised from Rs 7L in Budget 2025
TCS_RATE_INVESTMENT = 0.20               # on the EXCESS over the allowance

# --------------------------------------------------------------------------
# US estate tax, non-resident alien.
# --------------------------------------------------------------------------
US_ESTATE_EXEMPTION_USD = 60_000.0
US_ESTATE_TOP_RATE = 0.40

# --------------------------------------------------------------------------
# Dividend withholding reaching an Indian resident.
# --------------------------------------------------------------------------
US_DIVIDEND_WHT_DTAA = 0.25        # India-US treaty rate, creditable (Form 67)
IRISH_UCITS_FUND_LEVEL_WHT = 0.15  # US-Ireland treaty, suffered inside the fund
IRISH_WHT_TO_INDIAN_RESIDENT = 0.0

# --------------------------------------------------------------------------
# UK Stamp Duty Reserve Tax.
# --------------------------------------------------------------------------
UK_SDRT_RATE = 0.005               # UK-incorporated shares. LSE-listed Irish
                                   # UCITS ETFs are OUTSIDE this - not a
                                   # discount, a different security.

# --------------------------------------------------------------------------
# Indian capital gains on foreign equity.
# --------------------------------------------------------------------------
FOREIGN_LTCG_MONTHS = 24
FOREIGN_LTCG_RATE = 0.125          # without indexation, post 23 Jul 2024

SOURCES = {
    "tcs": ("TCS on foreign remittance, Rs 10 lakh threshold",
            "https://cleartax.in/s/tax-on-foreign-remittance"),
    "estate": ("Nonresident alien investors and Ireland-domiciled ETFs",
               "https://www.bogleheads.org/wiki/"
               "Nonresident_alien_investors_and_Ireland_domiciled_ETFs"),
    "domicile": ("US-domiciled vs Irish UCITS for non-US investors (SSGA)",
                 "https://www.ssga.com/us/en/institutional/insights/"
                 "considerations-for-non-us-investors-us-etfs-vs-irish-ucits"),
    "cgt": ("Foreign stocks and taxation (Zerodha Varsity)",
            "https://zerodha.com/varsity/chapter/foreign-stocks-and-taxation/"),
    "sdrt": ("Stamp duty and SDRT exemption for exchange traded funds (HMRC)",
             "https://assets.publishing.service.gov.uk/media/"
             "5a7c99c340f0b6629523a925/tiin-sd-sdrt.pdf"),
}

DISCLAIMER = (
    "Arithmetic, not advice. These are published rates applied to the numbers "
    f"you entered, verified on {VERIFIED_ON} against the sources listed. They "
    "do not know your tax slab, your residency history or your treaty "
    "position, and every one of them changes. Confirm with a chartered "
    "accountant before you file."
)


# --------------------------------------------------------------------------
# LRS / TCS
# --------------------------------------------------------------------------

@dataclass
class TcsResult:
    remitting_inr: float
    already_remitted_inr: float
    taxable_inr: float
    tcs_inr: float
    allowance_left_inr: float

    def note(self) -> str:
        if self.tcs_inr <= 0:
            return (f"No TCS: this remittance stays inside the "
                    f"₹{TCS_FREE_ALLOWANCE_INR:,.0f} annual allowance "
                    f"(₹{self.allowance_left_inr:,.0f} of it still unused).")
        return (
            f"TCS ₹{self.tcs_inr:,.0f} — {TCS_RATE_INVESTMENT:.0%} of the "
            f"₹{self.taxable_inr:,.0f} that sits above the "
            f"₹{TCS_FREE_ALLOWANCE_INR:,.0f} allowance. This is NOT a cost: it "
            f"is credited against your income tax and refunded if it exceeds "
            f"the liability. It is a cash-flow event — that money is out of "
            f"reach until you file."
        )


def tcs_on(remitting_inr: float, already_remitted_inr: float = 0.0) -> TcsResult:
    """
    TCS due on one remittance, given what the financial year has already used.

    The allowance is cumulative across every non-tour LRS remittance in the
    year and is not per bank or per transaction, so the running total is an
    input rather than something this function can infer.
    """
    remitting_inr = max(0.0, float(remitting_inr))
    already = max(0.0, float(already_remitted_inr))
    allowance_left = max(0.0, TCS_FREE_ALLOWANCE_INR - already)
    taxable = max(0.0, remitting_inr - allowance_left)
    return TcsResult(
        remitting_inr=remitting_inr, already_remitted_inr=already,
        taxable_inr=taxable, tcs_inr=taxable * TCS_RATE_INVESTMENT,
        allowance_left_inr=allowance_left,
    )


# --------------------------------------------------------------------------
# US estate tax exposure
# --------------------------------------------------------------------------

@dataclass
class EstateExposure:
    us_situs_usd: float
    non_us_situs_usd: float
    holdings: list = field(default_factory=list)   # (label, usd, is_us_situs)

    @property
    def over_exemption_usd(self) -> float:
        return max(0.0, self.us_situs_usd - US_ESTATE_EXEMPTION_USD)

    @property
    def headroom_usd(self) -> float:
        return max(0.0, US_ESTATE_EXEMPTION_USD - self.us_situs_usd)

    @property
    def breached(self) -> bool:
        return self.us_situs_usd > US_ESTATE_EXEMPTION_USD

    @property
    def indicative_tax_usd(self) -> float:
        """
        The excess at the top marginal rate.

        Deliberately the TOP rate rather than the graduated schedule: this is
        a warning light, not a return, and understating it would defeat the
        purpose. Labelled "indicative" everywhere it is shown.
        """
        return self.over_exemption_usd * US_ESTATE_TOP_RATE

    def note(self) -> str:
        if not self.breached:
            return (
                f"US-situs assets ${self.us_situs_usd:,.0f} — inside the "
                f"${US_ESTATE_EXEMPTION_USD:,.0f} non-resident exemption, with "
                f"${self.headroom_usd:,.0f} of headroom. "
                f"${self.non_us_situs_usd:,.0f} is held outside the US regime "
                f"and does not count toward this."
            )
        return (
            f"US-situs assets ${self.us_situs_usd:,.0f} — "
            f"${self.over_exemption_usd:,.0f} ABOVE the "
            f"${US_ESTATE_EXEMPTION_USD:,.0f} non-resident exemption. On death "
            f"that excess is exposed to US estate tax at up to "
            f"{US_ESTATE_TOP_RATE:.0%}, indicatively "
            f"${self.indicative_tax_usd:,.0f}, and India has no estate tax "
            f"treaty with the US to reduce it. An Ireland-domiciled UCITS "
            f"holding the same companies is not US-situs and sits outside this "
            f"entirely."
        )


def estate_exposure(holdings) -> EstateExposure:
    """
    Split holdings by situs and measure them against the $60,000 line.

    `holdings` is an iterable of objects with `.label`, `.value_usd` and
    `.us_situs`. Direct US shares and US-domiciled ETFs (SPUS, HLAL) are
    US-situs; Ireland-domiciled UCITS are not, and there is no look-through to
    the American companies a UCITS holds.
    """
    us = non_us = 0.0
    rows = []
    for h in holdings:
        value = float(getattr(h, "value_usd", 0.0) or 0.0)
        situs = bool(getattr(h, "us_situs", False))
        rows.append((getattr(h, "label", "?"), value, situs))
        if situs:
            us += value
        else:
            non_us += value
    return EstateExposure(us_situs_usd=us, non_us_situs_usd=non_us,
                          holdings=rows)


# --------------------------------------------------------------------------
# Frictions on one ticket
# --------------------------------------------------------------------------

@dataclass
class TicketCosts:
    market: str
    deployed_local: float
    currency: str
    sdrt: float = 0.0
    lines: list = field(default_factory=list)     # (label, amount, note)

    @property
    def total(self) -> float:
        return sum(amount for _, amount, _ in self.lines)

    @property
    def pct_of_deployed(self) -> float:
        return self.total / self.deployed_local if self.deployed_local else 0.0


def ticket_costs(market, deployed_local: float,
                 uk_incorporated: bool = True) -> TicketCosts:
    """
    Entry frictions on one position, in the market's own currency.

    Only the ones that are structural and knowable. Brokerage is deliberately
    absent: IBKR's tiers depend on monthly volume, and a guessed commission in
    a reward:risk calculation is worse than an acknowledged omission.

    `uk_incorporated` exists because SDRT follows the SECURITY, not the
    exchange. A UK-incorporated share bought on the LSE pays 0.5%; an
    Ireland-domiciled UCITS ETF listed on the very same exchange pays nothing.
    """
    costs = TicketCosts(market=market.key, deployed_local=deployed_local,
                        currency=market.currency)
    if market.key == "uk" and uk_incorporated:
        sdrt = deployed_local * UK_SDRT_RATE
        costs.sdrt = sdrt
        costs.lines.append((
            "UK Stamp Duty Reserve Tax", sdrt,
            f"{UK_SDRT_RATE:.1%} on purchases of UK-incorporated shares. It is "
            f"paid on the way in and is not recoverable, so it comes straight "
            f"off the reward side of the trade. LSE-listed Irish UCITS ETFs do "
            f"not pay it."))
    return costs


# --------------------------------------------------------------------------
# Holding-period and dividend treatment
# --------------------------------------------------------------------------

def cgt_note(months_held: float | None = None) -> str:
    """How India will tax the gain, and when the treatment changes."""
    base = (
        f"Indian capital gains on foreign equity: long-term after "
        f"{FOREIGN_LTCG_MONTHS} months at {FOREIGN_LTCG_RATE:.1%} without "
        f"indexation; short-term at your slab rate."
    )
    if months_held is None:
        return base + (
            " A swing held for days is short-term by a wide margin — the "
            "long-term rate is not reachable from this book and should not be "
            "assumed in any expectancy calculation.")
    if months_held >= FOREIGN_LTCG_MONTHS:
        return base + f" This position is {months_held:.0f} months old: long-term."
    remaining = FOREIGN_LTCG_MONTHS - months_held
    return base + (f" This position is {months_held:.0f} months old — "
                   f"{remaining:.0f} more months to long-term treatment.")


def withholding_note(domicile: str) -> str:
    """What reaches you from a dividend, by fund domicile."""
    if domicile.upper() in ("US", "USA"):
        return (
            f"US-domiciled: dividends are withheld at "
            f"{US_DIVIDEND_WHT_DTAA:.0%} under the India-US DTAA. That is "
            f"creditable against your Indian tax via Form 67, so it is a "
            f"timing and paperwork cost rather than a permanent one.")
    if domicile.upper() in ("IE", "IRELAND", "IRL"):
        return (
            f"Ireland-domiciled UCITS: the fund suffers "
            f"{IRISH_UCITS_FUND_LEVEL_WHT:.0%} on its US dividends under the "
            f"US-Ireland treaty, and Ireland withholds "
            f"{IRISH_WHT_TO_INDIAN_RESIDENT:.0%} on the way out to you. An "
            f"ACCUMULATING share class never distributes, so there is no "
            f"dividend event to tax in the year at all — the gain is taxed "
            f"when you sell.")
    return "Domicile not recognised - withholding treatment not computed."


SCHEDULE_FA_NOTE = (
    "Schedule FA is reported on a CALENDAR year, not the financial year: the "
    "return you file for FY2026-27 discloses foreign assets held between "
    "1 January and 31 December 2026. It is mandatory for a Resident and "
    "Ordinarily Resident holding any foreign brokerage account, and the "
    "penalty regime under the Black Money Act is severe and does not scale "
    "with the size of the omission. Disclose the account even in a year you "
    "did not trade it."
)
