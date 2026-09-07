"""
Total-return index history from niftyindices.com, for the Stage 0 question.

WHY THIS EXISTS AT ALL. Every benchmark in this repo is a PRICE index - the
swing book's `^NSEI` from yfinance, the factor sleeve's Kite token 256265 - so
every recorded excess over "Nifty" throws the dividends away on one side only.
That is tolerable when the comparison is a strategy against a rough yardstick.
It is not tolerable when the whole question is whether one index beats another
by a point or two a year, because the Shariah screen excludes financials, and
financials are a large part of the parent index's payout. Comparing price
indices there would answer a question about dividend policy while looking like
an answer about compliance.

WHY NOT AN AGGREGATOR. Several show rebase artefacts on this index, and the
failure mode is the reason to care: a rebase leaves a complete, monotonic,
entirely plausible series with one bad month buried in it. niftyindices is the
administrator (NSE Indices Ltd), so it is the primary source, and the endpoint
and fetch date are written into the parquet so a later reader can re-check
rather than trust.

THE ENDPOINT. `POST /BackPage/getTotalReturnIndexString` on
www.niftyindices.com, with the query nested as a STRING inside a `cinfo` field:
`{"cinfo": "{'name':...,'startDate':...}"}`. That double encoding is the site's
own. The older `Backpage.aspx/...` form that circulates online is dead - it now
answers 200 with the site's HTML shell, which is exactly the "looks like data"
failure `swing/universe.py` already guards against, so `_rows` checks the SHAPE
and never the status code.

The site's own JavaScript refuses a range longer than a year. That limit is
client-side only - the server returns the full history in one request, which is
the default here - but `--chunk-years` keeps a correct chunked path in case
that stops being true.

WHAT COMES BACK. Two columns, and the difference between them is a tax:
  TotalReturnsIndex - gross TRI, dividends reinvested with nothing withheld.
  NTR_Value         - net TRI, dividends reinvested after a withholding rate.
Both are stored. Which to headline is the report's decision, not the fetcher's;
`factor/indices.py` makes it and says why.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

#: The administrator's own host. VERIFY: re-read if a fetch starts returning
#: HTML - the site was redesigned once already and took `Backpage.aspx` with it.
BASE = "https://www.niftyindices.com"
ENDPOINT = BASE + "/BackPage/getTotalReturnIndexString"
VERIFIED_ON = "2026-09-06"

#: Requested first so the session carries whatever cookie the endpoint wants.
#: Same reasoning as the Referer header in `swing/universe.py`.
WARMUP = BASE + "/indices/equity/broad-based-indices/nifty-50"

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json; charset=utf-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": BASE + "/reports/historical-data",
    "Origin": BASE,
}

#: Index names exactly as `assets/json/IndexMapping.json` spells them. The
#: endpoint uppercases `name` itself; `indexName` keeps the original case.
#: A misspelt name returns an EMPTY list with a 200 rather than an error,
#: which is why `fetch` treats zero rows as a failure and names the index.
INDICES: dict[str, str] = {
    "NIFTY 50": "Nifty 50",
    "NIFTY 500": "Nifty 500",
    "NIFTY50 SHARIAH": "Nifty50 Shariah",
    "NIFTY500 SHARIAH": "Nifty500 Shariah",
}

CACHE_NAME = "nifty_tri.parquet"


def _fmt(day: date) -> str:
    """niftyindices wants dd-Mmm-yyyy and rejects anything else silently."""
    return day.strftime("%d-%b-%Y")


def _decode(body: bytes, label: str) -> object:
    """
    Parsed JSON, or an exception that names the actual failure.

    THE DEAD ENDPOINT ANSWERS 200 WITH THE SITE'S HTML SHELL. Left to
    `json.loads` that surfaces as "Expecting value: line 1 column 1", which
    reads like a corrupt download rather than like an endpoint that moved -
    and this one has moved once already, from `Backpage.aspx/...`. The status
    code proves nothing here, so the body is the only real check.
    """
    text = body.decode("utf-8-sig", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        head = text.lstrip()[:80].replace("\n", " ")
        raise ValueError(
            label + ": the endpoint returned something that is not JSON - "
            "this usually means it has moved again and the site served its "
            "HTML shell with a 200. First bytes: " + repr(head)) from None


def _rows(payload: object, label: str) -> list:
    """
    The response rows, or an exception naming what arrived instead.

    Shape, never status: a 200 is not evidence of data on this host.
    """
    if isinstance(payload, str):
        payload = _decode(payload.encode("utf-8"), label)
    if not isinstance(payload, list):
        raise ValueError(
            label + ": expected a list, got " + type(payload).__name__
            + " - the endpoint has probably moved again")
    for row in payload:
        if not isinstance(row, dict) or "TotalReturnsIndex" not in row:
            raise ValueError(
                label + ": rows carry no TotalReturnsIndex column "
                + "(keys: " + repr(sorted(row)[:6] if isinstance(row, dict)
                                   else row) + ")")
        break
    return payload


def _session():
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": _HEADERS["User-Agent"]})
    try:
        session.get(WARMUP, timeout=30)
    except Exception:                                   # noqa: BLE001
        pass          # the cookie is a nicety; the endpoint works without it
    return session


def _request(session, name: str, label: str, start: date, end: date,
             timeout: int) -> list:
    query = ("{'name':'" + name + "','startDate':'" + _fmt(start)
             + "','endDate':'" + _fmt(end) + "','indexName':'" + label + "'}")
    resp = session.post(ENDPOINT, headers=_HEADERS,
                        data=json.dumps({"cinfo": query}), timeout=timeout)
    resp.raise_for_status()
    return _rows(_decode(resp.content, label), label)


def _spans(start: date, end: date, years: int):
    """Calendar-year-aligned windows, inclusive, that cover [start, end]."""
    lo = start
    while lo <= end:
        hi = min(end, date(lo.year + years, 1, 1) - pd.Timedelta(days=1).to_pytimedelta())
        yield lo, hi
        lo = hi + pd.Timedelta(days=1).to_pytimedelta()


def fetch(name: str, label: str, start: date, end: date, timeout: int = 120,
          chunk_years: int = 0, pause: float = 0.5) -> pd.DataFrame:
    """
    One index, tidy long. Raises rather than returning an empty frame.

    An empty result is indistinguishable from a quiet market only if it is
    allowed to pass, and the likeliest cause by far is a misspelt index name -
    which this endpoint answers with `[]` and a 200.
    """
    session = _session()
    rows: list = []
    if chunk_years <= 0:
        rows = _request(session, name, label, start, end, timeout)
    else:
        for lo, hi in _spans(start, end, chunk_years):
            rows += _request(session, name, label, lo, hi, timeout)
            print("    " + label + ": " + _fmt(lo) + ".." + _fmt(hi)
                  + " -> " + str(len(rows)) + " rows", flush=True)
            time.sleep(pause)
    if not rows:
        raise ValueError(
            label + ": no rows. The name is the usual cause - this endpoint "
            "answers an unknown index with an empty list and a 200.")

    frame = pd.DataFrame(rows)
    out = pd.DataFrame({
        "index_name": label,
        "date": pd.to_datetime(frame["Date"], format="%d %b %Y"),
        "tri": pd.to_numeric(frame["TotalReturnsIndex"], errors="coerce"),
        "ntr": pd.to_numeric(frame["NTR_Value"], errors="coerce")
        if "NTR_Value" in frame.columns else pd.NA,
    })
    return (out.dropna(subset=["tri"]).drop_duplicates(subset=["date"])
               .sort_values("date").reset_index(drop=True))


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch Nifty TRI history.")
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default="data/cache/" + CACHE_NAME)
    ap.add_argument("--chunk-years", type=int, default=0,
                    help="0 = one request for the whole range (works today)")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    print("niftyindices TRI  " + str(start) + " .. " + str(end)
          + "  (endpoint verified " + VERIFIED_ON + ")", flush=True)

    frames, failures = [], []
    for name, label in INDICES.items():
        try:
            got = fetch(name, label, start, end, timeout=args.timeout,
                        chunk_years=args.chunk_years)
            span = (str(got["date"].iloc[0].date()) + " .. "
                    + str(got["date"].iloc[-1].date()))
            print("  " + label.ljust(20) + str(len(got)).rjust(6)
                  + " sessions  " + span + "  NTR on "
                  + str(int(got["ntr"].notna().sum())), flush=True)
            frames.append(got)
        except Exception as exc:                        # noqa: BLE001
            failures.append(label + ": " + str(exc))
            print("  " + label.ljust(20) + " FAILED - " + str(exc), flush=True)

    if failures:
        # Same rule as `membership.refresh`: a partial download must not
        # overwrite a good file, because a half-written cache reads exactly
        # like a short history.
        print("\nNothing written - a partial TRI cache reads like a short "
              "history.\n  " + "\n  ".join(failures))
        return 1

    out = pd.concat(frames, ignore_index=True)
    out["fetched_at"] = pd.Timestamp(datetime.now())
    out["source"] = ENDPOINT
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print("\nwrote " + str(path) + "  (" + str(len(out)) + " rows, "
          + str(out["index_name"].nunique()) + " indices)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
