"""
Headlines for the finalists, scored by lexicon.

WHY ONLY THE FINALISTS: this is one HTTP request per company. A hundred of
them takes minutes and gets rate-limited, and news cannot rescue a stock with
no setup anyway - it can only rank or veto ones that already have one. So the
scanner narrows on price first and asks about news last, for the fifteen names
that are still standing.

WHY GOOGLE NEWS RSS: it needs no key, no account and no quota negotiation, it
covers the Indian financial press broadly, and it returns a publication
timestamp, which the recency decay needs. It is an unofficial endpoint and it
can throttle or change; every failure path here degrades to "news unavailable"
rather than to a silent zero.

A ZERO SCORE AND A MISSING SCORE ARE DIFFERENT FACTS. "Nothing notable was
published" is information. "We could not reach the feed" is an absence of
information. Conflating them would let a network failure read as a clean bill
of health on every stock at once, which is the most dangerous possible way for
this module to fail. `available` carries the distinction and the ranking drops
the news component entirely when it is False.
"""
from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .news_lexicon import score_headline, veto_reason

GOOGLE_NEWS = "https://news.google.com/rss/search"

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/rss+xml,application/xml,text/xml,*/*",
}

#: Yahoo and Google both throttle bursts; a short gap between companies is
#: cheaper than the empty responses that follow a rate-limit.
_PAUSE_SECONDS = 0.5


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: datetime | None
    score: float
    matches: list[str] = field(default_factory=list)
    veto: str | None = None

    @property
    def age_hours(self) -> float | None:
        if self.published is None:
            return None
        now = datetime.now(timezone.utc)
        return max(0.0, (now - self.published).total_seconds() / 3600.0)


@dataclass
class NewsResult:
    symbol: str
    available: bool = False
    score: float = 0.0                 # -1..+1, recency-weighted
    items: list[NewsItem] = field(default_factory=list)
    veto: str | None = None            # set -> the candidate is dropped
    veto_headline: str | None = None
    note: str = "not fetched"

    @property
    def normalised(self) -> float:
        """Score mapped to 0..1 for the composite, where 0.5 is neutral."""
        return (self.score + 1.0) / 2.0

    def summary(self) -> str:
        if not self.available:
            return f"news unavailable - {self.note}"
        if not self.items:
            return "no headlines in the window - nothing being said either way"
        direction = ("tailwind" if self.score > 0.15 else
                     "headwind" if self.score < -0.15 else "neutral")
        return (f"{len(self.items)} headlines, score {self.score:+.2f} "
                f"({direction})")


def fetch_for(stocks, cfg, progress=None) -> dict[str, NewsResult]:
    """
    Headlines for each stock in `stocks` (an iterable of `universe.Stock`).

    Never raises. Every stock gets a NewsResult; an unreachable feed produces
    `available=False` with the reason attached.
    """
    stocks = list(stocks)
    out: dict[str, NewsResult] = {}

    try:
        import feedparser            # noqa: F401
        import requests              # noqa: F401
    except ImportError as e:
        note = (f"{e.name} is not installed - pip install feedparser requests. "
                f"News is excluded from the ranking.")
        return {s.symbol: NewsResult(s.symbol, note=note) for s in stocks}

    for i, stock in enumerate(stocks):
        if progress:
            progress(i, len(stocks), f"news: {stock.symbol}")
        out[stock.symbol] = fetch_one(stock, cfg)
        if i < len(stocks) - 1:
            time.sleep(_PAUSE_SECONDS)

    if progress and stocks:
        progress(len(stocks), len(stocks), "news complete")
    return out


def fetch_one(stock, cfg) -> NewsResult:
    """Headlines for one company. Never raises."""
    lookback = cfg.swing.news_lookback_hours
    days = max(1, round(lookback / 24))
    # The market-context clause is doing real work. A bare name query for
    # "Bosch" returns three days of global Bosch AG coverage - dishwashers,
    # German factory news, automotive supply deals - and scores the Indian
    # listing on all of it. Indian market reporting essentially always uses
    # one of these words, so requiring one cuts the foreign and consumer
    # coverage without needing to know which stories were which.
    query = (f'"{stock.search_name}" (share OR shares OR stock OR NSE) '
             f'when:{days}d')
    url = f"{GOOGLE_NEWS}?" + urllib.parse.urlencode({
        "q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en",
    })

    try:
        import feedparser
        import requests
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as e:
        return NewsResult(stock.symbol, note=f"feed error: {e}")

    entries = getattr(parsed, "entries", None)
    if entries is None:
        return NewsResult(stock.symbol, note="feed returned nothing parseable")

    items: list[NewsItem] = []
    for entry in entries:
        item = _to_item(entry, stock, lookback)
        if item is not None:
            items.append(item)

    # Newest first, then capped. A heavily-covered name can return dozens of
    # headlines in three days while a quiet one returns two, and without a cap
    # the aggregate would partly measure press attention rather than what was
    # said. The cap is applied AFTER the veto scan below, so a fraud story
    # buried at position forty still kills the trade.
    found = len(items)
    items.sort(key=lambda i: i.published or datetime.min.replace(
        tzinfo=timezone.utc), reverse=True)

    vetoed = next((i for i in items if i.veto), None)

    capped = items[:max(1, cfg.swing.news_max_items)]
    note = f"{found} relevant of {len(entries)} headlines"
    if found > len(capped):
        note += f", newest {len(capped)} scored"

    result = NewsResult(stock.symbol, available=True, items=capped, note=note)

    if vetoed is not None:
        result.veto = vetoed.veto
        result.veto_headline = vetoed.title
        if vetoed not in capped:
            # It has to stay visible, or the card shows a veto with no
            # headline behind it and there is nothing to disagree with.
            result.items = [vetoed] + capped

    result.score = _aggregate(capped, lookback)
    return result


# ---------------- internals ----------------

def _to_item(entry, stock, lookback_hours: int) -> NewsItem | None:
    title = (getattr(entry, "title", "") or "").strip()
    if not title:
        return None
    veto = veto_reason(title)
    score, matches = score_headline(title)

    # A veto headline is never filtered out. These rules are heuristics, and a
    # heuristic that can silently discard "auditor resigns" is worse than one
    # that occasionally lets a foreign story through.
    if veto is None:
        if not _is_relevant(title, stock):
            return None
        # A headline that neither scores nor reads as market coverage cannot
        # affect the ranking, and keeping it only adds noise to the card. This
        # is what removes the Netflix detective series from Bosch's feed
        # without also removing "Cipla gets USFDA approval", which scores.
        if not matches and not _is_market_news(title):
            return None

    published = _published(entry)
    if published is not None:
        age = (datetime.now(timezone.utc) - published).total_seconds() / 3600.0
        if age > lookback_hours:
            return None
    return NewsItem(
        title=title,
        link=(getattr(entry, "link", "") or ""),
        source=_source(entry),
        published=published,
        score=score,
        matches=matches,
        veto=veto_reason(title),
    )


def _is_relevant(title: str, stock) -> bool:
    """
    Whether this headline is plausibly about this company.

    A quoted-phrase query still returns sector round-ups and "shares to watch"
    lists. Requiring the symbol or a distinctive word from the company name in
    the headline itself throws away the worst of it. Short words are skipped -
    matching on "the" or "india" would let everything through.
    """
    text = title.lower()
    if stock.symbol.lower() in text:
        return True
    generic = {"limited", "ltd", "india", "indian", "company", "corporation",
               "industries", "the", "and", "products", "services", "motors",
               "bank", "finance", "technologies", "enterprises", "power"}
    tokens = [t for t in stock.search_name.lower().replace("(", " ")
              .replace(")", " ").replace("&", " ").split()
              if len(t) > 3 and t not in generic]
    if not tokens:
        # Nothing distinctive left (e.g. "ITC", "GAIL"). Fall back to the full
        # name, which is what the query asked for anyway.
        return stock.search_name.lower() in text
    return any(t in text for t in tokens)


#: Words that mark a headline as coverage of a traded instrument rather than
#: general news about a brand. See `_is_market_news`.
_MARKET_MARKERS = (
    "share", "stock", "nse", "bse", "sensex", "nifty", "equity", "equities",
    "rs ", "rs.", "₹", "crore", "cr ", "profit", "revenue", "results",
    "quarter", "q1", "q2", "q3", "q4", "dividend", "target", "brokerage",
    "rating", "ipo", "order", "contract", "deal", "52-week", "market cap",
    "investor", "analyst", "valuation", "earnings",
)


def _is_market_news(title: str) -> bool:
    return any(marker in title.lower() for marker in _MARKET_MARKERS)


def _published(entry) -> datetime | None:
    parsed = getattr(entry, "published_parsed", None) or \
        getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _source(entry) -> str:
    src = getattr(entry, "source", None)
    if src is not None:
        title = getattr(src, "title", None) or (
            src.get("title") if isinstance(src, dict) else None)
        if title:
            return str(title)
    # Google appends " - Publication" to its titles when there is no source tag.
    title = getattr(entry, "title", "") or ""
    return title.rsplit(" - ", 1)[-1] if " - " in title else "unknown"


def _aggregate(items: list[NewsItem], lookback_hours: int) -> float:
    """
    Recency-weighted mean of the scoring headlines.

    Headlines that matched no phrase are excluded from the mean rather than
    counted as zeros: twenty routine "shares gain 2%" items should not dilute
    one genuine upgrade down to nothing. They still appear on the card - you
    can see everything that was read, only not everything votes.
    """
    scoring = [i for i in items if i.matches]
    if not scoring:
        return 0.0

    total = 0.0
    weight_sum = 0.0
    for item in scoring:
        age = item.age_hours
        if age is None:
            weight = 0.5              # undated: counted, but at half voice
        else:
            weight = max(0.2, 1.0 - age / max(lookback_hours, 1))
        total += item.score * weight
        weight_sum += weight

    if weight_sum <= 0:
        return 0.0
    return max(-1.0, min(1.0, total / weight_sum))


def unavailable(symbols, note: str) -> dict[str, NewsResult]:
    """Every symbol marked as having no news read. Used when news is skipped."""
    return {s: NewsResult(s, note=note) for s in symbols}
