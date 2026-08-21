"""
The news read.

Two properties carry the weight here:

  1. A veto really does kill a candidate, regardless of the chart.
  2. "Nothing was published" and "we could not reach the feed" never collapse
     into the same value. Conflating them would let one network failure read
     as a clean bill of health on every stock at once.

The rest is lexicon arithmetic and the relevance filter that keeps sector
round-ups out of a single company's score.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nifty_algo.config import Config
from nifty_algo.swing import news
from nifty_algo.swing.news_lexicon import score_headline, veto_reason
from nifty_algo.swing.universe import Stock


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def acme() -> Stock:
    return Stock("ACME", "Acme Widgets Limited", "Capital Goods",
                 "Industrial Products", "ACME.NS")


def item(title, hours_ago=1.0, link="http://x", source="Test Wire") -> news.NewsItem:
    score, matches = score_headline(title)
    return news.NewsItem(
        title=title, link=link, source=source,
        published=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        score=score, matches=matches, veto=veto_reason(title))


# ---------------------------------------------------------------- lexicon

@pytest.mark.parametrize("headline", [
    "Jefferies raises target on Acme to Rs 1,650",
    "Acme Q1 profit jumps 24%, beats estimates",
    "Acme bags order worth Rs 4,200 crore from NHAI",
    "Acme gets approval for its new facility",
])
def test_bullish_headlines_score_positive(headline):
    score, matches = score_headline(headline)
    assert score > 0 and matches


@pytest.mark.parametrize("headline", [
    "Morgan Stanley cuts target on Acme after weak quarter",
    "Acme Q2 profit falls 18%, misses estimates",
    "Acme downgrade: brokerage sees limited upside",
    "USFDA issues Form 483 observations for Acme plant",
])
def test_bearish_headlines_score_negative(headline):
    score, matches = score_headline(headline)
    assert score < 0 and matches


def test_neutral_headline_scores_zero_and_matches_nothing():
    score, matches = score_headline("Acme shares in focus on Tuesday")
    assert score == 0.0 and matches == []


def test_score_is_capped_so_one_stuffed_headline_cannot_outvote_five_stories():
    stuffed = ("Acme upgrade: brokerage raises target, profit jumps, "
               "bags order, gets approval, announces buyback")
    score, _ = score_headline(stuffed)
    assert score == 1.0


@pytest.mark.parametrize("headline, expected", [
    ("SEBI probe into Acme promoter dealings", "SEBI investigation"),
    ("Acme auditor resigns citing lack of information", "auditor resignation"),
    ("Acme defaults on Rs 900 crore debt repayment", "debt default"),
    ("Forensic audit ordered into Acme accounts", "forensic audit ordered"),
    ("Lender invokes pledged shares of Acme promoter", "promoter pledge invoked"),
])
def test_vetoes_are_detected(headline, expected):
    assert veto_reason(headline) == expected


def test_ordinary_bad_news_is_not_a_veto():
    """A downgrade is a headwind, not a disqualification."""
    assert veto_reason("Acme downgraded to sell on valuation") is None


# ---------------------------------------------------------------- aggregate

def test_headlines_that_matched_nothing_do_not_dilute_the_score():
    """
    Twenty routine "shares gain 2%" items must not wash out one real upgrade.
    They still appear on the card; they simply do not vote.
    """
    real = [item("Brokerage raises target on Acme")]
    padded = real + [item(f"Acme shares in focus, session {i}") for i in range(20)]

    assert news._aggregate(real, 72) == pytest.approx(
        news._aggregate(padded, 72))


def test_recent_news_outweighs_stale_news():
    fresh = news._aggregate([item("Acme bags order", hours_ago=1),
                             item("Acme profit falls", hours_ago=70)], 72)
    stale = news._aggregate([item("Acme bags order", hours_ago=70),
                             item("Acme profit falls", hours_ago=1)], 72)
    assert fresh > 0 > stale


def test_no_scoring_headlines_aggregates_to_zero():
    assert news._aggregate([], 72) == 0.0
    assert news._aggregate([item("Acme shares in focus")], 72) == 0.0


def test_aggregate_stays_inside_the_range():
    many = [item("Acme upgrade, raises target, profit jumps") for _ in range(10)]
    assert -1.0 <= news._aggregate(many, 72) <= 1.0


# ---------------------------------------------------------------- relevance

def test_sector_roundups_are_filtered_out(acme):
    assert news._is_relevant("Acme Widgets bags Rs 400 crore order", acme)
    assert news._is_relevant("ACME hits 52-week high", acme)
    assert not news._is_relevant("Top 5 capital goods stocks to watch today", acme)
    assert not news._is_relevant("Sensex ends 400 points higher", acme)


def test_generic_words_alone_do_not_make_a_headline_relevant():
    """Matching on "India" or "Bank" would let the entire market through."""
    stock = Stock("XBANK", "Example Bank of India", "Financial Services",
                  "Private Sector Bank", "XBANK.NS")
    assert not news._is_relevant("Indian banks rally on rate cut hopes", stock)
    assert news._is_relevant("Example Bank of India posts record profit", stock)


# ---------------------------------------------------------------- availability

def test_unavailable_is_not_neutral():
    """
    The distinction the whole module is built around: a missing score is
    excluded from the ranking, a neutral score is counted at 0.5.
    """
    missing = news.NewsResult("ACME", available=False)
    neutral = news.NewsResult("ACME", available=True, score=0.0)
    assert missing.available is False
    assert neutral.normalised == 0.5
    assert "unavailable" in missing.summary()


def test_unavailable_helper_marks_every_symbol():
    out = news.unavailable(["A", "B"], "skipped")
    assert set(out) == {"A", "B"}
    assert all(r.available is False for r in out.values())


def test_a_dead_network_never_raises_and_never_scores(cfg, acme, monkeypatch):
    def explode(*a, **kw):
        raise OSError("network is down")

    import requests
    monkeypatch.setattr(requests, "get", explode)

    result = news.fetch_one(acme, cfg)
    assert result.available is False
    assert result.score == 0.0
    assert "feed error" in result.note


def test_fetch_for_returns_a_result_for_every_stock_even_when_all_fail(
        cfg, acme, monkeypatch):
    def explode(*a, **kw):
        raise OSError("nope")

    import requests
    monkeypatch.setattr(requests, "get", explode)

    other = Stock("BETA", "Beta Corp", "Chemicals", "Specialty Chemicals",
                  "BETA.NS")
    out = news.fetch_for([acme, other], cfg)
    assert set(out) == {"ACME", "BETA"}
    assert all(r.available is False for r in out.values())


# ---------------------------------------------------------------- parsing

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Acme Widgets bags Rs 4,200 crore order from NHAI</title>
  <link>http://example.test/1</link>
  <pubDate>{recent}</pubDate>
  <source url="http://wire.test">Test Wire</source>
</item>
<item>
  <title>Top 5 capital goods stocks to watch today</title>
  <link>http://example.test/2</link>
  <pubDate>{recent}</pubDate>
</item>
<item>
  <title>SEBI probe into Acme Widgets promoter dealings</title>
  <link>http://example.test/3</link>
  <pubDate>{recent}</pubDate>
</item>
</channel></rss>"""


class _Resp:
    def __init__(self, body: str):
        self.content = body.encode("utf-8")

    def raise_for_status(self):
        return None


def _serve(monkeypatch, body: str):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp(body))


def test_a_real_feed_is_parsed_scored_and_filtered(cfg, acme, monkeypatch):
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT")
    _serve(monkeypatch, RSS.format(recent=recent))

    result = news.fetch_one(acme, cfg)
    assert result.available is True

    titles = [i.title for i in result.items]
    assert any("bags Rs 4,200 crore" in t for t in titles)
    assert not any("Top 5 capital goods" in t for t in titles), \
        "the sector round-up should have been filtered out"

    assert result.veto == "SEBI investigation"
    assert "Acme" in (result.veto_headline or "")


def test_headlines_older_than_the_window_are_dropped(cfg, acme, monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT")
    _serve(monkeypatch, RSS.format(recent=old))

    result = news.fetch_one(acme, cfg)
    assert result.available is True
    assert result.items == []
    assert result.veto is None, "a month-old story must not veto today's trade"


# ---------------------------------------------------------------- the cap

def _many_rss(n: int, recent: str, veto_at: int | None = None) -> str:
    items = []
    for i in range(n):
        title = (f"SEBI probe into Acme Widgets dealings" if i == veto_at
                 else f"Acme Widgets raises target, item {i}")
        items.append(
            f"<item><title>{title}</title>"
            f"<link>http://example.test/{i}</link>"
            f"<pubDate>{recent}</pubDate></item>")
    return ('<?xml version="1.0"?><rss version="2.0"><channel>'
            + "".join(items) + "</channel></rss>")


def test_a_widely_covered_name_is_capped_to_the_newest_items(
        cfg, acme, monkeypatch):
    """
    Without a cap the aggregate partly measures press attention rather than
    what was said - a household name would out-score a quiet one on volume.
    """
    cfg.swing.news_max_items = 5
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT")
    _serve(monkeypatch, _many_rss(40, recent))

    result = news.fetch_one(acme, cfg)
    assert len(result.items) == 5
    assert "newest 5 scored" in result.note


def test_a_veto_buried_below_the_cap_still_fires_and_stays_visible(
        cfg, acme, monkeypatch):
    """
    The cap is an attention limit, not a licence to miss a fraud story that
    happened to be the fortieth result.
    """
    cfg.swing.news_max_items = 3
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT")
    _serve(monkeypatch, _many_rss(40, recent, veto_at=39))

    result = news.fetch_one(acme, cfg)
    assert result.veto == "SEBI investigation"
    assert any(i.veto for i in result.items), \
        "the headline behind the veto must still be shown"


# ---------------------------------------------------------------- ambiguity

def one_word(symbol="BOSCHLTD", name="Bosch") -> Stock:
    return Stock(symbol, name, "Automobile and Auto Components",
                 "Auto Components", f"{symbol}.NS")


def test_general_brand_news_is_dropped_but_scoring_news_is_kept(cfg,
                                                               monkeypatch):
    """
    "Bosch" returns the German parent, an unrelated Indian listing called
    Bosch Home Comfort, and a Netflix detective series. A headline that
    neither scores nor reads as market coverage cannot affect the ranking, so
    it is dropped - while anything that scores is kept regardless.
    """
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT")
    titles = [
        "Netflix is pulling the plug on the strategy for Bosch fans",
        "Bosch shares hit a new 52-week high",
        "Bosch bags order worth Rs 900 crore",
    ]
    items = "".join(
        f"<item><title>{t}</title><link>http://x/{i}</link>"
        f"<pubDate>{recent}</pubDate></item>" for i, t in enumerate(titles))
    _serve(monkeypatch, '<?xml version="1.0"?><rss version="2.0"><channel>'
           + items + "</channel></rss>")

    kept = [i.title for i in news.fetch_one(one_word(), cfg).items]
    assert not any("Netflix" in t for t in kept)
    assert any("52-week high" in t for t in kept)
    assert any("bags order" in t for t in kept)


def test_a_scoring_headline_survives_without_market_words(cfg, monkeypatch):
    """"Cipla gets USFDA approval" must not be thrown away as brand news."""
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT")
    body = ('<?xml version="1.0"?><rss version="2.0"><channel>'
            "<item><title>Cipla gets USFDA approval for its generic</title>"
            f"<link>http://x</link><pubDate>{recent}</pubDate></item>"
            "</channel></rss>")
    _serve(monkeypatch, body)

    cipla = Stock("CIPLA", "Cipla", "Healthcare", "Pharmaceuticals", "CIPLA.NS")
    result = news.fetch_one(cipla, cfg)
    assert len(result.items) == 1
    assert result.score > 0


def test_a_veto_headline_is_never_filtered_out(cfg, monkeypatch):
    """
    The relevance rules are heuristics, and one that can silently discard
    "auditor resigns" is worse than one that lets a foreign story through.
    """
    title = "Bosch premises searched in CBI probe"
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT")
    body = ('<?xml version="1.0"?><rss version="2.0"><channel>'
            f"<item><title>{title}</title>"
            f"<link>http://x</link><pubDate>{recent}</pubDate></item>"
            "</channel></rss>")
    _serve(monkeypatch, body)

    # It matches no lexicon phrase and carries no market word, so the noise
    # filter alone would have discarded it.
    assert score_headline(title) == (0.0, [])
    assert not news._is_market_news(title)

    result = news.fetch_one(one_word(), cfg)
    assert result.veto == "CBI investigation"
    assert result.veto_headline == title


# ------------------------------------------------------- veto false positives

@pytest.mark.parametrize("headline", [
    "NCLT approves amalgamation of Inzpera Healthsciences with Cipla",
    "NCLT clears Inzpera Healthsciences merger with Cipla",
])
def test_routine_tribunal_housekeeping_is_not_a_veto(headline):
    """
    The NCLT handles subsidiary amalgamations alongside insolvency. A bare
    "nclt" match vetoed Cipla on live data for a merger approval, which is
    housekeeping, not distress.
    """
    assert veto_reason(headline) is None


@pytest.mark.parametrize("headline, expected", [
    ("NCLT admits insolvency plea against Acme", "insolvency proceedings"),
    ("Acme admitted to NCLT after lender petition", "NCLT insolvency admission"),
    ("Creditor files NCLT petition against Acme", "NCLT petition"),
])
def test_real_insolvency_still_vetoes(headline, expected):
    assert veto_reason(headline) == expected
