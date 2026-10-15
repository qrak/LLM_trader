"""Deterministic ID generation and canonical article schema mapping.

The downstream consumers (NewsManager, ContextBuilder, IndexManager, dashboard)
expect articles in this exact shape:

    {
        "id":                str,   # deterministic from URL
        "title":             str,
        "body":              str,   # full body text
        "categories":        str,   # pipe-separated e.g. "BTC|DeFi"
        "tags":              str,   # pipe-separated keywords
        "published_on":      float, # epoch seconds (UTC)
        "source_info":       dict,  # {"name": "<source>"}
        "url":               str,
        # downstream-populated fields (set by ArticleProcessor / IndexManager):
        # "detected_coins":      list[str]
        # "detected_coins_str":  str
        # "_normalize()-added fields (title_lower etc.)
    }

This module is the canonical location for article-body boilerplate/tail-marker
sanitization rules.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

_BASE_TAIL_MARKERS: tuple[str, ...] = (
    "More For You",
    "Read full story",
    "Latest Crypto News",
    "Top Stories",
    "About About Us",
    "* * * * * * About",
    "* * * EthicsPrivacy",
    "EthicsPrivacyTerms",
    "Contact Contact Us",
    "Disclosure & Polices",
    "Disclosure & Policies",
    "NewsLatest",
    "Cointelegraph is committed",
)

_SOURCE_TAIL_MARKERS: dict[str, tuple[str, ...]] = {
    "coindesk": (
        "Newsletters CoinDesk",
        "CoinDesk Headlines",
        "Crypto Daybook Americas",
        "CoinDesk Podcast Network",
        "Market Data  * Index Offering",
        "Consensus Miami",
        "Research Hub  * Exchange Benchmark",
    ),
}

_BOILERPLATE_SECTION_PATTERN = re.compile(
    r"(?is)\babout\b.{0,220}\bcontact\b.{0,260}\bnewsletters?\b"
)

_BOILERPLATE_MENU_PATTERN = re.compile(
    r"(?is)\b(?:about|contact|newsletters|videos|podcasts|cryptocurrencies|"
    r"data\s*&\s*indices|consensus|sponsored|research)\b"
    r"(?:\s*\*\s*[\w&:'\-\s\(\)]+){6,}"
)

_MARKET_TICKER_END_PATTERN = re.compile(r"(?is)\bprice data by\b")
_MARKET_TICKER_PRICE_CHANGE_PATTERN = re.compile(
    r"\$[\d,.]+\s*[-+]?\d+(?:\.\d+)?%"
)
_MARKET_TICKER_TRAILING_NAV_PATTERN = re.compile(
    r"(?is)^\s*(?:\*\s*){3,}[^\*\n]{0,160}?(?:\*\s*){3,}"
)
_MARKET_TICKER_SEPARATOR_PREFIX_PATTERN = re.compile(r"(?i)^\s*(?:\*\s*){3,}")
_MARKET_TICKER_MIN_PRICE_CHANGES = 12
_MARKET_TICKER_SCAN_CHARS = 20_000
_NEWS_NAVIGATION_MARKERS = (
    "asset news", "all news news terminal", "guide categories",
    "review categories", "browse directory",
)


def has_news_boilerplate(text: str) -> bool:
    """Detect dense market tickers or clustered publisher navigation, not price prose."""
    prefix = text[:_MARKET_TICKER_SCAN_CHARS]
    return (
        len(_MARKET_TICKER_PRICE_CHANGE_PATTERN.findall(prefix)) >= _MARKET_TICKER_MIN_PRICE_CHANGES
        or sum(marker in prefix.lower() for marker in _NEWS_NAVIGATION_MARKERS) >= 3
    )


def make_article_id(url: str) -> str:
    """Return a 16-hex-char deterministic ID derived from the canonical URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def to_article_schema(item: dict[str, Any]) -> dict[str, Any]:
    """Map a raw ingested item (rss_primitives shape) to the canonical article schema.

    *item* is a dict produced by :func:`rss_primitives.parse_rss_items` or an
    equivalent ingestion source.  ``item["url"]`` must already be normalised.
    """
    url: str = item.get("url") or ""
    title: str = item.get("title") or ""
    source_name: str = str(item.get("source_name") or "").strip().lower()
    body: str = _clean_article_body(
        item.get("body_text") or item.get("summary") or "",
        title,
        source_name,
    )
    raw_categories: list[str] = item.get("categories") or []
    categories_str: str = "|".join(c for c in raw_categories if c)

    return {
        "id":           make_article_id(url),
        "title":        title,
        "body":         body,
        "categories":   categories_str,
        "tags":         "",
        "published_on": float(item.get("published_at_epoch") or 0.0),
        "source_info":  {"name": source_name},
        "url":          url,
    }


def normalize_article_whitespace(text: str) -> str:
    """Normalize article whitespace while preserving paragraph boundaries."""
    normalized = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    normalized = re.sub(r"\n\s*\n", "\n\n", normalized)
    return re.sub(r"\n{3,}", "\n\n", normalized)


_NEWS_CHROME_LINE_TOKENS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"latest news",
        r"published",
        r"preview",
        r"by\s+[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,2}\s*\|\s*edited by\s+[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,2}",
        r"by\s+[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,2}",
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}(?:,\s*\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.)?\s*[A-Z]{2,4})?",
        r"\d+\s+min read",
        r"make preferred on",
        r"share this article",
        r"copy link",
        r"copied to clipboard",
        r"x icon",
        r"x \(twitter\)",
        r"linkedin",
        r"facebook",
        r"email",
        r"newsletters?",
    )
)

_NEWS_SUMMARY_HEADER = re.compile(r"^\s*(summary|key takeaways?)\s*$", re.IGNORECASE)

_NEWS_SUMMARY_LINE = re.compile(r"^\s*(?:show|show more|hide)\s*$", re.IGNORECASE)

_NEWS_BULLET_LINE = re.compile(r"^\s*(?:\*|•|-|\u2013)\s+")

_NEWS_TAIL_MARKERS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"the post.{0,300}?appeared first on",
        r"appeared first on",
        r"subscribe to daily byte-sized",
        r"crypto, delivered directly to your inbox",
        r"sign up by signing up",
        r"by signing up, you will receive",
        r"preview sign up",
        r"newsletterseditorial policy",
        r"editorial policyads disclosure",
        r"financial risk disclaimer",
        r"cointelegraph is committed to providing independent",
        r"the views expressed in this column are those of the author",
        r"terms of service and privacy policy",
        r"©\s*cointelegraph",
        r"all rights reserved",
        r"disclaimer: the information provided",
        r"this article does not constitute",
        r"\s+newsletters?\s*$",
        r"\s+subscribe\s*$",
        r"_\*\*related:\*\*_",
        r"_\*\*magazine:\*\*_",
    )
)

NEWS_TAIL_SCAN_RATIO = 0.3

_NEWS_OUTLET_TOKENS = re.compile(
    r"(?:cryptoslate|cointelegraph|coindesk|the block|decrypt|coingape|bitcoinist|newsbtc|"
    r"cryptopotato|ambcrypto|utoday|bitcoin\.com|dl news|beincrypto)",
    re.IGNORECASE,
)


def _is_chrome_only(text: str) -> bool:
    """Whether the text carries no story at all, only publisher chrome."""
    remaining = text
    for token in (*_NEWS_CHROME_LINE_TOKENS, *_NEWS_TAIL_MARKERS):
        remaining = token.sub(" ", remaining)
    remaining = _NEWS_OUTLET_TOKENS.sub(" ", remaining)
    return not re.sub(r"[^A-Za-z]+", "", remaining)


def _is_chrome_line(line: str) -> bool:
    """Whether a line consists only of publisher chrome (nav, byline, share)."""
    remaining = line.strip()
    if not remaining:
        return False

    for token in _NEWS_CHROME_LINE_TOKENS:
        remaining = token.sub(" ", remaining)

    return not re.sub(r"[\s:|,\u2013\u2014-]+", "", remaining)


def _strip_summary_block(lines: list[str]) -> list[str]:
    """Drop the duplicated 'Summary/Show' bullet preview some outlets prepend."""
    for index, line in enumerate(lines):
        if not _NEWS_SUMMARY_HEADER.match(line):
            continue

        cursor = index + 1
        while cursor < len(lines) and (
            _NEWS_SUMMARY_LINE.match(lines[cursor]) or _NEWS_BULLET_LINE.match(lines[cursor])
        ):
            cursor += 1

        if cursor > index + 1:
            return lines[cursor:]
    return lines


def _strip_tail_boilerplate(text: str) -> str:
    """Cut publisher footers, newsletter calls to action and post credits."""
    scan_from = int(len(text) * NEWS_TAIL_SCAN_RATIO)
    cut_at = len(text)

    for marker in _NEWS_TAIL_MARKERS:
        match = marker.search(text, scan_from)
        if match and match.start() < cut_at:
            cut_at = match.start()

    if cut_at == len(text):
        return text

    line_start = text.rfind("\n", 0, cut_at)
    return text[: line_start if line_start != -1 else cut_at].rstrip()


def strip_news_boilerplate(text: str) -> str:
    """Remove publisher chrome from an article body before it reaches a prompt.

    Real feeds ship nav lines, bylines, share widgets, a duplicated summary block
    and footer/newsletter blocks; they eat the article token budget that should
    carry the story, and the model reads them as content. Bodies that lose all
    their content are reported as empty so the caller can drop the article.
    """
    normalized = _strip_market_ticker_prefix(normalize_article_whitespace(text))
    if not normalized:
        return ""

    if has_news_boilerplate(normalized):
        return ""

    lines = [line for line in normalized.split("\n") if not _is_chrome_line(line)]
    lines = _strip_summary_block(lines)

    cleaned = _strip_tail_boilerplate("\n".join(lines).strip())

    if not cleaned or _is_chrome_only(cleaned):
        return ""
    return cleaned


def _strip_market_ticker_prefix(text: str) -> str:
    marker_match = _MARKET_TICKER_END_PATTERN.search(text[:_MARKET_TICKER_SCAN_CHARS])
    if marker_match is None:
        return text

    prefix = text[:marker_match.start()]
    price_change_count = len(_MARKET_TICKER_PRICE_CHANGE_PATTERN.findall(prefix))
    if price_change_count < _MARKET_TICKER_MIN_PRICE_CHANGES:
        return text

    remainder = text[marker_match.end():]
    remainder = _MARKET_TICKER_TRAILING_NAV_PATTERN.sub("", remainder, count=1)
    remainder = _MARKET_TICKER_SEPARATOR_PREFIX_PATTERN.sub("", remainder, count=1)
    return remainder.lstrip(" \n\t:-|#")


def _find_tail_marker_index(text: str, marker: str) -> int:
    marker_pattern = re.compile(
        rf"(?is){re.escape(marker)}(?=\s*(?:$|\n|\*|:|-|\|))"
    )
    marker_match = marker_pattern.search(text)
    return marker_match.start() if marker_match else -1


def _clean_article_body(body: str, title: str, source_name: str = "") -> str:
    text = normalize_article_whitespace(body)
    text = _strip_market_ticker_prefix(text)

    if title:
        title_index = text.lower().find(title.lower())
        if 0 <= title_index <= 2500:
            text = text[title_index + len(title):].lstrip(" \n\t.:-|#")

    tail_markers = _BASE_TAIL_MARKERS + _SOURCE_TAIL_MARKERS.get(source_name.lower(), ())
    cut_at = len(text)

    separator_match = re.search(r"(?:\*\s*){3,}\s*About\b", text, flags=re.IGNORECASE)
    if separator_match:
        cut_at = min(cut_at, separator_match.start())

    boilerplate_match = _BOILERPLATE_SECTION_PATTERN.search(text)
    if boilerplate_match:
        cut_at = min(cut_at, boilerplate_match.start())

    menu_match = _BOILERPLATE_MENU_PATTERN.search(text)
    if menu_match:
        cut_at = min(cut_at, menu_match.start())

    for marker in tail_markers:
        marker_index = _find_tail_marker_index(text, marker)
        if marker_index != -1:
            cut_at = min(cut_at, marker_index)

    text = text[:cut_at].strip()
    return normalize_article_whitespace(text)
