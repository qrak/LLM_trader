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


def _clean_article_body(body: str, title: str, source_name: str = "") -> str:
    text = normalize_article_whitespace(body)

    if title:
        title_index = text.lower().find(title.lower())
        if 0 <= title_index <= 2500:
            text = text[title_index + len(title):].lstrip(" \n\t:-|#")

    tail_markers = _BASE_TAIL_MARKERS + _SOURCE_TAIL_MARKERS.get(source_name.lower(), ())
    lowered = text.lower()
    cut_at = len(text)

    separator_match = re.search(r"(?:\*\s*){3,}\s*About\b", text, flags=re.IGNORECASE)
    if separator_match and separator_match.start() >= 300:
        cut_at = min(cut_at, separator_match.start())

    boilerplate_match = _BOILERPLATE_SECTION_PATTERN.search(text)
    if boilerplate_match and boilerplate_match.start() >= 300:
        cut_at = min(cut_at, boilerplate_match.start())

    menu_match = _BOILERPLATE_MENU_PATTERN.search(text)
    if menu_match and menu_match.start() >= 300:
        cut_at = min(cut_at, menu_match.start())

    for marker in tail_markers:
        marker_index = lowered.find(marker.lower())
        if marker_index >= 300:
            cut_at = min(cut_at, marker_index)

    text = text[:cut_at].strip()
    return normalize_article_whitespace(text)
