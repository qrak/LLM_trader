"""RSS feed fetching and parsing primitives for the news ingestion pipeline.

These are the reusable building blocks shared between the runtime provider and
the operator preview scripts.
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import html as html_module

import aiohttp


# Source registry – default values; callers may supply a filtered subset.

_DEFAULT_RSS_SOURCES: list[dict[str, str]] = [
    {"name": "coindesk",      "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "cointelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "decrypt",       "url": "https://decrypt.co/feed"},
    {"name": "cryptoslate",   "url": "https://cryptoslate.com/feed/"},
]

RSS_SOURCES: list[dict[str, str]] = _DEFAULT_RSS_SOURCES


def get_sources(
    enabled_names: list[str] | None = None,
    source_urls: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Return the enabled subset of RSS_SOURCES.

    *enabled_names* is a list of lowercase source name keys (e.g. ``["coindesk",
    "decrypt"]``).  Pass ``None`` or an empty list to return all sources.
    *source_urls* can override the default source registry from config.
    """
    registry = RSS_SOURCES
    if source_urls:
        registry = [
            {"name": str(name).strip().lower(), "url": str(url).strip()}
            for name, url in source_urls.items()
            if name and url
        ]

    if not enabled_names:
        return registry
    names = {n.lower().strip() for n in enabled_names}
    return [s for s in registry if s["name"].lower() in names]


# URL normalisation

_TRACKING_PARAMS: frozenset[str] = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid",
})


def normalize_url(raw_url: str) -> str:
    """Strip tracking query-params and trailing slashes from *raw_url*."""
    if not raw_url:
        return ""
    parsed = urlparse(raw_url.strip())
    filtered_qs = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    new_query = urlencode(filtered_qs, doseq=True)
    normalized_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    cleaned = parsed._replace(query=new_query, fragment="", path=normalized_path)
    return urlunparse(cleaned)


# HTML text extraction

def strip_html(text: str) -> str:
    """Remove HTML tags, unescape HTML entities, and collapse whitespace."""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    unescaped = html_module.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def extract_html_body_text(html_text: str) -> str:
    """Extract readable body text from an HTML page or fragment."""
    soup_text = _extract_html_body_text_bs4(html_text)
    if soup_text:
        return soup_text

    parser_text = _extract_html_body_text_parser(html_text)
    if parser_text:
        return parser_text

    return strip_html(html_text)


class _HtmlBodyTextParser(HTMLParser):
    """Extract readable text while ignoring executable/non-content tags."""

    _IGNORED_TAGS: frozenset[str] = frozenset({
        "script", "style", "noscript", "svg", "header", "nav", "footer", "aside", "form"
    })
    _BLOCK_TAGS: frozenset[str] = frozenset({
        "article", "main", "section", "div", "p", "li", "blockquote", "h1", "h2", "h3", "br"
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        if tag_name in self._IGNORED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag_name in self._BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if tag_name in self._IGNORED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag_name in self._BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        chunk = re.sub(r"\s+", " ", data).strip()
        if chunk:
            self._parts.append(chunk)

    def get_text(self) -> str:
        joined = "".join(self._parts)
        lines = [re.sub(r"\s+", " ", line).strip() for line in joined.split("\n\n")]
        non_empty = [line for line in lines if line]
        return "\n\n".join(non_empty)


def _extract_html_body_text_parser(html_text: str) -> str:
    parser = _HtmlBodyTextParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:  # noqa: BLE001
        return ""
    return parser.get_text().strip()


def _extract_html_body_text_bs4(html_text: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:  # noqa: BLE001
        return ""

    for tag in soup.select("script, style, noscript, svg, header, nav, footer, aside, form"):
        tag.decompose()

    selector_groups: list[tuple[str, int]] = [
        (
            "[data-module-name='article-body'], [data-testid='article-body'], "
            "[class*='article-body'], [class*='articleBody'], [class*='ArticleBody']",
            200,
        ),
        (
            "article, [class*='article-content'], [class*='articleContent'], "
            "[class*='ArticleContent'], [class*='post-content'], [class*='entry-content']",
            1,
        ),
        ("main", 1),
    ]

    for selectors, min_len in selector_groups:
        candidates = [
            _text_from_soup_node(node)
            for node in soup.select(selectors)
        ]
        candidates = [candidate for candidate in candidates if len(candidate) >= min_len]
        if candidates:
            return max(candidates, key=len)

    return ""


def _text_from_soup_node(node: Any) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for element in node.find_all(["h1", "h2", "h3", "p", "li", "blockquote"]):
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)

    if not parts:
        text = node.get_text("\n\n", strip=True)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    return "\n\n".join(parts).strip()


# Date parsing

def parse_pub_date_to_epoch(raw_date: str | None) -> float:
    """Convert an RFC-2822 or ISO-8601 date string to a UTC epoch float.

    Returns 0.0 on parse failure so downstream callers always get a number.
    """
    if not raw_date:
        return 0.0
    try:
        parsed = parsedate_to_datetime(raw_date)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError):
        pass
    # Try ISO-8601 fallback
    try:
        parsed = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError):
        return 0.0


# RSS XML parsing

def _first_text(parent: ET.Element, path: str) -> str:
    node = parent.find(path)
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def parse_rss_items(
    payload_text: str,
    source_name: str,
    max_items: int = 50,
) -> list[dict[str, Any]]:
    """Parse an RSS feed XML payload into a list of normalised item dicts.

    Each item has the *raw ingested* shape (not the canonical article schema –
    use :func:`schema_mapper.to_article_schema` to convert).
    """
    results: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(payload_text)
    except ET.ParseError:
        return results

    now_iso = datetime.now(timezone.utc).isoformat()

    for item in root.findall("./channel/item"):
        if len(results) >= max_items:
            break

        title = _first_text(item, "title")
        link = _first_text(item, "link")
        guid = _first_text(item, "guid")
        description = _first_text(item, "description")
        content_encoded = _first_text(
            item, "{http://purl.org/rss/1.0/modules/content/}encoded"
        )
        author = _first_text(item, "{http://purl.org/dc/elements/1.1/}creator")
        pub_date = _first_text(item, "pubDate")

        categories = [
            (cat.text or "").strip()
            for cat in item.findall("category")
            if (cat.text or "").strip()
        ]

        clean_url = normalize_url(link)
        clean_summary = strip_html(description) if description else ""
        body_html = content_encoded or description or ""
        body_text = extract_html_body_text(body_html) if body_html else ""
        body_source = "content:encoded" if content_encoded else "description"

        if not title or not clean_url:
            continue

        results.append({
            "source_name": source_name,
            "source_type": "rss",
            "title": title,
            "url": clean_url,
            "published_at_epoch": parse_pub_date_to_epoch(pub_date),
            "summary": clean_summary,
            "body_text": body_text,
            "body_source": body_source,
            "author": author or None,
            "categories": categories,
            "raw_source_id": guid or None,
            "fetched_at": now_iso,
        })

    return results


# Async source fetching

@dataclass
class FetchResult:
    """Outcome of fetching one news source."""
    source_name: str
    source_type: str
    url: str
    success: bool
    status_code: int | None
    error: str | None
    normalized_items: list[dict[str, Any]] = field(default_factory=list)


async def fetch_source(
    session: aiohttp.ClientSession,
    source_name: str,
    source_url: str,
    max_items: int = 50,
    timeout: float = 20.0,
) -> FetchResult:
    """Fetch one RSS source and return a :class:`FetchResult`."""
    try:
        async with session.get(
            source_url,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            payload_text = await response.text(encoding="utf-8", errors="replace")
            if response.status != 200:
                return FetchResult(
                    source_name=source_name,
                    source_type="rss",
                    url=source_url,
                    success=False,
                    status_code=response.status,
                    error=f"HTTP {response.status}",
                )
            items = parse_rss_items(payload_text, source_name, max_items=max_items)
            return FetchResult(
                source_name=source_name,
                source_type="rss",
                url=source_url,
                success=True,
                status_code=response.status,
                error=None,
                normalized_items=items,
            )
    except asyncio.TimeoutError:
        return FetchResult(
            source_name=source_name,
            source_type="rss",
            url=source_url,
            success=False,
            status_code=None,
            error="TimeoutError",
        )
    except Exception as exc:  # noqa: BLE001
        return FetchResult(
            source_name=source_name,
            source_type="rss",
            url=source_url,
            success=False,
            status_code=None,
            error=f"{type(exc).__name__}: {exc}",
        )


# Deduplication

def dedupe_by_url(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate items by canonical URL, keeping the most recent version."""
    best: dict[str, dict[str, Any]] = {}
    for item in items:
        url = item.get("url") or ""
        if not url:
            continue
        current = best.get(url)
        if current is None:
            best[url] = item
            continue
        # Keep the item with the newer published_at_epoch
        if item.get("published_at_epoch", 0.0) > current.get("published_at_epoch", 0.0):
            best[url] = item
    return list(best.values())


def sort_by_date(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort items newest-first by ``published_at_epoch``."""
    return sorted(items, key=lambda x: x.get("published_at_epoch", 0.0), reverse=True)
