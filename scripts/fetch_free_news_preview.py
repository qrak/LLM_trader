"""Standalone free crypto news fetcher and preview utility.

Thin wrapper around the shared runtime ingestion code in
``src/rag/news_ingestion``.  Fetches from public no-key RSS feeds and
optionally from the cryptocurrency.cv API, prints a preview table, and saves
timestamped raw and normalized JSON snapshots to ``data/news_fetch_preview``.

Changes from earlier standalone version
-----------------------------------------
* RSS parsing, URL normalisation, and deduplication now come from the shared
  ``src.rag.news_ingestion.rss_primitives`` module so previewing reflects the
  real runtime behaviour.
* Body enrichment uses ``Crawl4AIEnricher`` (Crawl4AI if available, aiohttp
  fallback otherwise).
* cryptocurrency.cv remains a script-only extension (not a runtime source).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from rich.console import Console
from rich.table import Table

# Allow running from repo root without installing the package
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rag.news_ingestion.rss_primitives import (  # noqa: E402
    RSS_SOURCES,
    FetchResult,
    dedupe_by_url,
    extract_html_body_text,
    normalize_url,
    fetch_source,
    parse_pub_date_to_epoch,
    sort_by_date,
    strip_html,
)
from src.rag.news_ingestion.crawl4ai_enricher import Crawl4AIEnricher  # noqa: E402
from src.rag.news_ingestion.schema_mapper import to_article_schema  # noqa: E402


# ---------------------------------------------------------------------------
# Script-only: cryptocurrency.cv source (not used in runtime)
# ---------------------------------------------------------------------------

CRYPTOCURRENCY_CV_SOURCE = {
    "name": "cryptocurrency.cv",
    "url": "https://cryptocurrency.cv/api/news?limit=50",
}


def _parse_cryptocurrency_cv_items(payload_text: str) -> list[dict[str, Any]]:
    data = json.loads(payload_text)
    items = data.get("articles", [])
    now_iso = datetime.now(timezone.utc).isoformat()
    normalized: list[dict[str, Any]] = []

    for item in items:
        title = (item.get("title") or "").strip()
        raw_url = (item.get("link") or item.get("url") or "").strip()
        if not title or not raw_url:
            continue

        published_raw = item.get("pubDate") or item.get("published_at")
        published_at_epoch: float = 0.0
        if isinstance(published_raw, str):
            published_at_epoch = parse_pub_date_to_epoch(published_raw)

        summary = item.get("description") or ""
        if isinstance(summary, str):
            summary = strip_html(summary)
        else:
            summary = ""

        body_html = item.get("content") or item.get("description") or ""
        if not isinstance(body_html, str):
            body_html = ""
        body_text = extract_html_body_text(body_html) if body_html else ""

        categories = item.get("categories") or []
        if isinstance(categories, str):
            categories = [categories]

        normalized.append({
            "source_name": "cryptocurrency.cv",
            "source_type": "api",
            "title": title,
            "url": normalize_url(raw_url),
            "published_at_epoch": published_at_epoch,
            "summary": summary,
            "body_text": body_text,
            "body_source": "api_content_or_description",
            "author": item.get("author"),
            "categories": categories if isinstance(categories, list) else [],
            "raw_source_id": item.get("id"),
            "fetched_at": now_iso,
        })

    return normalized


async def _fetch_cryptocurrency_cv(
    session: aiohttp.ClientSession,
    timeout: float,
) -> FetchResult:
    source = CRYPTOCURRENCY_CV_SOURCE
    try:
        async with session.get(
            source["url"],
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            payload_text = await resp.text(encoding="utf-8", errors="replace")
            if resp.status != 200:
                return FetchResult(
                    source_name=source["name"],
                    source_type="api",
                    url=source["url"],
                    success=False,
                    status_code=resp.status,
                    error=f"HTTP {resp.status}",
                )
            items = _parse_cryptocurrency_cv_items(payload_text)
            return FetchResult(
                source_name=source["name"],
                source_type="api",
                url=source["url"],
                success=True,
                status_code=resp.status,
                error=None,
                normalized_items=items,
            )
    except Exception as exc:  # noqa: BLE001
        return FetchResult(
            source_name=source["name"],
            source_type="api",
            url=source["url"],
            success=False,
            status_code=None,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Preview rendering + snapshot saving
# ---------------------------------------------------------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_preview(items: list[dict[str, Any]], preview_limit: int) -> None:
    console = Console()
    table = Table(title=f"Free Crypto News Preview (top {preview_limit})")
    table.add_column("#", justify="right")
    table.add_column("Source", overflow="fold")
    table.add_column("Published (UTC)", overflow="fold")
    table.add_column("Title", overflow="fold")
    table.add_column("URL", overflow="fold")

    for idx, item in enumerate(items[:preview_limit], start=1):
        pub = item.get("published_on") or item.get("published_at_epoch") or 0.0
        pub_str = (
            datetime.fromtimestamp(pub, tz=timezone.utc).isoformat()
            if pub
            else ""
        )
        source_raw = item.get("source_info") or item.get("source_name") or ""
        source_name = source_raw.get("name") if isinstance(source_raw, dict) else str(source_raw)
        table.add_row(
            str(idx),
            source_name,
            pub_str,
            str(item.get("title") or ""),
            str(item.get("url") or ""),
        )

    console.print(table)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    timeout = float(args.timeout_seconds)
    concurrency = int(args.max_concurrency)
    min_body_chars = int(args.min_body_chars)

    async with aiohttp.ClientSession() as session:
        # Fetch all RSS sources in parallel
        rss_tasks = [
            fetch_source(session, s["name"], s["url"], max_items=50, timeout=timeout)
            for s in RSS_SOURCES
        ]
        if args.include_cryptocurrency_cv:
            rss_tasks.append(_fetch_cryptocurrency_cv(session, timeout))

        results: list[FetchResult] = await asyncio.gather(*rss_tasks)

        merged_items: list[dict[str, Any]] = []
        for r in results:
            merged_items.extend(r.normalized_items)

        # Enrichment
        enriched_count = 0
        if args.enrich_from_article_pages:
            enricher = Crawl4AIEnricher(
                concurrency=concurrency,
                timeout=timeout,
                min_chars=min_body_chars,
                use_crawl4ai=args.use_crawl4ai,
            )
            enriched_count = await enricher.enrich_items(merged_items, session)

    deduped = dedupe_by_url(merged_items)
    sorted_items = sort_by_date(deduped)
    canonical_items = [to_article_schema(item) for item in sorted_items]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    raw_path = out_dir / f"news_raw_{timestamp}.json"
    normalized_path = out_dir / f"news_normalized_{timestamp}.json"

    raw_payload: dict[str, Any] = {
        "created_at": now_utc_iso(),
        "sources": [
            {
                "source_name": r.source_name,
                "source_type": r.source_type,
                "url": r.url,
                "success": r.success,
                "status_code": r.status_code,
                "error": r.error,
            }
            for r in results
        ],
    }

    normalized_payload: dict[str, Any] = {
        "created_at": now_utc_iso(),
        "stats": {
            "source_count": len(results),
            "source_success_count": sum(1 for r in results if r.success),
            "raw_item_count": len(merged_items),
            "deduped_item_count": len(canonical_items),
            "page_enriched_count": enriched_count,
        },
        "items": canonical_items,
    }

    save_json(raw_path, raw_payload)
    save_json(normalized_path, normalized_payload)

    render_preview(canonical_items, preview_limit=int(args.limit))

    console = Console()
    console.print(f"\nRaw snapshot:        {raw_path}")
    console.print(f"Normalized snapshot: {normalized_path}")
    console.print(
        f"Summary: sources={len(results)}, "
        f"ok={sum(1 for r in results if r.success)}, "
        f"raw={len(merged_items)}, deduped={len(canonical_items)}, "
        f"enriched={enriched_count}"
    )

    for r in results:
        if not r.success:
            console.print(f"  [red]Source error:[/red] {r.source_name} -> {r.error}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and preview free no-key crypto news sources."
    )
    parser.add_argument("--limit", type=int, default=15,
                        help="Rows to show in terminal preview table.")
    parser.add_argument("--out-dir", type=str, default="data/news_fetch_preview",
                        help="Directory for timestamped JSON snapshots.")
    parser.add_argument("--include-cryptocurrency-cv", action="store_true",
                        help="Include cryptocurrency.cv API source.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0,
                        help="Per-request timeout in seconds.")
    parser.add_argument("--enrich-from-article-pages", action="store_true",
                        help="Enrich short bodies by fetching article pages.")
    parser.add_argument("--min-body-chars", type=int, default=600,
                        help="Minimum body_text length before enrichment.")
    parser.add_argument("--max-concurrency", type=int, default=6,
                        help="Max concurrent enrichment requests.")
    parser.add_argument("--use-crawl4ai", action="store_true", default=False,
                        help="Use Crawl4AI browser enrichment (requires crawl4ai installed).")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

