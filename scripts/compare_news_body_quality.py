"""Compare fresh feed article bodies with cached news quality.

This script is standalone and does not modify runtime behavior.
It reads allowed feeds from config/config.ini and compares:
1) data/news_cache/recent_news.json (cached baseline)
2) fresh fetch from CoinDesk Data endpoint

Output:
- Console summary
- JSON report saved under data/news_fetch_preview/
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse

import aiohttp

COINDESK_DATA_URL = "https://data-api.coindesk.com/news/v1/article/list"
DEFAULT_CONFIG_PATH = Path("config/config.ini")
DEFAULT_CACHE_PATH = Path("data/news_cache/recent_news.json")
DEFAULT_OUT_DIR = Path("data/news_fetch_preview")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fresh news body quality with current cached recent_news.json"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--cache", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--full-body-threshold", type=int, default=500)
    parser.add_argument("--sample-size", type=int, default=8)
    return parser.parse_args()


def load_allowed_feeds(config_path: Path) -> list[str]:
    config = configparser.ConfigParser()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config.read(config_path, encoding="utf-8")
    feeds_raw = config.get("rag", "news_allowed_feeds", fallback="")

    feeds = [feed.strip().lower() for feed in feeds_raw.split(",") if feed.strip()]
    return feeds


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def normalize_cached_articles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    articles = payload.get("articles")
    if not isinstance(articles, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in articles:
        if not isinstance(item, dict):
            continue

        source_key = _safe_str(item.get("source")).lower()
        source_name = ""
        source_info = item.get("source_info")
        if isinstance(source_info, dict):
            source_name = _safe_str(source_info.get("name"))

        body = _safe_str(item.get("body"))
        normalized.append(
            {
                "id": item.get("id"),
                "title": _safe_str(item.get("title")),
                "url": _safe_str(item.get("url")),
                "domain": _extract_domain(_safe_str(item.get("url"))),
                "published_on": item.get("published_on"),
                "source_key": source_key,
                "source_name": source_name,
                "body": body,
                "body_len": len(body),
            }
        )

    return normalized


def normalize_coindesk_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("Data")
    items: list[dict[str, Any]] = []

    if isinstance(data, list):
        items = [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        nested = data.get("LIST")
        if isinstance(nested, list):
            items = [x for x in nested if isinstance(x, dict)]

    normalized: list[dict[str, Any]] = []
    for item in items:
        source_data = item.get("SOURCE_DATA")
        source_data = source_data if isinstance(source_data, dict) else {}

        source_key = _safe_str(source_data.get("SOURCE_KEY") or item.get("SOURCE")).lower()
        source_name = _safe_str(source_data.get("NAME") or item.get("SOURCE"))

        body = _safe_str(item.get("BODY"))
        url = _safe_str(item.get("URL"))

        normalized.append(
            {
                "id": item.get("ID") or item.get("GUID") or item.get("id"),
                "title": _safe_str(item.get("TITLE") or item.get("title")),
                "url": url,
                "domain": _extract_domain(url),
                "published_on": item.get("PUBLISHED_ON") or item.get("published_on"),
                "source_key": source_key,
                "source_name": source_name,
                "body": body,
                "body_len": len(body),
            }
        )

    return normalized


def filter_by_allowed_feeds(
    items: list[dict[str, Any]],
    allowed_feeds: list[str],
) -> list[dict[str, Any]]:
    if not allowed_feeds:
        return items
    allowed = set(allowed_feeds)
    return [item for item in items if item.get("source_key") in allowed]


def summarize_items(items: list[dict[str, Any]], full_body_threshold: int) -> dict[str, Any]:
    body_lengths = [int(item.get("body_len") or 0) for item in items]
    full_count = sum(1 for x in body_lengths if x >= full_body_threshold)

    by_source: dict[str, dict[str, Any]] = {}
    for item in items:
        key = item.get("source_key") or "unknown"
        source_stats = by_source.setdefault(
            key,
            {"count": 0, "full_body_count": 0, "avg_body_len": 0.0, "_lens": []},
        )
        body_len = int(item.get("body_len") or 0)
        source_stats["count"] += 1
        source_stats["_lens"].append(body_len)
        if body_len >= full_body_threshold:
            source_stats["full_body_count"] += 1

    for key, value in by_source.items():
        lens = value.pop("_lens")
        value["avg_body_len"] = round(sum(lens) / max(1, len(lens)), 2)
        value["full_body_ratio"] = round(
            value["full_body_count"] / max(1, value["count"]),
            4,
        )

    return {
        "count": len(items),
        "full_body_threshold": full_body_threshold,
        "full_body_count": full_count,
        "full_body_ratio": round(full_count / max(1, len(items)), 4),
        "avg_body_len": round(sum(body_lengths) / max(1, len(body_lengths)), 2),
        "median_body_len": int(median(body_lengths)) if body_lengths else 0,
        "sources": by_source,
    }


def sample_items(items: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    sorted_items = sorted(items, key=lambda x: int(x.get("body_len") or 0), reverse=True)
    out: list[dict[str, Any]] = []

    for item in sorted_items[:size]:
        body = _safe_str(item.get("body"))
        out.append(
            {
                "source_key": item.get("source_key"),
                "source_name": item.get("source_name"),
                "title": item.get("title"),
                "url": item.get("url"),
                "published_on": item.get("published_on"),
                "body_len": item.get("body_len"),
                "body_preview": body[:400],
            }
        )

    return out


def compare_overlap(
    cached_items: list[dict[str, Any]],
    fresh_items: list[dict[str, Any]],
) -> dict[str, Any]:
    cached_urls = {str(item.get("url") or "") for item in cached_items if item.get("url")}
    fresh_urls = {str(item.get("url") or "") for item in fresh_items if item.get("url")}

    overlap = cached_urls.intersection(fresh_urls)
    return {
        "cached_url_count": len(cached_urls),
        "fresh_url_count": len(fresh_urls),
        "overlap_url_count": len(overlap),
        "overlap_ratio_vs_cached": round(len(overlap) / max(1, len(cached_urls)), 4),
        "overlap_ratio_vs_fresh": round(len(overlap) / max(1, len(fresh_urls)), 4),
    }


async def fetch_coindesk_news(limit: int, timeout_seconds: float) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    params = {"lang": "EN", "limit": int(limit)}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(COINDESK_DATA_URL, params=params) as response:
            text = await response.text(encoding="utf-8", errors="replace")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"_raw_text": text}

            return {
                "status_code": response.status,
                "url": str(response.url),
                "payload": payload,
            }


def print_console_summary(
    allowed_feeds: list[str],
    cache_summary: dict[str, Any],
    fresh_summary: dict[str, Any],
    overlap: dict[str, Any],
    report_path: Path,
) -> None:
    print("=== News Body Quality Comparison ===")
    print(f"Allowed feeds from config: {', '.join(allowed_feeds) if allowed_feeds else '(none)'}")
    print("")
    print("[Cached recent_news.json]")
    print(
        "count={count}, full_body_ratio={full_body_ratio}, avg_body_len={avg_body_len}, median_body_len={median_body_len}".format(
            **cache_summary
        )
    )
    print("[Fresh endpoint fetch]")
    print(
        "count={count}, full_body_ratio={full_body_ratio}, avg_body_len={avg_body_len}, median_body_len={median_body_len}".format(
            **fresh_summary
        )
    )
    print("[Overlap]")
    print(
        "cached_urls={cached_url_count}, fresh_urls={fresh_url_count}, overlap={overlap_url_count}".format(
            **overlap
        )
    )
    print(f"Report saved: {report_path}")


async def run() -> int:
    args = parse_args()

    config_path = Path(args.config)
    cache_path = Path(args.cache)
    out_dir = Path(args.out_dir)

    allowed_feeds = load_allowed_feeds(config_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")

    cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    cached_all = normalize_cached_articles(cached_payload)
    cached_filtered = filter_by_allowed_feeds(cached_all, allowed_feeds)

    fresh_result = await fetch_coindesk_news(limit=args.limit, timeout_seconds=args.timeout_seconds)
    fresh_payload = fresh_result["payload"]
    fresh_all = normalize_coindesk_items(fresh_payload if isinstance(fresh_payload, dict) else {})
    fresh_filtered = filter_by_allowed_feeds(fresh_all, allowed_feeds)

    cache_summary = summarize_items(cached_filtered, full_body_threshold=args.full_body_threshold)
    fresh_summary = summarize_items(fresh_filtered, full_body_threshold=args.full_body_threshold)
    overlap = compare_overlap(cached_filtered, fresh_filtered)

    report = {
        "created_at": now_iso(),
        "inputs": {
            "config": str(config_path),
            "cache": str(cache_path),
            "allowed_feeds": allowed_feeds,
            "coindesk_request_url": fresh_result.get("url"),
            "coindesk_status_code": fresh_result.get("status_code"),
            "limit": int(args.limit),
            "full_body_threshold": int(args.full_body_threshold),
        },
        "cache_summary": cache_summary,
        "fresh_summary": fresh_summary,
        "overlap": overlap,
        "samples": {
            "cached_top_by_body_len": sample_items(cached_filtered, args.sample_size),
            "fresh_top_by_body_len": sample_items(fresh_filtered, args.sample_size),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"news_body_quality_comparison_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_console_summary(
        allowed_feeds=allowed_feeds,
        cache_summary=cache_summary,
        fresh_summary=fresh_summary,
        overlap=overlap,
        report_path=report_path,
    )

    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
