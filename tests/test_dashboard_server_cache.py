from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.dashboard.server import DashboardServer


def _build_server(tmp_path):
    config = SimpleNamespace(
        DATA_DIR=str(tmp_path),
        DEMO_QUOTE_CAPITAL=10000.0,
        DASHBOARD_ENABLE_CORS=False,
        DASHBOARD_CORS_ORIGINS=[],
        TIMEFRAME="4h",
        STOP_LOSS_TYPE="hard",
        STOP_LOSS_CHECK_INTERVAL="15m",
        TAKE_PROFIT_TYPE="hard",
        TAKE_PROFIT_CHECK_INTERVAL="15m",
    )
    return DashboardServer(
        brain_service=MagicMock(),
        vector_memory=MagicMock(),
        analysis_engine=MagicMock(),
        config=config,
        logger=MagicMock(),
        unified_parser=MagicMock(),
        persistence=MagicMock(),
        exchange_manager=MagicMock(),
    )


def test_versioned_static_assets_are_immutable_for_cloudflare(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/main.js?v=5.0")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == (
        "public, max-age=31536000, immutable"
    )
    assert response.headers["CDN-Cache-Control"] == (
        "public, max-age=31536000, stale-while-revalidate=86400, stale-if-error=604800"
    )
    assert response.headers["Cloudflare-CDN-Cache-Control"] == response.headers["CDN-Cache-Control"]


def test_html_shell_keeps_browser_revalidation_but_short_edge_cache(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "public, max-age=30, must-revalidate"
    assert response.headers["CDN-Cache-Control"] == (
        "public, max-age=300, stale-while-revalidate=60, stale-if-error=600"
    )
    assert response.headers["Cloudflare-CDN-Cache-Control"] == response.headers["CDN-Cache-Control"]


def test_api_responses_use_short_cloudflare_edge_cache_only(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.headers["Cache-Control"] == "public, max-age=15"
    assert response.headers["CDN-Cache-Control"] == (
        "public, max-age=60, stale-while-revalidate=30, stale-if-error=300"
    )
    assert response.headers["Cloudflare-CDN-Cache-Control"] == response.headers["CDN-Cache-Control"]


def test_refresh_price_endpoint_bypasses_cache(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/api/brain/refresh-price")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate, proxy-revalidate"
    )
    assert response.headers["CDN-Cache-Control"] == "no-store"
    assert response.headers["Cloudflare-CDN-Cache-Control"] == response.headers["CDN-Cache-Control"]


def test_vector_query_endpoint_bypasses_cache_for_search_queries(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/api/brain/vectors?query=btc&limit=50")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate, proxy-revalidate"
    )
    assert response.headers["CDN-Cache-Control"] == "no-store"
    assert response.headers["Cloudflare-CDN-Cache-Control"] == response.headers["CDN-Cache-Control"]


def test_brain_lifecycle_endpoint_bypasses_cache(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/api/brain/lifecycle")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate, proxy-revalidate"
    )
    assert response.headers["CDN-Cache-Control"] == "no-store"
    assert response.headers["Cloudflare-CDN-Cache-Control"] == response.headers["CDN-Cache-Control"]


def test_brain_refresh_endpoint_bypasses_cache(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        response = client.post("/api/brain/refresh")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate, proxy-revalidate"
    )
    assert response.headers["CDN-Cache-Control"] == "no-store"
    assert response.headers["Cloudflare-CDN-Cache-Control"] == response.headers["CDN-Cache-Control"]


def test_cacheable_api_emits_etag_and_supports_not_modified(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        first = client.get("/api/brain/status")
        etag = first.headers.get("ETag")
        second = client.get("/api/brain/status", headers={"If-None-Match": etag})

    assert first.status_code == 200
    assert etag is not None
    assert second.status_code == 304
    assert second.headers.get("ETag") == etag


def test_no_store_endpoint_does_not_emit_etag(tmp_path):
    server = _build_server(tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/api/brain/refresh-price")

    assert response.status_code == 200
    assert response.headers.get("ETag") is None