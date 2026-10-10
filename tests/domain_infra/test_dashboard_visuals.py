"""Dense domain tests for the visuals router (src/dashboard/routers/visuals.py).

The chart endpoint hands out a few hundred kB of base64 per call, so its caching
contract is behaviour that must not regress: one ETag per chart image, a 304 on a
matching If-None-Match, and the unchanged debug payloads when there is no chart.
"""

import base64
import hashlib
from io import BytesIO

from fastapi import FastAPI
from starlette.testclient import TestClient

from src.dashboard.routers.visuals import _CHART_CACHE_CONTROL, VisualsRouter

CHART = b"\x89PNG\r\n\x1a\n" + b"chart-bytes" * 32


class _Engine:
    """Minimal stand-in for the analysis engine's chart buffer."""

    def __init__(self, chart: bytes | None):
        self.last_chart_buffer = BytesIO(chart) if chart is not None else None


def _client(engine) -> TestClient:
    app = FastAPI()
    app.include_router(VisualsRouter(analysis_engine=engine).router)
    return TestClient(app)


def _etag_for(chart: bytes) -> str:
    return '"' + hashlib.sha256(chart).hexdigest()[:32] + '"'


def test_returns_base64_chart_with_etag_and_cache_headers():
    r = _client(_Engine(CHART)).get("/api/visuals/charts/latest")
    assert r.status_code == 200
    body = r.json()
    assert base64.b64decode(body["chart_base64"]) == CHART
    assert body["timestamp"]
    assert r.headers["etag"] == _etag_for(CHART)
    assert r.headers["cache-control"] == _CHART_CACHE_CONTROL


def test_matching_if_none_match_gets_a_304_without_a_body():
    client = _client(_Engine(CHART))
    etag = client.get("/api/visuals/charts/latest").headers["etag"]
    r = client.get("/api/visuals/charts/latest", headers={"If-None-Match": etag})
    assert r.status_code == 304
    assert r.content == b""
    assert r.headers["etag"] == etag


def test_etag_changes_with_the_chart():
    first = _client(_Engine(CHART)).get("/api/visuals/charts/latest").headers["etag"]
    second = _client(_Engine(CHART + b"new-analysis")).get("/api/visuals/charts/latest").headers["etag"]
    assert first != second


def test_stale_etag_still_gets_the_image():
    r = _client(_Engine(CHART)).get("/api/visuals/charts/latest", headers={"If-None-Match": '"stale"'})
    assert r.status_code == 200
    assert r.json()["chart_base64"]


def test_empty_buffer_reports_the_empty_buffer_payload():
    r = _client(_Engine(b"")).get("/api/visuals/charts/latest")
    assert r.json() == {
        "error": "Chart buffer exists but is empty",
        "debug": {"buffer_exists": True, "bytes_length": 0},
    }
    assert "etag" not in r.headers


def test_idle_engine_reports_the_debug_payload():
    r = _client(_Engine(None)).get("/api/visuals/charts/latest")
    body = r.json()
    assert body["error"] == "No chart generated recently."
    assert body["debug"] == {
        "analysis_engine_exists": True,
        "has_buffer_attr": True,
        "buffer_value": "None",
    }


def test_missing_engine_flags_both_debug_fields():
    r = _client(None).get("/api/visuals/charts/latest")
    assert r.json()["debug"] == {
        "analysis_engine_exists": False,
        "has_buffer_attr": False,
        "buffer_value": "None",
    }


def test_visuals_path_uses_the_long_edge_policy():
    from src.dashboard.server import _api_cache_policies

    browser, edge = _api_cache_policies("/api/visuals/charts/latest", {})
    assert browser == "public, max-age=60, stale-while-revalidate=300"
    assert edge == "public, max-age=300, stale-while-revalidate=60, stale-if-error=600"
    # Every other API route keeps the short default policy.
    assert _api_cache_policies("/api/brain/position", {})[1].startswith("public, max-age=60")
