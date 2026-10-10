"""Router for fetching dashboard visualizations and charts."""
import base64
import hashlib
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request, Response

# Browsers may reuse the chart for a minute, the Cloudflare edge for five.
_CHART_CACHE_CONTROL = "public, max-age=60, stale-while-revalidate=300"


class VisualsRouter:
    """Handles endpoints related to charts and visuals."""
    def __init__(self, analysis_engine):
        self.router = APIRouter(prefix="/api/visuals", tags=["visuals"])
        self.analysis_engine = analysis_engine

        self.router.add_api_route("/charts/latest", self.get_latest_chart, methods=["GET"])

    async def get_latest_chart(self, request: Request, response: Response) -> Any:
        """Get the latest generated analysis chart as base64-encoded JSON.

        The body is a few hundred kB of base64 that only changes when an analysis
        produces a new chart, so it carries an ETag and cache headers: browsers and
        the Cloudflare edge revalidate with a 304 instead of pulling the image again
        on every poll.
        """
        last_chart_buffer = self.analysis_engine.last_chart_buffer if self.analysis_engine else None
        if last_chart_buffer:
            last_chart_buffer.seek(0)
            chart_bytes = last_chart_buffer.getvalue()
            if chart_bytes:
                etag = '"' + hashlib.sha256(chart_bytes).hexdigest()[:32] + '"'
                headers = {"ETag": etag, "Cache-Control": _CHART_CACHE_CONTROL}
                if request.headers.get("if-none-match") == etag:
                    return Response(status_code=304, headers=headers)
                response.headers.update(headers)
                chart_base64 = base64.b64encode(chart_bytes).decode("utf-8")
                return {
                    "chart_base64": chart_base64,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            return {
                "error": "Chart buffer exists but is empty",
                "debug": {"buffer_exists": True, "bytes_length": 0}
            }

        try:
            has_buffer_attr = True
            _ = self.analysis_engine.last_chart_buffer
        except AttributeError:
            has_buffer_attr = False

        return {
            "error": "No chart generated recently.",
            "debug": {
                "analysis_engine_exists": self.analysis_engine is not None,
                "has_buffer_attr": has_buffer_attr,
                "buffer_value": str(type(last_chart_buffer)) if last_chart_buffer else "None"
            }
        }

