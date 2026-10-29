
import base64
import hashlib
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request, Response

_CHART_CACHE_CONTROL = "public, max-age=60, stale-while-revalidate=300"


class VisualsRouter:
    """Handles endpoints related to charts and visuals."""
    def __init__(self, analysis_engine):
        self.router = APIRouter(prefix="/api/visuals", tags=["visuals"])
        self.analysis_engine = analysis_engine

        self.router.add_api_route("/charts/latest", self.get_latest_chart, methods=["GET"])

    async def get_latest_chart(self, request: Request, response: Response) -> Any:

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

        return {
            "error": "No chart generated recently.",
            "debug": {
                "analysis_engine_exists": self.analysis_engine is not None,
                "has_buffer_attr": self.analysis_engine is not None,
                "buffer_value": str(type(last_chart_buffer)) if last_chart_buffer else "None"
            }
        }

