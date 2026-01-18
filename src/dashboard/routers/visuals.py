from fastapi import APIRouter, Request
import base64
from datetime import datetime
from typing import Dict, Any

router = APIRouter(prefix="/api/visuals", tags=["visuals"])

@router.get("/charts/latest")
async def get_latest_chart(request: Request) -> Dict[str, Any]:
    """Get the latest generated analysis chart as base64-encoded JSON."""
    analysis_engine = request.app.state.analysis_engine
    last_chart_buffer = getattr(analysis_engine, "last_chart_buffer", None)
    if last_chart_buffer:
        last_chart_buffer.seek(0)
        chart_bytes = last_chart_buffer.getvalue()
        if chart_bytes:
            chart_base64 = base64.b64encode(chart_bytes).decode("utf-8")
            return {
                "chart_base64": chart_base64,
                "timestamp": datetime.now().isoformat()
            }
        return {
            "error": "Chart buffer exists but is empty",
            "debug": {"buffer_exists": True, "bytes_length": 0}
        }
    return {
        "error": "No chart generated recently.",
        "debug": {
            "analysis_engine_exists": analysis_engine is not None,
            "has_buffer_attr": hasattr(analysis_engine, "last_chart_buffer") if analysis_engine else False,
            "buffer_value": str(type(last_chart_buffer)) if last_chart_buffer else "None"
        }
    }

