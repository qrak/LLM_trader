from fastapi import APIRouter, Request
from typing import Dict, Any
import json
from pathlib import Path

router = APIRouter(prefix="/api/monitor", tags=["monitor"])

@router.get("/last_prompt")
async def get_last_prompt(request: Request) -> Dict[str, Any]:
    """Get the last prompt sent to the LLM."""
    analysis_engine = request.app.state.analysis_engine
    last_prompt = getattr(analysis_engine, "last_generated_prompt", None)
    if last_prompt:
        return {"prompt": last_prompt, "source": "memory"}
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    if prev_response_file.exists():
        try:
            with open(prev_response_file, "r") as f:
                data = json.load(f)
                prompt = data.get("prompt")
                if prompt:
                    return {
                        "prompt": prompt,
                        "source": "disk",
                        "timestamp": data.get("timestamp", "unknown")
                    }
        except Exception:
            pass
    return {"prompt": "No prompt generated yet.", "source": None}

@router.get("/last_response")
async def get_last_response(request: Request) -> Dict[str, Any]:
    """Get the last response received from the LLM."""
    analysis_engine = request.app.state.analysis_engine
    last_response = getattr(analysis_engine, "last_llm_response", None)
    if last_response:
        return {"response": last_response}
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    if prev_response_file.exists():
        try:
            with open(prev_response_file, "r") as f:
                data = json.load(f)
                response = data.get("response", {})
                text_analysis = response.get("text_analysis", "No analysis available")
                return {
                    "response": text_analysis,
                    "timestamp": data.get("timestamp"),
                    "indicators": {k: v for k, v in response.items() if k != "text_analysis"}
                }
        except Exception as e:
            return {"response": f"Error reading previous response: {str(e)}"}
    return {"response": "No response received yet."}


@router.get("/system_prompt")
async def get_system_prompt(request: Request) -> Dict[str, Any]:
    """Get the last system prompt (contains brain context and trading rules)."""
    analysis_engine = request.app.state.analysis_engine
    system_prompt = getattr(analysis_engine, "last_system_prompt", None)
    if system_prompt:
        has_brain = "TRADING BRAIN" in system_prompt
        return {
            "system_prompt": system_prompt,
            "source": "memory",
            "has_brain_context": has_brain
        }
    return {"system_prompt": "No system prompt generated yet.", "source": None, "has_brain_context": False}


@router.get("/costs")
async def get_api_costs() -> Dict[str, Any]:
    """Get current API cost tracking data."""
    from src.dashboard.dashboard_state import dashboard_state
    return dashboard_state.get_cost_data()


@router.post("/costs/reset")
async def reset_api_costs() -> Dict[str, Any]:
    """Reset API cost tracking to zero."""
    from src.dashboard.dashboard_state import dashboard_state
    await dashboard_state.reset_api_costs()
    return {"status": "ok", "message": "API costs reset to zero"}
