from fastapi import APIRouter, Request
import json
from pathlib import Path

router = APIRouter(prefix="/api/monitor", tags=["monitor"])

@router.get("/last_prompt")
async def get_last_prompt(request: Request):
    """Get the last prompt sent to the LLM."""
    analysis_engine = request.app.state.analysis_engine
    
    # First try the in-memory value
    last_prompt = getattr(analysis_engine, "last_generated_prompt", None)
    
    if last_prompt:
        return {"prompt": last_prompt, "source": "memory"}
    
    # Fallback: Read from previous_response.json
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    
    if prev_response_file.exists():
        try:
            with open(prev_response_file, "r") as f:
                data = json.load(f)
                # Read the persisted prompt
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
async def get_last_response(request: Request):
    """Get the last response received from the LLM."""
    analysis_engine = request.app.state.analysis_engine
    
    # First try the in-memory value
    last_response = getattr(analysis_engine, "last_llm_response", None)
    
    if last_response:
        return {"response": last_response}
    
    # Fallback: Read from previous_response.json
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

@router.get("/indicators")
async def get_last_indicators(request: Request):
    """Get the last technical indicators."""
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    
    if prev_response_file.exists():
        try:
            with open(prev_response_file, "r") as f:
                data = json.load(f)
                response = data.get("response", {})
                # Filter out text_analysis, return only numeric indicators
                indicators = {k: v for k, v in response.items() if k != "text_analysis"}
                return {
                    "timestamp": data.get("timestamp"),
                    "indicators": indicators
                }
        except Exception as e:
            return {"error": str(e)}
    
    return {"indicators": {}}
