from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
import io
import os
from pathlib import Path

router = APIRouter(prefix="/api/visuals", tags=["visuals"])

@router.get("/charts/latest")
async def get_latest_chart(request: Request):
    """Get the latest generated analysis chart."""
    analysis_engine = request.app.state.analysis_engine
    
    # Check if we have an in-memory chart
    last_chart_buffer = getattr(analysis_engine, "last_chart_buffer", None)
    
    if last_chart_buffer:
        # Reset pointer just in case
        last_chart_buffer.seek(0)
        return StreamingResponse(io.BytesIO(last_chart_buffer.getvalue()), media_type="image/png")
        
    return {"error": "No chart generated recently."}

@router.get("/charts")
async def list_charts(request: Request):
    """List available charts on disk (if debug saving is enabled)."""
    config = request.app.state.config
    # Access config directly if possible, or assume defaults
    # config object is passed from main implementation
    
    chart_dir = getattr(config, "chart_save_path", "chart_images")
    if not os.path.exists(chart_dir):
        return []
        
    files = sorted(Path(chart_dir).glob("*.png"), key=os.path.getmtime, reverse=True)
    return [f.name for f in files[:50]] # Return top 50 recent

@router.get("/charts/{filename}")
async def get_chart_image(request: Request, filename: str):
    """Serve a specific chart image."""
    config = request.app.state.config
    chart_dir = getattr(config, "chart_save_path", "chart_images")
    file_path = Path(chart_dir) / filename
    
    if file_path.exists():
        return FileResponse(file_path)
    
    return {"error": "File not found"}
