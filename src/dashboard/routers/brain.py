from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any, List
import json
from pathlib import Path

router = APIRouter(prefix="/api/brain", tags=["brain"])

@router.get("/status")
async def get_brain_status(request: Request):
    """Get the current thought process/status of the brain."""
    brain_service = request.app.state.brain_service
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    
    # Load last analysis for current state
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    stats_file = Path(data_dir) / "trading" / "statistics.json"
    
    status = {
        "status": "active",
        "trend": "--",
        "confidence": "--",
        "action": "--",
        "adx": None,
        "rsi": None
    }
    
    # Extract from previous_response.json
    if prev_response_file.exists():
        try:
            with open(prev_response_file, "r") as f:
                data = json.load(f)
                response = data.get("response", {})
                text = response.get("text_analysis", "")
                
                # Parse trend from text
                if "BULLISH" in text.upper():
                    status["trend"] = "BULLISH"
                elif "BEARISH" in text.upper():
                    status["trend"] = "BEARISH"
                else:
                    status["trend"] = "NEUTRAL"
                
                # Get key indicators
                status["adx"] = response.get("adx")
                status["rsi"] = response.get("rsi")
                
                # Parse action from JSON block in text
                if '"signal":' in text:
                    try:
                        json_start = text.find('```json')
                        json_end = text.find('```', json_start + 7)
                        if json_start != -1 and json_end != -1:
                            json_str = text[json_start + 7:json_end].strip()
                            analysis_json = json.loads(json_str)
                            analysis = analysis_json.get("analysis", {})
                            status["action"] = analysis.get("signal", "--")
                            status["confidence"] = analysis.get("confidence", "--")
                    except Exception:
                        pass
                        
        except Exception:
            pass
    
    # Add trading stats
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
                status["total_trades"] = stats.get("total_trades", 0)
                status["win_rate"] = stats.get("win_rate", 0)
                status["current_capital"] = stats.get("current_capital", 0)
        except Exception:
            pass
    
    return status

@router.get("/memory")
async def get_vector_memory(request: Request, limit: int = 100):
    """Get recent vector memories (synapses)."""
    vector_memory = request.app.state.vector_memory
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    
    result = {
        "experience_count": 0,
        "trades": [],
        "stats": {}
    }
    
    # Get from vector memory if available
    if vector_memory:
        result["experience_count"] = vector_memory.experience_count
        result["stats"] = vector_memory.compute_confidence_stats()
    
    # Also load trade history as "memories"
    trade_history_file = Path(data_dir) / "trading" / "trade_history.json"
    if trade_history_file.exists():
        try:
            with open(trade_history_file, "r") as f:
                trades = json.load(f)
                # Format for visualization
                result["trades"] = [
                    {
                        "id": f"trade_{i}",
                        "timestamp": t.get("timestamp"),
                        "action": t.get("action"),
                        "price": t.get("price"),
                        "confidence": t.get("confidence"),
                        "reasoning": t.get("reasoning", "")[:100]  # Truncate
                    }
                    for i, t in enumerate(trades[-limit:])
                ]
        except Exception:
            pass
    
    return result

@router.get("/rules")
async def get_active_rules(request: Request):
    """Get currently active semantic rules."""
    vector_memory = request.app.state.vector_memory
    if not vector_memory:
        return []
    
    try:
        rules = vector_memory.get_active_rules(n_results=20)
        return rules
    except Exception:
        return []

@router.get("/vectors")
async def get_vector_details(request: Request, query: str = None, limit: int = 50):
    """Get detailed vector memory contents from ChromaDB."""
    vector_memory = request.app.state.vector_memory
    
    result = {
        "experience_count": 0,
        "experiences": [],
        "confidence_stats": {},
        "adx_stats": {},
        "factor_stats": {},
        "rule_count": 0
    }
    
    if not vector_memory:
        return result
    
    try:
        # Get counts
        result["experience_count"] = vector_memory.experience_count
        result["rule_count"] = vector_memory.semantic_rule_count
        
        # Get stats breakdowns
        result["confidence_stats"] = vector_memory.compute_confidence_stats()
        result["adx_stats"] = vector_memory.compute_adx_performance()
        result["factor_stats"] = vector_memory.compute_factor_performance()
        
        # If query provided, retrieve similar experiences
        if query:
            experiences = vector_memory.retrieve_similar_experiences(query, k=limit)
            result["experiences"] = experiences
        else:
            # Get all experiences (if method available) or most recent
            # For now, use a generic query to get stored experiences
            experiences = vector_memory.retrieve_similar_experiences(
                "trading market conditions", k=limit, use_decay=False
            )
            result["experiences"] = experiences
            
    except Exception as e:
        result["error"] = str(e)
    
    return result
