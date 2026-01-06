from fastapi import APIRouter, Request
import json
from pathlib import Path
from datetime import datetime

router = APIRouter(prefix="/api/performance", tags=["performance"])

@router.get("/history")
async def get_performance_history(request: Request):
    """Get historical performance data for the chart."""
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    
    # Try to build equity curve from trade_history.json
    trade_history_file = Path(data_dir) / "trading" / "trade_history.json"
    stats_file = Path(data_dir) / "trading" / "statistics.json"
    
    equity_curve = []
    stats = {}
    
    # Load statistics for summary
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
        except Exception:
            pass
    
    # Build equity curve from trade history
    if trade_history_file.exists():
        try:
            with open(trade_history_file, "r") as f:
                trades = json.load(f)
            
            # Calculate running equity
            initial_capital = stats.get("initial_capital", 10000.0)
            running_capital = initial_capital
            
            for trade in trades:
                ts = trade.get("timestamp")
                action = trade.get("action", "")
                reasoning = trade.get("reasoning", "")
                
                # Extract P&L from CLOSE actions
                if "CLOSE" in action and "P&L:" in reasoning:
                    try:
                        # Parse "P&L: +3.55%" or "P&L: -2.85%"
                        pnl_str = reasoning.split("P&L:")[1].strip().split("%")[0].replace("+", "")
                        pnl_pct = float(pnl_str)
                        pnl_quote = running_capital * (pnl_pct / 100)
                        running_capital += pnl_quote
                        
                        equity_curve.append({
                            "time": ts,
                            "value": round(running_capital, 2),
                            "action": action
                        })
                    except (ValueError, IndexError):
                        pass
                elif action == "BUY":
                    # Add entry point marker
                    equity_curve.append({
                        "time": ts,
                        "value": round(running_capital, 2),
                        "action": action
                    })
                    
        except Exception as e:
            return {"error": f"Failed to load trade history: {str(e)}"}
    
    return {
        "history": equity_curve,
        "stats": stats
    }

@router.get("/stats")
async def get_statistics(request: Request):
    """Get trading statistics summary."""
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    
    stats_file = Path(data_dir) / "trading" / "statistics.json"
    
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load stats: {str(e)}"}
    
    return {}
