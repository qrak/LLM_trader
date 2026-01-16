"""Pure JSON I/O service for trading data persistence.

This service handles all file system operations for trading data without any business logic.
Follows Single Responsibility Principle by delegating calculations to other services.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from src.logger.logger import Logger
from src.utils.data_utils import serialize_for_json

if TYPE_CHECKING:
    from src.trading.dataclasses import Position, TradeDecision
    from src.trading.statistics_calculator import TradingStatistics


class PersistenceManager:
    """Pure persistence layer for trading data.
    
    Responsibilities:
    - Load/save positions, trade history, brain, statistics
    - No business logic (no P&L calculation, no insight extraction)
    """
    
    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    def __init__(self, logger: Logger, data_dir: str = "trading_data"):
        """Initialize trading persistence.
        
        Args:
            logger: Logger instance
            data_dir: Directory for trading data files
        """
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.positions_file = self.data_dir / "positions.json"
        self.history_file = self.data_dir / "trade_history.json"
        self.previous_response_file = self.data_dir / "previous_response.json"
        self.last_analysis_file = self.data_dir / "last_analysis.json"
        self.statistics_file = self.data_dir / "statistics.json"
    
    def save_position(self, position: Optional["Position"]) -> None:
        """Save current position to disk."""
        try:
            if position is None:
                if self.positions_file.exists():
                    self.positions_file.unlink()
                return
            
            data = {
                "entry_price": position.entry_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "size": position.size,
                "entry_time": position.entry_time.isoformat(),
                "confidence": position.confidence,
                "direction": position.direction,
                "symbol": position.symbol,
                "confluence_factors": [[name, score] for name, score in position.confluence_factors],
                "entry_fee": position.entry_fee,
                "size_pct": position.size_pct,
                "quote_amount": position.quote_amount,
                "atr_at_entry": position.atr_at_entry,
                "volatility_level": position.volatility_level,
                "sl_distance_pct": position.sl_distance_pct,
                "tp_distance_pct": position.tp_distance_pct,
                "rr_ratio_at_entry": position.rr_ratio_at_entry,
                "adx_at_entry": position.adx_at_entry,
                "rsi_at_entry": position.rsi_at_entry,
                "max_drawdown_pct": position.max_drawdown_pct,
                "max_profit_pct": position.max_profit_pct,
            }
            
            data = serialize_for_json(data)

            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug(f"Saved position: {position.direction} {position.symbol}")
        except Exception as e:
            self.logger.error(f"Error saving position: {e}")
    
    def load_position(self) -> Optional["Position"]:
        """Load current position from disk."""
        if not self.positions_file.exists():
            return None
        try:
            from src.trading.dataclasses import Position
            with open(self.positions_file, 'r') as f:
                data = json.load(f)
                cf_list = data.get("confluence_factors", [])
                cf_tuple = tuple((name, score) for name, score in cf_list)
                return Position(
                    entry_price=data["entry_price"],
                    stop_loss=data["stop_loss"],
                    take_profit=data["take_profit"],
                    size=data["size"],
                    entry_time=self._ensure_utc(datetime.fromisoformat(data["entry_time"])),
                    confidence=data.get("confidence", "MEDIUM"),
                    direction=data.get("direction", "LONG"),
                    symbol=data.get("symbol", "BTC/USDC"),
                    confluence_factors=cf_tuple,
                    entry_fee=data.get("entry_fee", 0.0),
                    quote_amount=data.get("quote_amount", 0.0),
                    size_pct=data.get("size_pct", 0.0),
                    atr_at_entry=data.get("atr_at_entry", 0.0),
                    volatility_level=data.get("volatility_level", "MEDIUM"),
                    sl_distance_pct=data.get("sl_distance_pct", 0.0),
                    tp_distance_pct=data.get("tp_distance_pct", 0.0),
                    rr_ratio_at_entry=data.get("rr_ratio_at_entry", 0.0),
                    adx_at_entry=data.get("adx_at_entry", 0.0),
                    rsi_at_entry=data.get("rsi_at_entry", 50.0),
                    max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
                    max_profit_pct=data.get("max_profit_pct", 0.0),
                )
        except Exception as e:
            self.logger.error(f"Error loading position: {e}")
            return None
    
    def save_trade_decision(self, decision: "TradeDecision") -> None:
        """Save a trade decision to history."""
        try:
            history = self.load_trade_history()
            
            decision_dict = decision.to_dict()
            sanitized_decision = serialize_for_json(decision_dict)
            history.append(sanitized_decision)
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            self.logger.info(f"Saved trade decision: {decision.action} @ ${decision.price:,.2f}")
        except Exception as e:
            self.logger.error(f"Error saving trade decision: {e}")
    
    def load_trade_history(self) -> List[Dict[str, Any]]:
        """Load full trade history."""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading trade history: {e}")
            return []
    
    def load_last_n_decisions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Load the last n trade decisions from history."""
        history = self.load_trade_history()
        valid_actions = {"BUY", "SELL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT"}
        
        filtered = [
            d for d in history
            if d.get("action", "").upper() in valid_actions
        ]
        
        filtered.sort(
            key=lambda x: self._ensure_utc(datetime.fromisoformat(x["timestamp"])),
            reverse=True
        )
        
        return filtered[:n]
    
    def get_entry_decision_for_position(self, entry_time: datetime) -> Optional["TradeDecision"]:
        """Retrieve the entry decision from trade history for a given position.
        
        Args:
            entry_time: The entry_time of the position to find
            
        Returns:
            TradeDecision with the original entry reasoning, or None if not found
        """
        try:
            from src.trading.dataclasses import TradeDecision
            history = self.load_trade_history()
            entry_actions = {"BUY", "SELL"}
            
            # Search for BUY/SELL action matching the entry_time
            for decision_dict in history:
                action = decision_dict.get("action", "")
                timestamp_str = decision_dict.get("timestamp", "")
                
                if action in entry_actions and timestamp_str:
                    decision_time = self._ensure_utc(datetime.fromisoformat(timestamp_str))
                    entry_time_utc = self._ensure_utc(entry_time)
                    
                    # Match by timestamp (allowing 1 second tolerance for floating point precision)
                    time_diff = abs((decision_time - entry_time_utc).total_seconds())
                    if time_diff < 1.0:
                        # Reconstruct TradeDecision from dictionary
                        return TradeDecision(
                            timestamp=decision_time,
                            symbol=decision_dict.get("symbol", "BTC/USDC"),
                            action=action,
                            confidence=decision_dict.get("confidence", "MEDIUM"),
                            price=decision_dict.get("price", 0.0),
                            stop_loss=decision_dict.get("stop_loss"),
                            take_profit=decision_dict.get("take_profit"),
                            position_size=decision_dict.get("position_size", 0.0),
                            quote_amount=decision_dict.get("quote_amount", 0.0),
                            quantity=decision_dict.get("quantity", 0.0),
                            fee=decision_dict.get("fee", 0.0),
                            reasoning=decision_dict.get("reasoning", "")
                        )
            
            self.logger.warning(f"Could not find entry decision for position at {entry_time}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving entry decision: {e}")
            return None
    
    def save_statistics(self, stats: "TradingStatistics") -> None:
        """Save trading statistics to disk."""
        try:
            with open(self.statistics_file, 'w') as f:
                json.dump(stats.to_dict(), f, indent=2)
            self.logger.debug(f"Saved statistics: {stats.total_trades} trades")
        except Exception as e:
            self.logger.error(f"Error saving statistics: {e}")
    
    def load_statistics(self) -> "TradingStatistics":
        """Load trading statistics from disk."""
        from src.trading.statistics_calculator import TradingStatistics
        if not self.statistics_file.exists():
            return TradingStatistics()
        try:
            with open(self.statistics_file, 'r') as f:
                data = json.load(f)
                return TradingStatistics.from_dict(data)
        except Exception as e:
            self.logger.error(f"Error loading statistics: {e}")
            return TradingStatistics()
    
    def save_previous_response(
        self, 
        response: str, 
        technical_data: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None
    ) -> None:
        """Save the previous AI response, technical indicator values, and prompt.
        
        Args:
            response: The AI response text
            technical_data: Dictionary of technical indicator values (RSI, MACD, ADX, etc.)
            prompt: The prompt that was sent to the AI
        """
        try:
            response_dict = {"text_analysis": response}
            
            if technical_data:
                serialized_data = serialize_for_json(technical_data)
                response_dict.update(serialized_data)
            
            data_to_save = {
                "response": response_dict,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add prompt if provided
            if prompt:
                data_to_save["prompt"] = prompt
            
            data_to_save = serialize_for_json(data_to_save)
            
            with open(self.previous_response_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
                
            self.logger.debug(f"Saved previous response with {len(technical_data) if technical_data else 0} indicators")
        except Exception as e:
            self.logger.error(f"Error saving previous response: {e}")
    
    def load_previous_response(self) -> Optional[Dict[str, Any]]:
        """Load the previous AI response and technical indicators.
        
        Returns:
            Dictionary with 'response' (str) and 'technical_indicators' (dict) keys,
            or None if file doesn't exist
        """
        if not self.previous_response_file.exists():
            return None
        
        try:
            with open(self.previous_response_file, 'r') as f:
                data = json.load(f)
                
                response_data = data.get("response", {})
                text_analysis = response_data.get("text_analysis", "")
                technical_indicators = {k: v for k, v in response_data.items() if k != "text_analysis"}
                
                return {
                    "response": text_analysis,
                    "technical_indicators": technical_indicators if technical_indicators else None,
                    "timestamp": data.get("timestamp")
                }
        except Exception as e:
            self.logger.error(f"Error loading previous response: {e}")
            return None
    
    def save_last_analysis_time(self, timestamp: Optional[datetime] = None) -> None:
        """Save the timestamp of the last successful analysis.
        
        Args:
            timestamp: Timestamp to save (defaults to now)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            
            with open(self.last_analysis_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp.isoformat()
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving last analysis time: {e}")
    
    def get_last_analysis_time(self) -> Optional[datetime]:
        """Get timestamp of last successful analysis."""
        if not self.last_analysis_file.exists():
            return None
        
        try:
            with open(self.last_analysis_file, 'r') as f:
                data = json.load(f)
                return self._ensure_utc(datetime.fromisoformat(data["timestamp"]))
        except Exception as e:
            self.logger.warning(f"Could not get last analysis time: {e}")
            return None
