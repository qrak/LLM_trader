"""Extract trading signals from AI responses."""

import re
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING
from re import Pattern

if TYPE_CHECKING:
    from src.parsing.unified_parser import UnifiedParser

from src.logger.logger import Logger


class PositionExtractor:
    """Extracts trading signals, stop loss, take profit from AI responses."""
    
    def __init__(self, logger: Optional[Logger] = None, unified_parser: "UnifiedParser" = None):
        """Initialize the position extractor.
        
        Args:
            logger: Optional logger instance
            unified_parser: UnifiedParser for JSON extraction (DRY)
        """
        self.logger = logger
        self.unified_parser = unified_parser
        
        # Regex patterns for extracting trading information
        self.signal_pattern: Pattern = re.compile(
            r'signal["\s:]*\[?(BUY|SELL|HOLD|CLOSE|CLOSE_LONG|CLOSE_SHORT|UPDATE)\]?',
            re.IGNORECASE
        )
        self.confidence_pattern: Pattern = re.compile(
            r'confidence["\s:]*\[?(HIGH|MEDIUM|LOW)\]?',
            re.IGNORECASE
        )
        self.stop_loss_pattern: Pattern = re.compile(
            r'stop[_\s]?loss["\s:]*\$?([0-9,]+(?:\.[0-9]+)?)',
            re.IGNORECASE
        )
        self.take_profit_pattern: Pattern = re.compile(
            r'take[_\s]?profit["\s:]*\$?([0-9,]+(?:\.[0-9]+)?)',
            re.IGNORECASE
        )
        self.position_size_pattern: Pattern = re.compile(
            r'position[_\s]?size["\s:]*\[?([0-9]+(?:\.[0-9]+)?)\]?%?',
            re.IGNORECASE
        )
        self.reasoning_pattern: Pattern = re.compile(
            r'reasoning["\s:]*["\']?([^"\'}\]]+)["\']?',
            re.IGNORECASE
        )
    
    def extract_from_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract trading decision from JSON in response.
        
        Uses UnifiedParser.extract_json_block() for JSON extraction (DRY).
        
        Args:
            text: AI response text
            
        Returns:
            Extracted JSON trading decision or None
        """
        if not self.unified_parser:
            if self.logger:
                self.logger.warning("No UnifiedParser provided, cannot extract JSON")
            return None
        
        # Try unwrapping 'analysis' key first
        result = self.unified_parser.extract_json_block(text, unwrap_key='analysis')
        if result:
            return result
        
        # Try 'trading_decision' key
        result = self.unified_parser.extract_json_block(text, unwrap_key='trading_decision')
        if result:
            return result
        
        # Try 'decision' key
        result = self.unified_parser.extract_json_block(text, unwrap_key='decision')
        if result:
            return result
        
        # Try raw extraction without unwrapping
        return self.unified_parser.extract_json_block(text)
    
    def extract_trading_info(self, text: str) -> Tuple[str, str, Optional[float], Optional[float], Optional[float], str]:
        """Extract trading information from AI response.
        
        Args:
            text: AI response text
            
        Returns:
            Tuple of (signal, confidence, stop_loss, take_profit, position_size, reasoning)
        """
        # First try JSON extraction
        json_data = self.extract_from_json(text)
        if json_data:
            return self._extract_from_dict(json_data)
        
        # Fall back to regex extraction
        return self._extract_from_text(text)
    
    def _extract_from_dict(self, data: Dict[str, Any]) -> Tuple[str, str, Optional[float], Optional[float], Optional[float], str]:
        """Extract trading info from a dictionary.
        
        Args:
            data: Dictionary containing trading decision
            
        Returns:
            Tuple of (signal, confidence, stop_loss, take_profit, position_size, reasoning)
        """
        signal = str(data.get("signal", data.get("action", "HOLD"))).upper()
        
        # Confidence can be numeric (0-100) or string (HIGH/MEDIUM/LOW)
        confidence_raw = data.get("confidence", "MEDIUM")
        if isinstance(confidence_raw, (int, float)):
            # Convert numeric confidence to string
            confidence = self._numeric_to_confidence_string(confidence_raw)
        else:
            confidence = str(confidence_raw).upper()
        
        stop_loss = data.get("stop_loss")
        if stop_loss is not None:
            stop_loss = float(str(stop_loss).replace(",", "").replace("$", ""))
        
        take_profit = data.get("take_profit")
        if take_profit is not None:
            take_profit = float(str(take_profit).replace(",", "").replace("$", ""))
        
        position_size = data.get("position_size")
        if position_size is not None:
            position_size = float(str(position_size).replace("%", "")) / 100
        
        reasoning = str(data.get("reasoning", data.get("rationale", "")))
        
        return signal, confidence, stop_loss, take_profit, position_size, reasoning
    
    def _numeric_to_confidence_string(self, confidence: float) -> str:
        """Convert numeric confidence (0-100) to string (HIGH/MEDIUM/LOW).
        
        Args:
            confidence: Numeric confidence 0-100
            
        Returns:
            String confidence level
        """
        if confidence >= 70:
            return "HIGH"
        elif confidence >= 50:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _extract_from_text(self, text: str) -> Tuple[str, str, Optional[float], Optional[float], Optional[float], str]:
        """Extract trading info using regex patterns.
        
        Args:
            text: Raw text response
            
        Returns:
            Tuple of (signal, confidence, stop_loss, take_profit, position_size, reasoning)
        """
        # Extract signal
        signal_match = self.signal_pattern.search(text)
        signal = signal_match.group(1).upper() if signal_match else "HOLD"
        
        # Extract confidence
        confidence_match = self.confidence_pattern.search(text)
        confidence = confidence_match.group(1).upper() if confidence_match else "MEDIUM"
        
        # Extract stop loss
        stop_loss_match = self.stop_loss_pattern.search(text)
        stop_loss = float(stop_loss_match.group(1).replace(",", "")) if stop_loss_match else None
        
        # Extract take profit
        take_profit_match = self.take_profit_pattern.search(text)
        take_profit = float(take_profit_match.group(1).replace(",", "")) if take_profit_match else None
        
        # Extract position size (convert percentage to decimal)
        position_size_match = self.position_size_pattern.search(text)
        position_size = None
        if position_size_match:
            size = float(position_size_match.group(1))
            position_size = size / 100 if size > 1 else size
        
        # Extract reasoning (take first 200 chars of any found reasoning)
        reasoning_match = self.reasoning_pattern.search(text)
        reasoning = reasoning_match.group(1).strip()[:200] if reasoning_match else ""
        
        return signal, confidence, stop_loss, take_profit, position_size, reasoning
    
    def extract_position_size(self, text: str) -> Optional[float]:
        """Extract just the position size from text.
        
        Args:
            text: AI response text
            
        Returns:
            Position size as decimal (e.g., 0.02 for 2%)
        """
        match = self.position_size_pattern.search(text)
        if not match:
            return None
        
        try:
            size = float(match.group(1))
            return size / 100 if size > 1 else size
        except (ValueError, IndexError):
            return None
    
    def validate_signal(self, signal: str) -> bool:
        """Validate if signal is a recognized trading action.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if valid signal
        """
        valid_signals = {"BUY", "SELL", "HOLD", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT", "UPDATE"}
        return signal.upper() in valid_signals
    

