from typing import Optional, Tuple, Pattern
import re

class PositionExtractor:
    def __init__(self) -> None:
        self.position_pattern: Pattern = re.compile(
            r'Position Size:[\s*]*\[?(\d+(?:\.\d+)?)]?%?(?:\s+of\s+portfolio)?',
            re.IGNORECASE
        )
        self.signal_pattern: Pattern = re.compile(r'Signal:[\s*]*\[?(CLOSE|BUY|SELL|HOLD)]?')
        self.confidence_pattern: Pattern = re.compile(r'Confidence:[\s*]*\[?(HIGH|MEDIUM|LOW)]?')
        self.stop_loss_pattern: Pattern = re.compile(r'Stop Loss:[\s*]*\$?([0-9,.]+)')
        self.take_profit_pattern: Pattern = re.compile(r'Take Profit:[\s*]*\$?([0-9,.]+)')

    def extract_position_size(self, text: str) -> Optional[float]:
        match = self.position_pattern.search(text)
        if not match:
            return None

        try:
            position_size = float(match.group(1))
            return position_size / 100
        except (ValueError, IndexError):
            return None

    def extract_trading_info(self, text: str) -> Tuple[str, str, Optional[float], Optional[float], Optional[float]]:
        signal = self.signal_pattern.search(text)
        confidence = self.confidence_pattern.search(text)
        stop_loss = self.stop_loss_pattern.search(text)
        take_profit = self.take_profit_pattern.search(text)
        position_size = self.extract_position_size(text)

        return (
            signal.group(1) if signal else "HOLD",
            confidence.group(1) if confidence else "MEDIUM",
            float(stop_loss.group(1).replace(',', '')) if stop_loss else None,
            float(take_profit.group(1).replace(',', '')) if take_profit else None,
            position_size
        )