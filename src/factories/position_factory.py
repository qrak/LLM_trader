from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.trading.dataclasses import Position, RiskAssessment
from src.logger.logger import Logger

class PositionFactory:
    """Factory for creating and updating Position objects."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def create_position(
        self,
        symbol: str,
        direction: str,
        confidence: str,
        risk_assessment: RiskAssessment,
        confluence_factors: tuple = (),
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Position:
        """Create a new Position instance."""
        market_conditions = market_conditions or {}

        return Position(
            entry_price=risk_assessment.entry_price,
            stop_loss=risk_assessment.stop_loss,
            take_profit=risk_assessment.take_profit,
            size=risk_assessment.quantity,
            entry_time=datetime.now(timezone.utc),
            confidence=confidence,
            direction=direction,
            symbol=symbol,
            confluence_factors=confluence_factors,
            entry_fee=risk_assessment.entry_fee,
            quote_amount=risk_assessment.quote_amount,
            size_pct=risk_assessment.size_pct,
            atr_at_entry=market_conditions.get('atr', 0.0),
            volatility_level=risk_assessment.volatility_level,
            sl_distance_pct=risk_assessment.sl_distance_pct,
            tp_distance_pct=risk_assessment.tp_distance_pct,
            rr_ratio_at_entry=risk_assessment.rr_ratio,
            adx_at_entry=market_conditions.get('adx', 0.0),
            rsi_at_entry=market_conditions.get('rsi', 50.0),
            trend_direction_at_entry=market_conditions.get('trend_direction', 'NEUTRAL'),
            macd_signal_at_entry=market_conditions.get('macd_signal', 'NEUTRAL'),
            bb_position_at_entry=market_conditions.get('bb_position', 'MIDDLE'),
            volume_state_at_entry=market_conditions.get('volume_state', 'NORMAL'),
            market_sentiment_at_entry=market_conditions.get('market_sentiment', 'NEUTRAL'),
            max_drawdown_pct=0.0,
            max_profit_pct=0.0
        )

    def create_updated_position(
        self,
        original_position: Position,
        new_stop_loss: float,
        new_take_profit: float
    ) -> Position:
        """Create a new Position instance with updated parameters."""
        # Create a new instance with updated values
        # We manually copy fields to ensure no side effects
        return Position(
            entry_price=original_position.entry_price,
            stop_loss=new_stop_loss,
            take_profit=new_take_profit,
            size=original_position.size,
            entry_time=original_position.entry_time,
            confidence=original_position.confidence,
            direction=original_position.direction,
            symbol=original_position.symbol,
            confluence_factors=original_position.confluence_factors,
            entry_fee=original_position.entry_fee,
            quote_amount=original_position.quote_amount,
            size_pct=original_position.size_pct,
            atr_at_entry=original_position.atr_at_entry,
            volatility_level=original_position.volatility_level,
            sl_distance_pct=original_position.sl_distance_pct,
            tp_distance_pct=original_position.tp_distance_pct,
            rr_ratio_at_entry=original_position.rr_ratio_at_entry,
            adx_at_entry=original_position.adx_at_entry,
            rsi_at_entry=original_position.rsi_at_entry,
            trend_direction_at_entry=original_position.trend_direction_at_entry,
            macd_signal_at_entry=original_position.macd_signal_at_entry,
            bb_position_at_entry=original_position.bb_position_at_entry,
            volume_state_at_entry=original_position.volume_state_at_entry,
            market_sentiment_at_entry=original_position.market_sentiment_at_entry,
            max_drawdown_pct=original_position.max_drawdown_pct,
            max_profit_pct=original_position.max_profit_pct
        )
