"""Trading strategy that wraps analysis with position management."""

from datetime import datetime
from typing import Optional, Any, Dict, TYPE_CHECKING

from src.logger.logger import Logger
from .dataclasses import Position, TradeDecision
from .data_persistence import DataPersistence


if TYPE_CHECKING:
    from src.parsing.unified_parser import UnifiedParser


class TradingStrategy:
    """Manages trading positions and decision execution based on AI analysis."""
    
    def __init__(
        self,
        logger: Logger,
        data_persistence: DataPersistence,
        config: Any = None,
        position_extractor=None,
    ):
        """Initialize the trading strategy with DI pattern.
        
        Args:
            logger: Logger instance
            data_persistence: Data persistence for positions and history
            config: Configuration module
            position_extractor: PositionExtractor instance (injected from app.py)
        """
        if position_extractor is None:
            raise ValueError("position_extractor is required - must be injected from app.py")
        
        self.logger = logger
        self.persistence = data_persistence
        self.config = config
        self.extractor = position_extractor
        
        # Load any existing position
        self.current_position: Optional[Position] = self.persistence.load_position()
        
        if self.current_position:
            self.logger.info(
                f"Loaded existing position: {self.current_position.direction} "
                f"{self.current_position.symbol} @ ${self.current_position.entry_price:,.2f}"
            )
    
    async def check_position(self, current_price: float) -> Optional[str]:
        """Check if current position hit stop loss or take profit.
        
        Args:
            current_price: Current market price
            
        Returns:
            Reason for closing position if hit, else None
        """
        if not self.current_position:
            return None
        
        # Update live performance metrics (MAE/MFE)
        self.current_position.update_metrics(current_price)
        self.persistence.save_position(self.current_position)
        
        if self.current_position.is_stop_hit(current_price):
            await self.close_position("stop_loss", current_price)
            return "stop_loss"
        
        if self.current_position.is_target_hit(current_price):
            await self.close_position("take_profit", current_price)
            return "take_profit"
        
        return None
    
    async def close_position(
        self, 
        reason: str, 
        current_price: float,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> None:
        """Close the current position and update trading brain.
        
        Args:
            reason: Reason for closing (stop_loss, take_profit, signal)
            current_price: Current market price
            market_conditions: Optional market conditions for brain learning
        """
        if not self.current_position:
            return
        
        pnl = self.current_position.calculate_pnl(current_price)
        
        # Calculate closing fee
        closing_fee = self.current_position.calculate_closing_fee(
            current_price, 
            self.config.TRANSACTION_FEE_PERCENT
        )
        
        decision = TradeDecision(
            timestamp=datetime.now(),
            symbol=self.current_position.symbol,
            action=f"CLOSE_{self.current_position.direction}",
            confidence=self.current_position.confidence,
            price=current_price,
            stop_loss=self.current_position.stop_loss,
            take_profit=self.current_position.take_profit,
            position_size=self.current_position.size_pct,
            quote_amount=self.current_position.quote_amount,
            quantity=self.current_position.size,
            fee=closing_fee,
            reasoning=f"Position closed: {reason}. P&L: {pnl:+.2f}%. Fee: ${closing_fee:.4f}",
        )
        
        self.logger.info(
            f"Closing {self.current_position.direction} position ({reason}) "
            f"@ ${current_price:,.2f}, P&L: {pnl:+.2f}%, Fee: ${closing_fee:.4f}"
        )
        
        # Update trading brain with closed trade insights
        try:
            self.persistence.update_brain_from_closed_trade(
                position=self.current_position,
                close_price=current_price,
                close_reason=reason,
                entry_decision=None,  # Could be loaded from history if needed
                market_conditions=market_conditions
            )
        except Exception as e:
            self.logger.error(f"Error updating trading brain: {e}")
        
        self.persistence.save_trade_decision(decision)
        self.persistence.save_position(None)
        self.current_position = None
    
    async def process_analysis(self, analysis_result: dict, symbol: str) -> Optional[TradeDecision]:
        """Process AI analysis result and execute trading decision.
        
        Args:
            analysis_result: Result from AnalysisEngine.analyze_market()
            symbol: Trading symbol
            
        Returns:
            TradeDecision if action taken, else None
        """
        try:
            raw_response = analysis_result.get("raw_response", "")
            current_price = self._extract_price_from_result(analysis_result)
            
            if not raw_response:
                self.logger.warning("No response to process")
                return None
            
            if current_price <= 0:
                self.logger.error("Invalid current_price extracted, cannot process trade")
                return None
            
            # Extract trading info from response
            signal, confidence, stop_loss, take_profit, position_size, reasoning = \
                self.extractor.extract_trading_info(raw_response)
            
            self.logger.info(f"Extracted Signal: {signal}, Confidence: {confidence}")
            
            # Validate signal
            if not self.extractor.validate_signal(signal):
                self.logger.warning(f"Invalid signal: {signal}")
                return None
            
            # Extract market conditions for brain learning
            market_conditions = self._extract_market_conditions(analysis_result)
            
            # Extract confluence factors for brain learning (Feature 1)
            confluence_factors = self._extract_confluence_factors(analysis_result)
            
            # Handle existing position
            if self.current_position:
                return await self._handle_existing_position(
                    signal, confidence, stop_loss, take_profit, 
                    current_price, symbol, reasoning, market_conditions
                )
            
            # Handle new position
            if signal in ("BUY", "SELL"):
                return await self._open_new_position(
                    signal, confidence, stop_loss, take_profit,
                    position_size, current_price, symbol, reasoning,
                    confluence_factors, market_conditions
                )
            
            # HOLD or no action
            if reasoning:
                self.logger.info(f"No action taken. Signal: {signal}. Reasoning: {reasoning[:200]}")
            else:
                self.logger.info(f"No action taken. Signal: {signal}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing analysis: {e}")
            return None
    
    async def _handle_existing_position(
        self,
        signal: str,
        confidence: str,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        current_price: float,
        symbol: str,
        reasoning: str,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradeDecision]:
        """Handle trading decision when position exists.
        
        Args:
            signal: Trading signal
            confidence: Confidence level
            stop_loss: New stop loss (for update)
            take_profit: New take profit (for update)
            current_price: Current price
            symbol: Trading symbol
            reasoning: AI reasoning
            market_conditions: Market state for brain learning
            
        Returns:
            TradeDecision if action taken
        """
        if signal == "CLOSE" or signal.startswith("CLOSE_"):
            self.logger.info("Closing position based on analysis signal...")
            await self.close_position("analysis_signal", current_price, market_conditions)
            return TradeDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action="CLOSE",
                confidence=confidence,
                price=current_price,
                fee=0.0, # Fee handled in close_position, this is just the signal return
                reasoning=reasoning,
            )
        
        # Update stop loss / take profit if provided
        updated = self._update_position_parameters(stop_loss, take_profit)
        
        if updated:
            decision = TradeDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action="UPDATE",
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                fee=0.0, # No fee for updates
                reasoning=f"Updated position parameters. {reasoning}",
            )
            self.persistence.save_trade_decision(decision)
            self.logger.info(f"Position updated: New SL=${stop_loss:,.2f}, TP=${take_profit:,.2f}")
            return decision
        
        return None
    
    async def _open_new_position(
        self,
        signal: str,
        confidence: str,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        position_size: Optional[float],
        current_price: float,
        symbol: str,
        reasoning: str,
        confluence_factors: tuple = (),
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> TradeDecision:
        """Open a new trading position with dynamic parameter calculation.

        Args:
            signal: BUY or SELL
            confidence: Confidence level
            stop_loss: Stop loss price (from AI)
            take_profit: Take profit price (from AI)
            position_size: Position size as decimal (e.g., 0.02 = 2% of capital)
            current_price: Entry price
            symbol: Trading symbol
            reasoning: AI reasoning
            confluence_factors: Tuple of (factor_name, score) pairs for brain learning
            market_conditions: Market state dict with ATR, volatility_level, etc.

        Returns:
            TradeDecision for the new position
        """
        direction = "LONG" if signal == "BUY" else "SHORT"
        market_conditions = market_conditions or {}

        # Extract ATR for dynamic calculation (fallback to 2% of price)
        atr = market_conditions.get("atr", current_price * 0.02)
        atr_pct = market_conditions.get("atr_percentage", (atr / current_price) * 100)
        adx = market_conditions.get("adx", 0.0)

        # Determine volatility level for Brain learning
        if atr_pct > 3:
            volatility_level = "HIGH"
        elif atr_pct < 1.5:
            volatility_level = "LOW"
        else:
            volatility_level = "MEDIUM"

        # Dynamic default calculation based on ATR (not hardcoded 2%/4%)
        # Use 2x ATR for SL, 4x ATR for TP (2:1 R/R default)
        dynamic_sl_distance = atr * 2
        dynamic_tp_distance = atr * 4

        if direction == "LONG":
            dynamic_sl = current_price - dynamic_sl_distance
            dynamic_tp = current_price + dynamic_tp_distance
        else:  # SHORT
            dynamic_sl = current_price + dynamic_sl_distance
            dynamic_tp = current_price - dynamic_tp_distance

        # Use AI-provided values if valid, otherwise use dynamic defaults
        if stop_loss and stop_loss > 0:
            final_sl = stop_loss
            self.logger.debug(f"Using AI-provided SL: ${final_sl:,.2f}")
        else:
            final_sl = dynamic_sl
            self.logger.info(f"Using dynamic SL (2x ATR): ${final_sl:,.2f}")

        if take_profit and take_profit > 0:
            final_tp = take_profit
            self.logger.debug(f"Using AI-provided TP: ${final_tp:,.2f}")
        else:
            final_tp = dynamic_tp
            self.logger.info(f"Using dynamic TP (4x ATR): ${final_tp:,.2f}")

        # Circuit Breaker: Validate and clamp extreme values
        sl_distance_raw = abs(current_price - final_sl) / current_price
        tp_distance_raw = abs(final_tp - current_price) / current_price

        # Clamp SL: min 0.5%, max 10%
        if sl_distance_raw > 0.10:
            self.logger.warning(f"SL distance {sl_distance_raw:.1%} exceeds 10% max, clamping")
            if direction == "LONG":
                final_sl = current_price * 0.90
            else:
                final_sl = current_price * 1.10
        elif sl_distance_raw < 0.005:
            self.logger.warning(f"SL distance {sl_distance_raw:.1%} below 0.5% min, expanding")
            if direction == "LONG":
                final_sl = current_price * 0.995
            else:
                final_sl = current_price * 1.005

        # Validate SL/TP direction makes sense
        if direction == "LONG":
            if final_sl >= current_price:
                self.logger.warning(f"Invalid SL for LONG ({final_sl} >= {current_price}), using dynamic")
                final_sl = dynamic_sl
            if final_tp <= current_price:
                self.logger.warning(f"Invalid TP for LONG ({final_tp} <= {current_price}), using dynamic")
                final_tp = dynamic_tp
        else:  # SHORT
            if final_sl <= current_price:
                self.logger.warning(f"Invalid SL for SHORT ({final_sl} <= {current_price}), using dynamic")
                final_sl = dynamic_sl
            if final_tp >= current_price:
                self.logger.warning(f"Invalid TP for SHORT ({final_tp} >= {current_price}), using dynamic")
                final_tp = dynamic_tp

        # Calculate final distances for Brain learning
        sl_distance_pct = abs(current_price - final_sl) / current_price
        tp_distance_pct = abs(final_tp - current_price) / current_price
        rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0

        # Position sizing (use AI value or confidence-based default)
        if position_size and position_size > 0:
            final_size_pct = position_size
        else:
            # Dynamic sizing based on confidence
            confidence_map = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}
            final_size_pct = confidence_map.get(confidence.upper(), 0.02)
            self.logger.info(f"Using confidence-based size: {final_size_pct*100:.1f}%")

        # Calculate quantity based on capital and size percentage
        capital = self.config.DEMO_QUOTE_CAPITAL
        allocation = capital * final_size_pct
        quantity = allocation / current_price

        # Calculate entry fee for limit order (based on allocated capital, not quantity)
        entry_fee = allocation * self.config.TRANSACTION_FEE_PERCENT

        self.logger.info(
            f"Position sizing: Capital=${capital:,.2f}, Size={final_size_pct*100:.2f}%, "
            f"Allocation=${allocation:,.2f}, Quantity={quantity:.6f}"
        )
        self.logger.info(
            f"Risk metrics: SL={sl_distance_pct*100:.2f}%, TP={tp_distance_pct*100:.2f}%, R/R={rr_ratio:.2f}"
        )

        # Create position with confluence factors and market conditions for brain learning
        self.current_position = Position(
            entry_price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
            size=quantity,
            entry_time=datetime.now(),
            confidence=confidence,
            direction=direction,
            symbol=symbol,
            confluence_factors=confluence_factors,
            entry_fee=entry_fee,
            quote_amount=allocation,
            size_pct=final_size_pct,
            atr_at_entry=atr,
            volatility_level=volatility_level,
            sl_distance_pct=sl_distance_pct,
            tp_distance_pct=tp_distance_pct,
            rr_ratio_at_entry=rr_ratio,
            adx_at_entry=adx,
        )
        self.persistence.save_position(self.current_position)
        self.logger.info(
            f"Opened {direction} position @ ${current_price:,.2f} "
            f"(SL: ${final_sl:,.2f}, TP: ${final_tp:,.2f}, Qty: {quantity:.6f}, Fee: ${entry_fee:.4f})"
        )

        # Create and save decision (store size_pct for history context)
        decision = TradeDecision(
            timestamp=datetime.now(),
            symbol=symbol,
            action=signal,
            confidence=confidence,
            price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
            position_size=final_size_pct,
            quote_amount=allocation,
            quantity=quantity,
            fee=entry_fee,
            reasoning=reasoning,
        )

        self.persistence.save_trade_decision(decision)

        return decision
    
    def _update_position_parameters(
        self,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> bool:
        """Update position stop loss and take profit.
        
        Args:
            stop_loss: New stop loss
            take_profit: New take profit
            
        Returns:
            True if anything was updated
        """
        if not self.current_position:
            return False
        
        updated = False
        new_sl = self.current_position.stop_loss
        new_tp = self.current_position.take_profit
        
        if stop_loss and stop_loss != self.current_position.stop_loss:
            new_sl = stop_loss
            self.logger.info(f"Updated Stop Loss: ${stop_loss:,.2f}")
            updated = True
        
        if take_profit and take_profit != self.current_position.take_profit:
            new_tp = take_profit
            self.logger.info(f"Updated Take Profit: ${take_profit:,.2f}")
            updated = True
        
        if updated:
            # Create new position with updated values (frozen dataclass)
            self.current_position = Position(
                entry_price=self.current_position.entry_price,
                stop_loss=new_sl,
                take_profit=new_tp,
                size=self.current_position.size,
                quote_amount=self.current_position.quote_amount,
                entry_time=self.current_position.entry_time,
                confidence=self.current_position.confidence,
                direction=self.current_position.direction,
                symbol=self.current_position.symbol,
                adx_at_entry=self.current_position.adx_at_entry,
                max_drawdown_pct=self.current_position.max_drawdown_pct,
                max_profit_pct=self.current_position.max_profit_pct,
            )
            self.persistence.save_position(self.current_position)
        
        return updated
    
    def _extract_price_from_result(self, result: dict) -> float:
        """Extract current price from analysis result.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            Current price
        """
        # Try different possible locations for price
        if "current_price" in result:
            return float(result["current_price"])
        
        if "context" in result and result["context"] is not None:
            return float(result["context"].current_price)
        
        # Default fallback
        if self.logger:
            self.logger.warning(f"Could not extract price from result keys: {list(result.keys())}")
        return 0.0
    
    def _extract_market_conditions(self, result: dict) -> Dict[str, Any]:
        """Extract market conditions from analysis result for brain learning.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            Dictionary with trend_direction, adx, volatility, etc.
        """
        conditions = {}
        
        try:
            # Try to extract from parsed JSON analysis
            parsed = result.get("parsed_json", {})
            analysis = parsed.get("analysis", {})
            
            # Get trend info
            trend = analysis.get("trend", {})
            if trend:
                conditions["trend_direction"] = trend.get("direction", "NEUTRAL")
                conditions["trend_strength"] = trend.get("strength", 50)
            
            # Try to get context data for more details
            context = result.get("context")
            if context and context.technical_data:
                tech_data = context.technical_data
                conditions["adx"] = tech_data.get("adx", 0)
                conditions["rsi"] = tech_data.get("rsi", 50)
                conditions["choppiness"] = tech_data.get("choppiness", None)

                # Extract ATR for dynamic SL/TP calculation
                conditions["atr"] = tech_data.get("atr", 0)
                atr_pct = tech_data.get("atr_percentage", 0)
                conditions["atr_percentage"] = atr_pct

                # Determine volatility from ATR or other indicators
                if atr_pct > 3:
                    conditions["volatility"] = "HIGH"
                elif atr_pct < 1.5:
                    conditions["volatility"] = "LOW"
                else:
                    conditions["volatility"] = "MEDIUM"
            
            # Fallback: try to extract from raw response keywords
            raw_response = result.get("raw_response", "").lower()
            if not conditions.get("trend_direction"):
                if "bullish" in raw_response or "uptrend" in raw_response:
                    conditions["trend_direction"] = "BULLISH"
                elif "bearish" in raw_response or "downtrend" in raw_response:
                    conditions["trend_direction"] = "BEARISH"
                else:
                    conditions["trend_direction"] = "NEUTRAL"
                    
        except Exception as e:
            self.logger.warning(f"Could not extract market conditions: {e}")
        
        return conditions
    
    def _extract_confluence_factors(self, result: dict) -> tuple:
        """Extract confluence factors from analysis result for brain learning.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            Tuple of (factor_name, score) pairs
        """
        factors = []
        
        try:
            # Try to extract from parsed JSON analysis
            parsed = result.get("parsed_json", {})
            analysis = parsed.get("analysis", {})
            confluence_factors = analysis.get("confluence_factors", {})
            
            if isinstance(confluence_factors, dict):
                for factor_name, score in confluence_factors.items():
                    try:
                        # Ensure score is numeric and in valid range
                        score_value = float(score)
                        if 0 <= score_value <= 100:
                            factors.append((factor_name, score_value))
                    except (ValueError, TypeError):
                        pass
                        
        except Exception as e:
            self.logger.warning(f"Could not extract confluence factors: {e}")
        
        return tuple(factors)
    
    def get_position_context(self, current_price: Optional[float] = None) -> str:
        """Get formatted context about current position for prompts.
        
        Args:
            current_price: Current market price for P&L calculation
            
        Returns:
            Formatted position context string
        """
        if not self.current_position:
            return "CURRENT POSITION: None"
        
        pos = self.current_position
        duration = datetime.now() - pos.entry_time
        hours = duration.total_seconds() / 3600
        
        context_lines = [
            "CURRENT POSITION:",
            f"- Direction: {pos.direction}",
            f"- Symbol: {pos.symbol}",
            f"- Entry Price: ${pos.entry_price:,.2f}",
            f"- Stop Loss: ${pos.stop_loss:,.2f}",
            f"- Take Profit: ${pos.take_profit:,.2f}",
            f"- Size: {pos.size * 100:.1f}%",
            f"- Duration: {hours:.1f} hours",
            f"- Confidence: {pos.confidence}",
        ]
        
        if current_price and current_price > 0:
            pnl_pct = pos.calculate_pnl(current_price)
            pnl_usdt = (current_price - pos.entry_price) * pos.size if pos.direction == 'LONG' else (pos.entry_price - current_price) * pos.size
            context_lines.append(f"- Unrealized P&L: {pnl_pct:+.2f}% (${pnl_usdt:+,.2f} USDT)")
        
        return "\n".join(context_lines)
