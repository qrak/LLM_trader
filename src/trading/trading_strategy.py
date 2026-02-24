"""Trading strategy that wraps analysis with position management."""

from datetime import datetime, timezone
from typing import Optional, Any, Dict, TYPE_CHECKING

from src.logger.logger import Logger
from src.contracts.risk_contract import RiskManagerProtocol
from .data_models import Position, TradeDecision
from .brain import TradingBrainService
from .statistics import TradingStatisticsService
from .memory import TradingMemoryService

if TYPE_CHECKING:
    from src.managers.persistence_manager import PersistenceManager


class TradingStrategy:
    """Manages trading positions and decision execution based on AI analysis."""

    def __init__(
        self,
        logger: Logger,
        persistence: "PersistenceManager",
        brain_service: TradingBrainService,
        statistics_service: TradingStatisticsService,
        memory_service: TradingMemoryService,
        risk_manager: RiskManagerProtocol,
        config: Any = None,
        position_extractor=None,
        position_factory=None,
    ):
        """Initialize the trading strategy with DI pattern.

        Args:
            logger: Logger instance
            persistence: Persistence service for loading/saving data
            brain_service: Brain service for learning and insights
            statistics_service: Statistics service for performance metrics
            memory_service: Memory service for recent decision context
            risk_manager: Risk Manager for position sizing and SL/TP
            config: Configuration module
            position_extractor: PositionExtractor instance (injected from app.py)
            position_factory: PositionFactory instance (injected from start.py)
        """
        self.logger = logger
        self.persistence = persistence
        self.brain_service = brain_service
        self.statistics_service = statistics_service
        self.memory_service = memory_service
        self.risk_manager = risk_manager
        self.config = config
        self.extractor = position_extractor
        self.position_factory = position_factory

        # Load any existing position
        self.current_position: Optional[Position] = self.persistence.load_position()

        if self.current_position:
            self.logger.info("Loaded existing position: %s %s @ $%s", self.current_position.direction, self.current_position.symbol, f"{self.current_position.entry_price:,.2f}")

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
        await self.persistence.async_save_position(self.current_position)

        if self.current_position.is_stop_hit(current_price):
            conditions = self._build_conditions_from_position(self.current_position)
            await self.close_position("stop_loss", current_price, conditions)
            return "stop_loss"

        if self.current_position.is_target_hit(current_price):
            conditions = self._build_conditions_from_position(self.current_position)
            await self.close_position("take_profit", current_price, conditions)
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
            timestamp=datetime.now(timezone.utc),
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

        self.logger.info("Closing %s position (%s) @ $%s, P&L: %s%%, Fee: $%.4f", self.current_position.direction, reason, f"{current_price:,.2f}", f"{pnl:+.2f}", closing_fee)

        # Retrieve entry decision from trade history for brain learning
        entry_decision = None
        try:
            entry_decision = self.persistence.get_entry_decision_for_position(
                self.current_position.entry_time
            )
            if entry_decision:
                reasoning_preview = entry_decision.reasoning[:500] if entry_decision.reasoning else "(no reasoning)"
                self.logger.debug("Retrieved entry decision with reasoning: %s...", reasoning_preview)
            else:
                self.logger.warning("Could not retrieve entry decision from trade history")
        except Exception as e:
            self.logger.error("Error retrieving entry decision: %s", e)

        # Update trading brain with closed trade insights
        try:
            self.brain_service.update_from_closed_trade(
                position=self.current_position,
                close_price=current_price,
                close_reason=reason,
                entry_decision=entry_decision,
                market_conditions=market_conditions
            )
        except Exception as e:
            self.logger.error("Error updating trading brain: %s", e)
        # Save close decision FIRST so statistics include this trade
        await self.persistence.async_save_trade_decision(decision)

        # Recalculate performance statistics (Sharpe, Sortino, drawdown, etc.)
        try:
            self.statistics_service.recalculate(self.config.DEMO_QUOTE_CAPITAL)
        except Exception as e:
            self.logger.error("Error recalculating statistics: %s", e)
        await self.persistence.async_save_position(None)
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

            self.logger.info("Extracted Signal: %s, Confidence: %s", signal, confidence)

            # Validate signal
            if not self.extractor.validate_signal(signal):
                self.logger.warning("Invalid signal: %s", signal)
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
                self.logger.info("No action taken. Signal: %s. Reasoning: %s", signal, reasoning[:200])
            else:
                self.logger.info("No action taken. Signal: %s", signal)
            return None

        except Exception as e:
            self.logger.error("Error processing analysis: %s", e)
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
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                action="CLOSE",
                confidence=confidence,
                price=current_price,
                fee=0.0,
                reasoning=reasoning,
            )

        old_sl = self.current_position.stop_loss
        old_tp = self.current_position.take_profit

        updated = await self._update_position_parameters(stop_loss, take_profit)

        if updated:
            try:
                current_pnl = self.current_position.calculate_pnl(current_price)
                self.brain_service.track_position_update(
                    position=self.current_position,
                    old_sl=old_sl,
                    old_tp=old_tp,
                    new_sl=stop_loss if stop_loss else old_sl,
                    new_tp=take_profit if take_profit else old_tp,
                    current_price=current_price,
                    current_pnl_pct=current_pnl,
                    market_conditions=market_conditions
                )
            except Exception as e:
                self.logger.warning("Failed to track position update: %s", e)

            decision = TradeDecision(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                action="UPDATE",
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                fee=0.0,
                reasoning=f"Updated position parameters. {reasoning}",
            )
            await self.persistence.async_save_trade_decision(decision)
            self.logger.info("Position updated: New SL=$%s, TP=$%s", f"{stop_loss:,.2f}", f"{take_profit:,.2f}")
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
        """Open a new trading position with dynamic parameter calculation."""
        direction = "LONG" if signal == "BUY" else "SHORT"
        market_conditions = market_conditions or {}

        # Calculate quantity based on CURRENT capital (not initial)
        capital = self.statistics_service.get_current_capital(self.config.DEMO_QUOTE_CAPITAL)

        # Delegate Risk Calculation to RiskManager
        risk_assessment = self.risk_manager.calculate_entry_parameters(
            signal=signal,
            current_price=current_price,
            capital=capital,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            market_conditions=market_conditions
        )

        final_sl = risk_assessment.stop_loss
        final_tp = risk_assessment.take_profit
        final_size_pct = risk_assessment.size_pct
        quantity = risk_assessment.quantity
        entry_fee = risk_assessment.entry_fee
        sl_distance_pct = risk_assessment.sl_distance_pct
        tp_distance_pct = risk_assessment.tp_distance_pct
        rr_ratio = risk_assessment.rr_ratio

        self.logger.info("Position sizing: Capital=$%s, Size=%.2f%%, Allocation=$%s, Quantity=%.6f", f"{capital:,.2f}", final_size_pct * 100, f"{risk_assessment.quote_amount:,.2f}", quantity)
        self.logger.info("Risk metrics: SL=%.2f%%, TP=%.2f%%, R/R=%.2f", sl_distance_pct * 100, tp_distance_pct * 100, rr_ratio)

        # Create position using Factory
        self.current_position = self.position_factory.create_position(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            risk_assessment=risk_assessment,
            confluence_factors=confluence_factors,
            market_conditions=market_conditions
        )

        await self.persistence.async_save_position(self.current_position)
        self.logger.info("Opened %s position @ $%s (SL: $%s, TP: $%s, Qty: %.6f, Fee: $%.4f)", direction, f"{current_price:,.2f}", f"{final_sl:,.2f}", f"{final_tp:,.2f}", quantity, entry_fee)

        # Create and save decision (store size_pct for history context)
        decision = TradeDecision(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            action=signal,
            confidence=confidence,
            price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
            position_size=final_size_pct,
            quote_amount=risk_assessment.quote_amount,
            quantity=quantity,
            fee=entry_fee,
            reasoning=reasoning,
        )

        await self.persistence.async_save_trade_decision(decision)

        return decision

    async def _update_position_parameters(
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
            direction = self.current_position.direction
            old_sl = self.current_position.stop_loss
            # FULL AI AUTONOMY: Allow AI to move stop loss in any direction
            if direction == "LONG" and stop_loss < old_sl:
                self.logger.info(
                    "AI Widening Stop Loss for LONG: $%.2f -> $%.2f (Risk Increased)",
                    old_sl, stop_loss
                )
                new_sl = stop_loss
                updated = True
            elif direction == "SHORT" and stop_loss > old_sl:
                self.logger.info(
                    "AI Widening Stop Loss for SHORT: $%.2f -> $%.2f (Risk Increased)",
                    old_sl, stop_loss
                )
                new_sl = stop_loss
                updated = True
            else:
                new_sl = stop_loss
                self.logger.info("Updated Stop Loss: $%s", f"{stop_loss:,.2f}")
                updated = True

        if take_profit and take_profit != self.current_position.take_profit:
            new_tp = take_profit
            self.logger.info("Updated Take Profit: $%s", f"{take_profit:,.2f}")
            updated = True

        if updated:
            # Create new position with updated values using factory
            self.current_position = self.position_factory.create_updated_position(
                original_position=self.current_position,
                new_stop_loss=new_sl,
                new_take_profit=new_tp
            )
            await self.persistence.async_save_position(self.current_position)

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
            self.logger.warning("Could not extract price from result keys: %s", list(result.keys()))
        return 0.0

    @staticmethod
    def _build_conditions_from_position(position: Position) -> Dict[str, Any]:
        """Reconstruct market conditions from Position's stored entry fields.

        Used when closing via SL/TP hit where no fresh analysis is available.
        """
        rsi = position.rsi_at_entry
        if rsi > 70:
            rsi_level = "OVERBOUGHT"
        elif rsi > 60:
            rsi_level = "STRONG"
        elif rsi < 30:
            rsi_level = "OVERSOLD"
        elif rsi < 40:
            rsi_level = "WEAK"
        else:
            rsi_level = "NEUTRAL"

        return {
            "trend_direction": position.trend_direction_at_entry,
            "adx": position.adx_at_entry,
            "rsi": rsi,
            "rsi_level": rsi_level,
            "volatility": position.volatility_level,
            "macd_signal": position.macd_signal_at_entry,
            "bb_position": position.bb_position_at_entry,
            "volume_state": position.volume_state_at_entry,
            "market_sentiment": position.market_sentiment_at_entry,
        }

    def _extract_market_conditions(self, result: dict) -> Dict[str, Any]:
        """Extract market conditions from analysis result for brain learning.

        Args:
            result: Analysis result dictionary

        Returns:
            Dictionary with trend_direction, adx, volatility, etc.
        """
        conditions = {}

        try:
            # Extract from analysis dict (result has 'analysis' at top level, not under 'parsed_json')
            analysis = result.get("analysis", {})

            # Get trend info
            trend = analysis.get("trend", {})
            if trend:
                conditions["trend_direction"] = trend.get("direction", "NEUTRAL")
                conditions["trend_strength"] = trend.get("strength_4h", trend.get("strength", 50))
                conditions["timeframe_alignment"] = trend.get("timeframe_alignment")

            # Get technical data (result has 'technical_data' at top level, not under 'context')
            tech_data = result.get("technical_data", {})
            if tech_data:
                conditions["adx"] = tech_data.get("adx", 0)
                conditions["rsi"] = tech_data.get("rsi", 50)
                # Derive RSI Level
                rsi_val = conditions["rsi"]
                if rsi_val > 70:
                    conditions["rsi_level"] = "OVERBOUGHT"
                elif rsi_val > 60:
                    conditions["rsi_level"] = "STRONG"
                elif rsi_val < 30:
                    conditions["rsi_level"] = "OVERSOLD"
                elif rsi_val < 40:
                    conditions["rsi_level"] = "WEAK"
                else:
                    conditions["rsi_level"] = "NEUTRAL"
                    
                conditions["choppiness"] = tech_data.get("choppiness", None)
                
                # MACD
                macd = tech_data.get("macd", {})
                if isinstance(macd, dict):
                    conditions["macd_signal"] = macd.get("signal", "NEUTRAL")
                
                # Bollinger Bands
                bb = tech_data.get("bollinger_bands", {})
                if isinstance(bb, dict):
                    pct_b = bb.get("percent_b", 0.5)
                    if pct_b > 0.95:
                        conditions["bb_position"] = "UPPER"
                    elif pct_b < 0.05:
                        conditions["bb_position"] = "LOWER"
                    else:
                        conditions["bb_position"] = "MIDDLE"

                # Volume
                vol_data = tech_data.get("volume", {})
                if isinstance(vol_data, dict):
                    conditions["volume_state"] = vol_data.get("state", "NORMAL")

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
            
            # Market Sentiment
            conditions["market_sentiment"] = result.get("context", {}).get("market_sentiment", "NEUTRAL") if "context" in result else "NEUTRAL"
            conditions["fear_greed_index"] = result.get("context", {}).get("fear_greed_index", 50) if "context" in result else 50

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
            self.logger.warning("Could not extract market conditions: %s", e)
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
            # Extract from analysis dict (result has 'analysis' at top level, not under 'parsed_json')
            analysis = result.get("analysis", {})
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
            self.logger.warning("Could not extract confluence factors: %s", e)
        return tuple(factors)

    def get_position_context(self, current_price: Optional[float] = None) -> str:
        """Get formatted context about current position for prompts.

        Args:
            current_price: Current market price for P&L calculation

        Returns:
            Formatted position context string with capital status
        """
        capital = self.statistics_service.get_current_capital(self.config.DEMO_QUOTE_CAPITAL)
        currency = self.config.QUOTE_CURRENCY
        if not self.current_position:
            return (
                f"## Capital Status\n"
                f"- Total Capital: ${capital:,.2f} {currency}\n"
                f"- Available: ${capital:,.2f} (100%)\n\n"
                f"CURRENT POSITION: None"
            )
        pos = self.current_position
        # Ensure both datetimes are timezone-aware for subtraction
        now = datetime.now(timezone.utc)
        entry_time = pos.entry_time
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        duration = now - entry_time
        hours = duration.total_seconds() / 3600
        allocated = pos.quote_amount
        available = capital - allocated
        allocation_pct = (allocated / capital) * 100 if capital > 0 else 0
        context_lines = [
            "## Capital Status",
            f"- Total Capital: ${capital:,.2f} {currency}",
            f"- Allocated: ${allocated:,.2f} ({allocation_pct:.1f}%)",
            f"- Available: ${available:,.2f} ({100 - allocation_pct:.1f}%)",
            "",
            "## Current Position",
            f"- Direction: {pos.direction}",
            f"- Symbol: {pos.symbol}",
            f"- Entry Price: ${pos.entry_price:,.2f}",
        ]
        if current_price and current_price > 0:
            context_lines.append(f"- Current Price: ${current_price:,.2f}")
        context_lines.extend([
            f"- Stop Loss: ${pos.stop_loss:,.2f}",
            f"- Take Profit: ${pos.take_profit:,.2f}",
            f"- Position Size: {pos.size_pct * 100:.2f}%",
            f"- Quantity: {pos.size:.6f}",
            f"- Entry Fee: ${pos.entry_fee:.4f}",
            f"- Duration: {hours:.1f} hours",
            f"- Confidence: {pos.confidence}",
        ])
        if current_price and current_price > 0:
            pnl_pct = pos.calculate_pnl(current_price)
            pnl_quote = (current_price - pos.entry_price) * pos.size if pos.direction == 'LONG' else (pos.entry_price - current_price) * pos.size
            context_lines.append(f"- Unrealized P&L: {pnl_pct:+.2f}% (${pnl_quote:+,.2f} {currency})")
        return "\n".join(context_lines)
