from datetime import datetime
from typing import Optional

from core.market_analyzer import MarketAnalyzer
from utils.dataclass import Position, TradeDecision, TimeframeConfig
from utils.position_extractor import PositionExtractor


class TradingStrategy(MarketAnalyzer):
    def __init__(self, logger) -> None:
        super().__init__(logger)
        self.interval: int = TimeframeConfig.get_seconds(self.timeframe)
        self.current_position: Optional[Position] = self.data_persistence.load_position()
        self.extractor = PositionExtractor()

    async def close(self) -> None:
        await super().close()

    async def check_position(self, current_price: float) -> None:
        if not self.current_position:
            return

        if self.current_position.direction == 'LONG':
            if current_price <= self.current_position.stop_loss:
                await self.close_position("stop_loss")
            elif current_price >= self.current_position.take_profit:
                await self.close_position("take_profit")
        else:
            if current_price >= self.current_position.stop_loss:
                await self.close_position("stop_loss")
            elif current_price <= self.current_position.take_profit:
                await self.close_position("take_profit")


    async def close_position(self, reason: str) -> None:
        if not self.current_position:
            return

        current_price = self.periods['3D'].data[-1].close
        position_size = self.current_position.size
        confidence = self.current_position.confidence if self.current_position.confidence else "HIGH"

        decision = TradeDecision(
            timestamp=datetime.now(),
            action=f"CLOSE_{self.current_position.direction}",
            price=current_price,
            confidence=confidence,
            stop_loss=self.current_position.stop_loss,
            take_profit=self.current_position.take_profit,
            position_size=position_size,
            reasoning=f"Position closed: {reason}"
        )

        self.logger.info(
            f"Closing {self.current_position.direction} position ({reason}) at {current_price:.2f}"
        )
        self.data_persistence.save_trade_decision(decision)
        self.data_persistence.save_position(None)
        self.current_position = None

    def _should_close_position(self, signal: str) -> bool:
        return self.current_position and signal == "CLOSE"

    def _update_position_parameters(self, stop_loss: Optional[float],
                                    take_profit: Optional[float]) -> None:
        if not self.current_position:
            return

        updated = False
        if stop_loss and stop_loss != self.current_position.stop_loss:
            self.current_position.stop_loss = stop_loss
            self.logger.info(f"Updated Stop Loss: {stop_loss:.2f}")
            updated = True

        if take_profit and take_profit != self.current_position.take_profit:
            self.current_position.take_profit = take_profit
            self.logger.info(f"Updated Take Profit: {take_profit:.2f}")
            updated = True

        if updated:
            self.data_persistence.save_position(self.current_position)

    async def _open_new_position(
            self,
            signal: str,
            current_price: float,
            confidence: str,
            stop_loss: Optional[float],
            take_profit: Optional[float],
            position_size: Optional[float] = None
    ) -> None:
        if signal == "BUY":
            direction = "LONG"
            default_sl = current_price * 0.98
            default_tp = current_price * 1.04
        elif signal == "SELL":
            direction = "SHORT"
            default_sl = current_price * 1.02
            default_tp = current_price * 0.96
        else:
            raise ValueError(f"Invalid signal for position opening: {signal}")

        final_sl = stop_loss if stop_loss else default_sl
        final_tp = take_profit if take_profit else default_tp
        final_position_size = position_size if position_size is not None else 0.1

        self.current_position = Position(
            entry_price=current_price,
            stop_loss=final_sl,
            take_profit=final_tp,
            size=final_position_size,
            entry_time=datetime.now(),
            confidence=confidence,
            direction=direction
        )

        decision = TradeDecision(
            timestamp=datetime.now(),
            action=signal.upper(),
            price=current_price,
            confidence=confidence,
            stop_loss=final_sl,
            take_profit=final_tp,
            position_size=final_position_size,
            reasoning=f"Opened new {direction} position"
        )
        self.data_persistence.save_position(self.current_position)
        self.data_persistence.save_trade_decision(decision)

    async def process_analysis(self, analysis: str) -> None:
        try:
            current_price = self.periods['3D'].data[-1].close
            signal, confidence, stop_loss, take_profit, position_size = self.extractor.extract_trading_info(analysis)
            self.logger.info(f"Extracted Signal: {signal}, Confidence: {confidence}")

            if self.current_position:
                try:
                    if self._should_close_position(signal):
                        self.logger.info("Closing position based on analysis signal...")
                        await self.close_position("analysis_signal")
                        return

                    self._update_position_parameters(stop_loss, take_profit)
                    return

                except AttributeError:
                    self.logger.warning("Position appears to be already closed")
                    self.current_position = None
                    return

            if signal in ["BUY", "SELL"]:
                await self._open_new_position(
                    signal,
                    current_price,
                    confidence,
                    stop_loss,
                    take_profit,
                    position_size
                )
            elif signal == "CLOSE":
                self.logger.warning("Received CLOSE signal without open position")
            else:
                self.logger.info(f"No valid trading signal ({signal}).")
        except Exception as e:
            self.logger.error(f"Error processing analysis: {e}")
            return