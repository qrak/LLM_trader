"""
Console Notifier - Fallback notification service when Discord is disabled.
Prints AI trading analysis to console with colored and formatted output.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.parsing.unified_parser import UnifiedParser


class ConsoleNotifier:
    """Console-based notifier as fallback when Discord is disabled."""

    def __init__(self, logger, config: "ConfigProtocol", unified_parser: "UnifiedParser") -> None:
        """Initialize ConsoleNotifier.
        
        Args:
            logger: Logger instance
            config: ConfigProtocol instance
            unified_parser: UnifiedParser for JSON extraction (DRY)
        """
        self.logger = logger
        self.config = config
        self.unified_parser = unified_parser
        self.is_initialized = True  # Always ready

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        pass  # No cleanup needed for console

    async def start(self) -> None:
        """Start the console notifier (no-op for console)."""
        self.logger.info("ConsoleNotifier: Using console output (Discord disabled)")

    async def wait_until_ready(self) -> None:
        """Console is always ready."""
        pass

    async def send_message(
            self,
            message: str,
            channel_id: int = None,
            expire_after: Optional[int] = None
    ) -> None:
        """Print a text message to console.
        
        Args:
            message: Message text
            channel_id: Ignored for console output
            expire_after: Ignored for console output
        """
        print(f"\n{message}")

    async def send_trading_decision(
            self,
            decision: Any,
            channel_id: int = None
    ) -> None:
        """Print a trading decision to console.
        
        Args:
            decision: TradingDecision dataclass
            channel_id: Ignored for console output
        """
        print("\n" + "=" * 60)
        print("TRADING DECISION")
        print("=" * 60)
        print(f"Time: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Symbol: {decision.symbol}")
        print(f"Action: {decision.action}")
        print(f"Confidence: {decision.confidence}")
        print(f"Price: ${decision.price:,.2f}")
        
        if decision.stop_loss:
            print(f"Stop Loss: ${decision.stop_loss:,.2f}")
        if decision.take_profit:
            print(f"Take Profit: ${decision.take_profit:,.2f}")
        if decision.position_size:
            print(f"Position Size: {decision.position_size * 100:.1f}%")
        
        print(f"\nReasoning: {decision.reasoning}")
        print("=" * 60 + "\n")

    async def send_analysis_notification(
            self,
            result: dict,
            symbol: str,
            timeframe: str,
            channel_id: int = None
    ) -> None:
        """Print full analysis notification with reasoning and JSON data.
        
        Args:
            result: Analysis result dict with raw_response
            symbol: Trading symbol
            timeframe: Trading timeframe
            channel_id: Ignored for console output
        """
        try:
            raw_response = result.get("raw_response", "")
            if not raw_response:
                return
            
            # Extract reasoning and JSON using UnifiedParser
            reasoning = self.unified_parser.extract_text_before_json(raw_response)
            analysis_json = self.unified_parser.extract_json_block(raw_response, unwrap_key='analysis')
            
            print("\n" + "=" * 60)
            print(f"📊 ANALYSIS: {symbol} ({timeframe})")
            print("=" * 60)
            
            # Print reasoning
            if reasoning:
                print(f"\n{reasoning}")
            
            # Print JSON data as formatted output
            if analysis_json:
                self._print_analysis_data(analysis_json, symbol, timeframe)
                
        except Exception as e:
            self.logger.error(f"Error printing analysis notification: {e}")

    async def send_position_status(
            self,
            position: Any,  # Position dataclass
            current_price: float,
            channel_id: int = None
    ) -> None:
        """Print current open position status.
        
        Args:
            position: Current Position object
            current_price: Current market price
            channel_id: Ignored for console output
        """
        try:
            # Calculate unrealized P&L
            pnl_pct = position.calculate_pnl(current_price)
            if position.direction == 'LONG':
                pnl_usdt = (current_price - position.entry_price) * position.size
            else:
                pnl_usdt = (position.entry_price - current_price) * position.size
            
            # Calculate distance to stop and target
            if position.direction == 'LONG':
                stop_distance_pct = ((position.stop_loss - current_price) / current_price) * 100
                target_distance_pct = ((position.take_profit - current_price) / current_price) * 100
            else:  # SHORT
                stop_distance_pct = ((current_price - position.stop_loss) / current_price) * 100
                target_distance_pct = ((current_price - position.take_profit) / current_price) * 100
            
            # Determine emoji based on P&L
            if pnl_pct > 0:
                emoji = "📈"
            elif pnl_pct < 0:
                emoji = "📉"
            else:
                emoji = "➡️"
            
            # Time held calculation
            time_held = datetime.now() - position.entry_time
            hours_held = time_held.total_seconds() / 3600
            
            print("\n" + "=" * 60)
            print(f"{emoji} OPEN {position.direction} POSITION - {position.symbol}")
            print("=" * 60)
            print(f"Entry Price:     ${position.entry_price:,.2f}")
            print(f"Current Price:   ${current_price:,.2f}")
            print(f"Position Size:   {position.size:.4f}")
            print("-" * 40)
            print(f"Unrealized P&L:  {pnl_pct:+.2f}%")
            print(f"P&L (USDT):      ${pnl_usdt:+,.2f}")
            print(f"Confidence:      {position.confidence}")
            print("-" * 40)
            print(f"Stop Loss:       ${position.stop_loss:,.2f} ({stop_distance_pct:+.2f}%)")
            print(f"Take Profit:     ${position.take_profit:,.2f} ({target_distance_pct:+.2f}%)")
            print(f"Time Held:       {hours_held:.1f}h")
            print(f"Entry Time:      {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
                
        except Exception as e:
            self.logger.error(f"Error printing position status: {e}")

    async def send_performance_stats(
            self,
            trade_history: List[Dict[str, Any]],
            symbol: str,
            channel_id: int = None
    ) -> None:
        """Print overall performance statistics.
        
        Args:
            trade_history: Full trade history list
            symbol: Trading symbol
            channel_id: Ignored for console output
        """
        try:
            if not trade_history:
                return
            
            # Calculate overall performance (same logic as DiscordNotifier)
            total_pnl_usdt = 0.0
            total_pnl_pct = 0.0
            closed_trades = 0
            winning_trades = 0
            
            open_position = None
            for decision_dict in trade_history:
                action = decision_dict.get('action', '')
                price = decision_dict.get('price', 0)
                position_size = decision_dict.get('position_size', 1.0)
                
                if action in ['BUY', 'SELL']:
                    open_position = decision_dict
                elif action in ['CLOSE', 'CLOSE_LONG', 'CLOSE_SHORT'] and open_position:
                    open_action = open_position.get('action', '')
                    open_price = open_position.get('price', 0)
                    
                    if open_action == 'BUY':
                        pnl_pct = ((price - open_price) / open_price) * 100
                        pnl_usdt = (price - open_price) * position_size
                    else:  # SELL
                        pnl_pct = ((open_price - price) / open_price) * 100
                        pnl_usdt = (open_price - price) * position_size
                    
                    total_pnl_usdt += pnl_usdt
                    total_pnl_pct += pnl_pct
                    closed_trades += 1
                    if pnl_pct > 0:
                        winning_trades += 1
                    open_position = None
            
            if closed_trades == 0:
                return
            
            avg_pnl_pct = total_pnl_pct / closed_trades
            win_rate = (winning_trades / closed_trades) * 100
            
            print("\n" + "=" * 60)
            print("📈 TRADING PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"Symbol:          {symbol}")
            print(f"Closed Trades:   {closed_trades}")
            print("-" * 40)
            print(f"Total P&L (USDT): ${total_pnl_usdt:+,.2f}")
            print(f"Total P&L (%):    {total_pnl_pct:+.2f}%")
            print(f"Avg P&L/Trade:    {avg_pnl_pct:+.2f}%")
            print(f"Win Rate:         {win_rate:.1f}% ({winning_trades}/{closed_trades})")
            print("=" * 60)
                
        except Exception as e:
            self.logger.error(f"Error printing performance stats: {e}")


    
    def _print_analysis_data(self, analysis: dict, symbol: str, timeframe: str) -> None:
        """Print analysis JSON data in formatted console output."""
        try:
            signal = analysis.get('signal', 'UNKNOWN')
            confidence = analysis.get('confidence', 0)
            reasoning = analysis.get('reasoning', 'No reasoning provided')
            
            print("\n" + "-" * 40)
            print(f"Signal:          {signal}")
            print(f"Confidence:      {confidence}%")
            
            if 'entry_price' in analysis:
                print(f"Entry:           ${analysis['entry_price']:,.2f}")
            if 'stop_loss' in analysis:
                print(f"Stop Loss:       ${analysis['stop_loss']:,.2f}")
            if 'take_profit' in analysis:
                print(f"Take Profit:     ${analysis['take_profit']:,.2f}")
            if 'risk_reward_ratio' in analysis:
                print(f"R:R Ratio:       {analysis['risk_reward_ratio']:.2f}")
            
            # Trend info
            trend = analysis.get('trend', {})
            if trend:
                direction = trend.get('direction', 'N/A')
                strength = trend.get('strength', 0)
                print(f"Trend:           {direction} ({strength}%)")
            
            # Key levels
            key_levels = analysis.get('key_levels', {})
            if key_levels:
                supports = key_levels.get('support', [])
                resistances = key_levels.get('resistance', [])
                if supports:
                    support_str = ", ".join([f"${s:,.2f}" for s in supports[:3]])
                    print(f"Support:         {support_str}")
                if resistances:
                    resistance_str = ", ".join([f"${r:,.2f}" for r in resistances[:3]])
                    print(f"Resistance:      {resistance_str}")
            
            print(f"Timeframe:       {timeframe}")
            print("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Error formatting analysis data: {e}")
