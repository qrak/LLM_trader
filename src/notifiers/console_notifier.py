"""
Console Notifier - Fallback notification service when Discord is disabled.
Prints AI trading analysis to console with colored and formatted output.
"""
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.parsing.unified_parser import UnifiedParser
    from src.utils.format_utils import FormatUtils

from .base_notifier import BaseNotifier



class ConsoleNotifier(BaseNotifier):
    """Console-based notifier as fallback when Discord is disabled."""

    def __init__(self, logger, config: "ConfigProtocol", unified_parser: "UnifiedParser", formatter: "FormatUtils") -> None:
        """Initialize ConsoleNotifier.

        Args:
            logger: Logger instance
            config: ConfigProtocol instance
            unified_parser: UnifiedParser for JSON extraction (DRY)
            formatter: FormatUtils instance for value formatting
        """
        super().__init__(logger, config, unified_parser, formatter)
        self.is_initialized = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        pass

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

    async def send_trading_decision(self, decision: Any, channel_id: int = None) -> None:
        """Print a trading decision to console.

        Args:
            decision: TradingDecision dataclass
            channel_id: Ignored for console output
        """
        _, emoji = self.get_action_styling(decision.action)

        print("\n" + "=" * 60)
        print(f"{emoji} TRADING DECISION")
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
            print(f"Position Size: {decision.position_size * 100:.2f}%")
        if decision.quote_amount:
            print(f"Invested:      ${decision.quote_amount:,.2f}")
        if decision.quantity:
            print(f"Quantity:      {self.formatter.fmt(decision.quantity)}")
        if decision.action in ['BUY', 'SELL', 'CLOSE', 'CLOSE_LONG', 'CLOSE_SHORT'] and decision.fee:
            print(f"Fee:       ${decision.fee:.4f}")

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

            reasoning, analysis_json = self.parse_analysis_response(raw_response)

            print("\n" + "=" * 60)
            print(f"📊 ANALYSIS: {symbol} ({timeframe})")
            print("=" * 60)

            if reasoning:
                print(f"\n{reasoning}")

            if analysis_json:
                self._print_analysis_data(analysis_json, timeframe)
        except Exception as e:
            self.logger.error(f"Error printing analysis notification: {e}")

    async def send_position_status(
            self,
            position: Any,
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
            pnl_pct, pnl_quote = self.calculate_position_pnl(position, current_price)
            stop_distance_pct, target_distance_pct = self.calculate_stop_target_distances(position, current_price)
            hours_held = self.calculate_time_held(position.entry_time)

            _, emoji = self.get_pnl_styling(pnl_pct)

            print("\n" + "=" * 60)
            print(f"{emoji} OPEN {position.direction} POSITION - {position.symbol}")
            print("=" * 60)
            print(f"Entry Price:     ${position.entry_price:,.2f}")
            print(f"Current Price:   ${current_price:,.2f}")
            print(f"Quantity:        {self.formatter.fmt(position.size)}")
            if hasattr(position, 'quote_amount') and position.quote_amount > 0:
                print(f"Invested:        ${position.quote_amount:,.2f}")
            print("-" * 40)
            print(f"Unrealized P&L:  {pnl_pct:+.2f}%")
            print(f"P&L ({self.config.QUOTE_CURRENCY}):  ${pnl_quote:+,.2f}")
            print(f"Confidence:      {position.confidence}")
            print(f"Position Size %: {position.size_pct * 100:.2f}%")
            print("-" * 40)
            print(f"Stop Loss:       ${position.stop_loss:,.2f} ({stop_distance_pct:+.2f}%)")
            print(f"Take Profit:     ${position.take_profit:,.2f} ({target_distance_pct:+.2f}%)")
            print(f"Entry Fee:       ${position.entry_fee:.4f}")
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
            stats = self.calculate_performance_stats(trade_history)
            if not stats:
                return

            print("\n" + "=" * 60)
            print("📈 TRADING PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"Symbol:           {symbol}")
            print(f"Closed Trades:    {stats['closed_trades']}")
            print("-" * 40)
            print(f"Total P&L ({self.config.QUOTE_CURRENCY}): ${stats['total_pnl_quote']:+,.2f}")
            print(f"Total P&L (%):    {stats['total_pnl_pct']:+.2f}%")
            print(f"Avg P&L/Trade:    {stats['avg_pnl_pct']:+.2f}%")
            print(f"Win Rate:         {stats['win_rate']:.1f}% ({stats['winning_trades']}/{stats['closed_trades']})")
            print(f"Total Fees:       ${stats['total_fees']:.4f}")
            print(f"Net P&L (USDT):   ${stats['net_pnl']:+,.2f}")
            print("=" * 60)
        except Exception as e:
            self.logger.error(f"Error printing performance stats: {e}")

    def _print_analysis_data(self, analysis: dict, timeframe: str) -> None:
        """Print analysis JSON data in formatted console output."""
        try:
            fields = self.extract_analysis_fields(analysis)

            print("\n" + "-" * 40)
            print(f"Signal:          {fields['signal']}")
            print(f"Confidence:      {fields['confidence']}%")

            if fields['entry_price']:
                print(f"Entry:           ${fields['entry_price']:,.2f}")
            if fields['stop_loss']:
                print(f"Stop Loss:       ${fields['stop_loss']:,.2f}")
            if fields['take_profit']:
                print(f"Take Profit:     ${fields['take_profit']:,.2f}")
            if fields['risk_reward_ratio']:
                print(f"R:R Ratio:       {fields['risk_reward_ratio']:.2f}")

            trend = fields['trend']
            if trend:
                direction = trend.get('direction', 'N/A')
                strength = trend.get('strength', 0)
                print(f"Trend:           {direction} ({strength}%)")

            key_levels = fields['key_levels']
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
