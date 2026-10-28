"""
Console Notifier - Fallback notification service when Discord is disabled.
Prints AI trading analysis to console with colored and formatted output.
"""
import io
from typing import TYPE_CHECKING, Any

from src.trading.executor_reconciliation import (
    VERIFICATION_EXCHANGE_VERIFIED,
    VERIFICATION_EXECUTOR_REPORTED,
    VERIFICATION_UNVERIFIED,
)
from src.utils.indicator_classifier import resolve_primary_trend_strength
from src.utils.time_utils import utc_stamp

from .base_notifier import BaseNotifier, format_commission

if TYPE_CHECKING:
    from src.config.loader import Config
    from src.parsing.unified_parser import UnifiedParser
    from src.utils.format_utils import FormatUtils



class ConsoleNotifier(BaseNotifier):
    """Console-based notifier as fallback when Discord is disabled."""

    def __init__(self, logger, config: "Config", unified_parser: "UnifiedParser", formatter: "FormatUtils") -> None:
        """Initialize ConsoleNotifier."""
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
        self.logger.debug("ConsoleNotifier: Console is always ready")

    async def shutdown(self) -> None:
        """Shutdown the console notifier (no-op for console).
        
        Console notifier has no resources to clean up, so this is a no-op.
        """
        self.logger.debug("ConsoleNotifier: Shutdown complete (no-op for console)")

    async def send_message(
            self,
            message: str,
            channel_id: int | None = None,
            expire_after: int | None = None
    ) -> None:
        """Print a text message to console."""
        print(f"\n{message}")

    async def send_trading_decision(self, decision: Any, channel_id: int | None = None) -> None:
        """Print a trading decision to console."""
        _, emoji = self.get_action_styling(decision.action)

        print("\n" + "=" * 60)
        print(f"{emoji} TRADING DECISION")
        print("=" * 60)
        print(f"Time: {utc_stamp(decision.timestamp)}")
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
        if decision.action in ["BUY", "SELL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT"]:
            print(f"Fee:       {format_commission(decision.fee)}")

        print(f"\nReasoning: {decision.reasoning}")
        print("=" * 60 + "\n")

    async def send_analysis_notification(
            self,
            result: dict,
            symbol: str,
            timeframe: str,
            channel_id: int | None = None,
            chart_image: io.BytesIO | None = None,
            execution_note: str | None = None
    ) -> None:
        """Print full analysis notification with reasoning and JSON data.

        ``execution_note`` marks a recommendation the bot did NOT execute; it is
        printed before the analysis so the console never shows a bare signal that
        reads as a performed action.
        """
        try:
            analysis = result.get("analysis")
            if not analysis:
                return

            raw_response = result.get("raw_response", "")
            reasoning = self.unified_parser.extract_text_before_json(raw_response) if raw_response else ""

            print("\n" + "=" * 60)
            print(f"📊 ANALYSIS: {symbol} ({timeframe})")
            print("=" * 60)

            if execution_note:
                print(f"🚫 RECOMMENDATION NOT EXECUTED: {execution_note}")

            if reasoning:
                print(f"\n{reasoning}")

            self._print_analysis_data(analysis, timeframe)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error printing analysis notification: %s", e)

    async def send_position_status(
            self,
            position: Any,
            current_price: float,
            channel_id: int | None = None,
            *,
            verification: str = VERIFICATION_UNVERIFIED,
            verified_at: Any = None,
            verification_detail: str | None = None
    ) -> None:
        """Print current open position status.

        ``verification`` mirrors the Discord card's venue, tracker-only or unknown state.
        """
        try:
            pnl_pct, pnl_quote = self.calculate_position_pnl(position, current_price)
            stop_distance_pct, target_distance_pct = self.calculate_stop_target_distances(position, current_price)
            hours_held = self.calculate_time_held(position.entry_time)

            _, emoji = self.get_pnl_styling(pnl_pct)

            exchange_verified = verification == VERIFICATION_EXCHANGE_VERIFIED
            reported = verification == VERIFICATION_EXECUTOR_REPORTED
            if exchange_verified:
                header = f"{emoji} OPEN {position.direction} POSITION - {position.symbol}"
                position_state = f"Exchange-verified ({verified_at or 'unknown'}) — protection active"
            elif reported:
                header = (
                    f"{emoji} OPEN {position.direction} POSITION (executor-reported, NOT "
                    f"exchange-verified) - {position.symbol}"
                )
                position_state = (
                    f"executor-reported ({verified_at or 'unknown'}) — NOT exchange-verified"
                )
            else:
                header = f"⚠️ POSITION STATUS UNVERIFIED - {position.symbol}"
                position_state = (
                    f"UNVERIFIED — {verification_detail or 'no exchange verification performed yet'}"
                )
            print("\n" + "=" * 60)
            print(header)
            print("=" * 60)
            print(f"Position State:  {position_state}")
            print(f"Entry Price:     ${position.entry_price:,.2f}")
            print(f"Current Price:   ${current_price:,.2f}")
            print(f"Quantity:        {self.formatter.fmt(position.size)}")
            if position.quote_amount > 0:
                print(f"Invested:        ${position.quote_amount:,.2f}")
            print("-" * 40)
            print(f"Unrealized P&L:  {pnl_pct:+.2f}%")
            print(f"P&L ({self.config.QUOTE_CURRENCY}):  ${pnl_quote:+,.2f}")
            print(f"Confidence:      {position.confidence}")
            print(f"Position Size %: {position.size_pct * 100:.2f}%")
            print("-" * 40)
            print(f"Stop Loss:       ${position.stop_loss:,.2f} ({stop_distance_pct:+.2f}%)")
            print(f"Take Profit:     ${position.take_profit:,.2f} ({target_distance_pct:+.2f}%)")
            print(f"Exit Monitoring: {self.format_exit_monitoring()}")
            print(f"Entry Fee:       {format_commission(position.entry_fee)}")
            print(f"Time Held:       {hours_held:.1f}h")
            print(f"Entry Time:      {utc_stamp(position.entry_time)}")
            print("=" * 60)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error printing position status: %s", e)

    async def send_performance_stats(
            self,
            trade_history: list[dict[str, Any]],
            symbol: str,
            channel_id: int | None = None
    ) -> None:
        """Print overall performance statistics."""
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
            total_fees_value = format_commission(stats["total_fees"])
            if not stats.get("fees_complete", True):
                total_fees_value += f" (+{stats['fees_unknown_trades']} trade(s) without fee data)"
            print(f"Total Fees:       {total_fees_value}")
            print(f"Net P&L (USDT):   ${stats['net_pnl']:+,.2f}")

            last_closed_trade = stats.get("last_closed_trade")
            if last_closed_trade:
                outcome = last_closed_trade.get("outcome", "UNKNOWN")
                close_reason = last_closed_trade.get("close_reason")
                print("-" * 40)
                print(f"Last Outcome:     {outcome}")
                if close_reason:
                    print(f"Close Reason:     {close_reason}")
                print(
                    f"Last Trade P&L ({self.config.QUOTE_CURRENCY}): "
                    f"${last_closed_trade['pnl_quote']:+,.2f} ({last_closed_trade['pnl_pct']:+.2f}%)"
                )

            print("=" * 60)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error printing performance stats: %s", e)

    def _print_analysis_data(self, analysis: dict, timeframe: str) -> None:
        """Print analysis JSON data in formatted console output."""
        try:
            fields = self.extract_analysis_fields(analysis)

            print("\n" + "-" * 40)
            print(f"Signal:          {fields['signal']}")
            print(f"Confidence:      {fields['confidence']}%")

            if fields["entry_price"]:
                print(f"Entry:           ${fields['entry_price']:,.2f}")
            if fields["stop_loss"]:
                print(f"Stop Loss:       ${fields['stop_loss']:,.2f}")
            if fields["take_profit"]:
                print(f"Take Profit:     ${fields['take_profit']:,.2f}")
            if fields["risk_reward_ratio"]:
                print(f"R:R Ratio:       {fields['risk_reward_ratio']:.2f}")

            trend = fields["trend"]
            if trend:
                direction = trend.get("direction", "N/A")
                strength = resolve_primary_trend_strength(trend)
                if strength is None:
                    strength = trend.get("strength_daily", 0)
                print(f"Trend:           {direction} ({strength}%)")

            key_levels = fields["key_levels"]
            if key_levels:
                supports = key_levels.get("support", [])
                resistances = key_levels.get("resistance", [])
                if supports:
                    support_str = ", ".join([f"${s:,.2f}" for s in supports[:3]])
                    print(f"Support:         {support_str}")
                if resistances:
                    resistance_str = ", ".join([f"${r:,.2f}" for r in resistances[:3]])
                    print(f"Resistance:      {resistance_str}")

            print(f"Timeframe:       {timeframe}")
            print("=" * 60)
        except Exception as e:  # noqa: BLE001
            self.logger.error("Error formatting analysis data: %s", e)

