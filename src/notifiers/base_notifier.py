"""
Base Notifier - Abstract base class providing shared logic for notifiers.
Subclasses implement rendering methods for their specific output medium.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol
    from src.parsing.unified_parser import UnifiedParser
    from src.utils.format_utils import FormatUtils


class BaseNotifier(ABC):
    """Abstract base class for notifiers with shared calculation logic."""

    def __init__(self, logger, config: "ConfigProtocol", unified_parser: "UnifiedParser", formatter: "FormatUtils") -> None:
        """Initialize BaseNotifier.

        Args:
            logger: Logger instance
            config: ConfigProtocol instance
            unified_parser: UnifiedParser for JSON extraction (DRY)
            formatter: FormatUtils instance for value formatting
        """
        self.logger = logger
        self.config = config
        self.unified_parser = unified_parser
        self.formatter = formatter
        self.is_initialized = False

    @abstractmethod
    async def start(self) -> None:
        """Start the notifier service."""

    @abstractmethod
    async def wait_until_ready(self) -> None:
        """Wait for the notifier to be fully initialized."""

    @abstractmethod
    async def send_message(
            self,
            message: str,
            channel_id: int,
            expire_after: Optional[int] = None
    ) -> Any:
        """Send a text message."""

    @abstractmethod
    async def send_trading_decision(self, decision: Any, channel_id: int) -> None:
        """Send a trading decision notification."""

    @abstractmethod
    async def send_analysis_notification(
            self,
            result: dict,
            symbol: str,
            timeframe: str,
            channel_id: int
    ) -> None:
        """Send full analysis notification."""

    @abstractmethod
    async def send_position_status(
            self,
            position: Any,
            current_price: float,
            channel_id: int
    ) -> None:
        """Send current open position status."""

    @abstractmethod
    async def send_performance_stats(
            self,
            trade_history: List[Dict[str, Any]],
            symbol: str,
            channel_id: int
    ) -> None:
        """Send overall performance statistics."""

    @staticmethod
    def get_action_styling(action: str) -> Tuple[str, str]:
        """Get color key and emoji for a trading action.

        Args:
            action: Trading action (BUY, SELL, HOLD, CLOSE, etc.)

        Returns:
            Tuple of (color_key, emoji)
        """
        color_map = {
            'BUY': 'green',
            'SELL': 'red',
            'HOLD': 'grey',
            'CLOSE': 'orange',
            'CLOSE_LONG': 'orange',
            'CLOSE_SHORT': 'orange',
            'UPDATE': 'blue',
        }
        emoji_map = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '⚪',
            'CLOSE': '🟠',
            'CLOSE_LONG': '🟠',
            'CLOSE_SHORT': '🟠',
            'UPDATE': '🔵',
        }
        return color_map.get(action, 'grey'), emoji_map.get(action, '📊')

    @staticmethod
    def get_pnl_styling(pnl_pct: float) -> Tuple[str, str]:
        """Get color key and emoji based on PnL percentage.

        Args:
            pnl_pct: Profit and Loss percentage

        Returns:
            Tuple of (color_key, emoji)
        """
        if pnl_pct > 0:
            return 'green', '📈'
        elif pnl_pct < 0:
            return 'red', '📉'
        return 'grey', '➡️'


    def calculate_position_pnl(
            self,
            position: Any,
            current_price: float
    ) -> Tuple[float, float]:
        """Calculate unrealized PnL for a position.

        Args:
            position: Position object with entry_price, size, direction
            current_price: Current market price

        Returns:
            Tuple of (pnl_percent, pnl_quote)
        """
        pnl_pct = position.calculate_pnl(current_price)
        if position.direction == 'LONG':
            pnl_quote = (current_price - position.entry_price) * position.size
        else:
            pnl_quote = (position.entry_price - current_price) * position.size
        return pnl_pct, pnl_quote

    def calculate_stop_target_distances(
            self,
            position: Any,
            current_price: float
    ) -> Tuple[float, float]:
        """Calculate percentage distances to stop loss and take profit.

        Args:
            position: Position object with stop_loss, take_profit, direction
            current_price: Current market price

        Returns:
            Tuple of (stop_distance_pct, target_distance_pct)
        """
        if position.direction == 'LONG':
            stop_distance_pct = ((position.stop_loss - current_price) / current_price) * 100
            target_distance_pct = ((position.take_profit - current_price) / current_price) * 100
        else:
            stop_distance_pct = ((current_price - position.stop_loss) / current_price) * 100
            target_distance_pct = ((current_price - position.take_profit) / current_price) * 100
        return stop_distance_pct, target_distance_pct

    @staticmethod
    def calculate_time_held(entry_time: datetime) -> float:
        """Calculate hours held since entry.

        Args:
            entry_time: Position entry timestamp

        Returns:
            Hours held as float
        """
        from datetime import timezone
        now = datetime.now(timezone.utc)
        # Handle naive datetime by assuming UTC
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        time_held = now - entry_time
        return time_held.total_seconds() / 3600

    def calculate_performance_stats(
            self,
            trade_history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Calculate overall performance statistics from trade history.

        Args:
            trade_history: List of trade decision dictionaries

        Returns:
            Dict with stats or None if no closed trades
        """
        if not trade_history:
            return None

        total_pnl_quote = 0.0
        total_pnl_pct = 0.0
        total_fees = 0.0
        closed_trades = 0
        winning_trades = 0
        open_position = None

        for decision_dict in trade_history:
            action = decision_dict.get('action', '')
            price = decision_dict.get('price', 0)
            quantity = decision_dict.get('quantity', 0.0)

            if action in ['BUY', 'SELL']:
                open_position = decision_dict
            elif action in ['CLOSE', 'CLOSE_LONG', 'CLOSE_SHORT'] and open_position:
                open_action = open_position.get('action', '')
                open_price = open_position.get('price', 0)
                open_quantity = open_position.get('quantity', 0.0)

                if open_action == 'BUY':
                    pnl_pct = ((price - open_price) / open_price) * 100
                    pnl_quote = (price - open_price) * open_quantity
                else:
                    pnl_pct = ((open_price - price) / open_price) * 100
                    pnl_quote = (open_price - price) * open_quantity

                entry_fee = open_position.get('fee', 0.0)
                exit_fee = decision_dict.get('fee', 0.0)
                
                # Fallback for old history if fee is 0.0 (though migration should have fixed this)
                if entry_fee == 0.0 and open_quantity > 0:
                     entry_fee = open_price * open_quantity * self.config.TRANSACTION_FEE_PERCENT
                if exit_fee == 0.0 and quantity > 0:
                     exit_fee = price * quantity * self.config.TRANSACTION_FEE_PERCENT
                     
                total_fees += entry_fee + exit_fee
                total_pnl_quote += pnl_quote
                total_pnl_pct += pnl_pct
                closed_trades += 1

                if pnl_pct > 0:
                    winning_trades += 1
                open_position = None

        if closed_trades == 0:
            return None

        return {
            'total_pnl_quote': total_pnl_quote,
            'total_pnl_pct': total_pnl_pct,
            'total_fees': total_fees,
            'closed_trades': closed_trades,
            'winning_trades': winning_trades,
            'avg_pnl_pct': total_pnl_pct / closed_trades,
            'win_rate': (winning_trades / closed_trades) * 100,
            'net_pnl': total_pnl_quote - total_fees,
        }

    @staticmethod
    def extract_analysis_fields(analysis: dict) -> Dict[str, Any]:
        """Extract common fields from analysis JSON.

        Args:
            analysis: Analysis dictionary from AI response

        Returns:
            Dict with extracted fields
        """
        return {
            'signal': analysis.get('signal', 'UNKNOWN'),
            'confidence': analysis.get('confidence', 0),
            'reasoning': analysis.get('reasoning', 'No reasoning provided'),
            'entry_price': analysis.get('entry_price'),
            'stop_loss': analysis.get('stop_loss'),
            'take_profit': analysis.get('take_profit'),
            'risk_reward_ratio': analysis.get('risk_reward_ratio'),
            'trend': analysis.get('trend', {}),
            'key_levels': analysis.get('key_levels', {}),
        }

    def parse_analysis_response(self, raw_response: str) -> Tuple[str, Optional[dict]]:
        """Parse raw AI response to extract reasoning and JSON.

        Args:
            raw_response: Raw response string from AI

        Returns:
            Tuple of (reasoning_text, analysis_json or None)
        """
        reasoning = self.unified_parser.extract_text_before_json(raw_response)
        analysis_json = self.unified_parser.extract_json_block(raw_response, unwrap_key='analysis')
        return reasoning, analysis_json
