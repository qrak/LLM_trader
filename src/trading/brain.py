"""Trading brain service for learning and adaptive parameters.

Handles brain state management, learning from closed trades, and providing AI context.
"""

from typing import Any, TYPE_CHECKING

from src.logger.logger import Logger
from .vector_memory import VectorMemoryService
from .data_models import ExitExecutionContext, MarketConditions, Position, TradeDecision
from .brain_exit_profiles import ExitProfileResolver
from .brain_patterns import TradePatternAnalyzer
from .brain_reflection import BrainReflectionEngine
from .brain_experience import BrainExperienceRecorder
from .brain_context import BrainContextProvider
from .stop_loss_tightening_policy import StopLossTighteningPolicy, TighteningEvaluation

if TYPE_CHECKING:
    from src.managers.persistence_manager import PersistenceManager


class TradingBrainService:
    """Service for managing trading brain and learning from trades.

    Responsibilities:
    - Update brain from closed trades
    - Provide brain context for AI prompts
    - Suggest parameters based on learned data
    - Get dynamic thresholds
    """

    DEFAULT_REFLECTION_TIMEFRAME_MINUTES = 240
    SCALPING_REFLECTION_INTERVAL = 10
    INTRADAY_REFLECTION_INTERVAL = 7
    SWING_REFLECTION_INTERVAL = 5
    POSITION_REFLECTION_INTERVAL = 3

    @classmethod
    def _derive_reflection_interval(cls, timeframe_minutes: int) -> int:
        """Derive closed-trade reflection cadence from active timeframe."""
        try:
            minutes = int(timeframe_minutes)
        except (TypeError, ValueError):
            minutes = cls.DEFAULT_REFLECTION_TIMEFRAME_MINUTES
        if minutes <= 0:
            minutes = cls.DEFAULT_REFLECTION_TIMEFRAME_MINUTES
        if minutes < 60:
            return cls.SCALPING_REFLECTION_INTERVAL
        if minutes < 240:
            return cls.INTRADAY_REFLECTION_INTERVAL
        if minutes < 1440:
            return cls.SWING_REFLECTION_INTERVAL
        return cls.POSITION_REFLECTION_INTERVAL

    def __init__(
        self,
        logger: Logger,
        persistence: "PersistenceManager",
        vector_memory: VectorMemoryService,
        exit_execution_context: "ExitExecutionContext | None" = None,
        timeframe_minutes: int = DEFAULT_REFLECTION_TIMEFRAME_MINUTES,
        tightening_policy: StopLossTighteningPolicy | None = None,
        post_mortem_repo: Any | None = None,
    ):
        """Initialize trading brain service.

        Args:
            logger: Logger instance
            persistence: Persistence service
            vector_memory: Injected vector memory service (required)
            exit_execution_context: Configured fallback SL/TP execution context.
            timeframe_minutes: Active analysis timeframe in minutes.
            tightening_policy: Stop-loss tightening policy for threshold exposure.
            post_mortem_repo: Optional PostMortemRepository for trade journal injection.
        """
        self.logger = logger
        self.persistence = persistence
        self.vector_memory = vector_memory
        self._default_exit_execution_context: ExitExecutionContext = exit_execution_context or ExitExecutionContext()
        self._timeframe_minutes: int = timeframe_minutes
        self._tightening_policy: StopLossTighteningPolicy = (
            tightening_policy if tightening_policy is not None else StopLossTighteningPolicy()
        )
        self.exit_profiles = ExitProfileResolver(self._default_exit_execution_context)
        self.pattern_analyzer = TradePatternAnalyzer(self.exit_profiles)
        self.reflection_engine = BrainReflectionEngine(
            logger=self.logger,
            vector_memory=self.vector_memory,
            analyzer=self.pattern_analyzer,
            exit_profiles=self.exit_profiles,
        )
        self.experience_recorder = BrainExperienceRecorder(
            logger=self.logger,
            vector_memory=self.vector_memory,
            pattern_analyzer=self.pattern_analyzer,
            default_exit_execution_context=self._default_exit_execution_context,
        )
        self.context_provider = BrainContextProvider(
            vector_memory=self.vector_memory,
            exit_profiles=self.exit_profiles,
            post_mortem_repo=post_mortem_repo,
        )

        self._reflection_interval: int = self._derive_reflection_interval(timeframe_minutes)
        self.logger.debug(
            "Trading brain reflection interval: every %s closed trades",
            self._reflection_interval,
        )

        # Initialize trade count from persistent storage
        # This ensures reflection triggers consistently across restarts
        self._trade_count: int = self.vector_memory.trade_count

    def update_from_closed_trade(
        self,
        position: Position,
        close_price: float,
        close_reason: str,
        entry_decision: TradeDecision | None = None,
        market_conditions: "MarketConditions | None" = None
    ) -> None:
        """Extract insights from a closed trade and update brain.

        Args:
            position: Closed position
            close_price: Exit price
            close_reason: Reason for closing
            entry_decision: Original entry decision (for reasoning)
            market_conditions: Market state at close (or from entry if preferred)
        """
        self.context_provider.clear_stats_cache()
        self.experience_recorder.record_closed_trade(
            position=position,
            close_price=close_price,
            close_reason=close_reason,
            entry_decision=entry_decision,
            market_conditions=market_conditions,
        )
        self._trade_count += 1
        if self._trade_count % self._reflection_interval == 0:
            self._trigger_reflection()
            self._trigger_loss_reflection()
            self._trigger_ai_mistake_reflection()

    def get_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        rsi: float = 50.0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        exit_execution_context: "ExitExecutionContext | None" = None,
    ) -> str:
        """Generate formatted brain context for prompt injection using vector retrieval.

        Args:
            trend_direction: Current trend (BULLISH/BEARISH/NEUTRAL)
            adx: Current ADX value
            rsi: Current RSI numeric value
            volatility_level: Current volatility (HIGH/MEDIUM/LOW)
            rsi_level: RSI state (OVERBOUGHT/STRONG/NEUTRAL/WEAK/OVERSOLD)
            macd_signal: MACD signal (BULLISH/BEARISH/NEUTRAL)
            volume_state: Volume state (ACCUMULATION/NORMAL/DISTRIBUTION)
            bb_position: Bollinger Band position (UPPER/MIDDLE/LOWER)
            is_weekend: Whether current day is Saturday or Sunday
            market_sentiment: Fear & Greed state (EXTREME_FEAR/FEAR/NEUTRAL/GREED/EXTREME_GREED)
            order_book_bias: Order book pressure (BUY_PRESSURE/SELL_PRESSURE/BALANCED)

        Returns:
            Formatted string with vector-retrieved experiences and confidence calibration.
        """
        return self.context_provider.get_context(
            trend_direction=trend_direction,
            adx=adx,
            rsi=rsi,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
        )

    def get_dynamic_thresholds(self) -> dict[str, Any]:
        """Get Brain-learned thresholds from vector store.

        Returns: dict with learned thresholds. Defaults used when insufficient data.
        """
        thresholds = self.context_provider.get_dynamic_thresholds()
        base_threshold = self._tightening_policy.get_base_threshold(self._timeframe_minutes)
        effective_threshold, source = self._tightening_policy._resolve_effective_threshold(
            base_threshold, thresholds
        )
        thresholds["sl_tightening_pct"] = round(effective_threshold * 100)
        thresholds["sl_tightening_source"] = source
        sl_payload: dict[str, Any] = thresholds.get("sl_tightening") or {}
        sl_payload["base_threshold"] = base_threshold
        sl_payload["effective_threshold"] = effective_threshold
        sl_payload["effective_threshold_pct"] = round(effective_threshold * 100)
        sl_payload["source"] = source
        thresholds["sl_tightening"] = sl_payload
        return thresholds

    def _build_rich_context_string(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        exit_execution_context: "ExitExecutionContext | None" = None,
    ) -> str:
        """Build rich semantic context string for vector storage and retrieval.

        This unified method ensures that the context stored in memory matches
        the format of the context used for querying, maximizing vector similarity.
        """
        return self.context_provider.build_rich_context_string(
            trend_direction=trend_direction,
            adx=adx,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
        )

    def get_vector_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        rsi: float = 50.0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        exit_execution_context: "ExitExecutionContext | None" = None,
        k: int = 5
    ) -> str:
        """Get context from similar past experiences via vector retrieval.

        Uses semantic search to find trades in similar market conditions.

        Args:
            trend_direction: Current trend (BULLISH/BEARISH/NEUTRAL)
            adx: Current ADX value
            rsi: Current RSI numeric value
            volatility_level: Current volatility (HIGH/MEDIUM/LOW)
            rsi_level: RSI state (OVERBOUGHT/STRONG/NEUTRAL/WEAK/OVERSOLD)
            macd_signal: MACD signal (BULLISH/BEARISH/NEUTRAL)
            volume_state: Volume state (ACCUMULATION/NORMAL/DISTRIBUTION)
            bb_position: Bollinger Band position (UPPER/MIDDLE/LOWER)
            is_weekend: Whether current day is Saturday or Sunday
            market_sentiment: Fear & Greed state
            order_book_bias: Order book pressure
            k: Number of experiences to retrieve

        Returns:
            Formatted string with similar past trades for prompt injection.
        """
        return self.context_provider.get_vector_context(
            trend_direction=trend_direction,
            adx=adx,
            rsi=rsi,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
            k=k,
        )

    def refresh_semantic_rules_if_stale(self) -> None:
        """Refresh semantic rules once when active rules still use unknown exit profiles."""
        self.reflection_engine.refresh_semantic_rules_if_stale()

    def _trigger_reflection(self) -> None:
        """Reflect on recent trades and synthesize best-practice semantic rules.

        Called automatically every N trades. Identifies profitable patterns from
        win/loss history and stores outcome-aware rules with richer diagnostics.
        """
        self.reflection_engine.trigger_reflection()

    def _trigger_loss_reflection(self) -> None:
        """Reflect on losing trades and synthesize anti-pattern and corrective rules.

        Called automatically every N trades. Analyzes LOSS and mixed-outcome patterns
        to identify conditions that hurt profitability, storing actionable guidance.
        """
        self.reflection_engine.trigger_loss_reflection()

    def _trigger_ai_mistake_reflection(self) -> None:
        """Reflect on cases where the AI's confidence or premise was wrong."""
        self.reflection_engine.trigger_ai_mistake_reflection()

    def track_position_update(
        self,
        position: Position,
        old_sl: float,
        old_tp: float,
        new_sl: float,
        new_tp: float,
        current_price: float,
        current_pnl_pct: float,
        market_conditions: "MarketConditions | None" = None,
        tightening_evaluation: TighteningEvaluation | None = None,
    ) -> None:
        """Track position update decisions for learning.

        Args:
            position: Active position
            old_sl: Previous stop loss
            old_tp: Previous take profit
            new_sl: New stop loss
            new_tp: New take profit
            current_price: Current market price
            current_pnl_pct: Current unrealized P&L
            market_conditions: Market state at time of update
            tightening_evaluation: Policy evaluation from StopLossTighteningPolicy (optional)
        """
        self.experience_recorder.track_position_update(
            position=position,
            old_sl=old_sl,
            old_tp=old_tp,
            new_sl=new_sl,
            new_tp=new_tp,
            current_price=current_price,
            current_pnl_pct=current_pnl_pct,
            market_conditions=market_conditions,
            tightening_evaluation=tightening_evaluation,
            timeframe_minutes=self._timeframe_minutes,
        )
