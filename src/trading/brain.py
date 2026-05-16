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

    UNKNOWN_EXIT_PROFILE = "SL unknown/unknown | TP unknown/unknown"
    UNKNOWN_EXIT_PROFILE_KEY = "sl_unknown_unknown|tp_unknown_unknown"
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
    ):
        """Initialize trading brain service.

        Args:
            logger: Logger instance
            persistence: Persistence service
            vector_memory: Injected vector memory service (required)
            exit_execution_context: Configured fallback SL/TP execution context.
            timeframe_minutes: Active analysis timeframe in minutes.
            tightening_policy: Stop-loss tightening policy for threshold exposure.
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

    def _build_query_document(
        self,
        trend_direction: str,
        adx: float,
        rsi: float,
        volatility_level: str,
        rsi_level: str,
        macd_signal: str,
        volume_state: str,
        bb_position: str,
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        exit_execution_context: "ExitExecutionContext | None" = None,
    ) -> str:
        """Build a query document that mirrors _build_experience_document format.

        Reduces embedding asymmetry between stored documents and query by using
        the same structural format (Indicators line with numeric values).

        Args:
            trend_direction: Current trend direction
            adx: Current ADX numeric value
            rsi: Current RSI numeric value
            volatility_level: Current volatility label
            rsi_level: RSI label (OVERBOUGHT/STRONG/NEUTRAL/WEAK/OVERSOLD)
            macd_signal: MACD signal label
            volume_state: Volume state label
            bb_position: BB position label
            is_weekend: Whether current day is Saturday or Sunday
            market_sentiment: Fear & Greed label
            order_book_bias: Order book bias label

        Returns:
            Query string formatted like stored experience documents.
        """
        return self.context_provider.build_query_document(
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

    def _get_cached_stats(self, key: str, compute_fn) -> dict[str, Any]:
        """Get stats from cache or compute and cache them.

        Args:
            key: Cache key for the stats type.
            compute_fn: Function to call if cache miss.

        Returns:
            Computed or cached statistics.
        """
        return self.context_provider.get_cached_stats(key, compute_fn)

    def _extract_factor_scores(self, confluence_factors: tuple) -> dict[str, float]:
        """Extract factor scores into flat dict for vector metadata."""
        return self.pattern_analyzer.extract_factor_scores(confluence_factors)

    def _count_strong_confluences(self, confluence_factors: tuple) -> int:
        """Count factors with score > 50 (supporting the trade)."""
        return self.pattern_analyzer.count_strong_confluences(confluence_factors)

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        """Return a float for optional numeric metadata values."""
        return TradePatternAnalyzer.as_float(value, default)

    @staticmethod
    def _safe_key_part(value: Any, default: str = "unknown") -> str:
        """Normalize metadata values for deterministic semantic rule IDs."""
        return ExitProfileResolver.safe_key_part(value, default)

    def _resolve_exit_execution_context(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> ExitExecutionContext:
        """Return SL/TP execution metadata with configured defaults filled in."""
        return self.exit_profiles.resolve_exit_execution_context(metadata)

    def _resolve_rule_exit_execution_context(self, metadata: dict[str, Any]) -> ExitExecutionContext:
        """Resolve exit execution context from semantic-rule metadata."""
        return self.exit_profiles.resolve_rule_exit_execution_context(metadata)

    def _format_exit_profile_from_context(self, context: ExitExecutionContext) -> str:
        """Format a normalized SL/TP execution context."""
        return self.exit_profiles.format_exit_profile_from_context(context)

    def _replace_unknown_exit_profile_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Replace legacy unknown exit-profile text with resolved rule/default metadata."""
        return self.exit_profiles.replace_unknown_exit_profile_text(text, metadata)

    def _render_rule_text(self, rule: dict[str, Any]) -> str:
        """Return rule text with legacy unknown exit profile corrected for display."""
        return self.exit_profiles.render_rule_text(rule)

    def _dominant_exit_execution_context(self, metas: list[dict[str, Any]]) -> ExitExecutionContext:
        """Return the most common resolved SL/TP execution context for a trade group."""
        return self.exit_profiles.dominant_exit_execution_context(metas)

    def _legacy_unknown_exit_pattern_key(self, pattern_key: str) -> str:
        """Return the equivalent rule key that used the old unknown exit profile."""
        return self.exit_profiles.legacy_unknown_exit_pattern_key(pattern_key)

    def _deactivate_legacy_unknown_exit_rule(self, rule_id_prefix: str, pattern_key: str) -> None:
        """Deactivate the stale unknown-profile rule once a resolved replacement exists."""
        self.reflection_engine.deactivate_legacy_unknown_exit_rule(rule_id_prefix, pattern_key)

    def refresh_semantic_rules_if_stale(self) -> None:
        """Refresh semantic rules once when active rules still use unknown exit profiles."""
        self.reflection_engine.refresh_semantic_rules_if_stale()

    def _sanitize_rule_key(self, value: str) -> str:
        """Create a stable Chroma ID suffix from a composite rule key."""
        return self.exit_profiles.sanitize_rule_key(value)

    def _normalize_close_reason(self, reason: Any) -> str:
        """Normalize exit reasons while preserving stop-loss semantics."""
        return self.pattern_analyzer.normalize_close_reason(reason)

    def _is_stop_loss_reason(self, reason: Any) -> bool:
        """Return whether a close reason represents any stop-loss style exit."""
        return self.pattern_analyzer.is_stop_loss_reason(reason)

    def _adx_bucket_for_meta(self, meta: dict[str, Any]) -> str:
        """Classify ADX into the reflection buckets used for rule keys."""
        return self.pattern_analyzer.adx_bucket_for_meta(meta)

    @staticmethod
    def _adx_level_from_bucket(adx_bucket: str) -> str:
        """Format an ADX bucket for rule text."""
        return TradePatternAnalyzer.adx_level_from_bucket(adx_bucket)

    def _build_exit_profile_key(self, meta: dict[str, Any]) -> str:
        """Build a deterministic key for hard/soft SL/TP execution settings."""
        return self.exit_profiles.build_exit_profile_key(meta)

    @staticmethod
    def _format_exit_profile(
        stop_type: str,
        stop_interval: str,
        take_profit_type: str,
        take_profit_interval: str,
    ) -> str:
        """Format a hard/soft SL/TP profile for rules and dashboard metadata."""
        return ExitProfileResolver.format_exit_profile(
            stop_type,
            stop_interval,
            take_profit_type,
            take_profit_interval,
        )

    def _meta_values(
        self,
        metas: list[dict[str, Any]],
        key: str,
        *,
        absolute: bool = False,
        positive_only: bool = False,
        non_zero: bool = False,
    ) -> list[float]:
        """Return normalized numeric metadata values with optional filtering."""
        return self.pattern_analyzer.meta_values(
            metas,
            key,
            absolute=absolute,
            positive_only=positive_only,
            non_zero=non_zero,
        )

    def _average_meta(self, metas: list[dict[str, Any]], key: str, **filters: Any) -> float:
        """Average a numeric metadata field after applying optional filters."""
        return self.pattern_analyzer.average_meta(metas, key, **filters)

    def _compute_loss_diagnostics(self, losses: list[dict[str, Any]]) -> dict[str, float]:
        """Compute reused diagnostic averages for losing trades."""
        return self.pattern_analyzer.compute_loss_diagnostics(losses)

    def _build_rule_metadata(
        self,
        rule_type: str,
        source_pattern: str,
        metrics: dict[str, Any],
        **extra: Any,
    ) -> dict[str, Any]:
        """Build common semantic-rule metadata for best, loss, and AI-mistake rules."""
        return self.pattern_analyzer.build_rule_metadata(rule_type, source_pattern, metrics, **extra)

    def _is_sideways_failure(self, meta: dict[str, Any]) -> bool:
        """Identify losses or flat outcomes where the market failed to trend."""
        return self.pattern_analyzer.is_sideways_failure(meta)

    def _classify_ai_mistake(self, meta: dict[str, Any]) -> str:
        """Classify whether an outcome contradicts the AI's entry confidence."""
        return self.pattern_analyzer.classify_ai_mistake(meta)

    def _derive_ai_assumption(self, metas: list[dict[str, Any]]) -> str:
        """Summarize the failed assumption from stored AI reasoning text."""
        return self.pattern_analyzer.derive_ai_assumption(metas)

    def _derive_ai_mistake_reason(
        self,
        mistake_metas: list[dict[str, Any]],
        mistake_type: str,
        failed_assumption: str,
    ) -> str:
        """Explain what the AI got wrong for a repeated mistake cluster."""
        return self.pattern_analyzer.derive_ai_mistake_reason(
            mistake_metas,
            mistake_type,
            failed_assumption,
        )

    def _derive_ai_mistake_adjustment(
        self,
        mistake_metas: list[dict[str, Any]],
        mistake_type: str,
    ) -> str:
        """Generate prompt-ready corrections for repeated AI judgment mistakes."""
        return self.pattern_analyzer.derive_ai_mistake_adjustment(mistake_metas, mistake_type)

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

    def _compute_group_metrics(self, group_metas: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute outcome statistics for a group of trade metadata records."""
        return self.pattern_analyzer.compute_group_metrics(group_metas)

    def _derive_failure_reason(self, metrics: dict[str, Any]) -> str:
        """Derive the primary cause of losses from trade group metrics."""
        return self.pattern_analyzer.derive_failure_reason(metrics)

    def _derive_recommended_adjustment(self, metrics: dict[str, Any]) -> str:
        """Generate actionable guidance to improve profitability for this pattern."""
        return self.pattern_analyzer.derive_recommended_adjustment(metrics)

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
