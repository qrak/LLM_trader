"""Dataclasses for trading system."""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.utils.data_utils import SerializableMixin

LONG_ENTRY_SIGNALS = frozenset({"BUY", "LONG"})
SHORT_ENTRY_SIGNALS = frozenset({"SELL", "SHORT"})
ENTRY_SIGNALS = LONG_ENTRY_SIGNALS | SHORT_ENTRY_SIGNALS
EXIT_SIGNALS = frozenset({"CLOSE", "CLOSE_LONG", "CLOSE_SHORT"})

#: Marker put into a CLOSE row's reasoning when the VENUE's fill price is not a price
#: the real market made (see executor_reconciliation / the executor's fill check).
#: Such a close is still booked — history must stay paired — but it is excluded from
#: statistics and from the trading brain: the 2026-10-17 sandbox take-profit filled
#: 4.7% below the real market and was counted as the best trade of the whole run.
FILL_DIVERGENCE_MARKER = "EXIT FILL DIVERGENCE"

#: ``fill_verification`` value the executor writes for such a fill.
FILL_VERIFICATION_DIVERGENT = "divergent"
#: ``fill_verification`` value of a fill whose price matched the real market.
FILL_VERIFICATION_VERIFIED = "verified"
#: ``fill_verification`` value meaning "there was nothing to compare against" —
#: a different finding from ``divergent``: the venue's price was never measured.
FILL_VERIFICATION_UNAVAILABLE = "unavailable"
#: How far back the executor's reference window may reach. A window that reaches
#: further would let a print from minutes ago pass as "the price at the fill".
MAX_MARKET_REFERENCE_WINDOW_SECONDS = 120.0
MARKET_VERIFICATION_FIELDS = (
    "fill_verification", "reference_price", "deviation_pct", "reference_source",
    "fill_timestamp", "reference_timestamp", "reference_window_start", "reference_window_end",
    "reference_window_seconds", "reference_symbol", "verification_reason",
    "reference_url", "reference_sample_count", "reference_low", "reference_high",
    "fill_checks", "verification_error",
)


def _reference_window_seconds(value: dict[str, Any]) -> float | None:
    """Seconds the executor's reference window reached back from the fill.

    The explicit field wins; otherwise the window is measured from its own bounds.
    ``None`` means the payload never stated one (legacy: a one-second window).
    """
    try:
        window = float(value["reference_window_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError):
        window = None
    if window is not None and math.isfinite(window) and window > 0:
        return window
    try:
        start = datetime.fromisoformat(str(value["reference_window_start"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(value["reference_window_end"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if start.tzinfo is None or end.tzinfo is None:
        return None
    span = (end - start).total_seconds()
    return span if math.isfinite(span) and span > 0 else None


def market_verification_is_verified(value: Any) -> bool:
    """A status without contemporaneous reference provenance is not verification.

    The reference print must sit INSIDE the window the executor reports, and that
    window must stay bounded — a widened window may not smuggle in a print from
    minutes away. A payload that states no window keeps the original one-second rule.
    """
    if not isinstance(value, dict) or value.get("fill_verification") != "verified":
        return False
    if value.get("reference_source") != "binance_spot_agg_trades":
        return False
    try:
        fill = datetime.fromisoformat(str(value["fill_timestamp"]).replace("Z", "+00:00"))
        reference = datetime.fromisoformat(str(value["reference_timestamp"]).replace("Z", "+00:00"))
        price = float(value["reference_price"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if fill.tzinfo is None or reference.tzinfo is None:
        return False
    window = _reference_window_seconds(value)
    if window is None:
        window = 1.0
    lag = (fill - reference).total_seconds()
    return (
        math.isfinite(price) and price > 0
        and window <= MAX_MARKET_REFERENCE_WINDOW_SECONDS
        and 0 <= lag <= window
    )


def execution_quality_is_eligible(value: Any) -> bool:
    """Legacy history remains readable; explicit quality requires affirmative eligibility."""
    if value is None:
        return True
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return False
    return isinstance(value, dict) and value.get("eligible") is True


def entry_direction(signal: str) -> str:
    """Return LONG/SHORT for an entry signal; raise ValueError when it opens nothing."""
    normalized = signal.upper()
    if normalized in LONG_ENTRY_SIGNALS:
        return "LONG"
    if normalized in SHORT_ENTRY_SIGNALS:
        return "SHORT"
    raise ValueError(f"{signal} is not an entry signal")


@dataclass(slots=True)
class MarketConditions(SerializableMixin):
    """Market state snapshot passed between trading, brain, and risk modules.

    Declared before Position on purpose: Position keeps the entry-time snapshot
    verbatim (conditions_at_entry) so the close path never has to rebuild it.
    """
    trend_direction: str = "NEUTRAL"
    adx: float = 0.0
    rsi: float = 50.0
    rsi_level: str = "NEUTRAL"
    volatility: str = "MEDIUM"
    atr: float = 0.0
    atr_percentage: float = 0.0
    macd_signal: str = "NEUTRAL"
    bb_position: str = "MIDDLE"
    volume_state: str = "NORMAL"
    is_weekend: bool = False
    market_sentiment: str = "NEUTRAL"
    order_book_bias: str = "BALANCED"
    fear_greed_index: int = 50
    trend_strength: float = 0.0
    timeframe_alignment: str | None = None
    choppiness: float | None = None
    vwap: float = 0.0
    mfi: float = 50.0
    cmf: float = 0.0
    bb_percent_b: float = 0.5
    chandelier_long: float = 0.0
    pfe: float = 0.0
    supertrend_direction: str = "NEUTRAL"
    social_sentiment_reddit: str = ""
    portfolio_pnl_pct: float = 0.0


@dataclass(slots=True)
class Position(SerializableMixin):

    """Represents an active trading position.

    Includes confluence_factors from entry for brain learning on close.
    """
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    entry_time: datetime
    confidence: str
    direction: str
    symbol: str
    conditions_at_entry: MarketConditions
    confluence_factors: tuple = field(default_factory=tuple)
    entry_fee: float | None = None
    entry_execution_quality: dict[str, Any] | None = None
    quote_amount: float = 0.0
    size_pct: float = 0.0
    atr_at_entry: float = 0.0
    atr_percentage_at_entry: float = 0.0
    volatility_level: str = "MEDIUM"
    sl_distance_pct: float = 0.0
    tp_distance_pct: float = 0.0
    rr_ratio_at_entry: float = 0.0
    adx_at_entry: float = 0.0
    rsi_at_entry: float = 50.0
    trend_direction_at_entry: str = "NEUTRAL"
    macd_signal_at_entry: str = "NEUTRAL"
    bb_position_at_entry: str = "MIDDLE"
    volume_state_at_entry: str = "NORMAL"
    market_sentiment_at_entry: str = "NEUTRAL"
    order_book_bias_at_entry: str = "BALANCED"
    stop_loss_type_at_entry: str = "unknown"
    stop_loss_check_interval_at_entry: str = "unknown"
    take_profit_type_at_entry: str = "unknown"
    take_profit_check_interval_at_entry: str = "unknown"
    max_drawdown_pct: float = 0.0
    max_profit_pct: float = 0.0
    regime_profile: str = "NEUTRAL"

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.direction == "LONG":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        return ((self.entry_price - current_price) / self.entry_price) * 100

    def update_metrics(self, current_price: float) -> None:
        """Update live performance metrics (MAE/MFE)."""
        pnl = self.calculate_pnl(current_price)

        if pnl < 0 and pnl < self.max_drawdown_pct:
            self.max_drawdown_pct = pnl

        if pnl > 0 and pnl > self.max_profit_pct:
            self.max_profit_pct = pnl

    def is_stop_hit(self, current_price: float) -> bool:
        """Check if stop loss is hit."""
        if self.direction == "LONG":
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def is_target_hit(self, current_price: float) -> bool:
        """Check if take profit is hit."""
        if self.direction == "LONG":
            return current_price >= self.take_profit
        return current_price <= self.take_profit


@dataclass(slots=True)
class TradeDecision(SerializableMixin):
    """Represents a trading decision from the AI."""
    timestamp: datetime
    symbol: str
    action: str
    confidence: str
    price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float = 0.0
    quote_amount: float = 0.0
    quantity: float = 0.0
    fee: float | None = None
    reasoning: str = ""
    indicators_json: str | dict[str, Any] | None = None
    order_id: str | None = None
    execution_quality: dict[str, Any] | None = None


@dataclass(slots=True)
class TradingMemory(SerializableMixin):
    """Rolling memory of recent trading decisions for context."""
    decisions: list[TradeDecision] = field(default_factory=list)
    max_decisions: int = 10

    @staticmethod
    def _decision_key(decision: TradeDecision) -> str:
        """Build a stable key for pairing decision annotations."""
        return f"{decision.timestamp.isoformat()}|{decision.action}|{decision.price:.8f}"

    @staticmethod
    def _extract_close_reason(reasoning: str) -> str | None:
        """Extract the raw close reason from persisted close reasoning text."""
        prefix = "Position closed: "
        if not reasoning.startswith(prefix):
            return None
        remainder = reasoning[len(prefix):]
        return remainder.split(".", 1)[0].strip() or None

    @staticmethod
    def _describe_close_reason(reason: str | None, pnl_pct: float | None) -> str | None:
        """Convert close reasons into explicit outcome-aware wording for prompts."""
        if not reason:
            return None

        normalized = reason.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized != "stop_loss":
            return reason.replace("_", " ")

        if pnl_pct is None:
            return reason
        if pnl_pct > 0:
            return "profit-protecting stop"
        if pnl_pct < 0:
            return "loss-cutting stop"
        return "breakeven stop"

    @classmethod
    def _format_recent_reasoning(cls, reasoning: str, pnl_pct: float | None) -> str:
        """Make close reasoning explicit about whether a stop protected profit or cut loss."""
        raw_reason = cls._extract_close_reason(reasoning)
        display_reason = cls._describe_close_reason(raw_reason, pnl_pct)
        if not raw_reason or not display_reason or display_reason == raw_reason:
            return reasoning
        return reasoning.replace(f"Position closed: {raw_reason}", f"Position closed: {display_reason}", 1)

    def add_decision(self, decision: TradeDecision) -> None:
        """Add a decision to memory, maintaining max size."""
        self.decisions.append(decision)
        if len(self.decisions) > self.max_decisions:
            self.decisions.pop(0)

    def get_context_summary(
        self,
        full_history: list["TradeDecision"] | None = None,
        initial_capital: float | None = None,
    ) -> str:
        """Generate a concise summary for prompt injection.
        Returns:
            Formatted summary of last 5 decisions with overall P&L data from all trades
        """
        if not self.decisions:
            return ""

        recent_source = full_history if full_history else self.decisions
        recent = recent_source[-5:]
        lines = []
        if recent:
            lines.append("## Recent Trading History (Last 5 Decisions):")

        history_to_analyze = full_history if full_history else self.decisions
        def _ensure_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        if full_history is not None and full_history is not self.decisions:
            history_to_analyze = sorted(history_to_analyze, key=lambda x: _ensure_utc(x.timestamp))
        total_pnl_quote = 0.0
        sum_trade_pnl_pct = 0.0
        total_entry_quote = 0.0
        closed_trades = 0
        winning_trades = 0
        close_pnl_by_key: dict[str, float] = {}

        open_position = None
        for decision in history_to_analyze:
            if decision.action in ENTRY_SIGNALS:
                open_position = decision
            elif decision.action in EXIT_SIGNALS and open_position:
                if (not execution_quality_is_eligible(decision.execution_quality)
                        or not execution_quality_is_eligible(open_position.execution_quality)
                        or FILL_DIVERGENCE_MARKER in decision.reasoning):
                    open_position = None
                    continue
                if open_position.action in LONG_ENTRY_SIGNALS:
                    pnl_pct = ((decision.price - open_position.price) / open_position.price) * 100
                    pnl_quote = (decision.price - open_position.price) * open_position.quantity
                else:
                    pnl_pct = ((open_position.price - decision.price) / open_position.price) * 100
                    pnl_quote = (open_position.price - decision.price) * open_position.quantity

                total_pnl_quote += pnl_quote
                sum_trade_pnl_pct += pnl_pct
                total_entry_quote += open_position.quote_amount or (open_position.price * open_position.quantity)
                closed_trades += 1
                if pnl_pct > 0:
                    winning_trades += 1
                close_pnl_by_key[self._decision_key(decision)] = pnl_pct
                open_position = None

        for decision in recent:
            time_str = decision.timestamp.strftime("%Y-%m-%d %H:%M")
            reasoning = self._format_recent_reasoning(
                decision.reasoning,
                close_pnl_by_key.get(self._decision_key(decision)),
            )
            lines.append(
                f"- [{time_str}] {decision.action} @ ${decision.price:,.2f} "
                f"(Conf: {decision.confidence}) - {reasoning}"
            )

        if closed_trades > 0:
            pnl_pct_basis = initial_capital if initial_capital is not None else total_entry_quote
            total_pnl_pct = (total_pnl_quote / pnl_pct_basis) * 100 if pnl_pct_basis > 0 else 0.0
            avg_pnl_pct = sum_trade_pnl_pct / closed_trades
            win_rate = (winning_trades / closed_trades) * 100
            lines.append("")
            lines.append(f"## Overall Performance ({closed_trades} Total Closed Trades):")
            lines.append(f"- Total P&L: ${total_pnl_quote:+,.2f} ({total_pnl_pct:+.2f}%)")
            lines.append(f"- Average P&L per Trade: {avg_pnl_pct:+.2f}%")
            lines.append(f"- Win Rate: {win_rate:.1f}% ({winning_trades}/{closed_trades} trades)")

        return "\n".join(lines)

@dataclass(frozen=True)
class LocalExitRequest:
    """A LOCAL exit condition that fired but must NOT be booked locally.

    Wave 3: the price that tripped the bracket is an observation, not an exit price.
    The request is what the monitor and the operator see while the local position
    stays exactly as it was, waiting for the executor's fill evidence.

    Wave 4: the analysis ``CLOSE`` signal produces the same object with
    ``reason="analysis_signal"`` and ``state="unknown"`` when the executor
    integration is off — there is no confirmation channel at all then, so nothing is
    booked and the position stays.

    Lives in ``data_models`` (not ``trading_strategy``) so the position-management
    mixin can build it without a circular import; ``trading_strategy`` re-exports it.
    """

    reason: str
    observed_price: float
    state: str
    intent_key: str
    order_id: str | None
    detail: str


@dataclass(slots=True)
class VectorSearchResult(SerializableMixin):
    """Represents a search result from VectorMemory."""
    id: str
    document: str
    similarity: float
    recency: float
    hybrid_score: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class ExitExecutionContext:
    """Normalized SL/TP execution settings for brain memory and query context."""
    stop_loss_type: str = "unknown"
    stop_loss_check_interval: str = "unknown"
    take_profit_type: str = "unknown"
    take_profit_check_interval: str = "unknown"

    def to_dict(self) -> dict[str, str]:
        """Return as a plain dict for spreading into metadata payloads."""
        return {
            "stop_loss_type": self.stop_loss_type,
            "stop_loss_check_interval": self.stop_loss_check_interval,
            "take_profit_type": self.take_profit_type,
            "take_profit_check_interval": self.take_profit_check_interval,
        }

    def with_defaults(self, defaults: "ExitExecutionContext") -> "ExitExecutionContext":
        """Return a new instance with unknown fields filled from *defaults*."""
        return ExitExecutionContext(
            stop_loss_type=self.stop_loss_type if self.stop_loss_type != "unknown" else defaults.stop_loss_type,
            stop_loss_check_interval=self.stop_loss_check_interval if self.stop_loss_check_interval != "unknown" else defaults.stop_loss_check_interval,
            take_profit_type=self.take_profit_type if self.take_profit_type != "unknown" else defaults.take_profit_type,
            take_profit_check_interval=self.take_profit_check_interval if self.take_profit_check_interval != "unknown" else defaults.take_profit_check_interval,
        )


@dataclass(slots=True)
class RiskAssessment(SerializableMixin):
    """Represents the calculated risk parameters for a trade."""
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    size_pct: float
    quote_amount: float
    entry_fee: float | None
    sl_distance_pct: float
    tp_distance_pct: float
    rr_ratio: float
    volatility_level: str
    regime_profile: str = "NEUTRAL"


@dataclass(slots=True)
class ClosedTradeResult(SerializableMixin):
    """Result of a closed trade for statistics calculation."""
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_quote: float
    quantity: float
    direction: str


@dataclass(slots=True)
class SessionCosts(SerializableMixin):
    """Cumulative session costs by provider."""
    openrouter: float = 0.0
    google: float = 0.0
    lmstudio: float = 0.0
    deepseek: float = 0.0

    @property
    def total(self) -> float:
        """Get total cost across all providers."""
        return self.openrouter + self.google + self.lmstudio + self.deepseek


@dataclass(slots=True)
class ProviderCostStats(SerializableMixin):
    """Persistent cost statistics for a single provider."""
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

@dataclass(slots=True, frozen=True)
class MarketSnapshot:
    """Classified market state handed to the brain/context layers.

    Replaces the 19-keyword block that used to be copied between the analysis
    engine, TradingBrainService, BrainContextProvider and the indicator
    classifiers — adding a field now means adding it once, here.
    """
    trend_direction: str = "NEUTRAL"
    adx: float = 0.0
    rsi: float = 50.0
    volatility_level: str = "MEDIUM"
    rsi_level: str = "NEUTRAL"
    macd_signal: str = "NEUTRAL"
    volume_state: str = "NORMAL"
    bb_position: str = "MIDDLE"
    is_weekend: bool = False
    market_sentiment: str = "NEUTRAL"
    order_book_bias: str = "BALANCED"
    exit_execution_context: ExitExecutionContext | None = None
    choppiness: float | None = None
    trend_strength: float = 0.0
    atr_percentage: float = 0.0
    mfi: float | None = None
    cmf: float | None = None
    vwap: float = 0.0
    supertrend_direction: str = "NEUTRAL"

    @classmethod
    def from_conditions(
        cls,
        conditions: "MarketConditions",
        exit_execution_context: "ExitExecutionContext | None" = None,
    ) -> "MarketSnapshot":
        """Build a snapshot from a stored MarketConditions record."""
        return cls(
            trend_direction=conditions.trend_direction,
            adx=float(conditions.adx),
            volatility_level=conditions.volatility,
            rsi_level=conditions.rsi_level,
            macd_signal=conditions.macd_signal,
            volume_state=conditions.volume_state,
            bb_position=conditions.bb_position,
            is_weekend=conditions.is_weekend,
            market_sentiment=conditions.market_sentiment,
            order_book_bias=conditions.order_book_bias,
            exit_execution_context=exit_execution_context,
        )

