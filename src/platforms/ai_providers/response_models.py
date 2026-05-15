"""Unified Pydantic response models for all AI providers."""
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MessageModel(BaseModel):
    """Message content from AI response."""
    role: str = "assistant"
    content: str = ""


class UsageModel(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceModel(BaseModel):
    """Single response choice from AI."""
    message: MessageModel
    finish_reason: str | None = None
    error: dict[str, Any] | None = None


class ChatResponseModel(BaseModel):
    """Unified response model for all AI providers."""
    choices: list[ChoiceModel]
    usage: UsageModel | None = None
    id: str | None = None
    model: str | None = None
    error: str | None = None

    @classmethod
    def from_error(cls, error: str) -> "ChatResponseModel":
        """Create error response."""
        return cls(choices=[], error=error)

    @classmethod
    def from_content(
        cls,
        content: str,
        role: str = "assistant",
        usage: UsageModel | None = None,
        model: str | None = None,
        response_id: str | None = None
    ) -> "ChatResponseModel":
        """Create response from content string."""
        return cls(
            choices=[ChoiceModel(message=MessageModel(role=role, content=content))],
            usage=usage,
            model=model,
            id=response_id
        )


class TradingSignal(str, Enum):
    """Allowed trading decision signals emitted by the analysis LLM."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    UPDATE = "UPDATE"


class TrendDirection(str, Enum):
    """Allowed market trend directions in the analysis response."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TimeframeAlignment(str, Enum):
    """Allowed timeframe alignment labels in the analysis response."""
    ALIGNED = "ALIGNED"
    MIXED = "MIXED"
    DIVERGENT = "DIVERGENT"


class ConfluenceFactorsModel(BaseModel):
    """Confluence scores used by trading-brain learning and dashboard logs."""
    model_config = ConfigDict(extra="allow")

    trend_alignment: float | None = Field(default=None, ge=0, le=100)
    momentum_strength: float | None = Field(default=None, ge=0, le=100)
    volume_support: float | None = Field(default=None, ge=0, le=100)
    pattern_quality: float | None = Field(default=None, ge=0, le=100)
    support_resistance_strength: float | None = Field(default=None, ge=0, le=100)


class KeyLevelsModel(BaseModel):
    """Support and resistance levels selected by the analysis response."""
    model_config = ConfigDict(extra="allow")

    support: list[float] = Field(default_factory=list)
    resistance: list[float] = Field(default_factory=list)


class TrendModel(BaseModel):
    """Trend summary emitted by the trading analysis response."""
    model_config = ConfigDict(extra="allow")

    direction: TrendDirection | None = None
    strength_4h: int | None = Field(default=None, ge=0, le=100)
    strength_daily: int | None = Field(default=None, ge=0, le=100)
    timeframe_alignment: TimeframeAlignment | None = None


class TradingAnalysisModel(BaseModel):
    """Validated shape for the `analysis` object in trading responses."""
    model_config = ConfigDict(extra="allow")

    signal: TradingSignal
    confidence: int = Field(ge=0, le=100)
    confluence_factors: ConfluenceFactorsModel | None = None
    entry_price: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    position_size: float | None = Field(default=None, ge=0, le=1)
    reasoning: str = ""
    key_levels: KeyLevelsModel | None = None
    trend: TrendModel | None = None
    risk_reward_ratio: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_signal_execution_fields(self) -> "TradingAnalysisModel":
        """Validate fields that are mandatory for immediate execution signals."""
        if self.signal in (TradingSignal.BUY, TradingSignal.SELL):
            required_fields = {
                "entry_price": self.entry_price,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "risk_reward_ratio": self.risk_reward_ratio,
                "position_size": self.position_size,
            }
            missing_fields = [field_name for field_name, field_value in required_fields.items() if field_value is None]
            if missing_fields:
                raise ValueError(
                    f"{self.signal.value} response is missing required execution fields: {', '.join(missing_fields)}"
                )
        if self.signal == TradingSignal.UPDATE:
            if self.entry_price is None:
                raise ValueError("UPDATE response requires entry_price to represent the current price")
            if self.stop_loss is None and self.take_profit is None:
                raise ValueError("UPDATE response must include a new stop_loss or take_profit")
        return self


class TradingAnalysisResponseModel(BaseModel):
    """Validated trading analysis response wrapper."""
    model_config = ConfigDict(extra="allow")

    schema_version: ClassVar[str] = "trading-analysis-response-v1"
    analysis: TradingAnalysisModel
