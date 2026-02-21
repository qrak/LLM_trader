"""Unified Pydantic response models for all AI providers."""
from typing import Optional, List, Dict, Any

from pydantic import BaseModel


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
    finish_reason: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class ChatResponseModel(BaseModel):
    """Unified response model for all AI providers."""
    choices: List[ChoiceModel]
    usage: Optional[UsageModel] = None
    id: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_error(cls, error: str) -> "ChatResponseModel":
        """Create error response."""
        return cls(choices=[], error=error)

    @classmethod
    def from_content(
        cls,
        content: str,
        role: str = "assistant",
        usage: Optional[UsageModel] = None,
        model: Optional[str] = None,
        response_id: Optional[str] = None
    ) -> "ChatResponseModel":
        """Create response from content string."""
        return cls(
            choices=[ChoiceModel(message=MessageModel(role=role, content=content))],
            usage=usage,
            model=model,
            id=response_id
        )
