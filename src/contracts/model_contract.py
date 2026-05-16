"""Protocol definition for ModelManager interface"""

import io
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.token_counter import TokenCounter


class ModelManagerProtocol(Protocol):
    """
    Protocol defining the interface for AI model management.

    This protocol allows for dependency injection and testing by defining
    the contract that any ModelManager implementation must fulfill.
    """

    token_counter: "TokenCounter"

    async def send_prompt(
        self,
        prompt: str,
        system_message: str = None,
        prepared_messages: list[dict[str, str]] | None = None,
        provider: str | None = None,
        model: str | None = None
    ) -> str:
        """
        Send a prompt to the model and get a response.
        """
        ...

    async def send_prompt_streaming(
        self,
        prompt: str,
        system_message: str | None = None,
        provider: str | None = None,
        model: str | None = None
    ) -> str:
        """
        Send a prompt to the model and get a streaming response.
        """
        ...

    async def send_prompt_with_chart_analysis(
        self,
        prompt: str,
        chart_image: io.BytesIO | bytes | str,
        system_message: str | None = None,
        provider: str | None = None,
        model: str | None = None
    ) -> str:
        """
        Send a prompt with chart image for pattern analysis.
        """
        ...

    def supports_image_analysis(self, provider_override: str | None = None) -> bool:
        """
        Check if the selected provider supports image analysis.
        """
        ...

    def describe_provider_and_model(
        self,
        provider_override: str | None,
        model_override: str | None,
        *,
        chart: bool = False
    ) -> tuple[str, str]:
        """
        Return provider + model description for logging and telemetry.
        """
        ...

    async def close(self) -> None:
        """Close all client connections and cleanup resources"""
        ...
