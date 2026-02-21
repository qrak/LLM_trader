"""Protocol definition for ModelManager interface"""

import io
from typing import Protocol, Optional, Union, Tuple, TYPE_CHECKING, Dict, List

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
        prepared_messages: List[Dict[str, str]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Send a prompt to the model and get a response.
        """
        ...

    async def send_prompt_streaming(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Send a prompt to the model and get a streaming response.
        """
        ...

    async def send_prompt_with_chart_analysis(
        self,
        prompt: str,
        chart_image: Union[io.BytesIO, bytes, str],
        system_message: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Send a prompt with chart image for pattern analysis.
        """
        ...

    def supports_image_analysis(self, provider_override: Optional[str] = None) -> bool:
        """
        Check if the selected provider supports image analysis.
        """
        ...

    def describe_provider_and_model(
        self,
        provider_override: Optional[str],
        model_override: Optional[str],
        *,
        chart: bool = False
    ) -> Tuple[str, str]:
        """
        Return provider + model description for logging and telemetry.
        """
        ...

    async def close(self) -> None:
        """Close all client connections and cleanup resources"""
        ...
