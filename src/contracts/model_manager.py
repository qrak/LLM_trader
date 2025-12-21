"""Protocol definition for ModelManager interface"""

import io
from typing import Protocol, Optional, Dict, Any, List, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.token_counter import TokenCounter


class ModelManagerProtocol(Protocol):
    """
    Protocol defining the interface for AI model management.
    
    This protocol allows for dependency injection and testing by defining
    the contract that any ModelManager implementation must fulfill.
    """
    
    token_counter: "TokenCounter"
    
    async def send_prompt_streaming(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Send a prompt to the model and get a streaming response.
        
        Args:
            prompt: User prompt
            system_message: Optional system instructions
            provider: Optional provider override (admin only)
            model: Optional model override (admin only)
            
        Returns:
            Complete response text from the AI model
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
        
        Args:
            prompt: User prompt
            chart_image: Chart image data
            system_message: Optional system instructions
            provider: Optional provider override (admin only)
            model: Optional model override (admin only)
            
        Returns:
            Response text from the AI model
        """
        ...
    
    def supports_image_analysis(self, provider_override: Optional[str] = None) -> bool:
        """
        Check if the selected provider supports image analysis.
        
        Args:
            provider_override: Optional provider to check instead of default
            
        Returns:
            True if image analysis is supported, False otherwise
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
        
        Args:
            provider_override: Optional provider override
            model_override: Optional model override
            chart: Whether this is for chart analysis
            
        Returns:
            Tuple of (provider_name, model_name)
        """
        ...
    
    async def close(self) -> None:
        """Close all client connections and cleanup resources"""
        ...
