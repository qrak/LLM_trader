import json
import io
import re
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

from src.logger.logger import Logger
from src.utils.serialize import serialize_for_json, safe_tolist

if TYPE_CHECKING:
    from src.contracts.manager_factory import ModelManagerProtocol


class AnalysisResultProcessor:
    """Processes and formats market analysis results from AI models"""
    
    def __init__(self, model_manager: "ModelManagerProtocol", logger: Logger, unified_parser=None):
        """Initialize the processor"""
        if unified_parser is None:
            raise ValueError("unified_parser is required - must be injected from app.py")
        self.model_manager = model_manager
        self.logger = logger
        self.unified_parser = unified_parser
        
    async def process_analysis(self, system_prompt: str, prompt: str, 
                              chart_image: Optional[Union[io.BytesIO, bytes, str]] = None,
                              provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Process analysis by sending prompts to AI model and formatting response.
        
        Args:
            system_prompt: System instructions for the AI model
            prompt: User prompt for analysis
            chart_image: Optional chart image for visual analysis
            provider: Optional provider override (admin only)
            model: Optional model override (admin only)
            
        Returns:
            Dictionary containing formatted analysis results
        """
        # Send the prompt to the model
        self.logger.debug("Sending prompt to AI model for analysis")
        
        # Use chart analysis if image is provided and model supports it
        use_chart_analysis = chart_image is not None and self.model_manager.supports_image_analysis(provider)
        if use_chart_analysis:
            prov_name, model_name = self.model_manager.describe_provider_and_model(provider, model, chart=True)
            prov_label = prov_name.upper() if prov_name else "UNKNOWN"
            self.logger.info(
                "Using chart image analysis via %s (model: %s)",
                prov_label,
                model_name
            )
            try:
                complete_response = await self.model_manager.send_prompt_with_chart_analysis(
                    prompt=prompt,
                    chart_image=chart_image,
                    system_message=system_prompt,
                    provider=provider,
                    model=model
                )
            except ValueError:
                # Chart analysis failed, re-raise to let analysis engine handle logging and fallback
                raise
        else:
            # Use the standard send_prompt_streaming method
            complete_response = await self.model_manager.send_prompt_streaming(
                prompt=prompt,
                system_message=system_prompt,
                provider=provider,
                model=model
            )
        
        self.logger.debug("Received response from AI model")
        cleaned_response = self._clean_response(complete_response)
        
        parsed_response = self.unified_parser.parse_ai_response(cleaned_response)
        
        if not self.unified_parser.validate_ai_response(parsed_response):
            self.logger.warning("Invalid response format from AI model")
            return {
                "error": "Invalid response format",
                "raw_response": cleaned_response
            }
        
        # Log the analysis result
        self._log_analysis_result(parsed_response)
        
        # Format the final response
        return self._format_analysis_response(parsed_response, cleaned_response)
    

        
    def _log_analysis_result(self, parsed_response: Dict[str, Any]) -> None:
        """Log analysis result information"""
        if "analysis" in parsed_response:
            analysis = parsed_response["analysis"]
            
            # Check if this is trading analysis (has signal field)
            if "signal" in analysis:
                signal = analysis.get("signal", "UNKNOWN")
                confidence = analysis.get("confidence", 0)
                trend_info = analysis.get("trend", {})
                direction = trend_info.get("direction", "UNKNOWN") if isinstance(trend_info, dict) else "UNKNOWN"
                strength = trend_info.get("strength", 0) if isinstance(trend_info, dict) else 0
                
                # Log confluence factors if available (Chain-of-Thought scoring)
                confluence_factors = analysis.get("confluence_factors", {})
                if confluence_factors and isinstance(confluence_factors, dict):
                    cf_str = ", ".join([f"{k}={v}" for k, v in confluence_factors.items()])
                    self.logger.debug(
                        f"Trading analysis complete: Signal {signal}, Confidence {confidence}, "
                        f"Trend {direction} ({strength}% strength) | Confluence: {cf_str}"
                    )
                else:
                    self.logger.debug(
                        f"Trading analysis complete: Signal {signal}, Confidence {confidence}, "
                        f"Trend {direction} ({strength}% strength)"
                    )
            else:
                # Legacy analysis format
                bias = analysis.get("technical_bias", "UNKNOWN")
                trend = analysis.get("observed_trend", "UNKNOWN")
                confidence = analysis.get("confidence_score", 0)
                self.logger.debug(f"Analysis complete: Technical bias {bias} with {trend} trend ({confidence}% confidence)")
        else:
            self.logger.warning("Analysis complete but response format may be incomplete")
            
    def _format_analysis_response(self, parsed_response: Dict[str, Any], 
                                cleaned_response: str) -> Dict[str, Any]:
        """Format the final analysis response"""
        parsed_response["raw_response"] = cleaned_response
        
        # Include current_price if available in context
        if hasattr(self, 'context') and hasattr(self.context, 'current_price'):
            parsed_response["current_price"] = self.context.current_price
        
        # Return formatted response - article_urls will be added by the caller
        return parsed_response
    
    @staticmethod
    def _clean_response(text: str) -> str:
        """Remove thinking sections and extra whitespace from AI responses"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
