import json
import io
import re
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

from src.logger.logger import Logger
from src.utils.profiler import profile_performance
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
    
    def process_mock_analysis(self, symbol: str, current_price: float,
                              article_urls: Optional[Dict[str, str]] = None,
                              technical_history: Optional[Dict[str, Any]] = None,
                              technical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process mock analysis for testing purposes"""
        self.logger.debug("Generating mock analysis instead of calling AI model")
        
        # Generate minimal mock data for trading (no dependency on non-existent src.html.mock)
        def get_mock_analysis_data(sym, price):
            return {
                "analysis": {
                    "summary": f"Mock trading analysis for {sym} at ${price}",
                    "confidence_score": 50,
                    "action": "HOLD",
                    "entry_price": price,
                    "stop_loss": price * 0.95,
                    "take_profit": price * 1.10
                },
                "markdown_content": f"## Mock Trading Analysis for {sym}\\n\\nCurrent Price: ${price}\\n\\nRecommendation: HOLD",
                "article_urls": {}
            }

        # Get all mock data (JSON analysis, Markdown, article URLs)
        mock_data = get_mock_analysis_data(symbol, current_price)
        
        # Compose the JSON part for the structured analysis and include indicators if available
        analysis_obj = mock_data["analysis"].copy()
        if technical_history is not None:
            analysis_obj["technical_history"] = {}
            # convert numpy arrays to lists for JSON serialization where applicable
            for k, v in technical_history.items():
                try:
                    analysis_obj["technical_history"][k] = safe_tolist(v)
                except Exception:
                    analysis_obj["technical_history"][k] = str(type(v))

        if technical_data is not None:
            analysis_obj["technical_data"] = technical_data

        # Create a JSON-serializable copy for the raw_response
        serializable_analysis = serialize_for_json(analysis_obj)
        mock_json_string = json.dumps({"analysis": serializable_analysis}, indent=2)

        # Use the generated Markdown content and append indicators if available
        mock_markdown_content = mock_data["markdown_content"]

        # Build compact indicators section to append to markdown
        indicators_section = ""
        try:
            if technical_data:
                indicators_section += "\n\n### Live Indicator Snapshot\n"
                for k, v in technical_data.items():
                    indicators_section += f"- **{k}**: {v}\n"

            if technical_history:
                indicators_section += "\n### Indicator Series (last 5 values each)\n"
                for k, series in list(technical_history.items())[:20]:
                    try:
                        vals = safe_tolist(series)
                        if isinstance(vals, list):
                            sample = vals[-5:]
                        else:
                            sample = list(vals)[-5:] if hasattr(vals, '__iter__') else [vals]
                        indicators_section += f"- **{k}** (last 5): {sample}\n"
                    except Exception:
                        indicators_section += f"- **{k}**: [unserializable]\n"
        except Exception:
            indicators_section = ""

        if indicators_section:
            mock_markdown_content = mock_markdown_content + "\n\n" + indicators_section
        # Use the mock article URLs or the provided ones
        mock_article_urls = article_urls or mock_data["article_urls"]

        mock_analysis = {
            "analysis": analysis_obj, # Use the structured dict here (with indicators)
            "raw_response": f"```json\n{mock_json_string}\n```\n{mock_markdown_content}",
            "article_urls": mock_article_urls,
            "technical_history_included": technical_history is not None,
            "technical_data_included": technical_data is not None
        }
        
        # Normalize numeric fields in mock analysis to match real analysis behavior
        mock_analysis = self.unified_parser._normalize_numeric_fields(mock_analysis)
        # Also attempt to parse the mock raw_response using the real parser to surface parsing issues
        try:
            parsed = self.unified_parser.parse_ai_response(mock_analysis["raw_response"])
            mock_analysis["parsed_response"] = parsed
            mock_analysis["parse_valid"] = self.unified_parser.validate_ai_response(parsed)
            if not mock_analysis["parse_valid"]:
                mock_analysis["parse_error"] = "Parsed response failed validation"
        except Exception as e:
            self.logger.error(f"Error while parsing mock response: {e}")
            mock_analysis["parse_exception"] = str(e)

        return mock_analysis
        
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
