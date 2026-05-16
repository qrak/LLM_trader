"""Analysis Result Processor.

Handles the processing and formatting of analysis results from the AI models.
"""
from __future__ import annotations
import io
import re
from typing import Any, Union, TYPE_CHECKING

from src.logger.logger import Logger
from src.analyzer.trend_validator import TrendValidator
from src.analyzer.pattern_quality_scorer import PatternQualityScorer

if TYPE_CHECKING:
    from src.analyzer.analysis_context import AnalysisContext
    from src.contracts.model_contract import ModelManagerProtocol


class AnalysisResultProcessor:
    """Processes and formats market analysis results from AI models"""

    def __init__(self, model_manager: "ModelManagerProtocol", logger: Logger, unified_parser=None):
        """Initialize the processor"""
        self.model_manager = model_manager
        self.logger = logger
        self.unified_parser = unified_parser
        self.context: "AnalysisContext" | None = None
        self._trend_validator = TrendValidator()
        self._quality_scorer = PatternQualityScorer()

    async def process_analysis(self, system_prompt: str, prompt: str,
                              chart_image: Union[io.BytesIO, bytes, str] | None = None,
                              provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
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
            except (ValueError, Exception) as chart_error:  # pylint: disable=broad-exception-caught
                self.logger.warning("Chart analysis failed: %s. Falling back to text-only analysis.", chart_error)
                complete_response = await self.model_manager.send_prompt_streaming(
                    prompt=prompt,
                    system_message=system_prompt,
                    provider=provider,
                    model=model
                )
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
        self._log_response_validation(parsed_response.get("response_validation"))
        # Log the analysis result
        self._log_analysis_result(parsed_response)

        # Validate LLM claims against computed data (trend ADX + pattern quality)
        self._validate_llm_claims(parsed_response)

        # Format the final response
        return self._format_analysis_response(parsed_response, cleaned_response)

    def _log_analysis_result(self, parsed_response: dict[str, Any]) -> None:
        """Log analysis result information"""
        if "analysis" in parsed_response:
            analysis = parsed_response["analysis"]

            # Check if this is trading analysis (has signal field)
            if "signal" in analysis:
                signal = analysis.get("signal", "UNKNOWN")
                confidence = analysis.get("confidence", 0)
                trend_info = analysis.get("trend", {})
                # No isinstance needed - analysis.get() with default {} always returns a dict
                direction = trend_info.get("direction", "UNKNOWN")

                # Try legacy 'strength' field first, then prefer daily (macro), fall back to 4h
                strength = trend_info.get("strength")
                if strength is None:
                    strength = trend_info.get("strength_daily", trend_info.get("strength_4h", 0))

                # Log confluence factors if available (Chain-of-Thought scoring)
                confluence_factors = analysis.get("confluence_factors", {})
                if confluence_factors:
                    cf_str = ", ".join([f"{k}={v}" for k, v in confluence_factors.items()])
                    self.logger.debug("Trading analysis complete: Signal %s, Confidence %s, Trend %s (%s%% strength) | Confluence: %s", signal, confidence, direction, strength, cf_str)
                else:
                    self.logger.debug("Trading analysis complete: Signal %s, Confidence %s, Trend %s (%s%% strength)", signal, confidence, direction, strength)
            else:
                # Legacy analysis format
                bias = analysis.get("technical_bias", "UNKNOWN")
                trend = analysis.get("observed_trend", "UNKNOWN")
                confidence = analysis.get("confidence_score", 0)
                self.logger.debug("Analysis complete: Technical bias %s with %s trend (%s%% confidence)", bias, trend, confidence)
        else:
            self.logger.warning("Analysis complete but response format may be incomplete")

    def _log_response_validation(self, validation: dict[str, Any] | None) -> None:
        """Log response-contract validation metadata without blocking legacy parsing."""
        if not validation:
            return
        status = validation.get("status")
        if status == "valid":
            self.logger.debug("AI response contract validation passed: %s", validation.get("schema"))
            return
        if status == "invalid":
            errors = validation.get("errors", [])
            self.logger.warning("AI response contract validation failed: %s", errors[:3])
            return
        self.logger.debug("AI response contract validation skipped: no trading signal found")

    def _format_analysis_response(self, parsed_response: dict[str, Any],
                                cleaned_response: str) -> dict[str, Any]:
        """Format the final analysis response."""
        parsed_response["raw_response"] = cleaned_response
        if self.context is not None:
            parsed_response["current_price"] = self.context.current_price
        return parsed_response

    @staticmethod
    def _clean_response(text: str) -> str:
        """Remove thinking sections and extra whitespace from AI responses"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _validate_llm_claims(self, parsed_response: dict[str, Any]) -> None:
        """Validate LLM-reported ADX strengths and pattern quality against computed data.

        Runs after parsing but before the response is returned. Overwrites
        LLM values with computed ones when discrepancies exceed thresholds.
        Logs warnings for any significant gaps found.
        """
        if self.context is None:
            return

        analysis = parsed_response.get("analysis")
        if not analysis:
            return

        trend = analysis.get("trend", {})
        tech_data = self.context.technical_data or {}
        long_term_data = self.context.long_term_data or {}

        # --- Trend ADX Validation ---
        trend_result = self._trend_validator.validate(
            strength_4h=trend.get("strength_4h"),
            strength_daily=trend.get("strength_daily"),
            computed_adx=tech_data.get("adx"),
            computed_daily_adx=long_term_data.get("daily_adx") if long_term_data else None,
        )

        if trend_result.has_computed_data:
            # Always overwrite with validated values (computed beats LLM guess)
            self._trend_validator.overwrite_llm_trend(analysis, trend_result)

            if not trend_result.passed:
                for d in trend_result.discrepancies:
                    self.logger.warning("ADX validation: %s", d)
            else:
                self.logger.debug(
                    "ADX validation passed: 4H=%s daily=%s",
                    trend_result.validated_4h, trend_result.validated_daily,
                )

        # --- Pattern Quality Validation ---
        llm_quality_raw = analysis.get("pattern_quality")
        try:
            llm_quality = float(llm_quality_raw) if llm_quality_raw is not None else None
        except (TypeError, ValueError):
            llm_quality = None

        quality = self._quality_scorer.score(
            patterns=self.context.technical_patterns,
            tech_data=tech_data,
            llm_quality=llm_quality,
        )
        self._quality_scorer.overwrite_llm_quality(analysis, quality)

        if not quality.passed:
            for d in quality.discrepancies:
                self.logger.warning("Pattern quality validation: %s", d)
        else:
            self.logger.debug(
                "Pattern quality: computed=%s (%s)",
                round(quality.overall), quality.label,
            )
