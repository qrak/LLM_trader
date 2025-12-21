import numpy as np
from typing import Optional, Any

from src.logger.logger import Logger
from ..core.analysis_context import AnalysisContext
from ..calculations.technical_calculator import TechnicalCalculator
from .template_manager import TemplateManager
from ..formatting.market_formatter import MarketFormatter
from ..formatting.technical_formatter import TechnicalFormatter
from .context_builder import ContextBuilder


class PromptBuilder:
    def __init__(self, timeframe: str = "1h", logger: Optional[Logger] = None, technical_calculator: Optional[TechnicalCalculator] = None, config: Any = None, format_utils=None, data_processor=None) -> None:
        """Initialize the PromptBuilder
        
        Args:
            timeframe: The primary timeframe for analysis (e.g. "1h")
            logger: Optional logger instance for debugging
            technical_calculator: Calculator for technical indicators
        """
        self.timeframe = timeframe
        self.logger = logger
        self.custom_instructions: list[str] = []
        self.language: Optional[str] = None
        self.context: Optional[AnalysisContext] = None
        self.technical_calculator = technical_calculator or TechnicalCalculator(logger, format_utils)
        self.config = config
        self.format_utils = format_utils
        self.data_processor = data_processor
        
        # Access indicator thresholds from the calculator
        self.INDICATOR_THRESHOLDS = self.technical_calculator.INDICATOR_THRESHOLDS
        
        # Initialize component managers
        self.template_manager = TemplateManager(config=self.config, logger=logger)
        self.market_formatter = MarketFormatter(logger, format_utils)
        self.technical_analysis_formatter = TechnicalFormatter(self.technical_calculator, logger, format_utils)
        self.context_builder = ContextBuilder(timeframe, logger, format_utils, data_processor)

    def build_prompt(self, context: AnalysisContext, has_chart_analysis: bool = False) -> str:
        """Build the complete prompt using component managers.
        
        Args:
            context: Analysis context containing all required data
            has_chart_analysis: Whether chart image analysis is available
            
        Returns:
            str: Complete formatted prompt
        """
        self.context = context

        sections = [
            self.context_builder.build_trading_context(context),
            self.context_builder.build_sentiment_section(context.sentiment),
        ]

        # Add market overview first before technical analysis to give it more prominence
        if context.market_overview:
            sections.append(self.market_formatter.format_market_overview(
                context.market_overview,
                analyzed_symbol=context.symbol
            ))
            
            # Add ticker data from coin_data if available
            coin_data = context.market_overview.get("coin_data", {})
            if coin_data and context.symbol:
                # Extract base symbol (e.g., "BTC" from "BTC/USDT")
                base_symbol = context.symbol.split('/')[0]
                ticker_info = coin_data.get(base_symbol)
                if ticker_info:
                    ticker_section = self.market_formatter.format_ticker_data(ticker_info, context.symbol)
                    if ticker_section:
                        sections.append(ticker_section)
        
        # Add market microstructure data (order book, trades, funding rate)
        if context.market_microstructure:
            microstructure = context.market_microstructure
            
            # Add order book depth
            if "order_book" in microstructure and microstructure["order_book"]:
                ob_section = self.market_formatter.format_order_book_depth(microstructure["order_book"], context.symbol)
                if ob_section:
                    sections.append(ob_section)
            
            # Add trade flow
            if "recent_trades" in microstructure and microstructure["recent_trades"]:
                trades_section = self.market_formatter.format_trade_flow(microstructure["recent_trades"], context.symbol)
                if trades_section:
                    sections.append(trades_section)
            
            # Add funding rate (if futures contract)
            if "funding_rate" in microstructure and microstructure["funding_rate"]:
                funding_section = self.market_formatter.format_funding_rate(microstructure["funding_rate"], context.symbol)
                if funding_section:
                    sections.append(funding_section)

        # Add cryptocurrency details if available
        coin_details_section = self.context_builder.build_coin_details_section(
            context.coin_details
        )
        if coin_details_section:
            sections.append(coin_details_section)

        sections.extend([
            self.context_builder.build_market_data_section(context.ohlcv_candles),
            self.technical_analysis_formatter.format_technical_analysis(context, self.timeframe),
            self.context_builder.build_market_period_metrics_section(context.market_metrics),
        ])
        
        # Build long-term analysis section (daily + weekly)
        long_term_sections = []
        
        # Daily macro analysis
        if context.long_term_data:
            daily_section = self.context_builder.build_long_term_analysis_section(
                context.long_term_data, 
                context.current_price
            )
            if daily_section:
                long_term_sections.append(daily_section)
        
        # Weekly macro analysis (200W SMA)
        if context.weekly_macro_indicators and 'weekly_macro_trend' in context.weekly_macro_indicators:
            weekly_section = self.market_formatter._format_weekly_macro_section(
                context.weekly_macro_indicators['weekly_macro_trend']
            )
            if weekly_section:
                long_term_sections.append(weekly_section)
        
        if long_term_sections:
            sections.append("\n\n".join(long_term_sections))

        # Add custom instructions if available
        if self.custom_instructions:
            sections.append("\n".join(self.custom_instructions))

        # Check if we have advanced support/resistance detected
        advanced_support_resistance_detected = self._has_advanced_support_resistance()

        # Get available periods from context builder for dynamic prompt generation
        available_periods = self.context_builder._calculate_period_candles()

        # Add analysis steps right before response template
        sections.append(self.template_manager.build_analysis_steps(context.symbol, advanced_support_resistance_detected, has_chart_analysis, available_periods))

        # Response template should always be last
        sections.append(self.template_manager.build_response_template(has_chart_analysis))

        final_prompt = "\n\n".join(filter(None, sections))

        return final_prompt
    
    def build_system_prompt(self, symbol: str, has_chart_image: bool = False) -> str:
        """Build system prompt using template manager.
        
        Args:
            symbol: Trading symbol
            has_chart_image: Whether a chart image is being provided
            
        Returns:
            str: Formatted system prompt
        """
        return self.template_manager.build_system_prompt(symbol, self.timeframe, self.language, has_chart_image)

    def add_custom_instruction(self, instruction: str) -> None:
        """Add custom instruction to the prompt.
        
        Args:
            instruction: Custom instruction to add
        """
        self.custom_instructions.append(instruction)
    
    def _has_advanced_support_resistance(self) -> bool:
        """Check if advanced support/resistance indicators are detected.
        
        Returns:
            bool: True if advanced S/R indicators are available and valid
        """
        td = self.context.technical_data
        
        # Get advanced indicators with defaults
        adv_support = td.get('advanced_support', np.nan)
        adv_resistance = td.get('advanced_resistance', np.nan)
        
        # Handle array indicators - take the last value
        try:
            if len(adv_support) > 0:
                adv_support = adv_support[-1]
        except TypeError:
            # adv_support is already a scalar value
            pass
            
        try:
            if len(adv_resistance) > 0:
                adv_resistance = adv_resistance[-1]
        except TypeError:
            # adv_resistance is already a scalar value
            pass
        
        # Both values must be valid (not NaN)
        return not np.isnan(adv_support) and not np.isnan(adv_resistance)
