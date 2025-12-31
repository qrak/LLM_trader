from typing import Dict, Any, Optional, TYPE_CHECKING
import io

from src.utils.timeframe_validator import TimeframeValidator
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.coingecko import CoinGeckoAPI
from src.utils.profiler import profile_performance
from .analysis_context import AnalysisContext
from .data_fetcher import DataFetcher
from src.logger.logger import Logger


import numpy as np

if TYPE_CHECKING:
    from src.contracts.manager_factory import ModelManagerProtocol
    from src.config.protocol import ConfigProtocol
    from src.rag import RagEngine


class AnalysisEngine:
    """Orchestrates market data collection, analysis, and publication"""
    
    def __init__(
        self,
        logger: Logger,
        rag_engine: "RagEngine",
        coingecko_api: CoinGeckoAPI,
        model_manager: "ModelManagerProtocol",
        alternative_me_api: AlternativeMeAPI,
        market_api,
        config: "ConfigProtocol",
        technical_calculator=None,
        pattern_analyzer=None,
        prompt_builder=None,
        data_collector=None,
        metrics_calculator=None,
        result_processor=None,
        chart_generator=None
    ) -> None:
        """
        Initialize AnalysisEngine with injected dependencies (DI pattern).
        
        Args:
            logger: Logger instance
            rag_engine: RAG engine for news and context
            coingecko_api: CoinGecko API client
            model_manager: AI model manager (Protocol-based)
            alternative_me_api: Alternative.me API for Fear & Greed Index
            market_api: CryptoCompare Market API client
            format_utils: Formatting utilities
            data_processor: Data processing utilities
            config: Configuration instance (Protocol-based)
            technical_calculator: TechnicalCalculator instance (injected from app.py)
            pattern_analyzer: PatternAnalyzer instance (injected from app.py)
            prompt_builder: PromptBuilder instance (injected from app.py)
            data_collector: MarketDataCollector instance (injected from app.py)
            metrics_calculator: MarketMetricsCalculator instance (injected from app.py)
            result_processor: AnalysisResultProcessor instance (injected from app.py)
            chart_generator: ChartGenerator instance (injected from app.py)
        """
        self.logger = logger
        
        # Validate required config dependency
        if config is None:
            raise ValueError("config is a required parameter and cannot be None")
        self.config = config

        # Basic properties
        self.exchange = None
        self.symbol = None
        self.base_symbol = None
        self.context = None
        self.article_urls = {}
        self.last_analysis_result = None

        # Load configuration
        try:
            self.timeframe = self.config.TIMEFRAME
            self.limit = self.config.CANDLE_LIMIT
            
            # Validate timeframe
            if not TimeframeValidator.validate(self.timeframe):
                self.logger.warning(
                    f"Timeframe '{self.timeframe}' is not fully supported. "
                    f"Supported timeframes: {', '.join(TimeframeValidator.SUPPORTED_TIMEFRAMES)}. "
                    f"Proceeding but expect potential calculation errors."
                )
        except Exception as e:
            self.logger.exception(f"Error loading configuration values: {e}.")
            raise

        # Validate required dependencies
        if model_manager is None:
            raise ValueError("model_manager is a required parameter and cannot be None")
        if alternative_me_api is None:
            raise ValueError("alternative_me_api is a required parameter and cannot be None")
        if market_api is None:
            raise ValueError("market_api is a required parameter and cannot be None")
        if technical_calculator is None:
            raise ValueError("technical_calculator is required - must be injected from app.py")
        if pattern_analyzer is None:
            raise ValueError("pattern_analyzer is required - must be injected from app.py")
        if prompt_builder is None:
            raise ValueError("prompt_builder is required - must be injected from app.py")
        if data_collector is None:
            raise ValueError("data_collector is required - must be injected from app.py")
        if metrics_calculator is None:
            raise ValueError("metrics_calculator is required - must be injected from app.py")
        if result_processor is None:
            raise ValueError("result_processor is required - must be injected from app.py")
        if chart_generator is None:
            raise ValueError("chart_generator is required - must be injected from app.py")

        # Store injected components
        self.model_manager = model_manager
        self.technical_calculator = technical_calculator
        self.pattern_analyzer = pattern_analyzer
        self.prompt_builder = prompt_builder
        self.data_collector = data_collector
        self.metrics_calculator = metrics_calculator
        self.result_processor = result_processor
        self.chart_generator = chart_generator

        # Store references to external services
        self.rag_engine = rag_engine
        self.coingecko_api = coingecko_api
        self.market_api = market_api
        
        # Use the token counter from model_manager
        self.token_counter = self.model_manager.token_counter

    def initialize_for_symbol(self, symbol: str, exchange, timeframe=None) -> None:
        """
        Initialize the analyzer for a specific symbol and exchange.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            exchange: Exchange instance
            timeframe: Optional timeframe override (uses config default if None)
        """
        self.symbol = symbol
        self.exchange = exchange
        self.base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Use provided timeframe or fall back to config
        effective_timeframe = timeframe if timeframe else self.timeframe
        
        # Validate effective timeframe
        try:
            effective_timeframe = TimeframeValidator.validate_and_normalize(effective_timeframe)
        except ValueError as e:
            self.logger.warning(f"Timeframe validation failed: {e}. Using config default: {self.timeframe}")
            effective_timeframe = self.timeframe
        
        # Initialize analysis context with effective timeframe
        self.context = AnalysisContext(symbol)
        self.context.exchange = exchange.name if hasattr(exchange, 'name') else str(exchange)
        self.context.timeframe = effective_timeframe
        
        # Create data fetcher and initialize data collector
        data_fetcher = DataFetcher(exchange=exchange, logger=self.logger)
        
        self.data_collector.initialize(
            data_fetcher=data_fetcher, 
            symbol=symbol, 
            exchange=exchange,
            timeframe=effective_timeframe,
            limit=self.limit
        )
        
        # Update prompt builder and context builder with effective timeframe
        self.prompt_builder.timeframe = effective_timeframe
        
        if self.prompt_builder.context_builder:
            self.prompt_builder.context_builder.timeframe = effective_timeframe
        
        # Reset analysis state
        self.article_urls = {}
        self.last_analysis_result = None
        
        # Reset token counter for new analysis using the model manager's token counter
        self.token_counter.reset_session_stats()

    async def close(self) -> None:
        """Clean up resources with proper null checks"""
        try:
            if self.exchange is not None:
                await self.exchange.close()
                
            if self.model_manager is not None:
                await self.model_manager.close()
                
            if self.rag_engine is not None:
                await self.rag_engine.close()
        except Exception as e:
            self.logger.error(f"Error during MarketAnalyzer cleanup: {e}")

    @profile_performance
    async def analyze_market(
        self, 
        provider: Optional[str] = None, 
        model: Optional[str] = None,
        additional_context: Optional[str] = None,
        previous_response: Optional[str] = None,
        previous_indicators: Optional[Dict[str, Any]] = None,
        position_context: Optional[str] = None,
        performance_context: Optional[str] = None,
        brain_context: Optional[str] = None,
        last_analysis_time: Optional[str] = None,
        current_ticker: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate the complete market analysis workflow.
        
        Args:
            provider: Optional AI provider override (admin only)
            model: Optional AI model override (admin only)
            additional_context: Additional context to append to prompt (e.g., extra instructions)
            previous_response: Optional previous AI response for continuity
            previous_indicators: Optional previous technical indicator values for trend comparison
            position_context: Current position details and unrealized P&L (goes to system prompt)
            performance_context: Recent trading history and performance (goes to system prompt)
            brain_context: Distilled trading insights from closed trades (goes to system prompt)
            last_analysis_time: Formatted timestamp of last analysis (e.g., "2025-12-26 14:30:00")
            current_ticker: Optional dict containing current ticker data to avoid redundant API calls
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Step 1: Collect all required data
            if not await self._collect_market_data():
                return {"error": "Failed to collect market data", "details": "Data collection failed"}
            
            # Step 2: Enrich context with external data
            await self._enrich_market_context(current_ticker=current_ticker)
            
            # Step 3: Perform technical analysis
            await self._perform_technical_analysis()
            
            # Step 4: Generate AI analysis
            analysis_result = await self._generate_ai_analysis(provider, model, additional_context, previous_response, previous_indicators, position_context, performance_context, brain_context, last_analysis_time)
            
            # Store the result for later publication
            self.last_analysis_result = analysis_result
                
            # Reset custom instructions for next run
            self.prompt_builder.custom_instructions = []
            
            return analysis_result
            
        except Exception as e:
            self.logger.exception(f"Analysis failed: {e}")
            return {"error": str(e), "recommendation": "HOLD"}

    async def _collect_market_data(self) -> bool:
        """Collect market data using data collector"""
        data_result = await self.data_collector.collect_data(self.context)
        if not data_result["success"]:
            self.logger.error(f"Failed to collect market data: {data_result['errors']}")
            return False
        
        # Store article URLs
        self.article_urls = self.data_collector.article_urls
        
        # Add market context to prompt builder if available
        market_context = data_result.get("market_context")
        if market_context:
            self.prompt_builder.add_custom_instruction(market_context)
        else:
            self.logger.warning(f"No market context available for {self.symbol}")
            
        return True

    async def _enrich_market_context(self, current_ticker: Optional[Dict[str, Any]] = None) -> None:
        """
        Enrich market context with overview, microstructure, and coin details
        
        Args:
            current_ticker: Optional ticker data to reuse
        """
        # Fetch market overview
        try:
            market_overview = await self.rag_engine.get_market_overview()
            self.context.market_overview = market_overview
        except Exception as e:
            self.logger.warning(f"Failed to fetch market overview: {e}")
            self.context.market_overview = {}
        
        # Fetch market microstructure
        try:
            microstructure = await self.data_collector.data_fetcher.fetch_market_microstructure(
                self.symbol, 
                cached_ticker=current_ticker
            )
            self.context.market_microstructure = microstructure
        except Exception as e:
            self.logger.warning(f"Failed to fetch market microstructure: {e}")
            self.context.market_microstructure = {}
        
        # Fetch cryptocurrency details
        if self.market_api:
            try:
                coin_details = await self.market_api.get_coin_details(self.base_symbol)
                self.context.coin_details = coin_details
                if coin_details:
                    # self.logger.debug(f"Coin details for {self.base_symbol} fetched and added to context")
                    pass
                else:
                    self.logger.warning(f"No coin details found for {self.base_symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to fetch coin details for {self.base_symbol}: {e}")
                self.context.coin_details = {}

    async def _perform_technical_analysis(self) -> None:
        """Perform all technical analysis steps"""
        # Calculate technical indicators
        await self._calculate_technical_indicators()
        
        # Process long-term data
        await self._process_long_term_data()
        
        # Calculate market metrics
        data = self.data_collector.extract_ohlcv_data(self.context)
        self.metrics_calculator.update_period_metrics(data, self.context)
        
        # Run technical pattern analysis
        technical_patterns = self.pattern_analyzer.detect_patterns(
            self.context.ohlcv_candles,
            self.context.technical_history,
            self.context.long_term_data,
            self.context.timestamps
        )
        
        if any(technical_patterns.values()):
            self.context.technical_patterns = technical_patterns

    async def _generate_ai_analysis(
        self, 
        provider: Optional[str], 
        model: Optional[str],
        additional_context: Optional[str] = None,
        previous_response: Optional[str] = None,
        previous_indicators: Optional[Dict[str, Any]] = None,
        position_context: Optional[str] = None,
        performance_context: Optional[str] = None,
        brain_context: Optional[str] = None,
        last_analysis_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate AI analysis using prompt builder and result processor"""
        
        # Check if chart analysis is supported by the current provider
        has_chart_analysis = self.model_manager.supports_image_analysis(provider)
        chart_image: Optional[io.BytesIO] = None
        
        if has_chart_analysis:
            chart_image = await self._generate_chart_image()
            if chart_image is None:
                has_chart_analysis = False
                self.logger.warning("Chart generation failed, proceeding without chart analysis")
        
        system_prompt = self.prompt_builder.build_system_prompt(
            self.symbol, 
            has_chart_analysis, 
            previous_response,
            position_context,
            performance_context,
            brain_context,
            last_analysis_time
        )
        prompt = self.prompt_builder.build_prompt(
            context=self.context,
            has_chart_analysis=has_chart_analysis,
            additional_context=additional_context,
            previous_indicators=previous_indicators
        )
        # Process analysis
        if self.config.TEST_ENVIRONMENT:
            self.logger.debug("TEST_ENVIRONMENT is True - using mock analysis")
            analysis_result = self.result_processor.process_mock_analysis(
                self.symbol,
                self.context.current_price,
                self.article_urls,
                technical_history=self.context.technical_history,
                technical_data=self.context.technical_data
            )
        else:
            analysis_result = await self._execute_ai_request(
                system_prompt, prompt, provider, model, chart_image
            )
            
        # Add metadata to result
        analysis_result["article_urls"] = self.article_urls
        analysis_result["timeframe"] = self.context.timeframe
        actual_provider, actual_model = self.model_manager.describe_provider_and_model(provider, model, chart=has_chart_analysis)
        analysis_result["provider"] = actual_provider
        analysis_result["model"] = actual_model
        analysis_result["chart_analysis"] = has_chart_analysis
        
        # Add technical_data for persistence (will be saved to previous_response.json)
        if self.context.technical_data:
            analysis_result["technical_data"] = self.context.technical_data
        
        return analysis_result

    async def _execute_ai_request(
        self, 
        system_prompt: str, 
        prompt: str, 
        provider: Optional[str], 
        model: Optional[str],
        chart_image: Optional[io.BytesIO] = None
    ) -> Dict[str, Any]:
        """Execute the AI request with optional chart image for visual analysis.
        
        Args:
            system_prompt: System instructions for the AI
            prompt: User prompt with market data
            provider: Optional provider override
            model: Optional model override
            chart_image: Optional chart image for visual analysis
            
        Returns:
            Analysis result dictionary
        """
        if provider and model:
            self.logger.info(f"Using admin-specified provider: {provider}, model: {model}")
        
        # Give result processor access to context for current_price
        self.result_processor.context = self.context
        
        # Pass chart image to result processor (it will use chart analysis if image provided)
        return await self.result_processor.process_analysis(
            system_prompt, prompt, chart_image=chart_image, provider=provider, model=model
        )
    
    async def _generate_chart_image(self) -> Optional[io.BytesIO]:
        """Generate chart image for AI visual analysis.
        
        Returns:
            BytesIO containing PNG chart image, or None if generation fails
        """
        try:
            # Get OHLCV data from context
            if self.context.ohlcv_candles is None or len(self.context.ohlcv_candles) == 0:
                self.logger.warning("No OHLCV data available for chart generation")
                return None
            
            # Get timestamps if available
            timestamps = self.context.timestamps
            
            # Get technical history for RSI overlay (optional)
            technical_history = self.context.technical_history
            
            # Generate chart image using chart generator
            # The chart_generator will automatically limit candles based on AI_CHART_CANDLE_LIMIT
            chart_image = self.chart_generator.create_chart_image(
                ohlcv=self.context.ohlcv_candles,
                technical_history=technical_history,
                pair_symbol=self.symbol,
                timeframe=self.context.timeframe,
                save_to_disk=self.config.DEBUG_SAVE_CHARTS,
                timestamps=timestamps
            )
            
            # If save_to_disk was True, chart_image will be a file path string
            # Otherwise it's a BytesIO object
            if isinstance(chart_image, str):
                # Chart was saved to disk, read it back into BytesIO
                with open(chart_image, 'rb') as f:
                    img_buffer = io.BytesIO(f.read())
                    img_buffer.seek(0)
                    return img_buffer
            else:
                return chart_image
                
        except Exception as e:
            self.logger.error(f"Failed to generate chart image: {e}")
            return None

    async def _calculate_technical_indicators(self) -> None:
        """Calculate technical indicators using the technical calculator"""
        indicators = self.technical_calculator.get_indicators(self.context.ohlcv_candles)
        
        # Store history in context
        self.context.technical_history = indicators
        
        # Extract latest values for each indicator for the technical_data
        technical_data = {}
        for key, values in indicators.items():
            try:
                # Handle different types of NumPy array returns based on shape
                if values is None or np.isnan(values).all():
                    continue
                    
                # Check if it's a standard 1D array (most indicators)
                if isinstance(values, np.ndarray) and values.ndim == 1:
                    technical_data[key] = float(values[-1])
                    
                # Handle 2D arrays - common for indicators that return multiple series like vortex_indicator
                elif isinstance(values, np.ndarray) and values.ndim > 1:
                    # For 2D array, take the last value from each series
                    technical_data[key] = [float(values[i, -1]) for i in range(values.shape[0])]
                    
                # Handle tuples - common return type from indicator functions (e.g., MACD, support_resistance)
                elif isinstance(values, tuple) and all(isinstance(item, np.ndarray) for item in values):
                    # For tuple of arrays, store as list of last values
                    technical_data[key] = [float(array[-1]) for array in values]
                    
                # Handle lists - could be lists of arrays or scalar values
                elif isinstance(values, list):
                    if all(isinstance(item, np.ndarray) for item in values):
                        technical_data[key] = [float(array[-1]) for array in values]
                    else:
                        technical_data[key] = values
                        
                # Handle scalar values
                else:
                    technical_data[key] = float(values)
                    
            except (IndexError, TypeError, ValueError) as e:
                self.logger.warning(f"Could not process indicator '{key}': {e}")
                continue
                
        self.context.technical_data = technical_data

    async def _process_long_term_data(self) -> None:
        """Process long-term historical data and calculate metrics"""
        if not hasattr(self.context, 'long_term_data') or self.context.long_term_data is None:
            self.logger.debug("No long-term data available to process")
            return
            
        if 'data' not in self.context.long_term_data or self.context.long_term_data['data'] is None:
            self.logger.debug("Long-term data contains no OHLCV data")
            return
            
        try:
            # Get long-term indicators and metrics
            long_term_indicators = self.technical_calculator.get_long_term_indicators(
                self.context.long_term_data['data']
            )
            
            # Update the context with calculated metrics
            self.context.long_term_data.update(long_term_indicators)
            
        except Exception as e:
            self.logger.error(f"Error processing long-term data: {str(e)}")
        
        # Calculate weekly macro (if available)
        # Check for dynamic attribute attached in DataCollector (see MarketDataCollector.collect_data)
        if hasattr(self.context, 'weekly_ohlcv') and self.context.weekly_ohlcv is not None:
            try:
                # self.logger.info("Calculating weekly macro indicators...")
                weekly_macro = self.technical_calculator.get_weekly_macro_indicators(self.context.weekly_ohlcv)
                self.context.weekly_macro_indicators = weekly_macro
                
                if 'weekly_macro_trend' in weekly_macro:
                    trend = weekly_macro['weekly_macro_trend']
                    self.logger.info(f"Weekly Macro: {trend.get('trend_direction')} ({trend.get('confidence_score')}%)")
                    if trend.get('cycle_phase'):
                        self.logger.info(f"Cycle Phase: {trend['cycle_phase']}")
            except Exception as e:
                self.logger.error(f"Error calculating weekly macro indicators: {str(e)}")
                self.context.weekly_macro_indicators = None
        else:
            self.context.weekly_macro_indicators = None

