"""Market Analysis Engine.

This module orchestrates the gathering of market data, technical analysis,
and AI generation to produce trading signals.
"""
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import io
import asyncio
from datetime import datetime

import numpy as np

from src.utils.timeframe_validator import TimeframeValidator
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.coingecko import CoinGeckoAPI
from src.utils.profiler import profile_performance
from src.utils.indicator_classifier import (
    classify_trend_direction,
    classify_volatility_level,
    classify_rsi_level,
    classify_macd_signal,
    classify_volume_state,
    classify_bb_position,
    classify_market_sentiment,
    classify_order_book_bias,
<<<<<<< HEAD
=======
    build_exit_execution_context_from_config,
>>>>>>> main
)
from src.logger.logger import Logger
from .analysis_context import AnalysisContext

if TYPE_CHECKING:
    from src.contracts.model_contract import ModelManagerProtocol
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
        _alternative_me_api: AlternativeMeAPI,
        market_api,
        config: "ConfigProtocol",
        technical_calculator=None,
        pattern_analyzer=None,
        prompt_builder=None,
        data_collector=None,
        metrics_calculator=None,
        result_processor=None,
        chart_generator=None,
        data_fetcher_factory=None
    ) -> None:
        """
        Initialize AnalysisEngine with injected dependencies (DI pattern).

        Args:
            logger: Logger instance
            rag_engine: RAG engine for news and context
            coingecko_api: CoinGecko API client
            model_manager: AI model manager (Protocol-based)
            _alternative_me_api: AlternativeMeAPI (Unused)
<<<<<<< HEAD
            market_api: CryptoCompare Market API client
=======
            market_api: Market metadata API client
>>>>>>> main
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
            data_fetcher_factory: DataFetcherFactory instance (injected from app.py)
        """
        # pylint: disable=too-many-arguments, too-many-locals
        self.logger = logger

        self.config = config
        # Basic properties
        self.exchange = None
        self.symbol = None
        self.base_symbol = None
        self.context = None
        self.article_urls = {}
<<<<<<< HEAD
        self.last_analysis_result = None
=======
>>>>>>> main
        self.previous_microstructure_snapshots: Dict[str, Dict[str, Any]] = {}

        # Load configuration
        try:
            self.timeframe = TimeframeValidator.validate_and_normalize(self.config.TIMEFRAME)
            self.limit = self.config.CANDLE_LIMIT
<<<<<<< HEAD

            # Validate timeframe
            if not TimeframeValidator.validate(self.timeframe):
                self.logger.warning("Timeframe '%s' is not fully supported. Supported timeframes: %s. Proceeding but expect potential calculation errors.", self.timeframe, ', '.join(TimeframeValidator.SUPPORTED_TIMEFRAMES))
=======
        except ValueError as e:
            self.logger.error("Invalid configured timeframe: %s", e)
            raise
>>>>>>> main
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.exception("Error loading configuration values: %s.", e)
            raise
        # Store injected components
        self.model_manager = model_manager
        self.technical_calculator = technical_calculator
        self.pattern_analyzer = pattern_analyzer
        self.prompt_builder = prompt_builder
        self.data_collector = data_collector
        self.metrics_calculator = metrics_calculator
        self.result_processor = result_processor
        self.chart_generator = chart_generator
        self.data_fetcher_factory = data_fetcher_factory

        # Store references to external services
        self.rag_engine = rag_engine
        self.coingecko_api = coingecko_api
        self.market_api = market_api

        # Use the token counter from model_manager
        self.token_counter = self.model_manager.token_counter

        # Dashboard Monitoring Data
        self.last_generated_prompt: Optional[str] = None
        self.last_prompt_timestamp: Optional[str] = None
        self.last_system_prompt: Optional[str] = None
        self.last_llm_response: Optional[str] = None
        self.last_response_timestamp: Optional[str] = None
        self.last_chart_buffer: Optional[io.BytesIO] = None

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
<<<<<<< HEAD
            self.logger.warning("Timeframe validation failed: %s. Using config default: %s", e, self.timeframe)
=======
            self.logger.warning("Timeframe validation failed: %s. Falling back to config default: %s", e, self.timeframe)
>>>>>>> main
            effective_timeframe = self.timeframe

        self.context = AnalysisContext(symbol)
        self.context.exchange = exchange.name if exchange.name else str(exchange)
        self.context.timeframe = effective_timeframe

        # Create data fetcher via factory
        data_fetcher = self.data_fetcher_factory.create(exchange)

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
<<<<<<< HEAD
        self.last_analysis_result = None
=======
>>>>>>> main

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
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Error during MarketAnalyzer cleanup: %s", e)

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
        brain_service = None,
        last_analysis_time: Optional[str] = None,
        current_ticker: Optional[Dict[str, Any]] = None,
        dynamic_thresholds: Optional[Dict[str, Any]] = None
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
            brain_service: TradingBrainService instance to generate context from CURRENT indicators
            last_analysis_time: Formatted timestamp of last analysis (e.g., "2025-12-26 14:30:00")
            current_ticker: Optional dict containing current ticker data to avoid redundant API calls
            dynamic_thresholds: Optional dict containing brain-learned thresholds for response template

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Step 1: Collect all required data
            if not await self._collect_market_data():
                return {"error": "Failed to collect market data", "details": "Data collection failed"}

            async def run_tech_and_chart():
                await self._perform_technical_analysis()
                has_chart_analysis = self.model_manager.supports_image_analysis(provider)
                if has_chart_analysis:
                    return await self._generate_chart_image(), has_chart_analysis
                return None, False

            async def run_rag_analysis():
                query = self.rag_engine.build_context_query(self.symbol)
                
                market_context = await self.rag_engine.retrieve_context(
                    query,
                    self.symbol,
                    k=self.rag_engine.config.RAG_NEWS_LIMIT
                )

                rag_urls = {}
                try:
<<<<<<< HEAD
                    rag_urls = self.rag_engine.context_builder.get_latest_article_urls()
=======
                    rag_urls = self.rag_engine.get_latest_article_urls_snapshot()
>>>>>>> main
                except Exception as e:  # pylint: disable=broad-exception-caught
                    self.logger.warning("Could not retrieve article URLs from RAG engine: %s", e)

                return market_context, rag_urls

            results = await asyncio.gather(
                self._enrich_market_context(current_ticker=current_ticker),
                run_tech_and_chart(),
                run_rag_analysis()
            )

            chart_image, has_chart_analysis = results[1]
            market_context, rag_urls = results[2]

            self.article_urls.update(rag_urls)

            if market_context:
                self.prompt_builder.add_custom_instruction(market_context)
            else:
                self.logger.warning("No market context available for %s", self.symbol)

            # Step 3.5: Generate brain context from CURRENT indicators (after technical analysis)
            brain_context = None
            if brain_service and self.context.technical_data:
                brain_context = await self._generate_brain_context_from_current_indicators(
                    brain_service, self.context.technical_data
                )

            # Step 4: Generate AI analysis
            analysis_result = await self._generate_ai_analysis(
                provider, model, additional_context, previous_response,
                previous_indicators, position_context, performance_context,
                brain_context, last_analysis_time, dynamic_thresholds,
                precomputed_chart=(chart_image, has_chart_analysis) # Pass precomputed chart
            )

<<<<<<< HEAD
            # Store the result for later publication
            self.last_analysis_result = analysis_result

=======
>>>>>>> main
            # Reset custom instructions for next run
            self.prompt_builder.custom_instructions = []

            return analysis_result

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.exception("Analysis failed: %s", e)
            return {"error": str(e), "recommendation": "HOLD"}

    async def _collect_market_data(self) -> bool:
        """Collect market data using data collector"""
        data_result = await self.data_collector.collect_data(self.context)
        if not data_result["success"]:
            self.logger.error("Failed to collect market data: %s", data_result['errors'])
            return False

        # Store article URLs (initial empty set, will be updated by parallel RAG task)
        self.article_urls = self.data_collector.article_urls

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
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("Failed to fetch market overview: %s", e)
            self.context.market_overview = {}

        # Fetch market microstructure
        try:
            microstructure = await self.data_collector.data_fetcher.fetch_market_microstructure(
                self.symbol,
                cached_ticker=current_ticker
            )
            microstructure = self._apply_microstructure_snapshot_context(microstructure)
            self.context.market_microstructure = microstructure
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("Failed to fetch market microstructure: %s", e)
            self.context.market_microstructure = {}

        # Fetch cryptocurrency details
        if self.market_api:
            try:
                coin_details = await self.market_api.get_coin_details(self.base_symbol)
                self.context.coin_details = coin_details
                if coin_details:
                    # self.logger.debug("Coin details for %s fetched and added to context", self.base_symbol)
                    pass
                else:
                    self.logger.warning("No coin details found for %s", self.base_symbol)
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.warning("Failed to fetch coin details for %s: %s", self.base_symbol, e)
                self.context.coin_details = {}

    def _copy_comparison_bucket(self, bucket: Dict[str, Any]) -> Dict[str, float]:
        """Copy only numeric fields needed for snapshot-to-snapshot comparisons."""
        return {
            'bid_depth': float(bucket.get('bid_depth', 0.0)),
            'ask_depth': float(bucket.get('ask_depth', 0.0)),
            'imbalance': float(bucket.get('imbalance', 0.0))
        }

    def _build_order_book_comparison_state(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """Persist only compact order book metrics needed for the next-cycle delta."""
        return {
            'timestamp': order_book.get('timestamp'),
            'spread': float(order_book.get('spread', 0.0)),
            'spread_percent': float(order_book.get('spread_percent', 0.0)),
            'bid_depth': float(order_book.get('bid_depth', 0.0)),
            'ask_depth': float(order_book.get('ask_depth', 0.0)),
            'imbalance': float(order_book.get('imbalance', 0.0)),
            'best_bid_size': float(order_book.get('best_bid_size', 0.0)),
            'best_ask_size': float(order_book.get('best_ask_size', 0.0)),
            'depth_by_level': {
                key: self._copy_comparison_bucket(bucket)
                for key, bucket in order_book.get('depth_by_level', {}).items()
            },
            'liquidity_near_mid': {
                key: self._copy_comparison_bucket(bucket)
                for key, bucket in order_book.get('liquidity_near_mid', {}).items()
            }
        }

    def _build_order_book_deltas(
        self,
        current_order_book: Dict[str, Any],
        previous_order_book: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build deltas versus the immediately previous analysis-cycle snapshot."""
        if not previous_order_book:
            return {}

        current_timestamp = current_order_book.get('timestamp')
        previous_timestamp = previous_order_book.get('timestamp')
        snapshot_interval_seconds = None
        if current_timestamp and previous_timestamp:
            snapshot_interval_seconds = max(0.0, (current_timestamp - previous_timestamp) / 1000)

        top_10_current = current_order_book.get('depth_by_level', {}).get('10', {})
        top_10_previous = previous_order_book.get('depth_by_level', {}).get('10', {})
        near_mid_current = current_order_book.get('liquidity_near_mid', {}).get('10bps', {})
        near_mid_previous = previous_order_book.get('liquidity_near_mid', {}).get('10bps', {})

        return {
            'snapshot_interval_seconds': snapshot_interval_seconds,
            'spread': float(current_order_book.get('spread', 0.0)) - float(previous_order_book.get('spread', 0.0)),
            'spread_percent': float(current_order_book.get('spread_percent', 0.0)) - float(previous_order_book.get('spread_percent', 0.0)),
            'bid_depth': float(current_order_book.get('bid_depth', 0.0)) - float(previous_order_book.get('bid_depth', 0.0)),
            'ask_depth': float(current_order_book.get('ask_depth', 0.0)) - float(previous_order_book.get('ask_depth', 0.0)),
            'imbalance': float(current_order_book.get('imbalance', 0.0)) - float(previous_order_book.get('imbalance', 0.0)),
            'best_bid_size': float(current_order_book.get('best_bid_size', 0.0)) - float(previous_order_book.get('best_bid_size', 0.0)),
            'best_ask_size': float(current_order_book.get('best_ask_size', 0.0)) - float(previous_order_book.get('best_ask_size', 0.0)),
            'top_10': {
                'bid_depth': float(top_10_current.get('bid_depth', 0.0)) - float(top_10_previous.get('bid_depth', 0.0)),
                'ask_depth': float(top_10_current.get('ask_depth', 0.0)) - float(top_10_previous.get('ask_depth', 0.0)),
                'imbalance': float(top_10_current.get('imbalance', 0.0)) - float(top_10_previous.get('imbalance', 0.0)),
            },
            'near_mid_10bps': {
                'bid_depth': float(near_mid_current.get('bid_depth', 0.0)) - float(near_mid_previous.get('bid_depth', 0.0)),
                'ask_depth': float(near_mid_current.get('ask_depth', 0.0)) - float(near_mid_previous.get('ask_depth', 0.0)),
                'imbalance': float(near_mid_current.get('imbalance', 0.0)) - float(near_mid_previous.get('imbalance', 0.0)),
            }
        }

    def _apply_microstructure_snapshot_context(self, microstructure: Dict[str, Any]) -> Dict[str, Any]:
        """Attach snapshot metadata and previous-cycle deltas to microstructure data."""
        snapshot_context = {
            'is_live_snapshot': True,
            'configured_timeframe': self.context.timeframe if self.context else self.timeframe,
            'comparison_basis': 'previous_analysis_cycle_snapshot',
            'comparison_available': False
        }

        order_book = microstructure.get('order_book')
        if not order_book:
            microstructure['snapshot_context'] = snapshot_context
            return microstructure

        previous_snapshot = self.previous_microstructure_snapshots.get(self.symbol, {})
        previous_order_book = previous_snapshot.get('order_book')
        order_book_delta = self._build_order_book_deltas(order_book, previous_order_book)
        if order_book_delta:
            order_book['delta_from_previous_snapshot'] = order_book_delta
            snapshot_context['comparison_available'] = True

        microstructure['order_book'] = order_book
        microstructure['snapshot_context'] = snapshot_context
        self.previous_microstructure_snapshots[self.symbol] = {
            'timestamp': microstructure.get('timestamp'),
            'order_book': self._build_order_book_comparison_state(order_book)
        }
        return microstructure

    async def _perform_technical_analysis(self) -> None:
        """Perform all technical analysis steps"""
        # Calculate technical indicators
        await self._calculate_technical_indicators()

        # Process long-term data
        await self._process_long_term_data()

        # Calculate market metrics
        await asyncio.to_thread(self.metrics_calculator.update_period_metrics, self.context)

        # Run technical pattern analysis
        technical_patterns = await asyncio.to_thread(
            self.pattern_analyzer.detect_patterns,
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
        last_analysis_time: Optional[str] = None,
        dynamic_thresholds: Optional[Dict[str, Any]] = None,
        precomputed_chart: Optional[Tuple[Optional[io.BytesIO], bool]] = None
    ) -> Dict[str, Any]:
        """Generate AI analysis using prompt builder and result processor"""

        # Use precomputed chart if available, otherwise generate (fallback)
        if precomputed_chart:
            chart_image, has_chart_analysis = precomputed_chart
        else:
            # Fallback for synchronous calls or if parallel logic bypassed
            has_chart_analysis = self.model_manager.supports_image_analysis(provider)
            chart_image: Optional[io.BytesIO] = None

            if has_chart_analysis:
                chart_image = await self._generate_chart_image()
                if chart_image is None:
                    has_chart_analysis = False
                    self.logger.warning("Chart generation failed, proceeding without chart analysis")

        system_prompt = self.prompt_builder.build_system_prompt(
            self.symbol,
            self.context,
            previous_response,
            performance_context,
            brain_context,
            last_analysis_time,
            has_chart_analysis,
            dynamic_thresholds,
            previous_indicators=previous_indicators
        )
        prompt = self.prompt_builder.build_prompt(
            context=self.context,
            additional_context=additional_context,
            previous_indicators=previous_indicators,
            position_context=position_context
        )
        # Process analysis
        analysis_result = await self._execute_ai_request(
            system_prompt, prompt, provider, model, chart_image
        )

        # Add metadata to result
        analysis_result["article_urls"] = self.article_urls
        analysis_result["timeframe"] = self.context.timeframe
        actual_provider, actual_model = self.model_manager.describe_provider_and_model(
            provider, model, chart=has_chart_analysis
        )
        analysis_result["provider"] = actual_provider
        analysis_result["model"] = actual_model
        analysis_result["chart_analysis"] = has_chart_analysis

        # Add technical_data for persistence (will be saved to previous_response.json)
        if self.context.technical_data:
            analysis_result["technical_data"] = self.context.technical_data

        # Add prompt for dashboard persistence
        if self.last_generated_prompt:
            analysis_result["generated_prompt"] = self.last_generated_prompt

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
            self.logger.info("Using admin-specified provider: %s, model: %s", provider, model)

        # Give result processor access to context for current_price
        self.result_processor.context = self.context

        # Pass chart image to result processor (it will use chart analysis if image provided)

        # Dashboard: Store both prompts for monitoring
        self.last_generated_prompt = prompt
        self.last_prompt_timestamp = datetime.now().isoformat()
        self.last_system_prompt = system_prompt
        if chart_image:
            self.last_chart_buffer = io.BytesIO(chart_image.getvalue())

        result = await self.result_processor.process_analysis(
            system_prompt, prompt, chart_image=chart_image, provider=provider, model=model
        )
        self.last_llm_response = result.get("raw_response")
        self.last_response_timestamp = datetime.now().isoformat()
        return result

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
            chart_image = await self.chart_generator.create_chart_image(
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

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to generate chart image: %s", e)
            return None

    async def _calculate_technical_indicators(self) -> None:
        """Calculate technical indicators using the technical calculator"""
        # Offload CPU-bound technical analysis to a separate thread to avoid blocking the event loop
        indicators = await asyncio.to_thread(
            self.technical_calculator.get_indicators,
            self.context.ohlcv_candles
        )

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
                self.logger.warning("Could not process indicator '%s': %s", key, e)
                continue

        self.context.technical_data = technical_data

    async def _process_long_term_data(self) -> None:
        """Process long-term historical data and calculate metrics"""
        if self.context.long_term_data is None:
            self.logger.debug("No long-term data available to process")
            return

        if 'data' not in self.context.long_term_data or self.context.long_term_data['data'] is None:
            self.logger.debug("Long-term data contains no OHLCV data")
            return

        try:
            # Get long-term indicators and metrics
            long_term_indicators = await asyncio.to_thread(
                self.technical_calculator.get_long_term_indicators,
                self.context.long_term_data['data']
            )

            # Update the context with calculated metrics
            self.context.long_term_data.update(long_term_indicators)

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Error processing long-term data: %s", str(e))

        # Calculate weekly macro (if available)
        # Check for dynamic attribute attached in DataCollector (see MarketDataCollector.collect_data)
        if self.context.weekly_ohlcv is not None:
            try:
                # self.logger.info("Calculating weekly macro indicators...")
                weekly_macro = await asyncio.to_thread(
                    self.technical_calculator.get_weekly_macro_indicators,
                    self.context.weekly_ohlcv
                )
                self.context.weekly_macro_indicators = weekly_macro

                if 'weekly_macro_trend' in weekly_macro:
                    trend = weekly_macro['weekly_macro_trend']
                    self.logger.info("Weekly Macro: %s (%s%%)", trend.get('trend_direction'), trend.get('confidence_score'))
                    if trend.get('cycle_phase'):
                        self.logger.info("Cycle Phase: %s", trend['cycle_phase'])
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error("Error calculating weekly macro indicators: %s", str(e))
                self.context.weekly_macro_indicators = None
        else:
            self.context.weekly_macro_indicators = None

    async def _generate_brain_context_from_current_indicators(self, brain_service, technical_data: Dict[str, Any]) -> str:
        """Generate brain context using CURRENT technical indicators.

        Offloads blocking ChromaDB queries and CPU-bound embedding operations
        to a thread pool via asyncio.to_thread to avoid stalling the event loop.

        Args:
            brain_service: TradingBrainService instance
            technical_data: Dict of current technical indicators

        Returns:
            Formatted brain context string
        """
        # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        # Extract trend direction from +DI/-DI
        trend_direction = classify_trend_direction(technical_data)

        # Extract ADX
        adx_value = technical_data.get("adx", 0.0)

        # Extract volatility from ATR%
        volatility_level = classify_volatility_level(technical_data)

        # Extract RSI level
        rsi_level = classify_rsi_level(technical_data)

        # Extract MACD signal from macd_line vs macd_signal
        macd_signal = classify_macd_signal(technical_data)

        # Extract volume state from obv_slope
        volume_state = classify_volume_state(technical_data)

        # Extract BB position from bb_upper/bb_lower
        bb_position = classify_bb_position(technical_data, self.context.current_price)

        # Extract weekend status (Saturday=5, Sunday=6)
        is_weekend = datetime.now().weekday() >= 5

        # Extract market sentiment from Fear & Greed data
        market_sentiment = classify_market_sentiment(self.context.sentiment)

        # Extract order book bias from microstructure
        order_book_bias = classify_order_book_bias(self.context.market_microstructure)

        # Extract raw RSI for numeric embedding in query document
        rsi_value = technical_data.get("rsi", 50.0)
<<<<<<< HEAD
=======
        exit_execution_context = build_exit_execution_context_from_config(
            self.config,
            self.timeframe,
        )
>>>>>>> main

        # Offload blocking ChromaDB + embedding calls to a thread to free the event loop
        return await asyncio.to_thread(
            brain_service.get_context,
            trend_direction=trend_direction,
            adx=adx_value,
            rsi=rsi_value,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
        )

