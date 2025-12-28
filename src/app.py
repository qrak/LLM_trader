import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from src.logger.logger import Logger
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.coingecko import CoinGeckoAPI
from src.platforms.cryptocompare import CryptoCompareAPI
from src.platforms.exchange_manager import ExchangeManager
from src.analyzer.analysis_engine import AnalysisEngine
from src.rag import RagEngine
from src.utils.token_counter import TokenCounter
from src.utils.format_utils import FormatUtils
from src.utils.timeframe_validator import TimeframeValidator
from src.analyzer.data_processor import DataProcessor
from src.contracts.manager import ModelManager
from src.trading import DataPersistence, TradingStrategy
from src.discord_interface import DiscordNotifier
from src.utils.keyboard_handler import KeyboardHandler
from src.rag.text_splitting import SentenceSplitter
from src.parsing.unified_parser import UnifiedParser
from src.factories import TechnicalIndicatorsFactory
from src.rag.article_processor import ArticleProcessor


class CryptoTradingBot:
    """Automated crypto trading bot - TRADING MODE ONLY."""
    
    def __init__(self, logger: Logger, config):
        self.logger = logger
        self.config = config
        self.market_analyzer = None
        self.coingecko_api = None
        self.cryptocompare_api = None
        self.alternative_me_api = None
        self.rag_engine = None
        self.symbol_manager = None
        self.token_counter = None
        self.sentence_splitter = None
        self.format_utils = None
        self.data_processor = None
        self.model_manager = None
        self.tasks = []
        self.running = False
        self._active_tasks = set()
        
        # Keyboard handler
        self.keyboard_handler: Optional[KeyboardHandler] = None
        self._force_analysis = asyncio.Event()
        
        # Trading components
        self.data_persistence: Optional[DataPersistence] = None
        self.trading_strategy: Optional[TradingStrategy] = None
        self.current_exchange = None
        self.current_symbol: Optional[str] = None
        self.current_timeframe: Optional[str] = None
        
        # Discord notifier
        self.discord_notifier: Optional[DiscordNotifier] = None
        self._discord_task: Optional[asyncio.Task] = None
        self._position_status_task: Optional[asyncio.Task] = None

    async def initialize(self):
        start_time = time.perf_counter()
        self.logger.info("Initializing Crypto Trading Bot...")

        # Initialize TokenCounter early
        self.token_counter = TokenCounter()
        self.logger.debug("TokenCounter initialized")

        # Initialize SentenceSplitter
        self.sentence_splitter = SentenceSplitter()
        self.logger.debug("SentenceSplitter initialized")

        # Initialize DataProcessor and FormatUtils
        self.data_processor = DataProcessor()
        self.format_utils = FormatUtils(self.data_processor)
        self.logger.debug("DataProcessor and FormatUtils initialized")

        # === COMPOSITION ROOT: Create shared utilities (no dependencies) ===
        # UnifiedParser - single instance shared across all components
        self.unified_parser = UnifiedParser(self.logger)
        self.logger.debug("UnifiedParser initialized")
        
        # TechnicalIndicatorsFactory - creates fresh TI instances for each calculation
        self.ti_factory = TechnicalIndicatorsFactory()
        self.logger.debug("TechnicalIndicatorsFactory initialized")
        
        # ArticleProcessor - shared instance for article processing
        self.article_processor = ArticleProcessor(
            logger=self.logger,
            format_utils=self.format_utils,
            sentence_splitter=self.sentence_splitter,
            unified_parser=self.unified_parser
        )
        self.logger.debug("ArticleProcessor initialized")

        self.symbol_manager = ExchangeManager(self.logger, self.config)
        await self.symbol_manager.initialize()
        self.logger.debug("SymbolManager initialized")

        # Initialize API clients
        self.coingecko_api = CoinGeckoAPI(
            logger=self.logger, 
            api_key=self.config.COINGECKO_API_KEY
        )
        await self.coingecko_api.initialize()
        self.logger.debug("CoinGeckoAPI initialized")
        
        self.cryptocompare_api = CryptoCompareAPI(logger=self.logger, config=self.config)
        await self.cryptocompare_api.initialize()
        self.logger.debug("CryptoCompareAPI initialized")
        
        self.alternative_me_api = AlternativeMeAPI(logger=self.logger)
        await self.alternative_me_api.initialize()
        self.logger.debug("AlternativeMeAPI initialized")

        # Pass token_counter, API clients, and shared utilities to RagEngine
        self.rag_engine = RagEngine(
            self.logger, 
            self.token_counter,
            self.config,
            coingecko_api=self.coingecko_api,
            cryptocompare_api=self.cryptocompare_api,
            format_utils=self.format_utils,
            sentence_splitter=self.sentence_splitter,
            article_processor=self.article_processor,
            unified_parser=self.unified_parser
        )
        await self.rag_engine.initialize()
        self.logger.debug("RagEngine initialized")
        
        self.rag_engine.set_symbol_manager(self.symbol_manager)
        self.logger.debug("Passed SymbolManager to RagEngine")

        # Initialize ModelManager with unified_parser
        self.model_manager = ModelManager(self.logger, self.config, unified_parser=self.unified_parser)
        self.logger.debug("ModelManager initialized")

        # Initialize AnalysisEngine with all shared utilities
        self.market_analyzer = AnalysisEngine(
            logger=self.logger,
            rag_engine=self.rag_engine,
            coingecko_api=self.coingecko_api,
            model_manager=self.model_manager,
            alternative_me_api=self.alternative_me_api,
            cryptocompare_api=self.cryptocompare_api,
            format_utils=self.format_utils,
            data_processor=self.data_processor,
            config=self.config,
            ti_factory=self.ti_factory,
            unified_parser=self.unified_parser
        )
        self.logger.debug("AnalysisEngine initialized")

        # Initialize trading components
        self.data_persistence = DataPersistence(self.logger, data_dir="data/trading", max_memory=10)
        self.trading_strategy = TradingStrategy(self.logger, self.data_persistence, self.config)
        self.logger.debug("Trading components initialized")

        # Initialize Discord notifier if configured and enabled
        if self.config.DISCORD_BOT_ENABLED and hasattr(self.config, 'BOT_TOKEN_DISCORD') and self.config.BOT_TOKEN_DISCORD:
            try:
                self.discord_notifier = DiscordNotifier(self.logger, self.config)
                self._discord_task = asyncio.create_task(self.discord_notifier.start())
                await self.discord_notifier.wait_until_ready()
                self.logger.debug("Discord notifier initialized and ready")
            except Exception as e:
                self.logger.warning(f"Discord initialization failed: {e}. Continuing without Discord.")
                self.discord_notifier = None
        else:
            self.logger.debug("Discord not configured, notifications disabled")

        await self.rag_engine.start_periodic_updates()
        
        # Initialize keyboard handler
        self.keyboard_handler = KeyboardHandler(logger=self.logger)
        self.keyboard_handler.register_command('a', self._force_analysis_now, "Force immediate analysis")
        self.keyboard_handler.register_command('h', self._show_help, "Show available keyboard commands")
        self.keyboard_handler.register_command('q', self._request_shutdown, "Quit the application")
        
        # Start keyboard handler task
        keyboard_task = asyncio.create_task(
            self.keyboard_handler.start_listening(),
            name="Keyboard-Handler"
        )
        self._active_tasks.add(keyboard_task)
        keyboard_task.add_done_callback(self._active_tasks.discard)
        self.tasks.append(keyboard_task)
        
        end_time = time.perf_counter()
        init_duration = end_time - start_time
        self.logger.info(f"Crypto Trading Bot initialized successfully in {init_duration:.2f} seconds")
        self.logger.info("Keyboard commands: 'a' = force analysis, 'h' = help, 'q' = quit")

    async def run(self, symbol: str, timeframe: str = None):
        """Run the trading bot in continuous mode.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Optional timeframe override
        """
        self.current_symbol = symbol
        self.current_timeframe = timeframe or self.config.TIMEFRAME
        
        # Find exchange that supports the symbol
        exchange, exchange_id = await self.symbol_manager.find_symbol_exchange(symbol)
        if not exchange:
            self.logger.error(f"Symbol {symbol} not found on any configured exchange")
            return
        
        self.current_exchange = exchange
        self.logger.info(f"Starting trading for {symbol} on {exchange_id}")
        self.logger.info(f"Timeframe: {self.current_timeframe}")
        
        # Initialize analyzer for this symbol
        self.market_analyzer.initialize_for_symbol(
            symbol=symbol,
            exchange=exchange,
            language="English",
            timeframe=self.current_timeframe
        )
        
        # Log current position if any
        if self.trading_strategy.current_position:
            pos = self.trading_strategy.current_position
            self.logger.info(f"Existing position: {pos.direction} @ ${pos.entry_price:,.2f}")
            # Start hourly position updates for existing position
            if self.discord_notifier:
                await self._start_position_status_updates()
        else:
            self.logger.info("No existing position")
        
        # Start the periodic trading loop
        self.running = True
        check_count = 0
        
        # Check if resuming from previous session (regardless of position status)
        last_analysis_time = self.data_persistence.get_last_analysis_time()
        if last_analysis_time:
            self.logger.info(f"Resuming from last analysis at {last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
            await self._wait_until_next_timeframe_after(last_analysis_time)
            self.logger.info("Ready for next analysis after wait")
        
        while self.running:
            try:
                check_count += 1
                await self._execute_trading_check(check_count)
                
                # Check if still running before waiting
                if not self.running:
                    break
                
                # Wait for next timeframe
                await self._wait_for_next_timeframe()
                
            except asyncio.CancelledError:
                self.logger.info("Trading cancelled")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                # Wait 60 seconds before retrying on error
                await self._interruptible_sleep(60)
    
    async def _execute_trading_check(self, check_count: int):
        """Execute a single trading check iteration."""
        current_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info(f"Trading Check #{check_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
        
        # Check if existing position hit stop/target
        if self.trading_strategy.current_position:
            try:
                current_price = await self._fetch_current_price()
                close_reason = await self.trading_strategy.check_position(current_price)
                if close_reason:
                    self.logger.info(f"Position closed: {close_reason}")
                    # Stop hourly position updates
                    await self._stop_position_status_updates()
                    # Send performance stats to Discord
                    if self.discord_notifier:
                        history = self.data_persistence.load_trade_history()
                        await self.discord_notifier.send_performance_stats(
                            trade_history=history,
                            symbol=self.current_symbol,
                            channel_id=self.config.MAIN_CHANNEL_ID
                        )
            except Exception as e:
                self.logger.error(f"Error checking position: {e}")
        
        # Run market analysis
        self.logger.info("Running market analysis...")
        
        # Fetch current price for P&L calculation
        try:
            current_price = await self._fetch_current_price()
        except Exception as e:
            self.logger.warning(f"Could not fetch current price for P&L: {e}")
            current_price = None
        
        # Build trading context with P&L data (separate for system prompt)
        position_context = self.trading_strategy.get_position_context(current_price)
        memory_context = self.data_persistence.get_memory_context(current_price)
        brain_context = self.data_persistence.get_brain_context()
        
        # Load previous response for AI continuity
        previous_data = self.data_persistence.load_previous_response()
        previous_response = None
        previous_indicators = None
        
        if previous_data:
            previous_response = previous_data.get("response")
            previous_indicators = previous_data.get("technical_indicators")
        
        # Get last analysis time for temporal context
        last_analysis_time_obj = self.data_persistence.get_last_analysis_time()
        last_analysis_time_str = None
        if last_analysis_time_obj:
            last_analysis_time_str = last_analysis_time_obj.strftime('%Y-%m-%d %H:%M:%S')
        
        result = await self.market_analyzer.analyze_market(
            previous_response=previous_response,
            previous_indicators=previous_indicators,
            position_context=position_context,
            performance_context=memory_context,
            brain_context=brain_context,
            last_analysis_time=last_analysis_time_str
        )
        
        if "error" in result:
            self.logger.error(f"Analysis failed: {result['error']}")
            return
        
        # Save the timestamp of successful analysis
        self.data_persistence.save_last_analysis_time()
        
        # Process the analysis for trading decision
        decision = await self.trading_strategy.process_analysis(result, self.current_symbol)
        
        if decision:
            self._print_trading_decision(decision)
            # Send initial position status and start hourly updates if a new position was opened
            if decision.action in ('BUY', 'SELL') and self.trading_strategy.current_position:
                if self.discord_notifier:
                    # Send initial position status
                    try:
                        current_price = await self._fetch_current_price()
                        await self.discord_notifier.send_position_status(
                            position=self.trading_strategy.current_position,
                            current_price=current_price,
                            channel_id=self.config.MAIN_CHANNEL_ID
                        )
                    except Exception as e:
                        self.logger.warning(f"Error sending initial position status: {e}")
                    # Start hourly updates
                    await self._start_position_status_updates()
        else:
            self.logger.info("No trading action taken")
        
        # Send Discord notification
        if self.discord_notifier:
            await self.discord_notifier.send_analysis_notification(
                result=result,
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                channel_id=self.config.MAIN_CHANNEL_ID
            )
        
        # Save the response and technical indicators for context
        raw_response = result.get("raw_response", "")
        technical_data = result.get("technical_data")  # Get technical indicators from result
        
        if raw_response:
            self.data_persistence.save_previous_response(raw_response, technical_data)
    
    async def _fetch_current_price(self) -> float:
        """Fetch current price from exchange."""
        try:
            ticker = await self.current_exchange.fetch_ticker(self.current_symbol)
            return float(ticker.get('last', ticker.get('close', 0)))
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return 0.0
    
    async def _wait_for_next_timeframe(self):
        """Wait until the next timeframe candle starts."""
        try:
            # Get current time from exchange if possible
            try:
                current_time_ms = await self.current_exchange.fetch_time()
            except Exception:
                current_time_ms = int(time.time() * 1000)
            
            # Calculate interval in milliseconds
            interval_seconds = TimeframeValidator.to_minutes(self.current_timeframe) * 60
            interval_ms = interval_seconds * 1000
            
            # Calculate next candle start
            next_candle_ms = ((current_time_ms // interval_ms) + 1) * interval_ms
            delay_ms = next_candle_ms - current_time_ms + 5000  # Add 5 second buffer
            delay_seconds = delay_ms / 1000
            
            next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)
            self.logger.info(f"Next check at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {delay_seconds:.0f}s)")
            
            await self._interruptible_sleep(delay_seconds)
            
        except Exception as e:
            self.logger.error(f"Error calculating next timeframe: {e}")
            # Default to 5 minute wait on error
            await self._interruptible_sleep(300)
    
    async def _wait_until_next_timeframe_after(self, last_time: datetime):
        """Wait until the next timeframe candle after a specific timestamp.
        
        Args:
            last_time: Timestamp of last analysis
        """
        try:
            # Get current time
            try:
                current_time_ms = await self.current_exchange.fetch_time()
            except Exception:
                current_time_ms = int(time.time() * 1000)
            

            # Calculate interval
            interval_seconds = TimeframeValidator.to_minutes(self.current_timeframe) * 60
            interval_ms = interval_seconds * 1000
            
            # Get last analysis time in ms
            last_time_ms = int(last_time.timestamp() * 1000)
            
            # Calculate next candle after last analysis
            next_candle_ms = ((last_time_ms // interval_ms) + 1) * interval_ms
            
            # Check if we're still within the same candle as the last analysis
            current_candle_ms = (current_time_ms // interval_ms) * interval_ms
            last_candle_ms = (last_time_ms // interval_ms) * interval_ms
            
            if current_candle_ms == last_candle_ms:
                # Still in same candle - wait for next one
                delay_ms = next_candle_ms - current_time_ms + 1000
                delay_seconds = max(0, delay_ms / 1000)
                next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)
                
                self.logger.info(
                    f"Resuming from last check at {last_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Still in same candle - next check at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {delay_seconds:.0f}s)"
                )
                await self._interruptible_sleep(delay_seconds)
            elif current_time_ms >= next_candle_ms:
                # Already past next candle - run immediately
                self.logger.info(
                    f"Resuming from last check at {last_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Next candle already passed - proceeding immediately"
                )
                return
            else:
                # In a different candle but not yet at next_candle_ms (edge case)
                delay_ms = next_candle_ms - current_time_ms + 1000
                delay_seconds = max(0, delay_ms / 1000)
                next_check_time = datetime.fromtimestamp(next_candle_ms / 1000, timezone.utc)
                
                self.logger.info(
                    f"Resuming from last check at {last_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Next check at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {delay_seconds:.0f}s)"
                )
                await self._interruptible_sleep(delay_seconds)
            
        except Exception as e:
            self.logger.error(f"Error calculating wait time: {e}")
            # Default to 1 minute wait on error
            await self._interruptible_sleep(60)
    
    async def _interruptible_sleep(self, seconds: float):
        """Sleep in small chunks to allow responsive shutdown and force analysis."""
        chunk_size = 1.0  # Check every second
        elapsed = 0.0
        
        # Clear force analysis flag before sleeping
        self._force_analysis.clear()
        
        try:
            while elapsed < seconds and self.running:
                # Check for force analysis
                if self._force_analysis.is_set():
                    self._force_analysis.clear()
                    self.logger.info("Force analysis triggered - interrupting wait")
                    return
                
                sleep_time = min(chunk_size, seconds - elapsed)
                await asyncio.sleep(sleep_time)
                elapsed += sleep_time
        except asyncio.CancelledError:
            # Allow immediate cancellation
            raise
    
    async def _force_analysis_now(self):
        """Force immediate analysis by interrupting the wait."""
        self.logger.info("Forcing immediate analysis...")
        self._force_analysis.set()
    
    async def _start_position_status_updates(self):
        """Start periodic position status updates to Discord (every hour)."""
        if self._position_status_task and not self._position_status_task.done():
            return  # Already running
        
        self._position_status_task = asyncio.create_task(
            self._position_status_loop(),
            name="Position-Status-Updates"
        )
        self._active_tasks.add(self._position_status_task)
        self._position_status_task.add_done_callback(self._active_tasks.discard)
        self.logger.debug("Started hourly position status updates")
    
    async def _stop_position_status_updates(self):
        """Stop periodic position status updates."""
        if self._position_status_task and not self._position_status_task.done():
            self._position_status_task.cancel()
            try:
                await self._position_status_task
            except asyncio.CancelledError:
                pass
            self._position_status_task = None
            self.logger.debug("Stopped position status updates")
    
    async def _position_status_loop(self):
        """Send position status updates every hour while position is open."""
        update_interval = 3600  # 1 hour in seconds
        
        try:
            while self.running:
                # Wait for 1 hour (in chunks for responsiveness)
                await self._interruptible_sleep(update_interval)
                
                if not self.running:
                    break
                
                # Check if position is still open
                if not self.trading_strategy.current_position:
                    self.logger.debug("Position closed, stopping status updates")
                    break
                
                # Send position status update
                if self.discord_notifier:
                    try:
                        current_price = await self._fetch_current_price()
                        await self.discord_notifier.send_position_status(
                            position=self.trading_strategy.current_position,
                            current_price=current_price,
                            channel_id=self.config.MAIN_CHANNEL_ID
                        )
                        self.logger.info("Sent hourly position status update to Discord")
                    except Exception as e:
                        self.logger.warning(f"Error sending position status update: {e}")
        except asyncio.CancelledError:
            self.logger.debug("Position status loop cancelled")
            raise
    
    async def _show_help(self):
        """Show help information about available commands."""
        self.logger.info("\n=== Available Keyboard Commands ===")
        self.keyboard_handler.display_help()
        self.logger.info("===================================")
    
    async def _request_shutdown(self):
        """Request application shutdown via keyboard."""
        self.logger.info("Shutdown requested via keyboard command")
        self.running = False
        for task in self.tasks:
            if not task.done():
                task.cancel()
    
    def _print_trading_decision(self, decision):
        """Print a trading decision to console."""
        print("\n" + "=" * 60)
        print("TRADING DECISION")
        print("=" * 60)
        print(f"Time: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Symbol: {decision.symbol}")
        print(f"Action: {decision.action}")
        print(f"Confidence: {decision.confidence}")
        print(f"Price: ${decision.price:,.2f}")
        
        if decision.stop_loss:
            print(f"Stop Loss: ${decision.stop_loss:,.2f}")
        if decision.take_profit:
            print(f"Take Profit: ${decision.take_profit:,.2f}")
        if decision.position_size:
            print(f"Position Size: {decision.position_size * 100:.1f}%")
        
        print(f"\nReasoning: {decision.reasoning}")
        print("=" * 60 + "\n")

    async def shutdown(self):
        self.logger.info("Shutting down gracefully...")
        self.running = False
        
        # Give a moment for loops to see running=False
        await asyncio.sleep(0.1)

        # Cancel and wait for active tasks
        await self._shutdown_active_tasks()
        
        # Close keyboard handler
        await self._shutdown_keyboard_handler()
        
        # Close Discord notifier
        await self._shutdown_discord_notifier()
        
        # Close components in reverse initialization order
        await self._shutdown_market_analyzer()
        await self._shutdown_rag_engine()
        await self._shutdown_api_clients()
        await self._shutdown_symbol_manager()
        
        self._cleanup_references()
        self.logger.info("Shutdown complete")

    async def _shutdown_active_tasks(self):
        """Cancel and wait for active tasks to complete."""
        pending_tasks = list(self._active_tasks)
        if not pending_tasks:
            return
            
        self.logger.info(f"Cancelling {len(pending_tasks)} active tasks...")
        for task in pending_tasks:
            if not task.done():
                task.cancel()

        try:
            await asyncio.wait(pending_tasks, timeout=3.0)
        except asyncio.TimeoutError:
            self.logger.warning("Some tasks didn't complete in time")

    async def _shutdown_market_analyzer(self):
        """Close market analyzer and model manager safely."""
        if hasattr(self, 'model_manager') and self.model_manager:
            try:
                self.logger.info("Closing ModelManager...")
                await asyncio.wait_for(self.model_manager.close(), timeout=3.0)
                self.model_manager = None
            except asyncio.TimeoutError:
                self.logger.warning("ModelManager shutdown timed out")
            except Exception as e:
                self.logger.warning(f"Error closing ModelManager: {e}")
        
        if not self.market_analyzer:
            return
            
        try:
            self.logger.info("Closing market analyzer...")
            await asyncio.wait_for(self.market_analyzer.close(), timeout=3.0)
            self.market_analyzer = None
        except asyncio.TimeoutError:
            self.logger.warning("Market analyzer shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing market analyzer: {e}")

    async def _shutdown_rag_engine(self):
        """Close RAG engine safely."""
        if not hasattr(self, 'rag_engine') or not self.rag_engine:
            return
            
        try:
            self.logger.info("Closing RAG engine...")
            await asyncio.wait_for(self.rag_engine.close(), timeout=3.0)
            self.rag_engine = None
        except asyncio.TimeoutError:
            self.logger.warning("RAG engine shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing RAG engine: {e}")

    async def _shutdown_api_clients(self):
        """Close all API clients safely."""
        api_clients = [
            ("AlternativeMeAPI", self.alternative_me_api),
            ("CryptoCompareAPI", self.cryptocompare_api),
            ("CoinGeckoAPI", self.coingecko_api)
        ]

        for client_name, client in api_clients:
            await self._shutdown_single_api_client(client_name, client)

    async def _shutdown_single_api_client(self, client_name: str, client):
        """Close a single API client safely."""
        if not client or not hasattr(client, 'close'):
            return
            
        try:
            self.logger.info(f"Closing {client_name}...")
            await asyncio.wait_for(client.close(), timeout=3.0)
        except asyncio.TimeoutError:
            self.logger.warning(f"{client_name} shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing {client_name}: {e}")

    async def _shutdown_symbol_manager(self):
        """Close symbol manager safely."""
        if not self.symbol_manager:
            return
            
        try:
            self.logger.info("Closing SymbolManager...")
            await asyncio.wait_for(self.symbol_manager.shutdown(), timeout=3.0)
            self.symbol_manager = None
        except asyncio.TimeoutError:
            self.logger.warning("SymbolManager shutdown timed out")
        except Exception as e:
            self.logger.warning(f"Error closing SymbolManager: {e}")
    
    async def _shutdown_discord_notifier(self):
        """Close Discord notifier safely."""
        if self._discord_task and not self._discord_task.done():
            self._discord_task.cancel()
            try:
                await asyncio.wait_for(self._discord_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        if self.discord_notifier:
            try:
                self.logger.info("Closing Discord notifier...")
                async with self.discord_notifier:
                    pass  # __aexit__ handles cleanup
                self.discord_notifier = None
            except Exception as e:
                self.logger.warning(f"Error closing Discord notifier: {e}")

    async def _shutdown_keyboard_handler(self):
        """Close keyboard handler safely."""
        if not self.keyboard_handler:
            return
        
        try:
            self.logger.info("Closing keyboard handler...")
            await self.keyboard_handler.stop_listening()
            self.keyboard_handler = None
        except Exception as e:
            self.logger.warning(f"Error closing keyboard handler: {e}")
    
    def _cleanup_references(self):
        """Set all component references to None to help garbage collection."""
        self.alternative_me_api = None
        self.cryptocompare_api = None
        self.coingecko_api = None
        self.discord_notifier = None
        self.keyboard_handler = None
