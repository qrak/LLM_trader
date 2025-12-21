import asyncio
import time
from datetime import datetime
from typing import Optional
import time

from src.logger.logger import Logger
from src.platforms.alternative_me import AlternativeMeAPI
from src.platforms.coingecko import CoinGeckoAPI
from src.platforms.cryptocompare import CryptoCompareAPI
from src.platforms.exchange_manager import ExchangeManager
from src.analyzer.core.analysis_engine import AnalysisEngine
from src.rag import RagEngine
from src.utils.token_counter import TokenCounter
from src.utils.loader import config
from src.utils.format_utils import FormatUtils
from src.utils.timeframe_validator import TimeframeValidator
from src.analyzer.data.data_processor import DataProcessor
from src.models.manager import ModelManager
from src.trading import DataPersistence, TradingStrategy


class CryptoTradingBot:
    """Automated crypto trading bot - TRADING MODE ONLY."""
    
    def __init__(self, logger: Logger):
        self.market_analyzer = None
        self.coingecko_api = None
        self.cryptocompare_api = None
        self.alternative_me_api = None
        self.rag_engine = None
        self.logger = logger
        self.symbol_manager = None
        self.token_counter = None
        self.format_utils = None
        self.data_processor = None
        self.model_manager = None
        self.tasks = []
        self.running = False
        self._active_tasks = set()
        
        # Trading components
        self.data_persistence: Optional[DataPersistence] = None
        self.trading_strategy: Optional[TradingStrategy] = None
        self.current_exchange = None
        self.current_symbol: Optional[str] = None
        self.current_timeframe: Optional[str] = None

    async def initialize(self):
        self.logger.info("Initializing Crypto Trading Bot...")

        # Initialize TokenCounter early
        self.token_counter = TokenCounter()
        self.logger.debug("TokenCounter initialized")

        # Initialize DataProcessor and FormatUtils
        self.data_processor = DataProcessor()
        self.format_utils = FormatUtils(self.data_processor)
        self.logger.debug("DataProcessor and FormatUtils initialized")

        self.symbol_manager = ExchangeManager(self.logger, config)
        await self.symbol_manager.initialize()
        self.logger.debug("SymbolManager initialized")

        # Initialize API clients
        self.coingecko_api = CoinGeckoAPI(logger=self.logger)
        await self.coingecko_api.initialize()
        self.logger.debug("CoinGeckoAPI initialized")
        
        self.cryptocompare_api = CryptoCompareAPI(logger=self.logger, config=config)
        await self.cryptocompare_api.initialize()
        self.logger.debug("CryptoCompareAPI initialized")
        
        self.alternative_me_api = AlternativeMeAPI(logger=self.logger)
        await self.alternative_me_api.initialize()
        self.logger.debug("AlternativeMeAPI initialized")

        # Pass token_counter and initialized API clients to RagEngine
        self.rag_engine = RagEngine(
            self.logger, 
            self.token_counter,
            config,
            coingecko_api=self.coingecko_api,
            cryptocompare_api=self.cryptocompare_api,
            format_utils=self.format_utils
        )
        await self.rag_engine.initialize()
        self.logger.debug("RagEngine initialized")
        
        self.rag_engine.set_symbol_manager(self.symbol_manager)
        self.logger.debug("Passed SymbolManager to RagEngine")

        # Initialize ModelManager before AnalysisEngine
        self.model_manager = ModelManager(self.logger, config)
        self.logger.debug("ModelManager initialized")

        self.market_analyzer = AnalysisEngine(
            logger=self.logger,
            rag_engine=self.rag_engine,
            coingecko_api=self.coingecko_api,
            model_manager=self.model_manager,
            alternative_me_api=self.alternative_me_api,
            cryptocompare_api=self.cryptocompare_api,
            format_utils=self.format_utils,
            data_processor=self.data_processor,
            config=config
        )
        self.logger.debug("AnalysisEngine initialized")

        # Initialize trading components
        self.data_persistence = DataPersistence(self.logger, data_dir="data/trading", max_memory=10)
        self.trading_strategy = TradingStrategy(self.logger, self.data_persistence, config)
        self.logger.debug("Trading components initialized")

        await self.rag_engine.start_periodic_updates()
        
        self.logger.info("Crypto Trading Bot initialized successfully")

    async def run(self, symbol: str, timeframe: str = None):
        """Run the trading bot in continuous mode.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Optional timeframe override
        """
        self.current_symbol = symbol
        self.current_timeframe = timeframe or config.TIMEFRAME
        
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
        else:
            self.logger.info("No existing position")
        
        # Start the periodic trading loop
        self.running = True
        check_count = 0
        
        # If resuming with existing position, wait for next timeframe before first check
        if self.trading_strategy.current_position:
            last_analysis_time = self.data_persistence.get_last_analysis_time()
            if last_analysis_time:
                await self._wait_until_next_timeframe_after(last_analysis_time)
                self.logger.info("Resumed from last check, ready for next analysis")
        
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
        
        # Build trading context with P&L data
        position_context = self.trading_strategy.get_position_context(current_price)
        memory_context = self.data_persistence.get_memory_context(current_price)
        
        # Load previous response for AI continuity
        previous_response = self.data_persistence.load_previous_response()
        
        result = await self.market_analyzer.analyze_market(
            additional_context=f"\n\n{position_context}\n\n{memory_context}",
            previous_response=previous_response
        )
        
        if "error" in result:
            self.logger.error(f"Analysis failed: {result['error']}")
            return
        
        # Process the analysis for trading decision
        decision = await self.trading_strategy.process_analysis(result, self.current_symbol)
        
        if decision:
            self._print_trading_decision(decision)
        else:
            self.logger.info("No trading action taken")
        
        # Save the response for context
        raw_response = result.get("raw_response", "")
        if raw_response:
            self.data_persistence.save_previous_response(raw_response)
    
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
            
            next_check_time = datetime.fromtimestamp(next_candle_ms / 1000)
            self.logger.info(f"Next check at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} (in {delay_seconds:.0f}s)")
            
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
            
            current_time = datetime.fromtimestamp(current_time_ms / 1000)
            
            # Calculate interval
            interval_seconds = TimeframeValidator.to_minutes(self.current_timeframe) * 60
            interval_ms = interval_seconds * 1000
            
            # Get last analysis time in ms
            last_time_ms = int(last_time.timestamp() * 1000)
            
            # Calculate next candle after last analysis
            next_candle_ms = ((last_time_ms // interval_ms) + 1) * interval_ms
            
            # If we're already past the next candle, calculate from current time
            if current_time_ms >= next_candle_ms:
                next_candle_ms = ((current_time_ms // interval_ms) + 1) * interval_ms
            
            delay_ms = next_candle_ms - current_time_ms + 5000  # Add 5 second buffer
            delay_seconds = max(0, delay_ms / 1000)
            
            next_check_time = datetime.fromtimestamp(next_candle_ms / 1000)
            
            if delay_seconds > 0:
                self.logger.info(
                    f"Resuming from last check at {last_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Next check at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} (in {delay_seconds:.0f}s)"
                )
                await self._interruptible_sleep(delay_seconds)
            else:
                self.logger.info("Next timeframe already arrived, proceeding immediately")
            
        except Exception as e:
            self.logger.error(f"Error calculating wait time: {e}")
            # Default to 1 minute wait on error
            await self._interruptible_sleep(60)
    
    async def _interruptible_sleep(self, seconds: float):
        """Sleep in small chunks to allow responsive shutdown."""
        chunk_size = 1.0  # Check every second
        elapsed = 0.0
        
        try:
            while elapsed < seconds and self.running:
                sleep_time = min(chunk_size, seconds - elapsed)
                await asyncio.sleep(sleep_time)
                elapsed += sleep_time
        except asyncio.CancelledError:
            # Allow immediate cancellation
            raise
    
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

    def _cleanup_references(self):
        """Set all component references to None to help garbage collection."""
        self.alternative_me_api = None
        self.cryptocompare_api = None
        self.coingecko_api = None
